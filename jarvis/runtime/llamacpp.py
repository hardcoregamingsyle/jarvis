"""llama.cpp: the serving path that makes a dense 27B usable on this CPU.

Ollama is the right default — one binary, rootless, handles pulls. But it does
not expose llama.cpp's ``--model-draft``, and that flag is the single largest
speedup available on a bandwidth-bound CPU. This module builds the launch
command for ``llama-server`` so the OpenAI-compatible backend can talk to it.

Why speculative decoding wins here
----------------------------------
Generating a token from a dense model means reading **every weight** from RAM.
On DDR4-2666 that is ~28 GB/s against 18 GB of Q4 weights: about 1.5 tok/s, and
the CPU sits idle waiting on memory the whole time.

Speculative decoding exploits that idle compute. A small draft model proposes
``k`` tokens cheaply, then the large model verifies **all k in a single
batched forward pass**. Verification costs one read of the target weights
regardless of ``k`` — the expensive resource is touched once per round instead
of once per token.

With acceptance rate ``a`` and draft depth ``k``, expected tokens per round:

.. math::   E = (1 - a^{k+1}) / (1 - a)

Same-family drafts agree often (Qwen3 0.6B drafting for Qwen3.8-27B lands
around 70-80% on ordinary prose), so::

    k=4, a=0.75, draft 0.4 GB:
        reads  = 4 x 0.4 + 18   = 19.6 GB per round
        tokens = (1 - 0.75^5)/0.25 = 3.05
        rate   = 28 / 19.6 x 3.05 = 4.4 tok/s        (~2.8x)

**Output is identical to the target model's.** Rejected drafts are discarded,
so this is a pure latency optimisation, not a quality trade — unlike dropping
to Q2, which buys similar speed by making the model worse.

Acceptance rate is the whole game, and it is workload-dependent: predictable
prose accepts well, dense code and unusual names accept poorly. When it drops
below roughly 50% the draft overhead outweighs the batching win and you are
better off without it. :func:`estimate_speedup` computes the crossover.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Binary names, newest first. Upstream renamed ``server`` to ``llama-server``.
SERVER_BINARIES = ("llama-server", "server")

#: Draft depth. Beyond ~6 the acceptance chain decays faster than the extra
#: tokens repay; 4 is the sweet spot for same-family drafts on CPU.
DEFAULT_DRAFT_TOKENS = 4

#: Below this measured acceptance rate, drafting costs more than it saves.
MIN_USEFUL_ACCEPTANCE = 0.5


@dataclass(frozen=True)
class ServerPlan:
    """A llama-server invocation, with the reasoning attached."""

    argv: List[str]
    host: str
    port: int
    model_path: str
    draft_path: str = ""
    threads: int = 0
    context: int = 8192
    notes: tuple = ()

    @property
    def uses_speculation(self) -> bool:
        return bool(self.draft_path)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command_line(self) -> str:
        """The command as you would type it, quoted where needed."""
        parts = []
        for arg in self.argv:
            parts.append(f'"{arg}"' if " " in arg else arg)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": list(self.argv),
            "base_url": self.base_url,
            "model_path": self.model_path,
            "draft_path": self.draft_path,
            "speculative": self.uses_speculation,
            "threads": self.threads,
            "context": self.context,
            "notes": list(self.notes),
        }


def find_server() -> Optional[str]:
    """Path to ``llama-server``, or ``None``."""
    for name in SERVER_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def is_available() -> bool:
    return find_server() is not None


def expected_tokens_per_round(acceptance: float, draft_tokens: int) -> float:
    """``(1 - a^(k+1)) / (1 - a)`` — expected accepted tokens per round.

    The verified token always lands even when every draft is rejected, which
    is why this is never below 1.0 and why speculation cannot be slower than
    the baseline in *token* terms (only in bytes read).
    """
    a = max(0.0, min(1.0, float(acceptance)))
    k = max(0, int(draft_tokens))
    if a >= 1.0:
        return float(k + 1)
    return (1.0 - a ** (k + 1)) / (1.0 - a)


def estimate_speedup(
    *,
    target_gb: float,
    draft_gb: float,
    acceptance: float,
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    bandwidth_gb_s: float = 28.0,
) -> Dict[str, float]:
    """Model the throughput change from speculation. Arithmetic, not a promise.

    Real acceptance depends on the workload; ``jarvis selftest`` measures the
    rate you actually get.
    """
    baseline = bandwidth_gb_s / max(target_gb, 0.01)
    tokens = expected_tokens_per_round(acceptance, draft_tokens)
    gb_per_round = draft_tokens * draft_gb + target_gb
    speculative = (bandwidth_gb_s / max(gb_per_round, 0.01)) * tokens
    return {
        "baseline_tok_s": round(baseline, 2),
        "speculative_tok_s": round(speculative, 2),
        "speedup": round(speculative / baseline, 2) if baseline else 0.0,
        "tokens_per_round": round(tokens, 2),
        "gb_per_round": round(gb_per_round, 2),
        "worthwhile": speculative > baseline,
    }


def physical_cores() -> int:
    """Physical cores — the right thread count for a bandwidth-bound load.

    Hyperthreaded siblings share one memory port, so pairing them contends for
    the exact resource that is already saturated.
    """
    try:
        ids = set()
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            physical_id = core_id = None
            for line in fh:
                low = line.lower()
                if low.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif low.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif not line.strip() and core_id is not None:
                    ids.add((physical_id, core_id))
                    physical_id = core_id = None
            if core_id is not None:
                ids.add((physical_id, core_id))
        if ids:
            return len(ids)
    except OSError:
        pass
    try:
        import os

        logical = os.cpu_count() or 1
        return max(1, logical // 2) if logical > 1 else 1
    except Exception:  # noqa: BLE001
        return 1


def build_server_plan(
    model_path: Any,
    *,
    draft_path: Any = "",
    host: str = "127.0.0.1",
    port: int = 8080,
    context: int = 8192,
    threads: int = 0,
    draft_tokens: int = DEFAULT_DRAFT_TOKENS,
    parallel: int = 1,
    extra_args: Optional[List[str]] = None,
) -> ServerPlan:
    """Build the ``llama-server`` argv for this machine.

    ``draft_path`` empty disables speculation. Nothing is executed here; the
    plan is printable so the exact command can be shown before it is run.
    """
    notes: List[str] = []
    model = str(model_path)
    draft = str(draft_path or "")

    if threads <= 0:
        threads = physical_cores()
        notes.append(
            f"Using {threads} threads (physical cores). Hyperthread siblings "
            "share a memory port, so pairing them slows a bandwidth-bound load."
        )

    argv: List[str] = [
        find_server() or "llama-server",
        "--model", model,
        "--host", host,
        "--port", str(int(port)),
        "--ctx-size", str(int(context)),
        "--threads", str(int(threads)),
        # Batch threads matter for prompt ingestion, which IS compute-bound.
        "--threads-batch", str(int(threads)),
        # Map weights from the page cache rather than copying them in.
        "--mlock" if False else "--no-mmap" if False else "--mmap",
    ]
    # `--mmap` is the default and older builds reject the explicit flag.
    argv = [a for a in argv if a != "--mmap"]

    if draft:
        argv += [
            "--model-draft", draft,
            "--draft-max", str(int(draft_tokens)),
            "--draft-min", "1",
        ]
        notes.append(
            f"Speculative decoding on: the draft proposes {draft_tokens} tokens "
            "and the 27B verifies them in ONE batched pass. Output is identical "
            "to running the 27B alone -- rejected drafts are discarded."
        )
    else:
        notes.append(
            "No draft model: every token costs a full read of the 27B weights. "
            "Pass draft_path to roughly double throughput."
        )

    if parallel > 1:
        argv += ["--parallel", str(int(parallel))]
        notes.append(
            f"{parallel} parallel slots: the KV cache is divided between them, "
            "so each gets ctx-size/parallel tokens."
        )

    if context > 32768:
        notes.append(
            f"A {context}-token context is ~{context * 64 / 1e6:.1f} GB of KV "
            "cache on top of the weights. Qwen3.8's hybrid attention (only 16 "
            "of 64 layers are full attention) keeps this affordable, but check "
            "it against free RAM -- swapping costs far more than it saves."
        )

    argv += list(extra_args or [])

    return ServerPlan(
        argv=argv,
        host=host,
        port=int(port),
        model_path=model,
        draft_path=draft,
        threads=int(threads),
        context=int(context),
        notes=tuple(notes),
    )


def gguf_candidates(models_dir: Any) -> List[Path]:
    """Every ``.gguf`` under ``models_dir``, largest first."""
    root = Path(models_dir)
    if not root.exists():
        return []
    try:
        files = [p for p in root.rglob("*.gguf") if p.is_file()]
    except OSError:
        return []
    return sorted(files, key=lambda p: p.stat().st_size, reverse=True)


def pick_target_and_draft(models_dir: Any) -> Dict[str, str]:
    """Guess which local GGUF is the target and which is the draft.

    Size is the signal: the largest file is the brain, and the smallest that
    is under a quarter of it is a plausible draft. A draft close in size to
    the target saves nothing, so it is rejected.
    """
    files = gguf_candidates(models_dir)
    if not files:
        return {"target": "", "draft": ""}
    target = files[0]
    draft = ""
    for candidate in reversed(files[1:]):
        if candidate.stat().st_size <= target.stat().st_size * 0.25:
            draft = str(candidate)
            break
    return {"target": str(target), "draft": draft}


__all__ = [
    "ServerPlan",
    "build_server_plan",
    "estimate_speedup",
    "expected_tokens_per_round",
    "find_server",
    "is_available",
    "physical_cores",
    "gguf_candidates",
    "pick_target_and_draft",
    "DEFAULT_DRAFT_TOKENS",
    "MIN_USEFUL_ACCEPTANCE",
    "SERVER_BINARIES",
]
