"""vLLM backend -- an OpenAI-compatible client aimed at a vLLM server.

Why vLLM at all
---------------
JARVIS runs a tree of agents: the main agent spawns subagents, which spawn
sub-subagents, and all of them call the model concurrently.  Ollama and AirLLM
effectively serialise that traffic, so the tree finishes no sooner than a single
agent would.  vLLM does continuous batching -- one resident copy of the weights
serves many simultaneous requests -- so N agents cost far less than N times one
agent.  The win being bought here is **throughput under concurrency**, not
single-request latency.

Platform reality, stated honestly
---------------------------------
* **vLLM is Linux-first.**  There is no supported Windows build.  On Windows,
  run the server inside WSL2 or point ``llm.vllm_host`` at another machine.
  This class is a pure HTTP client and works fine on Windows either way.
* **CPU-only is slow per token.**  vLLM's CPU backend exists but is tuned for
  server CPUs; on an i5-10210U class laptop a 27-30B model will still crawl on
  a per-token basis.  What does not degrade is the concurrency story: ten
  agents batched together cost far less than ten agents run one after another,
  because the expensive weight reads are amortised across the batch.
* **The pragmatic middle ground on a CPU-only box is Ollama with
  ``OLLAMA_NUM_PARALLEL>1``**, which also speaks this same ``/v1`` API -- point
  :class:`~jarvis.llm.openai_compat.OpenAICompatBackend` at
  ``http://127.0.0.1:11434/v1`` and the rest of JARVIS cannot tell the
  difference.

:meth:`VLLMBackend.server_command` returns the argv for a matching server so the
CLI and the docs can print it.  Nothing in this module ever launches a process.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional
from urllib import parse as _urlparse

from ..core.config import LLMConfig
from .openai_compat import (
    API_KEY_ENV,
    OpenAICompatBackend,
    config_value,
    normalise_base_url,
)


logger = logging.getLogger(__name__)


#: vLLM's own default bind address.
DEFAULT_VLLM_HOST = "http://127.0.0.1:8000/v1"

#: Concurrent sequences the server keeps in a batch when the client sets no cap.
#: Sized for an agent tree rather than for a single chat: a main agent plus two
#: layers of subagents comfortably produces a dozen simultaneous requests.
DEFAULT_MAX_NUM_SEQS = 16


def vllm_base_url(cfg: LLMConfig) -> str:
    """The API base for the configured vLLM server."""
    return normalise_base_url(
        config_value(cfg, "vllm_host", "openai_base_url") or DEFAULT_VLLM_HOST
    )


def _api_key(cfg: LLMConfig) -> str:
    return config_value(cfg, "vllm_api_key", "openai_api_key", "api_key") or os.environ.get(
        API_KEY_ENV, ""
    )


def _download_dir(cfg: LLMConfig) -> str:
    """Where the server should cache weights, if the config says anything.

    ``Config.models_dir()`` is a method and ``LLMConfig.layer_shards_dir`` is a
    string; both are accepted so a caller can pass either object.
    """
    return config_value(cfg, "models_dir", "layer_shards_dir")


def max_num_seqs(cfg: LLMConfig) -> int:
    """Batch width to serve. Mirrors the client-side in-flight cap when set."""
    limit = int(getattr(cfg, "max_concurrent_requests", 0) or 0)
    if limit > 0:
        return max(4, limit)
    return DEFAULT_MAX_NUM_SEQS


def server_command(cfg: LLMConfig) -> List[str]:
    """Return the argv that starts a vLLM server matching ``cfg``.

    The module-entrypoint form is used because it unambiguously accepts
    ``--model``; ``vllm serve <model> ...`` with the same flags is equivalent.
    Nothing here runs the command -- it exists so the CLI and documentation can
    show the user exactly what to start on the Linux box.
    """
    model = config_value(cfg, "openai_model", "model")
    base = vllm_base_url(cfg)
    parts = _urlparse.urlsplit(base)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 8000

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(int(getattr(cfg, "context_tokens", 8192) or 8192)),
        # Continuous batching width: the whole reason vLLM is here.
        "--max-num-seqs",
        str(max_num_seqs(cfg)),
    ]

    device = str(getattr(cfg, "device", "") or "").strip().lower()
    if device and device != "auto":
        cmd += ["--device", device]

    download_dir = _download_dir(cfg)
    if download_dir:
        cmd += ["--download-dir", download_dir]

    logger.debug("vLLM server argv: %s", " ".join(cmd))

    # Appended after the log line on purpose: a token has no business in a log.
    key = _api_key(cfg)
    if key:
        cmd += ["--api-key", key]

    return cmd


class VLLMBackend(OpenAICompatBackend):
    """OpenAI-compatible client pointed at a vLLM server.

    Defaults to ``cfg.vllm_host`` (``http://127.0.0.1:8000/v1``) and
    ``cfg.model``.  Everything about transport, retries, streaming and the
    in-flight cap is inherited from
    :class:`~jarvis.llm.openai_compat.OpenAICompatBackend`.
    """

    name = "vllm"

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            cfg,
            base_url=base_url or vllm_base_url(cfg),
            model=model or config_value(cfg, "model", "openai_model"),
            api_key=api_key,
        )

    @staticmethod
    def server_command(cfg: LLMConfig) -> List[str]:
        """See :func:`server_command`."""
        return server_command(cfg)

    def health(self) -> Dict[str, Any]:
        """Describe the server and the concurrency settings in force.

        Never raises: an unreachable server is reported as
        ``{"reachable": False, "error": ...}``.  A single attempt, no backoff --
        a health check should answer with the truth now rather than wait out a
        server that is still loading weights.
        """
        info: Dict[str, Any] = {
            "backend": self.name,
            "base_url": self.base_url,
            "reachable": False,
            "models": [],
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_attempts": int(self.max_attempts),
            "request_timeout": float(getattr(self.cfg, "request_timeout", 0.0) or 0.0),
            "server_max_num_seqs": max_num_seqs(self.cfg),
            "error": None,
        }
        try:
            models = self.list_models(attempts=1)
        except Exception as exc:
            info["error"] = str(exc)
            return info
        info["reachable"] = True
        info["models"] = models
        if models and not self._served_model:
            info["model"] = self._match_model(self.model, models)
        return info


__all__ = [
    "VLLMBackend",
    "server_command",
    "vllm_base_url",
    "max_num_seqs",
    "DEFAULT_VLLM_HOST",
    "DEFAULT_MAX_NUM_SEQS",
]
