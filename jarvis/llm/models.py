"""Model catalogue and resolution: which weights to run, and can we get them.

Nothing in JARVIS hardcodes a model.  ``cfg.llm.model`` is the single source of
truth, and this module is the layer that turns whatever string is in it — a
short alias, a full Hugging Face repo id, an Ollama tag, or a repo that was
published after this file was written — into a :class:`ModelSpec` with enough
information to size, download and run it.  Switching models is therefore one
line of config (or one ``JARVIS_LLM_MODEL=...``), including for gated repos
that need a token.

Three jobs:

* **Catalogue** — :data:`KNOWN_MODELS` holds curated presets keyed by a short
  alias, with honest sizes for the CPU-only target machine.  It is a
  convenience, never a whitelist: :func:`resolve` happily synthesises a spec
  for a repo it has never heard of, because an unknown repo id is the normal
  case rather than an error.
* **Sizing** — :func:`recommend` and :func:`estimate_footprint` answer "will
  this actually run on 32 GB of system RAM with no GPU", before 30 GB of
  download says otherwise.
* **Access** — :func:`hf_token`, :func:`check_access` and :func:`local_models`
  turn "mysterious 401 halfway through a download" into "this repo is gated,
  accept the licence here and set HF_TOKEN".

Units: ``GB`` throughout means 10^9 bytes — the unit Hugging Face and Ollama
use when they report download sizes — not GiB.

Secrets: a token is never logged, never placed in a returned dict, and never
interpolated into an exception message.  :func:`redact` masks token-shaped text
and is applied on every path that echoes a URL or an underlying error.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


HF_ENDPOINT = "https://huggingface.co"
HF_API_TIMEOUT = 10.0

# Environment variables the wider ecosystem uses for the HF token, in
# descending precedence.  ``huggingface-cli login`` writes the token to disk
# instead; see :data:`_TOKEN_FILE_ENV`.
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_TOKEN_FILE_ENV = "HF_TOKEN_PATH"


# --------------------------------------------------------------------------- #
#  Errors
# --------------------------------------------------------------------------- #
class ModelError(ValueError):
    """Base class for model-resolution problems."""


class UnknownModelError(ModelError):
    """The string is neither a known alias nor anything shaped like a model."""


class UnreleasedModelError(ModelError):
    """The model is in the catalogue but does not exist yet."""


# --------------------------------------------------------------------------- #
#  The spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Everything JARVIS needs to know about a candidate set of weights.

    Attributes:
        id: Hugging Face repo id (``org/name``) or an Ollama tag.
        label: Human-readable name for menus and logs.
        params: Total parameter count in billions.
        family: Architecture family, e.g. ``"qwen3"``.
        context: Native context window in tokens.
        quantised_size_gb: On-disk size of the usual 4-bit quantisation.
        notes: One line of guidance about when to pick this model.
        backends: JARVIS backend names that can run it.
        gated: The repo requires accepting a licence before download.
        exists: ``False`` for models that are announced, rumoured or merely
            wished for.  They stay in the catalogue so they can be *listed*,
            but :func:`resolve` refuses to select them.
        ollama_tag: Equivalent Ollama tag, when one is published.
        active_params: Parameters actually used per token.  Differs from
            ``params`` only for mixture-of-experts models, where it — not the
            total — governs how fast the thing feels on a CPU.
        alias: Catalogue key, if this spec came from the catalogue.
    """

    id: str
    label: str
    params: float
    family: str
    context: int
    quantised_size_gb: float
    notes: str = ""
    backends: tuple = ()
    gated: bool = False
    exists: bool = True
    ollama_tag: str = ""
    active_params: Optional[float] = None
    alias: str = ""

    @property
    def effective_params(self) -> float:
        """Parameters per token — the number that predicts CPU speed."""
        return float(self.active_params if self.active_params else self.params)

    @property
    def is_moe(self) -> bool:
        """True when only a fraction of the weights fire per token."""
        return bool(self.active_params) and float(self.active_params or 0) < self.params

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view for CLI output and JSON."""
        return {
            "alias": self.alias,
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "params_b": self.params,
            "active_params_b": self.effective_params,
            "moe": self.is_moe,
            "context": self.context,
            "quantised_size_gb": self.quantised_size_gb,
            "backends": list(self.backends),
            "ollama_tag": self.ollama_tag,
            "gated": self.gated,
            "exists": self.exists,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
#  Catalogue
# --------------------------------------------------------------------------- #
_HF_BACKENDS = ("transformers", "vllm", "airllm")
_ALL_BACKENDS = ("ollama", "transformers", "vllm", "airllm")

# Qwen shipped the July-2025 "-2507" Instruct refresh for the 4B, 30B-A3B and
# 235B-A22B checkpoints only; the dense 8B/14B/32B aliases therefore point at
# the original Qwen3 releases, which are still the current weights for those
# sizes.  If a -2507 dense checkpoint appears, changing the ``id`` here is the
# whole migration.
KNOWN_MODELS: Dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        id="Qwen/Qwen3-0.6B",
        label="Qwen3 0.6B",
        params=0.6,
        family="qwen3",
        context=32768,
        quantised_size_gb=0.5,
        notes="Smoke-test size. Runs anywhere; says very little worth hearing.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:0.6b",
    ),
    "qwen3-1.7b": ModelSpec(
        id="Qwen/Qwen3-1.7B",
        label="Qwen3 1.7B",
        params=1.7,
        family="qwen3",
        context=32768,
        quantised_size_gb=1.1,
        notes="Fast enough for wake-word-to-reply on weak hardware.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:1.7b",
    ),
    "qwen3-4b": ModelSpec(
        id="Qwen/Qwen3-4B-Instruct-2507",
        label="Qwen3 4B Instruct (2507)",
        params=4.0,
        family="qwen3",
        context=262144,
        quantised_size_gb=2.5,
        notes="The safe default on a small or busy machine; snappy on CPU.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:4b-instruct-2507-q4_K_M",
    ),
    "qwen3-8b": ModelSpec(
        id="Qwen/Qwen3-8B",
        label="Qwen3 8B",
        params=8.2,
        family="qwen3",
        context=32768,
        quantised_size_gb=4.9,
        notes="Noticeably better reasoning than 4B; borderline for CPU voice latency.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:8b",
    ),
    "qwen3-14b": ModelSpec(
        id="Qwen/Qwen3-14B",
        label="Qwen3 14B",
        params=14.8,
        family="qwen3",
        context=32768,
        quantised_size_gb=8.9,
        notes="Good batch/background model on CPU; too slow to converse with.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:14b",
    ),
    "qwen3-32b": ModelSpec(
        id="Qwen/Qwen3-32B",
        label="Qwen3 32B",
        params=32.8,
        family="qwen3",
        context=32768,
        quantised_size_gb=19.7,
        notes="Smartest dense Qwen3. Wants a GPU: every token touches 32B weights.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:32b",
    ),
    "qwen3-30b-a3b": ModelSpec(
        id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        label="Qwen3 30B-A3B Instruct (2507)",
        params=30.5,
        family="qwen3",
        context=262144,
        quantised_size_gb=18.3,
        notes="Best speed/quality trade-off on a 32 GB CPU box: 30B of knowledge, "
              "~3B of arithmetic per token.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3:30b-a3b-instruct-2507-q4_K_M",
        active_params=3.3,
    ),
    "qwen3-coder-30b-a3b": ModelSpec(
        id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        label="Qwen3 Coder 30B-A3B Instruct",
        params=30.5,
        family="qwen3-coder",
        context=262144,
        quantised_size_gb=18.3,
        notes="Same MoE shape as 30B-A3B, tuned for code and tool calls.",
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3-coder:30b",
        active_params=3.3,
    ),
    "qwen3-235b-a22b": ModelSpec(
        id="Qwen/Qwen3-235B-A22B-Instruct-2507",
        label="Qwen3 235B-A22B Instruct (2507)",
        params=235.0,
        family="qwen3",
        context=262144,
        quantised_size_gb=141.0,
        notes="Frontier-class, and far past what one laptop can hold.",
        backends=_HF_BACKENDS,
        ollama_tag="qwen3:235b-a22b",
        active_params=22.0,
    ),
    "llama3.1-8b": ModelSpec(
        id="meta-llama/Llama-3.1-8B-Instruct",
        label="Llama 3.1 8B Instruct",
        params=8.0,
        family="llama3",
        context=131072,
        quantised_size_gb=4.9,
        notes="Gated: accept Meta's licence on the model page before downloading.",
        backends=_ALL_BACKENDS,
        ollama_tag="llama3.1:8b",
        gated=True,
    ),
    # Requested by the owner; announced by nobody.  Kept here on purpose so
    # `jarvis model list` shows it as a known-but-unavailable target instead of
    # silently pretending the name was a typo.  When it ships: set exists=True
    # (and correct the figures).  That is the entire change.
    # Verified against the Hugging Face API on 2026-08-08: 15 safetensors
    # shards totalling 55.6 GB in bf16, ungated, 6.7M downloads. config.json
    # reports model_type "qwen3_5", architecture Qwen3_5ForConditionalGeneration,
    # 64 layers, hidden 5120, GQA 24/4, max_position_embeddings 262144, and a
    # vision encoder alongside the text tower.
    "qwen3.6-27b": ModelSpec(
        id="Qwen/Qwen3.6-27B",
        label="Qwen3.6 27B (dense, vision-language)",
        params=27.0,
        family="qwen3",
        context=262144,
        # Q4_K_S GGUF measured at 16.1 GB; IQ4_XS 15.7 GB; Q3_K_M 13.8 GB.
        quantised_size_gb=16.1,
        notes=(
            "Strongest model that still fits 32 GB at Q4, and multimodal "
            "(image and video in) with a 262K context extensible to ~1M via "
            "YaRN. DENSE, though: all 27B parameters are read per token, so on "
            "a CPU-only box expect roughly 1 tok/s against 4-8 for the "
            "30B-A3B MoE, which activates only ~3B. Thinking mode is ON by "
            "default and emits hundreds of <think> tokens before answering — "
            "disable it for anything interactive. Serve with vLLM/SGLang on a "
            "GPU, or Ollama GGUF on CPU. Needs transformers >= 4.57."
        ),
        backends=_ALL_BACKENDS,
        ollama_tag="qwen3.6:27b",
        exists=True,
    ),
    "qwen3.8-27b": ModelSpec(
        id="Qwen/Qwen3.8-27B",
        label="Qwen3.8 27B (unreleased)",
        params=27.0,
        family="qwen3",
        context=262144,
        quantised_size_gb=16.2,
        notes="Not released yet — the repository is not publicly readable. "
              "Figures are placeholders. Its released predecessor is "
              "qwen3.6-27b, which is the current default; switching when 3.8 "
              "ships means flipping exists=False here.",
        backends=_ALL_BACKENDS,
        exists=False,
    ),
}

DEFAULT_ALIAS = "qwen3.6-27b"

# Populated at the end of the module, once the normalisation helper exists.
_ID_INDEX: Dict[str, str] = {}
_TAG_INDEX: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
#  Redaction
# --------------------------------------------------------------------------- #
_REDACTED = "***REDACTED***"

# Real HF tokens are ``hf_`` plus ~34 alphanumerics.  The 16-character floor is
# what stops the mask from eating the literal word ``hf_token`` out of our own
# advice messages while still catching every genuine credential.
_TOKEN_PATTERNS: Tuple[re.Pattern, ...] = (
    # Hugging Face user and organisation tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bapi_org_[A-Za-z0-9]{16,}"),
    # Anything handed over in a query string or a header.
    #
    # The value group deliberately absorbs an optional ``Bearer``/``Token``
    # scheme word. Without that, ``Authorization: Bearer <secret>`` matches with
    # "Bearer" as the value — so only the scheme word gets masked, and having
    # consumed it this pattern leaves the actual credential in cleartext where
    # the standalone Bearer pattern below can no longer reach it.
    re.compile(
        r"(?i)\b(token|access_token|api_key|apikey|key|authorization)"
        r"(\s*[:=]\s*)((?:Bearer|Token)\s+)?([^\s&\"']+)"
    ),
    # A bearer credential with no key= prefix in front of it.
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}"),
)


def redact(text: Any) -> str:
    """Mask token-shaped substrings so a secret cannot reach a log or a reply.

    Deliberately blunt: it would rather mangle a harmless URL parameter than
    let ``hf_...`` through.  Applied to every error string and every URL this
    module hands back.
    """
    if text is None:
        return ""
    out = str(text)
    if not out:
        return out
    out = _TOKEN_PATTERNS[0].sub("hf_" + _REDACTED, out)
    out = _TOKEN_PATTERNS[1].sub("api_org_" + _REDACTED, out)
    out = _TOKEN_PATTERNS[2].sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{_REDACTED}", out
    )
    out = _TOKEN_PATTERNS[3].sub("Bearer " + _REDACTED, out)
    return out


# --------------------------------------------------------------------------- #
#  Identification & resolution
# --------------------------------------------------------------------------- #
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])")
_ACTIVE_RE = re.compile(r"[-_]A(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])")


def is_hf_repo_id(text: Any) -> bool:
    """True when ``text`` has the shape of a Hugging Face repo id (``org/name``).

    Shape only — this makes no network call and says nothing about whether the
    repository exists.  Rejects anything with whitespace, a missing or extra
    slash, path traversal, or a non-string value.
    """
    if not isinstance(text, str):
        return False
    candidate = text.strip()
    if not candidate or ".." in candidate:
        return False
    return bool(_REPO_ID_RE.match(candidate))


def is_ollama_tag(text: Any) -> bool:
    """True when ``text`` looks like ``name:tag`` rather than a repo id."""
    if not isinstance(text, str):
        return False
    candidate = text.strip()
    if not candidate or "/" in candidate or any(c.isspace() for c in candidate):
        return False
    return bool(re.match(r"^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+$", candidate))


def normalise_alias(text: Any) -> str:
    """Fold a user-typed alias to catalogue form (lowercase, dashes)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s_]+", "-", text.strip().lower())


def all_models(*, include_unreleased: bool = True) -> List[ModelSpec]:
    """The whole catalogue, smallest first — what ``jarvis model list`` shows."""
    specs = [
        spec for spec in KNOWN_MODELS.values()
        if include_unreleased or spec.exists
    ]
    return sorted(specs, key=lambda s: (s.params, s.id))


def lookup(name_or_alias: Any) -> Optional[ModelSpec]:
    """Catalogue lookup that never raises and never synthesises.

    Returns the spec for an alias, an exact repo id or a known Ollama tag —
    including models with ``exists=False``, which is what lets the CLI list a
    model it must refuse to select.  ``None`` when nothing matches.
    """
    key = normalise_alias(name_or_alias)
    if not key:
        return None
    if key in KNOWN_MODELS:
        return KNOWN_MODELS[key]
    if key in _ID_INDEX:
        return KNOWN_MODELS[_ID_INDEX[key]]
    if key in _TAG_INDEX:
        return KNOWN_MODELS[_TAG_INDEX[key]]
    return None


def _guess_params(repo_id: str) -> Tuple[float, Optional[float]]:
    """Pull ``(total_b, active_b)`` out of a model name, e.g. ``30B-A3B``."""
    name = repo_id.split("/")[-1]
    active_match = _ACTIVE_RE.search(name)
    active = float(active_match.group(1)) if active_match else None
    total = 0.0
    for match in _PARAM_RE.finditer(name):
        value = float(match.group(1))
        if active is not None and value == active and total:
            continue
        total = max(total, value)
    if active is not None and total and active >= total:
        active = None
    return total, active


def _guess_family(repo_id: str) -> str:
    name = repo_id.split("/")[-1].lower()
    match = re.match(r"^([a-z]+[0-9.]*)", name)
    return match.group(1) if match else "unknown"


def _synthesise(raw: str) -> ModelSpec:
    """Build a best-effort spec for a repo the catalogue has never seen.

    New checkpoints appear constantly; refusing to run one because it postdates
    this file would defeat the point.  The numbers are inferred from the name
    and flagged as estimates in ``notes``.
    """
    repo = raw.strip()
    total, active = _guess_params(repo)
    ollama = repo if is_ollama_tag(repo) else ""
    backends = ("ollama",) if ollama else _HF_BACKENDS
    if total:
        size = round(total * _BITS["q4"] / 8.0, 1)
        note = "Not in the built-in catalogue; size estimated from the name."
    else:
        size = 0.0
        note = ("Not in the built-in catalogue and the parameter count is not in "
                "the name; size unknown.")
    return ModelSpec(
        id=repo,
        label=repo.split("/")[-1],
        params=total,
        family=_guess_family(repo),
        context=32768,
        quantised_size_gb=size,
        notes=note,
        backends=backends,
        ollama_tag=ollama,
        active_params=active,
        alias="",
    )


def resolve(name_or_alias: Any, *, allow_unreleased: bool = False) -> ModelSpec:
    """Turn any model string into a :class:`ModelSpec` fit to be loaded.

    Accepts a catalogue alias (``qwen3-30b-a3b``), an exact Hugging Face repo id
    (``Qwen/Qwen3-8B``), an Ollama tag (``qwen3:8b``), or a repo id this build
    has never heard of — the last case is synthesised rather than rejected,
    because an unknown repo id is how every new model starts life.

    Raises:
        UnreleasedModelError: the catalogue knows this model but it does not
            exist yet, so selecting it could only ever 404.  Pass
            ``allow_unreleased=True`` to inspect it anyway.
        UnknownModelError: the string is not a usable model reference at all.
    """
    raw = name_or_alias.strip() if isinstance(name_or_alias, str) else ""
    if not raw:
        raise UnknownModelError(
            "No model given. Set llm.model to a Hugging Face repo id such as "
            f"'{KNOWN_MODELS[DEFAULT_ALIAS].id}', or to one of: "
            f"{', '.join(sorted(KNOWN_MODELS))}."
        )

    spec = lookup(raw)
    if spec is not None:
        if not spec.exists and not allow_unreleased:
            raise UnreleasedModelError(
                f"{spec.label} ({spec.id}) is not released yet: no such model "
                "exists, so it cannot be selected. "
                f"{spec.notes} "
                "When it is published, set exists=True for the "
                f"'{spec.alias or normalise_alias(raw)}' entry in "
                "jarvis/llm/models.py and it becomes selectable immediately."
            )
        return spec

    if is_hf_repo_id(raw) or is_ollama_tag(raw):
        return _synthesise(raw)

    raise UnknownModelError(
        f"{raw!r} is not a known alias, a Hugging Face repo id ('org/name'), "
        "or an Ollama tag ('name:tag'). Known aliases: "
        f"{', '.join(sorted(KNOWN_MODELS))}."
    )


def resolve_config(cfg: Any) -> ModelSpec:
    """Resolve whatever ``cfg`` currently points at (``Config`` or ``LLMConfig``)."""
    llm = getattr(cfg, "llm", cfg)
    name = getattr(llm, "model", "") or ""
    backend = str(getattr(llm, "backend", "") or "").lower()
    if backend == "ollama":
        name = getattr(llm, "ollama_model", "") or name
    return resolve(name)


# --------------------------------------------------------------------------- #
#  Sizing
# --------------------------------------------------------------------------- #
# Effective bits per weight, including the fp16 scales that every k-quant keeps
# alongside the packed values — which is why "4-bit" is really ~4.8.
_BITS: Dict[str, float] = {
    "": 16.0, "none": 16.0, "fp16": 16.0, "bf16": 16.0, "f16": 16.0,
    "fp32": 32.0, "float32": 32.0,
    "8bit": 8.5, "q8": 8.5, "q8_0": 8.5, "int8": 8.0,
    "6bit": 6.6, "q6": 6.6, "q6_k": 6.6,
    "5bit": 5.5, "q5": 5.5, "q5_k_m": 5.5,
    "4bit": 4.8, "q4": 4.8, "q4_k_m": 4.8, "int4": 4.5, "nf4": 4.5, "awq": 4.3,
    "3bit": 3.4, "q3": 3.4, "q3_k_m": 3.4,
    "2bit": 2.6, "q2": 2.6, "q2_k": 2.6,
}

# KV cache, fp16, per 1000 tokens per billion *active* parameters.  Calibrated
# against Qwen3-8B (36 layers, 8 KV heads, 128 head dim); accurate to roughly a
# factor of two across GQA models, which is all a budget check needs.
_KV_GB_PER_1K_PER_B = 0.018
# Runtime slack: allocator overhead, activations, the framework itself.
_WEIGHT_OVERHEAD = 1.08
_RUNTIME_BASE_GB = 0.7
# Default context for footprint maths: the model's native window can be 262k,
# and budgeting for a 262k KV cache would condemn every model on the list.
_FOOTPRINT_CONTEXT = 8192


def quantisation_bits(quantisation: Any) -> float:
    """Effective bits per weight for a quantisation name (fp16 if unknown)."""
    key = str(quantisation or "").strip().lower().replace("-", "_")
    if key in _BITS:
        return _BITS[key]
    match = re.match(r"^q?(\d+)", key)
    if match:
        bits = float(match.group(1))
        if 1.0 <= bits <= 32.0:
            return bits + 0.8 if bits < 8 else bits
    return _BITS[""]


def _positive_float(value: Any) -> float:
    """Best-effort coercion to a positive float, or ``0.0`` for anything else.

    Used to normalise ``vram_gb`` arguments: ``None``, ``0``, a negative number,
    or an unparsable value are all treated identically as "no VRAM figure was
    given", so callers never have to pre-validate what they pass in.
    """
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out > 0 else 0.0


def estimate_footprint(spec: Any, quantisation: str = "q4", *,
                       context: Optional[int] = None,
                       vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """Rough download and RAM figures for running ``spec`` at ``quantisation``.

    ``spec`` may be a :class:`ModelSpec` or any string :func:`resolve` accepts.
    Figures are deliberately approximate — their job is to say "this will not
    fit in 32 GB" before the download starts, not to predict the last megabyte.

    Returns a dict with ``download_gb``, ``weights_gb``, ``kv_cache_gb`` and
    ``ram_gb`` (all 10^9 bytes), plus the inputs used to compute them.

    ``vram_gb``, when given and positive, adds two more keys: ``fits_vram``
    (bool) and ``vram_headroom_gb`` (float, may be negative) — whether the
    weights plus KV cache alone (no system RAM) fit in that much VRAM. Omit it
    (the default) and the returned dict is byte-for-byte what earlier callers
    already depend on.
    """
    if not isinstance(spec, ModelSpec):
        try:
            spec = resolve(spec, allow_unreleased=True)
        except ModelError as exc:
            return {
                "model": str(spec),
                "error": redact(exc),
                "download_gb": 0.0,
                "weights_gb": 0.0,
                "kv_cache_gb": 0.0,
                "ram_gb": 0.0,
            }

    bits = quantisation_bits(quantisation)
    ctx = int(context or min(spec.context or _FOOTPRINT_CONTEXT, _FOOTPRINT_CONTEXT))
    ctx = max(ctx, 1)

    weights_gb = spec.params * bits / 8.0
    kv_gb = (ctx / 1000.0) * spec.effective_params * _KV_GB_PER_1K_PER_B
    ram_gb = weights_gb * _WEIGHT_OVERHEAD + kv_gb + _RUNTIME_BASE_GB

    result = {
        "model": spec.id,
        "label": spec.label,
        "quantisation": str(quantisation or "fp16").lower(),
        "bits_per_weight": round(bits, 2),
        "params_b": spec.params,
        "active_params_b": spec.effective_params,
        "context": ctx,
        "weights_gb": round(weights_gb, 2),
        "download_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_gb, 2),
        "ram_gb": round(ram_gb, 2),
        "note": "Estimates in 10^9-byte GB; assumes weights held in RAM.",
    }

    vgb = _positive_float(vram_gb)
    if vgb:
        headroom = vgb - (weights_gb + kv_gb)
        result["fits_vram"] = headroom >= 0
        result["vram_headroom_gb"] = round(headroom, 2)

    return result


# Fraction of system RAM the weights may claim.  The rest is KV cache, the OS,
# the browser the owner will inevitably have open, and JARVIS itself.
_RAM_BUDGET_FRACTION = 0.6
# Above this many active parameters, CPU-only generation stops feeling like a
# conversation and starts feeling like a batch job.
_INTERACTIVE_CPU_ACTIVE_B = 6.0

_INTERACTIVE_PURPOSES = {"chat", "interactive", "assistant", "voice", "agent", "general"}
_CODE_PURPOSES = {"code", "coding", "dev", "programming"}
_QUALITY_PURPOSES = {"quality", "reasoning", "deep", "batch", "analysis", "research"}
_SMALL_PURPOSES = {"tiny", "fast", "draft", "small", "cheap"}


def _purpose_score(spec: ModelSpec, *, wants_code: bool) -> tuple:
    """Shared ranking key: purpose match first, then biggest-but-cheapest-per-token."""
    specialised = "coder" in spec.family or "coder" in spec.id.lower()
    purpose_fit = 1 if specialised == wants_code else 0
    return (purpose_fit, spec.params, -spec.effective_params, spec.id)


def recommend(ram_gb: float = 32.0, has_gpu: bool = False,
              purpose: str = "chat", *, vram_gb: Optional[float] = None) -> ModelSpec:
    """Pick the best catalogue model that will actually run on this machine.

    Sizing is honest rather than aspirational: weights get at most 60% of
    system RAM, and on a CPU-only box an interactive model must keep its
    *active* parameter count small — which is why a 32 GB laptop is offered the
    30B-A3B mixture-of-experts (30B of knowledge, ~3B of arithmetic per token)
    and not the dense 32B that would technically almost fit and then answer at
    a word every few seconds.

    ``vram_gb``, when given and positive, is consulted first: a model whose Q4
    weights *and* KV cache both fit inside that much VRAM (via
    :func:`estimate_footprint`) is strongly preferred, because it runs entirely
    resident on the GPU rather than partially offloaded. When nothing fits
    whole, a mixture-of-experts model is preferred if one fits the RAM budget
    — Ollama offloads MoE expert layers to the GPU partially and keeps the
    rest in system RAM, which is a perfectly good outcome for "GPU present but
    not big enough for this model", not a failure. Only when neither of those
    applies does this fall back to the plain ``ram_gb``/``has_gpu`` logic
    below, exactly as if ``vram_gb`` had never been passed.

    Never raises: with an absurdly small budget it returns the smallest model
    in the catalogue.
    """
    try:
        budget = max(0.1, float(ram_gb) * _RAM_BUDGET_FRACTION)
    except (TypeError, ValueError):
        budget = 32.0 * _RAM_BUDGET_FRACTION

    key = normalise_alias(purpose) or "chat"
    candidates = [s for s in KNOWN_MODELS.values() if s.exists and not s.gated]
    wants_code = key in _CODE_PURPOSES

    vgb = _positive_float(vram_gb)
    if vgb:
        vram_fits = [
            s for s in candidates
            if estimate_footprint(s, "q4", vram_gb=vgb).get("fits_vram")
        ]
        if vram_fits:
            if key in _SMALL_PURPOSES:
                return min(vram_fits, key=lambda s: (s.params, s.id))
            return max(vram_fits, key=lambda s: _purpose_score(s, wants_code=wants_code))

        # Nothing fits whole in VRAM: a mixture-of-experts model that fits the
        # RAM budget is the honest recommendation for "GPU present but not big
        # enough" — see the docstring above. Falls through to the plain
        # ram_gb/has_gpu logic below when no MoE model fits either.
        moe_fits = [s for s in candidates if s.is_moe and s.quantised_size_gb <= budget]
        if moe_fits and key not in _SMALL_PURPOSES:
            return max(moe_fits, key=lambda s: _purpose_score(s, wants_code=wants_code))

    fits = [s for s in candidates if s.quantised_size_gb <= budget]
    if not fits:
        return min(candidates, key=lambda s: (s.quantised_size_gb, s.params))

    if key in _SMALL_PURPOSES:
        return min(fits, key=lambda s: (s.params, s.id))

    interactive = key in _INTERACTIVE_PURPOSES or key in _CODE_PURPOSES
    if interactive and not has_gpu:
        responsive = [s for s in fits if s.effective_params <= _INTERACTIVE_CPU_ACTIVE_B]
        if responsive:
            fits = responsive

    best = max(fits, key=lambda s: _purpose_score(s, wants_code=wants_code))
    if key in _QUALITY_PURPOSES and not has_gpu:
        logger.debug(
            "Recommending %s for %r on CPU: expect batch-speed, not conversation.",
            best.id, purpose,
        )
    return best


# --------------------------------------------------------------------------- #
#  Credentials
# --------------------------------------------------------------------------- #
def _token_file_candidates(environ: Dict[str, str]) -> List[Path]:
    """Where ``huggingface-cli login`` may have cached a token."""
    paths: List[Path] = []
    explicit = (environ.get(_TOKEN_FILE_ENV) or "").strip()
    if explicit:
        paths.append(Path(explicit))
    hf_home = (environ.get("HF_HOME") or "").strip()
    if hf_home:
        paths.append(Path(hf_home) / "token")
    xdg = (environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg:
        paths.append(Path(xdg) / "huggingface" / "token")
    try:
        home = Path.home()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return paths
    paths.append(home / ".cache" / "huggingface" / "token")
    paths.append(home / ".huggingface" / "token")
    return paths


def hf_token(cfg: Any = None, *, environ: Optional[dict] = None) -> Optional[str]:
    """Find a Hugging Face token, or return ``None``.

    Precedence, first non-empty wins:

    1. ``cfg.llm.hf_token`` (config file or ``JARVIS_LLM_HF_TOKEN``)
    2. ``HF_TOKEN``
    3. ``HUGGING_FACE_HUB_TOKEN``
    4. the file written by ``huggingface-cli login``

    An empty or whitespace-only value at any level is treated as absent, so a
    blank config field cannot shadow a perfectly good environment variable.
    Returns ``None`` rather than ``""`` — callers test truthiness, and an empty
    string sent as a bearer token produces a baffling 401.
    """
    llm = getattr(cfg, "llm", cfg) if cfg is not None else None
    configured = str(getattr(llm, "hf_token", "") or "").strip()
    if configured:
        return configured

    env = dict(environ) if environ is not None else dict(os.environ)
    for name in HF_TOKEN_ENV_VARS:
        value = str(env.get(name) or "").strip()
        if value:
            return value

    for path in _token_file_candidates(env):
        try:
            if path.is_file():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except (OSError, ValueError, UnicodeDecodeError):
            logger.debug("Unreadable HF token file: %s", path)
    return None


# --------------------------------------------------------------------------- #
#  Access check
# --------------------------------------------------------------------------- #
def _access(status: str, repo: str, url: str, message: str, *,
            http_status: Optional[int] = None, authenticated: bool = False,
            gated: bool = False) -> Dict[str, Any]:
    """Build the (token-free) result dict shared by every branch."""
    return {
        "status": status,
        "ok": status == "ok",
        "repo": repo,
        "url": redact(url),
        "http_status": http_status,
        "authenticated": authenticated,
        "gated": gated,
        "message": redact(message),
    }


def check_access(spec: Any, token: Optional[str] = None, *,
                 timeout: float = HF_API_TIMEOUT) -> Dict[str, Any]:
    """Ask Hugging Face whether these weights can actually be downloaded.

    One cheap API call, before the 30 GB one.  ``status`` is exactly one of:

    ``ok``          the repo exists and this token (or none) may read it
    ``gated``       it exists but the licence has not been accepted
    ``needs_token`` authentication is required and none was supplied
    ``not_found``   no such repo — a typo, or private and invisible to us
    ``offline``     huggingface.co could not be reached at all

    Never raises, never logs the token, and never puts the token in the
    returned dict; the URL and any underlying error text go through
    :func:`redact` first.
    """
    if isinstance(spec, ModelSpec):
        repo = spec.id
        known_gated = spec.gated
        exists = spec.exists
    else:
        repo = str(spec or "").strip()
        known_gated = False
        exists = True

    page = f"{HF_ENDPOINT}/{repo}"
    url = f"{HF_ENDPOINT}/api/models/{repo}"
    tok = str(token or "").strip() or None
    authed = tok is not None

    if not is_hf_repo_id(repo):
        return _access(
            "not_found", repo, url,
            f"{repo!r} is not a Hugging Face repo id ('org/name'), so there is "
            "nothing to check. Ollama tags are checked by the Ollama daemon.",
            authenticated=authed, gated=known_gated,
        )
    if not exists:
        return _access(
            "not_found", repo, url,
            f"{repo} is not released yet; the repository does not exist.",
            authenticated=authed, gated=known_gated,
        )

    try:
        import urllib.error
        import urllib.request
    except ImportError:  # pragma: no cover - urllib is stdlib
        return _access("offline", repo, url, "urllib is unavailable in this interpreter.",
                       authenticated=authed, gated=known_gated)

    headers = {"User-Agent": "jarvis/1.0"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    code: Optional[int] = None
    body = ""
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            code = int(getattr(response, "status", None) or response.getcode() or 0)
            try:
                raw = response.read(65536)
                body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            except (OSError, ValueError, AttributeError):
                body = ""
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
    except urllib.error.URLError as exc:
        reason = redact(getattr(exc, "reason", exc))
        return _access(
            "offline", repo, url,
            f"Could not reach huggingface.co ({reason}). "
            "Working offline is fine if the weights are already downloaded.",
            authenticated=authed, gated=known_gated,
        )
    except Exception as exc:  # noqa: BLE001 - a check must never take the app down
        logger.debug("check_access failed for %s: %s", repo, type(exc).__name__)
        return _access(
            "offline", repo, url,
            f"Access check failed ({type(exc).__name__}: {redact(exc)}).",
            authenticated=authed, gated=known_gated,
        )

    if code == 200:
        gated = known_gated or _body_is_gated(body)
        return _access(
            "ok", repo, url,
            f"{repo} is reachable and downloadable"
            + (" (gated, and this token has access)." if gated else "."),
            http_status=code, authenticated=authed, gated=gated,
        )
    if code == 401:
        if authed:
            return _access(
                "gated", repo, url,
                f"The supplied token was rejected for {repo}. Either it is not a "
                f"valid token, or access has not been granted: open {page} and "
                "accept the licence with the same account.",
                http_status=code, authenticated=authed, gated=True,
            )
        return _access(
            "needs_token", repo, url,
            f"{repo} requires authentication. Create a read token at "
            f"{HF_ENDPOINT}/settings/tokens and set HF_TOKEN (or llm.hf_token).",
            http_status=code, authenticated=authed, gated=known_gated,
        )
    if code == 403:
        return _access(
            "gated", repo, url,
            f"This repo is gated: accept the licence at {page}"
            + ("." if authed else " and set HF_TOKEN (or llm.hf_token)."),
            http_status=code, authenticated=authed, gated=True,
        )
    if code == 404:
        hint = ("" if authed else
                " If it is private, set HF_TOKEN (or llm.hf_token) and try again.")
        return _access(
            "not_found", repo, url,
            f"No repository named {repo} on Hugging Face. Check the spelling."
            + hint,
            http_status=code, authenticated=authed, gated=known_gated,
        )
    return _access(
        "offline", repo, url,
        f"Hugging Face answered HTTP {code}; try again shortly.",
        http_status=code, authenticated=authed, gated=known_gated,
    )


def _body_is_gated(body: str) -> bool:
    """Read the ``gated`` flag out of an API response without needing json schema."""
    if not body:
        return False
    try:
        import json
        data = json.loads(body)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    value = data.get("gated", False)
    return bool(value) and value is not False


# --------------------------------------------------------------------------- #
#  What is already on disk
# --------------------------------------------------------------------------- #
# Walking a model cache is cheap; walking a symlinked loop is not.  The cap is
# resource management, not policy: it bounds the work, never the reach.
_MAX_SCAN_ENTRIES = 200_000

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".onnx", ".npz")


def hf_cache_dir(environ: Optional[dict] = None) -> Optional[Path]:
    """Where the Hugging Face hub cache lives on this machine, if anywhere."""
    env = dict(environ) if environ is not None else dict(os.environ)
    explicit = (env.get("HF_HUB_CACHE") or "").strip()
    if explicit:
        return Path(explicit)
    hf_home = (env.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = (env.get("XDG_CACHE_HOME") or "").strip()
    if xdg:
        return Path(xdg) / "huggingface" / "hub"
    try:
        return Path.home() / ".cache" / "huggingface" / "hub"
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return None


def _dir_size(path: Path) -> Tuple[int, int]:
    """``(bytes, files)`` under ``path``; bounded, symlink-safe, never raises."""
    total = 0
    files = 0
    try:
        for root, dirnames, filenames in os.walk(str(path), followlinks=False):
            for name in filenames:
                if files >= _MAX_SCAN_ENTRIES:
                    dirnames[:] = []
                    return total, files
                files += 1
                try:
                    total += os.stat(os.path.join(root, name), follow_symlinks=True).st_size
                except OSError:
                    continue
    except OSError:
        logger.debug("Could not size %s", path)
    return total, files


def _cache_repo_id(dir_name: str) -> str:
    """``models--Qwen--Qwen3-8B`` -> ``Qwen/Qwen3-8B``."""
    body = dir_name[len("models--"):]
    return body.replace("--", "/", 1).replace("--", "/") if "--" in body else body


def _entry(name: str, path: Path, source: str) -> Dict[str, Any]:
    size, files = _dir_size(path) if path.is_dir() else (_file_size(path), 1)
    spec = lookup(name)
    return {
        "id": name,
        "label": spec.label if spec else name,
        "alias": spec.alias if spec else "",
        "path": str(path),
        "source": source,
        "size_bytes": size,
        "size_gb": round(size / 1e9, 2),
        "files": files,
    }


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def local_models(cfg: Any = None, *, environ: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Everything already downloaded, so the CLI can show what is on disk.

    Looks in two places: ``cfg.models_dir()`` (where AirLLM layer shards and
    hand-placed GGUF files live) and the Hugging Face hub cache.  Returns a
    list of plain dicts sorted largest-first; unreadable directories are
    skipped rather than raised.  Contains no credentials of any kind.
    """
    found: Dict[str, Dict[str, Any]] = {}

    roots: List[Tuple[Path, str]] = []
    models_dir = None
    if cfg is not None:
        try:
            models_dir = Path(cfg.models_dir())
        except (AttributeError, OSError, TypeError, ValueError):
            models_dir = None
    if models_dir is not None:
        roots.append((models_dir, "jarvis"))

    cache = hf_cache_dir(environ)
    if cache is not None:
        roots.append((cache, "hf-cache"))

    for root, source in roots:
        try:
            if not root.is_dir():
                continue
            children = sorted(root.iterdir())
        except OSError:
            logger.debug("Could not list %s", root)
            continue
        for child in children:
            try:
                if child.is_dir():
                    name = (
                        _cache_repo_id(child.name)
                        if child.name.startswith("models--")
                        else child.name
                    )
                    entry = _entry(name, child, source)
                elif child.suffix.lower() in _WEIGHT_SUFFIXES:
                    entry = _entry(child.stem, child, source)
                else:
                    continue
            except OSError:
                continue
            key = f"{source}:{entry['id']}"
            if key not in found:
                found[key] = entry

    return sorted(found.values(), key=lambda e: (-e["size_bytes"], e["id"]))


def describe(spec: ModelSpec) -> str:
    """One-line summary for a CLI listing."""
    bits = [
        f"{spec.params:g}B" if spec.params else "size unknown",
        f"{spec.context // 1024}k ctx" if spec.context >= 1024 else f"{spec.context} ctx",
        f"~{spec.quantised_size_gb:g} GB q4" if spec.quantised_size_gb else "",
    ]
    flags = []
    if spec.is_moe:
        flags.append(f"MoE {spec.effective_params:g}B active")
    if spec.gated:
        flags.append("gated")
    if not spec.exists:
        flags.append("NOT RELEASED")
    tail = f" [{', '.join(flags)}]" if flags else ""
    return f"{spec.label} ({spec.id}) — " + ", ".join(b for b in bits if b) + tail


def with_revision(spec: ModelSpec, revision: str) -> ModelSpec:
    """Copy of ``spec`` whose label records the pinned revision."""
    rev = (revision or "").strip()
    if not rev:
        return spec
    return replace(spec, label=f"{spec.label} @{rev}")


def iter_aliases(specs: Optional[Iterable[ModelSpec]] = None) -> List[str]:
    """Catalogue keys, sorted — for shell completion and error messages."""
    if specs is None:
        return sorted(KNOWN_MODELS)
    return sorted({s.alias for s in specs if s.alias})


# --------------------------------------------------------------------------- #
#  Catalogue post-processing: stamp each entry with its own alias and build the
#  reverse indices, using the same normalisation `lookup` applies to input.
# --------------------------------------------------------------------------- #
def _build_indices() -> None:
    for alias, spec in list(KNOWN_MODELS.items()):
        if not spec.alias:
            spec = replace(spec, alias=alias)
            KNOWN_MODELS[alias] = spec
        _ID_INDEX[normalise_alias(spec.id)] = alias
        if spec.ollama_tag:
            _TAG_INDEX.setdefault(normalise_alias(spec.ollama_tag), alias)


_build_indices()


__all__ = [
    "ModelSpec",
    "ModelError",
    "UnknownModelError",
    "UnreleasedModelError",
    "KNOWN_MODELS",
    "DEFAULT_ALIAS",
    "HF_ENDPOINT",
    "HF_TOKEN_ENV_VARS",
    "all_models",
    "check_access",
    "describe",
    "estimate_footprint",
    "hf_cache_dir",
    "hf_token",
    "is_hf_repo_id",
    "is_ollama_tag",
    "iter_aliases",
    "local_models",
    "lookup",
    "normalise_alias",
    "quantisation_bits",
    "recommend",
    "redact",
    "resolve",
    "resolve_config",
    "with_revision",
]
