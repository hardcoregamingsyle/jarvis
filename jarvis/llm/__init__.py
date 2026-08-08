"""LLM backend registry and factory.

Resolution order for ``create_llm(cfg)``:
    1. explicit ``backend=`` argument, else ``cfg.backend``.
    2. ``"auto"`` probes :data:`AUTO_PROBE_ORDER` and returns the first
       :meth:`is_available`.
    3. A named-but-unavailable backend falls through to auto when
       ``cfg.allow_fallback`` is true, otherwise raises ``RuntimeError``
       with an actionable install hint.

The probe order is deliberate:

* **vllm** first.  A vLLM server is never up by accident -- if one is listening
  it was started for JARVIS, and it is the only backend that serves an entire
  tree of concurrent agents from one resident copy of the weights (continuous
  batching).  Throughput under concurrency is what the agent tree needs.
* **ollama** next: the pragmatic default on a CPU-only laptop, and it
  parallelises reasonably with ``OLLAMA_NUM_PARALLEL>1``.
* **openai-compat** after that, for llama.cpp / LM Studio / TGI / hosted
  endpoints that happen to be configured.
* **transformers** for in-process weights.
* **airllm** last.  Its availability check is only an import test, so probing
  it earlier would cheerfully select the disk-paged backend (tens of seconds
  per token) even when a fully working server is running.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from ..core.config import LLMConfig
from ..core.contracts import LLMBackend
from .airllm_backend import AirLLMBackend
from .base import (
    BaseLLM,
    apply_stop_strings,
    estimate_tokens,
    extract_thinking,
    format_chat,
    strip_thinking,
    trim_to_context,
)
from .ollama_backend import OllamaBackend
from .openai_compat import OpenAICompatBackend
from .stub_backend import StubBackend
from .transformers_backend import TransformersBackend
from .vllm_backend import VLLMBackend


logger = logging.getLogger(__name__)


BackendFactory = Callable[[LLMConfig], BaseLLM]

BACKENDS: Dict[str, BackendFactory] = {
    "vllm": VLLMBackend,
    "ollama": OllamaBackend,
    "openai-compat": OpenAICompatBackend,
    "transformers": TransformersBackend,
    "airllm": AirLLMBackend,
    "stub": StubBackend,
}

# Names to probe when backend == "auto", in preference order.
AUTO_PROBE_ORDER = ("vllm", "ollama", "openai-compat", "transformers", "airllm")

_INSTALL_HINTS = {
    "airllm": "Install AirLLM with `pip install airllm`.",
    "ollama": (
        "Ollama is not reachable. Install from https://ollama.com and start it "
        "with `ollama serve` (default host http://127.0.0.1:11434)."
    ),
    "openai-compat": (
        "No OpenAI-compatible server answered. Point llm.openai_base_url at one "
        "(llama.cpp `llama-server`, LM Studio, TGI, or Ollama's /v1 shim on "
        "http://127.0.0.1:11434/v1)."
    ),
    "transformers": "Install with `pip install transformers torch`.",
    "vllm": (
        "No vLLM server answered on llm.vllm_host. vLLM is Linux-only: start it "
        "there (or in WSL2) with the argv from "
        "`jarvis.llm.vllm_backend.server_command(cfg)`, or point llm.vllm_host "
        "at the machine that runs it."
    ),
    "stub": "The stub backend has no dependencies and is always available.",
}


def available_backends(cfg: LLMConfig) -> List[str]:
    """Return the registered backends whose deps are currently satisfied."""
    out: List[str] = []
    for name, factory in BACKENDS.items():
        try:
            if factory(cfg).is_available():
                out.append(name)
        except Exception:
            logger.debug("is_available() raised for %s", name, exc_info=True)
    return out


def create_llm(cfg: LLMConfig, *, backend: Optional[str] = None) -> LLMBackend:
    """Instantiate the best available backend given ``cfg``."""
    requested = (backend or cfg.backend or "auto").strip().lower()

    if requested == "auto":
        return _auto_select(cfg)

    factory = BACKENDS.get(requested)
    if factory is None:
        if cfg.allow_fallback:
            logger.warning("Unknown LLM backend %r; falling back to auto.", requested)
            return _auto_select(cfg)
        raise RuntimeError(
            f"Unknown LLM backend {requested!r}. Known: {sorted(BACKENDS)}."
        )

    instance = factory(cfg)
    try:
        is_ok = instance.is_available()
    except Exception:
        logger.debug("is_available() raised for %s", requested, exc_info=True)
        is_ok = False

    if is_ok:
        return instance

    if cfg.allow_fallback:
        logger.warning(
            "Requested LLM backend %r unavailable; falling back to auto.",
            requested,
        )
        return _auto_select(cfg)

    hint = _INSTALL_HINTS.get(requested, "")
    raise RuntimeError(
        f"LLM backend {requested!r} is not available. {hint}"
    )


def _auto_select(cfg: LLMConfig) -> LLMBackend:
    for name in AUTO_PROBE_ORDER:
        factory = BACKENDS.get(name)
        if factory is None:
            continue
        try:
            instance = factory(cfg)
            if instance.is_available():
                logger.info("Auto-selected LLM backend: %s", name)
                return instance
        except Exception:
            logger.debug("Probe failed for %s", name, exc_info=True)
    logger.warning("No real LLM backend available; using stub.")
    return StubBackend(cfg)


__all__ = [
    "BACKENDS",
    "AUTO_PROBE_ORDER",
    "available_backends",
    "create_llm",
    # Backends
    "VLLMBackend",
    "OpenAICompatBackend",
    "OllamaBackend",
    "TransformersBackend",
    "AirLLMBackend",
    "StubBackend",
    # Helpers re-exported for convenience
    "BaseLLM",
    "apply_stop_strings",
    "estimate_tokens",
    "extract_thinking",
    "format_chat",
    "strip_thinking",
    "trim_to_context",
]
