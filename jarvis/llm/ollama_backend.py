"""Ollama HTTP backend, stdlib-only.

Ollama's REST API on http://127.0.0.1:11434 is a great fit for JARVIS on a
32 GB laptop: a 4-bit quantised Qwen3-30B-A3B MoE fits comfortably and does
~4-8 tokens/sec on CPU, which is realtime enough for spoken conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, List, Optional, Sequence
from urllib import error as _urlerror
from urllib import request as _urlrequest

from ..core.contracts import GenerationConfig, LLMResult, Message
from ..core.config import LLMConfig
from .base import (
    BaseLLM,
    apply_stop_strings,
    salvage_thinking,
    strip_thinking,
    wants_thinking_disabled,
)


logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = 1.5


def _default_thread_count() -> int:
    """Physical cores, which is what a bandwidth-bound workload wants.

    Hyperthreading does not help here: two threads on one physical core share
    a single memory port, so the pair contends for the exact resource that is
    already the bottleneck. On a 4-core/8-thread i5 the right answer is 4, and
    Ollama's own default of "all logical processors" is measurably worse.
    """
    try:
        import os

        physical = 0
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                ids = {
                    line.split(":", 1)[1].strip()
                    for line in fh
                    if line.lower().startswith("core id")
                }
            physical = len(ids)
        except OSError:
            physical = 0
        if physical > 0:
            return physical
        logical = os.cpu_count() or 0
        # No topology available: assume SMT and halve, floor of 1.
        return max(1, logical // 2) if logical > 1 else 1
    except Exception:  # noqa: BLE001 - a tuning hint must never break a call
        return 0


def _describe_ollama_error(exc: "_urlerror.URLError") -> str:
    """The useful half of a failed Ollama request, which ``str(exc)`` throws away.

    ``HTTPError`` (a ``URLError`` subclass) IS the response body -- it is a
    file-like object -- but nothing was ever reading it. Ollama's 4xx/5xx
    responses are JSON with a real diagnosis (``{"error": "model requires
    more system memory (X GiB) than is available (Y GiB)"}`` is the single
    most common one on a CPU box with a context window sized past what the
    machine can actually hold), so "HTTP Error 500: Internal Server Error"
    was hiding the one fact that would have made the failure self-explanatory
    instead of needing a bug report to decode.
    """
    body = ""
    if isinstance(exc, _urlerror.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 - the body is a bonus, not a requirement
            body = ""
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("error"):
                return f"{exc} -- {parsed['error']}"
        except json.JSONDecodeError:
            pass
        return f"{exc} -- {body[:500]}"
    return str(exc)


class OllamaBackend(BaseLLM):
    """Talks to a local Ollama daemon over HTTP."""

    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._host = (cfg.ollama_host or "http://127.0.0.1:11434").rstrip("/")
        self._model = cfg.ollama_model or cfg.model
        self._timeout = float(cfg.request_timeout or 600.0)
        # None = leave the model's own default alone; False = explicitly off.
        self._think: Optional[bool] = (
            False if wants_thinking_disabled(self._model) else None
        )

    # -- HTTP helpers ------------------------------------------------------- #
    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._host}{path}"

    def _post(self, path: str, payload: dict, *, timeout: Optional[float] = None):
        data = json.dumps(payload).encode("utf-8")
        req = _urlrequest.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _urlrequest.urlopen(req, timeout=timeout or self._timeout)

    def _get(self, path: str, *, timeout: Optional[float] = None):
        req = _urlrequest.Request(self._url(path), method="GET")
        return _urlrequest.urlopen(req, timeout=timeout or self._timeout)

    # -- availability ------------------------------------------------------- #
    def is_available(self) -> bool:
        try:
            with self._get("/api/tags", timeout=_PROBE_TIMEOUT) as resp:
                resp.read(1)
            return True
        except Exception:
            return False

    def _do_load(self) -> None:
        return None

    # -- options ------------------------------------------------------------ #
    def _build_options(self, gen: GenerationConfig) -> dict:
        # Ollama defaults to a 4096-token context and silently truncates when
        # the prompt exceeds it — always pass num_ctx explicitly.
        options: dict = {
            "num_ctx": int(self.cfg.context_tokens),
            "num_predict": int(gen.max_new_tokens),
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "top_k": int(gen.top_k),
        }

        # -- CPU tuning ----------------------------------------------------- #
        # Dense CPU inference is memory-bandwidth bound, not compute bound, and
        # these three settings are the difference between using the machine and
        # fighting it.
        threads = int(getattr(self.cfg, "num_threads", 0) or 0)
        if threads <= 0:
            threads = _default_thread_count()
        if threads > 0:
            options["num_thread"] = threads

        if getattr(self.cfg, "use_mmap", True):
            # Weights are mapped from the page cache instead of copied into the
            # process, so a second run starts warm rather than re-reading GB
            # from disk.
            options["use_mmap"] = True
        if getattr(self.cfg, "use_mlock", False):
            # Pins weights in RAM. Only worth it with headroom to spare; on a
            # tight machine it triggers swapping, which is far worse.
            options["use_mlock"] = True
        if gen.stop:
            options["stop"] = list(gen.stop)
        if gen.seed is not None:
            options["seed"] = int(gen.seed)
        return options

    def _payload(self, messages: Sequence[Message], gen: GenerationConfig, *, stream: bool) -> dict:
        payload = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": bool(stream),
            "options": self._build_options(gen),
        }
        # Qwen3.x reasons by default and will spend the entire num_predict
        # budget inside an unclosed <think> block, leaving nothing to say. On
        # a CPU-only box that is not a quality trade-off, it is the difference
        # between an answer and silence. Ollama exposes this as a top-level
        # boolean; older daemons ignore an unknown key rather than erroring.
        if self._think is False:
            payload["think"] = False
        return payload

    # -- generate ----------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:
        self.load()
        gen = self._gen_config(config)
        messages = self._fit(messages)
        payload = self._payload(messages, gen, stream=False)
        try:
            with self._post("/api/chat", payload) as resp:
                body = resp.read().decode("utf-8", "replace")
        except _urlerror.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {_describe_ollama_error(exc)}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {exc}") from exc

        raw_text = ""
        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict):
            raw_text = str(message.get("content") or "")
            # Newer daemons return reasoning on its own field rather than in
            # <think> tags; it must never be spoken as the answer.
            if not raw_text:
                thinking = message.get("thinking") or message.get("reasoning")
                if thinking:
                    raw_text = f"<think>{thinking}"
        text = strip_thinking(raw_text)
        if not text:
            # The whole budget went into reasoning. Salvage prose from it
            # rather than hand the agent loop an empty string.
            text = salvage_thinking(raw_text)
        text = apply_stop_strings(text, gen.stop)

        finish = "stop"
        if isinstance(data, dict) and data.get("done_reason"):
            finish = str(data["done_reason"])
        prompt_tokens = int(data.get("prompt_eval_count") or 0) if isinstance(data, dict) else 0
        completion_tokens = int(data.get("eval_count") or 0) if isinstance(data, dict) else 0

        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish,
            raw=data,
        )

    # -- stream ------------------------------------------------------------- #
    def stream(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        self.load()
        gen = self._gen_config(config)
        messages = self._fit(messages)
        payload = self._payload(messages, gen, stream=True)
        stops = tuple(gen.stop or ())
        buf = ""
        try:
            resp = self._post("/api/chat", payload)
        except _urlerror.URLError as exc:
            raise RuntimeError(
                f"Ollama streaming request failed: {_describe_ollama_error(exc)}"
            ) from exc
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = obj.get("message") if isinstance(obj, dict) else None
                if isinstance(message, dict):
                    delta = str(message.get("content") or "")
                    if delta:
                        buf += delta
                        cleaned = strip_thinking(buf)
                        if stops:
                            trimmed = apply_stop_strings(cleaned, stops)
                            if trimmed != cleaned:
                                if trimmed:
                                    yield trimmed
                                return
                        if delta:
                            yield delta
                if isinstance(obj, dict) and obj.get("done"):
                    break
        finally:
            try:
                resp.close()
            except Exception:  # pragma: no cover
                pass

    # -- introspection ------------------------------------------------------ #
    def list_models(self) -> List[str]:
        """Return the model tags installed on the daemon."""
        try:
            with self._get("/api/tags", timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except _urlerror.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}: "
                f"{_describe_ollama_error(exc)}. "
                f"Install and start it from https://ollama.com."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}: {exc}. "
                f"Install and start it from https://ollama.com."
            ) from exc
        if not isinstance(data, dict):
            return []
        return [str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]


__all__ = ["OllamaBackend"]
