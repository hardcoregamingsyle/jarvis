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
        payload = self._payload(messages, gen, stream=False)
        try:
            with self._post("/api/chat", payload) as resp:
                body = resp.read().decode("utf-8", "replace")
        except _urlerror.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
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
        payload = self._payload(messages, gen, stream=True)
        stops = tuple(gen.stop or ())
        buf = ""
        try:
            resp = self._post("/api/chat", payload)
        except _urlerror.URLError as exc:
            raise RuntimeError(f"Ollama streaming request failed: {exc}") from exc
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
        except (_urlerror.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}: {exc}. "
                f"Install and start it from https://ollama.com."
            ) from exc
        if not isinstance(data, dict):
            return []
        return [str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]


__all__ = ["OllamaBackend"]
