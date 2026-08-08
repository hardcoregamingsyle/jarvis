"""Zero-dependency stub backend used for tests and last-ditch fallback.

Produces a deterministic British-butler-style acknowledgement referencing the
last user message. Tests can inject a canned response list or callable.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterator, List, Optional, Sequence, Union

from ..core.contracts import GenerationConfig, LLMResult, Message, Role
from ..core.config import LLMConfig
from .base import BaseLLM, apply_stop_strings


Responder = Union[str, Sequence[str], Callable[[Sequence[Message]], str], None]


class StubBackend(BaseLLM):
    """A tiny, always-available backend.

    Constructor accepts:
      * ``responses=None`` — deterministic butler acknowledgement.
      * ``responses="one line"`` — always returns that line.
      * ``responses=[...]`` — cycles through the sequence, one per call.
      * ``responses=callable`` — invoked with the message list; returns text.

    Every call is recorded in ``self.calls`` (a list of message lists).
    """

    name = "stub"

    def __init__(self, cfg: Optional[LLMConfig] = None, responses: Responder = None) -> None:
        super().__init__(cfg or LLMConfig())
        self.calls: List[List[Message]] = []
        self._responses = responses
        self._cycle: Optional[itertools.cycle] = None
        if isinstance(responses, (list, tuple)) and responses:
            self._cycle = itertools.cycle(list(responses))

    def is_available(self) -> bool:
        return True

    def _last_user_text(self, messages: Sequence[Message]) -> str:
        for msg in reversed(list(messages)):
            if msg.role == Role.USER:
                return msg.content
        return ""

    def _pick(self, messages: Sequence[Message]) -> str:
        if callable(self._responses):
            return str(self._responses(list(messages)))
        if self._cycle is not None:
            return str(next(self._cycle))
        if isinstance(self._responses, str):
            return self._responses
        last = self._last_user_text(messages).strip()
        if last:
            snippet = last if len(last) <= 80 else last[:77] + "..."
            return f"Very good, Sir. Regarding \"{snippet}\", I shall attend to it promptly."
        return "At your service, Sir."

    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:
        self.load()
        gen = self._gen_config(config)
        self.calls.append(list(messages))
        text = self._pick(messages)
        text = apply_stop_strings(text, gen.stop)
        max_chars = max(4, int(gen.max_new_tokens) * 4)
        if len(text) > max_chars:
            text = text[:max_chars]
        return LLMResult(
            text=text,
            prompt_tokens=sum(max(1, len(m.content) // 4) for m in messages),
            completion_tokens=max(1, len(text) // 4),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        result = self.generate(messages, config)
        text = result.text or ""
        buf = ""
        for ch in text:
            buf += ch
            if ch.isspace():
                yield buf
                buf = ""
        if buf:
            yield buf


__all__ = ["StubBackend"]
