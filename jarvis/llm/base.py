"""Shared helpers for LLM backends.

Chat formatting, token estimation, context trimming, Qwen3 thinking-block
handling, stop-string truncation, and a :class:`BaseLLM` base that reduces
every concrete backend to filling in :meth:`generate`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Iterator, List, Optional, Sequence

from ..core.contracts import (
    GenerationConfig,
    LLMBackend,
    LLMResult,
    Message,
    Role,
)
from ..core.config import LLMConfig


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Chat formatting
# --------------------------------------------------------------------------- #
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def format_chat(messages: Sequence[Message], style: str = "chatml") -> str:
    """Render a chat transcript as a single prompt string.

    ``chatml`` matches Qwen3 (and OpenAI's original ChatML): each turn is
    ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` with a trailing
    ``<|im_start|>assistant\\n`` inviting the model to continue.

    ``plain`` is a human-readable fallback for backends whose tokenizer
    does not know the ChatML tokens.
    """
    if style == "chatml":
        parts: List[str] = []
        for msg in messages:
            role = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
            parts.append(f"{_IM_START}{role}\n{msg.content}{_IM_END}\n")
        parts.append(f"{_IM_START}assistant\n")
        return "".join(parts)

    if style == "plain":
        lines: List[str] = []
        for msg in messages:
            role = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
            lines.append(f"{role.upper()}: {msg.content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    raise ValueError(f"Unknown chat format style: {style!r}")


# --------------------------------------------------------------------------- #
#  Qwen3 thinking-block handling
# --------------------------------------------------------------------------- #
_THINK_CLOSED = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
# NOTE: MUST use `[\s\S]*$` — a `[^<]*` variant fails whenever the aborted
# thought contains a "<" character, which happens routinely with code snippets.
_THINK_UNCLOSED_TRAILER = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)
_THINK_EXTRACT = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
# Any stray opening or closing tag, for cleaning salvaged fragments.
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` blocks, plus any unclosed trailer."""
    if not text:
        return text
    cleaned = _THINK_CLOSED.sub("", text)
    cleaned = _THINK_UNCLOSED_TRAILER.sub("", cleaned)
    return cleaned.strip()


def salvage_thinking(text: str) -> str:
    """Recover something sayable from a reply that is *only* reasoning.

    The failure this exists for is the single most common way a thinking model
    goes mute: generation hits ``max_new_tokens`` while still inside an
    unclosed ``<think>``, so :func:`strip_thinking` correctly returns ``""``
    and the agent loop has no answer to speak. Silence is the worst possible
    output for a voice assistant, so rather than discard the tokens we were
    charged for, we salvage the tail of the reasoning as prose.

    Returns ``""`` when there is genuinely nothing to salvage, so callers can
    still fall back to their own message.
    """
    if not text:
        return ""

    # Anything outside the thinking blocks always wins.
    outside = strip_thinking(text)
    if outside:
        return outside

    interior = extract_thinking(text) or ""
    if not interior:
        match = _THINK_UNCLOSED_TRAILER.search(text)
        interior = match.group(0)[len("<think>"):] if match else ""
    # An empty pair of tags leaves a stray closing tag behind, which must
    # never reach a speaker.
    interior = _THINK_TAG_RE.sub("", interior).strip()
    if not interior:
        return ""

    # Keep only complete sentences: a truncated model stops mid-word, and
    # reading a half-word aloud sounds broken rather than thoughtful.
    sentences = re.findall(r"[^.!?]*[.!?]", interior)
    if sentences:
        salvaged = " ".join(s.strip() for s in sentences[-3:] if s.strip())
    else:
        salvaged = interior
    return salvaged.strip()


def wants_thinking_disabled(model: Any) -> bool:
    """True when ``model`` is a family that reasons by default.

    Qwen3, Qwen3.5, Qwen3.6 and Qwen3.8 all ship with thinking enabled and
    burn hundreds of tokens before emitting a single word of the answer. On a
    CPU-only box that is the difference between a reply in seconds and no
    reply at all, so every backend turns it off for interactive turns.
    """
    name = str(model or "").strip().lower()
    if not name:
        return False
    from .models import THINKING_DEFAULT_ON

    return any(family in name for family in THINKING_DEFAULT_ON)


def extract_thinking(text: str) -> Optional[str]:
    """Return the concatenated interior of every closed ``<think>`` block."""
    if not text:
        return None
    matches = _THINK_EXTRACT.findall(text)
    if not matches:
        return None
    joined = "\n".join(m.strip() for m in matches if m.strip())
    return joined or None


# --------------------------------------------------------------------------- #
#  Token estimation & context trimming
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Rough token count without pulling in a tokenizer.

    ~4 characters per token is the OpenAI/BPE folklore average; it is wrong
    in either direction by a factor of two on pathological inputs but good
    enough for budget arithmetic when the context window has slack built in.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _msg_tokens(msg: Message) -> int:
    # 4 tokens of framing overhead per message is the ChatGPT-cookbook number.
    return estimate_tokens(msg.content) + 4


#: Marker left in place of the elided middle of an oversized message. Visible
#: on purpose: the model must know something was removed rather than silently
#: reasoning about a fragment it believes is complete.
#: Below this budget, eliding the middle of a message destroys more than
#: it saves, so an oversized message is passed through intact instead.
_MIN_TRUNCATION_BUDGET = 256

ELISION_MARKER = "\n\n... [{dropped:,} characters elided from the middle] ...\n\n"


def _truncate_message(msg: Message, max_tokens: int) -> Message:
    """Shrink one oversized message, keeping its head and tail."""
    budget_chars = max(200, (max_tokens - 4) * 4)
    text = msg.content or ""
    if len(text) <= budget_chars:
        return msg
    head = int(budget_chars * 0.6)
    tail = max(0, budget_chars - head - 80)
    dropped = len(text) - head - tail
    body = text[:head] + ELISION_MARKER.format(dropped=dropped) + (
        text[-tail:] if tail else ""
    )
    logger.warning(
        "a single message of ~%d tokens exceeded the %d-token window; "
        "elided %d characters from its middle",
        estimate_tokens(text), max_tokens, dropped,
    )
    return Message(role=msg.role, content=body, metadata=dict(msg.metadata or {}))


def trim_to_context(
    messages: Sequence[Message],
    max_tokens: int,
    keep_system: bool = True,
) -> List[Message]:
    """Trim a transcript to fit within ``max_tokens`` estimated tokens.

    Drops the oldest non-system turns first. Guarantees:
      * the returned list is never empty,
      * the system message (if any and ``keep_system``) is retained,
      * the last user message is retained,
    even if a single retained message already exceeds ``max_tokens``.
    """
    msgs = list(messages)
    if not msgs:
        return msgs

    system_msg: Optional[Message] = None
    if keep_system and msgs and msgs[0].role == Role.SYSTEM:
        system_msg = msgs[0]
        body = msgs[1:]
    else:
        body = msgs

    last_user_idx: Optional[int] = None
    for i in range(len(body) - 1, -1, -1):
        if body[i].role == Role.USER:
            last_user_idx = i
            break

    kept: List[Message] = []
    if last_user_idx is not None:
        kept.append(body[last_user_idx])
    elif body:
        kept.append(body[-1])

    # A single message larger than the whole window cannot be dropped -- it is
    # the user's actual question, or the tool output they are asking about --
    # so it is kept and truncated in the MIDDLE. Head and tail survive because
    # that is where a file's structure and its conclusion live; cutting the
    # tail alone loses the answer, and cutting the head loses the context.
    #
    # This is the difference between "your prompt was too long" and the server
    # silently discarding the system prompt, which is what happens when an
    # oversized transcript is sent unmodified.
    # Only worth doing when there is a real budget to fit into. Below this a
    # "window" is not a window, the caller is misconfigured, and mangling the
    # user's actual question to satisfy it helps nobody -- pass it through and
    # let the server complain.
    if (
        kept
        and max_tokens >= _MIN_TRUNCATION_BUDGET
        and _msg_tokens(kept[0]) > max_tokens
    ):
        kept[0] = _truncate_message(kept[0], max_tokens)

    def total(with_extras: List[Message]) -> int:
        extra = _msg_tokens(system_msg) if system_msg else 0
        return extra + sum(_msg_tokens(m) for m in with_extras)

    if last_user_idx is not None:
        for j in range(last_user_idx - 1, -1, -1):
            candidate = [body[j], *kept]
            if total(candidate) > max_tokens:
                break
            kept = candidate

    if kept and body:
        anchor = kept[-1]
        try:
            anchor_pos = body.index(anchor)
        except ValueError:
            anchor_pos = len(body) - 1
        for j in range(anchor_pos + 1, len(body)):
            candidate = [*kept, body[j]]
            if total(candidate) > max_tokens:
                break
            kept = candidate

    out: List[Message] = []
    if system_msg is not None:
        out.append(system_msg)
    out.extend(kept)
    return out or list(msgs[:1])


# --------------------------------------------------------------------------- #
#  Stop-string truncation
# --------------------------------------------------------------------------- #
def apply_stop_strings(text: str, stop: Sequence[str]) -> str:
    """Truncate ``text`` at the first occurrence of any string in ``stop``."""
    if not text or not stop:
        return text
    cut: Optional[int] = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx == -1:
            continue
        if cut is None or idx < cut:
            cut = idx
    return text if cut is None else text[:cut]


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #
class BaseLLM(LLMBackend):
    """Boilerplate shared by every real backend."""

    name: str = "base"

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._loaded = False

    def is_available(self) -> bool:  # pragma: no cover - subclasses override
        return False

    def load(self) -> None:
        """Idempotent load. Subclasses override :meth:`_do_load`."""
        if self._loaded:
            return
        self._do_load()
        self._loaded = True

    def _do_load(self) -> None:
        """Real load work. Base does nothing so tests can use it."""
        return None

    def _fit(self, messages: Sequence[Message]) -> List[Message]:
        """Trim a transcript to the configured window before sending it.

        Every backend calls this. Without it an oversized prompt is handed
        straight to the server, which silently drops tokens off the front --
        taking the system prompt with them. A truncation we perform is
        visible, ordered and logged; one the server performs is none of those.
        """
        limit = int(getattr(self.cfg, "context_tokens", 0) or 0)
        if limit <= 0:
            return list(messages)
        # Leave room for the reply: the window is shared between prompt and
        # completion, and a prompt that exactly fills it leaves nothing to
        # generate into.
        reserve = int(getattr(self.cfg, "max_new_tokens", 512) or 512)
        budget = max(256, limit - reserve)
        fitted = trim_to_context(messages, budget)
        if len(fitted) < len(messages):
            logger.info(
                "trimmed %d message(s) to fit a %d-token window",
                len(messages) - len(fitted), limit,
            )
        return fitted

    def _gen_config(self, config: Optional[GenerationConfig]) -> GenerationConfig:
        """Merge caller-supplied :class:`GenerationConfig` with cfg defaults."""
        base = GenerationConfig(
            max_new_tokens=int(self.cfg.max_new_tokens),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            top_k=int(self.cfg.top_k),
        )
        if config is None:
            return base
        merged = replace(
            base,
            max_new_tokens=int(config.max_new_tokens or base.max_new_tokens),
            temperature=float(config.temperature if config.temperature is not None else base.temperature),
            top_p=float(config.top_p if config.top_p is not None else base.top_p),
            top_k=int(config.top_k if config.top_k is not None else base.top_k),
            stop=tuple(config.stop or ()),
            seed=config.seed,
        )
        return merged

    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:  # pragma: no cover - subclasses override
        raise NotImplementedError

    def stream(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Naive default: chunk :meth:`generate` output into whitespace words."""
        result = self.generate(messages, config)
        text = result.text or ""
        if not text:
            return
        buf = ""
        for ch in text:
            buf += ch
            if ch.isspace():
                yield buf
                buf = ""
        if buf:
            yield buf

    def unload(self) -> None:
        self._loaded = False


__all__ = [
    "format_chat",
    "strip_thinking",
    "extract_thinking",
    "estimate_tokens",
    "trim_to_context",
    "ELISION_MARKER",
    "apply_stop_strings",
    "BaseLLM",
]
