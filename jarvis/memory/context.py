"""Live conversation window + long-term recall glue.

Every turn is appended to a bounded in-memory window *and* persisted to the
store as a ``conversation`` memory.  When the live window outgrows
``cfg.summarize_after_turns`` the oldest tail is compressed — via the LLM if
one is provided, otherwise with a deterministic extractive fallback — and the
resulting summary is itself persisted (kind ``"summary"``) so it survives
restarts.

The whole point of this subsystem is that nothing important is lost, so a
persistence failure never fails silently: the write is retried once, a record
is appended to :attr:`ContextManager.failed_writes`, and
:attr:`ContextManager.persistence_healthy` flips to ``False`` for the
supervisor to notice.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from ..core.contracts import (
    GenerationConfig,
    LLMBackend,
    Message,
    MemoryRecord,
    MemoryStore,
    Role,
    now_ts,
)

log = logging.getLogger(__name__)


class ContextManager:
    """Maintains the live conversation window and drives long-term recall."""

    def __init__(
        self,
        store: MemoryStore,
        cfg,
        *,
        llm: Optional[LLMBackend] = None,
        system_prompt: str = "",
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.llm = llm
        self.system_prompt = system_prompt or ""
        self._live: List[Message] = []
        self.summary: str = ""
        self.failed_writes: List[dict] = []

    # ------------------------------------------------------------------ #
    # Persistence health
    # ------------------------------------------------------------------ #
    @property
    def persistence_healthy(self) -> bool:
        """False as soon as a single write has ever failed this session."""
        return not self.failed_writes

    def _persist(self, kind: str, text: str, **metadata: Any) -> Optional[str]:
        last_err: Optional[BaseException] = None
        for attempt in range(2):
            try:
                add_text = getattr(self.store, "add_text", None)
                if callable(add_text):
                    record = add_text(kind, text, **metadata)
                    return getattr(record, "id", None)
                record = MemoryRecord(
                    id="", kind=kind, text=text,
                    metadata=dict(metadata or {}), ts=now_ts(),
                )
                return self.store.add(record)
            except Exception as exc:  # noqa: BLE001 - the whole promise is to never crash the caller
                last_err = exc
                log.warning(
                    "memory persistence attempt %d/2 failed (%s)",
                    attempt + 1, exc,
                )
        log.error(
            "failed to persist %s memory after retry: %s | text=%r",
            kind, last_err, text,
        )
        self.failed_writes.append({
            "kind": kind,
            "text": text,
            "metadata": dict(metadata or {}),
            "error": str(last_err),
            "ts": now_ts(),
        })
        return None

    # ------------------------------------------------------------------ #
    # Turn recording
    # ------------------------------------------------------------------ #
    def add_user(self, text: str) -> Message:
        msg = Message.user(text)
        self._live.append(msg)
        self._persist("conversation", text, role="user")
        return msg

    def add_assistant(self, text: str) -> Message:
        msg = Message.assistant(text)
        self._live.append(msg)
        self._persist("conversation", text, role="assistant")
        return msg

    def add_tool(
        self, text: str, name: str, tool_call_id: Optional[str] = None
    ) -> Message:
        msg = Message.tool(text, name=name, tool_call_id=tool_call_id)
        self._live.append(msg)
        self._persist(
            "conversation", text,
            role="tool", tool=name, tool_call_id=tool_call_id,
        )
        return msg

    # ------------------------------------------------------------------ #
    # Facts
    # ------------------------------------------------------------------ #
    def remember_fact(self, text: str, **metadata: Any) -> Optional[str]:
        return self._persist("fact", text, **metadata)

    def facts(self, k: int = 100) -> List[MemoryRecord]:
        try:
            return list(self.store.recent(k=k, kind="fact"))
        except Exception:  # noqa: BLE001
            log.exception("could not list facts")
            return []

    def forget(self, record_id: str) -> bool:
        deleter = getattr(self.store, "delete", None)
        if not callable(deleter):
            return False
        try:
            return bool(deleter(record_id))
        except Exception:  # noqa: BLE001
            log.exception("failed to forget %s", record_id)
            return False

    def history(self) -> List[Message]:
        return list(self._live)

    def clear_live(self) -> None:
        """Drop the in-memory window *without* touching persistent storage."""
        self._live.clear()
        self.summary = ""

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #
    def build(
        self, user_input: str, *, extra: Sequence[Message] = ()
    ) -> List[Message]:
        """Compose the chat transcript to send to the LLM.

        Order: system prompt, then any relevant recollections, then the
        rolling summary, then the recent live turns, then the caller's new
        user input.  Anything already visible verbatim in the recent window
        is stripped from the recollections block.
        """
        messages: List[Message] = []
        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))

        recall_k = int(getattr(self.cfg, "recall_k", 8))
        min_score = float(getattr(self.cfg, "recall_min_score", 0.15))
        keep = max(0, int(getattr(self.cfg, "keep_recent_turns", 8)))
        recent_texts = {
            m.content.strip()
            for m in (self._live[-keep:] if keep else [])
            if m.content
        }

        recollections: List[MemoryRecord] = []
        if user_input:
            try:
                recalled = self.store.search(user_input, k=recall_k)
            except Exception:  # noqa: BLE001
                log.exception("memory search failed")
                recalled = []
            for rec in recalled:
                if getattr(rec, "score", 0.0) < min_score:
                    continue
                if rec.text and rec.text.strip() in recent_texts:
                    continue
                recollections.append(rec)

        if recollections:
            body = "\n".join(f"- ({r.kind}) {r.text}" for r in recollections)
            messages.append(
                Message.system(f"Relevant recollections:\n{body}")
            )

        if self.summary:
            messages.append(
                Message.system(f"Conversation summary so far:\n{self.summary}")
            )

        if keep:
            for msg in self._live[-keep:]:
                messages.append(msg)
        else:
            for msg in self._live:
                messages.append(msg)

        for msg in extra:
            messages.append(msg)

        if user_input is not None:
            messages.append(Message.user(user_input))
        return messages

    # ------------------------------------------------------------------ #
    # Summarisation
    # ------------------------------------------------------------------ #
    def maybe_summarize(self) -> bool:
        """Compress the oldest turns into the rolling summary if we've grown
        past ``cfg.summarize_after_turns``.  Returns True when it ran."""
        threshold = int(getattr(self.cfg, "summarize_after_turns", 20))
        keep = max(0, int(getattr(self.cfg, "keep_recent_turns", 8)))
        if len(self._live) <= threshold:
            return False
        to_compress = self._live[:-keep] if keep else list(self._live)
        if not to_compress:
            return False
        block = "\n".join(f"{m.role.value}: {m.content}" for m in to_compress)
        new_summary = self._summarise_text(self.summary, block)
        self.summary = new_summary
        self._persist(
            "summary", new_summary,
            turns=len(to_compress), rolling=True,
        )
        self._live = self._live[-keep:] if keep else []
        return True

    def _summarise_text(self, prior: str, block: str) -> str:
        if self.llm is not None:
            try:
                system_prompt = (
                    "You maintain a rolling summary of a long conversation. "
                    "Preserve names, decisions, open questions, and any facts "
                    "worth remembering. Reply with the updated summary only."
                )
                user_prompt_parts: List[str] = []
                if prior:
                    user_prompt_parts.append("Prior summary:\n" + prior)
                user_prompt_parts.append("New turns to fold in:\n" + block)
                messages = [
                    Message.system(system_prompt),
                    Message.user("\n\n".join(user_prompt_parts)),
                ]
                result = self.llm.generate(
                    messages, GenerationConfig(max_new_tokens=256)
                )
                text = (getattr(result, "text", "") or "").strip()
                if text:
                    return text
            except Exception:  # noqa: BLE001
                log.exception(
                    "LLM summarisation failed; using extractive fallback"
                )
        lines: List[str] = []
        if prior:
            lines.append(prior)
        for raw in block.splitlines():
            lines.append(raw if len(raw) <= 240 else raw[:237] + "...")
        merged = "\n".join(lines)
        if len(merged) > 4000:
            merged = merged[-4000:]
        return merged


__all__ = ["ContextManager"]
