"""Tests for the :class:`ContextManager` — the live conversation window
plus long-term recall glue."""

from __future__ import annotations

from typing import List

import pytest

from jarvis.core.config import Config
from jarvis.core.contracts import LLMResult, Message, Role
from jarvis.memory import (
    ContextManager,
    HashEmbedder,
    SQLiteMemoryStore,
    create_context,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class DummyLLM:
    """A tiny stand-in for an :class:`LLMBackend` — no deps, deterministic."""

    name = "dummy"

    def __init__(self, reply: str = "SUMMARY") -> None:
        self.reply = reply
        self.calls: List[list] = []

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def generate(self, messages, config=None) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=self.reply)

    def stream(self, messages, config=None):
        yield self.reply

    def unload(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "ctxhome"))
    (tmp_path / "ctxhome").mkdir(exist_ok=True)
    c = Config()
    c.memory.db_path = str(tmp_path / "memory.db")
    c.memory.embedder = "hash"
    c.memory.embed_dim = 64
    c.memory.recall_k = 5
    c.memory.recall_min_score = 0.0
    c.memory.summarize_after_turns = 6
    c.memory.keep_recent_turns = 2
    return c


@pytest.fixture
def store(cfg):
    s = SQLiteMemoryStore(
        cfg.memory.db_path,
        embedder=HashEmbedder(dim=cfg.memory.embed_dim),
    )
    try:
        yield s
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# build() ordering
# --------------------------------------------------------------------------- #
def test_build_orders_sections(store, cfg):
    ctx = ContextManager(store, cfg.memory, system_prompt="You are JARVIS.")
    store.add_text("fact", "The user's dog is Ada, a friendly golden retriever.")
    store.add_text("fact", "The user's favourite colour is teal.")
    ctx.add_user("Hello there")
    ctx.add_assistant("Good day.")
    ctx.summary = "Prior chat summary text."

    msgs = ctx.build("Tell me about my dog Ada")

    assert msgs[0].role == Role.SYSTEM
    assert "JARVIS" in msgs[0].content

    recall_idx = next(
        i for i, m in enumerate(msgs)
        if m.role == Role.SYSTEM and "Relevant recollections" in m.content
    )
    summary_idx = next(
        i for i, m in enumerate(msgs)
        if m.role == Role.SYSTEM and "Conversation summary" in m.content
    )
    hello_idx = next(
        i for i, m in enumerate(msgs) if m.content == "Hello there"
    )
    good_day_idx = next(
        i for i, m in enumerate(msgs) if m.content == "Good day."
    )

    assert 0 < recall_idx < summary_idx < hello_idx < good_day_idx
    assert msgs[-1].role == Role.USER
    assert msgs[-1].content == "Tell me about my dog Ada"


def test_build_excludes_verbatim_recent_turns_from_recollections(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    # These get persisted immediately, so a naive recall would surface them
    # again and produce awkward repetition.
    ctx.add_user("The password is swordfish")
    ctx.add_assistant("Noted.")

    msgs = ctx.build("What did I just tell you?")

    recall_blocks = [
        m for m in msgs
        if m.role == Role.SYSTEM and "Relevant recollections" in m.content
    ]
    for block in recall_blocks:
        assert "The password is swordfish" not in block.content
        assert "Noted." not in block.content


def test_build_omits_low_score_recollections(store, cfg):
    cfg.memory.recall_min_score = 0.99
    ctx = ContextManager(store, cfg.memory)
    store.add_text("fact", "totally unrelated background trivia")
    msgs = ctx.build("Something else entirely")
    recall_blocks = [
        m for m in msgs
        if m.role == Role.SYSTEM and "Relevant recollections" in m.content
    ]
    assert recall_blocks == []


def test_build_no_recall_when_user_input_empty(store, cfg):
    ctx = ContextManager(store, cfg.memory, system_prompt="sys")
    store.add_text("fact", "keep me")
    msgs = ctx.build("")
    assert not any("Relevant recollections" in m.content for m in msgs)


# --------------------------------------------------------------------------- #
# Turn recording -> persistence
# --------------------------------------------------------------------------- #
def test_turns_are_persisted_as_conversation(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    ctx.add_user("first line")
    ctx.add_assistant("reply")
    ctx.add_tool("tool output", name="calc", tool_call_id="call-1")
    assert store.count(kind="conversation") == 3
    conv = store.recent(k=10, kind="conversation")
    roles = {rec.metadata.get("role") for rec in conv}
    assert roles == {"user", "assistant", "tool"}
    for rec in conv:
        if rec.metadata.get("role") == "tool":
            assert rec.metadata.get("tool") == "calc"
            assert rec.metadata.get("tool_call_id") == "call-1"


# --------------------------------------------------------------------------- #
# Summarisation
# --------------------------------------------------------------------------- #
def test_summarize_at_threshold_preserves_recent(store, cfg):
    llm = DummyLLM(reply="Compact recap of earlier chat.")
    ctx = ContextManager(store, cfg.memory, llm=llm)
    for i in range(7):
        ctx.add_user(f"user #{i}")
        ctx.add_assistant(f"assistant #{i}")

    assert len(ctx.history()) == 14
    did = ctx.maybe_summarize()
    assert did is True

    kept = ctx.history()
    assert len(kept) == cfg.memory.keep_recent_turns == 2
    assert kept[-1].content == "assistant #6"
    assert kept[-2].content == "user #6"

    assert ctx.summary == "Compact recap of earlier chat."
    summaries = store.recent(k=5, kind="summary")
    assert summaries and summaries[0].text == "Compact recap of earlier chat."

    assert llm.calls, "LLM should have been asked to summarise"


def test_summarize_below_threshold_is_noop(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    ctx.add_user("only one turn")
    assert ctx.maybe_summarize() is False
    assert ctx.summary == ""
    assert store.count(kind="summary") == 0


def test_summarize_without_llm_uses_extractive_fallback(store, cfg):
    ctx = ContextManager(store, cfg.memory, llm=None)
    for i in range(7):
        ctx.add_user(f"user turn number {i}")
        ctx.add_assistant(f"assistant turn number {i}")
    assert ctx.maybe_summarize() is True
    assert ctx.summary, "extractive fallback must produce a non-empty summary"
    # The extractive fallback keeps some content from the compressed block.
    assert "turn number 0" in ctx.summary
    assert len(ctx.history()) == cfg.memory.keep_recent_turns


def test_summarize_when_llm_raises_falls_back(store, cfg):
    class BrokenLLM(DummyLLM):
        def generate(self, messages, config=None):
            raise RuntimeError("network down")

    ctx = ContextManager(store, cfg.memory, llm=BrokenLLM())
    for i in range(7):
        ctx.add_user(f"user #{i}")
        ctx.add_assistant(f"assistant #{i}")
    assert ctx.maybe_summarize() is True
    assert ctx.summary, "fallback must produce a summary even when LLM raises"


# --------------------------------------------------------------------------- #
# Persistence-failure signalling
# --------------------------------------------------------------------------- #
def test_persistence_failure_recorded_but_turn_succeeds(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    assert ctx.persistence_healthy is True

    class BrokenStore:
        def add_text(self, kind, text, **metadata):
            raise RuntimeError("disk full")

        def add(self, record):
            raise RuntimeError("still broken")

        def search(self, *a, **kw):
            return []

        def recent(self, **kw):
            return []

    ctx.store = BrokenStore()
    msg = ctx.add_user("something important")

    # The caller-visible Message is returned normally.
    assert isinstance(msg, Message) and msg.content == "something important"
    # The live window recorded the turn.
    assert ctx.history()[-1].content == "something important"
    # But the health signal has flipped, and the failure was logged into the list.
    assert ctx.persistence_healthy is False
    assert len(ctx.failed_writes) == 1
    entry = ctx.failed_writes[0]
    assert entry["text"] == "something important"
    assert entry["kind"] == "conversation"
    assert "disk full" in entry["error"] or "still broken" in entry["error"]


def test_persistence_falls_back_to_add_when_add_text_missing(cfg):
    class RecordOnlyStore:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)
            return record.id

        def search(self, *a, **kw):
            return []

        def recent(self, **kw):
            return []

    ctx = ContextManager(RecordOnlyStore(), cfg.memory)
    ctx.add_user("hello")
    assert ctx.persistence_healthy is True
    assert len(ctx.store.records) == 1
    assert ctx.store.records[0].kind == "conversation"
    assert ctx.store.records[0].text == "hello"


# --------------------------------------------------------------------------- #
# clear_live / facts / forget
# --------------------------------------------------------------------------- #
def test_clear_live_does_not_touch_the_store(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    ctx.add_user("hi")
    ctx.add_assistant("hello")
    ctx.summary = "some summary"
    assert store.count(kind="conversation") == 2
    ctx.clear_live()
    assert ctx.history() == []
    assert ctx.summary == ""
    # Store preserved everything.
    assert store.count(kind="conversation") == 2


def test_remember_facts_list_and_forget(store, cfg):
    ctx = ContextManager(store, cfg.memory)
    rid = ctx.remember_fact("The moon is small.", topic="astronomy")
    assert rid
    assert ctx.persistence_healthy is True
    facts = ctx.facts()
    assert any(f.id == rid for f in facts)

    ok = ctx.forget(rid)
    assert ok is True
    assert not any(f.id == rid for f in ctx.facts())
    # Forgetting the same id again returns False, not an exception.
    assert ctx.forget(rid) is False


def test_forget_returns_false_when_store_has_no_delete(cfg):
    class NoDeleteStore:
        def add_text(self, kind, text, **metadata):
            from jarvis.core.contracts import MemoryRecord
            return MemoryRecord(id="x", kind=kind, text=text)

        def recent(self, **kw):
            return []

        def search(self, *a, **kw):
            return []

    ctx = ContextManager(NoDeleteStore(), cfg.memory)
    assert ctx.forget("x") is False


def test_create_context_builds_working_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "cchome"))
    cfg = Config()
    cfg.memory.embedder = "hash"
    cfg.memory.embed_dim = 32
    ctx = create_context(cfg, system_prompt="Hi there.")
    try:
        assert isinstance(ctx.store, SQLiteMemoryStore)
        ctx.add_user("first")
        assert ctx.store.count(kind="conversation") == 1
        msgs = ctx.build("second question")
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "Hi there."
        assert msgs[-1].content == "second question"
    finally:
        ctx.store.close()
