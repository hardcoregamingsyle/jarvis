"""Tests for the memory subsystem: embeddings + SQLite store + factories."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jarvis.core.config import Config, MemoryConfig
from jarvis.core.contracts import MemoryRecord
from jarvis.memory import (
    HashEmbedder,
    NullEmbedder,
    SQLiteMemoryStore,
    cosine,
    create_embedder,
    create_memory,
)
from jarvis.memory import store as store_mod


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A Config wired to an isolated home and an isolated DB file."""
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "memtest_home"))
    (tmp_path / "memtest_home").mkdir(exist_ok=True)
    c = Config()
    c.memory.db_path = str(tmp_path / "memory.db")
    c.memory.embedder = "hash"
    c.memory.embed_dim = 128
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
# Embeddings
# --------------------------------------------------------------------------- #
def test_cosine_edges():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_hash_embedder_deterministic_and_normalised():
    a = HashEmbedder(dim=64)
    b = HashEmbedder(dim=64)
    va = a.embed_one("hello world")
    vb = b.embed_one("hello world")
    assert va == vb
    assert len(va) == 64
    assert any(abs(x) > 0 for x in va)
    magnitude = sum(x * x for x in va) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-4)
    vc = a.embed_one("completely different unrelated content indeed")
    assert va != vc


def test_hash_embedder_short_text():
    e = HashEmbedder(dim=32)
    vec = e.embed_one("hi")
    assert len(vec) == 32
    assert any(abs(x) > 0 for x in vec)


def test_hash_embedder_empty_text_zero_vector():
    e = HashEmbedder(dim=16)
    vec = e.embed_one("")
    assert vec == [0.0] * 16


def test_null_embedder():
    n = NullEmbedder()
    assert n.dim == 0
    assert n.embed(["x", "y", "z"]) == [[], [], []]
    assert n.is_available() is True


def test_create_embedder_selection():
    hashed = create_embedder(MemoryConfig(embedder="hash", embed_dim=32))
    assert isinstance(hashed, HashEmbedder) and hashed.dim == 32

    null = create_embedder(MemoryConfig(embedder="none"))
    assert isinstance(null, NullEmbedder)

    # "auto" always yields *something* callable; if the ST package is not
    # installed it must degrade to HashEmbedder rather than raise.
    auto = create_embedder(MemoryConfig(embedder="auto", embed_dim=32))
    assert auto.is_available()
    assert auto.dim > 0


# --------------------------------------------------------------------------- #
# Store: schema and CRUD
# --------------------------------------------------------------------------- #
def test_schema_reopen_durability(cfg):
    s = SQLiteMemoryStore(cfg.memory.db_path, embedder=HashEmbedder(dim=64))
    s.add_text("fact", "The capital of France is Paris", topic="geography")
    s.add_text("fact", "The user's dog is named Ada")
    s.close()

    reopened = SQLiteMemoryStore(cfg.memory.db_path, embedder=HashEmbedder(dim=64))
    try:
        facts = reopened.recent(k=10, kind="fact")
        assert len(facts) == 2
        texts = {f.text for f in facts}
        assert "The capital of France is Paris" in texts
        assert "The user's dog is named Ada" in texts
        for f in facts:
            if "capital" in f.text:
                assert f.metadata == {"topic": "geography"}
    finally:
        reopened.close()


def test_add_get_recent_all(store):
    r1 = store.add_text("note", "first")
    r2 = store.add_text("note", "second")
    r3 = store.add_text("fact", "third")

    fetched = store.get(r1.id)
    assert fetched is not None and fetched.text == "first"
    assert store.get("does-not-exist") is None

    notes = store.all(kind="note")
    assert [n.text for n in notes] == ["first", "second"]

    recent = store.recent(k=2)
    assert [r.text for r in recent] == ["third", "second"]

    everything = store.all()
    assert {r.id for r in everything} == {r1.id, r2.id, r3.id}


def test_add_idempotent_on_id(store):
    r = MemoryRecord(id="fixed_id", kind="fact", text="alpha")
    returned = store.add(r)
    assert returned == "fixed_id"
    # Re-add with the same id but different text: first write wins.
    store.add(MemoryRecord(id="fixed_id", kind="fact", text="beta"))
    got = store.get("fixed_id")
    assert got is not None and got.text == "alpha"
    assert store.count() == 1


def test_delete_removes_row_and_embedding(store):
    r = store.add_text("fact", "temporary")
    assert store.count() == 1
    assert store.delete(r.id) is True
    assert store.count() == 0
    assert store.get(r.id) is None
    # Vector row also gone.
    with store._lock:
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE id = ?", (r.id,)
        ).fetchone()[0]
    assert remaining == 0
    assert store.delete("no-such-id") is False


# --------------------------------------------------------------------------- #
# Store: search
# --------------------------------------------------------------------------- #
def test_search_relevance_with_hash_embedder(store):
    store.add_text("fact", "Ada is the user's dog. Ada was born in 2019.")
    store.add_text("fact", "The Eiffel Tower is in Paris, France.")
    store.add_text("note", "Grocery list: apples, bread, cheese.")

    results = store.search("Tell me about the dog Ada", k=3)
    assert results, "expected at least one result"
    assert results[0].text.startswith("Ada is the user's dog")
    for r in results:
        assert r.score >= 0.0
    # sorted descending by score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_with_no_embedder(cfg):
    s = SQLiteMemoryStore(cfg.memory.db_path, embedder=None)
    try:
        s.add_text("fact", "Zebras have stripes")
        s.add_text("fact", "Lions are large cats")
        found = s.search("zebra", k=5)
        assert any("Zebras" in x.text for x in found)
        assert not any("Lions" in x.text for x in found)
    finally:
        s.close()


def test_search_empty_store_returns_empty(store):
    assert store.search("anything") == []


def test_empty_query_returns_empty(store):
    store.add_text("fact", "Zebras are striped")
    store.add_text("fact", "Lions are cats")
    assert store.search("") == []
    assert store.search("   ") == []


def test_empty_fts_query_still_runs_keyword_search(cfg):
    """Punctuation-only sanitises to an empty FTS expression.  With no
    embedder to fall back on, that must yield an empty result set — the
    recency pool must NOT leak in.  A real keyword still hits regardless
    of whether FTS or LIKE handled it."""
    s = SQLiteMemoryStore(cfg.memory.db_path, embedder=None)
    try:
        s.add_text("fact", "Zebras are striped")
        s.add_text("fact", "Lions are cats")
        assert s.search("?!!") == []
        hits = s.search("zebra", k=3)
        assert any("Zebra" in x.text for x in hits)
    finally:
        s.close()


def test_fts_absent_falls_back_to_like(cfg, monkeypatch):
    """Simulate a SQLite build without FTS5: search must still work via LIKE.

    We use ``embedder=None`` here so the observed behaviour reflects LIKE
    alone — a HashEmbedder would smear soft cosine similarity across every
    record and hide the keyword path.
    """
    monkeypatch.setattr(store_mod, "_FTS5_AVAILABLE_CACHE", False)
    monkeypatch.setattr(store_mod, "_detect_fts5", lambda: False)
    s = SQLiteMemoryStore(cfg.memory.db_path, embedder=None)
    try:
        assert s._fts5 is False
        s.add_text("fact", "Alpha bravo charlie")
        s.add_text("fact", "Delta echo foxtrot")
        s.add_text("fact", "Golf hotel india bravo")
        results = s.search("bravo", k=5)
        texts = [r.text for r in results]
        assert "Alpha bravo charlie" in texts
        assert "Golf hotel india bravo" in texts
        assert "Delta echo foxtrot" not in texts
        # A keyword that appears in no record: LIKE returns nothing.
        assert s.search("qzxwjk", k=5) == []
    finally:
        s.close()


def test_unicode_and_emoji_roundtrip(store):
    text = "Café ☕ 你好 🌸 — naïve résumé"
    rec = store.add_text("note", text, mood="🎉", lang="mixed")
    got = store.get(rec.id)
    assert got is not None
    assert got.text == text
    assert got.metadata == {"mood": "🎉", "lang": "mixed"}
    # Keyword search finds the record on a Unicode token.
    found = store.search("Café", k=5)
    assert any(x.text == text for x in found)


def test_nested_metadata_roundtrip(store):
    md = {
        "tags": ["a", "b", "c"],
        "nested": {"level": 2, "items": [1, 2, {"x": True, "y": None}]},
        "flag": False,
    }
    rec = store.add_text("note", "with nested meta", **md)
    got = store.get(rec.id)
    assert got is not None
    assert got.metadata["tags"] == ["a", "b", "c"]
    assert got.metadata["nested"]["level"] == 2
    assert got.metadata["nested"]["items"][2] == {"x": True, "y": None}
    assert got.metadata["flag"] is False


# --------------------------------------------------------------------------- #
# Store: aggregation
# --------------------------------------------------------------------------- #
def test_count_and_stats(store):
    for i in range(5):
        store.add_text("fact", f"fact number {i}")
    for i in range(3):
        store.add_text("note", f"note number {i}")
    assert store.count() == 8
    assert store.count(kind="fact") == 5
    assert store.count(kind="note") == 3

    stats = store.stats()
    assert stats["total"] == 8
    assert stats["by_kind"] == {"fact": 5, "note": 3}
    assert stats["embeddings"] == 8
    assert stats["embedder"] == "hash"
    assert stats["embed_dim"] == 128
    assert "db_path" in stats


def test_export_import_roundtrip(cfg, tmp_path):
    src = SQLiteMemoryStore(cfg.memory.db_path, embedder=HashEmbedder(dim=32))
    for i in range(4):
        src.add_text("fact", f"the number is {i}", n=i, group="a")
    src.add_text("note", "a stray note", tag="misc")

    dump = tmp_path / "export.jsonl"
    n = src.export_jsonl(dump)
    assert n == 5
    lines = dump.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    payload = json.loads(lines[0])
    assert set(payload) >= {"id", "kind", "text", "metadata", "ts"}

    dst_path = tmp_path / "other.db"
    dst = SQLiteMemoryStore(dst_path, embedder=HashEmbedder(dim=32))
    try:
        loaded = dst.import_jsonl(dump)
        assert loaded == 5
        assert dst.count() == 5
        assert dst.count(kind="fact") == 4
        assert {r.id for r in dst.all()} == {r.id for r in src.all()}
        # Re-importing the same file is idempotent (add() ignores existing ids).
        assert dst.import_jsonl(dump) == 5
        assert dst.count() == 5
    finally:
        src.close()
        dst.close()


# --------------------------------------------------------------------------- #
# Store: concurrency
# --------------------------------------------------------------------------- #
def test_thread_safety(cfg):
    """Eight worker threads writing and reading concurrently: nothing is lost."""
    s = SQLiteMemoryStore(cfg.memory.db_path, embedder=HashEmbedder(dim=64))
    errors: list = []
    items_per_thread = 40
    thread_count = 8

    def worker(tid: int) -> None:
        try:
            for i in range(items_per_thread):
                s.add_text("note", f"thread {tid} item {i}", tid=tid, i=i)
                if i % 5 == 0:
                    _ = s.search(f"thread {tid}", k=3)
                    _ = s.recent(k=4)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(t,))
        for t in range(thread_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert errors == []
        assert s.count() == thread_count * items_per_thread
        stored_pairs = {
            (r.metadata.get("tid"), r.metadata.get("i")) for r in s.all()
        }
        expected = {
            (t, i)
            for t in range(thread_count)
            for i in range(items_per_thread)
        }
        assert stored_pairs == expected
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_create_memory_unset_db_path_is_absolute_under_data_dir(
    tmp_path, monkeypatch
):
    home = tmp_path / "jarvis_home_absolute"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    cfg = Config()
    cfg.memory.db_path = ""
    cfg.memory.embedder = "hash"
    cfg.memory.embed_dim = 32
    s = create_memory(cfg)
    try:
        db = Path(s.db_path)
        assert db.is_absolute()
        assert db.resolve() == (home / "memory.db").resolve()
    finally:
        s.close()


def test_create_memory_absolute_db_path_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "elsewhere"))
    explicit = tmp_path / "custom" / "brain.db"
    cfg = Config()
    cfg.memory.db_path = str(explicit)
    cfg.memory.embedder = "hash"
    cfg.memory.embed_dim = 32
    s = create_memory(cfg)
    try:
        assert Path(s.db_path).resolve() == explicit.resolve()
        s.add_text("fact", "custom-db lives")
        assert s.count() == 1
    finally:
        s.close()


def test_create_memory_returns_functional_store(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "jarvis_home_ok"))
    cfg = Config()
    cfg.memory.embedder = "hash"
    cfg.memory.embed_dim = 32
    s = create_memory(cfg)
    try:
        s.add_text("fact", "hello")
        assert s.count() == 1
        results = s.search("hello", k=1)
        assert results and results[0].text == "hello"
    finally:
        s.close()
