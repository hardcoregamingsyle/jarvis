"""Long-term memory subsystem: durable, searchable, and never destructive.

This package is the "practically forgets nothing" store.  Everything that comes
through the conversation loop is persisted to a local SQLite database with an
optional semantic index.  The design is deliberately conservative:

- writes are idempotent and never overwritten
- reads (search / recent / all) never mutate the store
- the store degrades gracefully when the semantic embedder is unavailable
- the SQLite build's FTS5 support is detected at runtime, and a LIKE fallback
  keeps keyword search working when FTS5 is not compiled in

Public factories:
    create_memory(cfg)   -> SQLiteMemoryStore
    create_context(cfg)  -> ContextManager
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.contracts import Embedder, MemoryRecord, MemoryStore
from ..core.platform_utils import data_dir, ensure_dir
from .context import ContextManager
from .embeddings import (
    HashEmbedder,
    NullEmbedder,
    SentenceTransformerEmbedder,
    cosine,
    create_embedder,
)
from .store import SQLiteMemoryStore


def create_memory(cfg) -> SQLiteMemoryStore:
    """Build a memory store from a :class:`Config` (or bare :class:`MemoryConfig`).

    An unset ``db_path`` resolves to an absolute path under
    :func:`jarvis.core.platform_utils.data_dir` — never a bare relative
    ``memory.db`` which would give a different database per launch directory.
    """
    mem_cfg = getattr(cfg, "memory", cfg)
    raw = getattr(mem_cfg, "db_path", "") or ""
    if not str(raw).strip():
        db_path = data_dir() / "memory.db"
    else:
        db_path = Path(raw).expanduser()
    ensure_dir(db_path.parent)
    embedder = create_embedder(mem_cfg)
    return SQLiteMemoryStore(db_path, embedder=embedder, config=mem_cfg)


def create_context(
    cfg,
    *,
    llm=None,
    system_prompt: str = "",
    store: Optional[MemoryStore] = None,
) -> ContextManager:
    """Build a :class:`ContextManager`, creating a memory store if one is not given."""
    mem_cfg = getattr(cfg, "memory", cfg)
    if store is None:
        store = create_memory(cfg)
    return ContextManager(store, mem_cfg, llm=llm, system_prompt=system_prompt)


__all__ = [
    "Embedder",
    "MemoryRecord",
    "MemoryStore",
    "HashEmbedder",
    "NullEmbedder",
    "SentenceTransformerEmbedder",
    "cosine",
    "create_embedder",
    "SQLiteMemoryStore",
    "ContextManager",
    "create_memory",
    "create_context",
]
