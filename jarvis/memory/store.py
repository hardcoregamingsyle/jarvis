"""SQLite-backed memory store — the durable "never forgets" tier.

Design notes:

* WAL journalling + ``check_same_thread=False`` + an internal ``RLock`` around
  every statement.  The store is safe to share between the STT thread, the
  agent thread, and any background workers.
* Records are stored with their embedding in a separate table (packed as a
  float32 blob via :mod:`struct`, not JSON — 4 bytes/dim instead of ~15).
* FTS5 is optional: some Windows Python builds ship without it.  We detect the
  capability at import time by attempting ``CREATE VIRTUAL TABLE`` against a
  ``:memory:`` connection, and fall back to a case-insensitive ``LIKE`` search
  otherwise.  The LIKE fallback also runs whenever the FTS5 query synthesised
  from the input is empty — so a punctuation-heavy query still exercises
  keyword search rather than silently returning nothing (or, worse, leaking
  a plain recency dump).
* Hybrid ``search()`` combines cosine similarity, keyword score, and a gentle
  recency boost, so it works usefully with any of {sentence-transformers,
  HashEmbedder, no embedder}.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..core.contracts import MemoryRecord, MemoryStore, new_id, now_ts
from .embeddings import cosine

log = logging.getLogger(__name__)


SCHEMA_VERSION = 1


_FTS5_AVAILABLE_CACHE: Optional[bool] = None


def _detect_fts5() -> bool:
    """Return True if the running SQLite build was compiled with FTS5."""
    global _FTS5_AVAILABLE_CACHE
    if _FTS5_AVAILABLE_CACHE is not None:
        return _FTS5_AVAILABLE_CACHE
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE __probe USING fts5(x)")
            _FTS5_AVAILABLE_CACHE = True
        finally:
            conn.close()
    except sqlite3.Error:
        _FTS5_AVAILABLE_CACHE = False
    return bool(_FTS5_AVAILABLE_CACHE)


def _pack_vector(vec: Sequence[float]) -> bytes:
    if not vec:
        return b""
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def _unpack_vector(blob: bytes) -> List[float]:
    if not blob:
        return []
    n = len(blob) // 4
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}f", blob))


_FTS_SPECIAL = re.compile(r'[^\w\s]+', re.UNICODE)


def _sanitise_fts_query(query: str) -> str:
    """Turn a natural-language query into an FTS5 MATCH expression.

    FTS5 has its own operators (``AND``/``OR``/``NEAR``/``^``/``:``/quoting)
    and punctuation trips the parser.  We strip everything but word characters
    and whitespace, then OR-join the surviving tokens as quoted phrases.
    Returns ``""`` for a punctuation-only or empty query so the caller knows
    to fall back to LIKE.
    """
    if not query:
        return ""
    cleaned = _FTS_SPECIAL.sub(" ", query)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


class SQLiteMemoryStore(MemoryStore):
    """Thread-safe SQLite memory store."""

    def __init__(
        self,
        db_path,
        embedder=None,
        *,
        config=None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.config = config
        self._lock = threading.RLock()
        self._fts5 = _detect_fts5()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=5000")
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);
                CREATE INDEX IF NOT EXISTS idx_records_ts   ON records(ts);
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL
                );
                """
            )
            row = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO schema_version(version) VALUES(?)",
                    (SCHEMA_VERSION,),
                )

            if self._fts5:
                try:
                    c.executescript(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
                            id UNINDEXED,
                            text,
                            kind UNINDEXED,
                            content='records',
                            content_rowid='rowid'
                        );
                        CREATE TRIGGER IF NOT EXISTS records_ai
                          AFTER INSERT ON records BEGIN
                            INSERT INTO records_fts(rowid, id, text, kind)
                            VALUES (new.rowid, new.id, new.text, new.kind);
                        END;
                        CREATE TRIGGER IF NOT EXISTS records_ad
                          AFTER DELETE ON records BEGIN
                            INSERT INTO records_fts(records_fts, rowid, id, text, kind)
                            VALUES ('delete', old.rowid, old.id, old.text, old.kind);
                        END;
                        CREATE TRIGGER IF NOT EXISTS records_au
                          AFTER UPDATE ON records BEGIN
                            INSERT INTO records_fts(records_fts, rowid, id, text, kind)
                            VALUES ('delete', old.rowid, old.id, old.text, old.kind);
                            INSERT INTO records_fts(rowid, id, text, kind)
                            VALUES (new.rowid, new.id, new.text, new.kind);
                        END;
                        """
                    )
                except sqlite3.Error:
                    log.warning(
                        "records_fts virtual table could not be created; "
                        "falling back to LIKE search"
                    )
                    self._fts5 = False

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add(self, record: MemoryRecord) -> str:
        """Insert a record.  Idempotent on ``record.id``: an existing id is
        left untouched (first write wins) — the store never overwrites."""
        if not record.id:
            record.id = new_id("mem")
        metadata_json = json.dumps(record.metadata or {}, ensure_ascii=False, default=str)
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM records WHERE id = ?", (record.id,)
            ).fetchone()
            if existing:
                return record.id
            self._conn.execute(
                "INSERT INTO records(id, kind, text, metadata, ts) VALUES(?, ?, ?, ?, ?)",
                (record.id, record.kind, record.text, metadata_json, float(record.ts)),
            )
            self._embed_and_store(record.id, record.text)
        return record.id

    def _embed_and_store(self, record_id: str, text: str) -> None:
        if self.embedder is None or not getattr(self.embedder, "dim", 0):
            return
        try:
            vec = self.embedder.embed_one(text)
        except Exception:  # noqa: BLE001
            log.exception("embedder failed for record %s", record_id)
            return
        if not vec:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings(id, dim, vec) VALUES(?, ?, ?)",
            (record_id, len(vec), _pack_vector(vec)),
        )

    def add_text(self, kind: str, text: str, **metadata) -> MemoryRecord:
        """Convenience: build a :class:`MemoryRecord` and add it."""
        record = MemoryRecord(
            id="", kind=kind, text=text, metadata=dict(metadata or {}), ts=now_ts()
        )
        self.add(record)
        return record

    def delete(self, record_id: str) -> bool:
        """Remove a record.  Returns True if something was deleted."""
        with self._lock:
            self._conn.execute("DELETE FROM embeddings WHERE id = ?", (record_id,))
            cur = self._conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            return bool(cur.rowcount and cur.rowcount > 0)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def _row_to_record(self, row) -> MemoryRecord:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        return MemoryRecord(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            metadata=metadata,
            ts=float(row["ts"]),
        )

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, text, metadata, ts FROM records WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def recent(self, *, k: int = 20, kind: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            if kind is None:
                rows = self._conn.execute(
                    "SELECT id, kind, text, metadata, ts FROM records "
                    "ORDER BY ts DESC, rowid DESC LIMIT ?",
                    (int(k),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, kind, text, metadata, ts FROM records "
                    "WHERE kind = ? ORDER BY ts DESC, rowid DESC LIMIT ?",
                    (kind, int(k)),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def all(self, *, kind: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            if kind is None:
                rows = self._conn.execute(
                    "SELECT id, kind, text, metadata, ts FROM records "
                    "ORDER BY ts ASC, rowid ASC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, kind, text, metadata, ts FROM records "
                    "WHERE kind = ? ORDER BY ts ASC, rowid ASC",
                    (kind,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, kind: Optional[str] = None) -> int:
        with self._lock:
            if kind is None:
                row = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM records WHERE kind = ?", (kind,)
                ).fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            by_kind_rows = self._conn.execute(
                "SELECT kind, COUNT(*) FROM records GROUP BY kind"
            ).fetchall()
            embs = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {
            "total": int(total),
            "by_kind": {r[0]: int(r[1]) for r in by_kind_rows},
            "embeddings": int(embs),
            "fts5": self._fts5,
            "db_path": str(self.db_path),
            "embedder": getattr(self.embedder, "name", None) if self.embedder else None,
            "embed_dim": int(getattr(self.embedder, "dim", 0) or 0),
        }

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self, query: str, *, k: int = 8, kind: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Hybrid recall: cosine + keyword + gentle recency boost.

        The result set is drawn only from records that matched by keyword or
        embedding — recency alone is never enough to appear.  Otherwise a
        blank-ish query would surface every recent turn as though it were
        relevant.
        """
        if not query or not str(query).strip():
            return []
        query = str(query)
        keyword_scores: dict = {}
        cosine_scores: dict = {}
        now = time.time()

        fts_query = _sanitise_fts_query(query)
        fts_hit = False
        if self._fts5 and fts_query:
            try:
                with self._lock:
                    if kind is None:
                        rows = self._conn.execute(
                            "SELECT records.id AS id, bm25(records_fts) AS score "
                            "FROM records_fts JOIN records "
                            "ON records.rowid = records_fts.rowid "
                            "WHERE records_fts MATCH ? "
                            "ORDER BY score LIMIT ?",
                            (fts_query, max(k * 4, 32)),
                        ).fetchall()
                    else:
                        rows = self._conn.execute(
                            "SELECT records.id AS id, bm25(records_fts) AS score "
                            "FROM records_fts JOIN records "
                            "ON records.rowid = records_fts.rowid "
                            "WHERE records_fts MATCH ? AND records.kind = ? "
                            "ORDER BY score LIMIT ?",
                            (fts_query, kind, max(k * 4, 32)),
                        ).fetchall()
                if rows:
                    fts_hit = True
                    for r in rows:
                        rid = r["id"]
                        raw = float(r["score"])
                        keyword_scores[rid] = max(keyword_scores.get(rid, 0.0), -raw)
            except sqlite3.Error:
                log.warning("FTS5 query failed; falling back to LIKE", exc_info=True)
                fts_hit = False

        if not fts_hit:
            tokens = [t for t in re.findall(r"\w+", query, re.UNICODE)]
            tokens = [t.lower() for t in tokens if t]
            for token in tokens[:8]:
                like_pat = f"%{token}%"
                with self._lock:
                    if kind is None:
                        rows = self._conn.execute(
                            "SELECT id FROM records WHERE lower(text) LIKE ? LIMIT ?",
                            (like_pat, max(k * 4, 32)),
                        ).fetchall()
                    else:
                        rows = self._conn.execute(
                            "SELECT id FROM records "
                            "WHERE lower(text) LIKE ? AND kind = ? LIMIT ?",
                            (like_pat, kind, max(k * 4, 32)),
                        ).fetchall()
                for r in rows:
                    rid = r[0]
                    keyword_scores[rid] = keyword_scores.get(rid, 0.0) + 1.0

        if keyword_scores:
            mx = max(keyword_scores.values())
            if mx > 0.0:
                inv = 1.0 / mx
                for rid in list(keyword_scores):
                    keyword_scores[rid] *= inv

        if self.embedder is not None and getattr(self.embedder, "dim", 0):
            try:
                qvec = self.embedder.embed_one(query)
            except Exception:  # noqa: BLE001
                log.exception("embedder failed on query")
                qvec = []
            if qvec:
                with self._lock:
                    if kind is None:
                        emb_rows = self._conn.execute(
                            "SELECT e.id AS id, e.vec AS vec FROM embeddings e "
                            "JOIN records r ON e.id = r.id"
                        ).fetchall()
                    else:
                        emb_rows = self._conn.execute(
                            "SELECT e.id AS id, e.vec AS vec FROM embeddings e "
                            "JOIN records r ON e.id = r.id WHERE r.kind = ?",
                            (kind,),
                        ).fetchall()
                for row in emb_rows:
                    v = _unpack_vector(row["vec"])
                    score = cosine(qvec, v)
                    if score > 0.0:
                        cosine_scores[row["id"]] = score

        pool_ids = set(keyword_scores) | set(cosine_scores)
        if not pool_ids:
            return []

        with self._lock:
            placeholders = ",".join("?" for _ in pool_ids)
            rows = self._conn.execute(
                f"SELECT id, kind, text, metadata, ts FROM records "
                f"WHERE id IN ({placeholders})",
                tuple(pool_ids),
            ).fetchall()

        half_life = 30.0 * 24.0 * 3600.0
        results: List[MemoryRecord] = []
        for row in rows:
            rid = row["id"]
            ks = keyword_scores.get(rid, 0.0)
            cs = cosine_scores.get(rid, 0.0)
            age = max(0.0, now - float(row["ts"]))
            recency = 0.5 ** (age / half_life)
            score = 0.55 * cs + 0.35 * ks + 0.10 * recency
            rec = self._row_to_record(row)
            rec.score = float(score)
            results.append(rec)

        seen: set = set()
        unique: List[MemoryRecord] = []
        for rec in sorted(results, key=lambda r: r.score, reverse=True):
            if rec.id in seen:
                continue
            seen.add(rec.id)
            unique.append(rec)
        return unique[:k]

    # ------------------------------------------------------------------ #
    # Import / export
    # ------------------------------------------------------------------ #
    def export_jsonl(self, path) -> int:
        """Dump every record as one JSON object per line.  Returns the count."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, text, metadata, ts FROM records "
                "ORDER BY ts ASC, rowid ASC"
            ).fetchall()
        written = 0
        with open(p, "w", encoding="utf-8") as f:
            for row in rows:
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                payload = {
                    "id": row["id"],
                    "kind": row["kind"],
                    "text": row["text"],
                    "metadata": metadata,
                    "ts": float(row["ts"]),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
        return written

    def import_jsonl(self, path) -> int:
        """Load records from a previous ``export_jsonl``.  Idempotent by id."""
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed JSONL line during import")
                    continue
                metadata = payload.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {"value": metadata}
                rec = MemoryRecord(
                    id=str(payload.get("id") or new_id("mem")),
                    kind=str(payload.get("kind", "note")),
                    text=str(payload.get("text", "")),
                    metadata=metadata,
                    ts=float(payload.get("ts", now_ts())),
                )
                self.add(rec)
                loaded += 1
        return loaded

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                log.debug("close raised", exc_info=True)


__all__ = ["SQLiteMemoryStore"]
