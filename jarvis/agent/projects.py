"""Projects and durable task state: work that survives a reboot.

Two problems solved together, because they are the same problem.

**Projects.** Work belongs somewhere. A subagent auditing a codebase and one
booking travel have nothing to say to each other, and pooling them means every
status question returns a mixed bag. A project owns its tasks, its notes and
its own slice of memory, and exactly one is *active* at a time — so "pause" and
"what are you working on" have an unambiguous subject.

**Durability.** The task tree in :mod:`jarvis.agent.task_manager` lives in RAM,
which is correct for scheduling and useless across a reboot. This module keeps
a SQLite mirror: every state change is written through, so a machine that comes
back up knows what was running, what was paused, and where each thing had got
to.

What "resume" honestly means
----------------------------
A paused task's *Python stack* is gone — it was a thread, and threads do not
survive `init`. What survives is the goal, the accumulated progress notes, and
the transcript. Resuming re-dispatches a subagent primed with all of it:

    Goal: audit the auth module for injection risks
    Progress so far:
      - read src/auth/*.py, found 3 raw SQL string builds
      - checking src/db/query.py next

so it continues rather than restarting. That is a re-entry, not a snapshot
restore, and the distinction is stated plainly because a tool that claims to
freeze and thaw a running process would be lying.

Every write is committed immediately. A power cut costs at most the update in
flight, never the project.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The project used when nobody has chosen one. Created on first use.
DEFAULT_PROJECT = "general"

# Durable task states. These extend the in-memory TaskState with the two that
# only mean anything across a restart.
PENDING = "pending"
RUNNING = "running"
PAUSED = "paused"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

#: States a task can be resumed from.
RESUMABLE = frozenset({PAUSED, PENDING, RUNNING})
#: States that never change again.
TERMINAL = frozenset({DONE, FAILED, CANCELLED})


def _now() -> float:
    return time.time()


@dataclass
class Project:
    """A named container for related work."""

    name: str
    created: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)
    description: str = ""
    active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created": self.created,
            "updated": self.updated,
            "description": self.description,
            "active": self.active,
            "metadata": dict(self.metadata),
        }


@dataclass
class DurableTask:
    """A task as it exists on disk, independent of any running thread."""

    id: str
    project: str
    goal: str
    state: str = PENDING
    created: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)
    parent_id: str = ""
    depth: int = 0
    progress: List[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def resumable(self) -> bool:
        return self.state in RESUMABLE

    def resume_briefing(self) -> str:
        """The context a re-dispatched subagent needs to continue, not restart."""
        lines = [f"Goal: {self.goal}"]
        if self.progress:
            lines.append("")
            lines.append("Progress already made (do not repeat this work):")
            lines.extend(f"  - {note}" for note in self.progress[-20:])
            lines.append("")
            lines.append("Continue from where this left off.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "goal": self.goal,
            "state": self.state,
            "created": self.created,
            "updated": self.updated,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "progress": list(self.progress),
            "result": self.result,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class ProjectStore:
    """SQLite-backed projects and durable tasks. Thread-safe, never raises.

    Mirrors the in-memory task tree rather than replacing it: the scheduler
    stays in RAM where it belongs, and this is the record that outlives it.
    """

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            # WAL so a reader never blocks the writer, and a crash mid-write
            # rolls back cleanly instead of corrupting the file.
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=5000")
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    name        TEXT PRIMARY KEY,
                    created     REAL NOT NULL,
                    updated     REAL NOT NULL,
                    description TEXT DEFAULT '',
                    active      INTEGER DEFAULT 0,
                    metadata    TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id        TEXT PRIMARY KEY,
                    project   TEXT NOT NULL,
                    goal      TEXT NOT NULL,
                    state     TEXT NOT NULL,
                    created   REAL NOT NULL,
                    updated   REAL NOT NULL,
                    parent_id TEXT DEFAULT '',
                    depth     INTEGER DEFAULT 0,
                    progress  TEXT DEFAULT '[]',
                    result    TEXT DEFAULT '',
                    error     TEXT DEFAULT '',
                    metadata  TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project);
                CREATE INDEX IF NOT EXISTS idx_tasks_state   ON tasks(state);
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            c.commit()
            self._ensure_project_locked(DEFAULT_PROJECT, make_active=True)

    # ------------------------------------------------------------------ #
    #  Projects
    # ------------------------------------------------------------------ #
    def _ensure_project_locked(self, name: str, *, make_active: bool = False) -> None:
        row = self._conn.execute(
            "SELECT name FROM projects WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            now = _now()
            self._conn.execute(
                "INSERT INTO projects(name, created, updated, active) VALUES(?,?,?,?)",
                (name, now, now, 0),
            )
            self._conn.commit()
        if make_active:
            has_active = self._conn.execute(
                "SELECT name FROM projects WHERE active = 1"
            ).fetchone()
            if has_active is None:
                self._conn.execute(
                    "UPDATE projects SET active = 1 WHERE name = ?", (name,)
                )
                self._conn.commit()

    def create_project(self, name: str, *, description: str = "") -> Project:
        """Create (or fetch) a project. Idempotent."""
        clean = str(name or "").strip() or DEFAULT_PROJECT
        with self._lock:
            self._ensure_project_locked(clean)
            if description:
                self._conn.execute(
                    "UPDATE projects SET description = ?, updated = ? WHERE name = ?",
                    (description, _now(), clean),
                )
                self._conn.commit()
        return self.get_project(clean) or Project(name=clean)

    def get_project(self, name: str) -> Optional[Project]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE name = ?", (str(name),)
            ).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self) -> List[Project]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM projects ORDER BY active DESC, updated DESC"
            ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def active_project(self) -> str:
        """The project new work is filed under."""
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM projects WHERE active = 1 LIMIT 1"
            ).fetchone()
        return str(row["name"]) if row else DEFAULT_PROJECT

    def set_active_project(self, name: str) -> str:
        """Switch projects, creating the target if needed."""
        clean = str(name or "").strip() or DEFAULT_PROJECT
        with self._lock:
            self._ensure_project_locked(clean)
            self._conn.execute("UPDATE projects SET active = 0")
            self._conn.execute(
                "UPDATE projects SET active = 1, updated = ? WHERE name = ?",
                (_now(), clean),
            )
            self._conn.commit()
        return clean

    @staticmethod
    def _row_to_project(row: Any) -> Project:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (ValueError, TypeError):
            metadata = {}
        return Project(
            name=str(row["name"]),
            created=float(row["created"]),
            updated=float(row["updated"]),
            description=str(row["description"] or ""),
            active=bool(row["active"]),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    #  Tasks
    # ------------------------------------------------------------------ #
    def record_task(
        self,
        task_id: str,
        goal: str,
        *,
        project: str = "",
        state: str = PENDING,
        parent_id: str = "",
        depth: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DurableTask:
        """Write a task to disk. Called when it is dispatched."""
        proj = str(project or "").strip() or self.active_project()
        now = _now()
        with self._lock:
            self._ensure_project_locked(proj)
            self._conn.execute(
                """INSERT OR REPLACE INTO tasks
                   (id, project, goal, state, created, updated, parent_id,
                    depth, progress, result, error, metadata)
                   VALUES (?,?,?,?,
                           COALESCE((SELECT created FROM tasks WHERE id = ?), ?),
                           ?,?,?,
                           COALESCE((SELECT progress FROM tasks WHERE id = ?), '[]'),
                           '','',?)""",
                (
                    task_id, proj, goal, state, task_id, now, now,
                    parent_id, int(depth), task_id,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return self.get_task(task_id) or DurableTask(
            id=task_id, project=proj, goal=goal, state=state
        )

    def update_state(
        self,
        task_id: str,
        state: str,
        *,
        result: str = "",
        error: str = "",
    ) -> bool:
        """Move a task to a new state. Returns False if it is unknown."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                """UPDATE tasks SET state = ?, updated = ?,
                          result = CASE WHEN ? != '' THEN ? ELSE result END,
                          error  = CASE WHEN ? != '' THEN ? ELSE error  END
                   WHERE id = ?""",
                (state, _now(), result, result, error, error, task_id),
            )
            self._conn.commit()
        return True

    def add_progress(self, task_id: str, note: str) -> bool:
        """Append a progress note — the thing that makes resuming meaningful.

        Without these a resumed task restarts from nothing; with them it knows
        what it already did.
        """
        clean = str(note or "").strip()
        if not clean:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT progress FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return False
            try:
                notes = json.loads(row["progress"] or "[]")
                if not isinstance(notes, list):
                    notes = []
            except (ValueError, TypeError):
                notes = []
            notes.append(clean)
            # Unbounded growth would eventually make the briefing larger than
            # the context window; the oldest notes are the least useful.
            del notes[:-200]
            self._conn.execute(
                "UPDATE tasks SET progress = ?, updated = ? WHERE id = ?",
                (json.dumps(notes), _now(), task_id),
            )
            self._conn.commit()
        return True

    def get_task(self, task_id: str) -> Optional[DurableTask]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (str(task_id),)
            ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(
        self,
        *,
        project: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[DurableTask]:
        sql = "SELECT * FROM tasks"
        clauses, params = [], []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if state:
            clauses.append("state = ?")
            params.append(state)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def resumable_tasks(self, *, project: Optional[str] = None) -> List[DurableTask]:
        """Everything that should come back after a restart."""
        return [t for t in self.list_tasks(project=project) if t.resumable]

    def mark_interrupted(self) -> int:
        """Called at startup: anything left RUNNING did not survive the restart.

        The thread is gone, so the honest state is PAUSED — recoverable, with
        its progress notes intact — rather than RUNNING, which would be a lie
        about a thread that no longer exists.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tasks SET state = ?, updated = ? WHERE state = ?",
                (PAUSED, _now(), RUNNING),
            )
            self._conn.commit()
            count = int(cur.rowcount or 0)
        if count:
            logger.info(
                "%d task(s) were interrupted by a restart and are now paused; "
                "they can be resumed with their progress intact",
                count,
            )
        return count

    @staticmethod
    def _row_to_task(row: Any) -> DurableTask:
        def _load(value: Any, fallback: Any) -> Any:
            try:
                out = json.loads(value or "null")
                return out if out is not None else fallback
            except (ValueError, TypeError):
                return fallback

        return DurableTask(
            id=str(row["id"]),
            project=str(row["project"]),
            goal=str(row["goal"]),
            state=str(row["state"]),
            created=float(row["created"]),
            updated=float(row["updated"]),
            parent_id=str(row["parent_id"] or ""),
            depth=int(row["depth"] or 0),
            progress=_load(row["progress"], []),
            result=str(row["result"] or ""),
            error=str(row["error"] or ""),
            metadata=_load(row["metadata"], {}),
        )

    # ------------------------------------------------------------------ #
    def summary(self, *, project: Optional[str] = None) -> str:
        """A spoken-prose status line. What the small model reads aloud."""
        proj = project or self.active_project()
        tasks = self.list_tasks(project=proj)
        if not tasks:
            return f"Nothing is running on {proj}, Sir."

        counts: Dict[str, int] = {}
        for task in tasks:
            counts[task.state] = counts.get(task.state, 0) + 1

        parts = [
            f"{n} {state}" for state, n in sorted(counts.items()) if n
        ]
        head = f"On {proj}: " + ", ".join(parts) + "."

        live = [t for t in tasks if t.state in (RUNNING, PENDING)]
        if live:
            head += " Currently: " + "; ".join(t.goal for t in live[:3]) + "."
        paused = [t for t in tasks if t.state == PAUSED]
        if paused:
            head += f" {len(paused)} on hold."
        return head

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("closing the project store failed", exc_info=True)


__all__ = [
    "Project",
    "DurableTask",
    "ProjectStore",
    "DEFAULT_PROJECT",
    "PENDING", "RUNNING", "PAUSED", "DONE", "FAILED", "CANCELLED",
    "RESUMABLE", "TERMINAL",
]
