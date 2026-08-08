"""Background task management — how the main agent stays free, and how the
tree of agents underneath it stays observable and bounded.

Every non-trivial request is handed to a subagent running on a worker thread.
The main agent returns to the user immediately; when a task finishes, its report
is queued and announced at the next natural opportunity.

Threads rather than asyncio: the tool layer is overwhelmingly blocking I/O
(subprocess, sqlite, filesystem, HTTP), and a thread pool keeps that honest
without forcing every tool to be written twice.

Tasks form a tree
-----------------
A subagent can spawn subagents of its own, so the set of running tasks is a
forest, not a list.  Every task records ``parent_id`` and ``depth`` in its
metadata, children are tracked on the parent, and a parent does not settle
until its whole subtree has settled — otherwise a parent would report success
while its children were still working and their findings would be lost.

Two resource limits keep that tree from becoming a fork bomb: ``max_depth``
bounds how far delegation may recurse, and ``max_total_tasks`` bounds how many
tasks are tracked at once.  Both are generous by default and neither asks
anyone's permission: exceeding one simply returns a failed :class:`Task` whose
``error`` names the limit and the current value, so the model that tripped it
can read the message and adapt instead of retrying forever.

Avoiding the classic pool deadlock
----------------------------------
"A parent is not done until its children are done" is exactly the shape that
deadlocks a fixed worker pool: the parent occupies a worker while waiting for
children that need workers.  Two independent mechanisms rule it out here.

1. The join is *event driven* and never occupies a worker.  When a runner
   returns, its outcome is parked on the handle and the worker thread is
   released immediately; the last child to settle is what finishes the parent.
   No thread ever blocks on a descendant.

2. There is one pool *per depth level*, created lazily.  A task at depth ``d``
   only ever waits on tasks at depth ``d + 1``, which live in a different pool,
   so even a runner that deliberately blocks on ``wait()`` for its own child
   cannot starve itself.  The number of pools is bounded by ``max_depth + 1``
   and each holds at most ``max_workers`` threads, so the thread ceiling is
   ``(max_depth + 1) * max_workers`` — 16 threads with the shipped defaults.
   (A task that blocks on a *sibling* at its own depth can still starve that
   level; nothing in JARVIS does that, and it is not a case worth pessimising
   the common path for.)
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from ..core.contracts import Task, TaskState, TaskUpdate, new_id, now_ts
from ..core.events import EventBus, Events

log = logging.getLogger(__name__)


# A runner receives the task and a progress callback, and returns the result.
Runner = Callable[..., Any]

#: States from which a task never moves again.
SETTLED_STATES = (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED)

#: How deep delegation may recurse.  Root tasks sit at depth 0, so the default
#: permits a root, its children, grandchildren and great-grandchildren.
DEFAULT_MAX_DEPTH = 3

#: How many tasks the manager will track at once.  Settled tasks that have
#: already been announced are reclaimed before this limit is enforced.
DEFAULT_MAX_TOTAL_TASKS = 64


class CancelledError(Exception):
    """Raised inside a runner when its task has been cancelled."""


class TaskHandle:
    """Internal bookkeeping for one running task, including its tree links."""

    __slots__ = (
        "task", "future", "cancel_event", "announced", "settled",
        "parent_id", "depth", "children", "runner_done", "outcome",
    )

    def __init__(
        self,
        task: Task,
        *,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> None:
        self.task = task
        self.future: Optional[Future] = None
        self.cancel_event = threading.Event()
        self.announced = False
        #: Set once the task has reached a terminal state *and* its subtree has.
        self.settled = threading.Event()
        self.parent_id = parent_id
        self.depth = depth
        self.children: List[str] = []
        self.runner_done = False
        #: ``(state, result, error)`` parked by the runner until children settle.
        self.outcome: Optional[tuple] = None


class TaskManager:
    """Owns the worker pools, the task tree, and the report queue."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        bus: Optional[EventBus] = None,
        default_timeout: float = 900.0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_total_tasks: int = DEFAULT_MAX_TOTAL_TASKS,
    ) -> None:
        self._max_workers = max(1, int(max_workers))
        self._bus = bus
        self._default_timeout = default_timeout
        self._max_depth = max(0, int(max_depth))
        self._max_total_tasks = max(1, int(max_total_tasks))
        self._pools: Dict[int, ThreadPoolExecutor] = {}
        self._handles: Dict[str, TaskHandle] = {}
        self._lock = threading.RLock()
        self._shutdown = False
        self._refused = 0

    # -- limits ------------------------------------------------------------- #
    @property
    def max_workers(self) -> int:
        """Concurrent tasks permitted at each depth level."""
        return self._max_workers

    @property
    def max_depth(self) -> int:
        """Deepest permitted task depth; roots are depth 0."""
        return self._max_depth

    @property
    def max_total_tasks(self) -> int:
        """How many tasks may be tracked at once."""
        return self._max_total_tasks

    def child_depth(self, parent_id: Optional[str] = None) -> int:
        """The depth a child spawned under ``parent_id`` would occupy."""
        if not parent_id:
            return 0
        with self._lock:
            handle = self._handles.get(parent_id)
        return 0 if handle is None else handle.depth + 1

    # -- pools -------------------------------------------------------------- #
    def _pool_for(self, depth: int) -> ThreadPoolExecutor:
        """One pool per depth level — see the module docstring on deadlock."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("TaskManager has been shut down")
            pool = self._pools.get(depth)
            if pool is None:
                pool = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="jarvis-task-d%d" % depth,
                )
                self._pools[depth] = pool
            return pool

    # -- lifecycle ---------------------------------------------------------- #
    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        """Stop accepting work and (optionally) wait for running tasks."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            handles = list(self._handles.values())
            pools = list(self._pools.values())

        if cancel_pending:
            for handle in handles:
                if handle.task.state in (TaskState.PENDING, TaskState.RUNNING):
                    handle.cancel_event.set()

        for pool in pools:
            # cancel_futures needs Python 3.9+; it only affects queued, not running.
            try:
                pool.shutdown(wait=wait, cancel_futures=cancel_pending)
            except TypeError:  # pragma: no cover - very old interpreters
                pool.shutdown(wait=wait)

        # A future the pool dropped before it ever ran would leave its task
        # un-settled forever, and wait() would block on it for its full timeout.
        for handle in handles:
            if not handle.settled.is_set():
                self._force_settle(
                    handle, TaskState.CANCELLED, "cancelled at shutdown"
                )

    def __enter__(self) -> "TaskManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=False)

    # -- submission --------------------------------------------------------- #
    def spawn(
        self,
        goal: str,
        runner: Runner,
        *,
        timeout: Optional[float] = None,
        metadata: Optional[dict] = None,
        task_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Task:
        """Queue ``runner`` to pursue ``goal`` under ``parent_id``; returns at once.

        Depth is derived from the parent, defaulting to 0 at the root.  When a
        resource limit would be exceeded the spawn is *refused* rather than
        raised: the returned :class:`Task` is already ``FAILED``, carries an
        explanatory ``error``, and is not registered with the manager.
        """
        if self._shutdown:
            raise RuntimeError("TaskManager has been shut down")

        with self._lock:
            parent = self._handles.get(parent_id) if parent_id else None
            if parent_id and parent is None:
                return self._refuse(
                    goal, metadata, parent_id, 0,
                    "spawn refused: no task '%s' is being tracked, so the new task "
                    "has no parent to attach to. Use task_tree or list_tasks to see "
                    "the live task ids." % parent_id,
                )

            depth = (parent.depth + 1) if parent is not None else 0
            if depth > self._max_depth:
                return self._refuse(
                    goal, metadata, parent_id, depth,
                    "spawn refused: max_depth=%d reached. Task %s is already at "
                    "depth %d, so a child of it would sit at depth %d. Do not "
                    "delegate any further down this branch — carry out this step "
                    "yourself and put the outcome in your report."
                    % (self._max_depth, parent_id, parent.depth, depth),
                )

            self._reap_announced_locked()
            tracked = len(self._handles)
            if tracked >= self._max_total_tasks:
                running = sum(
                    1 for h in self._handles.values()
                    if h.task.state in (TaskState.PENDING, TaskState.RUNNING)
                )
                return self._refuse(
                    goal, metadata, parent_id, depth,
                    "spawn refused: max_total_tasks=%d reached (%d tasks tracked, "
                    "%d still running). Wait for running work to finish, cancel a "
                    "task you no longer need, or do this step inline instead of "
                    "delegating." % (self._max_total_tasks, tracked, running),
                )

            task = Task(
                id=task_id or new_id("task"), goal=goal, metadata=dict(metadata or {})
            )
            task.metadata["depth"] = depth
            task.metadata["parent_id"] = parent_id
            handle = TaskHandle(task, parent_id=parent_id, depth=depth)
            self._handles[task.id] = handle
            if parent is not None:
                parent.children.append(task.id)

        self._emit(Events.TASK_CREATED, task)
        try:
            handle.future = self._pool_for(depth).submit(
                self._execute, handle, runner, timeout or self._default_timeout
            )
        except RuntimeError as exc:
            # A concurrent shutdown closed the pool between the check above and
            # the submit.  The task is already registered, so it must be settled
            # here or wait() would block on it forever.
            self._force_settle(handle, TaskState.FAILED, f"could not start: {exc}")
        return task

    def _refuse(
        self,
        goal: str,
        metadata: Optional[dict],
        parent_id: Optional[str],
        depth: int,
        message: str,
    ) -> Task:
        """Build an un-registered, already-failed Task explaining a limit."""
        with self._lock:
            self._refused += 1
        log.warning("%s", message)
        task = Task(
            id=new_id("task"),
            goal=goal,
            state=TaskState.FAILED,
            error=message,
            metadata=dict(metadata or {}),
        )
        task.metadata.update(
            {"depth": depth, "parent_id": parent_id, "refused": True}
        )
        task.updates.append(
            TaskUpdate(
                task_id=task.id, state=TaskState.FAILED,
                message=message, error=message,
            )
        )
        return task

    def _reap_announced_locked(self) -> int:
        """Reclaim settled, already-announced leaf tasks to stay under the cap.

        Purely resource management: without it a long session would fill the
        registry with finished work and refuse new tasks forever.  Their reports
        have already been delivered by the time they qualify.
        """
        if len(self._handles) < self._max_total_tasks:
            return 0
        reapable = [
            h for h in self._handles.values()
            if h.announced and h.settled.is_set() and not h.children
        ]
        reapable.sort(key=lambda h: h.task.created)
        freed = 0
        while reapable and len(self._handles) >= self._max_total_tasks:
            handle = reapable.pop(0)
            self._handles.pop(handle.task.id, None)
            freed += 1
        if freed:
            log.debug("reclaimed %d finished task(s) to stay under the cap", freed)
        return freed

    # -- execution ---------------------------------------------------------- #
    def _execute(self, handle: TaskHandle, runner: Runner, timeout: float) -> Any:
        task = handle.task

        if handle.settled.is_set():
            return None
        if handle.cancel_event.is_set():
            self._runner_finished(
                handle, TaskState.CANCELLED, error="cancelled before start"
            )
            return None

        self._set_state(handle, TaskState.RUNNING, "started")
        started = time.monotonic()

        def progress(message: str, fraction: Optional[float] = None) -> None:
            if handle.cancel_event.is_set():
                raise CancelledError(task.id)
            if timeout and (time.monotonic() - started) > timeout:
                raise TimeoutError(f"task exceeded {timeout:.0f}s")
            update = TaskUpdate(
                task_id=task.id, state=TaskState.RUNNING, message=message, progress=fraction
            )
            with self._lock:
                task.updates.append(update)
                task.updated = now_ts()
            self._emit(Events.TASK_UPDATE, update)

        try:
            result = runner(task, progress)
        except CancelledError:
            self._runner_finished(handle, TaskState.CANCELLED, error="cancelled")
            return None
        except TimeoutError as exc:
            self._runner_finished(handle, TaskState.FAILED, error=str(exc))
            return None
        except BaseException as exc:  # noqa: BLE001 - a task must never kill the pool
            log.exception("task %s failed", task.id)
            self._runner_finished(
                handle, TaskState.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
            return None

        if handle.cancel_event.is_set():
            self._runner_finished(handle, TaskState.CANCELLED, error="cancelled")
            return None

        self._runner_finished(handle, TaskState.DONE, result=result)
        return result

    def _runner_finished(
        self,
        handle: TaskHandle,
        state: TaskState,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Park the runner's outcome and release this worker thread.

        The task itself only settles once its children have; that handover
        happens on whichever thread settles the last child, never here.
        """
        with self._lock:
            handle.outcome = (state, result, error)
            handle.runner_done = True
            pending = self._pending_children_locked(handle)
            if pending:
                update = TaskUpdate(
                    task_id=handle.task.id,
                    state=TaskState.RUNNING,
                    message="awaiting %d child task(s)" % len(pending),
                )
                handle.task.updates.append(update)
                handle.task.updated = now_ts()
            else:
                update = None

        if update is not None:
            self._emit(Events.TASK_UPDATE, update)
        self._try_settle(handle)

    def _pending_children_locked(self, handle: TaskHandle) -> List[str]:
        return [
            cid for cid in handle.children
            if cid in self._handles and not self._handles[cid].settled.is_set()
        ]

    def _try_settle(self, handle: TaskHandle) -> None:
        """Settle ``handle`` if its runner and its whole subtree are finished."""
        with self._lock:
            if handle.settled.is_set() or not handle.runner_done:
                return
            if self._pending_children_locked(handle):
                return
            state, result, error = handle.outcome or (TaskState.DONE, None, None)
            if state is TaskState.DONE and handle.cancel_event.is_set():
                # Cancelled while it sat waiting on its children.
                state, error = TaskState.CANCELLED, error or "cancelled"
            result = self._fold_children_locked(handle, result)
            parent = self._handles.get(handle.parent_id) if handle.parent_id else None

        if self._finish(handle, state, result=result, error=error) and parent is not None:
            self._try_settle(parent)

    def _fold_children_locked(self, handle: TaskHandle, result: Any) -> Any:
        """Attach the children's reports to a dict result, so a parent's own
        report carries everything its subtree found."""
        reports = self._child_reports_locked(handle)
        if reports and isinstance(result, dict):
            merged = dict(result)
            merged["children"] = reports
            return merged
        return result

    def _child_reports_locked(self, handle: TaskHandle) -> List[dict]:
        out: List[dict] = []
        for cid in handle.children:
            child = self._handles.get(cid)
            if child is None:
                continue
            task = child.task
            entry: dict = {
                "task_id": task.id,
                "goal": task.goal,
                "state": task.state.value,
                "depth": child.depth,
            }
            if isinstance(task.result, dict):
                entry["report"] = task.result.get("report", "")
                nested = task.result.get("children")
                if nested:
                    entry["children"] = nested
            elif task.result is not None:
                entry["report"] = str(task.result)
            if task.error:
                entry["error"] = task.error
            out.append(entry)
        return out

    def _set_state(self, handle: TaskHandle, state: TaskState, message: str = "") -> None:
        with self._lock:
            if handle.settled.is_set():
                return
            handle.task.state = state
            handle.task.updated = now_ts()
            update = TaskUpdate(task_id=handle.task.id, state=state, message=message)
            handle.task.updates.append(update)
        self._emit(Events.TASK_UPDATE, update)

    def _finish(
        self,
        handle: TaskHandle,
        state: TaskState,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """Move a task to its terminal state exactly once.  Returns whether
        this call was the one that did it."""
        task = handle.task
        with self._lock:
            if handle.settled.is_set():
                return False
            task.state = state
            task.result = result
            task.error = error
            task.updated = now_ts()
            update = TaskUpdate(
                task_id=task.id, state=state, result=result, error=error,
                message=error or "completed",
            )
            task.updates.append(update)
            handle.settled.set()

        self._emit(Events.TASK_UPDATE, update)
        self._emit(Events.TASK_DONE if state is TaskState.DONE else Events.TASK_FAILED, task)
        return True

    def _force_settle(self, handle: TaskHandle, state: TaskState, error: str) -> None:
        """Settle a task out of band (cancellation, shutdown) and tell its parent."""
        if not self._finish(handle, state, error=error):
            return
        with self._lock:
            parent = self._handles.get(handle.parent_id) if handle.parent_id else None
        if parent is not None:
            self._try_settle(parent)

    def _emit(self, channel: str, payload: Any) -> None:
        if self._bus is not None:
            self._bus.emit(channel, payload)

    # -- queries ------------------------------------------------------------ #
    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            handle = self._handles.get(task_id)
            return handle.task if handle else None

    def list(self, *, state: Optional[TaskState] = None) -> list:
        with self._lock:
            tasks = [h.task for h in self._handles.values()]
        if state is not None:
            tasks = [t for t in tasks if t.state is state]
        return sorted(tasks, key=lambda t: t.created)

    def active(self) -> list:
        with self._lock:
            return [
                h.task
                for h in self._handles.values()
                if h.task.state in (TaskState.PENDING, TaskState.RUNNING)
            ]

    @property
    def busy(self) -> bool:
        return bool(self.active())

    # -- the tree ----------------------------------------------------------- #
    def depth_of(self, task_id: str) -> Optional[int]:
        """Depth of one task, or ``None`` if it is not tracked."""
        with self._lock:
            handle = self._handles.get(task_id)
        return None if handle is None else handle.depth

    def parent_of(self, task_id: str) -> Optional[str]:
        """The id of a task's parent, or ``None`` for a root or unknown task."""
        with self._lock:
            handle = self._handles.get(task_id)
        return None if handle is None else handle.parent_id

    def children(self, task_id: str) -> list:
        """Direct children of a task, oldest first."""
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is None:
                return []
            return [
                self._handles[cid].task
                for cid in handle.children
                if cid in self._handles
            ]

    def descendants(self, task_id: str) -> list:
        """Every task below ``task_id``, breadth-first."""
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is None:
                return []
            return [h.task for h in self._descendant_handles_locked(handle)]

    def _descendant_handles_locked(self, handle: TaskHandle) -> List[TaskHandle]:
        out: List[TaskHandle] = []
        seen = {handle.task.id}
        queue = list(handle.children)
        while queue:
            cid = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            child = self._handles.get(cid)
            if child is None:
                continue
            out.append(child)
            queue.extend(child.children)
        return out

    def ancestry(self, task_id: str) -> List[str]:
        """The chain of task ids from the root down to and including ``task_id``.

        Empty for an unknown task; ``[task_id]`` for a root.
        """
        chain: List[str] = []
        with self._lock:
            seen = set()
            current = self._handles.get(task_id)
            while current is not None and current.task.id not in seen:
                seen.add(current.task.id)
                chain.append(current.task.id)
                current = (
                    self._handles.get(current.parent_id) if current.parent_id else None
                )
        chain.reverse()
        return chain

    def roots(self) -> list:
        """Tasks with no tracked parent, oldest first."""
        with self._lock:
            handles = [
                h for h in self._handles.values()
                if not h.parent_id or h.parent_id not in self._handles
            ]
        return sorted((h.task for h in handles), key=lambda t: t.created)

    def tree(self, root_id: Optional[str] = None) -> list:
        """Nested ``{task_id, goal, state, depth, parent_id, children}`` nodes.

        Without ``root_id`` this is the whole forest; with one it is a single
        element list holding that task's subtree (empty if it is unknown).
        """
        with self._lock:
            if root_id is not None:
                handle = self._handles.get(root_id)
                return [] if handle is None else [self._node_locked(handle, set())]
            handles = [
                h for h in self._handles.values()
                if not h.parent_id or h.parent_id not in self._handles
            ]
            handles.sort(key=lambda h: h.task.created)
            return [self._node_locked(h, set()) for h in handles]

    def _node_locked(self, handle: TaskHandle, seen: set) -> dict:
        task = handle.task
        seen.add(task.id)
        node = {
            "task_id": task.id,
            "goal": task.goal,
            "state": task.state.value,
            "depth": handle.depth,
            "parent_id": handle.parent_id,
            "children": [],
        }
        for cid in handle.children:
            child = self._handles.get(cid)
            if child is not None and cid not in seen:
                node["children"].append(self._node_locked(child, seen))
        return node

    def render_tree(self, root_id: Optional[str] = None, *, max_goal: int = 70) -> str:
        """A compact, indented text rendering of the tree — one line per task."""
        lines: List[str] = []

        def walk(node: dict, indent: int) -> None:
            goal = node["goal"] or ""
            if len(goal) > max_goal:
                goal = goal[: max_goal - 3] + "..."
            lines.append(
                "%s%s [%s] %s" % ("  " * indent, node["task_id"], node["state"], goal)
            )
            for child in node["children"]:
                walk(child, indent + 1)

        for node in self.tree(root_id):
            walk(node, 0)
        return "\n".join(lines) if lines else "(no background tasks)"

    # -- control ------------------------------------------------------------ #
    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a task *and its whole subtree*.

        Cooperative: each runner stops at its next ``progress()`` call.  Returns
        False for an unknown or already-settled task.
        """
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is None:
                return False
            if handle.settled.is_set() or handle.task.state in SETTLED_STATES:
                return False
            targets = [handle] + self._descendant_handles_locked(handle)

        # Deepest first, so a parent never briefly observes an empty subtree.
        for target in reversed(targets):
            if target.settled.is_set():
                continue
            target.cancel_event.set()
            future = target.future
            # A task that never started cancels outright; a running one stops at
            # its next progress() call.
            if future is not None and future.cancel():
                self._force_settle(
                    target, TaskState.CANCELLED, "cancelled before start"
                )
        return True

    def cancel_all(self) -> int:
        return sum(1 for task in self.roots() if self.cancel(task.id))

    def wait(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """Block until a task *and its subtree* settle.  Mostly tests and the CLI."""
        with self._lock:
            handle = self._handles.get(task_id)
        if handle is None:
            return None
        handle.settled.wait(timeout)
        return handle.task

    def wait_all(self, timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        for task in self.list():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self.wait(task.id, remaining)

    # -- reporting ---------------------------------------------------------- #
    def pending_reports(self) -> list:
        """Finished tasks whose outcome has not yet been announced.

        Depth is irrelevant here on purpose: a grandchild's report reaches the
        top of the tree by exactly the same route a direct child's does.
        """
        with self._lock:
            return [
                h.task
                for h in self._handles.values()
                if not h.announced and h.task.state in SETTLED_STATES
            ]

    def mark_announced(self, task_id: str) -> None:
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is not None:
                handle.announced = True

    def take_reports(self) -> list:
        """Atomically fetch and mark the pending reports, oldest first."""
        with self._lock:
            ready = [
                h
                for h in self._handles.values()
                if not h.announced and h.task.state in SETTLED_STATES
            ]
            ready.sort(key=lambda h: (h.depth, h.task.updated))
            for handle in ready:
                handle.announced = True
            return [h.task for h in ready]

    def forget(self, task_id: str) -> bool:
        """Drop a settled task from the registry (frees its result)."""
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is None or handle.task.state in (TaskState.PENDING, TaskState.RUNNING):
                return False
            del self._handles[task_id]
            parent = self._handles.get(handle.parent_id) if handle.parent_id else None
            if parent is not None and task_id in parent.children:
                parent.children.remove(task_id)
            return True

    def stats(self) -> dict:
        counts: dict = {state.value: 0 for state in TaskState}
        with self._lock:
            handles = list(self._handles.values())
            refused = self._refused
        by_depth: Dict[int, int] = {}
        roots = 0
        deepest = 0
        for handle in handles:
            counts[handle.task.state.value] += 1
            by_depth[handle.depth] = by_depth.get(handle.depth, 0) + 1
            deepest = max(deepest, handle.depth)
            if not handle.parent_id:
                roots += 1
        counts["total"] = sum(counts[s.value] for s in TaskState)
        counts["workers"] = self._max_workers
        counts["tracked"] = len(handles)
        counts["roots"] = roots
        counts["deepest_depth"] = deepest
        counts["by_depth"] = by_depth
        counts["max_depth"] = self._max_depth
        counts["max_total_tasks"] = self._max_total_tasks
        counts["refused"] = refused
        return counts


__all__ = [
    "TaskManager",
    "TaskHandle",
    "CancelledError",
    "Runner",
    "SETTLED_STATES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_TOTAL_TASKS",
]
