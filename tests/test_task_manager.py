"""Background task pool: completion, failure, cancellation, timeouts, reporting."""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.agent.task_manager import TaskManager
from jarvis.core.contracts import TaskState
from jarvis.core.events import Events


@pytest.fixture
def manager():
    tm = TaskManager(max_workers=3, default_timeout=15.0)
    yield tm
    tm.shutdown(wait=False)


def _ticker(iterations=200, delay=0.01):
    """A runner that reports progress, so cancellation has somewhere to land."""

    def run(task, progress):
        for i in range(iterations):
            progress(f"step {i}", i / iterations)
            time.sleep(delay)
        return "finished"

    return run


def test_successful_task_records_result_and_progress(manager):
    def run(task, progress):
        progress("halfway", 0.5)
        return {"answer": 42}

    task = manager.spawn("compute", run)
    manager.wait(task.id, timeout=10)

    settled = manager.get(task.id)
    assert settled.state is TaskState.DONE
    assert settled.result == {"answer": 42}
    assert any(u.progress == 0.5 for u in settled.updates)


def test_failing_task_is_captured_not_raised(manager):
    def run(task, progress):
        raise ValueError("kaput")

    task = manager.spawn("break", run)
    manager.wait(task.id, timeout=10)

    settled = manager.get(task.id)
    assert settled.state is TaskState.FAILED
    assert "kaput" in settled.error
    assert "ValueError" in settled.error


def test_cancellation_is_cooperative(manager):
    task = manager.spawn("long", _ticker())
    deadline = time.monotonic() + 5
    while manager.get(task.id).state is not TaskState.RUNNING and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.cancel(task.id) is True
    manager.wait(task.id, timeout=10)
    assert manager.get(task.id).state is TaskState.CANCELLED


def test_cancel_returns_false_for_unknown_and_settled_tasks(manager):
    assert manager.cancel("does-not-exist") is False
    task = manager.spawn("quick", lambda t, p: "done")
    manager.wait(task.id, timeout=10)
    assert manager.cancel(task.id) is False


def test_timeout_fails_the_task():
    tm = TaskManager(max_workers=1, default_timeout=0.2)
    try:
        task = tm.spawn("slow", _ticker())
        tm.wait(task.id, timeout=10)
        settled = tm.get(task.id)
        assert settled.state is TaskState.FAILED
        assert "exceeded" in settled.error
    finally:
        tm.shutdown(wait=True)


def test_per_task_timeout_overrides_the_default():
    tm = TaskManager(max_workers=1, default_timeout=60.0)
    try:
        task = tm.spawn("slow", _ticker(), timeout=0.2)
        tm.wait(task.id, timeout=10)
        assert tm.get(task.id).state is TaskState.FAILED
    finally:
        tm.shutdown(wait=True)


def test_reports_are_delivered_exactly_once(manager):
    a = manager.spawn("a", lambda t, p: "A")
    b = manager.spawn("b", lambda t, p: "B")
    manager.wait(a.id, timeout=10)
    manager.wait(b.id, timeout=10)

    first = manager.take_reports()
    assert {t.id for t in first} == {a.id, b.id}
    assert manager.take_reports() == []


def test_pending_reports_does_not_consume(manager):
    task = manager.spawn("a", lambda t, p: "A")
    manager.wait(task.id, timeout=10)
    assert len(manager.pending_reports()) == 1
    assert len(manager.pending_reports()) == 1
    manager.mark_announced(task.id)
    assert manager.pending_reports() == []


def test_events_are_emitted(bus):
    seen = []
    bus.subscribe(Events.TASK_CREATED, lambda t: seen.append(("created", t.id)))
    bus.subscribe(Events.TASK_DONE, lambda t: seen.append(("done", t.id)))
    bus.subscribe(Events.TASK_FAILED, lambda t: seen.append(("failed", t.id)))

    tm = TaskManager(max_workers=2, bus=bus)
    try:
        ok = tm.spawn("ok", lambda t, p: 1)
        bad = tm.spawn("bad", lambda t, p: 1 / 0)
        tm.wait(ok.id, timeout=10)
        tm.wait(bad.id, timeout=10)
    finally:
        tm.shutdown(wait=True)

    assert ("created", ok.id) in seen
    assert ("done", ok.id) in seen
    assert ("failed", bad.id) in seen


def test_concurrency_actually_overlaps():
    """Three 0.3 s tasks on three workers must finish well under 0.9 s."""
    tm = TaskManager(max_workers=3)
    try:
        started = time.monotonic()
        tasks = [tm.spawn(f"t{i}", lambda t, p: time.sleep(0.3)) for i in range(3)]
        for task in tasks:
            tm.wait(task.id, timeout=10)
        elapsed = time.monotonic() - started
        assert elapsed < 0.8, f"tasks appear to have run serially ({elapsed:.2f}s)"
    finally:
        tm.shutdown(wait=True)


def test_list_and_stats(manager):
    manager.spawn("a", lambda t, p: 1)
    manager.spawn("b", lambda t, p: 1 / 0)
    manager.wait_all(timeout=10)

    assert len(manager.list()) == 2
    assert len(manager.list(state=TaskState.DONE)) == 1
    assert len(manager.list(state=TaskState.FAILED)) == 1

    stats = manager.stats()
    assert stats["total"] == 2 and stats["done"] == 1 and stats["failed"] == 1


def test_busy_and_active(manager):
    assert manager.busy is False
    task = manager.spawn("long", _ticker())
    deadline = time.monotonic() + 5
    while not manager.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.busy is True
    assert task.id in {t.id for t in manager.active()}
    manager.cancel(task.id)
    manager.wait(task.id, timeout=10)


def test_forget_only_removes_settled_tasks(manager):
    running = manager.spawn("long", _ticker())
    assert manager.forget(running.id) is False
    manager.cancel(running.id)
    manager.wait(running.id, timeout=10)
    assert manager.forget(running.id) is True
    assert manager.get(running.id) is None


def test_shutdown_joins_worker_threads():
    tm = TaskManager(max_workers=2)
    tm.spawn("a", lambda t, p: time.sleep(0.05))
    tm.shutdown(wait=True)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name.startswith("jarvis-task")]:
            break
        time.sleep(0.05)
    leftover = [t.name for t in threading.enumerate() if t.name.startswith("jarvis-task")]
    assert not leftover, f"worker threads survived shutdown: {leftover}"


def test_spawn_after_shutdown_raises():
    tm = TaskManager(max_workers=1)
    tm.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        tm.spawn("nope", lambda t, p: 1)


def test_baseexception_in_a_runner_does_not_kill_the_pool(manager):
    """A KeyboardInterrupt inside one task must not stop later tasks running."""
    def rude(task, progress):
        raise KeyboardInterrupt()

    bad = manager.spawn("rude", rude)
    manager.wait(bad.id, timeout=10)
    assert manager.get(bad.id).state is TaskState.FAILED

    good = manager.spawn("good", lambda t, p: "still working")
    manager.wait(good.id, timeout=10)
    assert manager.get(good.id).result == "still working"


def test_context_manager_shuts_down():
    with TaskManager(max_workers=1) as tm:
        task = tm.spawn("a", lambda t, p: 1)
        tm.wait(task.id, timeout=10)
    with pytest.raises(RuntimeError):
        tm.spawn("after", lambda t, p: 1)
