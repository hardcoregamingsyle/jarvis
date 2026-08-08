"""The tree of agents: depth, fan-out limits, subtree cancellation, and the
join that must not deadlock the worker pool.

Nothing here touches the network, a model, or a real subagent: runners are
plain callables and the LLMs are scripted.  Every wait is bounded so a
regression shows up as a failed assertion rather than a hung suite.
"""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.agent.protocol import format_tool_call
from jarvis.agent.subagent import (
    SubAgent,
    agent_context,
    current_agent_context,
    delegation_note,
)
from jarvis.agent.task_manager import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_TOTAL_TASKS,
    TaskManager,
)
from jarvis.core.contracts import Message, TaskState, ToolResult


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def manager():
    tm = TaskManager(max_workers=3, default_timeout=15.0)
    yield tm
    tm.shutdown(wait=False)


def leaf(value="leaf"):
    """A runner that does nothing but succeed."""

    def run(task, progress):
        progress("working", 0.5)
        return {"report": value}

    return run


def spawner(tm, count, child_runner, *, label="child"):
    """A runner that spawns ``count`` children under itself and returns at once."""

    def run(task, progress):
        ids = []
        for i in range(count):
            child = tm.spawn(
                f"{label}-{i} of {task.goal}", child_runner, parent_id=task.id
            )
            ids.append(child.id)
        return {"report": f"spawned {len(ids)}", "spawned": ids}

    return run


def chain(tm, remaining, *, label="link"):
    """A runner that spawns exactly one child, which spawns one, ... ``remaining``
    levels deep.  Every parent therefore has to wait on its descendants."""

    def run(task, progress):
        progress("delegating", 0.1)
        if remaining <= 0:
            return {"report": "bottom"}
        child = tm.spawn(
            f"{label}-{remaining}", chain(tm, remaining - 1, label=label),
            parent_id=task.id,
        )
        return {"report": f"delegated to {child.id}", "child": child.id}

    return run


class LiveRegistry:
    """A registry stand-in that actually *calls* the functions registered on it.

    The shared ``fake_registry`` fixture stubs ``register_function`` out, which
    is fine for loop tests but useless here: these tests are about what the
    orchestrator's own meta-tools return when the model invokes them.
    """

    def __init__(self, results=None):
        self.functions: dict = dict(results or {})
        self.calls: list = []

    def names(self) -> list:
        return sorted(self.functions)

    def has(self, name: str) -> bool:
        return name in self.functions

    def describe(self) -> str:
        return "\n".join(f"- {n}" for n in sorted(self.functions))

    def register_function(self, fn, **kwargs):
        self.functions[fn.__name__] = fn
        return fn

    def run(self, name: str, **kwargs) -> ToolResult:
        self.calls.append((name, kwargs))
        fn = self.functions.get(name)
        if fn is None:
            return ToolResult.failure(f"Unknown tool: {name!r}")
        try:
            value = fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - registry.run never raises
            return ToolResult.failure(f"{type(exc).__name__}: {exc}")
        if isinstance(value, ToolResult):
            return value
        return ToolResult.success(value)


def settle(tm, task_id, timeout=8.0):
    """Wait for one task and assert it actually finished inside the timeout."""
    tm.wait(task_id, timeout=timeout)
    task = tm.get(task_id)
    assert task is not None
    assert task.state in (
        TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED
    ), f"task {task_id} never settled (state={task.state})"
    return task


# --------------------------------------------------------------------------- #
#  Depth is recorded and inherited
# --------------------------------------------------------------------------- #
def test_root_tasks_sit_at_depth_zero(manager):
    task = manager.spawn("root", leaf())
    assert task.metadata["depth"] == 0
    assert task.metadata["parent_id"] is None
    assert manager.depth_of(task.id) == 0
    settle(manager, task.id)


def test_depth_is_inherited_from_the_parent(manager):
    root = manager.spawn("root", leaf())
    child = manager.spawn("child", leaf(), parent_id=root.id)
    grandchild = manager.spawn("grandchild", leaf(), parent_id=child.id)

    assert (child.metadata["depth"], grandchild.metadata["depth"]) == (1, 2)
    assert manager.depth_of(grandchild.id) == 2
    assert manager.parent_of(grandchild.id) == child.id
    assert manager.child_depth(grandchild.id) == 3
    assert manager.child_depth(None) == 0
    settle(manager, root.id)


def test_spawning_under_an_unknown_parent_is_refused_not_raised(manager):
    task = manager.spawn("orphan", leaf(), parent_id="task_nope")
    assert task.state is TaskState.FAILED
    assert "task_nope" in task.error
    assert manager.get(task.id) is None


# --------------------------------------------------------------------------- #
#  Limits: generous, explicit, and never raised into the caller
# --------------------------------------------------------------------------- #
def test_exceeding_max_depth_fails_the_spawn_with_a_readable_message():
    tm = TaskManager(max_workers=2, max_depth=2)
    try:
        root = tm.spawn("root", leaf())
        a = tm.spawn("a", leaf(), parent_id=root.id)
        b = tm.spawn("b", leaf(), parent_id=a.id)
        assert b.metadata["depth"] == 2

        refused = tm.spawn("too deep", leaf(), parent_id=b.id)
        assert refused.state is TaskState.FAILED
        assert refused.metadata["refused"] is True
        # The message must name the limit and the current value so the model
        # can adapt instead of retrying forever.
        assert "max_depth=2" in refused.error
        assert "depth 2" in refused.error and "depth 3" in refused.error
        # The refused task is not tracked, so it cannot consume the budget.
        assert tm.get(refused.id) is None
        assert len(tm.list()) == 3
    finally:
        tm.shutdown(wait=False)


def test_a_parent_survives_a_refused_child_spawn():
    """The whole point of refusing rather than raising."""
    tm = TaskManager(max_workers=2, max_depth=1)

    def greedy(task, progress):
        refusals = []
        for i in range(3):
            child = tm.spawn(f"child {i}", leaf(), parent_id=task.id)
            if child.state is TaskState.FAILED:
                refusals.append(child.error)
        return {"report": "carried on regardless", "refusals": refusals}

    try:
        root = tm.spawn("root", greedy)
        first = tm.spawn("depth one", greedy, parent_id=root.id)
        settled = settle(tm, first.id)

        assert settled.state is TaskState.DONE, settled.error
        assert settled.result["report"] == "carried on regardless"
        assert len(settled.result["refusals"]) == 3
        assert all("max_depth=1" in r for r in settled.result["refusals"])
    finally:
        tm.shutdown(wait=False)


def test_exceeding_max_total_tasks_fails_the_spawn_with_a_readable_message():
    tm = TaskManager(max_workers=2, max_total_tasks=4)
    try:
        accepted = [tm.spawn(f"t{i}", leaf()) for i in range(4)]
        assert all(t.state is not TaskState.FAILED for t in accepted)

        refused = tm.spawn("one too many", leaf())
        assert refused.state is TaskState.FAILED
        assert "max_total_tasks=4" in refused.error
        assert "4 tasks tracked" in refused.error
        assert tm.get(refused.id) is None
        assert len(tm.list()) == 4
    finally:
        tm.shutdown(wait=False)


def test_announced_finished_tasks_are_reclaimed_so_the_cap_is_not_a_dead_end():
    """A long session must not wedge once it has run max_total_tasks tasks."""
    tm = TaskManager(max_workers=2, max_total_tasks=3)
    try:
        for i in range(3):
            settle(tm, tm.spawn(f"t{i}", leaf()).id)
        assert tm.spawn("blocked", leaf()).state is TaskState.FAILED

        tm.take_reports()                      # the reports have been delivered
        fresh = tm.spawn("after reporting", leaf())
        assert fresh.state is not TaskState.FAILED
        settle(tm, fresh.id)
        assert len(tm.list()) <= 3
    finally:
        tm.shutdown(wait=False)


def test_limits_are_generous_by_default():
    tm = TaskManager(max_workers=2)
    try:
        assert tm.max_depth == DEFAULT_MAX_DEPTH == 3
        assert tm.max_total_tasks == DEFAULT_MAX_TOTAL_TASKS == 64
    finally:
        tm.shutdown(wait=False)


# --------------------------------------------------------------------------- #
#  Tree navigation
# --------------------------------------------------------------------------- #
@pytest.fixture
def three_levels(manager):
    """root -> (a, b); a -> (a1, a2); b -> (b1).  All finished."""
    root = manager.spawn("root", leaf())
    a = manager.spawn("a", leaf(), parent_id=root.id)
    b = manager.spawn("b", leaf(), parent_id=root.id)
    a1 = manager.spawn("a1", leaf(), parent_id=a.id)
    a2 = manager.spawn("a2", leaf(), parent_id=a.id)
    b1 = manager.spawn("b1", leaf(), parent_id=b.id)
    settle(manager, root.id)
    return {"root": root, "a": a, "b": b, "a1": a1, "a2": a2, "b1": b1}


def test_children_are_direct_only(manager, three_levels):
    t = three_levels
    assert [c.id for c in manager.children(t["root"].id)] == [t["a"].id, t["b"].id]
    assert [c.id for c in manager.children(t["a"].id)] == [t["a1"].id, t["a2"].id]
    assert manager.children(t["a1"].id) == []
    assert manager.children("task_unknown") == []


def test_descendants_reach_the_whole_subtree(manager, three_levels):
    t = three_levels
    assert {d.id for d in manager.descendants(t["root"].id)} == {
        t["a"].id, t["b"].id, t["a1"].id, t["a2"].id, t["b1"].id
    }
    assert {d.id for d in manager.descendants(t["a"].id)} == {t["a1"].id, t["a2"].id}
    assert manager.descendants(t["b1"].id) == []


def test_ancestry_is_the_path_from_the_root_down(manager, three_levels):
    t = three_levels
    assert manager.ancestry(t["a1"].id) == [t["root"].id, t["a"].id, t["a1"].id]
    assert manager.ancestry(t["root"].id) == [t["root"].id]
    assert manager.ancestry("task_unknown") == []


def test_tree_is_nested_and_matches_the_structure(manager, three_levels):
    t = three_levels
    forest = manager.tree()
    assert len(forest) == 1, "there should be exactly one root"

    root_node = forest[0]
    assert root_node["task_id"] == t["root"].id
    assert root_node["depth"] == 0 and root_node["parent_id"] is None

    by_goal = {node["goal"]: node for node in root_node["children"]}
    assert set(by_goal) == {"a", "b"}
    assert [c["goal"] for c in by_goal["a"]["children"]] == ["a1", "a2"]
    assert by_goal["a"]["children"][0]["depth"] == 2
    assert by_goal["b"]["children"][0]["goal"] == "b1"

    # Scoped to a branch it returns just that subtree.
    branch = manager.tree(t["a"].id)
    assert len(branch) == 1 and branch[0]["task_id"] == t["a"].id
    assert len(branch[0]["children"]) == 2
    assert manager.tree("task_unknown") == []


def test_render_tree_indents_by_depth(manager, three_levels):
    rendered = manager.render_tree()
    lines = rendered.splitlines()
    assert lines[0].startswith(three_levels["root"].id)
    a_line = next(line for line in lines if line.strip().endswith(" a"))
    a1_line = next(line for line in lines if line.strip().endswith(" a1"))
    assert len(a1_line) - len(a1_line.lstrip()) > len(a_line) - len(a_line.lstrip())
    assert TaskManager(max_workers=1).render_tree() == "(no background tasks)"


def test_roots_lists_only_top_level_tasks(manager, three_levels):
    assert [r.id for r in manager.roots()] == [three_levels["root"].id]


# --------------------------------------------------------------------------- #
#  A parent does not settle until its subtree does
# --------------------------------------------------------------------------- #
def test_parent_stays_running_until_its_child_finishes(manager):
    release = threading.Event()

    def slow_child(task, progress):
        progress("waiting", 0.1)
        assert release.wait(timeout=8), "the test never released the child"
        return {"report": "child done"}

    root = manager.spawn("root", spawner(manager, 1, slow_child))

    deadline = time.monotonic() + 5
    while not manager.children(root.id) and time.monotonic() < deadline:
        time.sleep(0.01)
    child = manager.children(root.id)[0]

    # The runner has long since returned, but the task must not be DONE yet.
    manager.wait(root.id, timeout=0.4)
    assert manager.get(root.id).state is TaskState.RUNNING
    assert manager.pending_reports() == []

    release.set()
    settle(manager, root.id)
    assert manager.get(root.id).state is TaskState.DONE
    assert manager.get(child.id).state is TaskState.DONE


def test_a_parents_result_carries_its_childrens_reports(manager):
    root = manager.spawn("root", spawner(manager, 2, leaf("child says hello")))
    settled = settle(manager, root.id)

    children = settled.result["children"]
    assert len(children) == 2
    assert all(c["report"] == "child says hello" for c in children)
    assert all(c["state"] == "done" and c["depth"] == 1 for c in children)


def test_child_reports_nest_all_the_way_down(manager):
    root = manager.spawn("root", spawner(manager, 1, spawner(manager, 1, leaf("deep"))))
    settled = settle(manager, root.id)

    child = settled.result["children"][0]
    grandchild = child["children"][0]
    assert grandchild["report"] == "deep"
    assert grandchild["depth"] == 2


def test_waiting_on_children_does_not_stop_the_worker_serving_others(manager):
    """The parent's join must not hold a worker slot."""
    release = threading.Event()

    def blocked_child(task, progress):
        assert release.wait(timeout=8)
        return {"report": "released"}

    parents = [
        manager.spawn(f"p{i}", spawner(manager, 1, blocked_child)) for i in range(3)
    ]
    deadline = time.monotonic() + 5
    while sum(len(manager.children(p.id)) for p in parents) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)

    # Three parents are mid-join on a three-worker pool; unrelated work must
    # still get through.
    quick = manager.spawn("unrelated", leaf("still responsive"))
    assert settle(manager, quick.id, timeout=5).result["report"] == "still responsive"

    release.set()
    for parent in parents:
        assert settle(manager, parent.id).state is TaskState.DONE


# --------------------------------------------------------------------------- #
#  *** POOL DEADLOCK ***
# --------------------------------------------------------------------------- #
def test_a_tree_deeper_than_the_worker_count_still_completes():
    """The classic fixed-pool deadlock, proved absent.

    One worker per depth level, a chain five deep where every parent is waiting
    on its child, and a hard eight-second budget.  A design that joined inside
    the worker slot, or that shared one pool across depths, would hang here.
    """
    tm = TaskManager(max_workers=1, max_depth=5, default_timeout=20.0)
    try:
        started = time.monotonic()
        root = tm.spawn("level 5", chain(tm, 5))
        tm.wait(root.id, timeout=8)
        elapsed = time.monotonic() - started

        settled = tm.get(root.id)
        assert settled.state is TaskState.DONE, (
            f"the tree deadlocked or failed: {settled.state} {settled.error}"
        )
        assert elapsed < 8, f"the tree took {elapsed:.1f}s — that is a hang"
        assert len(tm.list()) == 6
        assert max(t.metadata["depth"] for t in tm.list()) == 5
        assert all(t.state is TaskState.DONE for t in tm.list())
    finally:
        tm.shutdown(wait=False)


def test_a_runner_that_blocks_on_its_own_child_does_not_deadlock():
    """Even a runner that explicitly waits inside its worker slot survives,
    because each depth level owns its own pool."""
    tm = TaskManager(max_workers=1, max_depth=4, default_timeout=20.0)

    def blocking_parent(levels):
        def run(task, progress):
            if levels <= 0:
                return {"report": "bottom"}
            child = tm.spawn(
                f"blocking {levels}", blocking_parent(levels - 1), parent_id=task.id
            )
            settled = tm.wait(child.id, timeout=6)
            return {"report": f"child said {settled.result['report']}"}
        return run

    try:
        started = time.monotonic()
        root = tm.spawn("blocking root", blocking_parent(3))
        tm.wait(root.id, timeout=8)
        elapsed = time.monotonic() - started

        settled = tm.get(root.id)
        assert settled.state is TaskState.DONE, f"deadlocked: {settled.error}"
        assert elapsed < 8, f"took {elapsed:.1f}s — that is a hang"
        assert "child said" in settled.result["report"]
    finally:
        tm.shutdown(wait=False)


def test_independent_leaves_overlap_rather_than_serialising():
    """A tree of independent leaves must finish far under the serial sum."""
    tm = TaskManager(max_workers=4, max_depth=2)

    def sleeper(task, progress):
        progress("sleeping", 0.5)
        time.sleep(0.4)
        return {"report": "slept"}

    try:
        started = time.monotonic()
        root = tm.spawn("fan out", spawner(tm, 4, sleeper))
        tm.wait(root.id, timeout=8)
        elapsed = time.monotonic() - started

        assert tm.get(root.id).state is TaskState.DONE
        assert elapsed < 1.2, (
            f"four 0.4s leaves took {elapsed:.2f}s — they ran serially"
        )
    finally:
        tm.shutdown(wait=False)


def test_shutdown_joins_the_workers_of_every_depth():
    """One pool per depth is only affordable if they all go away again."""
    before = {t.ident for t in threading.enumerate()}
    tm = TaskManager(max_workers=2, max_depth=3)
    root = tm.spawn("root", chain(tm, 3))
    settle(tm, root.id)
    assert tm.stats()["deepest_depth"] == 3, "the chain never reached depth 3"
    tm.shutdown(wait=True)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        leftover = [
            t.name for t in threading.enumerate()
            if t.ident not in before and t.name.startswith("jarvis-task")
        ]
        if not leftover:
            break
        time.sleep(0.05)
    assert not leftover, f"worker threads survived shutdown: {leftover}"


def test_shutdown_settles_a_task_the_pool_never_got_to_run():
    """Otherwise wait() would block on it for its whole timeout."""
    tm = TaskManager(max_workers=1)
    gate = threading.Event()
    tm.spawn("blocker", lambda t, p: gate.wait(timeout=5))
    queued = tm.spawn("queued behind it", leaf())

    tm.shutdown(wait=False)
    gate.set()

    started = time.monotonic()
    tm.wait(queued.id, timeout=3)
    assert time.monotonic() - started < 2.5, "wait() hung on a dropped task"
    assert tm.get(queued.id).state in (TaskState.CANCELLED, TaskState.DONE)


# --------------------------------------------------------------------------- #
#  Cancellation takes the subtree with it
# --------------------------------------------------------------------------- #
def test_cancelling_a_parent_cancels_the_whole_subtree(manager):
    started = threading.Event()

    def patient(task, progress):
        started.set()
        for i in range(400):
            progress(f"step {i}", None)
            time.sleep(0.01)
        return {"report": "should never finish"}

    root = manager.spawn("root", spawner(manager, 1, spawner(manager, 1, patient)))

    assert started.wait(timeout=6), "the grandchild never started"
    subtree = manager.descendants(root.id)
    assert len(subtree) == 2

    assert manager.cancel(root.id) is True
    settle(manager, root.id)

    assert manager.get(root.id).state is TaskState.CANCELLED
    for task in subtree:
        assert manager.get(task.id).state is TaskState.CANCELLED, task.goal


def test_cancelling_a_branch_leaves_its_siblings_alone(manager):
    release = threading.Event()

    def waiter(task, progress):
        progress("waiting", 0.1)
        assert release.wait(timeout=8)
        return {"report": "finished normally"}

    root = manager.spawn("root", spawner(manager, 2, waiter))
    deadline = time.monotonic() + 5
    while len(manager.children(root.id)) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    first, second = manager.children(root.id)

    assert manager.cancel(first.id) is True
    release.set()
    settle(manager, root.id)

    assert manager.get(first.id).state is TaskState.CANCELLED
    assert manager.get(second.id).state is TaskState.DONE
    # The parent itself was not cancelled, so it reports its own outcome.
    assert manager.get(root.id).state is TaskState.DONE


# --------------------------------------------------------------------------- #
#  Stats
# --------------------------------------------------------------------------- #
def test_stats_reflect_depth_and_tree_shape(manager, three_levels):
    stats = manager.stats()
    assert stats["total"] == 6 and stats["done"] == 6
    assert stats["tracked"] == 6
    assert stats["roots"] == 1
    assert stats["deepest_depth"] == 2
    assert stats["by_depth"] == {0: 1, 1: 2, 2: 3}
    assert stats["max_depth"] == DEFAULT_MAX_DEPTH
    assert stats["max_total_tasks"] == DEFAULT_MAX_TOTAL_TASKS
    assert stats["refused"] == 0


def test_stats_count_refusals():
    tm = TaskManager(max_workers=1, max_depth=0)
    try:
        root = tm.spawn("root", leaf())
        tm.spawn("nope", leaf(), parent_id=root.id)
        tm.spawn("nope again", leaf(), parent_id=root.id)
        assert tm.stats()["refused"] == 2
        assert tm.stats()["tracked"] == 1
    finally:
        tm.shutdown(wait=False)


# --------------------------------------------------------------------------- #
#  SubAgent tree awareness
# --------------------------------------------------------------------------- #
def test_delegation_note_states_depth_and_remaining_budget():
    permitted = delegation_note(1, 3)
    assert "depth-1" in permitted and "depth 3" in permitted
    assert "2 further level(s)" in permitted

    exhausted = delegation_note(3, 3)
    assert "deepest permitted level" in exhausted
    assert "refused" in exhausted


def test_subagent_prompt_carries_its_depth_and_budget(scripted_llm, fake_registry):
    registry = fake_registry({"read_file": ToolResult.success("x")})
    sub = SubAgent(scripted_llm(["done"]), registry, depth=2, max_depth=3)
    system = sub.build_messages("do a thing", context="user asked nicely")[0].content

    assert "do a thing" in system
    assert "user asked nicely" in system, "the original context was dropped"
    assert "depth-2" in system
    assert "1 further level(s)" in system


def test_subagent_at_max_depth_is_told_not_to_delegate(scripted_llm, fake_registry):
    sub = SubAgent(scripted_llm(["done"]), fake_registry({}), depth=3, max_depth=3)
    system = sub.build_messages("do it yourself")[0].content
    assert "deepest permitted level" in system


def test_subagent_reports_its_place_in_the_tree(scripted_llm, fake_registry):
    registry = fake_registry({})
    tm = TaskManager(max_workers=2)
    try:
        root = tm.spawn("root", leaf())
        sub = SubAgent(scripted_llm(["All clear."]), registry)
        child = tm.spawn("child", sub.run, parent_id=root.id)
        settled = settle(tm, child.id)

        assert settled.result["report"] == "All clear."
        assert settled.result["depth"] == 1
        assert settled.result["parent_id"] == root.id
        assert settled.result["task_id"] == child.id
        assert settled.result["children"] == []
    finally:
        tm.shutdown(wait=False)


def test_subagent_publishes_its_task_on_the_running_thread(scripted_llm, fake_registry):
    """This is how spawn_task attributes a child without trusting the model."""
    seen: dict = {}

    def probe(**kwargs):
        seen.update(current_agent_context())
        return ToolResult.success("noted")

    registry = fake_registry({"probe": probe})
    llm = scripted_llm([format_tool_call("probe", {}), "Done."])

    tm = TaskManager(max_workers=1)
    try:
        task = tm.spawn("look around", SubAgent(llm, registry).run)
        settle(tm, task.id)
    finally:
        tm.shutdown(wait=False)

    assert seen["task_id"] == task.id
    assert seen["depth"] == 0
    # The context is scoped to the run, not leaked to the calling thread.
    assert current_agent_context() == {}


def test_agent_context_restores_the_previous_value():
    with agent_context(task_id="outer", depth=0):
        assert current_agent_context()["task_id"] == "outer"
        with agent_context(task_id="inner", depth=1):
            assert current_agent_context()["task_id"] == "inner"
        assert current_agent_context()["task_id"] == "outer"
    assert current_agent_context() == {}


# --------------------------------------------------------------------------- #
#  Orchestrator: the tree seen from the top
# --------------------------------------------------------------------------- #
@pytest.fixture
def agent(config, scripted_llm):
    """An Orchestrator wired to a scripted model and a registry that really runs."""
    from jarvis.agent.orchestrator import Orchestrator

    class Context:
        def __init__(self):
            self.messages: list = []
            self.store = self

        def add_user(self, text): self.messages.append(("user", text))
        def add_assistant(self, text): self.messages.append(("assistant", text))
        def build(self, text, extra=None): return list(extra or []) + [Message.user(text)]
        def maybe_summarize(self): return None
        def add_text(self, kind, text, **meta): return None
        def remember_fact(self, text, category="fact"): return None
        def search(self, query, k=5): return []

    config.agent.max_concurrent_tasks = 3
    llm = scripted_llm(["Very good, Sir."])
    orchestrator = Orchestrator(
        config, llm, LiveRegistry(), Context(), bus=None, tts=None
    )
    yield orchestrator
    orchestrator.shutdown(wait=False)


def test_orchestrator_reads_the_tree_limits_from_agent_config(config, agent):
    assert agent.max_agent_depth == getattr(
        config.agent, "max_agent_depth", DEFAULT_MAX_DEPTH
    )
    assert agent.tasks.max_depth == agent.max_agent_depth
    assert agent.tasks.max_total_tasks == agent.max_total_tasks


def test_spawn_task_tool_reports_depth_and_remaining_budget(agent):
    payload = agent.registry.run("spawn_task", goal="tidy the desk").output
    assert payload["depth"] == 0
    assert payload["parent_id"] is None
    assert payload["levels_remaining"] == agent.max_agent_depth
    assert payload["tasks_remaining"] == agent.max_total_tasks - payload["tasks_tracked"]

    nested = agent.registry.run(
        "spawn_task", goal="a sub-step", parent_task_id=payload["task_id"]
    ).output
    assert nested["depth"] == 1
    assert nested["parent_id"] == payload["task_id"]
    assert nested["levels_remaining"] == agent.max_agent_depth - 1


def test_spawn_task_tool_surfaces_a_refusal_instead_of_raising(agent):
    root = agent.spawn_task("root")
    current = root.id
    for _ in range(agent.max_agent_depth):
        current = agent.spawn_task("deeper", parent_id=current).id

    refused = agent.registry.run(
        "spawn_task", goal="one level too far", parent_task_id=current
    )
    assert refused.ok, "a refusal must be a result, not a tool failure"
    assert refused.output["refused"] is True
    assert "max_depth=%d" % agent.max_agent_depth in refused.output["error"]
    assert refused.output["state"] == "failed"


def test_a_subagent_spawn_is_attached_to_its_own_task(agent):
    """No explicit parent id, yet the child must not become a stray root."""
    root = agent.tasks.spawn("root", leaf())
    with agent_context(task_id=root.id, depth=0, parent_id=None):
        child = agent.spawn_task("delegated from inside")

    assert child.metadata["parent_id"] == root.id
    assert child.metadata["depth"] == 1
    assert agent.tasks.ancestry(child.id) == [root.id, child.id]
    settle(agent.tasks, root.id)


def test_a_stale_parent_id_does_not_lose_the_task(agent):
    task = agent.spawn_task("still worth doing", parent_id="task_long_gone")
    assert task.state is not TaskState.FAILED
    assert task.metadata["depth"] == 0
    settle(agent.tasks, task.id)


def test_task_tree_tool_renders_the_whole_tree(agent):
    root = agent.tasks.spawn("survey the disk", leaf())
    child = agent.tasks.spawn("count the files", leaf(), parent_id=root.id)
    settle(agent.tasks, root.id)

    rendered = agent.registry.run("task_tree").output
    assert "survey the disk" in rendered
    assert "count the files" in rendered
    assert root.id in rendered and child.id in rendered
    assert "deepest depth 1 of %d" % agent.max_agent_depth in rendered

    branch = agent.registry.run("task_tree", task_id=child.id).output
    assert "count the files" in branch
    assert "survey the disk" not in branch


def test_a_grandchilds_report_reaches_the_roots_pending_updates(agent):
    tm = agent.tasks
    root = tm.spawn(
        "root goal",
        spawner(tm, 1, spawner(tm, 1, leaf("the grandchild found the manifest"))),
    )
    settle(tm, root.id)

    updates = agent.pending_updates()
    joined = "\n".join(updates)
    assert "the grandchild found the manifest" in joined, joined
    # It arrives as a report in its own right, not only nested inside its parent's.
    assert any("depth 2" in u for u in updates), updates
    assert agent.pending_updates() == [], "a report was announced twice"


def test_reports_from_the_tree_are_folded_into_the_next_turn(agent):
    tm = agent.tasks
    root = tm.spawn("root goal", spawner(tm, 1, leaf("child found nothing amiss")))
    settle(tm, root.id)

    agent.llm.script = ["Nothing amiss, Sir."]
    reply = agent.chat("Any news?")

    assert reply == "Nothing amiss, Sir."
    prompt = agent.llm.last_prompt
    assert "child found nothing amiss" in prompt, "the subtree report never reached the model"
    assert "Background task update" in prompt


def test_cancel_task_tool_reports_the_subtree_it_took_with_it(agent):
    tm = agent.tasks
    started = threading.Event()

    def patient(task, progress):
        started.set()
        for i in range(400):
            progress(f"step {i}", None)
            time.sleep(0.01)
        return {"report": "never"}

    root = tm.spawn("root", spawner(tm, 1, patient))
    assert started.wait(timeout=6)

    result = agent.registry.run("cancel_task", task_id=root.id).output
    assert result["cancelled"] is True
    assert len(result["also_cancelled"]) == 1
    settle(tm, root.id)
    assert tm.get(root.id).state is TaskState.CANCELLED


def test_task_status_tool_exposes_the_lineage(agent):
    tm = agent.tasks
    root = tm.spawn("root", spawner(tm, 1, leaf("done and dusted")))
    settle(tm, root.id)
    child = tm.children(root.id)[0]

    status = agent.registry.run("task_status", task_id=child.id).output
    assert status["depth"] == 1
    assert status["parent_id"] == root.id
    assert status["ancestry"] == [root.id, child.id]

    parent_status = agent.registry.run("task_status", task_id=root.id).output
    assert parent_status["children"] == [child.id]
    assert parent_status["child_reports"][0]["report"] == "done and dusted"
