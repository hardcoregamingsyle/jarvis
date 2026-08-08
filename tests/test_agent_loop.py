"""The reason-act loop and the subagent that runs it."""

from __future__ import annotations

import pytest

from jarvis.agent.protocol import format_tool_call
from jarvis.agent.subagent import SubAgent, run_agent_loop
from jarvis.agent.task_manager import TaskManager
from jarvis.core.contracts import Message, TaskState, ToolResult


@pytest.fixture
def registry(fake_registry):
    return fake_registry({
        "read_file": ToolResult.success("file contents here"),
        "broken": ToolResult.failure("boom"),
    })


def test_answer_without_tools_completes_in_one_iteration(scripted_llm, registry):
    llm = scripted_llm(["The weather is fine, Sir."])
    turn = run_agent_loop(llm, registry, [Message.user("hello")], max_iterations=5)

    assert turn.text == "The weather is fine, Sir."
    assert turn.iterations == 1
    assert turn.used_tools == []
    assert turn.truncated is False


def test_tool_call_then_answer(scripted_llm, registry):
    llm = scripted_llm([
        format_tool_call("read_file", {"path": "a.txt"}),
        "The file says hello, Sir.",
    ])
    messages = [Message.system("s"), Message.user("read a.txt")]
    turn = run_agent_loop(llm, registry, messages, max_iterations=5)

    assert turn.text == "The file says hello, Sir."
    assert turn.used_tools == ["read_file"]
    assert registry.calls == [("read_file", {"path": "a.txt"})]

    tool_turns = [m for m in messages if m.role.value == "tool"]
    assert len(tool_turns) == 1
    assert "file contents here" in tool_turns[0].content
    assert messages[-1].role.value == "assistant"


def test_several_tool_calls_in_one_response_all_run(scripted_llm, registry):
    llm = scripted_llm([
        format_tool_call("read_file", {"path": "a"}) + format_tool_call("read_file", {"path": "b"}),
        "Both read, Sir.",
    ])
    turn = run_agent_loop(llm, registry, [Message.user("go")], max_iterations=5)
    assert [c.arguments["path"] for c in turn.tool_calls] == ["a", "b"]
    assert turn.text == "Both read, Sir."


def test_unknown_tool_is_reported_back_to_the_model(scripted_llm, registry):
    llm = scripted_llm([format_tool_call("nonexistent", {}), "Apologies, Sir."])
    messages = [Message.user("x")]
    turn = run_agent_loop(llm, registry, messages, max_iterations=5)

    feedback = [m.content for m in messages if m.role.value == "tool"][0]
    assert "no tool named 'nonexistent'" in feedback
    assert turn.text == "Apologies, Sir."


def test_unknown_tool_suggests_a_near_match(scripted_llm, registry):
    llm = scripted_llm([format_tool_call("read", {}), "Right."])
    messages = [Message.user("x")]
    run_agent_loop(llm, registry, messages, max_iterations=3)
    feedback = [m.content for m in messages if m.role.value == "tool"][0]
    assert "read_file" in feedback


def test_failing_tool_result_is_passed_through(scripted_llm, registry):
    llm = scripted_llm([format_tool_call("broken", {}), "That did not work, Sir."])
    messages = [Message.user("x")]
    run_agent_loop(llm, registry, messages, max_iterations=5)
    assert "ERROR: boom" in [m.content for m in messages if m.role.value == "tool"][0]


def test_registry_raising_is_contained(scripted_llm):
    class Raising:
        def names(self): return ["read_file"]
        def has(self, n): return True
        def describe(self): return ""
        def run(self, name, **kw): raise OSError("disk gone")

    llm = scripted_llm([format_tool_call("read_file", {"path": "a"}), "Recovered, Sir."])
    messages = [Message.user("x")]
    turn = run_agent_loop(llm, Raising(), messages, max_iterations=4)

    assert turn.text == "Recovered, Sir."
    assert "disk gone" in [m.content for m in messages if m.role.value == "tool"][0]


def test_llm_failure_becomes_a_spoken_apology(registry):
    class Exploding:
        name = "boom"
        def is_available(self): return True
        def load(self): pass
        def generate(self, messages, config=None): raise RuntimeError("gpu on fire")
        def stream(self, messages, config=None): raise RuntimeError("gpu on fire")

    turn = run_agent_loop(Exploding(), registry, [Message.user("x")], max_iterations=3)
    assert "gpu on fire" in turn.text
    assert turn.iterations == 1


def test_iteration_limit_forces_a_final_answer(scripted_llm, registry):
    llm = scripted_llm(
        [format_tool_call("read_file", {"path": "a"})] * 3 + ["Forced final."]
    )
    turn = run_agent_loop(llm, registry, [Message.user("loop")], max_iterations=3)
    assert turn.truncated is True
    assert turn.iterations == 3
    assert turn.text == "Forced final."


def test_never_returns_silence_when_the_model_keeps_calling_tools(scripted_llm, registry):
    """A voice assistant returning "" is the worst possible failure."""
    llm = scripted_llm([format_tool_call("read_file", {"path": "a"})] * 20)
    turn = run_agent_loop(llm, registry, [Message.user("loop")], max_iterations=3)

    assert turn.truncated is True
    assert turn.text.strip(), "the loop produced an empty reply"
    assert "read_file" in turn.text


def test_never_returns_silence_for_a_blank_model_response(scripted_llm, registry):
    turn = run_agent_loop(scripted_llm(["   \n\t "]), registry, [Message.user("hi")], max_iterations=2)
    assert turn.text.strip()


def test_no_successful_tools_produces_an_honest_failure_message(scripted_llm, registry):
    llm = scripted_llm([format_tool_call("broken", {})] * 10)
    turn = run_agent_loop(llm, registry, [Message.user("x")], max_iterations=2)
    assert "failed" in turn.text.lower()


def test_streaming_emits_chunks(scripted_llm, registry):
    chunks: list = []
    turn = run_agent_loop(
        scripted_llm(["Streamed reply."]), registry, [Message.user("hi")],
        max_iterations=2, stream=True, on_chunk=chunks.append,
    )
    assert turn.text == "Streamed reply."
    assert chunks == ["Streamed reply."]


def test_progress_callback_receives_updates(scripted_llm, registry):
    seen: list = []
    run_agent_loop(
        scripted_llm([format_tool_call("read_file", {"path": "a"}), "Done."]),
        registry, [Message.user("x")], max_iterations=3,
        progress=lambda msg, frac=None: seen.append(msg),
    )
    assert any("thinking" in s for s in seen)
    assert any("read_file" in s for s in seen)


def test_registry_none_means_no_tools(scripted_llm):
    llm = scripted_llm([format_tool_call("anything", {}), "No tools available, Sir."])
    messages = [Message.user("x")]
    turn = run_agent_loop(llm, None, messages, max_iterations=3)
    assert "no tool named" in [m.content for m in messages if m.role.value == "tool"][0]
    assert turn.text == "No tools available, Sir."


# --------------------------------------------------------------------------- #
#  SubAgent
# --------------------------------------------------------------------------- #
def test_subagent_prompt_contains_goal_and_tools(scripted_llm, registry):
    sub = SubAgent(scripted_llm(["done"]), registry)
    messages = sub.build_messages("index the photos folder", context="user is tidying up")
    system = messages[0].content
    assert "index the photos folder" in system
    assert "read_file" in system
    assert "user is tidying up" in system


def test_subagent_runs_through_the_task_manager(scripted_llm, registry):
    llm = scripted_llm([format_tool_call("read_file", {"path": "z"}), "Report: all clear."])
    sub = SubAgent(llm, registry)

    tm = TaskManager(max_workers=2)
    try:
        task = tm.spawn("check z", sub.run)
        tm.wait(task.id, timeout=20)
        settled = tm.get(task.id)

        assert settled.state is TaskState.DONE, settled.error
        assert settled.result["report"] == "Report: all clear."
        assert settled.result["tools_used"] == ["read_file"]
        assert settled.result["truncated"] is False
    finally:
        tm.shutdown(wait=True)


def test_subagent_failure_surfaces_in_the_report(registry):
    class Exploding:
        name = "boom"
        def is_available(self): return True
        def load(self): pass
        def generate(self, m, c=None): raise RuntimeError("model gone")

    tm = TaskManager(max_workers=1)
    try:
        task = tm.spawn("doomed", SubAgent(Exploding(), registry).run)
        tm.wait(task.id, timeout=20)
        settled = tm.get(task.id)
        # The loop catches generation errors, so the task completes with an
        # apologetic report rather than failing outright.
        assert settled.state is TaskState.DONE
        assert "model gone" in settled.result["report"]
    finally:
        tm.shutdown(wait=True)
