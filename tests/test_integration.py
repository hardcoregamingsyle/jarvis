"""End-to-end: boot the whole system and drive a real conversation.

These are the tests that catch a subsystem drifting out of contract with its
neighbours — the failure mode that unit tests, by construction, cannot see.
Everything runs against a scripted LLM and a temporary home, so the suite stays
hermetic: no models, no network, no audio device, and nothing written outside
``tmp_path``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from jarvis import app as app_module
from jarvis.agent.protocol import format_tool_call
from jarvis.core.contracts import TaskState
from jarvis.core.events import Events


@pytest.fixture
def booted(config, scripted_llm):
    """A fully wired JARVIS whose language model is scriptable.

    The real boot path runs first (so the wiring under test is the production
    wiring), then the stub LLM is swapped for a scripted one.  Tests drive
    behaviour with ``booted.llm.script = [...]``.
    """
    config.llm.backend = "stub"
    config.tts.engine = "null"
    config.stt.engine = "stub"
    subsystems = app_module.build(config, configure_logging=False)

    llm = scripted_llm()
    subsystems.llm = llm
    if subsystems.orchestrator is not None:
        subsystems.orchestrator.llm = llm
    if subsystems.registry is not None:
        subsystems.registry.ctx.extra["llm"] = llm

    yield subsystems
    app_module.shutdown(subsystems)


# --------------------------------------------------------------------------- #
#  Boot
# --------------------------------------------------------------------------- #
def test_everything_comes_up(booted):
    assert booted.orchestrator is not None, "orchestrator failed to build"
    assert booted.llm is not None
    assert booted.memory is not None
    assert booted.registry is not None

    status = booted.status()
    assert status["tools"] > 10, f"only {status['tools']} tools registered"
    assert status["memory"].startswith("sqlite")


def test_boot_creates_its_directories(booted, isolated_home):
    cfg = booted.config
    for directory in (cfg.tools_dir(), cfg.voices_dir(), cfg.logs_dir()):
        assert directory.is_dir()
    assert str(cfg.db_file()).startswith(str(isolated_home))


def test_the_expected_tools_are_registered(booted):
    names = set(booted.registry.names())
    expected = {
        "read_file", "write_file", "list_dir", "find_files",
        "run_command", "system_info",
        "list_processes",
        "spawn_task", "list_tasks", "task_status", "cancel_task",
        "remember", "recall",
        "create_tool",
    }
    missing = expected - names
    assert not missing, f"missing tools: {sorted(missing)}"


def test_every_tool_spec_is_well_formed(booted):
    for spec in booted.registry.list():
        assert spec.name and spec.description, f"{spec.name} is under-described"
        schema = spec.json_schema()
        assert schema["parameters"]["type"] == "object"
        json.dumps(schema)      # must be serialisable into a prompt


def test_shutdown_is_idempotent(config):
    config.llm.backend = "stub"
    subsystems = app_module.build(config, configure_logging=False)
    app_module.shutdown(subsystems)
    app_module.shutdown(subsystems)


# --------------------------------------------------------------------------- #
#  Conversation
# --------------------------------------------------------------------------- #
def test_a_plain_turn_produces_a_reply_and_is_remembered(booted):
    agent = booted.orchestrator
    reply = agent.chat("Good morning.")

    assert reply and isinstance(reply, str)
    history = agent.context.history()
    assert any(m.content == "Good morning." for m in history)

    stored = list(booted.memory.all(kind="conversation"))
    assert any("Good morning." in r.text for r in stored)


def test_the_system_prompt_carries_persona_and_tools(booted):
    prompt = booted.orchestrator.system_prompt()
    assert "British" in prompt
    assert "Sir" in prompt
    assert "read_file" in prompt
    assert "<tool_call>" in prompt


def test_a_tool_call_round_trip(booted, tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("the answer is 42", encoding="utf-8")

    agent = booted.orchestrator
    booted.llm.script = [
        format_tool_call("read_file", {"path": str(target)}),
        "The note says the answer is forty-two, Sir.",
    ]

    reply = agent.chat("What does that note say?")
    assert reply == "The note says the answer is forty-two, Sir."

    # The tool genuinely ran and its output reached the model.
    assert any(
        "the answer is 42" in m.content
        for m in booted.llm.calls[-1]
        if m.role.value == "tool"
    ), "the file contents never made it into the prompt"


def test_memory_survives_a_restart(config):
    config.llm.backend = "stub"
    first = app_module.build(config, configure_logging=False)
    try:
        first.orchestrator.context.remember_fact("The user prefers Earl Grey tea.")
    finally:
        app_module.shutdown(first)

    second = app_module.build(config, configure_logging=False)
    try:
        hits = second.memory.search("what tea does the user like", k=5)
        assert any("Earl Grey" in h.text for h in hits), [h.text for h in hits]
    finally:
        app_module.shutdown(second)


def test_recall_reaches_into_the_prompt(booted):
    agent = booted.orchestrator
    agent.context.remember_fact("The garage door code is 4417.")

    messages = agent.context.build("what is the garage code?")
    combined = "\n".join(m.content for m in messages)
    assert "4417" in combined, "long-term memory was not recalled into the prompt"


# --------------------------------------------------------------------------- #
#  Delegation
# --------------------------------------------------------------------------- #
def test_spawned_task_runs_and_reports_back(booted):
    agent = booted.orchestrator
    booted.llm.script = ["Subagent report: the scan found nothing untoward."]

    task = agent.spawn_task("scan the downloads folder")
    settled = agent.tasks.wait(task.id, timeout=30)

    assert settled.state is TaskState.DONE, settled.error
    assert "nothing untoward" in settled.result["report"]

    updates = agent.pending_updates()
    assert any("nothing untoward" in u for u in updates)
    assert agent.pending_updates() == [], "a report was announced twice"


def test_the_main_agent_stays_responsive_while_a_task_runs(booted):
    """The whole point of the subagent design."""
    agent = booted.orchestrator
    released = threading.Event()

    def slow_runner(task, progress):
        progress("working", 0.1)
        released.wait(timeout=10)
        return {"report": "eventually done"}

    task = agent.tasks.spawn("something slow", slow_runner)

    started = time.monotonic()
    reply = agent.chat("Are you still there?")
    elapsed = time.monotonic() - started

    assert reply, "the main agent was blocked by the background task"
    assert elapsed < 5.0, f"main agent took {elapsed:.1f}s while a task was running"
    assert agent.tasks.get(task.id).state is TaskState.RUNNING

    released.set()
    agent.tasks.wait(task.id, timeout=15)


def test_spawn_task_tool_is_callable_by_the_model(booted):
    result = booted.registry.run("spawn_task", goal="tidy the desktop")
    assert result.ok, result.error
    assert result.output["state"] in ("pending", "running", "done")

    listed = booted.registry.run("list_tasks")
    assert listed.ok
    assert any(t["goal"] == "tidy the desktop" for t in listed.output)


def test_task_status_and_cancel_tools(booted):
    booted.llm.script = ["done"]
    spawned = booted.registry.run("spawn_task", goal="a goal")
    task_id = spawned.output["task_id"]

    status = booted.registry.run("task_status", task_id=task_id)
    assert status.ok and status.output["goal"] == "a goal"

    assert booted.registry.run("task_status", task_id="nope").output.get("error")
    assert booted.registry.run("cancel_task", task_id=task_id).ok


# --------------------------------------------------------------------------- #
#  Memory tools
# --------------------------------------------------------------------------- #
def test_remember_and_recall_tools(booted):
    stored = booted.registry.run("remember", text="The wifi password is hunter2.")
    assert stored.ok

    found = booted.registry.run("recall", query="wifi password")
    assert found.ok
    assert any("hunter2" in hit["text"] for hit in found.output)


# --------------------------------------------------------------------------- #
#  Machine access
# --------------------------------------------------------------------------- #
def test_file_tools_operate_on_the_real_filesystem(booted, tmp_path):
    path = tmp_path / "sub" / "hello.txt"

    assert booted.registry.run("make_dir", path=str(tmp_path / "sub")).ok
    assert booted.registry.run("write_file", path=str(path), content="hello world").ok
    assert path.read_text(encoding="utf-8") == "hello world"

    read = booted.registry.run("read_file", path=str(path))
    assert read.ok and "hello world" in str(read.output)

    listing = booted.registry.run("list_dir", path=str(tmp_path / "sub"))
    assert listing.ok
    assert any("hello.txt" in str(entry) for entry in listing.output["entries"])

    found = booted.registry.run("find_files", root=str(tmp_path), pattern="*.txt")
    assert found.ok
    assert any("hello.txt" in match for match in found.output["matches"])


def test_system_info_works_without_optional_packages(booted):
    result = booted.registry.run("system_info")
    assert result.ok, result.error
    info = result.output
    assert info.get("os") in ("windows", "linux", "macos")
    assert info.get("python")


def test_run_command_executes(booted):
    result = booted.registry.run("run_command", command="echo integration-check")
    assert result.ok, result.error
    assert "integration-check" in str(result.output)


def test_unknown_tool_fails_cleanly(booted):
    result = booted.registry.run("no_such_tool")
    assert result.ok is False
    assert "no_such_tool" in (result.error or "")


def test_missing_required_argument_is_rejected(booted):
    result = booted.registry.run("read_file")
    assert result.ok is False
    assert result.error


# --------------------------------------------------------------------------- #
#  Security
# --------------------------------------------------------------------------- #
def test_path_protection_is_opt_in(config, tmp_path):
    """Nothing is protected by default; naming a path re-protects it.

    The write target is inside tmp_path, so if protection regresses this test
    fails on scratch space rather than writing into a real system directory.
    """
    sanctuary = tmp_path / "sanctuary"
    sanctuary.mkdir()
    target = str(sanctuary / "should_not_appear.txt")

    config.llm.backend = "stub"
    config.security.mode = "guarded"
    config.security.protected_paths = [str(sanctuary)]
    subsystems = app_module.build(config, configure_logging=False)
    try:
        result = subsystems.registry.run("write_file", path=target, content="nope")
        assert result.ok is False, "an explicitly protected path was written to"
        assert not Path(target).exists()
    finally:
        app_module.shutdown(subsystems)


def test_nothing_is_protected_out_of_the_box(config, tmp_path):
    """The shipped config has the rails off: writes go wherever they are aimed."""
    config.llm.backend = "stub"
    subsystems = app_module.build(config, configure_logging=False)
    try:
        assert subsystems.config.security.mode == "open"
        assert subsystems.config.security.protected_paths == []

        target = tmp_path / "anywhere.txt"
        result = subsystems.registry.run("write_file", path=str(target), content="ok")
        assert result.ok, result.error
        assert target.read_text(encoding="utf-8") == "ok"
    finally:
        app_module.shutdown(subsystems)


def test_readonly_mode_blocks_writes(config, tmp_path):
    config.llm.backend = "stub"
    config.security.mode = "readonly"
    subsystems = app_module.build(config, configure_logging=False)
    try:
        target = tmp_path / "readonly_probe.txt"
        result = subsystems.registry.run("write_file", path=str(target), content="x")
        assert result.ok is False
        assert not target.exists()
    finally:
        app_module.shutdown(subsystems)


def test_delete_path_refuses_a_drive_relative_target(booted, tmp_path, monkeypatch):
    """"C:" resolves to the working directory, not the drive root.

    This is the bug that once deleted this project's entire source tree, so it
    is pinned end-to-end through the registry, not just at the helper.
    """
    monkeypatch.chdir(tmp_path)
    canary = tmp_path / "canary.txt"
    canary.write_text("still here", encoding="utf-8")

    for target in ("C:", "c:"):
        result = booted.registry.run("delete_path", path=target, recursive=True)
        assert result.ok is False, f"delete_path accepted {target!r}"

    assert canary.exists(), "the working directory was modified"
    assert canary.read_text(encoding="utf-8") == "still here"


def test_delete_path_refuses_roots_and_home(booted, tmp_path, monkeypatch):
    """Redirected targets — the real root and home are never named.

    An earlier version of this test passed the actual ``Path.home()`` to a
    recursive delete and relied on the guard to refuse it. When the guard was
    later relaxed, that stopped being a test and became a live rmtree of the
    user profile. Aim these at stand-ins so a regression fails loudly on scratch
    space instead of succeeding quietly on someone's documents.
    """
    from jarvis.tools import file_tools

    fake_root = tmp_path / "pretend_root"
    fake_home = tmp_path / "pretend_home"
    for directory in (fake_root, fake_home):
        directory.mkdir()
        (directory / "canary.txt").write_text("irreplaceable", encoding="utf-8")

    monkeypatch.setattr(
        file_tools, "is_filesystem_root", lambda p: Path(p) == fake_root
    )
    monkeypatch.setattr(
        file_tools.os.path, "expanduser",
        lambda p: str(fake_home) if p == "~" else p,
    )

    assert booted.registry.run(
        "delete_path", path=str(fake_root), recursive=True
    ).ok is False
    assert booted.registry.run(
        "delete_path", path=str(fake_home), recursive=True
    ).ok is False

    for directory in (fake_root, fake_home):
        assert (directory / "canary.txt").exists(), f"{directory.name} was deleted"


def test_delete_path_still_works_for_ordinary_files(booted, tmp_path):
    victim = tmp_path / "disposable.txt"
    victim.write_text("bye", encoding="utf-8")
    result = booted.registry.run("delete_path", path=str(victim))
    assert result.ok, result.error
    assert not victim.exists()


def test_the_audit_log_records_actions(booted):
    booted.registry.run("run_command", command="echo audited")
    audit = booted.config.logs_dir() / "audit.jsonl"
    if audit.exists():
        lines = [
            json.loads(line)
            for line in audit.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert lines, "audit log is empty"
        assert all("ts" in entry and "allowed" in entry for entry in lines)


# --------------------------------------------------------------------------- #
#  Self-extension
# --------------------------------------------------------------------------- #
def test_jarvis_can_write_and_then_use_a_new_tool(booted):
    """The 'if a tool it needs is missing, it makes it' requirement."""
    source = '''
from __future__ import annotations

from jarvis.core.contracts import ToolResult
from jarvis.tools.registry import FunctionTool


def celsius_to_fahrenheit(celsius: float) -> ToolResult:
    """Convert a temperature from Celsius to Fahrenheit."""
    try:
        return ToolResult.success({"fahrenheit": float(celsius) * 9.0 / 5.0 + 32.0})
    except (TypeError, ValueError) as exc:
        return ToolResult.failure(str(exc))


def build_tools(ctx):
    return [FunctionTool(celsius_to_fahrenheit)]
'''
    from jarvis.tools.tool_maker import make_tool

    ctx = booted.registry.ctx
    result = make_tool(ctx, "temperature", "Temperature conversions",
                       "convert celsius to fahrenheit", source=source)
    assert result.ok, result.error

    written = booted.config.tools_dir() / "temperature.py"
    assert written.exists()

    loaded = booted.registry.load_generated()
    assert loaded >= 1
    assert booted.registry.has("celsius_to_fahrenheit")

    call = booted.registry.run("celsius_to_fahrenheit", celsius=100)
    assert call.ok, call.error
    assert call.output["fahrenheit"] == pytest.approx(212.0)


def test_a_dangerous_generated_tool_is_rejected(booted):
    from jarvis.tools.tool_maker import make_tool

    evil = '''
import os

def build_tools(ctx):
    os.system("echo pwned")
    return []
'''
    result = make_tool(booted.registry.ctx, "evil", "bad", "do harm", source=evil)
    assert result.ok is False
    assert not (booted.config.tools_dir() / "evil.py").exists(), \
        "a rejected tool was left on disk"


# --------------------------------------------------------------------------- #
#  Events
# --------------------------------------------------------------------------- #
def test_the_bus_reports_a_conversation(booted):
    seen = {"user": [], "reply": [], "tool": []}
    booted.bus.subscribe(Events.USER_UTTERANCE, seen["user"].append)
    booted.bus.subscribe(Events.ASSISTANT_REPLY, seen["reply"].append)
    booted.bus.subscribe(Events.TOOL_RESULT, seen["tool"].append)

    booted.llm.script = [
        format_tool_call("system_info", {}),
        "All nominal, Sir.",
    ]
    booted.orchestrator.chat("status report")

    assert seen["user"] == ["status report"]
    assert seen["reply"] == ["All nominal, Sir."]
    assert any(e["name"] == "system_info" for e in seen["tool"])
