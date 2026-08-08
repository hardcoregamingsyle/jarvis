"""Contracts, configuration, the event bus, and the platform layer."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jarvis.core import platform_utils as pu
from jarvis.core.config import Config, load_config
from jarvis.core.contracts import (
    Message,
    Role,
    Task,
    TaskState,
    ToolParam,
    ToolResult,
    ToolSpec,
    new_id,
)
from jarvis.core.events import EventBus, Events, get_bus, reset_bus


# --------------------------------------------------------------------------- #
#  Contracts
# --------------------------------------------------------------------------- #
def test_message_constructors_and_serialisation():
    assert Message.user("hi").role is Role.USER
    assert Message.system("s").role is Role.SYSTEM
    assert Message.assistant("a").to_dict() == {"role": "assistant", "content": "a"}

    tool = Message.tool("out", name="read_file", tool_call_id="c1")
    assert tool.to_dict() == {
        "role": "tool", "content": "out", "name": "read_file", "tool_call_id": "c1"
    }


def test_ids_are_unique_and_prefixed():
    ids = {new_id("mem") for _ in range(500)}
    assert len(ids) == 500
    assert all(i.startswith("mem_") for i in ids)


def test_tool_spec_json_schema_marks_required_correctly():
    spec = ToolSpec("t", "does a thing", [
        ToolParam("a", "string", "an a"),
        ToolParam("b", "integer", "a b", required=False),
        ToolParam("c", "string", "a c", enum=["x", "y"]),
    ])
    schema = spec.json_schema()
    assert schema["parameters"]["required"] == ["a", "c"]
    assert schema["parameters"]["properties"]["b"]["type"] == "integer"
    assert schema["parameters"]["properties"]["c"]["enum"] == ["x", "y"]


def test_tool_result_helpers():
    assert ToolResult.success("x").ok is True
    assert ToolResult.failure("bad").ok is False
    assert ToolResult.failure("bad").error == "bad"


def test_task_gets_an_id_and_starts_pending():
    task = Task(id="", goal="do a thing")
    assert task.id.startswith("task_")
    assert task.state is TaskState.PENDING


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
def test_defaults_are_the_documented_ones():
    cfg = load_config(use_env=False)
    assert cfg.llm.model.startswith("Qwen/Qwen3-")
    assert cfg.tts.edge_voice == "en-GB-RyanNeural"
    assert cfg.tts.piper_voice.startswith("en_GB")
    # Ships unrestricted: no prompts, no protected paths, no blocked commands.
    assert cfg.security.mode == "open"
    assert cfg.security.protected_paths == []
    assert cfg.security.dangerous_patterns == []
    assert cfg.memory.prune is False


def test_environment_overrides_are_typed():
    cfg = load_config(environ={
        "JARVIS_LLM_MAX_NEW_TOKENS": "128",
        "JARVIS_LLM_TEMPERATURE": "0.25",
        "JARVIS_TTS_ENABLED": "false",
        "JARVIS_LOG_LEVEL": "DEBUG",
    })
    assert cfg.llm.max_new_tokens == 128 and isinstance(cfg.llm.max_new_tokens, int)
    assert cfg.llm.temperature == pytest.approx(0.25)
    assert cfg.tts.enabled is False
    assert cfg.log_level == "DEBUG"


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("", False),
])
def test_boolean_coercion(raw, expected):
    cfg = load_config(environ={"JARVIS_TTS_ENABLED": raw})
    assert cfg.tts.enabled is expected


def test_unknown_environment_variables_are_ignored():
    cfg = load_config(environ={"JARVIS_NOT_A_SECTION_AT_ALL": "x"})
    assert isinstance(cfg, Config)


def test_config_file_round_trip(tmp_path):
    cfg = load_config(use_env=False)
    cfg.llm.model = "Qwen/Qwen3-8B"
    cfg.security.protected_paths = ["/nowhere"]
    cfg.agent.max_concurrent_tasks = 9

    path = cfg.save(tmp_path / "cfg.json")
    reloaded = load_config(path, use_env=False)

    assert reloaded.llm.model == "Qwen/Qwen3-8B"
    assert reloaded.security.protected_paths == ["/nowhere"]
    assert reloaded.agent.max_concurrent_tasks == 9


def test_partial_config_file_keeps_other_defaults(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"llm": {"model": "custom"}}), encoding="utf-8")

    cfg = load_config(path, use_env=False)
    assert cfg.llm.model == "custom"
    assert cfg.llm.temperature == 0.7          # untouched default
    assert cfg.tts.edge_voice == "en-GB-RyanNeural"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_malformed_config_file_raises_a_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, use_env=False)


def test_derived_paths_live_under_the_home(isolated_home):
    cfg = load_config(use_env=False)
    for path in (cfg.db_file(), cfg.tools_dir(), cfg.voices_dir(),
                 cfg.models_dir(), cfg.logs_dir()):
        assert str(path).startswith(str(isolated_home))
    assert cfg.tools_dir().is_dir()


def test_to_dict_is_json_serialisable():
    payload = load_config(use_env=False).to_dict()
    json.dumps(payload)          # must not raise
    assert payload["security"]["mode"] == "open"
    assert isinstance(payload["security"]["protected_paths"], list)


# --------------------------------------------------------------------------- #
#  Event bus
# --------------------------------------------------------------------------- #
def test_subscribe_emit_unsubscribe():
    bus = EventBus()
    seen = []
    unsubscribe = bus.subscribe(Events.SPEAK, seen.append)

    bus.emit(Events.SPEAK, "one")
    unsubscribe()
    bus.emit(Events.SPEAK, "two")

    assert seen == ["one"]


def test_a_failing_handler_does_not_stop_the_others():
    bus = EventBus()
    seen = []

    def explode(_payload):
        raise RuntimeError("bad handler")

    bus.subscribe(Events.SPEAK, explode)
    bus.subscribe(Events.SPEAK, seen.append)
    bus.emit(Events.SPEAK, "still delivered")

    assert seen == ["still delivered"]


def test_decorator_form():
    bus = EventBus()
    seen = []

    @bus.on(Events.WAKE)
    def _handler(payload):
        seen.append(payload)

    bus.emit(Events.WAKE, "jarvis")
    assert seen == ["jarvis"]


def test_emit_to_a_channel_with_no_handlers_is_a_no_op():
    EventBus().emit("nobody.listening", 123)


def test_clear_removes_handlers():
    bus = EventBus()
    seen = []
    bus.subscribe(Events.SPEAK, seen.append)
    bus.subscribe(Events.WAKE, seen.append)

    bus.clear(Events.SPEAK)
    bus.emit(Events.SPEAK, "gone")
    bus.emit(Events.WAKE, "kept")
    assert seen == ["kept"]

    bus.clear()
    bus.emit(Events.WAKE, "also gone")
    assert seen == ["kept"]


def test_async_handlers_are_awaited_by_emit_async():
    import asyncio

    bus = EventBus()
    seen = []

    async def handler(payload):
        await asyncio.sleep(0)
        seen.append(payload)

    bus.subscribe(Events.SPEAK, handler)
    asyncio.run(bus.emit_async(Events.SPEAK, "async"))
    assert seen == ["async"]


def test_emit_is_thread_safe():
    bus = EventBus()
    seen = []
    lock = threading.Lock()

    def handler(payload):
        with lock:
            seen.append(payload)

    bus.subscribe("chan", handler)
    threads = [
        threading.Thread(target=lambda i=i: bus.emit("chan", i)) for i in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 50


def test_default_bus_is_a_singleton_until_reset():
    reset_bus()
    try:
        assert get_bus() is get_bus()
        first = get_bus()
        reset_bus()
        assert get_bus() is not first
    finally:
        reset_bus()


# --------------------------------------------------------------------------- #
#  Platform layer
# --------------------------------------------------------------------------- #
def test_os_name_is_recognised():
    assert pu.os_name() in ("windows", "linux", "macos")
    assert sum([pu.IS_WINDOWS, pu.IS_LINUX, pu.IS_MAC]) <= 1


def test_run_command_captures_stdout():
    result = pu.run_command("echo hello-jarvis")
    assert result.ok
    assert "hello-jarvis" in result.stdout


def test_run_command_reports_a_non_zero_exit():
    result = pu.run_command("exit 3")
    assert result.returncode == 3
    assert result.ok is False


def test_run_command_times_out_without_raising():
    sleeper = "Start-Sleep -Seconds 30" if pu.IS_WINDOWS else "sleep 30"
    result = pu.run_command(sleeper, timeout=0.5)
    assert result.timed_out is True
    assert result.ok is False


def test_run_command_with_an_argv_list():
    import sys as _sys
    result = pu.run_command([_sys.executable, "-c", "print('argv-mode')"])
    assert result.ok and "argv-mode" in result.stdout


def test_run_command_on_a_missing_binary_does_not_raise():
    result = pu.run_command(["definitely-not-a-real-binary-xyz"])
    assert result.ok is False


def test_run_command_honours_cwd_and_env(tmp_path):
    import sys as _sys
    result = pu.run_command(
        [_sys.executable, "-c", "import os;print(os.getcwd());print(os.environ.get('JV_TEST'))"],
        cwd=str(tmp_path), env={"JV_TEST": "marker"},
    )
    assert "marker" in result.stdout
    assert str(tmp_path.resolve()).lower() in result.stdout.lower()


def test_which_finds_the_interpreter():
    assert pu.which("python") or pu.which("python3") or pu.which("py")


def test_directories_are_absolute_and_creatable(isolated_home):
    for directory in (pu.data_dir(), pu.config_dir(), pu.cache_dir()):
        assert Path(directory).is_absolute()
    made = pu.ensure_dir(Path(isolated_home) / "a" / "b" / "c")
    assert made.is_dir()
    pu.ensure_dir(made)          # idempotent


def test_system_summary_has_the_expected_keys():
    summary = pu.system_summary()
    for key in ("os", "platform", "machine", "python", "cpu", "hostname"):
        assert summary.get(key)


def test_data_dir_honours_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "custom"))
    assert pu.data_dir() == tmp_path / "custom"


# --------------------------------------------------------------------------- #
#  Path safety — the helpers that exist because a bare "C:" once wiped the repo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["C:", "c:", "D:", "C:foo", "Z:relative\\path"])
def test_drive_relative_paths_are_detected(raw):
    assert pu.is_drive_relative(raw) is True


@pytest.mark.parametrize("raw", [
    "C:\\", "C:/", "C:\\Users", "C:/Users", "/etc", "relative/path",
    "", "\\\\server\\share", "file.txt",
])
def test_ordinary_paths_are_not_drive_relative(raw):
    assert pu.is_drive_relative(raw) is False


def test_resolve_path_maps_drive_relative_to_the_drive_root(tmp_path, monkeypatch):
    """"C:" means the drive root, never the working directory.

    Python resolves a bare drive letter against the process CWD. That silent
    behaviour destroyed this project's source tree once, so the helper maps it to
    the root the caller obviously meant.
    """
    monkeypatch.chdir(tmp_path)
    if not pu.IS_WINDOWS:
        pytest.skip("drive-relative paths are a Windows concept")

    resolved = pu.resolve_path("C:")
    assert str(resolved) in ("C:\\", "C:/")
    assert resolved != Path.cwd(), "resolved to the working directory — the old bug"
    assert Path("C:").resolve() == Path.cwd(), "the underlying trap should still be real"


def test_resolve_path_strict_still_rejects_drive_relative():
    """Callers that would rather refuse than guess can still opt in."""
    with pytest.raises(ValueError) as excinfo:
        pu.resolve_path("C:", strict=True)
    assert "drive-relative" in str(excinfo.value)


def test_resolve_path_expands_and_resolves(tmp_path):
    nested = tmp_path / "a" / ".." / "b"
    resolved = pu.resolve_path(str(nested))
    assert resolved == (tmp_path / "b").resolve()
    assert resolved.is_absolute()


def test_filesystem_root_detection(tmp_path):
    root = Path(tmp_path.anchor or "/")
    assert pu.is_filesystem_root(root) is True
    assert pu.is_filesystem_root(tmp_path) is False


def test_a_bare_drive_letter_resolves_to_the_cwd_not_the_root(tmp_path, monkeypatch):
    """The exact trap the helpers guard against, pinned as a regression."""
    if not pu.IS_WINDOWS:
        pytest.skip("drive-relative paths are a Windows concept")

    monkeypatch.chdir(tmp_path)
    drive = Path.cwd().drive          # e.g. "C:"
    assert Path(drive).resolve() == Path.cwd()
    assert Path(drive + "\\").resolve() != Path.cwd()
    assert pu.is_drive_relative(drive) is True
