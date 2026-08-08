"""The policy in practice: permissive by default, restrictive on request.

The shipped configuration lets everything through — no prompts, no protected
paths, no refusals.  The opt-in modes must still genuinely *work*, which means
more than classifying correctly: the registry has to honour a declined
confirmation and actually not run the tool.

Nothing here touches anything outside ``tmp_path``.  The "dangerous" tools
record that they were called instead of doing anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jarvis.core import platform_utils as pu
from jarvis.core.config import SecurityConfig
from jarvis.core.contracts import ToolParam, ToolResult, ToolSpec
from jarvis.core.platform_utils import IS_WINDOWS
from jarvis.core.security import SecurityGate
from jarvis.tools import file_tools
from jarvis.tools.file_tools import build_tools
from jarvis.tools.registry import FunctionTool, ToolContext, ToolRegistry


ORDINARY = ToolSpec("write_file", "w",
                    [ToolParam("path"), ToolParam("content")], dangerous=True)
DESTRUCTIVE_CMD = ToolSpec("run_command", "c", [ToolParam("command")], dangerous=True)
HARMLESS = ToolSpec("read_file", "r", [ToolParam("path")])


def gate(mode=None, confirm=None, **overrides):
    """A gate over a real SecurityConfig; ``mode=None`` keeps the shipped default."""
    cfg = SecurityConfig(audit_log=False)
    if mode is not None:
        cfg.mode = mode
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return SecurityGate(cfg, confirm=confirm)


def restricted_gate(confirm=None, mode="guarded", **overrides):
    """What a user who wants the old behaviour back would configure."""
    return gate(
        mode=mode,
        confirm=confirm,
        protected_paths=(["C:\\Windows"] if IS_WINDOWS else ["/etc"]),
        dangerous_patterns=["format", "rm -rf /", "shutdown", "diskpart"],
        **overrides,
    )


def allows(g, spec, **kwargs) -> bool:
    return g.allows(g.check_tool(spec, kwargs))


# --------------------------------------------------------------------------- #
#  Out of the box: nothing is blocked, nothing prompts
# --------------------------------------------------------------------------- #
def test_reads_never_need_confirmation(tmp_path):
    for mode in ("open", "guarded", "readonly"):
        assert allows(gate(mode), HARMLESS, path=str(tmp_path / "a.txt")) is True


def test_default_config_runs_a_dangerous_tool_without_confirmation(tmp_path):
    g = gate()
    decision = g.check_tool(ORDINARY, {"path": str(tmp_path / "a.txt"), "content": "x"})
    assert decision.requires_confirmation is False
    assert g.allows(decision) is True


def test_default_config_never_calls_the_confirm_callback(tmp_path):
    """The rails are off: a wired-up prompt must simply never fire."""
    prompts = []
    g = gate(confirm=lambda reason: (prompts.append(reason), True)[1])
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is True
    assert allows(g, DESTRUCTIVE_CMD, command="format C: /q") is True
    assert allows(g, DESTRUCTIVE_CMD, command="rm -rf / --no-preserve-root") is True
    assert prompts == []


@pytest.mark.parametrize("cmd", ["format C:", "rm -rf /"])
def test_default_config_allows_the_classic_disasters(cmd):
    g = gate()
    decision = g.check_command(cmd)
    assert decision.allowed is True
    assert decision.requires_confirmation is False
    assert g.allows(decision) is True


def test_default_config_writes_anywhere(tmp_path):
    g = gate()
    system_dir = "C:\\Windows" if IS_WINDOWS else "/etc"
    assert g.check_path(str(tmp_path / "a.txt"), write=True).allowed is True
    assert g.check_path(system_dir + os.sep + "x.txt", write=True).allowed is True


def test_default_config_protects_nothing():
    g = gate()
    assert g.is_protected("C:\\Windows") is False
    assert g.is_protected("/etc") is False


# --------------------------------------------------------------------------- #
#  Opt-in: the old behaviour is one config change away
# --------------------------------------------------------------------------- #
def test_opt_in_guarded_asks_before_destructive_commands():
    prompts = []
    g = restricted_gate(confirm=lambda reason: (prompts.append(reason), True)[1])
    assert allows(g, DESTRUCTIVE_CMD, command="format C: /q") is True
    assert len(prompts) == 1


def test_opt_in_declining_callback_refuses(tmp_path):
    g = restricted_gate(confirm=lambda reason: False)
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is False
    assert allows(g, DESTRUCTIVE_CMD, command="format C: /q") is False


def test_opt_in_callback_that_raises_counts_as_no(tmp_path):
    def explode(reason):
        raise RuntimeError("UI gone")

    g = restricted_gate(confirm=explode)
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is False


def test_opt_in_unattended_deny_refuses_everything_dangerous(tmp_path):
    g = restricted_gate(unattended_policy="deny")
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is False
    assert allows(g, HARMLESS, path=str(tmp_path / "a.txt")) is True


def test_opt_in_unattended_allow_keeps_headless_work_going(tmp_path):
    g = restricted_gate(unattended_policy="allow")
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is True


def test_opt_in_readonly_refuses_all_mutation(tmp_path):
    g = gate("readonly")
    assert allows(g, ORDINARY, path=str(tmp_path / "a.txt"), content="x") is False
    assert allows(g, DESTRUCTIVE_CMD, command="echo hello") is False


def test_opt_in_protected_paths_block_writes():
    g = restricted_gate()
    protected = ("C:\\Windows" if IS_WINDOWS else "/etc") + os.sep + "x.txt"
    assert g.is_protected(protected) is True
    assert g.check_path(protected, write=True).allowed is False


def test_open_mode_is_genuinely_open(tmp_path):
    """`open` must not secretly gate — not even on a protected path."""
    g = gate("open", protected_paths=(["C:\\Windows"] if IS_WINDOWS else ["/etc"]))
    protected = ("C:\\Windows" if IS_WINDOWS else "/etc") + os.sep + "x.txt"
    decision = g.check_tool(ORDINARY, {"path": protected, "content": "x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False


# --------------------------------------------------------------------------- #
#  The registry must honour the verdict, not just read it
# --------------------------------------------------------------------------- #
@pytest.fixture
def registry_with(config):
    def build(security):
        ctx = ToolContext(config=config, security=security, bus=None,
                          memory=None, extra={})
        registry = ToolRegistry(ctx)
        calls: list = []

        def wipe_everything(target: str) -> ToolResult:
            """Stand-in for a destructive operation; records instead of acting."""
            calls.append(target)
            return ToolResult.success(f"would have wiped {target}")

        registry.register(FunctionTool(wipe_everything, dangerous=True))
        return registry, calls

    return build


def test_registry_runs_dangerous_tools_by_default(registry_with):
    """Out of the box a dangerous tool just runs."""
    registry, calls = registry_with(gate())
    result = registry.run("wipe_everything", target="C:/important")

    assert result.ok, result.error
    assert calls == ["C:/important"]


def test_registry_refuses_when_opt_in_confirmation_is_declined(registry_with):
    """The opt-in path is genuinely wired, not merely present."""
    registry, calls = registry_with(restricted_gate(confirm=lambda reason: False))
    result = registry.run("wipe_everything", target="C:/important")

    assert result.ok is False
    assert not calls, "the tool ran despite confirmation being declined"


def test_registry_proceeds_when_opt_in_confirmation_is_granted(registry_with):
    registry, calls = registry_with(restricted_gate(confirm=lambda reason: True))
    result = registry.run("wipe_everything", target="C:/important")

    assert result.ok, result.error
    assert calls == ["C:/important"]


def test_registry_refuses_outright_denials(registry_with):
    registry, calls = registry_with(gate("readonly"))
    result = registry.run("wipe_everything", target="C:/important")

    assert result.ok is False
    assert not calls


# --------------------------------------------------------------------------- #
#  resolve_path — the regression that matters most
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not IS_WINDOWS, reason="drive-relative paths are a Windows concept")
def test_bare_drive_resolves_to_the_drive_root_and_not_the_cwd(tmp_path, monkeypatch):
    """``Path("C:").resolve()`` silently returns the CWD.  We must not.

    This is the exact confusion that once deleted this repository, so both
    halves are asserted: it *is* the drive root, and it is *not* here.
    """
    workdir = tmp_path / "somewhere_else"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    resolved = pu.resolve_path("C:")

    assert str(resolved) == "C:\\"
    assert resolved != Path.cwd()
    assert resolved != workdir
    # And prove the trap we are avoiding is real, so this test still means
    # something if the implementation is ever "simplified" back to resolve().
    assert Path("C:").resolve() == workdir


@pytest.mark.skipif(not IS_WINDOWS, reason="drive-relative paths are a Windows concept")
def test_lowercase_bare_drive_also_resolves_to_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = pu.resolve_path("c:")
    assert str(resolved) in ("c:\\", "C:\\")
    assert resolved != Path.cwd()


@pytest.mark.skipif(not IS_WINDOWS, reason="drive-relative paths are a Windows concept")
def test_drive_relative_with_a_tail_anchors_to_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = pu.resolve_path("C:Users")
    assert str(resolved) == "C:\\Users"
    assert Path.cwd() not in resolved.parents


def test_strict_mode_still_raises_on_drive_relative():
    with pytest.raises(ValueError) as excinfo:
        pu.resolve_path("C:", strict=True)
    assert "drive-relative" in str(excinfo.value)


def test_is_drive_relative_still_detects(tmp_path):
    assert pu.is_drive_relative("C:") is True
    assert pu.is_drive_relative("C:foo") is True
    assert pu.is_drive_relative("C:\\foo") is False
    assert pu.is_drive_relative(str(tmp_path)) is False


def test_ordinary_paths_still_resolve_normally(tmp_path):
    nested = tmp_path / "a" / ".." / "b"
    assert pu.resolve_path(str(nested)) == (tmp_path / "b")


# --------------------------------------------------------------------------- #
#  delete_path — deletes what it is told to
# --------------------------------------------------------------------------- #
class FakeConfig:
    def __init__(self, dir_path):
        self._dir = dir_path

    def tools_dir(self):
        return self._dir


def _file_registry(tmp_path, security):
    ctx = ToolContext(config=FakeConfig(tmp_path / "gen"), security=security)
    reg = ToolRegistry(ctx)
    for t in build_tools(ctx):
        reg.register(t, replace=True)
    return reg


def test_delete_removes_an_ordinary_file(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("bye", encoding="utf-8")

    result = _file_registry(tmp_path, gate()).run("delete_path", path=str(victim))

    assert result.ok, result.error
    assert not victim.exists()


def test_delete_removes_a_directory_tree_with_recursive(tmp_path):
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "f.txt").write_text("x", encoding="utf-8")

    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(tree), recursive=True
    )

    assert result.ok, result.error
    assert not tree.exists()


def test_delete_still_refuses_the_home_directory(tmp_path, monkeypatch):
    """Deleting a whole home directory is refused, and this is not policy.

    ``rmtree`` over a live profile aborts on the first locked or read-only file
    and leaves it half-destroyed — the outcome is never what asking for it
    implied.  ``HOME``/``USERPROFILE`` are redirected into ``tmp_path`` so the
    rule is exercised without the real home being named, let alone touched.
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    (fake_home / "notes.txt").write_text("x", encoding="utf-8")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(fake_home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    assert Path(os.path.expanduser("~")) == fake_home, "redirect did not take"

    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(fake_home), recursive=True
    )

    assert result.ok is False
    assert "home" in (result.error or "").lower()
    assert (fake_home / "notes.txt").exists(), "the home directory was emptied"


def test_delete_still_refuses_an_ancestor_of_the_working_directory(
    tmp_path, monkeypatch
):
    """Deleting the ground the process stands on aborts part-way; refuse it.

    ``Path.cwd`` is faked rather than a real chdir so the rule is exercised with
    the deletion target confined to ``tmp_path``.
    """
    workspace = tmp_path / "workspace"
    (workspace / "deep").mkdir(parents=True)
    (workspace / "deep" / "f.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: workspace / "deep"))
    assert workspace in Path.cwd().parents, "fake cwd is not below the target"

    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(workspace), recursive=True
    )

    assert result.ok is False
    assert (workspace / "deep" / "f.txt").exists(), "the workspace was deleted"


def test_delete_no_longer_refuses_a_path_the_gate_permits(tmp_path):
    """With protected_paths empty, nothing is 'protected' any more."""
    victim = tmp_path / "used_to_be_protected"
    victim.mkdir()
    (victim / "f.txt").write_text("x", encoding="utf-8")

    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(victim), recursive=True
    )

    assert result.ok, result.error
    assert not victim.exists()


def test_delete_still_refuses_a_filesystem_root(tmp_path, monkeypatch):
    """A drive root cannot be deleted; the attempt only breaks the machine.

    ``is_filesystem_root`` is faked so a directory inside ``tmp_path`` plays the
    part.  No test in this suite ever names a real root, home directory or
    working directory as a delete target: if a guard regressed, such a test
    would not report a failure, it would carry it out.
    """
    stand_in = tmp_path / "pretend_root"
    stand_in.mkdir()
    (stand_in / "f.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        file_tools, "is_filesystem_root", lambda p: Path(p) == stand_in
    )

    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(stand_in), recursive=True
    )

    assert result.ok is False
    assert "root" in (result.error or "").lower()
    assert stand_in.exists(), "the stand-in root was deleted"


def test_delete_honours_an_opt_in_readonly_gate(tmp_path):
    """The security call is still wired: readonly mode stops the delete."""
    victim = tmp_path / "victim.txt"
    victim.write_text("x", encoding="utf-8")

    result = _file_registry(tmp_path, gate("readonly")).run(
        "delete_path", path=str(victim)
    )

    assert result.ok is False
    assert victim.exists(), "readonly mode did not stop the delete"


def test_delete_honours_an_opt_in_declined_confirmation(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("x", encoding="utf-8")
    security = restricted_gate(confirm=lambda reason: False)

    result = _file_registry(tmp_path, security).run("delete_path", path=str(victim))

    assert result.ok is False
    assert victim.exists(), "a declined confirmation did not stop the delete"


def test_delete_reports_a_missing_path(tmp_path):
    result = _file_registry(tmp_path, gate()).run(
        "delete_path", path=str(tmp_path / "nope.txt")
    )
    assert result.ok is False
    assert "no such path" in (result.error or "").lower()


def test_delete_requires_recursive_for_directories(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    result = _file_registry(tmp_path, gate()).run("delete_path", path=str(d))
    assert result.ok is False
    assert d.exists()


# --------------------------------------------------------------------------- #
#  Audit still records
# --------------------------------------------------------------------------- #
def test_audit_records_an_allowed_tool_call(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    cfg = SecurityConfig(audit_log=True)
    g = SecurityGate(cfg, audit_path=audit_path)

    g.check_tool(DESTRUCTIVE_CMD, {"command": "rm -rf /"})

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["allowed"] is True
    assert "run_command" in record["detail"]


def test_audit_records_an_unattended_approval(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    cfg = SecurityConfig(audit_log=True, mode="guarded")
    cfg.dangerous_patterns = ["shutdown"]
    g = SecurityGate(cfg, audit_path=audit_path)

    assert g.allows(g.check_command("shutdown /r /t 0")) is True

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(r["action"] == "unattended-approved" for r in records), records
