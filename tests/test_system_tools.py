"""System-tool tests — must run against stdlib alone."""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

from jarvis.core.contracts import ToolResult
from jarvis.core.platform_utils import IS_WINDOWS
from jarvis.tools.registry import ToolContext, ToolRegistry
from jarvis.tools.system_tools import build_tools


class OpenSecurity:
    def check_tool(self, spec, cleaned):
        class D:
            allowed = True
            requires_confirmation = False

        return D()

    def check_command(self, cmd):
        class D:
            allowed = True
            requires_confirmation = False

        return D()

    def is_protected(self, path: str) -> bool:
        return False

    def allows(self, decision) -> bool:
        return True


class DenySecurity(OpenSecurity):
    def check_command(self, cmd):
        class D:
            allowed = False
            reason = "denied by policy"
            requires_confirmation = False

        return D()


class FakeConfig:
    def __init__(self, tools_dir):
        self._dir = tools_dir

        class Sec:
            command_timeout = 30.0

        class C:
            security = Sec()

        self.security = Sec()

    def tools_dir(self):
        return self._dir


def _mk_registry(tmp_path, security=None):
    ctx = ToolContext(
        config=FakeConfig(tmp_path / "gen"),
        security=security if security is not None else OpenSecurity(),
    )
    reg = ToolRegistry(ctx)
    for t in build_tools(ctx):
        reg.register(t, replace=True)
    return reg


# --------------------------------------------------------------------------- #
#  system_info without psutil
# --------------------------------------------------------------------------- #
def _no_psutil_importer(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *args, **kw):
        if name == "psutil" or name.startswith("psutil."):
            raise ImportError("psutil is disabled for the test")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)


def test_system_info_without_psutil(tmp_path, monkeypatch):
    _no_psutil_importer(monkeypatch)
    reg = _mk_registry(tmp_path)
    r = reg.run("system_info")
    assert r.ok is True
    out = r.output
    assert out["psutil"] is False
    assert out["cpu_count"] >= 1
    assert "os" in out and "python" in out
    assert out["home_disk"] is None or "total" in out["home_disk"]


def test_system_info_shape(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("system_info")
    assert r.ok is True
    for key in ("os", "cpu_count", "hostname", "python"):
        assert key in r.output


# --------------------------------------------------------------------------- #
#  run_command
# --------------------------------------------------------------------------- #
def test_run_command_echo_roundtrip(tmp_path):
    reg = _mk_registry(tmp_path)
    if IS_WINDOWS:
        cmd = 'Write-Output "hello-from-jarvis"'
    else:
        cmd = "echo hello-from-jarvis"
    r = reg.run("run_command", command=cmd, timeout=15)
    assert r.ok is True
    assert r.output["returncode"] == 0
    assert "hello-from-jarvis" in r.output["stdout"]


def test_run_command_nonzero_exit(tmp_path):
    reg = _mk_registry(tmp_path)
    if IS_WINDOWS:
        cmd = "exit 7"
    else:
        cmd = "exit 7"
    r = reg.run("run_command", command=cmd, timeout=15)
    assert r.ok is True
    assert r.output["returncode"] == 7
    assert r.output["ok"] is False


def test_run_command_security_denial(tmp_path):
    reg = _mk_registry(tmp_path, security=DenySecurity())
    r = reg.run("run_command", command="echo x", timeout=5)
    assert r.ok is False
    assert "denied" in r.error.lower() or "policy" in r.error.lower()


def test_run_command_empty_fails(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("run_command", command="   ", timeout=5)
    assert r.ok is False


# --------------------------------------------------------------------------- #
#  get_env / set_env
# --------------------------------------------------------------------------- #
def test_get_env_and_set_env(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_TESTVAR", raising=False)
    reg = _mk_registry(tmp_path)

    r = reg.run("get_env", name="JARVIS_TESTVAR")
    assert r.ok is False

    r = reg.run("set_env", name="JARVIS_TESTVAR", value="hello")
    assert r.ok is True
    assert os.environ["JARVIS_TESTVAR"] == "hello"

    r = reg.run("get_env", name="JARVIS_TESTVAR")
    assert r.ok is True
    assert r.output["value"] == "hello"


def test_get_env_missing_name(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("get_env", name="")
    assert r.ok is False


# --------------------------------------------------------------------------- #
#  disk_usage
# --------------------------------------------------------------------------- #
def test_disk_usage_for_specific_path(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("disk_usage", path=str(tmp_path))
    assert r.ok is True
    assert r.output["total"] > 0
    assert r.output["free"] >= 0


def test_disk_usage_no_psutil_still_works(tmp_path, monkeypatch):
    _no_psutil_importer(monkeypatch)
    reg = _mk_registry(tmp_path)
    r = reg.run("disk_usage")
    assert r.ok is True
    assert r.output["count"] >= 0


# --------------------------------------------------------------------------- #
#  Capabilities that should degrade gracefully
# --------------------------------------------------------------------------- #
def test_battery_status_without_psutil(tmp_path, monkeypatch):
    _no_psutil_importer(monkeypatch)
    reg = _mk_registry(tmp_path)
    r = reg.run("battery_status")
    assert r.ok is False
    assert "psutil" in r.error.lower()


def test_screenshot_without_mss(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def guard(name, *args, **kw):
        if name == "mss" or name.startswith("mss."):
            raise ImportError("mss is disabled")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.delitem(sys.modules, "mss", raising=False)
    reg = _mk_registry(tmp_path)
    r = reg.run("take_screenshot")
    assert r.ok is False
    assert "mss" in r.error.lower()


def test_power_action_unknown(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("power_action", action="teleport")
    assert r.ok is False
    assert "supported" in r.error.lower() or "not available" in r.error.lower()


def test_power_action_security_refusal(tmp_path):
    reg = _mk_registry(tmp_path, security=DenySecurity())
    r = reg.run("power_action", action="shutdown")
    assert r.ok is False
    assert "denied" in r.error.lower() or "policy" in r.error.lower()


# --------------------------------------------------------------------------- #
#  Clipboard on Linux without any clipboard tool
# --------------------------------------------------------------------------- #
def test_clipboard_get_gracefully_fails_when_unavailable(tmp_path, monkeypatch):
    from jarvis.tools import system_tools as st

    monkeypatch.setattr(st, "IS_WINDOWS", False)
    monkeypatch.setattr(st, "IS_MAC", False)
    monkeypatch.setattr(st, "IS_LINUX", True)
    monkeypatch.setattr(st, "which", lambda name: None)
    reg = _mk_registry(tmp_path)
    r = reg.run("clipboard_get")
    assert r.ok is False
    assert "clipboard" in r.error.lower()


def test_clipboard_set_gracefully_fails_when_unavailable(tmp_path, monkeypatch):
    from jarvis.tools import system_tools as st

    monkeypatch.setattr(st, "IS_WINDOWS", False)
    monkeypatch.setattr(st, "IS_MAC", False)
    monkeypatch.setattr(st, "IS_LINUX", True)
    monkeypatch.setattr(st, "which", lambda name: None)
    reg = _mk_registry(tmp_path)
    r = reg.run("clipboard_set", text="hi")
    assert r.ok is False


# --------------------------------------------------------------------------- #
#  Notify + volume tools on a platform with no helpers installed
# --------------------------------------------------------------------------- #
def test_notify_without_helper(tmp_path, monkeypatch):
    from jarvis.tools import system_tools as st

    monkeypatch.setattr(st, "IS_WINDOWS", False)
    monkeypatch.setattr(st, "IS_MAC", False)
    monkeypatch.setattr(st, "IS_LINUX", True)
    monkeypatch.setattr(st, "which", lambda name: None)
    reg = _mk_registry(tmp_path)
    r = reg.run("notify", title="t", message="m")
    assert r.ok is False


def test_volume_set_range_validation(tmp_path, monkeypatch):
    from jarvis.tools import system_tools as st

    monkeypatch.setattr(st, "IS_WINDOWS", False)
    monkeypatch.setattr(st, "IS_MAC", False)
    monkeypatch.setattr(st, "IS_LINUX", True)
    monkeypatch.setattr(st, "which", lambda name: None)
    reg = _mk_registry(tmp_path)
    r = reg.run("volume_set", level=50)
    assert r.ok is False
    assert "volume" in r.error.lower() or "no volume" in r.error.lower() or "available" in r.error.lower()
