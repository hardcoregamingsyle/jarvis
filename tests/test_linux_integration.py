"""Linux integration: systemd user service, desktop, audio — tested from Windows.

Nothing here needs Linux, systemd, an audio device or a desktop session.  The
host is faked (``IS_LINUX``/``IS_WINDOWS``), the binaries are faked (``which``
returns paths for a chosen set of tools), and their output is faked
(``run_command`` replays canned text).  That is the point: the modules have to
be correct on a machine that cannot run them, and every one of them must also
stay inert — False, empty, never an exception — when it *is* the wrong host.

Safety: the autouse ``xdg`` fixture redirects ``XDG_CONFIG_HOME`` into
``tmp_path`` before any test runs.  Without it a test that fakes a Linux host
would write a real ``~/.config/systemd/user/jarvis.service`` and a real
``~/.config/autostart/jarvis.desktop`` on the developer's machine.  No test
here names a real home, root or working directory.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from jarvis.core import platform_utils
from jarvis.core.platform_utils import CommandResult
from jarvis.linux import audio, desktop, service

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
class FakeShell:
    """Stands in for ``which`` and ``run_command``.

    ``tools`` are the binaries that "exist"; anything else resolves to None, as
    on a machine where the package was never installed.  ``responses`` maps a
    command line — ``"pactl info"``, the program's base name followed by its
    arguments — to ``(returncode, stdout, stderr)``.  The longest key that is a
    prefix of the actual command wins, so ``"systemctl --user"`` can catch a
    whole family while ``"systemctl --user is-active jarvis.service"`` overrides
    one member of it.

    An unmatched command returns exit 127, which is what a shell reports for a
    command that is not there — a test that forgets to script something fails
    on its assertion rather than silently passing on a fabricated success.
    """

    def __init__(
        self,
        tools: Tuple[str, ...] = (),
        responses: Optional[Dict[str, Tuple[int, str, str]]] = None,
        *,
        raises: bool = False,
    ) -> None:
        self.tools = {name: f"/usr/bin/{name}" for name in tools}
        self.responses = dict(responses or {})
        self.raises = raises
        self.calls: List[List[str]] = []
        self.timeouts: List[Optional[float]] = []

    # -- the two platform_utils entry points ------------------------------ #
    def which(self, name: str) -> Optional[str]:
        return self.tools.get(name)

    def run_command(self, command, *, timeout=60.0, cwd=None, env=None, shell=False):
        argv = [str(part) for part in command]
        self.calls.append(argv)
        self.timeouts.append(timeout)
        if self.raises:
            raise OSError("the fake shell was told to fail")
        line = " ".join([Path(argv[0]).name] + argv[1:])
        best = None
        for key in self.responses:
            if line.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        if best is None:
            return CommandResult(127, "", f"no fake response for: {line}")
        code, out, err = self.responses[best]
        return CommandResult(code, out, err)

    # -- assertions helpers ------------------------------------------------ #
    def ran(self, fragment: str) -> bool:
        return any(fragment in " ".join(argv) for argv in self.calls)

    def command_lines(self) -> List[str]:
        return [" ".join(argv) for argv in self.calls]


#: The interpreter a faked Linux host is running.  Faking the host has to
#: include this: the real ``sys.executable`` here is ``C:\\...\\python.exe``,
#: which systemd would rightly reject as a non-absolute ExecStart.
LINUX_PYTHON = "/opt/jarvis/.venv/bin/python"


def fake_host(monkeypatch: pytest.MonkeyPatch, shell: FakeShell, *, linux: bool = True) -> None:
    """Point the whole package at ``shell`` and pretend to be Linux (or not)."""
    monkeypatch.setattr(platform_utils, "IS_LINUX", linux)
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", not linux)
    monkeypatch.setattr(platform_utils, "IS_MAC", False)
    monkeypatch.setattr(platform_utils, "which", shell.which)
    monkeypatch.setattr(platform_utils, "run_command", shell.run_command)
    if linux:
        monkeypatch.setattr(sys, "executable", LINUX_PYTHON)


@pytest.fixture(autouse=True)
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every XDG path into tmp_path and clear the session variables."""
    root = tmp_path / "xdg-config"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER", "testuser")
    return root


# --------------------------------------------------------------------------- #
#  Inert on the wrong host
# --------------------------------------------------------------------------- #
def test_modules_are_inert_on_a_real_windows_box():
    """No fakes at all: the real dev host must get honest negatives, not errors."""
    assert service.is_available() is False
    assert audio.is_available() is False
    assert audio.backend() == "none"
    assert audio.list_input_devices() == []
    assert audio.list_output_devices() == []
    assert audio.default_input() is None
    assert audio.package_manager() is None
    assert service.linger_enabled() is None

    status = service.status()
    assert status["supported"] is False
    assert status["error"] and "Linux-only" in status["error"]

    report = audio.check()
    assert report["ok"] is False
    assert report["backend"] == "none"
    assert any("not Linux" in p for p in report["problems"])

    capabilities = desktop.is_available()
    assert capabilities["window_control"] is False
    assert capabilities["global_hotkeys"] is False
    assert capabilities["autostart"] is False


def test_operations_refuse_on_windows_without_touching_the_filesystem(xdg: Path):
    """A Linux-only operation fails with an explanation and writes nothing."""
    result = service.install()
    assert result.ok is False
    assert "Linux-only" in (result.error or "")
    assert not service.unit_path().exists()

    for outcome in (desktop.list_windows(), desktop.active_window(), desktop.focus_window("x")):
        assert outcome.ok is False
        assert "windows" in (outcome.error or "").lower()

    assert audio.set_default_input("mic").ok is False
    assert list(xdg.rglob("*.service")) == []


def test_nothing_installed_on_linux_reports_false_and_never_raises(monkeypatch):
    """A bare Linux box with no tooling at all: negatives everywhere, no crash."""
    shell = FakeShell(tools=())
    fake_host(monkeypatch, shell)

    assert service.is_available() is False
    assert audio.is_available() is False
    assert audio.backend() == "none"
    assert audio.list_input_devices() == []
    assert desktop.notify("t", "m") == "log"

    windows = desktop.list_windows()
    assert windows.ok is False
    assert "wmctrl" in windows.error and "xdotool" in windows.error

    report = audio.check()
    assert report["ok"] is False
    assert any("sound server" in p for p in report["problems"])


def test_a_raising_shell_does_not_escape_any_probe(monkeypatch):
    """Every probe swallows an exploding subprocess layer and answers anyway."""
    shell = FakeShell(tools=("pactl", "systemctl", "notify-send", "wmctrl"), raises=True)
    fake_host(monkeypatch, shell)

    assert service.is_available() is False
    assert audio.backend() == "none"
    assert audio.list_input_devices() == []
    assert audio.in_audio_group() is None
    assert desktop.notify("t", "m") == "log"
    assert isinstance(audio.check(), dict)
    assert isinstance(service.status(), dict)
    assert isinstance(desktop.is_available(), dict)


# --------------------------------------------------------------------------- #
#  service: the unit file
# --------------------------------------------------------------------------- #
SPACED_PYTHON = "/opt/My Apps/jarvis/.venv/bin/python"


def test_unit_text_has_the_directives_systemd_needs():
    command = f'{service.quote_exec(SPACED_PYTHON)} -m jarvis voice'
    text = service.unit_text(command, description="JARVIS test unit")

    assert "[Unit]" in text and "[Service]" in text and "[Install]" in text
    assert f'ExecStart="{SPACED_PYTHON}" -m jarvis voice' in text
    assert "Restart=always" in text
    assert "RestartSec=5" in text
    assert "WantedBy=default.target" in text
    assert "Description=JARVIS test unit" in text
    # The runaway-restart guard is resource management, and must be present.
    assert "StartLimitBurst=5" in text


def test_unit_text_documents_the_audio_and_target_choices():
    text = service.unit_text(service.default_command())
    assert "XDG_RUNTIME_DIR" in text
    assert "graphical-session.target" in text  # explained as the choice NOT made
    assert "pipewire" in text.lower()


def test_unit_text_can_disable_restart():
    text = service.unit_text(service.default_command(), restart=False)
    assert "Restart=no" in text
    assert "Restart=always" not in text


def test_quote_exec_always_quotes_and_is_idempotent():
    assert service.quote_exec(SPACED_PYTHON) == f'"{SPACED_PYTHON}"'
    assert service.quote_exec("/usr/bin/python3") == '"/usr/bin/python3"'
    assert service.quote_exec('"/usr/bin/python3"') == '"/usr/bin/python3"'


@pytest.mark.parametrize(
    "command, expected",
    [
        ('"/opt/My Apps/python" -m jarvis voice', True),
        ("/usr/bin/python3 -m jarvis voice", True),
        ("-/usr/bin/python3 -m jarvis", True),      # systemd's ignore-failure prefix
        ("python3 -m jarvis voice", False),
        ("./jarvis voice", False),
        ("", False),
    ],
)
def test_exec_start_absolute_detection(command, expected):
    assert service.exec_start_is_absolute(command) is expected


# --------------------------------------------------------------------------- #
#  service: install / lifecycle
# --------------------------------------------------------------------------- #
def systemd_shell(**extra) -> FakeShell:
    responses = {
        "systemctl --user show-environment": (0, "LANG=en_GB.UTF-8\n", ""),
        "systemctl --user daemon-reload": (0, "", ""),
        "systemctl --user enable": (0, "", ""),
        "systemctl --user disable": (0, "", ""),
        "systemctl --user start": (0, "", ""),
        "systemctl --user stop": (0, "", ""),
        "systemctl --user restart": (0, "", ""),
        "systemctl --user is-enabled": (0, "enabled\n", ""),
        "systemctl --user is-active": (0, "active\n", ""),
        "loginctl show-user testuser --property=Linger": (0, "Linger=yes\n", ""),
        "journalctl --user": (0, "Aug 08 20:00:00 box jarvis[1]: ready\n", ""),
    }
    responses.update(extra)
    return FakeShell(tools=("systemctl", "journalctl", "loginctl"), responses=responses)


def test_install_writes_the_unit_and_reloads_systemd(monkeypatch, xdg: Path):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)
    monkeypatch.setattr(sys, "executable", SPACED_PYTHON)

    result = service.install(description="JARVIS voice assistant")
    assert result.ok, result.error

    unit = xdg / "systemd" / "user" / "jarvis.service"
    assert unit.exists()
    text = unit.read_text(encoding="utf-8")
    assert f'ExecStart="{SPACED_PYTHON}" -m jarvis voice' in text
    assert "WantedBy=default.target" in text

    assert result.output["daemon_reloaded"] is True
    assert shell.ran("systemctl --user daemon-reload")
    assert any("enable-linger testuser" in step for step in result.output["next_steps"])


def test_install_reports_a_failed_daemon_reload_without_losing_the_unit(monkeypatch, xdg: Path):
    shell = systemd_shell(**{"systemctl --user daemon-reload": (1, "", "Failed to connect")})
    fake_host(monkeypatch, shell)

    result = service.install()
    assert result.ok
    assert result.output["daemon_reloaded"] is False
    assert (xdg / "systemd" / "user" / "jarvis.service").exists()


def test_install_rejects_a_relative_exec_start(monkeypatch):
    fake_host(monkeypatch, systemd_shell())
    result = service.install("python3 -m jarvis voice")
    assert result.ok is False
    assert "absolute" in result.error
    assert not service.unit_path().exists()


@pytest.mark.parametrize("command", ["", "   ", "python\n-m jarvis"])
def test_install_rejects_empty_and_multiline_commands(monkeypatch, command):
    fake_host(monkeypatch, systemd_shell())
    result = service.install(command)
    assert result.ok is False
    assert not service.unit_path().exists()


def test_lifecycle_verbs_reach_systemctl(monkeypatch):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)
    assert service.install().ok

    for verb, call in (
        ("enable", service.enable),
        ("start", service.start),
        ("restart", service.restart),
        ("stop", service.stop),
        ("disable", service.disable),
    ):
        result = call()
        assert result.ok, result.error
        assert shell.ran(f"systemctl --user {verb} jarvis.service")


def test_lifecycle_refuses_before_install(monkeypatch):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)
    result = service.start()
    assert result.ok is False
    assert "install()" in result.error
    assert not shell.ran("systemctl --user start")


def test_lifecycle_surfaces_the_systemctl_error(monkeypatch):
    shell = systemd_shell(
        **{"systemctl --user start": (1, "", "Job for jarvis.service failed")}
    )
    fake_host(monkeypatch, shell)
    assert service.install().ok

    result = service.start()
    assert result.ok is False
    assert "Job for jarvis.service failed" in result.error


def test_uninstall_removes_the_unit_and_is_idempotent(monkeypatch, xdg: Path):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)
    assert service.install().ok
    unit = xdg / "systemd" / "user" / "jarvis.service"
    assert unit.exists()

    assert service.uninstall().ok
    assert not unit.exists()
    assert shell.ran("systemctl --user disable jarvis.service")
    assert service.uninstall().ok       # already gone is still success


def test_logs_calls_journalctl_and_clamps_the_line_count(monkeypatch):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)

    result = service.logs(10)
    assert result.ok
    assert "ready" in result.output
    assert shell.ran("journalctl --user -u jarvis.service -n 10 --no-pager")

    service.logs(10 ** 9)
    assert shell.ran(f"-n {service.MAX_LOG_LINES}")


def test_logs_without_journalctl_points_at_systemctl_status(monkeypatch):
    shell = FakeShell(tools=("systemctl",), responses={})
    fake_host(monkeypatch, shell)
    result = service.logs()
    assert result.ok is False
    assert "systemctl --user status" in result.error


def test_is_available_needs_a_reachable_user_bus(monkeypatch):
    reachable = systemd_shell()
    fake_host(monkeypatch, reachable)
    assert service.is_available() is True

    broken = systemd_shell(
        **{"systemctl --user show-environment": (1, "", "Failed to connect to bus")}
    )
    fake_host(monkeypatch, broken)
    assert service.is_available() is False


# --------------------------------------------------------------------------- #
#  service: lingering — the reason "it stopped when I closed the lid"
# --------------------------------------------------------------------------- #
def test_status_leads_with_the_linger_warning_when_lingering_is_off(monkeypatch):
    shell = systemd_shell(
        **{"loginctl show-user testuser --property=Linger": (0, "Linger=no\n", "")}
    )
    fake_host(monkeypatch, shell)
    assert service.install().ok

    report = service.status()
    assert report["linger"] is False
    assert report["linger_command"] == "loginctl enable-linger testuser"
    first = report["advice"][0]
    assert "loginctl enable-linger testuser" in first
    assert "log out" in first.lower()


def test_status_stays_quiet_about_linger_when_it_is_enabled(monkeypatch):
    shell = systemd_shell()
    fake_host(monkeypatch, shell)
    assert service.install().ok

    report = service.status()
    assert report["linger"] is True
    assert report["enabled"] == "enabled"
    assert report["active"] == "active"
    assert not any("enable-linger" in line for line in report["advice"])


def test_status_says_so_when_lingering_cannot_be_determined(monkeypatch):
    shell = FakeShell(
        tools=("systemctl",),
        responses={
            "systemctl --user show-environment": (0, "", ""),
            "systemctl --user daemon-reload": (0, "", ""),
            "systemctl --user is-enabled": (0, "disabled\n", ""),
            "systemctl --user is-active": (3, "inactive\n", ""),
        },
    )
    fake_host(monkeypatch, shell)
    assert service.install().ok

    report = service.status()
    assert report["linger"] in (None, False)
    assert any("enable-linger testuser" in line for line in report["advice"])
    assert any("not enabled" in line for line in report["advice"])


def test_status_tells_you_to_install_when_no_unit_exists(monkeypatch):
    fake_host(monkeypatch, systemd_shell())
    report = service.status()
    assert report["installed"] is False
    assert any("install()" in line for line in report["advice"])


def test_linger_is_read_from_loginctl(monkeypatch):
    shell = systemd_shell(
        **{"loginctl show-user testuser --property=Linger": (0, "Linger=yes\n", "")}
    )
    fake_host(monkeypatch, shell)
    assert service.linger_enabled() is True
    assert shell.ran("loginctl show-user testuser --property=Linger")


# --------------------------------------------------------------------------- #
#  desktop: session detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "environment, expected",
    [
        ({"XDG_SESSION_TYPE": "wayland"}, "wayland"),
        ({"XDG_SESSION_TYPE": "x11"}, "x11"),
        ({"XDG_SESSION_TYPE": "tty"}, "tty"),
        ({"XDG_SESSION_TYPE": "Wayland"}, "wayland"),          # case-insensitive
        ({"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "wayland-0"}, "wayland"),
        ({"WAYLAND_DISPLAY": "wayland-0"}, "wayland"),
        ({"DISPLAY": ":0"}, "x11"),
        ({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, "wayland"),  # XWayland
        ({}, "unknown"),
    ],
)
def test_session_type_reads_the_environment(monkeypatch, environment, expected):
    for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert desktop.session_type() == expected
    assert desktop.is_wayland() is (expected == "wayland")


# --------------------------------------------------------------------------- #
#  desktop: Wayland refuses instead of pretending
# --------------------------------------------------------------------------- #
def wayland(monkeypatch) -> FakeShell:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    shell = FakeShell(
        tools=("wmctrl", "xdotool"),
        responses={"wmctrl -lpG": (0, "0x01 0 1 0 0 100 100 box Should never be read\n", "")},
    )
    fake_host(monkeypatch, shell)
    return shell


@pytest.mark.parametrize(
    "operation",
    [
        lambda: desktop.list_windows(),
        lambda: desktop.active_window(),
        lambda: desktop.focus_window("Firefox"),
    ],
)
def test_window_operations_fail_actionably_under_wayland(monkeypatch, operation):
    shell = wayland(monkeypatch)
    result = operation()

    assert result.ok is False
    error = result.error
    assert "Wayland" in error
    assert "wmctrl" in error and "xdotool" in error
    assert "ydotool" in error
    assert "Xorg" in error
    assert "extension" in error.lower()
    # The decisive property: it refused *before* running anything, so it cannot
    # have reported a success that never happened.
    assert shell.calls == []


def test_global_hotkeys_under_wayland_name_the_compositor_route(monkeypatch):
    wayland(monkeypatch)
    result = desktop.global_hotkeys()
    assert result.ok is False
    assert "Wayland" in result.error
    assert "Custom Shortcut" in result.error
    assert "evdev" in result.error


def test_global_hotkeys_on_x11_report_the_backends(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    fake_host(monkeypatch, FakeShell(tools=("xdotool",)))

    result = desktop.global_hotkeys()
    assert result.ok
    assert result.output["session"] == "x11"
    assert result.output["backends"]["xdotool"] is True


def test_global_hotkeys_on_x11_without_display_explain_import_environment(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    fake_host(monkeypatch, FakeShell(tools=("xdotool",)))

    result = desktop.global_hotkeys()
    assert result.ok is False
    assert "import-environment" in result.error


def test_global_hotkeys_on_a_tty_say_there_is_no_display(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "tty")
    fake_host(monkeypatch, FakeShell())

    result = desktop.global_hotkeys()
    assert result.ok is False
    assert "text console" in result.error


# --------------------------------------------------------------------------- #
#  desktop: windows on X11
# --------------------------------------------------------------------------- #
WMCTRL_OUTPUT = (
    "0x03400007  0 3421   0    27   1920 1053 jarvis-box Firefox\n"
    "0x03600004  1 3999   1920 27   1280 720  jarvis-box Terminal - bash\n"
    "malformed line\n"
)


def x11(monkeypatch, shell: FakeShell) -> FakeShell:
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    fake_host(monkeypatch, shell)
    return shell


def test_list_windows_parses_wmctrl(monkeypatch):
    x11(monkeypatch, FakeShell(tools=("wmctrl",), responses={"wmctrl -lpG": (0, WMCTRL_OUTPUT, "")}))

    result = desktop.list_windows()
    assert result.ok
    windows = result.output
    assert [w["title"] for w in windows] == ["Firefox", "Terminal - bash"]
    assert windows[0] == {
        "id": "0x03400007", "desktop": 0, "pid": 3421,
        "x": 0, "y": 27, "width": 1920, "height": 1053,
        "host": "jarvis-box", "title": "Firefox",
    }


def test_list_windows_falls_back_to_xdotool(monkeypatch):
    shell = FakeShell(
        tools=("xdotool",),
        responses={
            "xdotool search --onlyvisible --name .": (0, "111\n222\n", ""),
            "xdotool getwindowname 111": (0, "Firefox\n", ""),
            "xdotool getwindowname 222": (0, "Terminal\n", ""),
        },
    )
    x11(monkeypatch, shell)

    result = desktop.list_windows()
    assert result.ok
    assert [w["title"] for w in result.output] == ["Firefox", "Terminal"]
    assert all(w["backend"] == "xdotool" for w in result.output)


def test_focus_window_prefers_wmctrl(monkeypatch):
    shell = x11(monkeypatch, FakeShell(tools=("wmctrl", "xdotool"),
                                       responses={"wmctrl -a Firefox": (0, "", "")}))
    result = desktop.focus_window("Firefox")
    assert result.ok
    assert result.output["backend"] == "wmctrl"
    assert not shell.ran("xdotool")


def test_focus_window_falls_back_to_xdotool_then_fails_honestly(monkeypatch):
    shell = x11(
        monkeypatch,
        FakeShell(
            tools=("wmctrl", "xdotool"),
            responses={
                "wmctrl -a Firefox": (1, "", "no such window"),
                "xdotool search --name Firefox windowactivate %1": (0, "", ""),
            },
        ),
    )
    assert desktop.focus_window("Firefox").ok
    assert shell.ran("xdotool search --name Firefox windowactivate")

    missing = desktop.focus_window("Nothing")
    assert missing.ok is False
    assert "Nothing" in missing.error


def test_focus_window_needs_a_title(monkeypatch):
    x11(monkeypatch, FakeShell(tools=("wmctrl",)))
    assert desktop.focus_window("   ").ok is False


def test_active_window_uses_xdotool(monkeypatch):
    shell = FakeShell(
        tools=("wmctrl", "xdotool"),
        responses={
            "xdotool getactivewindow": (0, "12345\n", ""),
            "xdotool getwindowname 12345": (0, "Firefox\n", ""),
        },
    )
    x11(monkeypatch, shell)

    result = desktop.active_window()
    assert result.ok
    assert result.output == {"id": "12345", "title": "Firefox", "backend": "xdotool"}


def test_active_window_without_xdotool_says_wmctrl_cannot_do_it(monkeypatch):
    x11(monkeypatch, FakeShell(tools=("wmctrl",)))
    result = desktop.active_window()
    assert result.ok is False
    assert "wmctrl cannot report the focused window" in result.error


# --------------------------------------------------------------------------- #
#  desktop: notifications
# --------------------------------------------------------------------------- #
def test_notify_prefers_notify_send_and_passes_urgency_and_icon(monkeypatch):
    shell = FakeShell(tools=("notify-send", "gdbus"),
                      responses={"notify-send": (0, "", "")})
    fake_host(monkeypatch, shell)

    assert desktop.notify("Title", "Body", urgency="critical", icon="audio-input-microphone") \
        == "notify-send"
    argv = shell.calls[0]
    assert argv[-2:] == ["Title", "Body"]
    assert "-u" in argv and argv[argv.index("-u") + 1] == "critical"
    assert "-i" in argv and argv[argv.index("-i") + 1] == "audio-input-microphone"
    assert not shell.ran("gdbus")


def test_notify_falls_back_to_gdbus(monkeypatch):
    shell = FakeShell(
        tools=("notify-send", "gdbus"),
        responses={"notify-send": (1, "", "no notification daemon"), "gdbus call": (0, "(u 1,)", "")},
    )
    fake_host(monkeypatch, shell)

    assert desktop.notify("Title", "Body") == "gdbus"
    gdbus_argv = shell.calls[-1]
    assert "org.freedesktop.Notifications.Notify" in gdbus_argv
    # Strings are GVariant-quoted so a title starting with '[' cannot be parsed
    # as an array and blow the call up.
    assert '"Title"' in gdbus_argv and '"Body"' in gdbus_argv
    assert "{'urgency': <byte 1>}" in gdbus_argv
    assert gdbus_argv[-1] == "5000"


def test_gdbus_notification_marks_critical_as_never_expiring(monkeypatch):
    shell = FakeShell(tools=("gdbus",), responses={"gdbus call": (0, "(u 1,)", "")})
    fake_host(monkeypatch, shell)

    assert desktop.notify("[warning]", "Body", urgency="critical") == "gdbus"
    argv = shell.calls[-1]
    assert "{'urgency': <byte 2>}" in argv
    assert argv[-1] == "0"          # 0 means "do not time out"
    assert '"[warning]"' in argv    # quoted, or gdbus would read it as an array


def test_notify_ends_at_the_log_channel(monkeypatch, caplog):
    fake_host(monkeypatch, FakeShell(tools=()))
    with caplog.at_level("INFO", logger="jarvis.linux.desktop"):
        assert desktop.notify("Title", "Body") == "log"
    assert "Title" in caplog.text


def test_notify_clamps_an_unknown_urgency(monkeypatch):
    shell = FakeShell(tools=("notify-send",), responses={"notify-send": (0, "", "")})
    fake_host(monkeypatch, shell)
    desktop.notify("t", "m", urgency="apocalyptic")
    argv = shell.calls[0]
    assert argv[argv.index("-u") + 1] == "normal"


# --------------------------------------------------------------------------- #
#  desktop: autostart
# --------------------------------------------------------------------------- #
def test_autostart_round_trips_in_a_fake_xdg_directory(monkeypatch, xdg: Path):
    fake_host(monkeypatch, FakeShell())
    entry = xdg / "autostart" / "jarvis.desktop"

    assert desktop.autostart_is_enabled() is False
    assert not entry.exists()

    enabled = desktop.autostart_enable('"/opt/My Apps/python" -m jarvis voice')
    assert enabled.ok, enabled.error
    assert Path(enabled.output["path"]) == entry
    assert entry.exists()

    text = entry.read_text(encoding="utf-8")
    assert text.startswith("[Desktop Entry]")
    assert 'Exec="/opt/My Apps/python" -m jarvis voice' in text
    assert "Type=Application" in text
    assert "X-GNOME-Autostart-enabled=true" in text

    assert desktop.autostart_is_enabled() is True
    assert desktop.autostart_command() == '"/opt/My Apps/python" -m jarvis voice'

    assert desktop.autostart_disable().ok
    assert not entry.exists()
    assert desktop.autostart_is_enabled() is False
    assert desktop.autostart_disable().ok        # idempotent


def test_autostart_defaults_to_this_interpreter(monkeypatch, xdg: Path):
    fake_host(monkeypatch, FakeShell())
    monkeypatch.setattr(sys, "executable", SPACED_PYTHON)

    assert desktop.autostart_enable().ok
    assert desktop.autostart_command() == f'"{SPACED_PYTHON}" -m jarvis voice'


@pytest.mark.parametrize("command", ["", "   ", "python\n-m jarvis"])
def test_autostart_rejects_empty_and_multiline_commands(monkeypatch, command, xdg: Path):
    fake_host(monkeypatch, FakeShell())
    result = desktop.autostart_enable(command)
    assert result.ok is False
    assert not (xdg / "autostart" / "jarvis.desktop").exists()


def test_autostart_still_works_under_wayland(monkeypatch, xdg: Path):
    """The compositor reads this file, so unlike wmctrl it is unaffected."""
    wayland(monkeypatch)
    assert desktop.autostart_enable().ok
    assert (xdg / "autostart" / "jarvis.desktop").exists()
    assert desktop.autostart_is_enabled() is True


def test_autostart_shares_its_path_with_the_windows_module():
    """One entry, not two — two would launch two assistants on one microphone."""
    from jarvis.win import autostart as win_autostart

    assert desktop.autostart_path() == win_autostart.desktop_entry_path()


# --------------------------------------------------------------------------- #
#  desktop: capability report
# --------------------------------------------------------------------------- #
def test_is_available_reports_x11_capabilities_and_missing_packages(monkeypatch):
    shell = FakeShell(tools=("wmctrl", "notify-send", "apt-get"), responses={"wmctrl": (0, "", "")})
    x11(monkeypatch, shell)

    report = desktop.is_available()
    assert report["session"] == "x11"
    assert report["wmctrl"] is True
    assert report["xdotool"] is False
    assert report["notify_send"] is True
    assert report["window_control"] is True
    assert report["autostart"] is True
    # xdotool is missing, so the advice must carry a runnable install command.
    assert "sudo apt-get install -y xdotool" in report["advice"]


def test_is_available_marks_window_control_dead_under_wayland(monkeypatch):
    wayland(monkeypatch)
    report = desktop.is_available()
    assert report["session"] == "wayland"
    assert report["wmctrl"] is True          # installed ...
    assert report["window_control"] is False  # ... and useless here
    assert report["global_hotkeys"] is False
    assert any("Wayland" in line for line in report["advice"])


# --------------------------------------------------------------------------- #
#  audio: backend detection
# --------------------------------------------------------------------------- #
PACTL_INFO_PIPEWIRE = (
    "Server String: /run/user/1000/pulse/native\n"
    "Library Protocol Version: 35\n"
    "Server Name: PulseAudio (on PipeWire 1.0.5)\n"
    "Server Version: 15.0.0\n"
    "Default Sink: alsa_output.pci-0000_00_1f.3.analog-stereo\n"
    "Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo\n"
)

PACTL_INFO_PULSE = (
    "Server Name: pulseaudio\n"
    "Server Version: 14.2\n"
    "Default Sink: alsa_output.pci-0000_00_1f.3.analog-stereo\n"
    "Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo\n"
)

APLAY_OUTPUT = (
    "**** List of PLAYBACK Hardware Devices ****\n"
    "card 0: PCH [HDA Intel PCH], device 0: ALC257 Analog [ALC257 Analog]\n"
    "  Subdevices: 1/1\n"
    "  Subdevice #0: subdevice #0\n"
)

ARECORD_OUTPUT = (
    "**** List of CAPTURE Hardware Devices ****\n"
    "card 0: PCH [HDA Intel PCH], device 0: ALC257 Analog [ALC257 Analog]\n"
    "  Subdevices: 1/1\n"
    "card 1: Snowball [Blue Snowball], device 0: USB Audio [USB Audio]\n"
)

PACTL_SOURCES = (
    "0\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tSUSPENDED\n"
    "1\talsa_input.pci-0000_00_1f.3.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"
)

PACTL_SINKS = (
    "0\talsa_output.pci-0000_00_1f.3.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"
)


def pipewire_shell(**extra) -> FakeShell:
    responses = {
        "pactl info": (0, PACTL_INFO_PIPEWIRE, ""),
        "pactl list short sources": (0, PACTL_SOURCES, ""),
        "pactl list short sinks": (0, PACTL_SINKS, ""),
        "pactl get-default-source": (0, "alsa_input.pci-0000_00_1f.3.analog-stereo\n", ""),
        "pactl get-default-sink": (0, "alsa_output.pci-0000_00_1f.3.analog-stereo\n", ""),
        "ldconfig -p": (0, "\tlibportaudio.so.2 (libc6,x86-64) => /usr/lib/libportaudio.so.2\n", ""),
        "id -nG": (0, "testuser audio video\n", ""),
    }
    responses.update(extra)
    return FakeShell(
        tools=("pactl", "aplay", "arecord", "ldconfig", "id", "ffmpeg", "espeak-ng", "apt-get"),
        responses=responses,
    )


def test_backend_detects_pipewire_behind_the_pulse_shim(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())
    assert audio.backend() == "pipewire"
    assert audio.is_available() is True


def test_backend_detects_real_pulseaudio(monkeypatch):
    fake_host(monkeypatch, pipewire_shell(**{"pactl info": (0, PACTL_INFO_PULSE, "")}))
    assert audio.backend() == "pulseaudio"


def test_backend_detects_pipewire_without_pactl(monkeypatch):
    shell = FakeShell(tools=("pw-cli",), responses={"pw-cli info 0": (0, "id 0, type PipeWire", "")})
    fake_host(monkeypatch, shell)
    assert audio.backend() == "pipewire"


def test_backend_falls_back_to_alsa(monkeypatch):
    shell = FakeShell(tools=("aplay",), responses={"aplay -l": (0, APLAY_OUTPUT, "")})
    fake_host(monkeypatch, shell)
    assert audio.backend() == "alsa"


def test_backend_is_none_when_aplay_finds_no_card(monkeypatch):
    shell = FakeShell(tools=("aplay",), responses={"aplay -l": (1, "", "no soundcards found")})
    fake_host(monkeypatch, shell)
    assert audio.backend() == "none"
    assert audio.is_available() is False


# --------------------------------------------------------------------------- #
#  audio: devices
# --------------------------------------------------------------------------- #
def test_pactl_sources_are_parsed_and_monitors_flagged(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())

    inputs = audio.list_input_devices()
    assert [d["name"] for d in inputs] == [
        "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
        "alsa_input.pci-0000_00_1f.3.analog-stereo",
    ]
    assert inputs[0]["monitor"] is True
    assert inputs[1]["monitor"] is False
    assert inputs[1]["state"] == "RUNNING"
    assert inputs[1]["driver"] == "PipeWire"

    outputs = audio.list_output_devices()
    assert len(outputs) == 1
    assert outputs[0]["kind"] == "output"


def test_devices_fall_back_to_arecord_and_aplay(monkeypatch):
    shell = FakeShell(
        tools=("aplay", "arecord"),
        responses={"arecord -l": (0, ARECORD_OUTPUT, ""), "aplay -l": (0, APLAY_OUTPUT, "")},
    )
    fake_host(monkeypatch, shell)

    inputs = audio.list_input_devices()
    assert [d["name"] for d in inputs] == ["hw:0,0", "hw:1,0"]
    assert inputs[1]["description"] == "Blue Snowball — USB Audio"
    assert inputs[0]["driver"] == "alsa"
    assert [d["name"] for d in audio.list_output_devices()] == ["hw:0,0"]


def test_defaults_come_from_pactl(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())
    assert audio.default_input() == "alsa_input.pci-0000_00_1f.3.analog-stereo"
    assert audio.default_output() == "alsa_output.pci-0000_00_1f.3.analog-stereo"


def test_defaults_fall_back_to_pactl_info_on_older_servers(monkeypatch):
    """PulseAudio before 15 has no get-default-source, only `pactl info`."""
    fake_host(
        monkeypatch,
        pipewire_shell(
            **{
                "pactl get-default-source": (1, "", "Unknown command"),
                "pactl get-default-sink": (1, "", "Unknown command"),
            }
        ),
    )
    assert audio.default_input() == "alsa_input.pci-0000_00_1f.3.analog-stereo"
    assert audio.default_output() == "alsa_output.pci-0000_00_1f.3.analog-stereo"


def test_set_default_input(monkeypatch):
    shell = pipewire_shell(**{"pactl set-default-source alsa_input": (0, "", "")})
    fake_host(monkeypatch, shell)

    result = audio.set_default_input("alsa_input.pci-0000_00_1f.3.analog-stereo")
    assert result.ok
    assert shell.ran("pactl set-default-source alsa_input.pci-0000_00_1f.3.analog-stereo")


def test_set_default_input_lists_the_real_names_on_failure(monkeypatch):
    fake_host(monkeypatch, pipewire_shell(**{"pactl set-default-source": (1, "", "No such entity")}))
    result = audio.set_default_input("nope")
    assert result.ok is False
    assert "No such entity" in result.error
    assert "alsa_input.pci-0000_00_1f.3.analog-stereo" in result.error


def test_set_default_input_without_pactl_names_the_package(monkeypatch):
    shell = FakeShell(tools=("apt-get",))
    fake_host(monkeypatch, shell)
    result = audio.set_default_input("anything")
    assert result.ok is False
    assert "pulseaudio-utils" in result.error


def test_set_default_input_rejects_an_empty_name(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())
    assert audio.set_default_input("").ok is False


# --------------------------------------------------------------------------- #
#  audio: diagnosis
# --------------------------------------------------------------------------- #
def test_check_on_a_healthy_pipewire_box(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())
    report = audio.check()

    assert report["backend"] == "pipewire"
    assert report["microphones"] == 1          # the .monitor source does not count
    assert report["portaudio"] is True
    assert report["audio_group"] is True
    assert report["package_manager"] == "apt"
    assert report["problems"] == []
    assert report["ok"] is True


def test_check_names_the_exact_packages_that_are_missing(monkeypatch):
    shell = FakeShell(
        tools=("pactl", "ldconfig", "id", "apt-get"),
        responses={
            "pactl info": (0, PACTL_INFO_PIPEWIRE, ""),
            "pactl list short sources": (0, PACTL_SOURCES, ""),
            "pactl list short sinks": (0, PACTL_SINKS, ""),
            "pactl get-default-source": (0, "alsa_input.x\n", ""),
            "pactl get-default-sink": (0, "alsa_output.x\n", ""),
            "ldconfig -p": (0, "\tlibc.so.6 (libc6,x86-64) => /lib/libc.so.6\n", ""),
            "id -nG": (0, "testuser video\n", ""),
        },
    )
    fake_host(monkeypatch, shell)
    report = audio.check()

    assert report["portaudio"] is False
    assert any("libportaudio2" in p for p in report["problems"])
    assert any("ffmpeg" in p for p in report["problems"])
    assert any("espeak-ng" in p for p in report["problems"])

    command = report["fixes"][0]
    assert command.startswith("sudo apt-get install -y ")
    assert "portaudio19-dev" in command
    assert "ffmpeg" in command
    assert "espeak-ng" in command
    assert report["ok"] is False


def test_check_only_blames_the_audio_group_on_bare_alsa(monkeypatch):
    """On PipeWire the group is irrelevant, and saying otherwise wastes an hour."""
    fake_host(monkeypatch, pipewire_shell(**{"id -nG": (0, "testuser video\n", "")}))
    pipewire_report = audio.check()
    assert pipewire_report["audio_group"] is False
    assert not any("audio' group" in p for p in pipewire_report["problems"])

    alsa = FakeShell(
        tools=("aplay", "arecord", "id", "ldconfig", "apt-get"),
        responses={
            "aplay -l": (0, APLAY_OUTPUT, ""),
            "arecord -l": (0, ARECORD_OUTPUT, ""),
            "id -nG": (0, "testuser video\n", ""),
            "ldconfig -p": (0, "\tlibportaudio.so.2 => /usr/lib/libportaudio.so.2\n", ""),
        },
    )
    fake_host(monkeypatch, alsa)
    alsa_report = audio.check()
    assert alsa_report["backend"] == "alsa"
    assert any("audio' group" in p for p in alsa_report["problems"])
    assert any(fix.startswith("sudo usermod -aG audio") for fix in alsa_report["fixes"])


def test_check_without_a_known_package_manager_says_so(monkeypatch):
    shell = FakeShell(tools=("pactl",), responses={"pactl info": (0, PACTL_INFO_PIPEWIRE, "")})
    fake_host(monkeypatch, shell)
    report = audio.check()

    assert report["package_manager"] is None
    assert any("apt/dnf/pacman/zypper were not found" in fix for fix in report["fixes"])


def test_portaudio_probe_uses_ldconfig(monkeypatch):
    shell = pipewire_shell()
    fake_host(monkeypatch, shell)
    assert audio.portaudio_installed() is True
    assert shell.ran("ldconfig -p")

    fake_host(monkeypatch, pipewire_shell(**{"ldconfig -p": (0, "\tlibc.so.6\n", "")}))
    assert audio.portaudio_installed() is False


def test_in_audio_group_reads_id(monkeypatch):
    fake_host(monkeypatch, pipewire_shell())
    assert audio.in_audio_group() is True

    fake_host(monkeypatch, pipewire_shell(**{"id -nG": (0, "testuser video\n", "")}))
    assert audio.in_audio_group() is False


# --------------------------------------------------------------------------- #
#  Distro package advice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "probe, manager, expected",
    [
        ("apt-get", "apt", "sudo apt-get install -y portaudio19-dev"),
        ("dnf", "dnf", "sudo dnf install -y portaudio-devel"),
        ("pacman", "pacman", "sudo pacman -S --needed portaudio"),
        ("zypper", "zypper", "sudo zypper install -y portaudio-devel"),
    ],
)
def test_package_advice_matches_the_package_manager(monkeypatch, probe, manager, expected):
    fake_host(monkeypatch, FakeShell(tools=(probe,)))
    assert audio.package_manager() == manager
    assert audio.install_command(["portaudio"]) == expected


@pytest.mark.parametrize(
    "manager, expected",
    [
        ("apt", "sudo apt-get install -y ffmpeg libnotify-bin"),
        ("dnf", "sudo dnf install -y ffmpeg-free libnotify"),
        ("pacman", "sudo pacman -S --needed ffmpeg libnotify"),
        ("zypper", "sudo zypper install -y ffmpeg libnotify-tools"),
    ],
)
def test_install_command_preserves_order_and_distro_names(manager, expected):
    assert audio.install_command(["ffmpeg", "libnotify"], manager) == expected


def test_package_advice_is_silent_when_the_manager_is_unknown(monkeypatch):
    fake_host(monkeypatch, FakeShell(tools=()))
    assert audio.package_manager() is None
    assert audio.install_command(["portaudio"]) is None


def test_python_venv_is_only_a_package_on_debian():
    assert audio.package_for("python-venv", "apt") == "python3-venv"
    assert audio.package_for("python-venv", "dnf") is None
    assert audio.install_command(["python-venv"], "dnf") is None
    assert audio.install_command(["python-venv"], "apt") == "sudo apt-get install -y python3-venv"


def test_install_command_deduplicates_and_skips_bundled_packages():
    command = audio.install_command(["portaudio", "portaudio", "python-venv"], "pacman")
    assert command == "sudo pacman -S --needed portaudio"


def test_unknown_capability_yields_nothing():
    assert audio.package_for("nonexistent-capability", "apt") is None
    assert audio.install_command(["nonexistent-capability"], "apt") is None


# --------------------------------------------------------------------------- #
#  Package-level report
# --------------------------------------------------------------------------- #
def test_is_linux_integration_available_never_raises_and_covers_every_area(monkeypatch):
    import jarvis.linux as linux_package

    fake_host(monkeypatch, pipewire_shell())
    report = linux_package.is_linux_integration_available()

    assert set(report) >= {"os", "is_linux", "session", "package_manager",
                           "service", "desktop", "audio"}
    assert report["is_linux"] is True
    assert report["package_manager"] == "apt"
    assert isinstance(report["service"], dict)
    assert isinstance(report["desktop"], dict)
    assert report["audio"]["backend"] == "pipewire"


def test_package_reexports_are_the_real_callables():
    import jarvis.linux as linux_package

    assert linux_package.notify is desktop.notify
    assert linux_package.session_type is desktop.session_type
    assert linux_package.package_manager is audio.package_manager
    assert linux_package.service is service


# --------------------------------------------------------------------------- #
#  install.sh
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
def test_install_sh_is_syntactically_valid_bash():
    result = subprocess.run(
        [shutil.which("bash"), "-n", str(INSTALL_SH)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def installer_text() -> str:
    return INSTALL_SH.read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", ["--min", "--lean", "--full", "--no-voice", "--venv",
                                  "--service", "--vllm", "--model"])
def test_install_sh_accepts_every_documented_flag(flag):
    text = installer_text()
    assert f"{flag})" in text, f"{flag} is not handled in the argument loop"
    assert flag in text.split("set -euo pipefail")[0], f"{flag} is not in the help header"


@pytest.mark.parametrize(
    "manager, package",
    [
        ("portaudio:apt", "portaudio19-dev"),
        ("portaudio:dnf", "portaudio-devel"),
        ("portaudio:pacman", "portaudio"),
        ("portaudio:zypper", "portaudio-devel"),
        ("libnotify:apt", "libnotify-bin"),
        ("libnotify:zypper", "libnotify-tools"),
        ("ffmpeg:dnf", "ffmpeg-free"),
        ("venv:apt", "python3-venv"),
    ],
)
def test_install_sh_maps_packages_per_distro(manager, package):
    text = installer_text()
    assert f"{manager})" in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{manager})"))
    assert package in line


def extract_shell_function(name: str) -> str:
    """Pull one function out of install.sh so bash can run it in isolation."""
    text = installer_text()
    body = text.split(f"{name}() {{\n", 1)[1].split("\n}\n", 1)[0]
    return f"{name}() {{\n{body}\n}}\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
@pytest.mark.parametrize(
    "manager, expected",
    [
        ("apt", {"portaudio": "portaudio19-dev", "ffmpeg": "ffmpeg",
                 "libnotify": "libnotify-bin", "pactl": "pulseaudio-utils",
                 "venv": "python3-venv", "espeak": "espeak-ng", "alsa": "alsa-utils"}),
        ("dnf", {"portaudio": "portaudio-devel", "ffmpeg": "ffmpeg-free",
                 "libnotify": "libnotify", "pactl": "pulseaudio-utils",
                 "venv": "", "espeak": "espeak-ng", "alsa": "alsa-utils"}),
        ("pacman", {"portaudio": "portaudio", "ffmpeg": "ffmpeg",
                    "libnotify": "libnotify", "pactl": "libpulse",
                    "venv": "", "espeak": "espeak-ng", "alsa": "alsa-utils"}),
        ("zypper", {"portaudio": "portaudio-devel", "ffmpeg": "ffmpeg",
                    "libnotify": "libnotify-tools", "pactl": "pulseaudio-utils",
                    "venv": "", "espeak": "espeak-ng", "alsa": "alsa-utils"}),
    ],
)
def test_install_sh_package_mapping_actually_runs(tmp_path: Path, manager, expected):
    """Run the installer's own pkg_for() under bash, one distro at a time.

    Grepping the source proves the lines exist; running them proves the case
    patterns match, which is where a shell ``case`` typically goes wrong.
    """
    script = tmp_path / "pkg_for.sh"
    script.write_text(
        "set -eu\n"
        f'PM="{manager}"\n'
        + extract_shell_function("pkg_for")
        + 'for cap in portaudio ffmpeg espeak libnotify pactl alsa venv; do\n'
        '    printf "%s=%s\\n" "$cap" "$(pkg_for "$cap")"\n'
        "done\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [shutil.which("bash"), str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr

    produced = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    assert produced == expected


def test_install_sh_detects_all_four_package_managers():
    text = installer_text()
    for probe, command in (
        ("apt-get", "sudo apt-get install -y"),
        ("dnf", "sudo dnf install -y"),
        ("pacman", "sudo pacman -S --needed"),
        ("zypper", "sudo zypper install -y"),
    ):
        assert f'command -v {probe} >/dev/null 2>&1' in text
        assert command in text


def test_install_sh_mentions_linger_and_the_service_unit():
    text = installer_text()
    assert "loginctl enable-linger" in text
    assert "jarvis.service" in text
    assert "systemctl --user" in text


def test_install_sh_warns_about_wayland():
    text = installer_text()
    assert "wayland" in text.lower()
    assert "ydotool" in text
    assert "Xorg" in text


def test_install_sh_treats_vllm_as_a_separate_linux_only_install():
    text = installer_text()
    vllm_section = text.split("#  vLLM")[1].split("#  Voice model")[0]
    assert "Linux-only" in vllm_section
    assert "pip install vllm" in vllm_section
    assert "VLLM_TARGET_DEVICE=cpu" in vllm_section


def test_install_sh_writes_the_model_into_config_yaml():
    text = installer_text()
    assert "config.example.yaml" in text
    assert "ollama_model" in text
    assert "JARVIS_SET_MODEL" in text


def test_install_sh_only_tells_you_to_run_real_subcommands():
    """Every 'jarvis <verb>' the installer prints must exist in the CLI.

    Printed advice is the first thing the owner types on a fresh box; a
    subcommand that was never implemented turns the installer's last screen
    into a dead end.
    """
    import re

    cli_source = (REPO_ROOT / "jarvis" / "cli.py").read_text(encoding="utf-8")
    known = set(re.findall(r'add_parser\(\s*"([a-z][a-z0-9_-]*)"', cli_source))
    assert known, "could not read the subcommand list out of jarvis/cli.py"

    # Comments are not advice — nobody types them. Scanning them anyway made
    # this test fire on ordinary English: a comment reading ".../bin/jarvis so
    # that it works" parses as the subcommand "so". Strip comment lines and
    # keep the check strict on everything that is executed or printed.
    executable_text = "\n".join(
        line for line in installer_text().splitlines()
        if not line.lstrip().startswith("#")
    )

    used = set(
        re.findall(r"(?:\./jarvis|/jarvis|-m jarvis)\s+([a-z][a-z0-9_-]*)", executable_text)
    )
    assert used, "the installer no longer mentions any jarvis subcommand"
    assert used <= known, f"install.sh names subcommands that do not exist: {sorted(used - known)}"


def test_install_sh_runs_nothing_destructive():
    """No delete/format/overwrite of anything outside this directory, ever."""
    text = installer_text()
    for pattern in ("rm -rf", "rm -f /", "mkfs", "dd if=", "> /dev/sd", ":(){", "chown -R /"):
        assert pattern not in text, f"install.sh contains {pattern!r}"

    # sudo is only ever *printed* as advice, never executed.
    for number, line in enumerate(text.splitlines(), start=1):
        assert not line.strip().startswith("sudo "), f"line {number} executes sudo"
