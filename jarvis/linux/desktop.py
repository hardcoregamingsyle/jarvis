"""The Linux counterpart of :mod:`jarvis.win`: notifications, autostart, windows.

Windows gets a tray icon, global hotkeys, a Run-key autostart entry and toast
notifications.  Linux can do three of those four honestly, and this module is
careful about which.

**Wayland is the caveat that matters.**  ``wmctrl`` and ``xdotool`` are X11
clients.  Under Wayland — the default on current Fedora and Ubuntu GNOME —
they see only XWayland windows, which on a modern desktop is usually none of
them, and window activation either fails or *silently does nothing*.  A tool
that reports success after doing nothing is worse than one that refuses, so
every window operation here checks :func:`session_type` first and returns an
explicit failure naming Wayland, with the three real ways forward:

1. log out and choose "GNOME on Xorg" at the login screen (everything below
   then works as on any X11 desktop);
2. ``ydotool`` — a uinput-based input injector that works under Wayland, needs
   the ``ydotoold`` daemon and membership of the ``input`` group, and can
   synthesise keystrokes and clicks but **cannot enumerate or focus windows**,
   because Wayland gives no client that information;
3. a GNOME Shell extension that exposes window control over D-Bus (the
   "Window Calls" family), driven with ``gdbus`` — the only route to real
   window management on a stock Wayland GNOME session.

Global hotkeys have the same shape of problem: under Wayland no ordinary
client can grab a key combination globally, by design.  The working answer is
to let the compositor own the binding — a custom shortcut in GNOME Settings
that runs a JARVIS command — and :func:`global_hotkeys` says exactly that.

Honesty note: developed and unit-tested on a Windows box with faked binaries
and faked environment variables.  The command lines and the Wayland logic are
verified; no real ``wmctrl``, ``notify-send`` or GNOME session has run against
this code.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import platform_utils
from ..core.contracts import ToolResult
from .audio import install_command

log = logging.getLogger(__name__)

_COMMAND_TIMEOUT = 10.0
_NOTIFY_TIMEOUT = 15.0

#: Windows returned by one enumeration.  Resource management, not policy: an
#: unbounded list would be fed straight into a model's context window.
MAX_WINDOWS = 200

#: Same file name jarvis.win.autostart writes on Linux, deliberately: two
#: autostart entries would launch two assistants that fight over the microphone.
APP_NAME = "JARVIS"
DESKTOP_FILE_NAME = "jarvis.desktop"

VALID_URGENCIES = ("low", "normal", "critical")


# --------------------------------------------------------------------------- #
#  Session detection
# --------------------------------------------------------------------------- #
def session_type() -> str:
    """``"wayland"``, ``"x11"``, ``"tty"`` or ``"unknown"``.

    ``XDG_SESSION_TYPE`` is authoritative when the display manager set it.
    When it is missing or says something unexpected — common over SSH, in a
    container, or under a bare ``startx`` — fall back to the display sockets:
    ``WAYLAND_DISPLAY`` means Wayland, ``DISPLAY`` means X11 (or XWayland).
    ``unknown`` is returned rather than guessed, because every caller here
    treats "unknown" as "try it and report what actually happened".

    This reads the environment and nothing else — it does not check that the
    host is Linux, so on a Windows box where some shell has exported ``DISPLAY``
    (Git Bash does, with a placeholder value) it will say ``x11``.  Every caller
    in this package tests ``IS_LINUX`` separately; treat the answer as "what the
    environment claims", not as proof of a display server.
    """
    declared = (os.environ.get("XDG_SESSION_TYPE") or "").strip().lower()
    if declared in ("wayland", "x11", "tty"):
        return declared
    if declared == "mir":  # rare, but it is not X11 and must not be treated as such
        return "wayland"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def is_wayland() -> bool:
    """True when this is a Wayland session and X11 window tools will not work."""
    return session_type() == "wayland"


def _wayland_failure(operation: str) -> str:
    """The one message every blocked window operation returns."""
    return (
        f"cannot {operation}: this is a Wayland session "
        f"(XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE') or 'unset'}, "
        f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY') or 'unset'}). "
        "wmctrl and xdotool are X11 clients: under Wayland they can see only "
        "XWayland windows and window activation fails or silently does nothing, "
        "so JARVIS refuses rather than reporting a success that did not happen. "
        "Three ways forward: (1) log out and pick 'GNOME on Xorg' at the login "
        "screen, after which wmctrl/xdotool work normally; (2) install ydotool "
        "(uinput-based, needs the ydotoold daemon and the 'input' group) — it can "
        "type and click under Wayland but cannot list or focus windows; "
        "(3) install a GNOME Shell extension that exposes window control over "
        "D-Bus (the 'Window Calls' family) and drive it with gdbus, which is the "
        "only real window management available on a stock Wayland GNOME."
    )


def _missing_tools_failure(operation: str, tools: List[str]) -> str:
    hint = install_command(tools)
    text = (
        f"cannot {operation}: none of {', '.join(tools)} is installed. "
        "These are the X11 window-management helpers JARVIS shells out to."
    )
    return text + (f" Install with: {hint}" if hint else "")


# --------------------------------------------------------------------------- #
#  Notifications
# --------------------------------------------------------------------------- #
def _gvariant_string(text: str) -> str:
    """Quote a string as GVariant text for ``gdbus call``.

    ``gdbus`` parses each parameter as GVariant, so a title beginning with
    ``[`` or ``{`` would be read as an array or dictionary and the call would
    fail with a type error.  Quoting removes the ambiguity.
    """
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _clamp_urgency(urgency: Any) -> str:
    value = str(urgency or "normal").strip().lower()
    return value if value in VALID_URGENCIES else "normal"


def _notify_send(title: str, message: str, urgency: str, icon: Optional[str]) -> bool:
    exe = platform_utils.which("notify-send")
    if not exe:
        return False
    argv = [exe, "-a", APP_NAME, "-u", urgency]
    if icon:
        argv += ["-i", str(icon)]
    argv += [title, message]
    result = platform_utils.run_command(argv, timeout=_NOTIFY_TIMEOUT)
    if not result.ok:
        log.debug("notify-send failed (rc=%s): %s", result.returncode, result.stderr.strip()[:200])
    return bool(result.ok)


#: freedesktop urgency hint values, for the gdbus path where ``-u`` does not exist.
_URGENCY_VALUES = {"low": 0, "normal": 1, "critical": 2}


def _gdbus_notify(title: str, message: str, urgency: str, icon: Optional[str]) -> bool:
    """Raise the notification straight over D-Bus, for boxes without libnotify."""
    exe = platform_utils.which("gdbus")
    if not exe:
        return False
    timeout_ms = 0 if urgency == "critical" else 5000
    hints = "{'urgency': <byte " + str(_URGENCY_VALUES.get(urgency, 1)) + ">}"
    argv = [
        exe, "call", "--session",
        "--dest", "org.freedesktop.Notifications",
        "--object-path", "/org/freedesktop/Notifications",
        "--method", "org.freedesktop.Notifications.Notify",
        _gvariant_string(APP_NAME),
        "0",
        _gvariant_string(icon or ""),
        _gvariant_string(title),
        _gvariant_string(message),
        "[]",
        hints,
        str(timeout_ms),
    ]
    result = platform_utils.run_command(argv, timeout=_NOTIFY_TIMEOUT)
    if not result.ok:
        log.debug("gdbus notify failed (rc=%s): %s", result.returncode, result.stderr.strip()[:200])
    return bool(result.ok)


def notify(
    title: str,
    message: str,
    *,
    urgency: str = "normal",
    icon: Optional[str] = None,
) -> str:
    """Show a desktop notification; return the channel that delivered it.

    ``"notify-send"``, then ``"gdbus"`` (same D-Bus interface, no libnotify
    needed), then ``"log"`` — which cannot fail, so a missing notification
    daemon never costs the caller anything.  Never raises and never blocks
    longer than the per-channel timeout: a notification that hangs the
    assistant is worse than no notification.

    ``urgency`` is one of ``low``/``normal``/``critical``; anything else is
    treated as ``normal``.  A ``critical`` notification does not time out, which
    is the behaviour the freedesktop spec defines and most shells honour.
    """
    title = str(title or APP_NAME)
    message = str(message or "")
    level = _clamp_urgency(urgency)

    if platform_utils.IS_LINUX:
        for name, attempt in (
            ("notify-send", lambda: _notify_send(title, message, level, icon)),
            ("gdbus", lambda: _gdbus_notify(title, message, level, icon)),
        ):
            try:
                if attempt():
                    return name
            except Exception as exc:  # noqa: BLE001 - always fall through to the next
                log.debug("notification channel %s failed: %s", name, exc)

    log.info("notification | %s | %s | %s", level, title, message)
    return "log"


# --------------------------------------------------------------------------- #
#  Autostart (XDG .desktop entry)
# --------------------------------------------------------------------------- #
def autostart_dir() -> Path:
    """``$XDG_CONFIG_HOME/autostart`` (``~/.config/autostart`` by default)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path(os.path.expanduser("~")) / ".config")
    return Path(base).expanduser() / "autostart"


def autostart_path() -> Path:
    """Full path of the autostart entry this module manages."""
    return autostart_dir() / DESKTOP_FILE_NAME


def default_command() -> str:
    """``<python> -m jarvis voice`` for the interpreter running right now."""
    interpreter = sys.executable or "python3"
    quoted = f'"{interpreter}"' if " " in interpreter else interpreter
    return f"{quoted} -m jarvis voice"


def desktop_entry_text(command: str) -> str:
    """The ``.desktop`` file contents for ``command``.

    ``X-GNOME-Autostart-Delay`` is not set: a voice assistant that starts a few
    seconds late is fine, but one that starts before the sound server can be
    left with no working audio device.  If JARVIS comes up deaf at login, add
    ``X-GNOME-Autostart-Delay=10`` here, or use the systemd user service
    (:mod:`jarvis.linux.service`), which can be ordered after PipeWire properly.
    """
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        f"Name={APP_NAME}\n"
        "Comment=Local voice assistant\n"
        f"Exec={command}\n"
        "Terminal=false\n"
        "NoDisplay=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def autostart_command() -> Optional[str]:
    """The command currently configured to run at login, or ``None``."""
    try:
        text = autostart_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        if line.startswith("Exec="):
            return line[len("Exec="):].strip() or None
    return None


def autostart_is_enabled() -> bool:
    """True when a JARVIS autostart entry exists and names a command."""
    try:
        return bool(autostart_command())
    except Exception:  # noqa: BLE001 - a status query never raises
        log.debug("autostart probe failed", exc_info=True)
        return False


def autostart_enable(command: Optional[str] = None) -> ToolResult:
    """Write ``~/.config/autostart/jarvis.desktop`` so JARVIS starts at login.

    Every mainstream desktop honours this directory, Wayland included — it is
    the compositor that reads it, not an X11 client, so this is one Linux
    facility that is *not* affected by the session type.

    Note this is the same file :mod:`jarvis.win.autostart` writes when it runs
    on Linux; that is intentional, so enabling autostart twice cannot produce
    two assistants competing for the microphone.
    """
    text = default_command() if command is None else str(command).strip()
    if not text or "\n" in text or "\r" in text:
        return ToolResult.failure("refusing to write an empty or multi-line Exec= line")

    path = autostart_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(desktop_entry_text(text), encoding="utf-8")
        try:
            path.chmod(0o755)
        except OSError:
            log.debug("could not chmod %s", path, exc_info=True)
    except OSError as exc:
        return ToolResult.failure(f"could not write {path}: {exc}")
    return ToolResult.success({"path": str(path), "command": text})


def autostart_disable() -> ToolResult:
    """Remove the autostart entry.  Already-absent counts as success."""
    path = autostart_path()
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        return ToolResult.failure(f"could not remove {path}: {exc}")
    return ToolResult.success({"path": str(path), "enabled": False})


# --------------------------------------------------------------------------- #
#  Windows
# --------------------------------------------------------------------------- #
def _x11_guard(operation: str) -> Optional[ToolResult]:
    """The shared precondition for every window operation."""
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(
            f"cannot {operation}: jarvis.linux.desktop is Linux-only and this host is "
            f"{platform_utils.os_name()}. Use jarvis.tools.window_tools instead."
        )
    if is_wayland():
        return ToolResult.failure(_wayland_failure(operation))
    return None


def _parse_wmctrl(text: str) -> List[Dict[str, Any]]:
    """Parse ``wmctrl -lpG``: id, desktop, pid, x, y, w, h, host, title."""
    windows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split(None, 8)
        if len(fields) < 8:
            continue
        try:
            entry = {
                "id": fields[0],
                "desktop": int(fields[1]),
                "pid": int(fields[2]),
                "x": int(fields[3]),
                "y": int(fields[4]),
                "width": int(fields[5]),
                "height": int(fields[6]),
                "host": fields[7],
                "title": fields[8] if len(fields) > 8 else "",
            }
        except ValueError:
            continue
        windows.append(entry)
        if len(windows) >= MAX_WINDOWS:
            break
    return windows


def list_windows() -> ToolResult:
    """Enumerate open windows via ``wmctrl -lpG``, falling back to ``xdotool``.

    Fails explicitly under Wayland (see the module docstring) and names the
    package to install when neither helper is present.
    """
    blocked = _x11_guard("list windows")
    if blocked is not None:
        return blocked

    wmctrl = platform_utils.which("wmctrl")
    if wmctrl:
        result = platform_utils.run_command([wmctrl, "-lpG"], timeout=_COMMAND_TIMEOUT)
        if result.ok:
            return ToolResult.success(_parse_wmctrl(result.stdout))
        log.debug("wmctrl -lpG failed: %s", result.stderr.strip()[:200])

    xdotool = platform_utils.which("xdotool")
    if xdotool:
        found = platform_utils.run_command(
            [xdotool, "search", "--onlyvisible", "--name", "."], timeout=_COMMAND_TIMEOUT
        )
        if found.ok:
            windows: List[Dict[str, Any]] = []
            for window_id in found.stdout.split()[:MAX_WINDOWS]:
                name = platform_utils.run_command(
                    [xdotool, "getwindowname", window_id], timeout=_COMMAND_TIMEOUT
                )
                windows.append(
                    {
                        "id": window_id,
                        "title": name.stdout.strip() if name.ok else "",
                        "backend": "xdotool",
                    }
                )
            return ToolResult.success(windows)
        log.debug("xdotool search failed: %s", found.stderr.strip()[:200])

    return ToolResult.failure(_missing_tools_failure("list windows", ["wmctrl", "xdotool"]))


def active_window() -> ToolResult:
    """The window with input focus, via ``xdotool getactivewindow``.

    ``wmctrl`` has no equivalent query, so without ``xdotool`` this reports a
    failure rather than picking a plausible-looking window from the list.
    """
    blocked = _x11_guard("read the active window")
    if blocked is not None:
        return blocked

    xdotool = platform_utils.which("xdotool")
    if not xdotool:
        return ToolResult.failure(
            _missing_tools_failure("read the active window", ["xdotool"])
            + " (wmctrl cannot report the focused window.)"
        )

    found = platform_utils.run_command([xdotool, "getactivewindow"], timeout=_COMMAND_TIMEOUT)
    if not found.ok or not found.stdout.strip():
        detail = found.stderr.strip() or f"exit {found.returncode}"
        return ToolResult.failure(f"xdotool getactivewindow failed: {detail}")

    window_id = found.stdout.split()[0]
    name = platform_utils.run_command(
        [xdotool, "getwindowname", window_id], timeout=_COMMAND_TIMEOUT
    )
    return ToolResult.success(
        {
            "id": window_id,
            "title": name.stdout.strip() if name.ok else "",
            "backend": "xdotool",
        }
    )


def focus_window(title: str) -> ToolResult:
    """Raise and focus the window whose title contains ``title``.

    ``wmctrl -a`` first (it matches substrings and asks the window manager
    properly), then ``xdotool windowactivate``.  ``wmctrl`` exits non-zero when
    nothing matched, which is reported as a failure naming the title rather
    than as a silent success.
    """
    needle = str(title or "").strip()
    if not needle:
        return ToolResult.failure("no window title given")

    blocked = _x11_guard("focus a window")
    if blocked is not None:
        return blocked

    wmctrl = platform_utils.which("wmctrl")
    if wmctrl:
        result = platform_utils.run_command([wmctrl, "-a", needle], timeout=_COMMAND_TIMEOUT)
        if result.ok:
            return ToolResult.success({"title": needle, "backend": "wmctrl"})
        log.debug("wmctrl -a failed: %s", result.stderr.strip()[:200])

    xdotool = platform_utils.which("xdotool")
    if xdotool:
        result = platform_utils.run_command(
            [xdotool, "search", "--name", needle, "windowactivate", "%1"],
            timeout=_COMMAND_TIMEOUT,
        )
        if result.ok:
            return ToolResult.success({"title": needle, "backend": "xdotool"})
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return ToolResult.failure(f"no window matched {needle!r} (xdotool: {detail})")

    if wmctrl:
        return ToolResult.failure(
            f"no window matched {needle!r} (wmctrl -a found nothing). "
            "Install xdotool for a second matching strategy."
        )
    return ToolResult.failure(_missing_tools_failure("focus a window", ["wmctrl", "xdotool"]))


# --------------------------------------------------------------------------- #
#  Global hotkeys
# --------------------------------------------------------------------------- #
def global_hotkeys() -> ToolResult:
    """Whether a global (system-wide) hotkey can be grabbed on this session.

    Success on X11 when ``xdotool``/an X connection is available.  Under
    Wayland this always fails, and that is not a missing feature to work
    around: the protocol deliberately gives no client the ability to grab keys
    it does not have focus for, which is why every Wayland-native app asks the
    compositor instead.  The failure text gives the route that does work.
    """
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(
            f"this host is {platform_utils.os_name()}; use jarvis.win.HotkeyManager"
        )
    session = session_type()
    if session == "wayland":
        return ToolResult.failure(
            "global hotkeys cannot be grabbed under Wayland by an ordinary client — "
            "the protocol does not allow it, so python 'keyboard'/'pynput' hooks and "
            "xdotool all fail or see nothing. Let the compositor own the binding "
            "instead: GNOME Settings -> Keyboard -> View and Customise Shortcuts -> "
            "Custom Shortcuts, bound to a command such as "
            f"'{default_command()}'. KDE Plasma has the same "
            "under System Settings -> Shortcuts. The other option is reading "
            "/dev/input directly with python-evdev, which needs membership of the "
            "'input' group and grabs keys before the compositor sees them."
        )
    if session == "tty":
        return ToolResult.failure(
            "this is a text console with no display server, so there is no global "
            "hotkey to grab. Use push-to-talk on stdin, or the systemd user service."
        )
    backends = {
        "xdotool": bool(platform_utils.which("xdotool")),
        "display": bool(os.environ.get("DISPLAY")),
    }
    if not backends["display"]:
        return ToolResult.failure(
            "DISPLAY is unset, so no X server can be reached and no key can be "
            "grabbed. If JARVIS is running as a systemd user service, run "
            "'systemctl --user import-environment DISPLAY XAUTHORITY' from a "
            "desktop session and restart it."
        )
    return ToolResult.success({"session": session, "backends": backends})


# --------------------------------------------------------------------------- #
#  Capability report
# --------------------------------------------------------------------------- #
def is_available() -> Dict[str, Any]:
    """Which desktop-integration capabilities actually exist on this machine.

    Always a dict, never raises, safe to call on Windows (where everything is
    reported absent).  ``advice`` holds the exact commands for whatever is
    missing, so ``jarvis doctor`` can print it verbatim.
    """
    report: Dict[str, Any] = {
        "os": platform_utils.os_name(),
        "session": "unknown",
        "notify_send": False,
        "gdbus": False,
        "notifications": True,          # the log channel always works
        "wmctrl": False,
        "xdotool": False,
        "ydotool": False,
        "window_control": False,
        "global_hotkeys": False,
        "autostart": False,
        "autostart_enabled": False,
        "autostart_path": str(autostart_path()),
        "advice": [],
    }
    advice: List[str] = report["advice"]

    try:
        report["session"] = session_type()
        report["autostart_enabled"] = autostart_is_enabled()

        if not platform_utils.IS_LINUX:
            advice.append(
                f"this host is {platform_utils.os_name()}; jarvis.linux.desktop "
                "reports everything absent by design. See jarvis.win for the "
                "Windows equivalents."
            )
            return report

        report["autostart"] = True
        for tool in ("notify-send", "gdbus", "wmctrl", "xdotool", "ydotool"):
            report[tool.replace("-", "_")] = bool(platform_utils.which(tool))

        report["window_control"] = bool(
            not is_wayland() and (report["wmctrl"] or report["xdotool"])
        )
        hotkeys = global_hotkeys()
        report["global_hotkeys"] = bool(hotkeys.ok)

        missing = [
            name
            for name, key in (("wmctrl", "wmctrl"), ("xdotool", "xdotool"),
                              ("libnotify", "notify_send"))
            if not report[key]
        ]
        command = install_command(missing)
        if command:
            advice.append(command)

        if is_wayland():
            advice.append(_wayland_failure("control windows"))
        if hotkeys.error:
            advice.append(hotkeys.error)
        if not report["autostart_enabled"]:
            advice.append(
                "JARVIS is not set to start at login. Either "
                "jarvis.linux.desktop.autostart_enable() for a desktop-session "
                "entry, or jarvis.linux.service.install() for a systemd user "
                "service that also survives logout."
            )
    except Exception as exc:  # noqa: BLE001 - a capability report never raises
        log.debug("desktop capability probe failed", exc_info=True)
        advice.append(f"the capability probe itself failed: {type(exc).__name__}: {exc}")

    return report


__all__ = [
    "APP_NAME",
    "DESKTOP_FILE_NAME",
    "MAX_WINDOWS",
    "VALID_URGENCIES",
    "session_type",
    "is_wayland",
    "notify",
    "autostart_dir",
    "autostart_path",
    "autostart_command",
    "autostart_enable",
    "autostart_disable",
    "autostart_is_enabled",
    "desktop_entry_text",
    "default_command",
    "list_windows",
    "active_window",
    "focus_window",
    "global_hotkeys",
    "is_available",
]
