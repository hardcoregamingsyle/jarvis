"""Run JARVIS as a systemd **user** service.

Never a system service, never ``sudo``.  A system unit runs as root outside any
login session, which means no ``XDG_RUNTIME_DIR``, no PipeWire/PulseAudio
socket, and therefore no microphone and no speakers — the exact opposite of
what a voice assistant needs.  Everything here writes to
``~/.config/systemd/user/`` and talks to ``systemctl --user``.

Two facts about user services decide whether this actually works, and both are
handled explicitly:

**Lingering.**  A user manager is started when you log in and torn down when
your last session ends.  Close the lid, log out, or reboot into the login
screen and your "always on" assistant is simply gone — with no error anywhere,
which is why this is the single most common "it stopped working" report.  The
fix is ``loginctl enable-linger $USER``, which tells systemd to start your user
manager at boot and keep it after logout.  :func:`status` detects the current
state with ``loginctl show-user`` and says plainly whether that command is
needed.  This module never runs it for you: it is the one step that changes
system state outside your home directory.

**The audio session.**  Because this is a user unit, systemd sets
``XDG_RUNTIME_DIR`` itself, so the PipeWire/PulseAudio socket at
``$XDG_RUNTIME_DIR/pulse/native`` is reachable with no extra configuration.
That is the whole reason for the user/system distinction above.  The unit is
ordered ``After=`` the sound services so the socket exists before JARVIS opens
a microphone, and installed ``WantedBy=default.target`` rather than
``graphical-session.target`` — ``default.target`` is reached whenever the user
manager starts, with or without a desktop login, so combined with lingering the
assistant survives logout.

Honesty note: this module is written and unit-tested on a Windows development
box against a faked ``systemctl``.  The unit text, the argument lists and the
control flow are verified; the behaviour of a real systemd on the target laptop
is not.  Treat the first ``jarvis linux service status`` on Linux as the real
acceptance test.
"""

from __future__ import annotations

import getpass
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import platform_utils
from ..core.contracts import ToolResult

log = logging.getLogger(__name__)


#: Unit file name.  Stable, so install() is idempotent and enable() finds it.
UNIT_NAME = "jarvis.service"
DEFAULT_DESCRIPTION = "JARVIS voice assistant"

#: systemctl is fast, but a hung D-Bus call must not hang the assistant.
_CONTROL_TIMEOUT = 20.0
_QUERY_TIMEOUT = 10.0
_LOG_TIMEOUT = 30.0

#: Journal lines a single logs() call may return.  Not a permission boundary —
#: a runaway journal dump would pin megabytes of text in memory and, worse, get
#: fed to a language model.
MAX_LOG_LINES = 5000


# --------------------------------------------------------------------------- #
#  Paths and the unit text
# --------------------------------------------------------------------------- #
def unit_dir() -> Path:
    """``$XDG_CONFIG_HOME/systemd/user`` (``~/.config/systemd/user`` by default)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path(os.path.expanduser("~")) / ".config")
    return Path(base).expanduser() / "systemd" / "user"


def unit_path() -> Path:
    """Full path of the unit file this module manages."""
    return unit_dir() / UNIT_NAME


def quote_exec(value: str) -> str:
    """Quote one ``ExecStart`` token for systemd.

    systemd splits ``ExecStart`` with shell-like rules, so a path such as
    ``/opt/My Apps/venv/bin/python`` must be quoted or systemd tries to execute
    ``/opt/My`` and the unit fails at every start with a message most people
    never see.  Quoting a path that does not need it is harmless, so it is done
    unconditionally.  Embedded double quotes are dropped rather than escaped:
    they cannot appear in a legitimate interpreter path.
    """
    text = str(value or "").replace('"', "")
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text
    return '"' + text + '"'


def default_command() -> str:
    """``"<python>" -m jarvis voice`` for the interpreter running right now.

    On the dedicated Linux box this resolves to the project virtualenv's
    interpreter, which is exactly what the unit should run: it carries the
    installed dependencies without any activation step.
    """
    interpreter = sys.executable or "/usr/bin/python3"
    return f"{quote_exec(interpreter)} -m jarvis voice"


def _clean_command(command: Optional[str]) -> Optional[str]:
    """Validate a caller-supplied command; ``None`` means "use the default"."""
    if command is None:
        return default_command()
    text = str(command).strip()
    if not text or "\n" in text or "\r" in text:
        return None
    return text


def exec_start_is_absolute(command: str) -> bool:
    """True when the first token of ``command`` is an absolute path.

    systemd rejects a relative ``ExecStart`` outright ("Executable path is not
    absolute"), and the unit then fails to load with an error that only appears
    in ``systemctl --user status``.  Checked before writing rather than after.
    """
    text = str(command or "").strip()
    if not text:
        return False
    if text.startswith('"'):
        end = text.find('"', 1)
        first = text[1:end] if end > 1 else ""
    else:
        first = text.split()[0]
    # Absolute for systemd means POSIX-absolute.  Also accept the '-'/'@'/':'
    # prefixes systemd allows before the path.
    first = first.lstrip("-@:+!")
    return first.startswith("/")


def unit_text(
    command: str,
    *,
    description: str = DEFAULT_DESCRIPTION,
    restart: bool = True,
) -> str:
    """Render the systemd unit file.

    ``restart=True`` yields ``Restart=always``: on a dedicated box an assistant
    that exits for any reason should come back.  The start-limit settings below
    it are resource management, not policy — without them a unit that dies
    immediately (a bad model path, say) would respawn forever and burn the CPU
    the assistant is supposed to be using.  systemd gives up after
    ``StartLimitBurst`` failures and leaves the unit visibly ``failed``.

    Use ``restart=False`` while debugging a crash loop, or edit the written file
    to ``Restart=on-failure`` if you want a clean ``exit 0`` to stay stopped.
    """
    restart_line = "Restart=always" if restart else "Restart=no"
    return "\n".join(
        [
            "# Written by JARVIS (jarvis.linux.service). Safe to edit by hand;",
            "# re-running the installer overwrites it.",
            "[Unit]",
            f"Description={description}",
            "# Ordering only, no requirement: naming a unit that does not exist in",
            "# After= is harmless, so this line is correct on PipeWire and on",
            "# PulseAudio boxes alike.",
            "After=pipewire.service pulseaudio.service sound.target",
            "StartLimitIntervalSec=120",
            "StartLimitBurst=5",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={command}",
            restart_line,
            "RestartSec=5",
            "TimeoutStopSec=20",
            "# Unbuffered, or the journal shows nothing until the process dies.",
            "Environment=PYTHONUNBUFFERED=1",
            "# Audio needs no configuration here: this is a *user* unit, so the",
            "# user manager exports XDG_RUNTIME_DIR and the PipeWire/PulseAudio",
            "# socket at $XDG_RUNTIME_DIR/pulse/native is already reachable.",
            "# X11 helpers (wmctrl/xdotool) additionally need DISPLAY, which a",
            "# unit started before login will not have; run",
            "#   systemctl --user import-environment DISPLAY XAUTHORITY WAYLAND_DISPLAY",
            "# from a desktop session, or set Environment=DISPLAY=:0 below.",
            "",
            "[Install]",
            "# default.target, not graphical-session.target: default.target is",
            "# reached whenever the user manager runs, so with lingering enabled",
            "# JARVIS keeps going with nobody logged in.",
            "WantedBy=default.target",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
#  Talking to systemd
# --------------------------------------------------------------------------- #
def _systemctl() -> Optional[str]:
    return platform_utils.which("systemctl") if platform_utils.IS_LINUX else None


def _journalctl() -> Optional[str]:
    return platform_utils.which("journalctl") if platform_utils.IS_LINUX else None


def _run_systemctl(args: List[str], *, timeout: float = _CONTROL_TIMEOUT):
    """Run ``systemctl --user <args>``; returns None when systemctl is absent."""
    exe = _systemctl()
    if not exe:
        return None
    return platform_utils.run_command([exe, "--user", *args], timeout=timeout)


def _unavailable() -> str:
    """The reason systemd cannot be used here, as one actionable sentence."""
    if not platform_utils.IS_LINUX:
        return (
            f"systemd user services are Linux-only; this host is "
            f"{platform_utils.os_name()}. On Windows use 'jarvis.win.autostart' instead."
        )
    if not _systemctl():
        return (
            "systemctl was not found on PATH, so this system is not running systemd. "
            "Use your init system's own user-service mechanism, or start JARVIS from "
            "~/.config/autostart (see jarvis.linux.desktop.autostart_enable)."
        )
    return (
        "the systemd user bus is not reachable. This usually means XDG_RUNTIME_DIR is "
        "unset (common over 'sudo su' or a bare SSH session). Log in as the target user "
        "directly, or run: export XDG_RUNTIME_DIR=/run/user/$(id -u)"
    )


def is_available() -> bool:
    """True when systemd is present *and* this process can reach the user bus.

    Both halves matter.  ``systemctl`` exists on plenty of machines where
    ``systemctl --user`` fails with "Failed to connect to bus" — inside a
    container, under ``sudo``, or over an SSH session with no ``XDG_RUNTIME_DIR``.
    Never raises.
    """
    try:
        if not platform_utils.IS_LINUX or not _systemctl():
            return False
        result = _run_systemctl(["show-environment"], timeout=_QUERY_TIMEOUT)
        return bool(result is not None and result.ok)
    except Exception as exc:  # noqa: BLE001 - an availability probe never raises
        log.debug("systemd probe failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
#  Lingering
# --------------------------------------------------------------------------- #
def current_user() -> str:
    """The login name this module manages the service for."""
    for key in ("USER", "LOGNAME", "USERNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry (containers do this)
        return "unknown"


def linger_command() -> str:
    """The exact command that makes JARVIS survive logout."""
    return f"loginctl enable-linger {current_user()}"


def linger_enabled() -> Optional[bool]:
    """True/False when lingering could be determined, ``None`` when it could not.

    ``None`` is a real answer here and is not treated as False anywhere: on a
    machine with no ``loginctl`` we genuinely do not know, and claiming the
    service will die at logout would be as misleading as claiming it will not.
    """
    if not platform_utils.IS_LINUX:
        return None
    try:
        loginctl = platform_utils.which("loginctl")
        if loginctl:
            result = platform_utils.run_command(
                [loginctl, "show-user", current_user(), "--property=Linger"],
                timeout=_QUERY_TIMEOUT,
            )
            if result.ok:
                for line in result.stdout.splitlines():
                    if line.strip().startswith("Linger="):
                        return line.split("=", 1)[1].strip().lower() in ("yes", "true", "1")
        # loginctl absent or unhelpful: systemd records lingering as an empty
        # marker file, which is readable without any tool at all.
        marker = Path("/var/lib/systemd/linger") / current_user()
        if marker.parent.is_dir():
            return marker.exists()
    except Exception as exc:  # noqa: BLE001
        log.debug("linger probe failed: %s", exc)
    return None


# --------------------------------------------------------------------------- #
#  Install / uninstall
# --------------------------------------------------------------------------- #
def install(
    command: Optional[str] = None,
    *,
    description: str = DEFAULT_DESCRIPTION,
    restart: bool = True,
) -> ToolResult:
    """Write ``~/.config/systemd/user/jarvis.service`` and reload systemd.

    Writing the file is not enough — systemd caches unit files, so a new or
    edited unit is invisible until ``systemctl --user daemon-reload`` runs.  The
    reload is therefore part of installing, not a separate step to forget.

    Installing does not start or enable anything; call :func:`enable` and
    :func:`start` for that.  Returns the unit path plus the lingering advice, so
    a caller can print the one warning that matters in the same breath.
    """
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(_unavailable())

    cleaned = _clean_command(command)
    if not cleaned:
        return ToolResult.failure("refusing to install an empty or multi-line ExecStart")
    if not exec_start_is_absolute(cleaned):
        return ToolResult.failure(
            f"ExecStart must begin with an absolute path; got {cleaned!r}. systemd rejects "
            "relative commands. Use the full interpreter path, e.g. "
            f"{default_command()}"
        )

    path = unit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            unit_text(cleaned, description=description, restart=restart), encoding="utf-8"
        )
        try:
            path.chmod(0o644)
        except OSError:
            log.debug("could not chmod %s", path, exc_info=True)
    except OSError as exc:
        return ToolResult.failure(f"could not write {path}: {exc}")

    reload_result = _run_systemctl(["daemon-reload"])
    reloaded = bool(reload_result is not None and reload_result.ok)
    if not reloaded:
        detail = "systemctl is not installed" if reload_result is None else (
            reload_result.stderr.strip() or f"exit {reload_result.returncode}"
        )
        log.warning("unit written but daemon-reload failed: %s", detail)

    linger = linger_enabled()
    return ToolResult.success(
        {
            "path": str(path),
            "command": cleaned,
            "daemon_reloaded": reloaded,
            "linger": linger,
            "next_steps": [
                "systemctl --user enable --now jarvis.service",
                linger_command()
                + "   # without this JARVIS stops when you log out",
            ],
        }
    )


def uninstall() -> ToolResult:
    """Stop, disable and remove the unit file, then reload systemd.

    Only ``~/.config/systemd/user/jarvis.service`` is removed — nothing else is
    touched, and a missing file is reported as success because the requested
    end state (no JARVIS unit) already holds.
    """
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(_unavailable())

    for args in (["disable", UNIT_NAME], ["stop", UNIT_NAME]):
        result = _run_systemctl(args)
        if result is not None and not result.ok:
            log.debug("systemctl %s: %s", args[0], result.stderr.strip()[:200])

    path = unit_path()
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        return ToolResult.failure(f"could not remove {path}: {exc}")

    _run_systemctl(["daemon-reload"])
    return ToolResult.success({"path": str(path), "removed": True})


# --------------------------------------------------------------------------- #
#  Lifecycle
# --------------------------------------------------------------------------- #
def _control(verb: str, extra: Optional[List[str]] = None) -> ToolResult:
    if not platform_utils.IS_LINUX or not _systemctl():
        return ToolResult.failure(_unavailable())
    if not unit_path().exists():
        return ToolResult.failure(
            f"{unit_path()} does not exist — run install() first "
            "(jarvis.linux.service.install)."
        )
    result = _run_systemctl([verb, *(extra or []), UNIT_NAME])
    if result is None:  # pragma: no cover - guarded above
        return ToolResult.failure(_unavailable())
    if not result.ok:
        detail = (result.stderr.strip() or result.stdout.strip() or
                  f"exit {result.returncode}")
        return ToolResult.failure(f"systemctl --user {verb} {UNIT_NAME} failed: {detail}")
    return ToolResult.success({"action": verb, "unit": UNIT_NAME, "output": result.stdout.strip()})


def enable() -> ToolResult:
    """Start JARVIS whenever the user manager starts (``WantedBy=default.target``).

    Enabling alone is not enough to survive logout; see :func:`linger_enabled`.
    """
    return _control("enable")


def disable() -> ToolResult:
    """Stop JARVIS starting automatically.  Does not stop a running instance."""
    return _control("disable")


def start() -> ToolResult:
    """Start the service now."""
    return _control("start")


def stop() -> ToolResult:
    """Stop the service now."""
    return _control("stop")


def restart() -> ToolResult:
    """Restart the service — the usual way to pick up a config change."""
    return _control("restart")


def logs(lines: int = 50) -> ToolResult:
    """The last ``lines`` journal entries for the unit.

    ``journalctl --user`` reads the caller's own journal, so no privileges are
    involved.  ``lines`` is clamped to :data:`MAX_LOG_LINES` to keep a runaway
    journal out of memory (and out of a model's context window).
    """
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(_unavailable())
    try:
        count = max(1, min(int(lines), MAX_LOG_LINES))
    except (TypeError, ValueError):
        count = 50

    exe = _journalctl()
    if not exe:
        return ToolResult.failure(
            "journalctl was not found. Without systemd's journal, try: "
            "systemctl --user status jarvis.service"
        )
    result = platform_utils.run_command(
        [exe, "--user", "-u", UNIT_NAME, "-n", str(count), "--no-pager"],
        timeout=_LOG_TIMEOUT,
    )
    if not result.ok:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return ToolResult.failure(f"journalctl failed: {detail}")
    return ToolResult.success(result.stdout, is_artifact=True)


# --------------------------------------------------------------------------- #
#  Status
# --------------------------------------------------------------------------- #
def _query(verb: str) -> Optional[str]:
    """``is-enabled``/``is-active`` return non-zero for perfectly normal states."""
    result = _run_systemctl([verb, UNIT_NAME], timeout=_QUERY_TIMEOUT)
    if result is None:
        return None
    text = (result.stdout or "").strip() or (result.stderr or "").strip()
    return text.splitlines()[0].strip() if text else None


def status() -> Dict[str, Any]:
    """Everything ``jarvis doctor`` should say about the Linux service.

    Always returns a dict and never raises.  ``advice`` is the part worth
    printing: it names the exact next command for whatever is missing, and it
    leads with the lingering warning because that is the failure people actually
    hit — the service works perfectly until they close the lid.
    """
    report: Dict[str, Any] = {
        "platform": platform_utils.os_name(),
        "supported": False,
        "systemctl": None,
        "user_bus": False,
        "unit_path": str(unit_path()),
        "installed": False,
        "enabled": None,
        "active": None,
        "linger": None,
        "linger_command": linger_command(),
        "user": current_user(),
        "advice": [],
        "error": None,
    }
    advice: List[str] = report["advice"]

    try:
        report["systemctl"] = _systemctl()
        report["installed"] = unit_path().exists()

        if not platform_utils.IS_LINUX:
            report["error"] = _unavailable()
            advice.append(_unavailable())
            return report

        report["user_bus"] = is_available()
        report["supported"] = bool(report["systemctl"]) and report["user_bus"]
        if not report["supported"]:
            report["error"] = _unavailable()
            advice.append(_unavailable())
            return report

        report["enabled"] = _query("is-enabled")
        report["active"] = _query("is-active")
        report["linger"] = linger_enabled()

        if not report["installed"]:
            advice.append(
                "No unit installed yet — run jarvis.linux.service.install() "
                f"(writes {unit_path()})."
            )
        else:
            if report["enabled"] != "enabled":
                advice.append(
                    "The service is installed but not enabled; it will not start on boot. "
                    "Run: systemctl --user enable --now jarvis.service"
                )
            if report["active"] != "active":
                advice.append(
                    "The service is not running right now. Run: "
                    "systemctl --user start jarvis.service   "
                    "(then 'journalctl --user -u jarvis.service -n 50' if it does not stay up)"
                )

        if report["linger"] is False:
            advice.insert(
                0,
                "Lingering is OFF, so the user manager — and JARVIS with it — is torn "
                "down when you log out or close the lid, with no error logged anywhere. "
                f"To keep it running with nobody logged in, run: {linger_command()}",
            )
        elif report["linger"] is None:
            advice.append(
                "Could not determine whether lingering is enabled (loginctl missing or "
                f"unreadable). If JARVIS stops when you log out, run: {linger_command()}"
            )
    except Exception as exc:  # noqa: BLE001 - a diagnosis never crashes the caller
        log.debug("service status failed", exc_info=True)
        report["error"] = f"{type(exc).__name__}: {exc}"

    return report


__all__ = [
    "UNIT_NAME",
    "DEFAULT_DESCRIPTION",
    "MAX_LOG_LINES",
    "unit_dir",
    "unit_path",
    "unit_text",
    "quote_exec",
    "default_command",
    "exec_start_is_absolute",
    "is_available",
    "install",
    "uninstall",
    "enable",
    "disable",
    "start",
    "stop",
    "restart",
    "logs",
    "status",
    "current_user",
    "linger_enabled",
    "linger_command",
]
