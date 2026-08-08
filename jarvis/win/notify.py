"""Desktop notifications, with a ladder of fallbacks that ends in a log line.

Order of attempts:

1. a real Windows toast, raised through PowerShell and the
   ``Windows.UI.Notifications`` API (no third-party package, works on Windows 10
   and 11, and survives being called from a background thread);
2. a balloon from the live tray icon, if one has been registered;
3. ``notify-send`` on Linux / ``msg`` on Windows Server-style installs;
4. a log line, which always succeeds.

:func:`notify` returns the name of the channel that worked.  It never raises and
never blocks for longer than the timeout it grants each channel, because a
notification that hangs the assistant is worse than no notification at all.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional, Tuple

from ..core import platform_utils

log = logging.getLogger(__name__)


#: Beyond this a "notification" is really a dialogue box; below it, invisible.
MIN_DURATION = 1
MAX_DURATION = 60

_tray_lock = threading.Lock()
_tray: Optional[Any] = None


# --------------------------------------------------------------------------- #
#  Tray registration
# --------------------------------------------------------------------------- #
def set_tray(tray: Optional[Any]) -> None:
    """Register (or clear) the live tray icon used for balloon notifications."""
    global _tray
    with _tray_lock:
        _tray = tray


def get_tray() -> Optional[Any]:
    with _tray_lock:
        return _tray


# --------------------------------------------------------------------------- #
#  Escaping
# --------------------------------------------------------------------------- #
def xml_escape(text: str) -> str:
    """Escape text for the toast XML payload."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def powershell_quote(text: str) -> str:
    """Quote text as a PowerShell single-quoted string (``'`` doubles itself)."""
    return "'" + str(text).replace("'", "''") + "'"


def _applescript_string(text: str) -> str:
    """Escape text for embedding in an AppleScript double-quoted literal."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _clamp_duration(duration: Any) -> int:
    try:
        value = int(duration)
    except (TypeError, ValueError):
        return 5
    return max(MIN_DURATION, min(MAX_DURATION, value))


# --------------------------------------------------------------------------- #
#  Channel 1: a real Windows toast
# --------------------------------------------------------------------------- #
#: PowerShell's own AppUserModelID.  A toast must be raised under an AUMID that
#: is actually installed, and PowerShell's always is; using an invented one
#: makes Windows swallow the toast silently.
POWERSHELL_AUMID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def _toast_script(title: str, message: str, duration: int) -> str:
    # Double quotes throughout the XML, and every text node XML-escaped, so the
    # payload contains no apostrophe at all — it then survives being wrapped in
    # a PowerShell single-quoted string without any further mangling.
    payload = (
        '<toast duration="{dur}"><visual><binding template="ToastGeneric">'
        "<text>{title}</text><text>{body}</text>"
        "</binding></visual></toast>"
    ).format(
        dur="long" if duration > 7 else "short",
        title=xml_escape(title),
        body=xml_escape(message),
    )
    return (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] > $null; "
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
        "ContentType = WindowsRuntime] > $null; "
        "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument; "
        f"$xml.LoadXml({powershell_quote(payload)}); "
        "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml; "
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        f"{powershell_quote(POWERSHELL_AUMID)}).Show($toast)"
    )


def _windows_toast(title: str, message: str, duration: int) -> bool:
    if not platform_utils.IS_WINDOWS:
        return False
    shell = platform_utils.which("powershell") or platform_utils.which("pwsh")
    if not shell:
        return False
    result = platform_utils.run_command(
        [shell, "-NoProfile", "-NonInteractive", "-Command", _toast_script(title, message, duration)],
        timeout=max(10.0, float(duration) + 5.0),
    )
    if not result.ok:
        log.debug("toast failed (rc=%s): %s", result.returncode, result.stderr.strip()[:300])
    return bool(result.ok)


# --------------------------------------------------------------------------- #
#  Channel 2: the tray balloon
# --------------------------------------------------------------------------- #
def _tray_balloon(title: str, message: str, duration: int, tray: Optional[Any]) -> bool:
    target = tray if tray is not None else get_tray()
    if target is None:
        return False
    method = getattr(target, "notify", None)
    if not callable(method):
        return False
    return bool(method(title, message))


# --------------------------------------------------------------------------- #
#  Channel 3: command-line notifiers
# --------------------------------------------------------------------------- #
def _command_line_notifier(title: str, message: str, duration: int) -> bool:
    if platform_utils.IS_WINDOWS:
        msg = platform_utils.which("msg")
        if not msg:
            return False
        result = platform_utils.run_command(
            [msg, "*", f"/time:{duration}", f"{title}: {message}"],
            timeout=max(5.0, float(duration) + 5.0),
        )
        return bool(result.ok)

    notify_send = platform_utils.which("notify-send")
    if notify_send:
        result = platform_utils.run_command(
            [notify_send, "-t", str(duration * 1000), str(title), str(message)],
            timeout=10.0,
        )
        return bool(result.ok)

    if platform_utils.IS_MAC:
        osascript = platform_utils.which("osascript")
        if osascript:
            script = (
                f'display notification "{_applescript_string(message)[:200]}" '
                f'with title "{_applescript_string(title)[:100]}"'
            )
            result = platform_utils.run_command([osascript, "-e", script], timeout=10.0)
            return bool(result.ok)
    return False


# --------------------------------------------------------------------------- #
#  Channel 4: the log
# --------------------------------------------------------------------------- #
def _log_only(title: str, message: str, duration: int) -> bool:
    log.info("notification | %s | %s", title, message)
    return True


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def notify(
    title: str,
    message: str,
    *,
    duration: int = 5,
    tray: Optional[Any] = None,
) -> str:
    """Show a desktop notification and report which channel delivered it.

    Returns one of ``"toast"``, ``"tray"``, ``"command"`` or ``"log"``.  Never
    raises: a failing channel is logged at debug level and the next one is
    tried, and the log channel cannot fail.
    """
    title = str(title or "JARVIS")
    message = str(message or "")
    seconds = _clamp_duration(duration)

    chain: List[Tuple[str, Callable[[], bool]]] = [
        ("toast", lambda: _windows_toast(title, message, seconds)),
        ("tray", lambda: _tray_balloon(title, message, seconds, tray)),
        ("command", lambda: _command_line_notifier(title, message, seconds)),
        ("log", lambda: _log_only(title, message, seconds)),
    ]

    for name, attempt in chain:
        try:
            if attempt():
                if name != "log":
                    log.debug("notification delivered via %s", name)
                return name
        except Exception as exc:  # noqa: BLE001 - try the next channel, always
            log.debug("notification channel %s failed: %s", name, exc)

    # Only reachable if logging itself was replaced by something that fails.
    return "none"


def available_channels() -> dict:
    """Which notification channels this machine can use, for ``jarvis doctor``."""
    try:
        shell = platform_utils.which("powershell") or platform_utils.which("pwsh")
        cli = (platform_utils.which("msg") if platform_utils.IS_WINDOWS
               else platform_utils.which("notify-send"))
        return {
            "toast": bool(platform_utils.IS_WINDOWS and shell),
            "tray": get_tray() is not None,
            "command": bool(cli),
            "log": True,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("channel probe failed: %s", exc)
        return {"toast": False, "tray": False, "command": False, "log": True}


__all__ = [
    "notify", "available_channels", "set_tray", "get_tray",
    "xml_escape", "powershell_quote", "POWERSHELL_AUMID",
]
