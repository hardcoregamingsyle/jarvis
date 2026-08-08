"""Desktop integration: tray icon, global hotkeys, autostart, notifications.

This is what turns JARVIS from a program you run into an assistant that is
simply *there*: resident in the notification area, summoned by a key, started
with the session.

Despite the package name, every module here imports and behaves on Linux and
macOS too — they report themselves unavailable instead of exploding, so
``jarvis doctor`` can say honestly what this machine supports.

One naming wrinkle worth knowing: ``jarvis.win.notify`` re-exported here is the
*function*, which shadows the submodule of the same name on this package
object.  ``from jarvis.win import notify`` therefore gives you something you can
call; reach the module itself as ``jarvis.win.notify`` only via
``importlib.import_module``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from . import autostart
from .hotkey import (
    DEFAULT_PUSH_TO_TALK_COMBO,
    DEFAULT_TOGGLE_COMBO,
    CallbackDispatcher,
    HotkeyManager,
    default_hotkeys,
    normalise_combo,
    parse_combo,
    voice_loop_callbacks,
)
from .notify import available_channels, get_tray, notify, set_tray
from .tray import STATE_COLOURS, TrayIcon, render_icon

log = logging.getLogger(__name__)


def _probe(name: str, fn: Any) -> Any:
    """Run one availability check, turning any failure into a readable string."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a diagnosis must never itself crash
        log.debug("%s probe failed", name, exc_info=True)
        return f"error: {type(exc).__name__}: {exc}"


def is_windows_integration_available() -> Dict[str, Any]:
    """Report which desktop-integration pieces work on this machine.

    Consumed by ``jarvis doctor``.  Every value is either a bool, a string, or a
    nested dict; nothing here raises, and nothing here has a side effect beyond
    importing optional packages.
    """
    from ..core.platform_utils import os_name

    keys = HotkeyManager()
    tray = TrayIcon()

    result: Dict[str, Any] = {
        "os": os_name(),
        "hotkeys": _probe("hotkeys", keys.is_available),
        "hotkey_backend": _probe("hotkey_backend", lambda: keys.backend_name),
        "hotkey_error": keys.last_error,
        "tray": _probe("tray", tray.is_available),
        "tray_error": tray.last_error,
        "notifications": _probe("notifications", available_channels),
        "autostart": _probe("autostart", autostart.is_supported),
        "autostart_enabled": _probe("autostart_enabled", autostart.is_enabled),
        "suggested_hotkeys": default_hotkeys(),
    }
    return result


__all__ = [
    "HotkeyManager",
    "CallbackDispatcher",
    "TrayIcon",
    "render_icon",
    "STATE_COLOURS",
    "autostart",
    "notify",
    "set_tray",
    "get_tray",
    "available_channels",
    "default_hotkeys",
    "normalise_combo",
    "parse_combo",
    "voice_loop_callbacks",
    "DEFAULT_TOGGLE_COMBO",
    "DEFAULT_PUSH_TO_TALK_COMBO",
    "is_windows_integration_available",
]
