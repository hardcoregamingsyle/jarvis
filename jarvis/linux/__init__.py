"""Linux integration: systemd user service, desktop notifications, audio.

The Linux counterpart of :mod:`jarvis.win`.  Windows has a tray icon, global
hotkeys, a Run-key autostart entry and toasts; this package covers the same
ground on the dedicated Linux box, plus the two things Linux needs and Windows
does not — a supervised background service and an honest account of which
sound server is running.

* :mod:`jarvis.linux.service` — a **user** systemd unit (never system-wide,
  never ``sudo``), with the lingering warning that decides whether JARVIS
  survives you closing the lid.
* :mod:`jarvis.linux.desktop` — notifications, XDG autostart, window control,
  and a straight answer about what Wayland will and will not permit.
* :mod:`jarvis.linux.audio` — PipeWire/PulseAudio/ALSA detection, device
  listing, and a diagnosis with the exact package command for this distro.

Every module here imports and behaves on Windows and macOS: they report
themselves unavailable instead of raising, so ``jarvis doctor`` can say
honestly what the machine it is on supports.  Nothing in this package imports a
third-party library, and nothing touches Linux-only stdlib at import time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from . import audio, desktop, service
from .audio import install_command, package_manager
from .desktop import notify, session_type

log = logging.getLogger(__name__)


def _probe(name: str, fn: Any) -> Any:
    """Run one availability check, turning any failure into a readable string."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a diagnosis must never itself crash
        log.debug("%s probe failed", name, exc_info=True)
        return f"error: {type(exc).__name__}: {exc}"


def is_linux_integration_available() -> Dict[str, Any]:
    """Report which Linux-integration pieces work on this machine.

    Consumed by ``jarvis doctor``.  Mirrors
    :func:`jarvis.win.is_windows_integration_available`: every value is a bool,
    a string or a nested dict, nothing raises, and the only side effect is
    running short read-only probe commands.
    """
    from ..core.platform_utils import IS_LINUX, os_name

    return {
        "os": os_name(),
        "is_linux": IS_LINUX,
        "session": _probe("session", session_type),
        "package_manager": _probe("package_manager", package_manager),
        "service": _probe("service", service.status),
        "desktop": _probe("desktop", desktop.is_available),
        "audio": _probe("audio", audio.check),
    }


__all__ = [
    "service",
    "desktop",
    "audio",
    "notify",
    "session_type",
    "package_manager",
    "install_command",
    "is_linux_integration_available",
]
