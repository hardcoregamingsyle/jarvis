"""Start JARVIS when the user logs in — per user, never for the machine.

Windows
    A value under ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run``.
    Deliberately *not* ``HKLM``: writing there needs elevation, affects every
    account on the computer, and is precisely the kind of thing an assistant
    should not do behind its owner's back.

Linux
    A ``.desktop`` file in ``$XDG_CONFIG_HOME/autostart`` (``~/.config/autostart``),
    which every mainstream desktop environment honours.

The interpreter path is always quoted.  On a normal Windows install it is
``C:\\Program Files\\Python313\\python.exe``, and an unquoted Run value stops at
the first space: Windows tries to launch ``C:\\Program`` and simply does
nothing — no error, no assistant, no clue why.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from ..core.platform_utils import IS_MAC, IS_WINDOWS

log = logging.getLogger(__name__)


#: Name of the registry value / desktop file.  Stable, so enable() is idempotent.
APP_NAME = "JARVIS"
RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
DESKTOP_FILE_NAME = "jarvis.desktop"


# --------------------------------------------------------------------------- #
#  Command construction
# --------------------------------------------------------------------------- #
def quote_argument(value: str) -> str:
    """Wrap ``value`` in double quotes unless it already is.

    Applied unconditionally to the interpreter path: quoting a path that does
    not need it is harmless, while failing to quote one that does produces a
    silent no-op at every login.
    """
    text = str(value or "")
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text
    return '"' + text.replace('"', '') + '"'


def default_command() -> str:
    """``"<python>" -m jarvis voice`` for the interpreter running right now."""
    interpreter = sys.executable or ("python" if not IS_WINDOWS else "pythonw.exe")
    return f"{quote_argument(interpreter)} -m jarvis voice"


def _clean_command(command: Optional[str]) -> Optional[str]:
    """Validate a caller-supplied command; ``None`` means "use the default"."""
    if command is None:
        return default_command()
    text = str(command).strip()
    if not text or "\n" in text or "\r" in text:
        return None
    return text


# --------------------------------------------------------------------------- #
#  Windows: the per-user Run key
# --------------------------------------------------------------------------- #
def _winreg() -> Optional[Any]:
    """The :mod:`winreg` module, or ``None`` anywhere it does not exist."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg  # type: ignore

        return winreg
    except ImportError as exc:  # a stripped-down or non-CPython Windows build
        log.debug("winreg unavailable: %s", exc)
        return None


def _registry_read(winreg: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(value, error)`` for our Run entry."""
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_READ)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot open the Run key: {exc}"
    try:
        value, _kind = winreg.QueryValueEx(key, APP_NAME)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot read the Run value: {exc}"
    finally:
        _close(winreg, key)
    return (str(value) if value is not None else None), None


def _registry_write(winreg: Any, command: str) -> Tuple[bool, Optional[str]]:
    try:
        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE
        )
    except OSError as exc:
        return False, f"cannot open the Run key for writing: {exc}"
    try:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
    except OSError as exc:
        return False, f"cannot write the Run value: {exc}"
    finally:
        _close(winreg, key)
    return True, None


def _registry_delete(winreg: Any) -> Tuple[bool, Optional[str]]:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE)
    except FileNotFoundError:
        return True, None            # no key at all: already not starting
    except OSError as exc:
        return False, f"cannot open the Run key for writing: {exc}"
    try:
        winreg.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        return True, None            # no value: already not starting
    except OSError as exc:
        return False, f"cannot delete the Run value: {exc}"
    finally:
        _close(winreg, key)
    return True, None


def _close(winreg: Any, key: Any) -> None:
    closer = getattr(winreg, "CloseKey", None)
    if callable(closer):
        try:
            closer(key)
        except Exception:  # noqa: BLE001 - closing must never be the thing that fails
            log.debug("CloseKey failed", exc_info=True)


# --------------------------------------------------------------------------- #
#  Linux: an XDG autostart entry
# --------------------------------------------------------------------------- #
def desktop_entry_path() -> Path:
    """Where the ``.desktop`` file lives (``$XDG_CONFIG_HOME/autostart``)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path(os.path.expanduser("~")) / ".config")
    return Path(base).expanduser() / "autostart" / DESKTOP_FILE_NAME


def desktop_entry_text(command: str) -> str:
    """The contents of the autostart entry for ``command``."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        f"Name={APP_NAME}\n"
        "Comment=Voice assistant\n"
        f"Exec={command}\n"
        "Terminal=false\n"
        "NoDisplay=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def _desktop_command() -> Optional[str]:
    path = desktop_entry_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        if line.startswith("Exec="):
            return line[len("Exec="):].strip()
    return None


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def is_supported() -> bool:
    """True when this platform has an autostart mechanism we can write to.

    macOS is reported as unsupported on purpose: it uses launchd agents, not
    XDG ``.desktop`` files, so writing one there would be a file that never
    runs — worse than an honest "no".
    """
    if IS_WINDOWS:
        return _winreg() is not None
    if IS_MAC:
        return False
    return True   # Linux and the other XDG desktops


def current_command() -> Optional[str]:
    """The command that is currently configured to run at login, if any."""
    try:
        if IS_WINDOWS:
            winreg = _winreg()
            if winreg is None:
                return None
            value, _error = _registry_read(winreg)
            return value
        return _desktop_command()
    except Exception:  # noqa: BLE001 - a status query never raises
        log.debug("could not read the autostart entry", exc_info=True)
        return None


def is_enabled() -> bool:
    """True when JARVIS is registered to start at login."""
    return bool(current_command())


def enable(command: Optional[str] = None) -> bool:
    """Register JARVIS to start at login.  Returns False on failure.

    ``command`` defaults to :func:`default_command`; pass your own to launch a
    packaged executable or add flags.
    """
    cleaned = _clean_command(command)
    if not cleaned:
        log.warning("autostart: refusing to install an empty or multi-line command")
        return False

    try:
        if IS_WINDOWS:
            winreg = _winreg()
            if winreg is None:
                log.warning("autostart: winreg is unavailable on this interpreter")
                return False
            ok, error = _registry_write(winreg, cleaned)
            if not ok:
                log.warning("autostart: %s", error)
                return False
            log.info("autostart enabled: %s", cleaned)
            return True

        if IS_MAC:
            log.warning("autostart: macOS needs a launchd agent, which is not implemented")
            return False

        path = desktop_entry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(desktop_entry_text(cleaned), encoding="utf-8")
        try:
            path.chmod(0o755)
        except OSError:
            log.debug("could not chmod %s", path, exc_info=True)
        log.info("autostart enabled: %s", path)
        return True
    except Exception as exc:  # noqa: BLE001 - never raise out of a settings toggle
        log.warning("autostart could not be enabled: %s", exc)
        return False


def disable() -> bool:
    """Stop JARVIS starting at login.  True when it is off afterwards."""
    try:
        if IS_WINDOWS:
            winreg = _winreg()
            if winreg is None:
                return False
            ok, error = _registry_delete(winreg)
            if not ok:
                log.warning("autostart: %s", error)
            return ok

        path = desktop_entry_path()
        if path.exists():
            path.unlink()
        return not path.exists()
    except Exception as exc:  # noqa: BLE001
        log.warning("autostart could not be disabled: %s", exc)
        return False


def location() -> str:
    """A human-readable description of where the setting is stored."""
    if IS_WINDOWS:
        return f"HKEY_CURRENT_USER\\{RUN_KEY_PATH}\\{APP_NAME}"
    return str(desktop_entry_path())


def status() -> dict:
    """Everything ``jarvis doctor`` wants to say about autostart."""
    supported = False
    error: Optional[str] = None
    try:
        supported = is_supported()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    command = current_command()
    return {
        "platform": "windows" if IS_WINDOWS else "posix",
        "supported": supported,
        "enabled": bool(command),
        "command": command,
        "default_command": default_command(),
        "location": location(),
        "error": error,
    }


__all__ = [
    "enable", "disable", "is_enabled", "status", "is_supported",
    "current_command", "default_command", "quote_argument", "location",
    "desktop_entry_path", "desktop_entry_text",
    "APP_NAME", "RUN_KEY_PATH",
]
