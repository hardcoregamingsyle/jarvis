"""Launch and control applications and windows.

Every path is a "best effort" — different desktops ship wildly different apps.
When a required helper is missing we return :class:`ToolResult.failure` with an
actionable message, we never throw.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Optional

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import (
    IS_LINUX,
    IS_MAC,
    IS_WINDOWS,
    open_path,
    resolve_path,
    run_command,
    which,
)
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


_LAUNCH_TIMEOUT = 15.0
_QUERY_TIMEOUT = 10.0
_CLOSE_TIMEOUT = 15.0


# --------------------------------------------------------------------------- #
#  Aliases
# --------------------------------------------------------------------------- #
_WINDOWS_ALIASES: Dict[str, str] = {
    "browser": "msedge.exe",
    "web": "msedge.exe",
    "edge": "msedge.exe",
    "chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "notepad": "notepad.exe",
    "text": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "terminal": "wt.exe",
    "shell": "powershell.exe",
    "powershell": "powershell.exe",
    "cmd": "cmd.exe",
    "explorer": "explorer.exe",
    "files": "explorer.exe",
    "code": "code",
    "vscode": "code",
    "spotify": "spotify.exe",
    "paint": "mspaint.exe",
}

_LINUX_ALIASES: Dict[str, List[str]] = {
    "browser": ["xdg-open", "firefox", "google-chrome", "chromium"],
    "web": ["xdg-open", "firefox"],
    "firefox": ["firefox"],
    "chrome": ["google-chrome", "chromium"],
    "notepad": ["gnome-text-editor", "gedit", "kate", "mousepad"],
    "text": ["gnome-text-editor", "gedit", "kate", "mousepad"],
    "calculator": ["gnome-calculator", "kcalc", "galculator"],
    "calc": ["gnome-calculator", "kcalc", "galculator"],
    "terminal": ["gnome-terminal", "konsole", "xterm", "alacritty", "kitty"],
    "shell": ["gnome-terminal", "konsole", "xterm"],
    "explorer": ["nautilus", "dolphin", "thunar", "nemo"],
    "files": ["nautilus", "dolphin", "thunar", "nemo"],
    "code": ["code"],
    "vscode": ["code"],
    "spotify": ["spotify"],
}


# --------------------------------------------------------------------------- #
#  Windows helpers
# --------------------------------------------------------------------------- #
def _windows_app_paths_lookup(name: str) -> Optional[str]:
    """Consult HKLM App Paths for an executable name."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg  # type: ignore
    except ImportError:
        return None
    candidates = (name, f"{name}.exe" if not name.lower().endswith(".exe") else None)
    for candidate in candidates:
        if not candidate:
            continue
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            key_path = (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\\"
                + candidate
            )
            try:
                key = winreg.OpenKey(hive, key_path)
            except OSError:
                continue
            try:
                value, _ = winreg.QueryValueEx(key, "")
            except OSError:
                winreg.CloseKey(key)
                continue
            winreg.CloseKey(key)
            if isinstance(value, str) and value:
                return value.strip('"')
    return None


def _launch_windows(name: str) -> ToolResult:
    alias = _WINDOWS_ALIASES.get(name.lower(), name)
    resolved = which(alias) or _windows_app_paths_lookup(alias)
    argv = resolved or alias
    script = (
        f"Start-Process -FilePath {_ps_quote(argv)}"
    )
    result = run_command(
        ["powershell", "-NoProfile", "-Command", script],
        timeout=_LAUNCH_TIMEOUT,
    )
    if not result.ok:
        return ToolResult.failure(
            f"launch_app({name!r}) failed: {safe_truncate(result.stderr, 300)}"
        )
    return ToolResult.success(output={"name": name, "target": argv})


def _ps_quote(value: str) -> str:
    """PowerShell-safe single-quoted string (double any embedded quote)."""
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


# --------------------------------------------------------------------------- #
#  Linux helpers
# --------------------------------------------------------------------------- #
def _launch_linux(name: str) -> ToolResult:
    candidates = _LINUX_ALIASES.get(name.lower(), [name])
    for cand in candidates:
        if not which(cand):
            continue
        try:
            subprocess.Popen(
                [cand],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(f"launch_app failed: {exc}")
        return ToolResult.success(output={"name": name, "target": cand})
    # Fall back to xdg-open / gio for a .desktop lookup.
    if which("gio"):
        result = run_command(
            ["gio", "launch", f"{name}.desktop"], timeout=_LAUNCH_TIMEOUT
        )
        if result.ok:
            return ToolResult.success(output={"name": name, "target": "gio"})
    if which("xdg-open"):
        try:
            subprocess.Popen(
                ["xdg-open", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(f"launch_app failed: {exc}")
        return ToolResult.success(output={"name": name, "target": "xdg-open"})
    return ToolResult.failure(
        f"no launcher found for {name!r} — install one of {candidates}"
    )


def _launch_mac(name: str) -> ToolResult:
    if not which("open"):
        return ToolResult.failure("macOS 'open' command is unavailable")
    result = run_command(["open", "-a", name], timeout=_LAUNCH_TIMEOUT)
    if not result.ok:
        return ToolResult.failure(
            f"launch_app({name!r}) failed: {safe_truncate(result.stderr, 300)}"
        )
    return ToolResult.success(output={"name": name, "target": name})


# --------------------------------------------------------------------------- #
#  Public tools
# --------------------------------------------------------------------------- #
def _launch_app(name: str) -> ToolResult:
    """Launch an application by short name or executable."""
    if not name or not str(name).strip():
        return ToolResult.failure("name is required")
    name = str(name).strip()
    if IS_WINDOWS:
        return _launch_windows(name)
    if IS_MAC:
        return _launch_mac(name)
    if IS_LINUX:
        return _launch_linux(name)
    return ToolResult.failure("launch_app: unsupported platform")


def _close_app(name: str) -> ToolResult:
    """Ask an application to exit (dangerous — may lose unsaved work)."""
    if not name or not str(name).strip():
        return ToolResult.failure("name is required")
    name = str(name).strip()
    if IS_WINDOWS:
        alias = _WINDOWS_ALIASES.get(name.lower(), name)
        if not alias.lower().endswith(".exe"):
            alias = f"{alias}.exe"
        result = run_command(
            ["taskkill", "/IM", alias, "/T"], timeout=_CLOSE_TIMEOUT
        )
        if not result.ok:
            return ToolResult.failure(
                f"close_app failed: {safe_truncate(result.stderr or result.stdout, 300)}"
            )
        return ToolResult.success(output={"name": name, "target": alias})
    if IS_LINUX or IS_MAC:
        if which("pkill"):
            result = run_command(["pkill", "-x", name], timeout=_CLOSE_TIMEOUT)
            if result.returncode in (0, 1):
                return ToolResult.success(
                    output={"name": name, "returncode": result.returncode}
                )
            return ToolResult.failure(
                f"pkill failed: {safe_truncate(result.stderr, 300)}"
            )
        if which("killall"):
            result = run_command(["killall", name], timeout=_CLOSE_TIMEOUT)
            if result.returncode in (0, 1):
                return ToolResult.success(
                    output={"name": name, "returncode": result.returncode}
                )
            return ToolResult.failure(
                f"killall failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.failure("no process-close helper found (install pkill)")
    return ToolResult.failure("close_app: unsupported platform")


def _list_windows() -> ToolResult:
    """List top-level windows visible to the user."""
    if IS_WINDOWS:
        script = (
            "Get-Process | Where-Object { $_.MainWindowTitle -ne '' } | "
            "Select-Object Id, ProcessName, MainWindowTitle | "
            "ConvertTo-Json -Compress"
        )
        result = run_command(
            ["powershell", "-NoProfile", "-Command", script],
            timeout=_QUERY_TIMEOUT,
        )
        if not result.ok:
            return ToolResult.failure(
                f"list_windows failed: {safe_truncate(result.stderr, 300)}"
            )
        import json as _json
        try:
            parsed = _json.loads(result.stdout or "[]")
        except _json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, dict):
            parsed = [parsed]
        windows = [
            {
                "pid": entry.get("Id"),
                "process": entry.get("ProcessName"),
                "title": entry.get("MainWindowTitle"),
            }
            for entry in parsed
            if isinstance(entry, dict)
        ]
        return ToolResult.success(output={"windows": windows, "count": len(windows)})

    if IS_LINUX:
        if which("wmctrl"):
            result = run_command(["wmctrl", "-l"], timeout=_QUERY_TIMEOUT)
            if not result.ok:
                return ToolResult.failure(
                    f"wmctrl failed: {safe_truncate(result.stderr, 300)}"
                )
            windows: List[Dict[str, Any]] = []
            for raw in result.stdout.splitlines():
                parts = raw.split(None, 3)
                if len(parts) < 4:
                    continue
                windows.append(
                    {"id": parts[0], "desktop": parts[1], "host": parts[2], "title": parts[3]}
                )
            return ToolResult.success(output={"windows": windows, "count": len(windows)})
        return ToolResult.failure("wmctrl is not installed")

    if IS_MAC:
        return ToolResult.failure(
            "list_windows on macOS requires a native helper (not implemented)"
        )
    return ToolResult.failure("list_windows: unsupported platform")


def _focus_window(title: str) -> ToolResult:
    """Bring the first window whose title contains ``title`` to the front."""
    if not title or not str(title).strip():
        return ToolResult.failure("title is required")
    if IS_WINDOWS:
        script = (
            "Add-Type -AssemblyName Microsoft.VisualBasic;"
            f"[Microsoft.VisualBasic.Interaction]::AppActivate({_ps_quote(title)})"
        )
        result = run_command(
            ["powershell", "-NoProfile", "-Command", script],
            timeout=_QUERY_TIMEOUT,
        )
        if not result.ok:
            return ToolResult.failure(
                f"focus_window failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"title": str(title)})
    if IS_LINUX and which("wmctrl"):
        result = run_command(["wmctrl", "-a", str(title)], timeout=_QUERY_TIMEOUT)
        if not result.ok:
            return ToolResult.failure(
                f"wmctrl -a failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"title": str(title)})
    return ToolResult.failure("focus_window is unavailable on this platform")


def _open_file_with(path: str, app: Optional[str] = None) -> ToolResult:
    """Open a file with the OS default handler or an explicit ``app``."""
    if not path or not str(path).strip():
        return ToolResult.failure("path is required")
    try:
        resolved = resolve_path(str(path))
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"no such path: {resolved}")
    target = str(resolved)
    if not app:
        result = open_path(target)
        if not result.ok:
            return ToolResult.failure(
                f"open_file_with failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"path": target})
    if IS_WINDOWS:
        script = (
            f"Start-Process -FilePath {_ps_quote(app)} -ArgumentList {_ps_quote(target)}"
        )
        result = run_command(
            ["powershell", "-NoProfile", "-Command", script],
            timeout=_LAUNCH_TIMEOUT,
        )
        if not result.ok:
            return ToolResult.failure(
                f"open_file_with failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"path": target, "app": app})
    if IS_MAC and which("open"):
        result = run_command(["open", "-a", str(app), target], timeout=_LAUNCH_TIMEOUT)
        if not result.ok:
            return ToolResult.failure(
                f"open failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"path": target, "app": app})
    if IS_LINUX:
        if not which(str(app)):
            return ToolResult.failure(f"application not found: {app}")
        try:
            subprocess.Popen(
                [str(app), target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return ToolResult.failure(f"open_file_with failed: {exc}")
        return ToolResult.success(output={"path": target, "app": app})
    return ToolResult.failure("open_file_with is unavailable on this platform")


def _open_folder(path: str) -> ToolResult:
    """Show ``path`` in the OS file browser."""
    if not path or not str(path).strip():
        return ToolResult.failure("path is required")
    try:
        resolved = resolve_path(str(path))
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists() or not resolved.is_dir():
        return ToolResult.failure(f"not a directory: {resolved}")
    result = open_path(str(resolved))
    if not result.ok:
        return ToolResult.failure(
            f"open_folder failed: {safe_truncate(result.stderr, 300)}"
        )
    return ToolResult.success(output={"path": str(resolved)})


_MEDIA_ACTIONS = ("play", "pause", "playpause", "next", "previous", "stop")


def _media_control(action: str) -> ToolResult:
    """Send a media transport control (play/pause/next/previous/stop)."""
    if not action:
        return ToolResult.failure("action is required")
    act = str(action).strip().lower()
    if act == "prev":
        act = "previous"
    if act not in _MEDIA_ACTIONS:
        return ToolResult.failure(
            f"unsupported action {action!r}; use one of {list(_MEDIA_ACTIONS)}"
        )
    if IS_WINDOWS:
        keys = {
            "play": "^{MEDIA_PLAY_PAUSE}",
            "pause": "^{MEDIA_PLAY_PAUSE}",
            "playpause": "^{MEDIA_PLAY_PAUSE}",
            "next": "^{MEDIA_NEXT_TRACK}",
            "previous": "^{MEDIA_PREV_TRACK}",
            "stop": "^{MEDIA_STOP}",
        }
        # Windows SendKeys does not expose media keys reliably; fall back to a
        # PowerShell + WScript.Shell dance.
        vk_map = {
            "play": 179, "pause": 179, "playpause": 179,
            "next": 176, "previous": 177, "stop": 178,
        }
        vk = vk_map[act]
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            f"$sig = @'\n"
            "using System.Runtime.InteropServices;\n"
            "public static class K { [DllImport(\"user32.dll\")] "
            "public static extern void keybd_event(byte v, byte s, uint f, "
            "System.UIntPtr e); }\n'@;"
            "Add-Type -TypeDefinition $sig -Language CSharp;"
            f"[K]::keybd_event({vk}, 0, 0, [System.UIntPtr]::Zero);"
            f"[K]::keybd_event({vk}, 0, 2, [System.UIntPtr]::Zero);"
        )
        del keys  # kept for reference; VK path is used
        result = run_command(
            ["powershell", "-NoProfile", "-Command", script], timeout=_QUERY_TIMEOUT
        )
        if not result.ok:
            return ToolResult.failure(
                f"media_control failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"action": act})
    if IS_LINUX:
        if not which("playerctl"):
            return ToolResult.failure("playerctl is not installed")
        mapping = {
            "play": "play",
            "pause": "pause",
            "playpause": "play-pause",
            "next": "next",
            "previous": "previous",
            "stop": "stop",
        }
        result = run_command(["playerctl", mapping[act]], timeout=_QUERY_TIMEOUT)
        if not result.ok:
            return ToolResult.failure(
                f"playerctl failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"action": act})
    if IS_MAC:
        if not which("osascript"):
            return ToolResult.failure("osascript is unavailable")
        applescript_map = {
            "play": 'tell application "Music" to play',
            "pause": 'tell application "Music" to pause',
            "playpause": 'tell application "Music" to playpause',
            "next": 'tell application "Music" to next track',
            "previous": 'tell application "Music" to previous track',
            "stop": 'tell application "Music" to stop',
        }
        result = run_command(
            ["osascript", "-e", applescript_map[act]], timeout=_QUERY_TIMEOUT
        )
        if not result.ok:
            return ToolResult.failure(
                f"osascript failed: {safe_truncate(result.stderr, 300)}"
            )
        return ToolResult.success(output={"action": act})
    return ToolResult.failure("media_control: unsupported platform")


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the built-in application-control tools bound to *ctx*."""
    del ctx  # unused
    tools: List[Tool] = [
        FunctionTool(
            _launch_app,
            name="launch_app",
            description="Launch an application by short name or executable.",
        ),
        FunctionTool(
            _close_app,
            name="close_app",
            description="Ask an application to exit (may lose unsaved work).",
            dangerous=True,
        ),
        FunctionTool(
            _list_windows,
            name="list_windows",
            description="List visible top-level windows.",
        ),
        FunctionTool(
            _focus_window,
            name="focus_window",
            description="Focus a window by title substring.",
        ),
        FunctionTool(
            _open_file_with,
            name="open_file_with",
            description="Open a file with the OS default handler or a specific app.",
        ),
        FunctionTool(
            _open_folder,
            name="open_folder",
            description="Show a folder in the file browser.",
        ),
        FunctionTool(
            _media_control,
            name="media_control",
            description="Send a media transport control (play/pause/next/previous/stop).",
        ),
    ]
    return tools


__all__ = ["build_tools"]
