"""Window management — enumerate, focus, move, snap and close desktop windows.

The Windows implementation talks to ``win32gui``/``win32con``/``win32process``
directly; Linux shells out to ``wmctrl`` and ``xdotool``.  Both paths funnel
through the same title-matching rules and return the same window dictionaries,
so the agent sees one behaviour regardless of platform.

Two things here are deliberate and easy to get wrong:

* ``SetForegroundWindow`` fails *silently* when the calling process does not own
  the foreground window — Windows blocks focus stealing.  Every focus attempt
  is therefore verified with ``GetForegroundWindow`` and reported as a failure
  when it did not take, because a focus call that quietly did nothing sends the
  next ``type_text`` into the wrong application.
* Title matching never guesses.  Exact match wins over case-insensitive match,
  which wins over substring; several substring matches are reported as an
  ambiguity with the candidate list rather than silently resolved.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import IS_LINUX, IS_MAC, IS_WINDOWS, run_command, which
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)

_COMMAND_TIMEOUT = 10.0
_MAX_CANDIDATES_SHOWN = 8
_MAX_CHILD_TEXTS = 200
_MIN_WINDOW_SIZE = 80


# --------------------------------------------------------------------------- #
#  Windows backend
# --------------------------------------------------------------------------- #
@dataclass
class Win32Backend:
    """The pywin32 modules this module needs, bundled so tests can fake them."""

    gui: Any
    con: Any
    process: Any
    api: Any


_BACKEND_LOCK = threading.RLock()
_backend_cache: Dict[str, Any] = {"tried": False, "backend": None}


def reset_backend_cache() -> None:
    """Forget the cached pywin32 bundle (used by the test-suite)."""
    with _BACKEND_LOCK:
        _backend_cache["tried"] = False
        _backend_cache["backend"] = None


def _backend() -> Optional[Win32Backend]:
    """Import pywin32 once and return the cached bundle, or None."""
    if not IS_WINDOWS:
        return None
    with _BACKEND_LOCK:
        if _backend_cache["tried"]:
            return _backend_cache["backend"]
        _backend_cache["tried"] = True
        bundle: Optional[Win32Backend] = None
        try:
            import win32api  # type: ignore
            import win32con  # type: ignore
            import win32gui  # type: ignore
            import win32process  # type: ignore

            bundle = Win32Backend(
                gui=win32gui, con=win32con, process=win32process, api=win32api
            )
        except ImportError:
            log.debug("pywin32 is not installed", exc_info=True)
        _backend_cache["backend"] = bundle
        return bundle


def is_available() -> bool:
    """True when some window backend can be reached.  Never raises."""
    try:
        if _backend() is not None:
            return True
        if IS_LINUX:
            return bool(which("wmctrl") or which("xdotool"))
        return False
    except Exception:  # noqa: BLE001
        log.debug("is_available() check failed", exc_info=True)
        return False


_PSUTIL_CACHE: Dict[str, Any] = {"tried": False, "module": None}


def _psutil() -> Optional[Any]:
    with _BACKEND_LOCK:
        if _PSUTIL_CACHE["tried"]:
            return _PSUTIL_CACHE["module"]
        _PSUTIL_CACHE["tried"] = True
        try:
            import psutil  # type: ignore

            _PSUTIL_CACHE["module"] = psutil
        except ImportError:
            log.debug("psutil is not installed; window process names unavailable")
        return _PSUTIL_CACHE["module"]


def _process_name(pid: Optional[int]) -> Optional[str]:
    """Best-effort executable name for *pid* (None when it cannot be known)."""
    if not pid:
        return None
    psutil = _psutil()
    if psutil is None:
        return None
    try:
        return str(psutil.Process(int(pid)).name())
    except Exception:  # noqa: BLE001 - NoSuchProcess/AccessDenied are both normal
        return None


def _rect_dict(rect: Sequence[int]) -> Dict[str, int]:
    left, top, right, bottom = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _window_info(backend: Win32Backend, hwnd: Any) -> Dict[str, Any]:
    """Build the standard window dictionary for one HWND."""
    gui = backend.gui
    try:
        title = gui.GetWindowText(hwnd) or ""
    except Exception:  # noqa: BLE001
        title = ""
    try:
        rect = _rect_dict(gui.GetWindowRect(hwnd))
    except Exception:  # noqa: BLE001
        rect = {"left": 0, "top": 0, "right": 0, "bottom": 0, "width": 0, "height": 0}
    pid: Optional[int] = None
    try:
        ids = backend.process.GetWindowThreadProcessId(hwnd)
        pid = int(ids[1]) if isinstance(ids, (list, tuple)) and len(ids) > 1 else None
    except Exception:  # noqa: BLE001
        pid = None
    try:
        minimized = bool(gui.IsIconic(hwnd))
    except Exception:  # noqa: BLE001
        minimized = False
    return {
        "handle": int(hwnd),
        "title": str(title),
        "process": _process_name(pid),
        "pid": pid,
        "rect": rect,
        "minimized": minimized,
    }


def _enumerate_win32(backend: Win32Backend, visible_only: bool = True) -> List[Dict[str, Any]]:
    """Enumerate top-level windows through EnumWindows."""
    handles: List[Any] = []

    def collect(hwnd: Any, _extra: Any) -> bool:
        handles.append(hwnd)
        return True

    backend.gui.EnumWindows(collect, None)

    windows: List[Dict[str, Any]] = []
    for hwnd in handles:
        if visible_only:
            try:
                if not backend.gui.IsWindowVisible(hwnd):
                    continue
            except Exception:  # noqa: BLE001
                continue
        info = _window_info(backend, hwnd)
        if visible_only and not info["title"].strip():
            # Untitled visible windows are tool windows, tray hosts and the
            # desktop itself — noise the agent can never usefully target.
            continue
        windows.append(info)
    return windows


# --------------------------------------------------------------------------- #
#  Linux backend
# --------------------------------------------------------------------------- #
def _enumerate_linux(visible_only: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Enumerate windows with ``wmctrl -lpG``."""
    wmctrl = which("wmctrl")
    if not wmctrl:
        return (None, "wmctrl is not installed (apt install wmctrl)")
    result = run_command([wmctrl, "-lpG"], timeout=_COMMAND_TIMEOUT)
    if not result.ok:
        return (None, f"wmctrl failed: {safe_truncate(result.stderr, 300)}")
    windows: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        raw_id, _desktop, raw_pid, x, y, w, h, _host, title = parts
        try:
            handle = int(raw_id, 16)
            pid = int(raw_pid)
            left, top = int(x), int(y)
            width, height = int(w), int(h)
        except ValueError:
            continue
        if visible_only and not title.strip():
            continue
        windows.append(
            {
                "handle": handle,
                "title": title,
                "process": _process_name(pid),
                "pid": pid,
                "rect": {
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "width": width,
                    "height": height,
                },
                "minimized": False,
            }
        )
    return (windows, None)


def _wmctrl(args: Sequence[str]) -> Optional[str]:
    """Run wmctrl.  Returns an error message or None."""
    wmctrl = which("wmctrl")
    if not wmctrl:
        return "wmctrl is not installed (apt install wmctrl)"
    result = run_command([wmctrl, *args], timeout=_COMMAND_TIMEOUT)
    if not result.ok:
        return f"wmctrl failed: {safe_truncate(result.stderr, 300)}"
    return None


def _xdotool(args: Sequence[str]) -> Optional[str]:
    """Run xdotool.  Returns an error message or None."""
    exe = which("xdotool")
    if not exe:
        return "xdotool is not installed (apt install xdotool)"
    result = run_command([exe, *args], timeout=_COMMAND_TIMEOUT)
    if not result.ok:
        return f"xdotool failed: {safe_truncate(result.stderr, 300)}"
    return None


def _hex_id(handle: int) -> str:
    return f"0x{int(handle):08x}"


# --------------------------------------------------------------------------- #
#  Enumeration + matching (platform independent)
# --------------------------------------------------------------------------- #
def _all_windows(visible_only: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Every window the platform can see, or an explanatory error."""
    backend = _backend()
    if backend is not None:
        try:
            return (_enumerate_win32(backend, visible_only), None)
        except Exception as exc:  # noqa: BLE001
            return (None, f"window enumeration failed: {exc}")
    if IS_LINUX:
        return _enumerate_linux(visible_only)
    if IS_MAC:
        return (None, "window management on macOS needs a native helper (not implemented)")
    return (None, "no window backend available on this platform")


def _describe(windows: Sequence[Dict[str, Any]]) -> str:
    shown = [f"{w['title']!r} (handle {w['handle']})" for w in windows[:_MAX_CANDIDATES_SHOWN]]
    if len(windows) > _MAX_CANDIDATES_SHOWN:
        shown.append(f"... and {len(windows) - _MAX_CANDIDATES_SHOWN} more")
    return "; ".join(shown)


def _select_window(
    windows: Sequence[Dict[str, Any]], target: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Pick the one window *target* names, or explain why it cannot be picked.

    Order: window handle, exact title, case-insensitive title, case-insensitive
    substring.  Several matches at the winning level is an ambiguity, never a
    silent pick — choosing the wrong window means typing into the wrong app.
    """
    if target is None or not str(target).strip():
        return (None, "a window title or handle is required")
    raw = str(target).strip()

    handle: Optional[int] = None
    if isinstance(target, int) and not isinstance(target, bool):
        handle = int(target)
    elif raw.isdigit():
        handle = int(raw)
    elif raw.lower().startswith("0x"):
        try:
            handle = int(raw, 16)
        except ValueError:
            handle = None
    if handle is not None:
        for window in windows:
            if int(window["handle"]) == handle:
                return (window, None)

    exact = [w for w in windows if w["title"] == raw]
    if len(exact) == 1:
        return (exact[0], None)
    if len(exact) > 1:
        return (None, f"{len(exact)} windows are titled exactly {raw!r}: {_describe(exact)} "
                      "— pass the handle instead")

    lowered = raw.lower()
    insensitive = [w for w in windows if w["title"].lower() == lowered]
    if len(insensitive) == 1:
        return (insensitive[0], None)
    if len(insensitive) > 1:
        return (None, f"{len(insensitive)} windows match {raw!r} ignoring case: "
                      f"{_describe(insensitive)} — pass the handle instead")

    partial = [w for w in windows if lowered in w["title"].lower()]
    if len(partial) == 1:
        return (partial[0], None)
    if len(partial) > 1:
        return (None, f"{len(partial)} windows match {raw!r}: {_describe(partial)} "
                      "— be more specific or pass the handle")

    if not windows:
        return (None, f"no window matches {raw!r} (no windows are open)")
    return (None, f"no window matches {raw!r}; open windows: {_describe(windows)}")


def _resolve(target: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Enumerate then select in one step."""
    windows, error = _all_windows(visible_only=True)
    if error or windows is None:
        return (None, error or "window enumeration failed")
    return _select_window(windows, target)


# --------------------------------------------------------------------------- #
#  Focus
# --------------------------------------------------------------------------- #
def _foreground_handle(backend: Win32Backend) -> Optional[int]:
    try:
        return int(backend.gui.GetForegroundWindow())
    except Exception:  # noqa: BLE001
        return None


def _nudge_with_alt(backend: Win32Backend) -> None:
    """Tap ALT.

    Windows only lets a process set the foreground window if it has recently
    received input.  Synthesising an ALT press is the documented way to satisfy
    that rule when AttachThreadInput is unavailable or refused.
    """
    api = backend.api
    keybd_event = getattr(api, "keybd_event", None)
    if keybd_event is None:
        return
    vk_menu = getattr(backend.con, "VK_MENU", 0x12)
    keyup = getattr(backend.con, "KEYEVENTF_KEYUP", 0x0002)
    try:
        keybd_event(vk_menu, 0, 0, 0)
        keybd_event(vk_menu, 0, keyup, 0)
    except Exception:  # noqa: BLE001
        log.debug("ALT nudge failed", exc_info=True)


def _focus_win32(backend: Win32Backend, hwnd: int) -> ToolResult:
    """Focus *hwnd*, working around the focus-stealing block, and verify it."""
    gui, con = backend.gui, backend.con
    try:
        if gui.IsIconic(hwnd):
            gui.ShowWindow(hwnd, con.SW_RESTORE)
    except Exception:  # noqa: BLE001
        log.debug("could not restore window before focusing", exc_info=True)

    def try_set() -> None:
        try:
            gui.SetForegroundWindow(hwnd)
        except Exception:  # noqa: BLE001
            # SetForegroundWindow raises pywintypes.error when it is refused;
            # that is not fatal, the verification below decides the outcome.
            log.debug("SetForegroundWindow was refused", exc_info=True)

    try_set()
    if _foreground_handle(backend) == int(hwnd):
        return ToolResult.success({"handle": int(hwnd), "method": "direct"})

    # Attach our input queue to the current foreground thread's: while attached
    # Windows treats us as the foreground thread and permits the change.
    attached = False
    current_thread: Optional[int] = None
    foreground_thread: Optional[int] = None
    try:
        foreground = _foreground_handle(backend)
        if foreground:
            foreground_thread = int(backend.process.GetWindowThreadProcessId(foreground)[0])
        get_thread = getattr(backend.api, "GetCurrentThreadId", None)
        current_thread = int(get_thread()) if get_thread is not None else None
        if (
            foreground_thread
            and current_thread
            and foreground_thread != current_thread
        ):
            attached = bool(
                backend.process.AttachThreadInput(current_thread, foreground_thread, True)
            )
    except Exception:  # noqa: BLE001
        log.debug("AttachThreadInput failed", exc_info=True)
    try:
        _nudge_with_alt(backend)
        try_set()
    finally:
        if attached and current_thread and foreground_thread:
            try:
                backend.process.AttachThreadInput(current_thread, foreground_thread, False)
            except Exception:  # noqa: BLE001
                log.debug("detaching the input queue failed", exc_info=True)

    if _foreground_handle(backend) == int(hwnd):
        return ToolResult.success(
            {"handle": int(hwnd), "method": "attach_thread_input" if attached else "alt_nudge"}
        )
    return ToolResult.failure(
        f"focus_window: Windows refused to move focus to handle {int(hwnd)} "
        "(another application is holding the foreground); try again after "
        "clicking on the desktop, or launch JARVIS with the same privilege "
        "level as the target window"
    )


def _focus_linux(window: Dict[str, Any]) -> ToolResult:
    handle = int(window["handle"])
    error = _wmctrl(["-i", "-a", _hex_id(handle)])
    if error is not None:
        error = _xdotool(["windowactivate", str(handle)])
        if error is not None:
            return ToolResult.failure(f"focus_window: {error}")
    if which("xdotool"):
        exe = which("xdotool")
        result = run_command([str(exe), "getactivewindow"], timeout=_COMMAND_TIMEOUT)
        if result.ok:
            try:
                active = int(result.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                active = -1
            if active != handle:
                return ToolResult.failure(
                    f"focus_window: focus did not move to {window['title']!r} "
                    f"(active window is {active})"
                )
    return ToolResult.success({"handle": handle, "title": window["title"]})


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #
def _list_windows(visible_only: bool = True) -> ToolResult:
    """List the top-level windows currently open."""
    windows, error = _all_windows(bool(visible_only))
    if error or windows is None:
        return ToolResult.failure(f"list_windows: {error or 'unavailable'}")
    return ToolResult.success({"windows": windows, "count": len(windows)})


def _focus_window(target: str) -> ToolResult:
    """Bring a window to the front and give it the keyboard."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"focus_window: {error or 'not found'}")
    backend = _backend()
    if backend is not None:
        return _focus_win32(backend, int(window["handle"]))
    if IS_LINUX:
        return _focus_linux(window)
    return ToolResult.failure("focus_window: unsupported platform")


def _close_window(target: str) -> ToolResult:
    """Ask a window to close (unsaved work in it may be lost)."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"close_window: {error or 'not found'}")
    handle = int(window["handle"])
    backend = _backend()
    if backend is not None:
        wm_close = getattr(backend.con, "WM_CLOSE", 0x0010)
        try:
            backend.gui.PostMessage(handle, wm_close, 0, 0)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"close_window failed: {exc}")
        return ToolResult.success({"handle": handle, "title": window["title"]})
    if IS_LINUX:
        error = _wmctrl(["-i", "-c", _hex_id(handle)])
        if error is not None:
            return ToolResult.failure(f"close_window: {error}")
        return ToolResult.success({"handle": handle, "title": window["title"]})
    return ToolResult.failure("close_window: unsupported platform")


def _show_window(target: Any, action: str) -> ToolResult:
    """Shared implementation of minimize / maximize / restore."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"{action}_window: {error or 'not found'}")
    handle = int(window["handle"])
    backend = _backend()
    if backend is not None:
        constant = {
            "minimize": "SW_MINIMIZE",
            "maximize": "SW_MAXIMIZE",
            "restore": "SW_RESTORE",
        }[action]
        try:
            backend.gui.ShowWindow(handle, getattr(backend.con, constant))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"{action}_window failed: {exc}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"], "action": action}
        )
    if IS_LINUX:
        if action == "minimize":
            error = _xdotool(["windowminimize", str(handle)])
        elif action == "maximize":
            error = _wmctrl(
                ["-i", "-r", _hex_id(handle), "-b", "add,maximized_vert,maximized_horz"]
            )
        else:
            error = _wmctrl(
                ["-i", "-r", _hex_id(handle), "-b", "remove,maximized_vert,maximized_horz"]
            )
        if error is not None:
            return ToolResult.failure(f"{action}_window: {error}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"], "action": action}
        )
    return ToolResult.failure(f"{action}_window: unsupported platform")


def _minimize_window(target: str) -> ToolResult:
    """Minimise a window to the taskbar."""
    return _show_window(target, "minimize")


def _maximize_window(target: str) -> ToolResult:
    """Maximise a window to fill the screen."""
    return _show_window(target, "maximize")


def _restore_window(target: str) -> ToolResult:
    """Restore a minimised or maximised window to its previous size."""
    return _show_window(target, "restore")


def _screen_metrics(backend: Optional[Win32Backend]) -> Optional[Tuple[int, int]]:
    """Primary screen size, used to keep window geometry on the display."""
    if backend is not None:
        getter = getattr(backend.api, "GetSystemMetrics", None)
        if getter is not None:
            try:
                width, height = int(getter(0)), int(getter(1))
                if width > 0 and height > 0:
                    return (width, height)
            except Exception:  # noqa: BLE001
                log.debug("GetSystemMetrics failed", exc_info=True)
    if IS_LINUX and which("xdotool"):
        exe = which("xdotool")
        result = run_command([str(exe), "getdisplaygeometry"], timeout=_COMMAND_TIMEOUT)
        if result.ok:
            parts = result.stdout.split()
            if len(parts) >= 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except ValueError:
                    return None
    return None


def _bounded_geometry(
    screen: Optional[Tuple[int, int]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Keep a requested rectangle on the display and above a usable size."""
    width = max(_MIN_WINDOW_SIZE, int(width))
    height = max(_MIN_WINDOW_SIZE, int(height))
    if screen is None:
        return (int(x), int(y), width, height)
    screen_w, screen_h = screen
    width = min(width, screen_w)
    height = min(height, screen_h)
    x = min(max(int(x), 0), max(0, screen_w - width))
    y = min(max(int(y), 0), max(0, screen_h - height))
    return (x, y, width, height)


def _move_window(
    target: str,
    x: int,
    y: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> ToolResult:
    """Move (and optionally resize) a window."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"move_window: {error or 'not found'}")
    try:
        new_x, new_y = int(x), int(y)
    except (TypeError, ValueError):
        return ToolResult.failure("move_window: x and y must be numbers")
    rect = window["rect"]
    try:
        new_w = int(width) if width is not None else int(rect["width"])
        new_h = int(height) if height is not None else int(rect["height"])
    except (TypeError, ValueError):
        return ToolResult.failure("move_window: width and height must be numbers")
    handle = int(window["handle"])
    backend = _backend()
    new_x, new_y, new_w, new_h = _bounded_geometry(
        _screen_metrics(backend), new_x, new_y, new_w, new_h
    )
    if backend is not None:
        try:
            backend.gui.MoveWindow(handle, new_x, new_y, new_w, new_h, True)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"move_window failed: {exc}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"],
             "rect": {"left": new_x, "top": new_y, "width": new_w, "height": new_h}}
        )
    if IS_LINUX:
        error = _wmctrl(
            ["-i", "-r", _hex_id(handle), "-e", f"0,{new_x},{new_y},{new_w},{new_h}"]
        )
        if error is not None:
            return ToolResult.failure(f"move_window: {error}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"],
             "rect": {"left": new_x, "top": new_y, "width": new_w, "height": new_h}}
        )
    return ToolResult.failure("move_window: unsupported platform")


_SNAP_SIDES = ("left", "right", "top", "bottom", "maximize")
_SNAP_CHORDS = {
    "left": ["win", "left"],
    "right": ["win", "right"],
    "top": ["win", "up"],
    "bottom": ["win", "down"],
}


def _send_hotkey(keys: Sequence[str]) -> ToolResult:
    """Indirection so window snapping can be tested without a keyboard backend."""
    from .input_tools import send_hotkey

    return send_hotkey(list(keys))


def _snap_window(target: str, side: str) -> ToolResult:
    """Snap a window to a screen edge (left/right/top/bottom) or maximise it.

    On Windows this drives the shell's own Win+arrow snapping, so the result
    matches what the user gets by hand — including Snap Assist.  ``maximize``
    goes through ShowWindow instead, because Win+Up on an already-snapped
    window only half-restores it.
    """
    name = str(side or "").strip().lower()
    if name in ("max", "full", "fullscreen"):
        name = "maximize"
    if name not in _SNAP_SIDES:
        return ToolResult.failure(
            f"snap_window: side must be one of {list(_SNAP_SIDES)}, got {side!r}"
        )
    if name == "maximize":
        return _show_window(target, "maximize")

    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"snap_window: {error or 'not found'}")
    backend = _backend()
    if backend is not None:
        focused = _focus_win32(backend, int(window["handle"]))
        if not focused.ok:
            return ToolResult.failure(
                f"snap_window: cannot snap a window that will not take focus — {focused.error}"
            )
        chord = _SNAP_CHORDS[name]
        sent = _send_hotkey(chord)
        if not sent.ok:
            return ToolResult.failure(f"snap_window: {sent.error}")
        return ToolResult.success(
            {"handle": int(window["handle"]), "title": window["title"],
             "side": name, "chord": chord}
        )
    if IS_LINUX:
        screen = _screen_metrics(None)
        if screen is None:
            return ToolResult.failure(
                "snap_window: cannot read the display geometry (install xdotool)"
            )
        screen_w, screen_h = screen
        halves = {
            "left": (0, 0, screen_w // 2, screen_h),
            "right": (screen_w // 2, 0, screen_w // 2, screen_h),
            "top": (0, 0, screen_w, screen_h // 2),
            "bottom": (0, screen_h // 2, screen_w, screen_h // 2),
        }[name]
        handle = int(window["handle"])
        error = _wmctrl(
            ["-i", "-r", _hex_id(handle), "-b", "remove,maximized_vert,maximized_horz"]
        )
        if error is not None:
            return ToolResult.failure(f"snap_window: {error}")
        error = _wmctrl(
            ["-i", "-r", _hex_id(handle), "-e",
             f"0,{halves[0]},{halves[1]},{halves[2]},{halves[3]}"]
        )
        if error is not None:
            return ToolResult.failure(f"snap_window: {error}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"], "side": name}
        )
    return ToolResult.failure("snap_window: unsupported platform")


def _active_window() -> ToolResult:
    """Report the window that currently has focus."""
    backend = _backend()
    if backend is not None:
        handle = _foreground_handle(backend)
        if not handle:
            return ToolResult.failure("active_window: no window has focus")
        try:
            return ToolResult.success(_window_info(backend, handle))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"active_window failed: {exc}")
    if IS_LINUX:
        exe = which("xdotool")
        if not exe:
            return ToolResult.failure("active_window: xdotool is not installed")
        result = run_command([exe, "getactivewindow"], timeout=_COMMAND_TIMEOUT)
        if not result.ok:
            return ToolResult.failure(
                f"active_window: {safe_truncate(result.stderr, 300)}"
            )
        try:
            handle = int(result.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return ToolResult.failure("active_window: xdotool returned no window id")
        windows, error = _enumerate_linux(visible_only=False)
        for window in windows or []:
            if int(window["handle"]) == handle:
                return ToolResult.success(window)
        return ToolResult.success({"handle": handle, "title": "", "process": None,
                                   "pid": None, "rect": None, "minimized": False})
    return ToolResult.failure("active_window: unsupported platform")


def _window_text(target: str) -> ToolResult:
    """Read a window's title and, on Windows, the text of its child controls."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"window_text: {error or 'not found'}")
    handle = int(window["handle"])
    payload: Dict[str, Any] = {"handle": handle, "title": window["title"], "text": []}
    backend = _backend()
    if backend is not None:
        enum_children = getattr(backend.gui, "EnumChildWindows", None)
        if enum_children is not None:
            texts: List[str] = []

            def collect(child: Any, _extra: Any) -> bool:
                if len(texts) >= _MAX_CHILD_TEXTS:
                    return False
                try:
                    value = backend.gui.GetWindowText(child)
                except Exception:  # noqa: BLE001
                    return True
                if value and str(value).strip():
                    texts.append(str(value))
                return True

            try:
                enum_children(handle, collect, None)
            except Exception:  # noqa: BLE001
                # EnumChildWindows raises when the callback stops it early, and
                # for windows that vanish mid-walk; the texts gathered so far
                # are still worth returning.
                log.debug("EnumChildWindows stopped early", exc_info=True)
            payload["text"] = texts
    return ToolResult.success(payload)


def _set_always_on_top(target: str, on: bool = True) -> ToolResult:
    """Pin a window above all others, or unpin it."""
    window, error = _resolve(target)
    if error or window is None:
        return ToolResult.failure(f"set_always_on_top: {error or 'not found'}")
    handle = int(window["handle"])
    pinned = bool(on)
    backend = _backend()
    if backend is not None:
        con = backend.con
        insert_after = getattr(con, "HWND_TOPMOST", -1) if pinned else getattr(
            con, "HWND_NOTOPMOST", -2
        )
        flags = getattr(con, "SWP_NOMOVE", 0x0002) | getattr(con, "SWP_NOSIZE", 0x0001)
        try:
            backend.gui.SetWindowPos(handle, insert_after, 0, 0, 0, 0, flags)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"set_always_on_top failed: {exc}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"], "always_on_top": pinned}
        )
    if IS_LINUX:
        verb = "add" if pinned else "remove"
        error = _wmctrl(["-i", "-r", _hex_id(handle), "-b", f"{verb},above"])
        if error is not None:
            return ToolResult.failure(f"set_always_on_top: {error}")
        return ToolResult.success(
            {"handle": handle, "title": window["title"], "always_on_top": pinned}
        )
    return ToolResult.failure("set_always_on_top: unsupported platform")


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the window-management tools bound to *ctx*."""
    del ctx  # these tools need no context
    return [
        FunctionTool(
            _list_windows,
            name="list_windows",
            description="List open windows with title, process, pid, rect and state.",
        ),
        FunctionTool(
            _focus_window,
            name="focus_window",
            description="Focus a window by title or handle, verifying that focus moved.",
        ),
        FunctionTool(
            _close_window,
            name="close_window",
            description="Ask a window to close (unsaved work may be lost).",
            dangerous=True,
        ),
        FunctionTool(
            _minimize_window,
            name="minimize_window",
            description="Minimise a window to the taskbar.",
        ),
        FunctionTool(
            _maximize_window,
            name="maximize_window",
            description="Maximise a window to fill the screen.",
        ),
        FunctionTool(
            _restore_window,
            name="restore_window",
            description="Restore a minimised or maximised window.",
        ),
        FunctionTool(
            _move_window,
            name="move_window",
            description="Move and optionally resize a window.",
        ),
        FunctionTool(
            _snap_window,
            name="snap_window",
            description="Snap a window left/right/top/bottom, or maximise it.",
        ),
        FunctionTool(
            _active_window,
            name="active_window",
            description="Report which window currently has focus.",
        ),
        FunctionTool(
            _window_text,
            name="window_text",
            description="Read a window's title and visible control text.",
        ),
        FunctionTool(
            _set_always_on_top,
            name="set_always_on_top",
            description="Pin a window above all others, or unpin it.",
        ),
    ]


__all__ = ["build_tools", "is_available", "reset_backend_cache", "Win32Backend"]
