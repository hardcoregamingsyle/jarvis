"""Keyboard and mouse control — the tools that let JARVIS drive the desktop.

Everything here synthesises real user input, so every entry point is written
defensively: coordinates are clamped to the actual screen, repeat counts and
durations are bounded, and a missing backend produces a
:class:`~jarvis.core.contracts.ToolResult` failure rather than an exception.

Backend preference, in order:

* ``pyautogui`` — cross-platform, handles keyboard and mouse alike.
* ``win32api`` on Windows — lower level, and it keeps working in windows that
  pyautogui's higher-level path can miss.
* ``xdotool`` on Linux / ``osascript`` on macOS — last resorts.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.contracts import Tool, ToolParam, ToolResult, ToolSpec
from ..core.platform_utils import IS_LINUX, IS_MAC, IS_WINDOWS, which
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Bounds.  A language model will happily ask for 99999 of anything.
# --------------------------------------------------------------------------- #
_MAX_TEXT_CHARS = 20000
_MAX_PRESSES = 50
_MAX_CLICKS = 10
_MAX_KEYS_IN_COMBO = 6
_MAX_DURATION = 5.0
_MAX_INTERVAL = 0.5
_MAX_SCROLL = 5000
_COMMAND_TIMEOUT = 10.0


# --------------------------------------------------------------------------- #
#  Backend loading (lazy + cached)
# --------------------------------------------------------------------------- #
_BACKEND_LOCK = threading.RLock()
_backend_cache: Dict[str, Any] = {"pyautogui_tried": False, "pyautogui": None}


def reset_backend_cache() -> None:
    """Forget the cached backend module.

    Used by the test-suite, and useful in real life after a display change
    (docking a laptop) when the backend needs to re-read the screen geometry.
    """
    with _BACKEND_LOCK:
        _backend_cache["pyautogui_tried"] = False
        _backend_cache["pyautogui"] = None


def _configure_pyautogui(module: Any) -> None:
    """Disable pyautogui's corner failsafe and its implicit inter-call sleep.

    FAILSAFE makes pyautogui raise ``FailSafeException`` the instant the pointer
    lands in a screen corner.  JARVIS is the thing moving the pointer, so a
    perfectly legitimate instruction ("click the Start button at 0,1079") would
    abort the call — and if it aborts mid-drag the mouse button is left held
    down.  A half-completed input action is worse than none, so the failsafe is
    turned off deliberately; the real stop button is the user telling JARVIS to
    stop, which the voice loop already honours.

    PAUSE is a global 0.1s sleep pyautogui inserts after *every* call; dictating
    a paragraph one key at a time would spend minutes asleep.  Timing is managed
    explicitly by the ``interval`` arguments instead.
    """
    try:
        module.FAILSAFE = False
        module.PAUSE = 0.0
    except Exception:  # noqa: BLE001 - a stubbed module may reject attributes
        log.debug("could not configure pyautogui globals", exc_info=True)


def _pyautogui() -> Optional[Any]:
    """Import pyautogui once and return the cached module (None if unusable).

    The import is deferred because it is slow (it drags in Pillow and, on X11,
    opens a display connection) and because on a headless machine it raises
    rather than failing to import — hence the broad except.
    """
    with _BACKEND_LOCK:
        if _backend_cache["pyautogui_tried"]:
            return _backend_cache["pyautogui"]
        _backend_cache["pyautogui_tried"] = True
        module: Optional[Any] = None
        try:
            import pyautogui  # type: ignore

            module = pyautogui
        except ImportError:
            log.debug("pyautogui is not installed", exc_info=True)
        except Exception:  # noqa: BLE001 - headless X11 raises KeyError('DISPLAY')
            log.debug("pyautogui is installed but could not initialise", exc_info=True)
        if module is not None:
            _configure_pyautogui(module)
        _backend_cache["pyautogui"] = module
        return module


def _win32_input() -> Optional[Tuple[Any, Any]]:
    """Return ``(win32api, win32con)`` on Windows, else None."""
    if not IS_WINDOWS:
        return None
    try:
        import win32api  # type: ignore
        import win32con  # type: ignore
    except ImportError:
        log.debug("pywin32 is not available", exc_info=True)
        return None
    return (win32api, win32con)


def _xdotool() -> Optional[str]:
    return which("xdotool") if IS_LINUX else None


def _osascript() -> Optional[str]:
    return which("osascript") if IS_MAC else None


def is_available() -> bool:
    """True when some input backend can be reached.  Never raises."""
    try:
        if _pyautogui() is not None:
            return True
        if _win32_input() is not None:
            return True
        return bool(_xdotool() or _osascript())
    except Exception:  # noqa: BLE001
        log.debug("is_available() check failed", exc_info=True)
        return False


def _no_backend(action: str) -> str:
    return (
        f"{action}: no input backend available — install pyautogui "
        f"(`pip install pyautogui`)"
        + ("" if not IS_LINUX else ", or install xdotool")
    )


# --------------------------------------------------------------------------- #
#  Key vocabulary
# --------------------------------------------------------------------------- #
def _build_common_keys() -> Tuple[str, ...]:
    keys = [
        "enter", "tab", "escape", "space", "backspace", "delete", "insert",
        "home", "end", "pageup", "pagedown", "up", "down", "left", "right",
        "ctrl", "alt", "shift", "win", "command", "option", "capslock",
        "numlock", "scrolllock", "printscreen", "pause",
        "volumeup", "volumedown", "volumemute",
        "playpause", "nexttrack", "prevtrack",
    ]
    keys += [f"f{n}" for n in range(1, 25)]
    keys += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    keys += [str(d) for d in range(10)]
    keys += list("`-=[]\\;',./+*")
    return tuple(keys)


_COMMON_KEYS: Tuple[str, ...] = _build_common_keys()

_KEY_ALIASES: Dict[str, str] = {
    "return": "enter",
    "ret": "enter",
    "cr": "enter",
    "newline": "enter",
    "esc": "escape",
    "spacebar": "space",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "page_up": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "page_down": "pagedown",
    "control": "ctrl",
    "ctl": "ctrl",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "caps": "capslock",
    "prtsc": "printscreen",
    "printscr": "printscreen",
    "plus": "+",
    "minus": "-",
}

_META_KEYS = ("win", "windows", "super", "meta", "cmd", "command", "start")


def _normalize_key(raw: Any) -> str:
    """Map the many names a model invents onto the backend's own vocabulary."""
    key = str(raw).strip()
    if not key:
        return ""
    lowered = key.lower()
    if lowered in _META_KEYS:
        return "command" if IS_MAC else "win"
    if lowered == "option":
        return "option" if IS_MAC else "alt"
    lowered = _KEY_ALIASES.get(lowered, lowered)
    # Single printable characters keep their case ("A" is shift+a to a human,
    # but pyautogui accepts the literal character and does the right thing).
    if len(key) == 1:
        return key
    return lowered


def _split_combo(text: str) -> List[str]:
    """Split ``"ctrl+shift+s"`` (or ``"alt tab"``) into its parts.

    A trailing ``+`` is the literal plus key (``"ctrl++"`` is zoom-in), which a
    naive split would swallow.
    """
    stripped = str(text).strip()
    if not stripped:
        return []
    if len(stripped) == 1:
        return [stripped]
    trailing_plus = stripped.endswith("+")
    body = stripped[:-1] if trailing_plus else stripped
    parts = [p for p in re.split(r"[+\s]+", body) if p]
    if trailing_plus:
        parts.append("+")
    return parts


def _parse_keys(value: Any) -> List[str]:
    """Accept ``"ctrl+s"``, ``["ctrl", "s"]`` or ``["ctrl+s"]`` alike.

    A language model produces all three shapes for the same intent, so all
    three have to land on the same key sequence.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: List[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    keys: List[str] = []
    for item in raw_items:
        for part in _split_combo(str(item)):
            normalized = _normalize_key(part)
            if normalized:
                keys.append(normalized)
    return keys


def _known_keys(module: Optional[Any]) -> Tuple[List[str], str]:
    """The accepted key names plus where the list came from."""
    if module is not None:
        candidate = getattr(module, "KEYBOARD_KEYS", None)
        if isinstance(candidate, (list, tuple)) and candidate:
            return ([str(k) for k in candidate], "pyautogui")
    return (list(_COMMON_KEYS), "builtin")


def _validate_key(key: str, module: Optional[Any]) -> Optional[str]:
    """Return an error message when *key* is not something the backend accepts."""
    if not key:
        return "key is required"
    accepted, _source = _known_keys(module)
    if key in accepted or key.lower() in accepted:
        return None
    if len(key) == 1:
        return None
    return (
        f"unknown key {key!r} — call key_names() for the list the backend accepts"
    )


# --------------------------------------------------------------------------- #
#  Numeric guards
# --------------------------------------------------------------------------- #
def _as_int(value: Any, name: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        return (int(round(float(value))), None)
    except (TypeError, ValueError):
        return (None, f"{name} must be a number, got {value!r}")


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _clamp_float(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        return float(_clamp(float(value), low, high))
    except (TypeError, ValueError):
        return fallback


def _clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return int(_clamp(int(round(float(value))), low, high))
    except (TypeError, ValueError):
        return fallback


def _screen_bounds() -> Optional[Tuple[int, int]]:
    """The primary screen size as ``(width, height)``, or None if unknowable."""
    module = _pyautogui()
    if module is not None:
        try:
            size = module.size()
            width, height = int(size[0]), int(size[1])
            if width > 0 and height > 0:
                return (width, height)
        except Exception:  # noqa: BLE001
            log.debug("pyautogui.size() failed", exc_info=True)
    win32 = _win32_input()
    if win32 is not None:
        api, _con = win32
        try:
            width = int(api.GetSystemMetrics(0))
            height = int(api.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return (width, height)
        except Exception:  # noqa: BLE001
            log.debug("GetSystemMetrics failed", exc_info=True)
    return None


def _clamp_point(
    x: Any, y: Any, *, label: str = "coordinate"
) -> Tuple[Optional[Tuple[int, int]], bool, Optional[str]]:
    """Validate and clamp ``(x, y)`` onto the screen.

    Returns ``(point, was_clamped, error)``.  A hallucinated (99999, 99999)
    becomes the bottom-right pixel instead of an exception from the backend.
    """
    xi, err = _as_int(x, f"{label} x")
    if err:
        return (None, False, err)
    yi, err = _as_int(y, f"{label} y")
    if err:
        return (None, False, err)
    bounds = _screen_bounds()
    if bounds is None:
        return (None, False, "screen size is unknown — cannot validate coordinates")
    width, height = bounds
    cx = int(_clamp(int(xi if xi is not None else 0), 0, width - 1))
    cy = int(_clamp(int(yi if yi is not None else 0), 0, height - 1))
    return ((cx, cy), (cx != xi or cy != yi), None)


# --------------------------------------------------------------------------- #
#  Low-level senders
# --------------------------------------------------------------------------- #
def _build_win32_vk() -> Dict[str, int]:
    """Virtual-key codes for the fallback that does not go through pyautogui."""
    table: Dict[str, int] = {
        "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "shift": 0x10,
        "ctrl": 0x11, "alt": 0x12, "pause": 0x13, "capslock": 0x14,
        "escape": 0x1B, "space": 0x20, "pageup": 0x21, "pagedown": 0x22,
        "end": 0x23, "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27,
        "down": 0x28, "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E,
        "win": 0x5B, "numlock": 0x90, "scrolllock": 0x91,
    }
    for index in range(1, 25):
        table[f"f{index}"] = 0x6F + index
    for code in range(ord("a"), ord("z") + 1):
        table[chr(code)] = ord(chr(code).upper())
    for digit in range(10):
        table[str(digit)] = ord(str(digit))
    return table


_WIN32_VK: Dict[str, int] = _build_win32_vk()


def _win32_send_keys(keys: Sequence[str]) -> Optional[str]:
    """Press *keys* as a chord through win32.  Returns an error message or None."""
    win32 = _win32_input()
    if win32 is None:
        return "pywin32 is not available"
    api, con = win32
    codes: List[int] = []
    for key in keys:
        code = _WIN32_VK.get(key.lower())
        if code is None:
            return f"the win32 fallback cannot press {key!r}"
        codes.append(code)
    try:
        for code in codes:
            api.keybd_event(code, 0, 0, 0)
        for code in reversed(codes):
            api.keybd_event(code, 0, con.KEYEVENTF_KEYUP, 0)
    except Exception as exc:  # noqa: BLE001
        return f"win32 keybd_event failed: {exc}"
    return None


def _win32_type(text: str, interval: float) -> Optional[str]:
    """Type *text* as UTF-16 units through win32.  Returns an error or None."""
    win32 = _win32_input()
    if win32 is None:
        return "pywin32 is not available"
    api, con = win32
    unicode_flag = getattr(con, "KEYEVENTF_UNICODE", 0x0004)
    keyup_flag = getattr(con, "KEYEVENTF_KEYUP", 0x0002)
    raw = text.encode("utf-16-le")
    units = [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw) - 1, 2)]
    try:
        for unit in units:
            api.keybd_event(0, unit, unicode_flag, 0)
            api.keybd_event(0, unit, unicode_flag | keyup_flag, 0)
            if interval > 0:
                time.sleep(interval)
    except Exception as exc:  # noqa: BLE001
        return f"win32 keybd_event failed: {exc}"
    return None


def _run_helper(argv: Sequence[str], *, timeout: float = _COMMAND_TIMEOUT) -> Optional[str]:
    """Run a short helper command.  Returns an error message or None."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        return f"could not run {argv[0]}: {exc}"
    try:
        _out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return f"{argv[0]} timed out after {timeout}s"
    if proc.returncode != 0:
        return f"{argv[0]} exited {proc.returncode}: {safe_truncate(err, 300)}"
    return None


def send_hotkey(keys: Sequence[str]) -> ToolResult:
    """Press *keys* together as a chord.  Public so window snapping can reuse it."""
    normalized = [k for k in (_normalize_key(k) for k in keys) if k]
    if not normalized:
        return ToolResult.failure("hotkey: no keys given")
    if len(normalized) > _MAX_KEYS_IN_COMBO:
        return ToolResult.failure(
            f"hotkey: {len(normalized)} keys is more than the {_MAX_KEYS_IN_COMBO} allowed"
        )
    module = _pyautogui()
    if module is not None:
        try:
            module.hotkey(*normalized)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"hotkey failed: {exc}")
        return ToolResult.success({"keys": normalized, "backend": "pyautogui"})
    if IS_WINDOWS:
        error = _win32_send_keys(normalized)
        if error is None:
            return ToolResult.success({"keys": normalized, "backend": "win32"})
        return ToolResult.failure(f"hotkey failed: {error}")
    xdotool = _xdotool()
    if xdotool:
        combo = "+".join("super" if k == "win" else k for k in normalized)
        error = _run_helper([xdotool, "key", "--clearmodifiers", combo])
        if error is None:
            return ToolResult.success({"keys": normalized, "backend": "xdotool"})
        return ToolResult.failure(f"hotkey failed: {error}")
    return ToolResult.failure(_no_backend("hotkey"))


# --------------------------------------------------------------------------- #
#  Clipboard (used by clipboard_paste_text)
# --------------------------------------------------------------------------- #
def _set_clipboard(text: str) -> ToolResult:
    """Put *text* on the system clipboard."""
    payload = "" if text is None else str(text)
    if IS_WINDOWS:
        try:
            import win32clipboard  # type: ignore
        except ImportError:
            win32clipboard = None  # type: ignore[assignment]
        if win32clipboard is not None:
            try:
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(payload, 13)  # CF_UNICODETEXT
                finally:
                    win32clipboard.CloseClipboard()
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"clipboard write failed: {exc}")
            return ToolResult.success({"chars": len(payload), "backend": "win32clipboard"})
        powershell = which("powershell") or which("pwsh")
        if powershell:
            return _pipe_to(
                [powershell, "-NoProfile", "-Command", "$Input | Set-Clipboard"], payload
            )
        return ToolResult.failure("no clipboard mechanism available on this machine")
    if IS_LINUX:
        for exe, args in (
            ("wl-copy", []),
            ("xclip", ["-selection", "clipboard"]),
            ("xsel", ["--clipboard", "--input"]),
        ):
            if which(exe):
                return _pipe_to([exe, *args], payload)
        return ToolResult.failure(
            "no clipboard tool found (install xclip, xsel or wl-clipboard)"
        )
    if IS_MAC:
        if which("pbcopy"):
            return _pipe_to(["pbcopy"], payload)
        return ToolResult.failure("pbcopy is unavailable")
    return ToolResult.failure("clipboard access is not supported on this platform")


def _pipe_to(argv: Sequence[str], payload: str) -> ToolResult:
    """Feed *payload* to a helper process on stdin."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError) as exc:
        return ToolResult.failure(f"could not run {argv[0]}: {exc}")
    try:
        _out, err = proc.communicate(payload, timeout=_COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return ToolResult.failure(f"{argv[0]} timed out")
    if proc.returncode != 0:
        return ToolResult.failure(
            f"{argv[0]} exited {proc.returncode}: {safe_truncate(err, 300)}"
        )
    return ToolResult.success({"chars": len(payload), "backend": argv[0]})


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #
def _type_text(text: str, interval: float = 0.01) -> ToolResult:
    """Type a string into whatever window currently has keyboard focus."""
    if text is None:
        return ToolResult.failure("text is required")
    payload = str(text)
    if not payload:
        return ToolResult.failure("text is empty")
    if len(payload) > _MAX_TEXT_CHARS:
        return ToolResult.failure(
            f"text is {len(payload)} characters, over the {_MAX_TEXT_CHARS} limit — "
            "use clipboard_paste_text for anything this long"
        )
    delay = _clamp_float(interval, 0.0, _MAX_INTERVAL, 0.01)
    report: Dict[str, Any] = {"chars": len(payload), "interval": delay}
    non_ascii = sorted({ch for ch in payload if ord(ch) > 127})
    if non_ascii:
        report["non_ascii"] = "".join(non_ascii[:20])
        report["note"] = (
            "characters outside ASCII may not survive a keystroke-by-keystroke "
            "path; clipboard_paste_text is more reliable for them"
        )

    module = _pyautogui()
    if module is not None:
        try:
            module.write(payload, interval=delay)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"type_text failed: {exc}")
        report["backend"] = "pyautogui"
        return ToolResult.success(report)

    if IS_WINDOWS:
        error = _win32_type(payload, delay)
        if error is None:
            report["backend"] = "win32"
            return ToolResult.success(report)
        return ToolResult.failure(f"type_text failed: {error}")

    xdotool = _xdotool()
    if xdotool:
        error = _run_helper(
            [xdotool, "type", "--clearmodifiers", "--delay", str(int(delay * 1000)), "--", payload],
            timeout=max(_COMMAND_TIMEOUT, len(payload) * delay + 5.0),
        )
        if error is None:
            report["backend"] = "xdotool"
            return ToolResult.success(report)
        return ToolResult.failure(f"type_text failed: {error}")

    osascript = _osascript()
    if osascript:
        escaped = payload.replace("\\", "\\\\").replace('"', '\\"')
        error = _run_helper(
            [osascript, "-e", f'tell application "System Events" to keystroke "{escaped}"']
        )
        if error is None:
            report["backend"] = "osascript"
            return ToolResult.success(report)
        return ToolResult.failure(f"type_text failed: {error}")

    return ToolResult.failure(_no_backend("type_text"))


def _press_key(key: str, presses: int = 1) -> ToolResult:
    """Press a named key such as enter, tab, escape, f5 or up."""
    if key is None or not str(key).strip():
        return ToolResult.failure("key is required")
    module = _pyautogui()
    normalized = _normalize_key(key)
    error = _validate_key(normalized, module)
    if error:
        return ToolResult.failure(f"press_key: {error}")
    count = _clamp_int(presses, 1, _MAX_PRESSES, 1)
    if module is not None:
        try:
            module.press(normalized, presses=count, interval=0.02)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"press_key failed: {exc}")
        return ToolResult.success(
            {"key": normalized, "presses": count, "backend": "pyautogui"}
        )
    for _ in range(count):
        result = send_hotkey([normalized])
        if not result.ok:
            return ToolResult.failure(f"press_key failed: {result.error}")
    return ToolResult.success({"key": normalized, "presses": count, "backend": "fallback"})


def _hotkey(keys: Any = None, combo: Optional[str] = None) -> ToolResult:
    """Press a key combination such as ctrl+s, alt+tab or win+d."""
    parsed = _parse_keys(keys) or _parse_keys(combo)
    if not parsed:
        return ToolResult.failure(
            "hotkey: give either keys=['ctrl','s'] or combo='ctrl+s'"
        )
    return send_hotkey(parsed)


def _mouse_move(x: int, y: int, duration: float = 0.2) -> ToolResult:
    """Move the mouse pointer to an absolute screen position."""
    module = _pyautogui()
    point, clamped, error = _clamp_point(x, y)
    if error or point is None:
        return ToolResult.failure(f"mouse_move: {error or 'invalid coordinates'}")
    seconds = _clamp_float(duration, 0.0, _MAX_DURATION, 0.2)
    if module is not None:
        try:
            module.moveTo(point[0], point[1], duration=seconds)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_move failed: {exc}")
        return ToolResult.success(
            {"x": point[0], "y": point[1], "clamped": clamped, "backend": "pyautogui"}
        )
    win32 = _win32_input()
    if win32 is not None:
        api, _con = win32
        try:
            api.SetCursorPos((point[0], point[1]))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_move failed: {exc}")
        return ToolResult.success(
            {"x": point[0], "y": point[1], "clamped": clamped, "backend": "win32"}
        )
    xdotool = _xdotool()
    if xdotool:
        err = _run_helper([xdotool, "mousemove", str(point[0]), str(point[1])])
        if err is None:
            return ToolResult.success(
                {"x": point[0], "y": point[1], "clamped": clamped, "backend": "xdotool"}
            )
        return ToolResult.failure(f"mouse_move failed: {err}")
    return ToolResult.failure(_no_backend("mouse_move"))


_BUTTONS = ("left", "right", "middle")


def _mouse_click(
    button: str = "left",
    clicks: int = 1,
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> ToolResult:
    """Click the mouse, optionally moving to an absolute position first."""
    name = str(button or "left").strip().lower()
    if name in ("primary",):
        name = "left"
    if name in ("secondary", "context"):
        name = "right"
    if name not in _BUTTONS:
        return ToolResult.failure(
            f"mouse_click: button must be one of {list(_BUTTONS)}, got {button!r}"
        )
    count = _clamp_int(clicks, 1, _MAX_CLICKS, 1)
    point: Optional[Tuple[int, int]] = None
    clamped = False
    if x is not None or y is not None:
        if x is None or y is None:
            return ToolResult.failure("mouse_click: give both x and y, or neither")
        point, clamped, error = _clamp_point(x, y)
        if error:
            return ToolResult.failure(f"mouse_click: {error}")
    module = _pyautogui()
    if module is not None:
        try:
            if point is None:
                module.click(clicks=count, button=name, interval=0.05)
            else:
                module.click(x=point[0], y=point[1], clicks=count, button=name, interval=0.05)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_click failed: {exc}")
        return ToolResult.success(
            {
                "button": name,
                "clicks": count,
                "x": None if point is None else point[0],
                "y": None if point is None else point[1],
                "clamped": clamped,
                "backend": "pyautogui",
            }
        )
    if point is not None:
        moved = _mouse_move(point[0], point[1], duration=0.0)
        if not moved.ok:
            return moved
    win32 = _win32_input()
    if win32 is not None:
        api, _con = win32
        down_up = {
            "left": (0x0002, 0x0004),
            "right": (0x0008, 0x0010),
            "middle": (0x0020, 0x0040),
        }[name]
        try:
            for _ in range(count):
                api.mouse_event(down_up[0], 0, 0, 0, 0)
                api.mouse_event(down_up[1], 0, 0, 0, 0)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_click failed: {exc}")
        return ToolResult.success(
            {
                "button": name,
                "clicks": count,
                "x": None if point is None else point[0],
                "y": None if point is None else point[1],
                "clamped": clamped,
                "backend": "win32",
            }
        )
    xdotool = _xdotool()
    if xdotool:
        index = {"left": "1", "middle": "2", "right": "3"}[name]
        err = _run_helper([xdotool, "click", "--repeat", str(count), index])
        if err is None:
            return ToolResult.success(
                {"button": name, "clicks": count, "clamped": clamped, "backend": "xdotool"}
            )
        return ToolResult.failure(f"mouse_click failed: {err}")
    return ToolResult.failure(_no_backend("mouse_click"))


def _mouse_scroll(
    amount: int, x: Optional[int] = None, y: Optional[int] = None
) -> ToolResult:
    """Scroll the wheel — positive scrolls up, negative scrolls down."""
    clicks, error = _as_int(amount, "amount")
    if error:
        return ToolResult.failure(f"mouse_scroll: {error}")
    clicks = int(_clamp(int(clicks or 0), -_MAX_SCROLL, _MAX_SCROLL))
    if clicks == 0:
        return ToolResult.failure("mouse_scroll: amount must not be zero")
    point: Optional[Tuple[int, int]] = None
    clamped = False
    if x is not None or y is not None:
        if x is None or y is None:
            return ToolResult.failure("mouse_scroll: give both x and y, or neither")
        point, clamped, error = _clamp_point(x, y)
        if error:
            return ToolResult.failure(f"mouse_scroll: {error}")
    module = _pyautogui()
    if module is not None:
        try:
            if point is None:
                module.scroll(clicks)
            else:
                module.scroll(clicks, x=point[0], y=point[1])
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_scroll failed: {exc}")
        return ToolResult.success(
            {"amount": clicks, "clamped": clamped, "backend": "pyautogui"}
        )
    if point is not None:
        moved = _mouse_move(point[0], point[1], duration=0.0)
        if not moved.ok:
            return moved
    win32 = _win32_input()
    if win32 is not None:
        api, _con = win32
        try:
            api.mouse_event(0x0800, 0, 0, clicks * 120, 0)  # MOUSEEVENTF_WHEEL
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_scroll failed: {exc}")
        return ToolResult.success({"amount": clicks, "clamped": clamped, "backend": "win32"})
    xdotool = _xdotool()
    if xdotool:
        index = "4" if clicks > 0 else "5"
        err = _run_helper([xdotool, "click", "--repeat", str(abs(clicks)), index])
        if err is None:
            return ToolResult.success(
                {"amount": clicks, "clamped": clamped, "backend": "xdotool"}
            )
        return ToolResult.failure(f"mouse_scroll failed: {err}")
    return ToolResult.failure(_no_backend("mouse_scroll"))


def _mouse_drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    button: str = "left",
    duration: float = 0.3,
) -> ToolResult:
    """Press the button at (x1, y1), drag to (x2, y2) and release."""
    name = str(button or "left").strip().lower()
    if name not in _BUTTONS:
        return ToolResult.failure(
            f"mouse_drag: button must be one of {list(_BUTTONS)}, got {button!r}"
        )
    start, start_clamped, error = _clamp_point(x1, y1, label="start")
    if error:
        return ToolResult.failure(f"mouse_drag: {error}")
    end, end_clamped, error = _clamp_point(x2, y2, label="end")
    if error:
        return ToolResult.failure(f"mouse_drag: {error}")
    if start is None or end is None:
        return ToolResult.failure("mouse_drag: invalid coordinates")
    seconds = _clamp_float(duration, 0.0, _MAX_DURATION, 0.3)
    module = _pyautogui()
    if module is None:
        return ToolResult.failure(_no_backend("mouse_drag"))
    try:
        module.moveTo(start[0], start[1], duration=0.0)
        module.dragTo(end[0], end[1], duration=seconds, button=name)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"mouse_drag failed: {exc}")
    return ToolResult.success(
        {
            "from": [start[0], start[1]],
            "to": [end[0], end[1]],
            "button": name,
            "clamped": bool(start_clamped or end_clamped),
            "backend": "pyautogui",
        }
    )


def _mouse_position() -> ToolResult:
    """Report where the mouse pointer currently is."""
    module = _pyautogui()
    if module is not None:
        try:
            pos = module.position()
            return ToolResult.success({"x": int(pos[0]), "y": int(pos[1])})
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"mouse_position failed: {exc}")
    win32 = _win32_input()
    if win32 is not None:
        api, _con = win32
        getter = getattr(api, "GetCursorPos", None)
        if getter is not None:
            try:
                pos = getter()
                return ToolResult.success({"x": int(pos[0]), "y": int(pos[1])})
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"mouse_position failed: {exc}")
    return ToolResult.failure(_no_backend("mouse_position"))


def _screen_size() -> ToolResult:
    """Report the primary screen size in pixels."""
    bounds = _screen_bounds()
    if bounds is None:
        return ToolResult.failure(_no_backend("screen_size"))
    return ToolResult.success({"width": bounds[0], "height": bounds[1]})


def _paste_chord() -> List[str]:
    return ["command", "v"] if IS_MAC else ["ctrl", "v"]


def _clipboard_paste_text(text: str, chord: Optional[str] = None) -> ToolResult:
    """Put text on the clipboard and paste it into the focused window.

    Far faster and far more reliable than typing for long or non-ASCII text.
    """
    if text is None:
        return ToolResult.failure("text is required")
    payload = str(text)
    if not payload:
        return ToolResult.failure("text is empty")
    copied = _set_clipboard(payload)
    if not copied.ok:
        # Sending the paste chord now would paste whatever was on the clipboard
        # before, which is a genuinely destructive surprise in an editor.
        return ToolResult.failure(f"clipboard_paste_text: {copied.error}")
    keys = _parse_keys(chord) if chord else _paste_chord()
    if not keys:
        return ToolResult.failure(f"clipboard_paste_text: cannot parse chord {chord!r}")
    pasted = send_hotkey(keys)
    if not pasted.ok:
        return ToolResult.failure(
            f"clipboard_paste_text: text is on the clipboard but the paste "
            f"keystroke failed: {pasted.error}"
        )
    return ToolResult.success(
        {"chars": len(payload), "chord": keys, "clipboard": copied.output}
    )


def _key_names() -> ToolResult:
    """List the key names press_key and hotkey accept."""
    module = _pyautogui()
    keys, source = _known_keys(module)
    cleaned = sorted({str(k) for k in keys if str(k).strip()})
    return ToolResult.success(
        {"keys": cleaned, "count": len(cleaned), "source": source}
    )


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
_HOTKEY_SPEC = ToolSpec(
    name="hotkey",
    description=(
        "Press a key combination, e.g. keys=['ctrl','s'] or combo='ctrl+s'. "
        "Accepts 'alt+tab', 'win+d', ['ctrl','shift','t']."
    ),
    params=(
        ToolParam(
            name="keys",
            type="array",
            description="Keys to press together, or a single combo string.",
            required=False,
            default=None,
        ),
        ToolParam(
            name="combo",
            type="string",
            description="Alternative single-string form, e.g. 'ctrl+alt+delete'.",
            required=False,
            default=None,
        ),
    ),
    dangerous=True,
)


def build_tools(ctx: Any) -> List[Tool]:
    """Return the keyboard/mouse tools bound to *ctx*."""
    del ctx  # these tools need no context
    return [
        FunctionTool(
            _type_text,
            name="type_text",
            description="Type a string into the window that currently has focus.",
            dangerous=True,
        ),
        FunctionTool(
            _press_key,
            name="press_key",
            description="Press a named key (enter, tab, escape, f5, up, ...).",
        ),
        FunctionTool(_hotkey, spec=_HOTKEY_SPEC),
        FunctionTool(
            _mouse_move,
            name="mouse_move",
            description="Move the mouse pointer to an absolute screen position.",
        ),
        FunctionTool(
            _mouse_click,
            name="mouse_click",
            description="Click the mouse, optionally at an absolute position.",
            dangerous=True,
        ),
        FunctionTool(
            _mouse_scroll,
            name="mouse_scroll",
            description="Scroll the wheel; positive is up, negative is down.",
        ),
        FunctionTool(
            _mouse_drag,
            name="mouse_drag",
            description="Press at one point, drag to another, and release.",
            dangerous=True,
        ),
        FunctionTool(
            _mouse_position,
            name="mouse_position",
            description="Report the current mouse pointer position.",
        ),
        FunctionTool(
            _screen_size,
            name="screen_size",
            description="Report the primary screen size in pixels.",
        ),
        FunctionTool(
            _clipboard_paste_text,
            name="clipboard_paste_text",
            description=(
                "Copy text to the clipboard and paste it into the focused window "
                "— faster and safer than typing for long or non-ASCII text."
            ),
            dangerous=True,
        ),
        FunctionTool(
            _key_names,
            name="key_names",
            description="List the key names press_key and hotkey accept.",
        ),
    ]


__all__ = ["build_tools", "send_hotkey", "is_available", "reset_backend_cache"]
