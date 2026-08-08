"""Global hotkeys: summon JARVIS from anywhere, including push-to-talk.

Two backends, tried in order:

``keyboard``
    The installed third-party package.  It hooks the keyboard globally, works
    on Windows and Linux, and — crucially — can report key *release*, which is
    what makes a held push-to-talk key possible.

``win32``
    A ``RegisterHotKey`` fallback driven through :mod:`ctypes` (no pywin32
    needed).  It only ever sees key *presses*, so push-to-talk is emulated with
    a short ``GetAsyncKeyState`` poll on the backend's own thread.

**Honesty about registration.**  On Windows the ``keyboard`` package installs a
low-level hook that, in some configurations (an elevated foreground window, or
a locked-down machine), quietly never starts — leaving an application convinced
it owns a hotkey that will never fire.  Every registration is therefore probed
afterwards and a failure is reported through the return value of
:meth:`HotkeyManager.register` and through :attr:`HotkeyManager.last_error`.
We would rather say "no" than lie.

**Never block the hook thread.**  Callbacks arrive on the hook's own thread, and
whatever runs there delays *every subsequent keystroke on the machine*.  So the
manager hands every callback to :class:`CallbackDispatcher`, which runs it on a
private worker thread; the hook thread only ever performs a queue put.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.platform_utils import IS_WINDOWS

log = logging.getLogger(__name__)


#: Toggle hands-free listening on and off.
DEFAULT_TOGGLE_COMBO = "ctrl+alt+j"
#: Hold to talk, release to send.
DEFAULT_PUSH_TO_TALK_COMBO = "ctrl+alt+space"


# --------------------------------------------------------------------------- #
#  Callback hand-off
# --------------------------------------------------------------------------- #
class CallbackDispatcher:
    """Run callbacks on a private worker thread instead of the caller's.

    Shared by the hotkey hooks and the tray menu.  Both are driven by threads
    owned by the operating system or by a UI toolkit, and a slow callback there
    is not merely rude: on the keyboard hook it stalls typing system-wide.

    :meth:`submit` never blocks and never raises.  If the queue is full the
    callback is dropped and counted in :attr:`dropped` — a visible, countable
    loss is better than an unbounded backlog of stale key presses.
    """

    def __init__(self, *, name: str = "jarvis-callbacks", maxsize: int = 256) -> None:
        self._queue: "queue.Queue" = queue.Queue(maxsize=max(1, maxsize))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._name = name
        self.dropped = 0

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        """Start the worker thread if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Drain and stop the worker.  Safe to call more than once."""
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def pending(self) -> int:
        return self._queue.qsize()

    # -- work ------------------------------------------------------------ #
    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Queue ``fn`` for the worker thread.  Returns False if it was dropped."""
        if not callable(fn):
            return False
        self.start()
        try:
            self._queue.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            self.dropped += 1
            log.warning("hotkey callback queue is full; dropped %s", getattr(fn, "__name__", fn))
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 - one bad callback must not end the worker
                log.exception("hotkey callback raised")


# --------------------------------------------------------------------------- #
#  Combo parsing (pure, and therefore testable on any OS)
# --------------------------------------------------------------------------- #
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001

_MODIFIERS: Dict[str, int] = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "ctl": MOD_CONTROL,
    "alt": MOD_ALT, "menu": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN, "windows": MOD_WIN, "super": MOD_WIN, "meta": MOD_WIN, "cmd": MOD_WIN,
}

#: Virtual-key codes for the keys a hotkey is plausibly bound to.
_VK_NAMES: Dict[str, int] = {
    "space": 0x20, "spacebar": 0x20,
    "enter": 0x0D, "return": 0x0D,
    "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "ins": 0x2D,
    "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "printscreen": 0x2C, "pause": 0x13,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, "\\": 0xDC, "`": 0xC0,
}

#: Which physical keys stand for each modifier bit, for GetAsyncKeyState polling.
_MODIFIER_VKS: Dict[int, Tuple[int, ...]] = {
    MOD_ALT: (0x12,),
    MOD_CONTROL: (0x11,),
    MOD_SHIFT: (0x10,),
    MOD_WIN: (0x5B, 0x5C),
}


def normalise_combo(combo: str) -> str:
    """Canonical spelling of a combo: lower case, no spaces around ``+``."""
    if not combo:
        return ""
    parts = [p.strip().lower() for p in str(combo).split("+")]
    return "+".join(p for p in parts if p)


def parse_combo(combo: str) -> Optional[Tuple[int, int]]:
    """Translate ``"ctrl+alt+j"`` into ``(modifier_mask, virtual_key_code)``.

    Returns ``None`` when the combo has no key, more than one key, or a key
    this module has no virtual-key code for — the caller must then report the
    hotkey as unregistered rather than pretend.
    """
    text = normalise_combo(combo)
    if not text:
        return None

    modifiers = 0
    key: Optional[int] = None
    for part in text.split("+"):
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
            continue
        if key is not None:
            return None  # two non-modifier keys is not a Win32 hotkey
        if len(part) == 1 and (part.isalpha() or part.isdigit()):
            key = ord(part.upper())
        elif part in _VK_NAMES:
            key = _VK_NAMES[part]
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            key = 0x70 + int(part[1:]) - 1
        else:
            return None
    if key is None:
        return None
    return modifiers, key


# --------------------------------------------------------------------------- #
#  Backend: the `keyboard` package
# --------------------------------------------------------------------------- #
def _import_keyboard() -> Tuple[Optional[Any], str]:
    """Import ``keyboard`` lazily.  Returns ``(module, error_message)``.

    On Linux the package raises ``ImportError`` at import time unless the
    process is root, so the failure text is worth keeping for the diagnosis
    printed by ``jarvis doctor``.
    """
    try:
        import keyboard  # type: ignore

        return keyboard, ""
    except ImportError as exc:
        return None, f"keyboard: {exc}"
    except Exception as exc:  # noqa: BLE001 - the package touches devices on import
        return None, f"keyboard: {type(exc).__name__}: {exc}"


class _KeyboardBackend:
    """Hotkeys via the ``keyboard`` package, with a post-registration probe."""

    name = "keyboard"
    supports_release = True

    def __init__(self, module: Any) -> None:
        self._kb = module
        self._handles: Dict[str, List[Any]] = {}
        self.last_error: Optional[str] = None

    def is_available(self) -> bool:
        return self._kb is not None

    # -- internals ------------------------------------------------------- #
    def _hook_is_live(self) -> Optional[bool]:
        """Is the global hook actually listening?  ``None`` when unknowable."""
        listener = getattr(self._kb, "_listener", None)
        if listener is None:
            return None
        listening = getattr(listener, "listening", None)
        if listening is None:
            return None
        return bool(listening)

    def _add_one(self, combo: str, callback: Callable[[], None], *,
                 suppress: bool, on_release: bool) -> Any:
        try:
            return self._kb.add_hotkey(
                combo, callback, suppress=suppress, trigger_on_release=on_release
            )
        except TypeError:
            # A build of `keyboard` with a narrower signature: press-only.
            if on_release:
                raise
            return self._kb.add_hotkey(combo, callback)

    def _drop(self, handles: List[Any]) -> None:
        remove = getattr(self._kb, "remove_hotkey", None)
        for handle in handles:
            if not callable(remove):
                break
            try:
                remove(handle)
            except Exception:  # noqa: BLE001
                log.debug("remove_hotkey failed", exc_info=True)

    _DEAD_HOOK = (
        "the global keyboard hook did not start; on Windows the 'keyboard' package "
        "needs to run with administrator rights for system-wide hotkeys"
    )

    # -- api ------------------------------------------------------------- #
    def add(self, combo: str, callback: Callable[[], None], *, suppress: bool = False) -> Tuple[bool, str]:
        handles: List[Any] = []
        try:
            handles.append(self._add_one(combo, callback, suppress=suppress, on_release=False))
        except Exception as exc:  # noqa: BLE001
            return False, f"keyboard.add_hotkey({combo!r}) failed: {exc}"
        if self._hook_is_live() is False:
            self._drop(handles)
            return False, self._DEAD_HOOK
        self._handles[combo] = handles
        return True, ""

    def add_ptt(
        self,
        combo: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        *,
        suppress: bool = False,
    ) -> Tuple[bool, str]:
        held = threading.Event()

        def _press() -> None:
            # Key auto-repeat fires the press hotkey again and again; a second
            # "begin utterance" mid-utterance would truncate the recording.
            if held.is_set():
                return
            held.set()
            on_press()

        def _release() -> None:
            if not held.is_set():
                return
            held.clear()
            on_release()

        handles: List[Any] = []
        try:
            handles.append(self._add_one(combo, _press, suppress=suppress, on_release=False))
            handles.append(self._add_one(combo, _release, suppress=False, on_release=True))
        except Exception as exc:  # noqa: BLE001
            self._drop(handles)
            return False, f"push-to-talk on {combo!r} failed: {exc}"
        if self._hook_is_live() is False:
            self._drop(handles)
            return False, self._DEAD_HOOK
        self._handles[combo] = handles
        return True, ""

    def remove(self, combo: str) -> bool:
        handles = self._handles.pop(combo, None)
        if not handles:
            return False
        self._drop(handles)
        return True

    def clear(self) -> None:
        for combo in list(self._handles):
            self.remove(combo)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.clear()


# --------------------------------------------------------------------------- #
#  Backend: Win32 RegisterHotKey via ctypes
# --------------------------------------------------------------------------- #
@dataclass
class _PttWatch:
    modifiers: int
    vk: int
    on_press: Callable[[], None]
    on_release: Callable[[], None]
    held: bool = False


@dataclass
class _Win32Entry:
    hotkey_id: int
    callback: Callable[[], None]


class _Win32Backend:
    """``RegisterHotKey`` on a dedicated message-pump thread.

    ``RegisterHotKey`` binds the hotkey to the *calling thread*, and the
    ``WM_HOTKEY`` message is delivered to that thread's message queue.  Every
    registration therefore has to be executed by the pump thread; callers post
    a command and wait briefly for the outcome so that :meth:`add` can return
    an honest success flag.

    Key release is not reported by ``WM_HOTKEY`` at all, so push-to-talk is
    emulated by polling ``GetAsyncKeyState`` on the same thread.
    """

    name = "win32"
    supports_release = True
    _POLL_SECONDS = 0.02
    _COMMAND_TIMEOUT = 3.0

    def __init__(self) -> None:
        self.last_error: Optional[str] = None
        self._commands: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._next_id = 1
        self._entries: Dict[str, _Win32Entry] = {}
        self._ptt: Dict[str, _PttWatch] = {}

    # -- availability ----------------------------------------------------- #
    def _user32(self) -> Optional[Any]:
        if not IS_WINDOWS:
            return None
        try:
            import ctypes

            return ctypes.windll.user32  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - no windll off Windows
            self.last_error = f"win32: {exc}"
            return None

    def is_available(self) -> bool:
        if not IS_WINDOWS:
            self.last_error = "win32: RegisterHotKey exists only on Windows"
            return False
        return self._user32() is not None

    # -- command plumbing -------------------------------------------------- #
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._ready.clear()
            self._thread = threading.Thread(target=self._run, name="jarvis-hotkeys-win32",
                                            daemon=True)
            self._thread.start()
        self._ready.wait(timeout=self._COMMAND_TIMEOUT)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._COMMAND_TIMEOUT)

    def _submit(self, action: str, payload: dict) -> Tuple[bool, str]:
        self.start()
        done = threading.Event()
        box: List[Tuple[bool, str]] = []
        self._commands.put((action, payload, box, done))
        if not done.wait(timeout=self._COMMAND_TIMEOUT):
            return False, f"win32 hotkey thread did not answer the {action} request"
        return box[0] if box else (False, "win32 hotkey thread returned nothing")

    # -- api --------------------------------------------------------------- #
    def add(self, combo: str, callback: Callable[[], None], *, suppress: bool = False) -> Tuple[bool, str]:
        parsed = parse_combo(combo)
        if parsed is None:
            return False, f"{combo!r} is not a combo RegisterHotKey understands"
        return self._submit("add", {"combo": combo, "parsed": parsed, "callback": callback})

    def add_ptt(
        self,
        combo: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        *,
        suppress: bool = False,
    ) -> Tuple[bool, str]:
        parsed = parse_combo(combo)
        if parsed is None:
            return False, f"{combo!r} is not a combo this backend can watch"
        return self._submit("add_ptt", {
            "combo": combo, "parsed": parsed,
            "on_press": on_press, "on_release": on_release,
        })

    def remove(self, combo: str) -> bool:
        if combo not in self._entries and combo not in self._ptt:
            return False
        ok, _ = self._submit("remove", {"combo": combo})
        return ok

    def clear(self) -> None:
        for combo in list(self._entries) + list(self._ptt):
            self.remove(combo)

    # -- the pump ---------------------------------------------------------- #
    def _run(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"win32: {exc}"
            self._ready.set()
            return

        user32 = self._user32()
        if user32 is None:
            self._ready.set()
            return
        user32.GetAsyncKeyState.restype = ctypes.c_short
        msg = wintypes.MSG()
        self._ready.set()

        try:
            while not self._stop.is_set():
                self._drain_commands(user32)
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY:
                        self._fire(int(msg.wParam))
                self._poll_push_to_talk(user32)
                time.sleep(self._POLL_SECONDS)
        finally:
            for entry in self._entries.values():
                try:
                    user32.UnregisterHotKey(None, entry.hotkey_id)
                except Exception:  # noqa: BLE001
                    log.debug("UnregisterHotKey failed", exc_info=True)
            self._entries.clear()
            self._ptt.clear()

    def _drain_commands(self, user32: Any) -> None:
        while True:
            try:
                action, payload, box, done = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                box.append(self._apply(user32, action, payload))
            except Exception as exc:  # noqa: BLE001
                box.append((False, f"win32 {action} failed: {exc}"))
            finally:
                done.set()

    def _apply(self, user32: Any, action: str, payload: dict) -> Tuple[bool, str]:
        combo = payload.get("combo", "")
        if action == "add":
            modifiers, vk = payload["parsed"]
            hotkey_id = self._next_id
            self._next_id += 1
            if not user32.RegisterHotKey(None, hotkey_id, modifiers | MOD_NOREPEAT, vk):
                return False, f"RegisterHotKey refused {combo!r} (another program may own it)"
            self._entries[combo] = _Win32Entry(hotkey_id, payload["callback"])
            return True, ""
        if action == "add_ptt":
            modifiers, vk = payload["parsed"]
            self._ptt[combo] = _PttWatch(modifiers, vk, payload["on_press"], payload["on_release"])
            return True, ""
        if action == "remove":
            entry = self._entries.pop(combo, None)
            if entry is not None:
                user32.UnregisterHotKey(None, entry.hotkey_id)
            watch = self._ptt.pop(combo, None)
            return (entry is not None or watch is not None), ""
        return False, f"unknown command {action!r}"

    def _fire(self, hotkey_id: int) -> None:
        for entry in self._entries.values():
            if entry.hotkey_id == hotkey_id:
                try:
                    entry.callback()
                except Exception:  # noqa: BLE001
                    log.exception("hotkey callback raised")
                return

    def _poll_push_to_talk(self, user32: Any) -> None:
        if not self._ptt:
            return
        for watch in self._ptt.values():
            down = bool(user32.GetAsyncKeyState(watch.vk) & 0x8000)
            if down:
                for bit, vks in _MODIFIER_VKS.items():
                    if watch.modifiers & bit and not any(
                        user32.GetAsyncKeyState(v) & 0x8000 for v in vks
                    ):
                        down = False
                        break
            if down and not watch.held:
                watch.held = True
                watch.on_press()
            elif not down and watch.held:
                watch.held = False
                watch.on_release()


# --------------------------------------------------------------------------- #
#  Manager
# --------------------------------------------------------------------------- #
@dataclass
class Registration:
    """One live hotkey, as the manager sees it."""

    combo: str
    kind: str                 # "hotkey" | "push-to-talk"
    suppress: bool = False
    backend: str = ""
    callbacks: tuple = field(default_factory=tuple)


class HotkeyManager:
    """Register global hotkeys and run their callbacks off the hook thread.

    Usage::

        with HotkeyManager() as keys:
            if not keys.register("ctrl+alt+j", toggle_listening):
                log.warning("no hotkey: %s", keys.last_error)

    Every public method returns a value rather than raising, so a machine that
    forbids global hooks degrades to "JARVIS runs, but you must click it".
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        dispatcher: Optional[CallbackDispatcher] = None,
    ) -> None:
        self._backend_choice = backend
        self._backend: Optional[Any] = None
        self._resolved = False
        self._dispatcher = dispatcher or CallbackDispatcher(name="jarvis-hotkeys")
        self._registrations: Dict[str, Registration] = {}
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None

    # -- backend ---------------------------------------------------------- #
    def _resolve(self) -> Optional[Any]:
        with self._lock:
            if self._resolved:
                return self._backend
            self._resolved = True
            order = (["keyboard", "win32"] if self._backend_choice == "auto"
                     else [self._backend_choice])
            problems: List[str] = []
            for name in order:
                if name == "keyboard":
                    module, error = _import_keyboard()
                    if module is not None:
                        self._backend = _KeyboardBackend(module)
                        return self._backend
                    problems.append(error)
                elif name == "win32":
                    candidate = _Win32Backend()
                    if candidate.is_available():
                        self._backend = candidate
                        return self._backend
                    problems.append(candidate.last_error or "win32: unavailable")
                else:
                    problems.append(f"unknown hotkey backend {name!r}")
            self.last_error = "; ".join(problems) or "no hotkey backend is usable"
            return None

    @property
    def backend_name(self) -> str:
        backend = self._resolve()
        return getattr(backend, "name", "none")

    def is_available(self) -> bool:
        """True when some backend can plausibly register a global hotkey."""
        try:
            return self._resolve() is not None
        except Exception as exc:  # noqa: BLE001 - availability never raises
            self.last_error = str(exc)
            log.debug("hotkey availability probe failed", exc_info=True)
            return False

    # -- registration ------------------------------------------------------ #
    def _wrap(self, callback: Callable[..., Any]) -> Callable[..., None]:
        """Return a hook-thread-safe shim that only enqueues work."""

        def _fire(*_args: Any, **_kwargs: Any) -> None:
            self._dispatcher.submit(callback)

        return _fire

    def register(self, combo: str, callback: Callable[..., Any], *, suppress: bool = False) -> bool:
        """Bind ``combo``.  Returns False (and sets ``last_error``) if it did not take."""
        key = normalise_combo(combo)
        if not key:
            self.last_error = "an empty combo cannot be registered"
            return False
        if not callable(callback):
            self.last_error = f"the callback for {key!r} is not callable"
            return False
        backend = self._resolve()
        if backend is None:
            return False

        with self._lock:
            self._remove_locked(key)
            ok, error = self._guarded(backend.add, key, self._wrap(callback), suppress=suppress)
            if not ok:
                self.last_error = error
                log.warning("hotkey %s not registered: %s", key, error)
                return False
            self._registrations[key] = Registration(
                combo=key, kind="hotkey", suppress=suppress,
                backend=getattr(backend, "name", ""), callbacks=(callback,),
            )
        self._dispatcher.start()
        log.info("hotkey registered: %s", key)
        return True

    def register_push_to_talk(
        self,
        combo: str,
        on_press: Callable[..., Any],
        on_release: Callable[..., Any],
        *,
        suppress: bool = False,
    ) -> bool:
        """Bind a *held* key: ``on_press`` when it goes down, ``on_release`` when it comes up."""
        key = normalise_combo(combo)
        if not key:
            self.last_error = "an empty combo cannot be registered"
            return False
        if not callable(on_press) or not callable(on_release):
            self.last_error = f"push-to-talk on {key!r} needs two callables"
            return False
        backend = self._resolve()
        if backend is None:
            return False
        if not getattr(backend, "supports_release", False):
            self.last_error = f"the {getattr(backend, 'name', '?')} backend cannot detect key release"
            return False

        with self._lock:
            self._remove_locked(key)
            ok, error = self._guarded(
                backend.add_ptt, key, self._wrap(on_press), self._wrap(on_release),
                suppress=suppress,
            )
            if not ok:
                self.last_error = error
                log.warning("push-to-talk %s not registered: %s", key, error)
                return False
            self._registrations[key] = Registration(
                combo=key, kind="push-to-talk", suppress=suppress,
                backend=getattr(backend, "name", ""), callbacks=(on_press, on_release),
            )
        self._dispatcher.start()
        log.info("push-to-talk registered: %s", key)
        return True

    def bind_push_to_talk(
        self,
        voice_loop: Any,
        *,
        combo: str = DEFAULT_PUSH_TO_TALK_COMBO,
        suppress: bool = False,
    ) -> bool:
        """Wire a held key to ``voice_loop.begin_utterance()``/``end_utterance()``.

        Those methods are looked up with ``getattr`` at call time: a voice loop
        that does not implement them yet simply logs, so the hotkey remains
        harmless rather than raising on every key press.
        """
        press, release = voice_loop_callbacks(voice_loop)
        return self.register_push_to_talk(combo, press, release, suppress=suppress)

    @staticmethod
    def _guarded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[bool, str]:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a backend must not take us down
            log.debug("hotkey backend raised", exc_info=True)
            return False, f"{type(exc).__name__}: {exc}"
        if isinstance(result, tuple):
            ok, error = result
            return bool(ok), str(error or "")
        return bool(result), "" if result else "the backend refused the registration"

    # -- removal ----------------------------------------------------------- #
    def _remove_locked(self, key: str) -> bool:
        if key not in self._registrations:
            return False
        backend = self._backend
        if backend is not None:
            try:
                backend.remove(key)
            except Exception:  # noqa: BLE001
                log.debug("backend.remove(%s) failed", key, exc_info=True)
        self._registrations.pop(key, None)
        return True

    def unregister(self, combo: str) -> bool:
        """Release one hotkey.  False when it was not registered here."""
        with self._lock:
            return self._remove_locked(normalise_combo(combo))

    def unregister_all(self) -> None:
        """Release every hotkey this manager owns.  Never raises."""
        with self._lock:
            for key in list(self._registrations):
                self._remove_locked(key)
            backend = self._backend
        if backend is not None:
            try:
                backend.clear()
            except Exception:  # noqa: BLE001
                log.debug("backend.clear() failed", exc_info=True)

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> bool:
        """Start the dispatcher and the backend.  False when nothing is usable."""
        backend = self._resolve()
        self._dispatcher.start()
        if backend is None:
            return False
        try:
            backend.start()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.debug("backend.start() failed", exc_info=True)
            return False
        return True

    def stop(self) -> None:
        """Release everything and stop the worker threads.  Never raises."""
        self.unregister_all()
        backend = self._backend
        if backend is not None:
            try:
                backend.stop()
            except Exception:  # noqa: BLE001
                log.debug("backend.stop() failed", exc_info=True)
        self._dispatcher.stop()

    def __enter__(self) -> "HotkeyManager":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.stop()
        return False

    # -- introspection ------------------------------------------------------ #
    def registered(self) -> List[str]:
        with self._lock:
            return sorted(self._registrations)

    def status(self) -> dict:
        """A dictionary suitable for ``jarvis doctor``."""
        return {
            "available": self.is_available(),
            "backend": self.backend_name,
            "registered": self.registered(),
            "dropped_callbacks": self._dispatcher.dropped,
            "last_error": self.last_error,
        }


def voice_loop_callbacks(voice_loop: Any) -> Tuple[Callable[[], None], Callable[[], None]]:
    """Build ``(on_press, on_release)`` for a :class:`~jarvis.voice.VoiceLoop`.

    Resolved at call time via ``getattr`` so this keeps working whether or not
    the voice loop has grown ``begin_utterance``/``end_utterance`` yet.
    """

    def _call(name: str) -> None:
        method = getattr(voice_loop, name, None)
        if not callable(method):
            log.debug("voice loop has no %s(); push-to-talk press ignored", name)
            return
        method()

    return (lambda: _call("begin_utterance"), lambda: _call("end_utterance"))


def default_hotkeys() -> Dict[str, str]:
    """The suggested bindings, for the CLI and documentation to agree on."""
    return {"toggle_listening": DEFAULT_TOGGLE_COMBO,
            "push_to_talk": DEFAULT_PUSH_TO_TALK_COMBO}


__all__ = [
    "HotkeyManager",
    "CallbackDispatcher",
    "Registration",
    "DEFAULT_TOGGLE_COMBO",
    "DEFAULT_PUSH_TO_TALK_COMBO",
    "default_hotkeys",
    "normalise_combo",
    "parse_combo",
    "voice_loop_callbacks",
]
