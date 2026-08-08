"""A system-tray presence, so JARVIS is a resident application, not a terminal.

The icon is drawn in code with PIL — a small arc-reactor disc whose colour
tracks what JARVIS is doing (idle, listening, thinking, speaking, muted).  No
binary asset ships with the package and nothing is downloaded: the icon is
sixteen lines of geometry, and generating it means it can be re-rendered at any
size for any DPI.

``pystray``'s :meth:`run` blocks forever, which is useless to an application
that also has a voice loop, so the icon is driven with ``run_detached()`` where
the backend supports it and a daemon thread otherwise.  Menu callbacks arrive on
pystray's own thread, so — exactly as with the keyboard hook — the real work is
handed to a :class:`~jarvis.win.hotkey.CallbackDispatcher` and the UI thread
returns immediately.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..core import platform_utils
from .hotkey import CallbackDispatcher

log = logging.getLogger(__name__)


#: What each state looks like.  Unknown states fall back to ``idle`` rather
#: than raising — a tray icon is not worth crashing an assistant over.
STATE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "idle": (72, 132, 186),
    "listening": (0, 200, 140),
    "thinking": (247, 181, 41),
    "speaking": (0, 176, 255),
    "muted": (122, 128, 138),
    "error": (214, 69, 65),
    "offline": (92, 96, 104),
}

DEFAULT_STATE = "idle"


def _import_pystray() -> Optional[Any]:
    """Import ``pystray`` lazily; ``None`` when it is missing or unusable."""
    try:
        import pystray  # type: ignore

        return pystray
    except ImportError as exc:
        log.debug("pystray unavailable: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - pystray probes the desktop on import
        log.debug("pystray failed to import: %s", exc)
        return None


def render_icon(state: str = DEFAULT_STATE, size: int = 64) -> Optional[Any]:
    """Draw the tray icon for ``state``.  Returns ``None`` without PIL.

    The image is an "arc reactor": a dark disc, a bright ring in the state
    colour, and a pale core, which stays legible at 16x16 in a crowded tray.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("PIL unusable: %s", exc)
        return None

    try:
        size = max(16, int(size))
        colour = STATE_COLOURS.get(str(state).strip().lower(), STATE_COLOURS[DEFAULT_STATE])
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        edge = size * 0.05
        draw.ellipse(
            [edge, edge, size - edge - 1, size - edge - 1],
            fill=(16, 20, 26, 255),
            outline=colour + (255,),
            width=max(1, size // 20),
        )
        ring = size * 0.24
        draw.ellipse(
            [ring, ring, size - ring - 1, size - ring - 1],
            outline=colour + (255,),
            width=max(2, size // 12),
        )
        core = size * 0.38
        draw.ellipse(
            [core, core, size - core - 1, size - core - 1],
            fill=(236, 248, 255, 255),
        )
        return image
    except Exception as exc:  # noqa: BLE001 - never let decoration break the app
        log.debug("could not render the tray icon: %s", exc)
        return None


class TrayIcon:
    """JARVIS in the notification area: state at a glance, control on right-click.

    All callbacks are optional and all of them run on a worker thread, so a
    handler that takes two seconds to restart the microphone does not freeze
    the user's tray.
    """

    #: Menu labels, kept as class attributes so tests and translations have a
    #: single place to look.
    LABEL_LISTENING = "Listening"
    LABEL_MUTE = "Mute voice"
    LABEL_LOGS = "Show log folder"
    LABEL_RESTART = "Restart listening"
    LABEL_QUIT = "Quit"

    def __init__(
        self,
        on_toggle_listen: Optional[Callable[[bool], Any]] = None,
        on_quit: Optional[Callable[[], Any]] = None,
        *,
        on_toggle_mute: Optional[Callable[[bool], Any]] = None,
        on_restart_listen: Optional[Callable[[], Any]] = None,
        on_open_logs: Optional[Callable[[], Any]] = None,
        log_dir: Optional[Path] = None,
        name: str = "jarvis",
        title: str = "JARVIS",
        state: str = DEFAULT_STATE,
        listening: bool = True,
        muted: bool = False,
        icon_factory: Optional[Callable[[str], Any]] = None,
        dispatcher: Optional[CallbackDispatcher] = None,
    ) -> None:
        self._on_toggle_listen = on_toggle_listen
        self._on_quit = on_quit
        self._on_toggle_mute = on_toggle_mute
        self._on_restart_listen = on_restart_listen
        self._on_open_logs = on_open_logs
        self._log_dir = Path(log_dir) if log_dir else None
        self._name = name
        self._title = title
        self._state = str(state or DEFAULT_STATE)
        self._listening = bool(listening)
        self._muted = bool(muted)
        self._icon_factory = icon_factory or (lambda s: render_icon(s))
        self._dispatcher = dispatcher or CallbackDispatcher(name="jarvis-tray")

        self._icon: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """True when a tray icon can actually be drawn and shown.  Never raises."""
        try:
            if _import_pystray() is None:
                self.last_error = "pystray is not installed"
                return False
            if self._make_image(DEFAULT_STATE) is None:
                self.last_error = "Pillow (PIL) is not installed, so no icon can be drawn"
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.debug("tray availability probe failed", exc_info=True)
            return False

    def _make_image(self, state: str) -> Optional[Any]:
        try:
            return self._icon_factory(state)
        except Exception as exc:  # noqa: BLE001
            log.debug("icon factory failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """Show the icon without blocking.  False when the tray is unusable."""
        with self._lock:
            if self._icon is not None:
                return True
            pystray = _import_pystray()
            if pystray is None:
                self.last_error = "pystray is not installed"
                return False
            image = self._make_image(self._state)
            if image is None:
                self.last_error = "no icon image could be drawn (is Pillow installed?)"
                return False
            try:
                icon = pystray.Icon(
                    self._name,
                    icon=image,
                    title=self._tooltip(),
                    menu=self._build_menu(pystray),
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"could not create the tray icon: {exc}"
                log.warning("%s", self.last_error)
                return False
            self._icon = icon
            self._dispatcher.start()

            run_detached = getattr(icon, "run_detached", None)
            if callable(run_detached):
                try:
                    run_detached()
                    log.info("tray icon running (detached)")
                    return True
                except Exception as exc:  # noqa: BLE001 - not every backend implements it
                    log.debug("run_detached unsupported (%s); falling back to a thread", exc)

            self._thread = threading.Thread(target=self._run_blocking, name="jarvis-tray",
                                            daemon=True)
            self._thread.start()
            log.info("tray icon running (threaded)")
            return True

    def _run_blocking(self) -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.run()
        except Exception:  # noqa: BLE001 - a dead tray must not kill the process
            log.exception("the tray icon stopped unexpectedly")

    def stop(self) -> None:
        """Remove the icon and stop its threads.  Safe to call twice."""
        with self._lock:
            icon, thread = self._icon, self._thread
            self._icon, self._thread = None, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                log.debug("icon.stop() failed", exc_info=True)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._dispatcher.stop()

    @property
    def running(self) -> bool:
        return self._icon is not None

    # ------------------------------------------------------------------ #
    #  State
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def muted(self) -> bool:
        return self._muted

    def _tooltip(self) -> str:
        return f"{self._title} — {self._state}"

    def set_state(self, state: str) -> bool:
        """Re-render the icon for ``idle``/``listening``/``thinking``/``speaking``.

        Returns True when the visible icon was updated.  An unknown state is
        recorded and drawn in the idle colour rather than rejected.
        """
        self._state = str(state or DEFAULT_STATE).strip().lower() or DEFAULT_STATE
        icon = self._icon
        if icon is None:
            return False
        image = self._make_image(self._state)
        try:
            if image is not None:
                icon.icon = image
            icon.title = self._tooltip()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"could not update the tray icon: {exc}"
            log.debug("%s", self.last_error)
            return False
        self._refresh_menu()
        return True

    def set_listening(self, listening: bool) -> None:
        """Reflect the listening state in the menu tick without invoking callbacks."""
        self._listening = bool(listening)
        self._refresh_menu()

    def set_muted(self, muted: bool) -> None:
        """Reflect the mute state in the menu tick without invoking callbacks."""
        self._muted = bool(muted)
        self._refresh_menu()

    def _refresh_menu(self) -> None:
        icon = self._icon
        update = getattr(icon, "update_menu", None) if icon is not None else None
        if callable(update):
            try:
                update()
            except Exception:  # noqa: BLE001
                log.debug("update_menu failed", exc_info=True)

    def notify(self, title: str, message: str) -> bool:
        """Show a balloon from the tray icon.  False when unsupported."""
        icon = self._icon
        if icon is None:
            return False
        balloon = getattr(icon, "notify", None)
        if not callable(balloon):
            return False
        try:
            # pystray's notify() returns None on success, so "it did not raise"
            # is the only signal available.
            balloon(str(message), str(title))
            return True
        except Exception as exc:  # noqa: BLE001 - several backends raise NotImplementedError
            log.debug("tray notification unsupported: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self, pystray: Any) -> Any:
        item = pystray.MenuItem
        separator = getattr(pystray.Menu, "SEPARATOR", None)
        entries = [
            item(self.LABEL_LISTENING, self._menu_toggle_listen,
                 checked=lambda _item: self._listening),
            item(self.LABEL_MUTE, self._menu_toggle_mute,
                 checked=lambda _item: self._muted),
        ]
        if separator is not None:
            entries.append(separator)
        entries.extend([
            item(self.LABEL_LOGS, self._menu_open_logs),
            item(self.LABEL_RESTART, self._menu_restart),
        ])
        if separator is not None:
            entries.append(separator)
        entries.append(item(self.LABEL_QUIT, self._menu_quit))
        return pystray.Menu(*entries)

    def _dispatch(self, callback: Optional[Callable[..., Any]], *args: Any) -> None:
        if callback is None:
            return
        self._dispatcher.submit(callback, *args)

    def _menu_toggle_listen(self, _icon: Any = None, _item: Any = None) -> None:
        self._listening = not self._listening
        self._refresh_menu()
        self._dispatch(self._on_toggle_listen, self._listening)

    def _menu_toggle_mute(self, _icon: Any = None, _item: Any = None) -> None:
        self._muted = not self._muted
        self._refresh_menu()
        self._dispatch(self._on_toggle_mute, self._muted)

    def _menu_restart(self, _icon: Any = None, _item: Any = None) -> None:
        self._dispatch(self._on_restart_listen)

    def _menu_open_logs(self, _icon: Any = None, _item: Any = None) -> None:
        if self._on_open_logs is not None:
            self._dispatch(self._on_open_logs)
            return
        if self._log_dir is None:
            log.debug("no log folder configured for the tray menu")
            return
        self._dispatch(self._open_log_dir)

    def _open_log_dir(self) -> None:
        target = self._log_dir
        if target is None:
            return
        try:
            platform_utils.ensure_dir(target)
            platform_utils.open_path(str(target))
        except Exception:  # noqa: BLE001
            log.exception("could not open the log folder %s", target)

    def _menu_quit(self, _icon: Any = None, _item: Any = None) -> None:
        self._dispatch(self._on_quit)
        icon = self._icon
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # noqa: BLE001
                log.debug("icon.stop() from the menu failed", exc_info=True)
        with self._lock:
            self._icon = None

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        """A dictionary suitable for ``jarvis doctor``."""
        return {
            "available": self.is_available(),
            "running": self.running,
            "state": self._state,
            "listening": self._listening,
            "muted": self._muted,
            "last_error": self.last_error,
        }


__all__ = ["TrayIcon", "render_icon", "STATE_COLOURS", "DEFAULT_STATE"]
