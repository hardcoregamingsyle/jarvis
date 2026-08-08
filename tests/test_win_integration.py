"""Desktop integration: hotkeys, tray icon, autostart, notifications.

Everything here is hermetic.  No global hotkey is ever registered, no tray icon
is ever shown, the real registry is never touched (``winreg`` is replaced by an
in-memory fake), and no notification subprocess is ever launched.  Every test
also has to pass on Linux, where none of these facilities exist — which is
exactly the property the modules promise.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types

import pytest

from jarvis.core import platform_utils
from jarvis.win import autostart, hotkey, tray

# ``jarvis.win`` re-exports the notify() *function* under the name ``notify``,
# which shadows the submodule of the same name on the package object.  Reach
# the module through sys.modules so the tests can patch its internals.
notify_mod = importlib.import_module("jarvis.win.notify")


def block_package(monkeypatch, prefix: str) -> None:
    """Make ``import <prefix>`` fail, as it would on a machine without it.

    Dropping the entry from ``sys.modules`` is not enough: a stale
    ``PIL.Image`` left there by another test satisfies ``from PIL import
    Image`` even when ``PIL`` itself has been replaced.  So the submodules go
    too, and a meta-path finder refuses any fresh attempt.
    """
    for name in [n for n in list(sys.modules) if n == prefix or n.startswith(prefix + ".")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == prefix or name.startswith(prefix + "."):
                raise ImportError(f"{name} is blocked by the test")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + list(sys.meta_path))


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
class FakeKeyboard:
    """Enough of the ``keyboard`` package to drive _KeyboardBackend.

    ``listening`` models the one failure that matters in practice: on Windows
    the low-level hook may never start, in which case the package accepts the
    registration and silently does nothing.
    """

    def __init__(self, *, listening: bool = True, has_listener: bool = True, error=None) -> None:
        self.registrations = []          # (combo, callback, suppress, on_release, handle)
        self.removed = []
        self.error = error
        self._counter = 0
        if has_listener:
            self._listener = types.SimpleNamespace(listening=listening)

    def add_hotkey(self, combo, callback, suppress=False, trigger_on_release=False):
        if self.error is not None:
            raise self.error
        self._counter += 1
        handle = f"handle-{self._counter}"
        self.registrations.append((combo, callback, suppress, trigger_on_release, handle))
        return handle

    def remove_hotkey(self, handle):
        self.removed.append(handle)

    # -- test driver ------------------------------------------------------ #
    def fire(self, combo, *, on_release=False):
        """Do what the hook thread does: call the callback synchronously."""
        fired = 0
        for reg_combo, callback, _s, reg_release, handle in list(self.registrations):
            if handle in self.removed:
                continue
            if reg_combo == combo and bool(reg_release) == bool(on_release):
                callback()
                fired += 1
        return fired

    def live_combos(self):
        return [r[0] for r in self.registrations if r[4] not in self.removed]


class FakeKey:
    def __init__(self, registry, path):
        self.registry = registry
        self.path = path


class FakeWinreg:
    """An in-memory stand-in for :mod:`winreg`.

    Only the handful of calls autostart.py makes; anything else would be a
    silent lie about what the module does.
    """

    HKEY_CURRENT_USER = "HKCU"
    HKEY_LOCAL_MACHINE = "HKLM"
    KEY_READ = 0x20019
    KEY_SET_VALUE = 0x0002
    REG_SZ = 1

    def __init__(self):
        self.store = {}
        self.closed = 0
        self.write_error = None

    def OpenKey(self, root, path, reserved=0, access=0):  # noqa: N802 - winreg's spelling
        if (root, path) not in self.store:
            raise FileNotFoundError(path)
        return FakeKey(self, (root, path))

    def CreateKeyEx(self, root, path, reserved=0, access=0):  # noqa: N802
        self.store.setdefault((root, path), {})
        return FakeKey(self, (root, path))

    def QueryValueEx(self, key, name):  # noqa: N802
        values = self.store[key.path]
        if name not in values:
            raise FileNotFoundError(name)
        return values[name], self.REG_SZ

    def SetValueEx(self, key, name, reserved, kind, value):  # noqa: N802
        if self.write_error is not None:
            raise self.write_error
        self.store[key.path][name] = value

    def DeleteValue(self, key, name):  # noqa: N802
        values = self.store[key.path]
        if name not in values:
            raise FileNotFoundError(name)
        del values[name]

    def CloseKey(self, key):  # noqa: N802
        self.closed += 1

    # -- test helper ------------------------------------------------------- #
    def run_values(self):
        return self.store.get((self.HKEY_CURRENT_USER, autostart.RUN_KEY_PATH), {})


class FakeMenuItem:
    def __init__(self, text, action=None, checked=None, **kwargs):
        self.text = text
        self.action = action
        self.checked = checked


class FakeMenu:
    SEPARATOR = "----"

    def __init__(self, *items):
        self.items = list(items)


class FakeTrayIconBackend:
    """A pystray.Icon stand-in.  ``run()`` blocks until ``stop()``, as the real one does."""

    def __init__(self, name, icon=None, title=None, menu=None):
        self.name = name
        self.icon = icon
        self.title = title
        self.menu = menu
        self.detached = False
        self.detach_error = None
        self.menu_updates = 0
        self.notifications = []
        self.run_entered = threading.Event()
        self.stopped = threading.Event()

    def run_detached(self, setup=None):
        if self.detach_error is not None:
            raise self.detach_error
        self.detached = True

    def run(self, setup=None):
        self.run_entered.set()
        self.stopped.wait(5.0)

    def stop(self):
        self.stopped.set()

    def update_menu(self):
        self.menu_updates += 1

    def notify(self, message, title=None):
        self.notifications.append((title, message))
        return None                      # exactly what pystray returns


def make_fake_pystray(*, detach_error=None):
    module = types.SimpleNamespace()
    created = []

    def _icon(name, icon=None, title=None, menu=None):
        obj = FakeTrayIconBackend(name, icon=icon, title=title, menu=menu)
        obj.detach_error = detach_error
        created.append(obj)
        return obj

    module.Icon = _icon
    module.MenuItem = FakeMenuItem
    module.Menu = FakeMenu
    module.created = created
    return module


def menu_item(icon_backend, label):
    for item in icon_backend.menu.items:
        if getattr(item, "text", None) == label:
            return item
    raise AssertionError(f"no menu item labelled {label!r}")


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def keyboard_manager(monkeypatch):
    """A HotkeyManager bound to a fake ``keyboard`` module."""
    fake = FakeKeyboard()
    monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (fake, ""))
    manager = hotkey.HotkeyManager()
    try:
        yield manager, fake
    finally:
        manager.stop()


@pytest.fixture
def fake_tray(monkeypatch):
    """A TrayIcon wired to a fake pystray and a fake icon renderer (no PIL needed)."""
    pystray = make_fake_pystray()
    monkeypatch.setattr(tray, "_import_pystray", lambda: pystray)
    drawn = []

    def factory(state):
        image = f"image:{state}"
        drawn.append(state)
        return image

    def build(**kwargs):
        kwargs.setdefault("icon_factory", factory)
        return tray.TrayIcon(**kwargs)

    yield types.SimpleNamespace(pystray=pystray, build=build, drawn=drawn)


# =========================================================================== #
#  hotkey.py — combo parsing
# =========================================================================== #
class TestComboParsing:
    def test_normalisation_is_forgiving_about_spacing_and_case(self):
        assert hotkey.normalise_combo("  Ctrl + Alt + J ") == "ctrl+alt+j"
        assert hotkey.normalise_combo("") == ""

    def test_default_combos_parse_to_the_expected_virtual_keys(self):
        assert hotkey.parse_combo(hotkey.DEFAULT_TOGGLE_COMBO) == (
            hotkey.MOD_CONTROL | hotkey.MOD_ALT, ord("J")
        )
        assert hotkey.parse_combo(hotkey.DEFAULT_PUSH_TO_TALK_COMBO) == (
            hotkey.MOD_CONTROL | hotkey.MOD_ALT, 0x20
        )

    @pytest.mark.parametrize("combo,expected", [
        ("shift+f5", (hotkey.MOD_SHIFT, 0x74)),
        ("win+7", (hotkey.MOD_WIN, ord("7"))),
        ("super+enter", (hotkey.MOD_WIN, 0x0D)),
        ("alt+f24", (hotkey.MOD_ALT, 0x87)),
        ("ctrl+shift+escape", (hotkey.MOD_CONTROL | hotkey.MOD_SHIFT, 0x1B)),
    ])
    def test_recognised_keys(self, combo, expected):
        assert hotkey.parse_combo(combo) == expected

    @pytest.mark.parametrize("combo", [
        "", "   ", "ctrl+alt",            # modifiers with no key
        "ctrl+j+k",                       # two real keys
        "ctrl+f25",                       # out of range
        "ctrl+play/pause",                # not in the table
    ])
    def test_unparseable_combos_return_none_rather_than_guessing(self, combo):
        assert hotkey.parse_combo(combo) is None

    def test_default_hotkeys_are_the_documented_pair(self):
        defaults = hotkey.default_hotkeys()
        assert defaults["toggle_listening"] == "ctrl+alt+j"
        assert defaults["push_to_talk"] == "ctrl+alt+space"


# =========================================================================== #
#  hotkey.py — dispatcher
# =========================================================================== #
class TestCallbackDispatcher:
    def test_work_runs_on_another_thread(self):
        dispatcher = hotkey.CallbackDispatcher(name="test-dispatch")
        seen = {}
        done = threading.Event()

        def record():
            seen["thread"] = threading.current_thread().name
            done.set()

        try:
            assert dispatcher.submit(record) is True
            assert done.wait(3.0) is True
            assert seen["thread"] != threading.current_thread().name
            assert seen["thread"] == "test-dispatch"
        finally:
            dispatcher.stop()

    def test_a_raising_callback_does_not_kill_the_worker(self):
        dispatcher = hotkey.CallbackDispatcher()
        survived = threading.Event()
        try:
            dispatcher.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            dispatcher.submit(survived.set)
            assert survived.wait(3.0) is True
        finally:
            dispatcher.stop()

    def test_a_full_queue_drops_work_instead_of_growing_without_limit(self):
        dispatcher = hotkey.CallbackDispatcher(maxsize=2)
        blocked = threading.Event()
        release = threading.Event()

        def blocker():
            blocked.set()
            release.wait(5.0)

        try:
            dispatcher.submit(blocker)
            assert blocked.wait(3.0) is True      # worker is now busy, queue empty
            assert dispatcher.submit(lambda: None) is True
            assert dispatcher.submit(lambda: None) is True
            assert dispatcher.submit(lambda: None) is False
            assert dispatcher.dropped == 1
        finally:
            release.set()
            dispatcher.stop()

    def test_submitting_a_non_callable_is_refused_not_queued(self):
        dispatcher = hotkey.CallbackDispatcher()
        try:
            assert dispatcher.submit("not a function") is False
            assert dispatcher.pending() == 0
        finally:
            dispatcher.stop()


# =========================================================================== #
#  hotkey.py — manager
# =========================================================================== #
class TestHotkeyManager:
    def test_registration_uses_the_keyboard_backend(self, keyboard_manager):
        manager, fake = keyboard_manager
        assert manager.register("ctrl+alt+j", lambda: None, suppress=True) is True
        assert manager.registered() == ["ctrl+alt+j"]
        assert manager.backend_name == "keyboard"
        combo, _cb, suppress, on_release, _handle = fake.registrations[0]
        assert (combo, suppress, on_release) == ("ctrl+alt+j", True, False)

    def test_a_slow_callback_never_blocks_the_hook_thread(self, keyboard_manager):
        manager, fake = keyboard_manager
        finished = threading.Event()

        def slow():
            time.sleep(0.4)
            finished.set()

        assert manager.register("ctrl+alt+j", slow) is True

        start = time.monotonic()
        fake.fire("ctrl+alt+j")            # this is the hook thread's call
        elapsed = time.monotonic() - start

        assert elapsed < 0.15, f"the hook thread was blocked for {elapsed:.2f}s"
        assert finished.wait(3.0) is True, "the callback was never run by the worker"

    def test_a_dead_global_hook_is_reported_not_swallowed(self, monkeypatch):
        """The `keyboard` package accepts the call, but nothing is listening."""
        fake = FakeKeyboard(listening=False)
        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (fake, ""))
        manager = hotkey.HotkeyManager()
        try:
            assert manager.register("ctrl+alt+j", lambda: None) is False
            assert manager.registered() == []
            assert "administrator" in (manager.last_error or "")
            # The half-registered hotkey was cleaned up rather than left behind.
            assert fake.removed == ["handle-1"]
        finally:
            manager.stop()

    def test_a_backend_exception_is_reported_through_the_return_value(self, monkeypatch):
        fake = FakeKeyboard(error=OSError("SetWindowsHookEx failed"))
        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (fake, ""))
        manager = hotkey.HotkeyManager()
        try:
            assert manager.register("ctrl+alt+j", lambda: None) is False
            assert "SetWindowsHookEx failed" in (manager.last_error or "")
        finally:
            manager.stop()

    def test_no_backend_means_unavailable_rather_than_an_exception(self, monkeypatch):
        monkeypatch.setattr(hotkey, "_import_keyboard",
                            lambda: (None, "keyboard: No module named 'keyboard'"))
        monkeypatch.setattr(hotkey._Win32Backend, "is_available", lambda self: False)
        manager = hotkey.HotkeyManager()
        try:
            assert manager.is_available() is False
            assert manager.register("ctrl+alt+j", lambda: None) is False
            assert manager.backend_name == "none"
            assert "keyboard" in (manager.last_error or "")
        finally:
            manager.stop()

    def test_empty_combo_and_non_callable_are_refused(self, keyboard_manager):
        manager, fake = keyboard_manager
        assert manager.register("", lambda: None) is False
        assert manager.register("ctrl+alt+j", "not callable") is False
        assert fake.registrations == []

    def test_re_registering_a_combo_replaces_the_old_binding(self, keyboard_manager):
        manager, fake = keyboard_manager
        first, second = [], []
        rang = threading.Event()
        manager.register("ctrl+alt+j", lambda: first.append(1))
        manager.register("ctrl+alt+j", lambda: (second.append(1), rang.set()))
        assert fake.removed == ["handle-1"]
        assert manager.registered() == ["ctrl+alt+j"]

        fake.fire("ctrl+alt+j")
        assert rang.wait(3.0) is True
        assert first == [], "the replaced callback must no longer fire"
        assert second == [1]

    def test_unregister_and_unregister_all(self, keyboard_manager):
        manager, fake = keyboard_manager
        manager.register("ctrl+alt+j", lambda: None)
        manager.register("ctrl+alt+k", lambda: None)
        assert manager.unregister("CTRL+ALT+J") is True     # normalised on the way in
        assert manager.registered() == ["ctrl+alt+k"]
        assert manager.unregister("ctrl+alt+j") is False
        manager.unregister_all()
        assert manager.registered() == []
        assert sorted(fake.removed) == ["handle-1", "handle-2"]

    def test_context_manager_releases_everything(self, monkeypatch):
        fake = FakeKeyboard()
        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (fake, ""))
        with hotkey.HotkeyManager() as manager:
            assert manager.register("ctrl+alt+j", lambda: None) is True
            assert manager.registered() == ["ctrl+alt+j"]
        assert manager.registered() == []
        assert fake.removed == ["handle-1"]

    def test_status_describes_the_installation(self, keyboard_manager):
        manager, _fake = keyboard_manager
        manager.register("ctrl+alt+j", lambda: None)
        status = manager.status()
        assert status["available"] is True
        assert status["backend"] == "keyboard"
        assert status["registered"] == ["ctrl+alt+j"]
        assert status["dropped_callbacks"] == 0


class TestPushToTalk:
    def test_press_and_release_are_both_bound(self, keyboard_manager):
        manager, fake = keyboard_manager
        assert manager.register_push_to_talk(
            "ctrl+alt+space", lambda: None, lambda: None
        ) is True
        triggers = sorted(r[3] for r in fake.registrations)
        assert triggers == [False, True], "push-to-talk needs a press AND a release hook"

    def test_holding_the_key_starts_and_stops_capture_once(self, keyboard_manager):
        manager, fake = keyboard_manager
        events = []
        pressed = threading.Event()
        released = threading.Event()

        def on_press():
            events.append("press")
            pressed.set()

        def on_release():
            events.append("release")
            released.set()

        assert manager.register_push_to_talk("ctrl+alt+space", on_press, on_release) is True

        fake.fire("ctrl+alt+space")                      # key down
        fake.fire("ctrl+alt+space")                      # auto-repeat while held
        fake.fire("ctrl+alt+space")
        assert pressed.wait(3.0) is True
        fake.fire("ctrl+alt+space", on_release=True)     # key up
        assert released.wait(3.0) is True

        assert events == ["press", "release"], f"auto-repeat leaked through: {events}"

    def test_a_second_press_after_release_is_honoured(self, keyboard_manager):
        manager, fake = keyboard_manager
        presses = []
        second = threading.Event()

        def on_press():
            presses.append(1)
            if len(presses) == 2:
                second.set()

        manager.register_push_to_talk("ctrl+alt+space", on_press, lambda: None)
        fake.fire("ctrl+alt+space")
        fake.fire("ctrl+alt+space", on_release=True)
        fake.fire("ctrl+alt+space")
        assert second.wait(3.0) is True
        assert len(presses) == 2

    def test_binding_a_voice_loop_calls_begin_and_end_utterance(self, keyboard_manager):
        manager, fake = keyboard_manager
        began = threading.Event()
        ended = threading.Event()

        class Loop:
            def begin_utterance(self):
                began.set()

            def end_utterance(self):
                ended.set()

        assert manager.bind_push_to_talk(Loop()) is True
        fake.fire("ctrl+alt+space")
        assert began.wait(3.0) is True
        fake.fire("ctrl+alt+space", on_release=True)
        assert ended.wait(3.0) is True

    def test_a_voice_loop_without_those_methods_is_tolerated(self, keyboard_manager):
        """VoiceLoop may not have grown begin_utterance() yet; nothing may explode."""
        manager, fake = keyboard_manager
        assert manager.bind_push_to_talk(object()) is True
        survived = threading.Event()
        fake.fire("ctrl+alt+space")
        manager._dispatcher.submit(survived.set)
        assert survived.wait(3.0) is True     # the worker is still alive

    def test_push_to_talk_is_refused_when_the_backend_cannot_see_releases(self, monkeypatch):
        class PressOnlyBackend:
            name = "press-only"
            supports_release = False

            def add(self, *a, **k):
                return True, ""

            def start(self):
                pass

            def stop(self):
                pass

            def clear(self):
                pass

            def remove(self, combo):
                return False

        manager = hotkey.HotkeyManager()
        monkeypatch.setattr(manager, "_resolve", lambda: PressOnlyBackend())
        try:
            assert manager.register_push_to_talk("ctrl+alt+space", lambda: None, lambda: None) is False
            assert "release" in (manager.last_error or "")
        finally:
            manager.stop()


class TestWin32Backend:
    def test_it_reports_itself_unavailable_off_windows(self, monkeypatch):
        monkeypatch.setattr(hotkey, "IS_WINDOWS", False)
        backend = hotkey._Win32Backend()
        assert backend.is_available() is False
        assert "Windows" in (backend.last_error or "")

    def test_an_unparseable_combo_is_refused_without_starting_a_thread(self, monkeypatch):
        monkeypatch.setattr(hotkey, "IS_WINDOWS", False)
        backend = hotkey._Win32Backend()
        ok, error = backend.add("ctrl+play/pause", lambda: None)
        assert ok is False
        assert "RegisterHotKey" in error
        assert backend._thread is None

    def test_selecting_it_explicitly_off_windows_leaves_the_manager_unavailable(self, monkeypatch):
        monkeypatch.setattr(hotkey, "IS_WINDOWS", False)
        manager = hotkey.HotkeyManager(backend="win32")
        try:
            assert manager.is_available() is False
            assert manager.register("ctrl+alt+j", lambda: None) is False
        finally:
            manager.stop()


# =========================================================================== #
#  tray.py
# =========================================================================== #
class TestIconRendering:
    def test_states_are_visually_distinct(self):
        pytest.importorskip("PIL", reason="Pillow is needed to draw the icon")
        idle = tray.render_icon("idle", size=48)
        listening = tray.render_icon("listening", size=48)
        assert idle is not None and listening is not None
        assert idle.size == (48, 48)
        assert idle.mode == "RGBA"
        assert idle.tobytes() != listening.tobytes()

    def test_an_unknown_state_falls_back_to_the_idle_colour(self):
        pytest.importorskip("PIL", reason="Pillow is needed to draw the icon")
        idle = tray.render_icon("idle", size=32)
        unknown = tray.render_icon("no-such-state", size=32)
        assert unknown is not None
        assert unknown.tobytes() == idle.tobytes()

    def test_without_pillow_it_returns_none_rather_than_raising(self, monkeypatch):
        block_package(monkeypatch, "PIL")
        assert tray.render_icon("idle") is None


class TestTrayAvailability:
    def test_unavailable_without_pystray(self, monkeypatch):
        monkeypatch.setattr(tray, "_import_pystray", lambda: None)
        icon = tray.TrayIcon()
        assert icon.is_available() is False
        assert "pystray" in (icon.last_error or "")
        assert icon.start() is False

    def test_unavailable_when_no_image_can_be_drawn(self, monkeypatch):
        monkeypatch.setattr(tray, "_import_pystray", lambda: make_fake_pystray())
        icon = tray.TrayIcon(icon_factory=lambda state: None)
        assert icon.is_available() is False
        assert icon.start() is False

    def test_a_raising_icon_factory_is_contained(self, monkeypatch):
        monkeypatch.setattr(tray, "_import_pystray", lambda: make_fake_pystray())

        def explode(state):
            raise RuntimeError("no display")

        icon = tray.TrayIcon(icon_factory=explode)
        assert icon.is_available() is False


class TestTrayLifecycle:
    def test_start_is_non_blocking_and_prefers_run_detached(self, fake_tray):
        icon = fake_tray.build()
        assert icon.start() is True
        backend = fake_tray.pystray.created[0]
        assert backend.detached is True
        assert backend.run_entered.is_set() is False
        assert icon.running is True
        icon.stop()
        assert backend.stopped.is_set() is True
        assert icon.running is False

    def test_it_falls_back_to_a_daemon_thread_when_detaching_is_unsupported(self, monkeypatch):
        pystray = make_fake_pystray(detach_error=NotImplementedError("no detach"))
        monkeypatch.setattr(tray, "_import_pystray", lambda: pystray)
        icon = tray.TrayIcon(icon_factory=lambda state: f"img:{state}")
        assert icon.start() is True
        backend = pystray.created[0]
        assert backend.run_entered.wait(3.0) is True
        assert backend.detached is False
        icon.stop()
        assert backend.stopped.is_set() is True

    def test_starting_twice_does_not_create_a_second_icon(self, fake_tray):
        icon = fake_tray.build()
        assert icon.start() is True
        assert icon.start() is True
        assert len(fake_tray.pystray.created) == 1
        icon.stop()

    def test_stop_is_idempotent(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        icon.stop()
        icon.stop()
        assert icon.running is False


class TestTrayState:
    def test_set_state_re_renders_the_icon_and_the_tooltip(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        backend = fake_tray.pystray.created[0]
        assert backend.icon == "image:idle"

        assert icon.set_state("listening") is True
        assert backend.icon == "image:listening"
        assert "listening" in backend.title
        assert icon.state == "listening"

        assert icon.set_state("thinking") is True
        assert backend.icon == "image:thinking"
        icon.stop()

    def test_every_documented_state_renders_without_raising(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        for state in ("idle", "listening", "thinking", "speaking", "muted", "error"):
            assert icon.set_state(state) is True
        assert fake_tray.drawn[-6:] == [
            "idle", "listening", "thinking", "speaking", "muted", "error"
        ]
        icon.stop()

    def test_an_unknown_state_is_recorded_rather_than_rejected(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        assert icon.set_state("Reticulating Splines") is True
        assert icon.state == "reticulating splines"
        icon.stop()

    def test_set_state_before_start_records_but_reports_nothing_was_shown(self, fake_tray):
        icon = fake_tray.build()
        assert icon.set_state("speaking") is False
        assert icon.state == "speaking"
        icon.start()
        assert fake_tray.pystray.created[0].icon == "image:speaking"
        icon.stop()

    def test_notify_uses_the_tray_balloon(self, fake_tray):
        icon = fake_tray.build()
        assert icon.notify("t", "m") is False, "no balloon is possible before start()"
        icon.start()
        assert icon.notify("Task done", "The report is ready") is True
        assert fake_tray.pystray.created[0].notifications == [
            ("Task done", "The report is ready")
        ]
        icon.stop()


class TestTrayMenu:
    def test_the_menu_offers_the_five_documented_actions(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        backend = fake_tray.pystray.created[0]
        labels = [getattr(i, "text", None) for i in backend.menu.items]
        for expected in ("Listening", "Mute voice", "Show log folder",
                         "Restart listening", "Quit"):
            assert expected in labels
        icon.stop()

    def test_listening_and_mute_are_checkable_and_reflect_state(self, fake_tray):
        icon = fake_tray.build(listening=True, muted=False)
        icon.start()
        backend = fake_tray.pystray.created[0]
        listening_item = menu_item(backend, tray.TrayIcon.LABEL_LISTENING)
        mute_item = menu_item(backend, tray.TrayIcon.LABEL_MUTE)
        assert listening_item.checked(listening_item) is True
        assert mute_item.checked(mute_item) is False

        icon.set_muted(True)
        assert mute_item.checked(mute_item) is True
        icon.stop()

    def test_toggling_listening_flips_the_tick_and_calls_back_off_the_ui_thread(self, fake_tray):
        seen = []
        done = threading.Event()

        def on_toggle(value):
            seen.append((value, threading.current_thread().name))
            done.set()

        icon = fake_tray.build(on_toggle_listen=on_toggle, listening=True)
        icon.start()
        backend = fake_tray.pystray.created[0]
        item = menu_item(backend, tray.TrayIcon.LABEL_LISTENING)

        item.action(backend, item)                     # pystray's UI thread does this
        assert done.wait(3.0) is True
        assert seen[0][0] is False
        assert seen[0][1] != threading.current_thread().name
        assert icon.listening is False
        assert backend.menu_updates >= 1
        icon.stop()

    def test_a_slow_menu_callback_does_not_block_the_ui_thread(self, fake_tray):
        finished = threading.Event()

        def slow(_value):
            time.sleep(0.4)
            finished.set()

        icon = fake_tray.build(on_toggle_mute=slow)
        icon.start()
        backend = fake_tray.pystray.created[0]
        item = menu_item(backend, tray.TrayIcon.LABEL_MUTE)

        start = time.monotonic()
        item.action(backend, item)
        elapsed = time.monotonic() - start

        assert elapsed < 0.15, f"the tray UI thread was blocked for {elapsed:.2f}s"
        assert finished.wait(3.0) is True
        icon.stop()

    def test_restart_listening_invokes_its_callback(self, fake_tray):
        called = threading.Event()
        icon = fake_tray.build(on_restart_listen=called.set)
        icon.start()
        backend = fake_tray.pystray.created[0]
        item = menu_item(backend, tray.TrayIcon.LABEL_RESTART)
        item.action(backend, item)
        assert called.wait(3.0) is True
        icon.stop()

    def test_show_log_folder_opens_the_configured_directory(self, fake_tray, tmp_path,
                                                            monkeypatch):
        opened = []
        finished = threading.Event()

        def fake_open(target):
            opened.append(target)
            finished.set()
            return platform_utils.CommandResult(0, "", "")

        monkeypatch.setattr(platform_utils, "open_path", fake_open)
        log_dir = tmp_path / "logs"
        icon = fake_tray.build(log_dir=log_dir)
        icon.start()
        backend = fake_tray.pystray.created[0]
        item = menu_item(backend, tray.TrayIcon.LABEL_LOGS)
        item.action(backend, item)
        assert finished.wait(3.0) is True
        assert opened == [str(log_dir)]
        assert log_dir.exists(), "the folder should be created before it is opened"
        icon.stop()

    def test_quit_calls_back_and_takes_the_icon_down(self, fake_tray):
        quit_called = threading.Event()
        icon = fake_tray.build(on_quit=quit_called.set)
        icon.start()
        backend = fake_tray.pystray.created[0]
        item = menu_item(backend, tray.TrayIcon.LABEL_QUIT)
        item.action(backend, item)
        assert quit_called.wait(3.0) is True
        assert backend.stopped.is_set() is True
        assert icon.running is False
        icon.stop()

    def test_status_reports_what_the_tray_is_doing(self, fake_tray):
        icon = fake_tray.build()
        icon.start()
        icon.set_state("speaking")
        status = icon.status()
        assert status["running"] is True
        assert status["state"] == "speaking"
        assert status["available"] is True
        icon.stop()


# =========================================================================== #
#  autostart.py
# =========================================================================== #
@pytest.fixture
def windows_registry(monkeypatch):
    """Pretend to be Windows, with an in-memory registry."""
    fake = FakeWinreg()
    monkeypatch.setattr(autostart, "IS_WINDOWS", True)
    monkeypatch.setattr(autostart, "_winreg", lambda: fake)
    return fake


@pytest.fixture
def linux_autostart(monkeypatch, tmp_path):
    """Pretend to be Linux, with XDG_CONFIG_HOME inside tmp_path."""
    monkeypatch.setattr(autostart, "IS_WINDOWS", False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "autostart" / autostart.DESKTOP_FILE_NAME


class TestQuoting:
    def test_a_path_with_spaces_is_quoted(self):
        quoted = autostart.quote_argument(r"C:\Program Files\Python313\python.exe")
        assert quoted == r'"C:\Program Files\Python313\python.exe"'

    def test_an_already_quoted_path_is_left_alone(self):
        assert autostart.quote_argument('"C:\\py\\python.exe"') == '"C:\\py\\python.exe"'

    def test_the_default_command_runs_the_voice_module(self, monkeypatch):
        monkeypatch.setattr(sys, "executable", r"C:\Program Files\Python313\python.exe")
        command = autostart.default_command()
        assert command == r'"C:\Program Files\Python313\python.exe" -m jarvis voice'
        assert not command.startswith("C:\\Program Files"), "an unquoted path fails silently"


class TestWindowsAutostart:
    def test_enable_is_enabled_disable_round_trip(self, windows_registry):
        assert autostart.is_enabled() is False
        assert autostart.enable() is True
        assert autostart.is_enabled() is True
        assert autostart.current_command() == autostart.default_command()
        assert autostart.disable() is True
        assert autostart.is_enabled() is False

    def test_the_value_lands_under_the_per_user_run_key_only(self, windows_registry):
        autostart.enable()
        assert (FakeWinreg.HKEY_CURRENT_USER, autostart.RUN_KEY_PATH) in windows_registry.store
        assert autostart.APP_NAME in windows_registry.run_values()
        machine_keys = [k for k in windows_registry.store
                        if k[0] == FakeWinreg.HKEY_LOCAL_MACHINE]
        assert machine_keys == [], "autostart must never write to HKEY_LOCAL_MACHINE"

    def test_the_stored_command_quotes_an_interpreter_path_containing_spaces(
        self, windows_registry, monkeypatch
    ):
        monkeypatch.setattr(sys, "executable", r"C:\Program Files\Python313\python.exe")
        assert autostart.enable() is True
        stored = windows_registry.run_values()[autostart.APP_NAME]
        assert stored == r'"C:\Program Files\Python313\python.exe" -m jarvis voice'

    def test_a_custom_command_is_stored_verbatim(self, windows_registry):
        assert autostart.enable('"C:\\x y\\pythonw.exe" -m jarvis voice --quiet') is True
        assert windows_registry.run_values()[autostart.APP_NAME] == (
            '"C:\\x y\\pythonw.exe" -m jarvis voice --quiet'
        )

    def test_enable_is_idempotent(self, windows_registry):
        autostart.enable()
        autostart.enable()
        assert len(windows_registry.run_values()) == 1

    def test_disabling_when_nothing_is_registered_is_still_success(self, windows_registry):
        assert autostart.disable() is True
        autostart.enable()
        autostart.disable()
        assert autostart.disable() is True

    def test_an_empty_or_multiline_command_is_refused(self, windows_registry):
        assert autostart.enable("   ") is False
        assert autostart.enable("python -m jarvis\nformat c:") is False
        assert windows_registry.run_values() == {}

    def test_a_registry_write_failure_is_reported_not_raised(self, windows_registry):
        windows_registry.write_error = PermissionError("access denied")
        assert autostart.enable() is False
        assert autostart.is_enabled() is False

    def test_every_opened_key_is_closed(self, windows_registry):
        autostart.enable()
        autostart.is_enabled()
        autostart.disable()
        assert windows_registry.closed == 3

    def test_without_winreg_it_degrades_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr(autostart, "IS_WINDOWS", True)
        monkeypatch.setattr(autostart, "_winreg", lambda: None)
        assert autostart.is_supported() is False
        assert autostart.enable() is False
        assert autostart.disable() is False
        assert autostart.is_enabled() is False
        assert autostart.status()["supported"] is False

    def test_status_points_at_the_current_user_hive(self, windows_registry):
        status = autostart.status()
        assert status["platform"] == "windows"
        assert status["location"].startswith("HKEY_CURRENT_USER\\")
        assert "CurrentVersion\\Run" in status["location"]
        assert status["enabled"] is False


class TestLinuxAutostart:
    def test_it_writes_a_plausible_desktop_entry(self, linux_autostart):
        assert autostart.is_enabled() is False
        assert autostart.enable("/usr/bin/python3 -m jarvis voice") is True
        assert linux_autostart.exists()

        text = linux_autostart.read_text(encoding="utf-8")
        assert text.startswith("[Desktop Entry]")
        assert "Type=Application" in text
        assert "Name=JARVIS" in text
        assert "Exec=/usr/bin/python3 -m jarvis voice" in text
        assert "X-GNOME-Autostart-enabled=true" in text

    def test_round_trip(self, linux_autostart):
        autostart.enable("/usr/bin/python3 -m jarvis voice")
        assert autostart.is_enabled() is True
        assert autostart.current_command() == "/usr/bin/python3 -m jarvis voice"
        assert autostart.disable() is True
        assert linux_autostart.exists() is False
        assert autostart.is_enabled() is False

    def test_a_path_with_spaces_keeps_its_quotes_in_the_exec_line(self, linux_autostart,
                                                                  monkeypatch):
        monkeypatch.setattr(sys, "executable", "/opt/my python/bin/python3")
        assert autostart.enable() is True
        text = linux_autostart.read_text(encoding="utf-8")
        assert 'Exec="/opt/my python/bin/python3" -m jarvis voice' in text

    def test_disabling_when_nothing_is_installed_is_success(self, linux_autostart):
        assert autostart.disable() is True

    def test_status_points_at_the_desktop_file(self, linux_autostart):
        status = autostart.status()
        assert status["platform"] == "posix"
        assert status["supported"] is True
        assert status["location"] == str(linux_autostart)
        assert status["enabled"] is False

    def test_an_unreadable_entry_reports_disabled_rather_than_raising(self, linux_autostart):
        linux_autostart.parent.mkdir(parents=True, exist_ok=True)
        linux_autostart.write_bytes(b"\xff\xfe\x00garbage")
        assert autostart.is_enabled() is False

    def test_macos_says_no_rather_than_writing_a_file_that_never_runs(
        self, linux_autostart, monkeypatch
    ):
        monkeypatch.setattr(autostart, "IS_MAC", True)
        assert autostart.is_supported() is False
        assert autostart.enable() is False
        assert linux_autostart.exists() is False


# =========================================================================== #
#  notify.py
# =========================================================================== #
@pytest.fixture
def silent_notify(monkeypatch):
    """Block every channel that would touch the machine."""
    monkeypatch.setattr(notify_mod, "_windows_toast", lambda *a, **k: False)
    monkeypatch.setattr(notify_mod, "_command_line_notifier", lambda *a, **k: False)
    notify_mod.set_tray(None)
    yield
    notify_mod.set_tray(None)


class TestEscaping:
    def test_xml_escaping_protects_the_toast_payload(self):
        assert notify_mod.xml_escape("Tom & <Jerry>") == "Tom &amp; &lt;Jerry&gt;"

    def test_powershell_quoting_doubles_single_quotes(self):
        assert notify_mod.powershell_quote("it's fine") == "'it''s fine'"

    def test_the_toast_script_escapes_hostile_text(self):
        script = notify_mod._toast_script("A & B", "don't <break> it", 5)
        assert "A &amp; B" in script
        assert "don&apos;t" in script
        assert "<break>" not in script
        assert "&lt;break&gt;" in script

    def test_a_long_notification_asks_for_a_long_toast(self):
        assert 'duration="long"' in notify_mod._toast_script("t", "m", 20)
        assert 'duration="short"' in notify_mod._toast_script("t", "m", 3)

    def test_applescript_escaping_protects_quotes_and_backslashes(self):
        assert notify_mod._applescript_string('say "hi" \\ now') == 'say \\"hi\\" \\\\ now'

    def test_the_xml_payload_never_contains_a_bare_apostrophe(self):
        """An apostrophe would break out of the PowerShell single-quoted string."""
        script = notify_mod._toast_script("Jarvis' report", "it's ready — o'clock", 5)
        payload = script.split("$xml.LoadXml(", 1)[1].split(");", 1)[0]
        assert payload.startswith("'") and payload.endswith("'")
        assert "'" not in payload[1:-1]


class TestNotifyChain:
    def test_it_falls_all_the_way_through_to_the_log(self, silent_notify):
        assert notify_mod.notify("Title", "Message") == "log"

    def test_a_live_tray_takes_the_notification(self, silent_notify):
        received = []

        class Tray:
            def notify(self, title, message):
                received.append((title, message))
                return True

        notify_mod.set_tray(Tray())
        assert notify_mod.notify("Task done", "Report ready") == "tray"
        assert received == [("Task done", "Report ready")]

    def test_an_explicit_tray_argument_wins_over_the_registered_one(self, silent_notify):
        calls = []

        class Tray:
            def __init__(self, tag):
                self.tag = tag

            def notify(self, title, message):
                calls.append(self.tag)
                return True

        notify_mod.set_tray(Tray("registered"))
        assert notify_mod.notify("t", "m", tray=Tray("explicit")) == "tray"
        assert calls == ["explicit"]

    def test_a_tray_that_declines_falls_through(self, silent_notify):
        class Tray:
            def notify(self, title, message):
                return False

        notify_mod.set_tray(Tray())
        assert notify_mod.notify("t", "m") == "log"

    def test_a_raising_channel_does_not_escape(self, silent_notify, monkeypatch):
        def explode(*_a, **_k):
            raise RuntimeError("WinRT is unavailable")

        monkeypatch.setattr(notify_mod, "_windows_toast", explode)

        class Tray:
            def notify(self, title, message):
                raise OSError("the shell is gone")

        notify_mod.set_tray(Tray())
        assert notify_mod.notify("t", "m") == "log"

    def test_the_toast_is_preferred_when_it_works(self, monkeypatch):
        monkeypatch.setattr(notify_mod, "_windows_toast", lambda *a, **k: True)
        assert notify_mod.notify("t", "m") == "toast"

    def test_empty_input_is_tolerated(self, silent_notify):
        assert notify_mod.notify(None, None) == "log"
        assert notify_mod.notify("", "") == "log"

    @pytest.mark.parametrize("given,expected", [
        (0, 1), (-4, 1), (5, 5), (999, 60), ("not a number", 5), (None, 5),
    ])
    def test_duration_is_clamped_to_something_a_human_can_read(self, given, expected):
        assert notify_mod._clamp_duration(given) == expected

    def test_notify_send_is_used_on_a_linux_desktop(self, monkeypatch):
        monkeypatch.setattr(platform_utils, "IS_WINDOWS", False)
        monkeypatch.setattr(platform_utils, "IS_MAC", False)
        monkeypatch.setattr(platform_utils, "which",
                            lambda name: "/usr/bin/notify-send" if name == "notify-send" else None)
        calls = []

        def fake_run(command, **kwargs):
            calls.append((list(command), kwargs))
            return platform_utils.CommandResult(0, "", "")

        monkeypatch.setattr(platform_utils, "run_command", fake_run)
        notify_mod.set_tray(None)

        assert notify_mod.notify("Task done", "Report ready", duration=4) == "command"
        argv, kwargs = calls[0]
        assert argv == ["/usr/bin/notify-send", "-t", "4000", "Task done", "Report ready"]
        assert kwargs["timeout"] > 0, "every subprocess must carry a timeout"

    def test_no_subprocess_is_launched_when_no_notifier_exists(self, monkeypatch):
        monkeypatch.setattr(platform_utils, "which", lambda name: None)
        ran = []
        monkeypatch.setattr(platform_utils, "run_command",
                            lambda *a, **k: ran.append(a) or platform_utils.CommandResult(1, "", ""))
        notify_mod.set_tray(None)
        assert notify_mod.notify("t", "m") == "log"
        assert ran == []

    def test_available_channels_describes_the_machine(self, monkeypatch):
        monkeypatch.setattr(platform_utils, "which", lambda name: None)
        notify_mod.set_tray(None)
        channels = notify_mod.available_channels()
        assert channels["log"] is True
        assert channels["toast"] is False
        assert channels["command"] is False
        assert channels["tray"] is False

        notify_mod.set_tray(object())
        try:
            assert notify_mod.available_channels()["tray"] is True
        finally:
            notify_mod.set_tray(None)

    def test_set_tray_round_trips_and_clears(self):
        sentinel = object()
        notify_mod.set_tray(sentinel)
        assert notify_mod.get_tray() is sentinel
        notify_mod.set_tray(None)
        assert notify_mod.get_tray() is None

    def test_the_toast_channel_is_skipped_off_windows(self, monkeypatch):
        monkeypatch.setattr(platform_utils, "IS_WINDOWS", False)
        called = []
        monkeypatch.setattr(platform_utils, "which", lambda name: called.append(name) or None)
        assert notify_mod._windows_toast("t", "m", 5) is False
        assert called == [], "no shell should even be looked for off Windows"


# =========================================================================== #
#  Package facade
# =========================================================================== #
class TestPackageFacade:
    def test_the_public_names_are_exported(self):
        import jarvis.win as win

        assert win.HotkeyManager is hotkey.HotkeyManager
        assert win.TrayIcon is tray.TrayIcon
        assert win.notify is notify_mod.notify        # the function, not the module
        assert win.autostart is autostart
        assert win.set_tray is notify_mod.set_tray
        assert win.render_icon is tray.render_icon
        assert win.DEFAULT_PUSH_TO_TALK_COMBO == "ctrl+alt+space"

    def test_the_availability_report_is_honest_when_nothing_is_installed(self, monkeypatch):
        import jarvis.win as win

        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (None, "keyboard: absent"))
        monkeypatch.setattr(hotkey._Win32Backend, "is_available", lambda self: False)
        monkeypatch.setattr(tray, "_import_pystray", lambda: None)
        monkeypatch.setattr(autostart, "IS_WINDOWS", True)
        monkeypatch.setattr(autostart, "_winreg", lambda: FakeWinreg())
        monkeypatch.setattr(platform_utils, "which", lambda name: None)

        report = win.is_windows_integration_available()

        assert report["os"] == platform_utils.os_name()
        assert report["hotkeys"] is False
        assert report["hotkey_backend"] == "none"
        assert report["tray"] is False
        assert report["autostart"] is True          # winreg fake is present
        assert report["autostart_enabled"] is False
        assert report["notifications"]["log"] is True
        assert report["suggested_hotkeys"]["push_to_talk"] == "ctrl+alt+space"

    def test_a_probe_that_raises_becomes_a_string_not_an_exception(self, monkeypatch):
        import jarvis.win as win

        def explode(self):
            raise RuntimeError("the desktop session went away")

        monkeypatch.setattr(tray.TrayIcon, "is_available", explode)
        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (None, "absent"))
        monkeypatch.setattr(hotkey._Win32Backend, "is_available", lambda self: False)

        report = win.is_windows_integration_available()
        assert isinstance(report["tray"], str)
        assert "desktop session went away" in report["tray"]

    def test_everything_still_works_with_no_optional_package_installed(self, monkeypatch):
        """The state a fresh Linux box is in: nothing installed, nothing crashing."""
        monkeypatch.setattr(hotkey, "_import_keyboard", lambda: (None, "keyboard: absent"))
        monkeypatch.setattr(hotkey, "IS_WINDOWS", False)
        monkeypatch.setattr(tray, "_import_pystray", lambda: None)
        block_package(monkeypatch, "PIL")

        manager = hotkey.HotkeyManager()
        try:
            assert manager.is_available() is False
            assert manager.register(hotkey.DEFAULT_TOGGLE_COMBO, lambda: None) is False
            assert manager.bind_push_to_talk(object()) is False
        finally:
            manager.stop()

        icon = tray.TrayIcon()
        assert icon.is_available() is False
        assert icon.start() is False
        assert icon.set_state("listening") is False
        assert icon.notify("t", "m") is False
        icon.stop()

        # ...and the pure, dependency-free parts still answer.
        assert hotkey.parse_combo("ctrl+alt+j") is not None
        assert autostart.default_command().endswith("-m jarvis voice")
        assert tray.STATE_COLOURS["listening"] != tray.STATE_COLOURS["idle"]
