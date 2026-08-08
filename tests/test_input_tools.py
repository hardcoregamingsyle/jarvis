"""Keyboard and mouse tool tests.

Nothing in here may touch a real device: the backends are replaced with
recorders, and the real ones are blocked at the import system so a forgotten
patch fails loudly instead of moving the developer's mouse.
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import pytest

from jarvis.core.platform_utils import IS_MAC, IS_WINDOWS
from jarvis.tools import input_tools
from jarvis.tools.input_tools import build_tools
from jarvis.tools.registry import ToolContext, ToolRegistry


# --------------------------------------------------------------------------- #
#  Doubles
# --------------------------------------------------------------------------- #
class FakePyAutoGUI:
    """Records every call the tools make, and reports a known screen size."""

    KEYBOARD_KEYS = [
        "enter", "tab", "escape", "space", "backspace", "delete", "f5",
        "up", "down", "left", "right", "ctrl", "alt", "shift", "win",
        "command", "option", "a", "d", "s", "t", "v",
    ]

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        # The real module ships with the corner failsafe on; the code under
        # test has to turn it off.
        self.FAILSAFE = True
        self.PAUSE = 0.1
        self.calls: List[Tuple[Any, ...]] = []
        self._size = (width, height)
        self._position = (7, 9)
        self.raises_on: Optional[str] = None

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self.raises_on == name:
            raise RuntimeError(f"{name} exploded")
        self.calls.append((name, args, kwargs))

    def named(self, name: str) -> List[Tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == name]

    # -- keyboard ---------------------------------------------------------- #
    def write(self, text: str, interval: float = 0.0) -> None:
        self._record("write", text, interval=interval)

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._record("press", key, presses=presses, interval=interval)

    def hotkey(self, *keys: str) -> None:
        self._record("hotkey", *keys)

    # -- mouse ------------------------------------------------------------- #
    def moveTo(self, x: int, y: int, duration: float = 0.0) -> None:  # noqa: N802
        self._record("moveTo", x, y, duration=duration)

    def click(self, x: Any = None, y: Any = None, clicks: int = 1,
              button: str = "left", interval: float = 0.0) -> None:
        self._record("click", x, y, clicks=clicks, button=button)

    def scroll(self, amount: int, x: Any = None, y: Any = None) -> None:
        self._record("scroll", amount, x, y)

    def dragTo(self, x: int, y: int, duration: float = 0.0,  # noqa: N802
               button: str = "left") -> None:
        self._record("dragTo", x, y, duration=duration, button=button)

    def position(self) -> Tuple[int, int]:
        self._record("position")
        return self._position

    def size(self) -> Tuple[int, int]:
        return self._size


class FakeWin32Api:
    """Enough of win32api for the Windows fallback paths."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.events: List[Tuple[int, int, int, int]] = []
        self._size = (width, height)

    def keybd_event(self, vk: int, scan: int, flags: int, extra: int) -> None:
        self.events.append((vk, scan, flags, extra))

    def GetSystemMetrics(self, index: int) -> int:  # noqa: N802
        return self._size[0] if index == 0 else self._size[1]


class FakeWin32Con:
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004


class OpenSecurity:
    """Permits everything, so the tests exercise the tools and not the gate."""

    def check_tool(self, spec: Any, cleaned: Any) -> Any:
        class Decision:
            allowed = True
            requires_confirmation = False

        return Decision()

    def allows(self, decision: Any) -> bool:
        return True


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch):
    """Block every real input backend for the duration of a test.

    Binding a name to ``None`` in ``sys.modules`` makes ``import name`` raise
    ImportError, which is exactly the situation the failure paths must handle.
    """
    for name in ("pyautogui", "win32api", "win32con", "win32clipboard"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(input_tools, "which", lambda name: None)
    input_tools.reset_backend_cache()
    yield
    input_tools.reset_backend_cache()


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> FakePyAutoGUI:
    fake = FakePyAutoGUI()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    input_tools.reset_backend_cache()
    return fake


@pytest.fixture
def registry() -> ToolRegistry:
    ctx = ToolContext(config=None, security=OpenSecurity())
    reg = ToolRegistry(ctx)
    for tool in build_tools(ctx):
        reg.register(tool, replace=True)
    return reg


# --------------------------------------------------------------------------- #
#  type_text
# --------------------------------------------------------------------------- #
def test_type_text_passes_the_exact_string_through(registry, gui):
    sentence = "Dear Sir, thank you for your letter."
    result = registry.run("type_text", text=sentence)
    assert result.ok is True, result.error
    assert gui.named("write") == [("write", (sentence,), {"interval": 0.01})]
    assert result.output["chars"] == len(sentence)
    assert result.output["backend"] == "pyautogui"


def test_type_text_disables_the_corner_failsafe(registry, gui):
    assert gui.FAILSAFE is True, "the double must start with the real default"
    registry.run("type_text", text="hello")
    assert gui.FAILSAFE is False, (
        "pyautogui's failsafe would abort a legitimate action mid-way"
    )
    assert gui.PAUSE == 0.0


def test_type_text_clamps_a_hallucinated_interval(registry, gui):
    result = registry.run("type_text", text="hi", interval=900)
    assert result.ok is True
    assert gui.named("write")[0][2]["interval"] == input_tools._MAX_INTERVAL


def test_type_text_rejects_empty_and_oversized_text(registry, gui):
    assert registry.run("type_text", text="").ok is False
    huge = "x" * (input_tools._MAX_TEXT_CHARS + 1)
    result = registry.run("type_text", text=huge)
    assert result.ok is False
    assert "clipboard_paste_text" in (result.error or "")
    assert gui.named("write") == []


def test_type_text_flags_non_ascii_characters(registry, gui):
    result = registry.run("type_text", text="Grüße")
    assert result.ok is True
    assert "ü" in result.output["non_ascii"]
    assert "clipboard_paste_text" in result.output["note"]


def test_type_text_reports_a_backend_explosion_instead_of_raising(registry, gui):
    gui.raises_on = "write"
    result = registry.run("type_text", text="boom")
    assert result.ok is False
    assert "exploded" in (result.error or "")


# --------------------------------------------------------------------------- #
#  press_key / hotkey
# --------------------------------------------------------------------------- #
def test_press_key_normalizes_an_alias(registry, gui):
    result = registry.run("press_key", key="Return")
    assert result.ok is True
    assert gui.named("press")[0][1] == ("enter",)
    assert result.output["key"] == "enter"


def test_press_key_clamps_the_repeat_count(registry, gui):
    result = registry.run("press_key", key="tab", presses=5000)
    assert result.ok is True
    assert gui.named("press")[0][2]["presses"] == input_tools._MAX_PRESSES


def test_press_key_rejects_a_key_the_backend_does_not_know(registry, gui):
    result = registry.run("press_key", key="wibble")
    assert result.ok is False
    assert "key_names" in (result.error or "")
    assert gui.named("press") == []


def test_a_combo_string_and_a_list_reach_the_backend_identically(registry, gui):
    assert registry.run("hotkey", combo="ctrl+s").ok is True
    assert registry.run("hotkey", keys=["ctrl", "s"]).ok is True
    assert registry.run("hotkey", keys="ctrl+s").ok is True
    sent = [call[1] for call in gui.named("hotkey")]
    assert sent == [("ctrl", "s"), ("ctrl", "s"), ("ctrl", "s")]


def test_hotkey_maps_the_meta_key_to_the_platform_name(registry, gui):
    assert registry.run("hotkey", combo="super+d").ok is True
    expected = "command" if IS_MAC else "win"
    assert gui.named("hotkey")[0][1] == (expected, "d")


def test_hotkey_keeps_a_trailing_plus_as_a_key():
    assert input_tools._parse_keys("ctrl++") == ["ctrl", "+"]
    assert input_tools._parse_keys(["ctrl", "shift", "t"]) == ["ctrl", "shift", "t"]
    assert input_tools._parse_keys("alt tab") == ["alt", "tab"]


def test_hotkey_without_any_keys_fails(registry, gui):
    result = registry.run("hotkey")
    assert result.ok is False
    assert "combo" in (result.error or "")
    assert gui.named("hotkey") == []


def test_hotkey_refuses_an_absurd_number_of_keys(registry, gui):
    result = registry.run("hotkey", keys=["ctrl", "alt", "shift", "win", "a", "b", "c"])
    assert result.ok is False
    assert gui.named("hotkey") == []


# --------------------------------------------------------------------------- #
#  Coordinates
# --------------------------------------------------------------------------- #
def test_coordinates_are_clamped_to_the_screen(registry, gui):
    result = registry.run("mouse_move", x=99999, y=99999)
    assert result.ok is True, result.error
    assert gui.named("moveTo")[0][1] == (1919, 1079)
    assert result.output["clamped"] is True


def test_negative_coordinates_are_clamped_to_the_origin(registry, gui):
    result = registry.run("mouse_move", x=-4000, y=-1)
    assert result.ok is True
    assert gui.named("moveTo")[0][1] == (0, 0)


def test_in_range_coordinates_are_not_reported_as_clamped(registry, gui):
    result = registry.run("mouse_move", x=100, y=200, duration=0.5)
    assert result.output["clamped"] is False
    assert gui.named("moveTo")[0][1] == (100, 200)
    assert gui.named("moveTo")[0][2]["duration"] == 0.5


def test_mouse_move_rejects_non_numeric_coordinates(gui):
    result = input_tools._mouse_move("over there", 5)
    assert result.ok is False
    assert "number" in (result.error or "")


def test_mouse_move_clamps_a_runaway_duration(registry, gui):
    registry.run("mouse_move", x=1, y=1, duration=600)
    assert gui.named("moveTo")[0][2]["duration"] == input_tools._MAX_DURATION


# --------------------------------------------------------------------------- #
#  Clicking, scrolling, dragging
# --------------------------------------------------------------------------- #
def test_mouse_click_passes_button_and_clamps_clicks(registry, gui):
    result = registry.run("mouse_click", button="right", clicks=99)
    assert result.ok is True
    call = gui.named("click")[0]
    assert call[2]["button"] == "right"
    assert call[2]["clicks"] == input_tools._MAX_CLICKS


def test_mouse_click_rejects_an_unknown_button(registry, gui):
    result = registry.run("mouse_click", button="paw")
    assert result.ok is False
    assert "left" in (result.error or "")
    assert gui.named("click") == []


def test_mouse_click_requires_both_coordinates_or_neither(registry, gui):
    result = registry.run("mouse_click", x=10)
    assert result.ok is False
    assert "both" in (result.error or "")
    assert gui.named("click") == []


def test_mouse_click_clamps_its_target(registry, gui):
    result = registry.run("mouse_click", x=50000, y=20)
    assert result.ok is True
    assert gui.named("click")[0][1] == (1919, 20)


def test_mouse_scroll_passes_the_amount_and_rejects_zero(registry, gui):
    assert registry.run("mouse_scroll", amount=-3).ok is True
    assert gui.named("scroll")[0][1][0] == -3
    zero = registry.run("mouse_scroll", amount=0)
    assert zero.ok is False
    assert len(gui.named("scroll")) == 1


def test_mouse_drag_records_both_ends(registry, gui):
    result = registry.run("mouse_drag", x1=10, y1=20, x2=99999, y2=40)
    assert result.ok is True, result.error
    assert gui.named("moveTo")[0][1] == (10, 20)
    assert gui.named("dragTo")[0][1] == (1919, 40)
    assert result.output["clamped"] is True


def test_mouse_position_and_screen_size_report_the_backend_values(registry, gui):
    position = registry.run("mouse_position")
    assert position.output == {"x": 7, "y": 9}
    size = registry.run("screen_size")
    assert size.output == {"width": 1920, "height": 1080}


# --------------------------------------------------------------------------- #
#  Clipboard paste
# --------------------------------------------------------------------------- #
def test_clipboard_paste_sets_the_clipboard_then_sends_the_chord(
    registry, gui, monkeypatch
):
    from jarvis.core.contracts import ToolResult

    copied: List[str] = []

    def record(text: str) -> ToolResult:
        copied.append(text)
        return ToolResult.success({"chars": len(text)})

    monkeypatch.setattr(input_tools, "_set_clipboard", record)
    result = registry.run("clipboard_paste_text", text="a long dictated paragraph")
    assert result.ok is True, result.error
    assert copied == ["a long dictated paragraph"]
    expected = ("command", "v") if IS_MAC else ("ctrl", "v")
    assert gui.named("hotkey")[0][1] == expected


def test_clipboard_paste_does_not_paste_when_the_copy_failed(registry, gui, monkeypatch):
    from jarvis.core.contracts import ToolResult

    monkeypatch.setattr(
        input_tools, "_set_clipboard", lambda text: ToolResult.failure("no clipboard")
    )
    result = registry.run("clipboard_paste_text", text="secret")
    assert result.ok is False
    assert "no clipboard" in (result.error or "")
    assert gui.named("hotkey") == [], "pasting would have dumped the old clipboard"


def test_clipboard_paste_accepts_a_custom_chord(registry, gui, monkeypatch):
    from jarvis.core.contracts import ToolResult

    monkeypatch.setattr(
        input_tools, "_set_clipboard", lambda text: ToolResult.success({"chars": len(text)})
    )
    result = registry.run("clipboard_paste_text", text="ls -la", chord="ctrl+shift+v")
    assert result.ok is True
    assert gui.named("hotkey")[0][1] == ("ctrl", "shift", "v")


def test_set_clipboard_fails_cleanly_when_nothing_can_do_it(gui):
    result = input_tools._set_clipboard("text")
    assert result.ok is False
    assert result.error


# --------------------------------------------------------------------------- #
#  key_names
# --------------------------------------------------------------------------- #
def test_key_names_comes_from_the_backend_when_there_is_one(registry, gui):
    result = registry.run("key_names")
    assert result.ok is True
    assert result.output["source"] == "pyautogui"
    assert "enter" in result.output["keys"]
    assert result.output["count"] == len(result.output["keys"])


def test_key_names_still_answers_without_a_backend(registry):
    result = registry.run("key_names")
    assert result.ok is True
    assert result.output["source"] == "builtin"
    assert result.output["count"] > 20
    for key in ("enter", "escape", "f12", "up"):
        assert key in result.output["keys"]


# --------------------------------------------------------------------------- #
#  No backend at all
# --------------------------------------------------------------------------- #
# key_names is deliberately absent: it answers from a built-in vocabulary and
# has no device to fail against.
NO_BACKEND_CALLS = [
    ("type_text", {"text": "hello"}),
    ("press_key", {"key": "enter"}),
    ("hotkey", {"combo": "ctrl+s"}),
    ("mouse_move", {"x": 10, "y": 10}),
    ("mouse_click", {}),
    ("mouse_scroll", {"amount": 3}),
    ("mouse_drag", {"x1": 0, "y1": 0, "x2": 10, "y2": 10}),
    ("mouse_position", {}),
    ("screen_size", {}),
    ("clipboard_paste_text", {"text": "hi"}),
]


@pytest.mark.parametrize("name,kwargs", NO_BACKEND_CALLS, ids=[c[0] for c in NO_BACKEND_CALLS])
def test_every_tool_fails_rather_than_raises_without_a_backend(registry, name, kwargs):
    result = registry.run(name, **kwargs)
    assert result.ok is False, f"{name} claimed success with no backend"
    assert isinstance(result.error, str) and result.error.strip(), (
        f"{name} failed without saying why"
    )


def test_is_available_is_false_with_no_backend():
    assert input_tools.is_available() is False


def test_is_available_is_true_with_a_backend(gui):
    assert input_tools.is_available() is True


# --------------------------------------------------------------------------- #
#  Windows fallback
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not IS_WINDOWS, reason="the win32 fallback only exists on Windows")
def test_win32_fallback_types_the_text_as_unicode_units(monkeypatch):
    api, con = FakeWin32Api(), FakeWin32Con()
    monkeypatch.setitem(sys.modules, "win32api", api)
    monkeypatch.setitem(sys.modules, "win32con", con)
    input_tools.reset_backend_cache()

    result = input_tools._type_text("Hi é", interval=0)
    assert result.ok is True, result.error
    assert result.output["backend"] == "win32"
    typed = "".join(
        chr(scan)
        for _vk, scan, flags, _extra in api.events
        if flags == con.KEYEVENTF_UNICODE
    )
    assert typed == "Hi é"


@pytest.mark.skipif(not IS_WINDOWS, reason="the win32 fallback only exists on Windows")
def test_win32_fallback_presses_a_chord(monkeypatch):
    api, con = FakeWin32Api(), FakeWin32Con()
    monkeypatch.setitem(sys.modules, "win32api", api)
    monkeypatch.setitem(sys.modules, "win32con", con)
    input_tools.reset_backend_cache()

    result = input_tools.send_hotkey(["ctrl", "s"])
    assert result.ok is True, result.error
    assert result.output["backend"] == "win32"
    downs = [vk for vk, _s, flags, _e in api.events if flags == 0]
    ups = [vk for vk, _s, flags, _e in api.events if flags == con.KEYEVENTF_KEYUP]
    assert downs == [0x11, ord("S")]
    assert ups == [ord("S"), 0x11], "keys must be released in reverse order"


@pytest.mark.skipif(not IS_WINDOWS, reason="the win32 fallback only exists on Windows")
def test_screen_size_falls_back_to_system_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "win32api", FakeWin32Api(1280, 800))
    monkeypatch.setitem(sys.modules, "win32con", FakeWin32Con())
    input_tools.reset_backend_cache()
    result = input_tools._screen_size()
    assert result.output == {"width": 1280, "height": 800}


# --------------------------------------------------------------------------- #
#  Specs
# --------------------------------------------------------------------------- #
def test_build_tools_exposes_the_expected_surface():
    tools = build_tools(ToolContext(config=None, security=OpenSecurity()))
    by_name = {t.spec.name: t.spec for t in tools}
    assert set(by_name) == {
        "type_text", "press_key", "hotkey", "mouse_move", "mouse_click",
        "mouse_scroll", "mouse_drag", "mouse_position", "screen_size",
        "clipboard_paste_text", "key_names",
    }
    for spec in by_name.values():
        assert spec.description, f"{spec.name} has no description"
    for name in ("type_text", "hotkey", "mouse_click", "mouse_drag", "clipboard_paste_text"):
        assert by_name[name].dangerous is True, f"{name} should be marked dangerous"
    assert by_name["mouse_position"].dangerous is False


def test_hotkey_spec_declares_keys_as_an_array():
    tools = {t.spec.name: t.spec for t in build_tools(None)}
    params = {p.name: p for p in tools["hotkey"].params}
    assert params["keys"].type == "array", (
        "an array parameter is what lets the registry turn 'ctrl+s' into a list"
    )
    assert params["combo"].type == "string"
    assert params["keys"].required is False and params["combo"].required is False
