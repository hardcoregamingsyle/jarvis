"""Window-management tool tests.

No test here may enumerate, focus, move or close a window on the developer's
real desktop: the pywin32 bundle is replaced with recorders, and the autouse
fixture pins the backend to "absent" so a test that forgets to inject one
exercises the failure path instead of the machine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from jarvis.core.contracts import ToolResult
from jarvis.core.platform_utils import IS_WINDOWS
from jarvis.tools import window_tools
from jarvis.tools.registry import ToolContext, ToolRegistry
from jarvis.tools.window_tools import build_tools

# Captured before any fixture replaces them, so the read-only tests can reach
# the genuine articles.
_REAL_BACKEND = window_tools._backend
_REAL_PROCESS_NAME = window_tools._process_name


# --------------------------------------------------------------------------- #
#  Doubles
# --------------------------------------------------------------------------- #
class FakeWin32Con:
    SW_MINIMIZE = 6
    SW_MAXIMIZE = 3
    SW_RESTORE = 9
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    WM_CLOSE = 0x0010
    VK_MENU = 0x12
    KEYEVENTF_KEYUP = 0x0002


class FakeWin32Gui:
    """Records every window call and honours a configurable focus policy."""

    def __init__(self, windows: Sequence[Dict[str, Any]], state: Dict[str, Any]) -> None:
        self._windows = {int(w["hwnd"]): dict(w) for w in windows}
        self._order = [int(w["hwnd"]) for w in windows]
        self._titles: Dict[int, str] = {
            int(w["hwnd"]): str(w.get("title", "")) for w in windows
        }
        for window in windows:
            for child_handle, child_text in window.get("children", []):
                self._titles[int(child_handle)] = str(child_text)
        self.state = state
        self.calls: List[Tuple[Any, ...]] = []

    def EnumWindows(self, callback: Any, extra: Any) -> None:  # noqa: N802
        for handle in self._order:
            callback(handle, extra)

    def EnumChildWindows(self, handle: int, callback: Any, extra: Any) -> None:  # noqa: N802
        for child_handle, _text in self._windows[int(handle)].get("children", []):
            callback(int(child_handle), extra)

    def IsWindowVisible(self, handle: int) -> bool:  # noqa: N802
        return bool(self._windows[int(handle)].get("visible", True))

    def GetWindowText(self, handle: int) -> str:  # noqa: N802
        return self._titles.get(int(handle), "")

    def GetWindowRect(self, handle: int) -> Tuple[int, int, int, int]:  # noqa: N802
        return tuple(self._windows[int(handle)].get("rect", (0, 0, 0, 0)))  # type: ignore[return-value]

    def IsIconic(self, handle: int) -> bool:  # noqa: N802
        return bool(self._windows[int(handle)].get("iconic", False))

    def ShowWindow(self, handle: int, command: int) -> bool:  # noqa: N802
        self.calls.append(("ShowWindow", int(handle), int(command)))
        if command == FakeWin32Con.SW_RESTORE:
            self._windows[int(handle)]["iconic"] = False
        return True

    def SetForegroundWindow(self, handle: int) -> bool:  # noqa: N802
        self.calls.append(("SetForegroundWindow", int(handle)))
        policy = self.state["focus_policy"]
        if policy == "always" or (policy == "after_attach" and self.state["attached"]):
            self.state["foreground"] = int(handle)
            return True
        return False

    def GetForegroundWindow(self) -> int:  # noqa: N802
        return int(self.state["foreground"])

    def MoveWindow(self, handle, x, y, width, height, repaint):  # noqa: N802
        self.calls.append(("MoveWindow", int(handle), x, y, width, height, repaint))
        return True

    def SetWindowPos(self, handle, insert_after, x, y, width, height, flags):  # noqa: N802
        self.calls.append(("SetWindowPos", int(handle), insert_after, flags))
        return True

    def PostMessage(self, handle, message, wparam, lparam):  # noqa: N802
        self.calls.append(("PostMessage", int(handle), int(message)))
        return True

    def named(self, name: str) -> List[Tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == name]


class FakeWin32Process:
    def __init__(self, pids: Dict[int, int], state: Dict[str, Any]) -> None:
        self._pids = pids
        self.state = state
        self.attach_calls: List[Tuple[int, int, bool]] = []

    def GetWindowThreadProcessId(self, handle: int) -> Tuple[int, int]:  # noqa: N802
        return (1000 + int(handle) % 7, self._pids.get(int(handle), 0))

    def AttachThreadInput(self, source: int, target: int, attach: bool) -> bool:  # noqa: N802
        self.attach_calls.append((int(source), int(target), bool(attach)))
        self.state["attached"] = bool(attach)
        return True


class FakeWin32Api:
    def __init__(self, screen: Tuple[int, int] = (1920, 1080)) -> None:
        self._screen = screen
        self.events: List[Tuple[int, int, int, int]] = []

    def GetCurrentThreadId(self) -> int:  # noqa: N802
        return 999

    def GetSystemMetrics(self, index: int) -> int:  # noqa: N802
        return self._screen[0] if index == 0 else self._screen[1]

    def keybd_event(self, vk: int, scan: int, flags: int, extra: int) -> None:
        self.events.append((vk, scan, flags, extra))


class OpenSecurity:
    """Permits everything, so the tests exercise the tools and not the gate."""

    def check_tool(self, spec: Any, cleaned: Any) -> Any:
        class Decision:
            allowed = True
            requires_confirmation = False

        return Decision()

    def allows(self, decision: Any) -> bool:
        return True


DEFAULT_WINDOWS: List[Dict[str, Any]] = [
    {"hwnd": 101, "title": "Notepad", "rect": (0, 0, 800, 600), "pid": 4242,
     "children": [(9001, "File name:"), (9002, "")]},
    {"hwnd": 102, "title": "Untitled - Notepad", "rect": (10, 10, 810, 610), "pid": 4243},
    {"hwnd": 103, "title": "a hidden helper", "rect": (0, 0, 10, 10), "pid": 4244,
     "visible": False},
    {"hwnd": 104, "title": "   ", "rect": (0, 0, 10, 10), "pid": 4245},
]


def make_backend(
    windows: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    focus_policy: str = "always",
    foreground: int = 0,
    screen: Tuple[int, int] = (1920, 1080),
) -> window_tools.Win32Backend:
    spec = list(DEFAULT_WINDOWS if windows is None else windows)
    state = {"foreground": foreground, "attached": False, "focus_policy": focus_policy}
    return window_tools.Win32Backend(
        gui=FakeWin32Gui(spec, state),
        con=FakeWin32Con,
        process=FakeWin32Process({int(w["hwnd"]): int(w.get("pid", 0)) for w in spec}, state),
        api=FakeWin32Api(screen),
    )


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch):
    """Default to "no backend at all" so nothing reaches the real desktop."""
    monkeypatch.setattr(window_tools, "_backend", lambda: None)
    monkeypatch.setattr(window_tools, "which", lambda name: None)
    monkeypatch.setattr(
        window_tools, "_process_name", lambda pid: f"proc{pid}.exe" if pid else None
    )
    window_tools.reset_backend_cache()
    yield
    window_tools.reset_backend_cache()


@pytest.fixture
def make_bundle(monkeypatch: pytest.MonkeyPatch):
    def factory(**kwargs: Any) -> window_tools.Win32Backend:
        bundle = make_backend(**kwargs)
        monkeypatch.setattr(window_tools, "_backend", lambda: bundle)
        return bundle

    return factory


@pytest.fixture
def backend(make_bundle) -> window_tools.Win32Backend:
    return make_bundle()


@pytest.fixture
def hotkeys(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    sent: List[List[str]] = []

    def fake_send(keys: Sequence[str]) -> ToolResult:
        sent.append(list(keys))
        return ToolResult.success({"keys": list(keys)})

    monkeypatch.setattr(window_tools, "_send_hotkey", fake_send)
    return sent


@pytest.fixture
def registry() -> ToolRegistry:
    ctx = ToolContext(config=None, security=OpenSecurity())
    reg = ToolRegistry(ctx)
    for tool in build_tools(ctx):
        reg.register(tool, replace=True)
    return reg


# --------------------------------------------------------------------------- #
#  Enumeration
# --------------------------------------------------------------------------- #
def test_enumeration_builds_the_expected_dicts(backend):
    windows = window_tools._enumerate_win32(backend, visible_only=True)
    assert windows == [
        {
            "handle": 101,
            "title": "Notepad",
            "process": "proc4242.exe",
            "pid": 4242,
            "rect": {"left": 0, "top": 0, "right": 800, "bottom": 600,
                     "width": 800, "height": 600},
            "minimized": False,
        },
        {
            "handle": 102,
            "title": "Untitled - Notepad",
            "process": "proc4243.exe",
            "pid": 4243,
            "rect": {"left": 10, "top": 10, "right": 810, "bottom": 610,
                     "width": 800, "height": 600},
            "minimized": False,
        },
    ]


def test_enumeration_can_include_hidden_and_untitled_windows(backend):
    windows = window_tools._enumerate_win32(backend, visible_only=False)
    assert [w["handle"] for w in windows] == [101, 102, 103, 104]


def test_minimized_state_is_reported(make_bundle):
    bundle = make_bundle(
        windows=[{"hwnd": 7, "title": "Mail", "rect": (0, 0, 10, 10), "pid": 3,
                  "iconic": True}]
    )
    windows = window_tools._enumerate_win32(bundle, visible_only=True)
    assert windows[0]["minimized"] is True


def test_list_windows_tool_wraps_the_enumeration(registry, backend):
    result = registry.run("list_windows")
    assert result.ok is True, result.error
    assert result.output["count"] == 2
    assert [w["title"] for w in result.output["windows"]] == [
        "Notepad", "Untitled - Notepad"
    ]


def test_process_name_is_none_for_a_missing_pid():
    # The real lookup, not the fixture's stub: an unknown pid must come back as
    # None rather than raising out of psutil.
    assert _REAL_PROCESS_NAME(None) is None
    assert _REAL_PROCESS_NAME(0) is None
    assert _REAL_PROCESS_NAME(4294967295) is None


# --------------------------------------------------------------------------- #
#  Title matching
# --------------------------------------------------------------------------- #
def _w(handle: int, title: str) -> Dict[str, Any]:
    return {"handle": handle, "title": title}


WINDOWS = [_w(1, "Notepad"), _w(2, "Untitled - Notepad"), _w(3, "Mail — Inbox")]


def test_exact_title_beats_a_substring_match():
    window, error = window_tools._select_window(WINDOWS, "Notepad")
    assert error is None
    assert window["handle"] == 1


def test_case_insensitive_match_is_used_when_nothing_is_exact():
    window, error = window_tools._select_window(WINDOWS, "MAIL — INBOX")
    assert error is None
    assert window["handle"] == 3


def test_a_unique_substring_matches():
    window, error = window_tools._select_window(WINDOWS, "untitled")
    assert error is None
    assert window["handle"] == 2


def test_several_substring_matches_are_reported_as_ambiguous():
    window, error = window_tools._select_window(WINDOWS, "note")
    assert window is None
    assert error and "2 windows match" in error
    assert "Notepad" in error and "Untitled - Notepad" in error
    assert "handle" in error


def test_duplicate_exact_titles_are_ambiguous_too():
    windows = [_w(1, "Chrome"), _w(2, "Chrome")]
    window, error = window_tools._select_window(windows, "Chrome")
    assert window is None
    assert error and "handle" in error


def test_a_handle_selects_directly():
    for target in (2, "2", "0x2"):
        window, error = window_tools._select_window(WINDOWS, target)
        assert error is None, f"{target!r} did not resolve"
        assert window["handle"] == 2


def test_an_unmatched_title_lists_what_is_open():
    window, error = window_tools._select_window(WINDOWS, "Photoshop")
    assert window is None
    assert error and "no window matches" in error
    assert "Notepad" in error


def test_an_empty_target_is_rejected():
    window, error = window_tools._select_window(WINDOWS, "   ")
    assert window is None
    assert error and "required" in error


# --------------------------------------------------------------------------- #
#  Focus
# --------------------------------------------------------------------------- #
def test_focus_succeeds_directly_when_windows_allows_it(registry, make_bundle):
    bundle = make_bundle(focus_policy="always")
    result = registry.run("focus_window", target="Notepad")
    assert result.ok is True, result.error
    assert result.output["method"] == "direct"
    assert bundle.gui.state["foreground"] == 101
    assert bundle.process.attach_calls == [], "no workaround was needed"


def test_focus_failure_is_detected_and_reported(registry, make_bundle):
    bundle = make_bundle(focus_policy="never", foreground=555)
    result = registry.run("focus_window", target="Notepad")
    assert result.ok is False, "a focus call that did nothing must not report success"
    assert "refused" in (result.error or "")
    assert bundle.gui.named("SetForegroundWindow"), "it must actually have tried"
    assert bundle.gui.state["foreground"] == 555


def test_focus_attaches_the_input_queue_and_detaches_again(registry, make_bundle):
    bundle = make_bundle(focus_policy="after_attach", foreground=555)
    result = registry.run("focus_window", target="Notepad")
    assert result.ok is True, result.error
    assert result.output["method"] == "attach_thread_input"
    assert [call[2] for call in bundle.process.attach_calls] == [True, False]
    assert any(vk == FakeWin32Con.VK_MENU for vk, _s, _f, _e in bundle.api.events), (
        "the ALT nudge is what makes Windows accept the focus change"
    )


def test_focus_restores_a_minimized_window_before_raising_it(registry, make_bundle):
    bundle = make_bundle(
        windows=[{"hwnd": 101, "title": "Notepad", "rect": (0, 0, 8, 6), "pid": 1,
                  "iconic": True}]
    )
    result = registry.run("focus_window", target="Notepad")
    assert result.ok is True, result.error
    kinds = [call[0] for call in bundle.gui.calls]
    assert kinds[0] == "ShowWindow"
    assert bundle.gui.calls[0][2] == FakeWin32Con.SW_RESTORE
    assert "SetForegroundWindow" in kinds


def test_focus_reports_an_ambiguous_title_instead_of_guessing(registry, backend):
    result = registry.run("focus_window", target="note")
    assert result.ok is False
    assert "2 windows match" in (result.error or "")
    assert backend.gui.named("SetForegroundWindow") == []


# --------------------------------------------------------------------------- #
#  Show / close / pin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tool,constant",
    [
        ("minimize_window", FakeWin32Con.SW_MINIMIZE),
        ("maximize_window", FakeWin32Con.SW_MAXIMIZE),
        ("restore_window", FakeWin32Con.SW_RESTORE),
    ],
)
def test_show_window_commands(registry, backend, tool, constant):
    result = registry.run(tool, target="Notepad")
    assert result.ok is True, result.error
    assert backend.gui.named("ShowWindow") == [("ShowWindow", 101, constant)]


def test_close_window_posts_wm_close(registry, backend):
    result = registry.run("close_window", target="Untitled - Notepad")
    assert result.ok is True, result.error
    assert backend.gui.named("PostMessage") == [
        ("PostMessage", 102, FakeWin32Con.WM_CLOSE)
    ]


def test_always_on_top_toggles_the_topmost_flag(registry, backend):
    assert registry.run("set_always_on_top", target="Notepad", on=True).ok is True
    assert registry.run("set_always_on_top", target="Notepad", on=False).ok is True
    inserts = [call[2] for call in backend.gui.named("SetWindowPos")]
    assert inserts == [FakeWin32Con.HWND_TOPMOST, FakeWin32Con.HWND_NOTOPMOST]
    flags = backend.gui.named("SetWindowPos")[0][3]
    assert flags == FakeWin32Con.SWP_NOMOVE | FakeWin32Con.SWP_NOSIZE


def test_window_text_reads_the_child_controls(registry, backend):
    result = registry.run("window_text", target="Notepad")
    assert result.ok is True, result.error
    assert result.output["title"] == "Notepad"
    assert result.output["text"] == ["File name:"], "blank child text is noise"


def test_active_window_reports_the_foreground_window(registry, make_bundle):
    make_bundle(foreground=102)
    result = registry.run("active_window")
    assert result.ok is True, result.error
    assert result.output["handle"] == 102
    assert result.output["title"] == "Untitled - Notepad"


def test_active_window_fails_when_nothing_has_focus(registry, make_bundle):
    make_bundle(foreground=0)
    result = registry.run("active_window")
    assert result.ok is False
    assert "focus" in (result.error or "")


# --------------------------------------------------------------------------- #
#  Move
# --------------------------------------------------------------------------- #
def test_move_window_keeps_the_current_size_when_none_is_given(registry, backend):
    result = registry.run("move_window", target="Notepad", x=100, y=120)
    assert result.ok is True, result.error
    assert backend.gui.named("MoveWindow") == [
        ("MoveWindow", 101, 100, 120, 800, 600, True)
    ]


def test_move_window_clamps_a_hallucinated_geometry(registry, backend):
    result = registry.run(
        "move_window", target="Notepad", x=99999, y=99999, width=8000, height=8000
    )
    assert result.ok is True, result.error
    assert backend.gui.named("MoveWindow") == [
        ("MoveWindow", 101, 0, 0, 1920, 1080, True)
    ]
    assert result.output["rect"] == {"left": 0, "top": 0, "width": 1920, "height": 1080}


def test_move_window_refuses_a_hopeless_size(registry, backend):
    registry.run("move_window", target="Notepad", x=10, y=10, width=1, height=1)
    call = backend.gui.named("MoveWindow")[0]
    assert call[4] >= window_tools._MIN_WINDOW_SIZE
    assert call[5] >= window_tools._MIN_WINDOW_SIZE


# --------------------------------------------------------------------------- #
#  Snap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "side,chord",
    [("left", ["win", "left"]), ("right", ["win", "right"]),
     ("top", ["win", "up"]), ("bottom", ["win", "down"])],
)
def test_snap_sends_the_windows_chord(registry, backend, hotkeys, side, chord):
    result = registry.run("snap_window", target="Notepad", side=side)
    assert result.ok is True, result.error
    assert hotkeys == [chord]
    assert result.output["side"] == side


def test_snap_maximize_uses_show_window_not_a_chord(registry, backend, hotkeys):
    result = registry.run("snap_window", target="Notepad", side="maximize")
    assert result.ok is True, result.error
    assert hotkeys == []
    assert backend.gui.named("ShowWindow") == [
        ("ShowWindow", 101, FakeWin32Con.SW_MAXIMIZE)
    ]


def test_snap_rejects_an_unknown_side_without_touching_anything(
    registry, backend, hotkeys
):
    result = registry.run("snap_window", target="Notepad", side="diagonal")
    assert result.ok is False
    assert "left" in (result.error or "") and "maximize" in (result.error or "")
    assert hotkeys == []
    assert backend.gui.calls == []


def test_snap_does_not_send_a_chord_when_focus_fails(registry, make_bundle, hotkeys):
    make_bundle(focus_policy="never", foreground=555)
    result = registry.run("snap_window", target="Notepad", side="left")
    assert result.ok is False
    assert "focus" in (result.error or "")
    assert hotkeys == [], "the chord would have snapped whatever was focused instead"


# --------------------------------------------------------------------------- #
#  No backend
# --------------------------------------------------------------------------- #
NO_BACKEND_CALLS = [
    ("list_windows", {}),
    ("focus_window", {"target": "Notepad"}),
    ("close_window", {"target": "Notepad"}),
    ("minimize_window", {"target": "Notepad"}),
    ("maximize_window", {"target": "Notepad"}),
    ("restore_window", {"target": "Notepad"}),
    ("move_window", {"target": "Notepad", "x": 0, "y": 0}),
    ("snap_window", {"target": "Notepad", "side": "left"}),
    ("active_window", {}),
    ("window_text", {"target": "Notepad"}),
    ("set_always_on_top", {"target": "Notepad"}),
]


@pytest.mark.parametrize("name,kwargs", NO_BACKEND_CALLS, ids=[c[0] for c in NO_BACKEND_CALLS])
def test_every_tool_fails_rather_than_raises_without_a_backend(registry, name, kwargs):
    result = registry.run(name, **kwargs)
    assert result.ok is False, f"{name} claimed success with no backend"
    assert isinstance(result.error, str) and result.error.strip(), (
        f"{name} failed without saying why"
    )


def test_is_available_is_false_without_a_backend():
    assert window_tools.is_available() is False


# --------------------------------------------------------------------------- #
#  Specs
# --------------------------------------------------------------------------- #
def test_build_tools_exposes_the_expected_surface():
    by_name = {t.spec.name: t.spec for t in build_tools(None)}
    assert set(by_name) == {
        "list_windows", "focus_window", "close_window", "minimize_window",
        "maximize_window", "restore_window", "move_window", "snap_window",
        "active_window", "window_text", "set_always_on_top",
    }
    for spec in by_name.values():
        assert spec.description, f"{spec.name} has no description"
    assert by_name["close_window"].dangerous is True
    assert by_name["list_windows"].dangerous is False
    params = {p.name: p for p in by_name["move_window"].params}
    assert params["x"].type == "integer" and params["x"].required is True
    assert params["width"].required is False


# --------------------------------------------------------------------------- #
#  Real Windows, read-only
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not IS_WINDOWS, reason="pywin32 enumeration is Windows-only")
def test_the_real_enumeration_parses_this_desktop(monkeypatch):
    """Read-only smoke test: walk the real window list, change nothing."""
    bundle = _REAL_BACKEND()
    if bundle is None:
        pytest.skip("pywin32 is not installed")
    monkeypatch.setattr(window_tools, "_process_name", lambda pid: None)
    windows = window_tools._enumerate_win32(bundle, visible_only=False)
    assert len(windows) >= 1, "a Windows session always has top-level windows"
    for window in windows:
        assert set(window) == {
            "handle", "title", "process", "pid", "rect", "minimized"
        }
        assert isinstance(window["handle"], int) and window["handle"] > 0
        assert set(window["rect"]) == {
            "left", "top", "right", "bottom", "width", "height"
        }
        assert isinstance(window["minimized"], bool)
