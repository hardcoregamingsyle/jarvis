"""Every built-in tool module must actually be registered at startup.

``input_tools`` and ``window_tools`` were fully implemented and fully tested
while being absent from ``ToolRegistry.load_builtin()``, so at runtime JARVIS
could not type, click, press a key, or minimise a window. The unit tests passed
throughout — they imported the modules directly.

These tests check the startup path itself, and pin the deliberate ordering that
resolves the name collision between ``app_tools`` and ``window_tools``.
"""

from __future__ import annotations

import importlib

import pytest

from jarvis.core.security import SecurityGate
from jarvis.tools import create_registry
from jarvis.tools.registry import ToolContext, ToolRegistry


BUILTIN_MODULES = (
    "file_tools", "system_tools", "process_tools", "web_tools",
    "app_tools", "input_tools", "window_tools", "tool_maker",
)


@pytest.fixture
def registry(config):
    return create_registry(config, SecurityGate(config.security))


@pytest.mark.parametrize("modname", BUILTIN_MODULES)
def test_every_builtin_module_contributes_at_runtime(modname, registry, config):
    """Each module must put at least one of its tools into a live registry.

    Checked behaviourally rather than by reading the source: a module can be
    named in the tuple and still contribute nothing (a failed import is caught
    and logged), which is exactly the state this test exists to detect.
    """
    ctx = ToolContext(
        config=config, security=SecurityGate(config.security),
        bus=None, memory=None, extra={},
    )
    module = importlib.import_module(f"jarvis.tools.{modname}")
    produced = {t.name for t in module.build_tools(ctx)}
    assert produced, f"{modname}.build_tools() returned nothing"

    registered = set(registry.names())
    assert produced & registered, (
        f"{modname} produces {sorted(produced)[:5]}... but none of it reached a "
        f"live registry — it is missing from ToolRegistry.load_builtin()"
    )


@pytest.mark.parametrize("tool_name", [
    # input_tools — none of these existed at runtime before
    "type_text", "press_key", "hotkey", "mouse_move", "mouse_click",
    "mouse_scroll", "mouse_position", "screen_size", "clipboard_paste_text",
    # window_tools
    "list_windows", "focus_window", "minimize_window", "maximize_window",
    "restore_window", "move_window", "snap_window", "active_window",
])
def test_desktop_control_tools_are_registered(registry, tool_name):
    assert registry.has(tool_name), f"{tool_name} is not available at runtime"


def test_registry_exposes_more_than_the_pre_registration_count(registry):
    """Guards against a module silently dropping back out of the tuple."""
    assert len(registry.names()) >= 70, (
        f"only {len(registry.names())} tools registered; input_tools/window_tools "
        f"may have fallen out of load_builtin()"
    )


# --------------------------------------------------------------------------- #
#  The app_tools / window_tools collision
# --------------------------------------------------------------------------- #
COLLIDING = ("list_windows", "focus_window")


@pytest.mark.parametrize("name", COLLIDING)
def test_window_tools_wins_the_collision(registry, name):
    """Registration order is load-bearing: replace=True means last wins.

    window_tools is listed after app_tools on purpose — it tries exact then
    case-insensitive title matching, reports ambiguity instead of guessing, and
    verifies with GetForegroundWindow that focus actually took effect (Windows
    blocks focus stealing, so the naive call reports success while doing
    nothing).
    """
    tool = registry.get(name)
    fn = getattr(tool, "fn", None) or getattr(tool, "_fn", None)
    assert fn is not None, f"{name} is not a FunctionTool"
    assert fn.__module__ == "jarvis.tools.window_tools", (
        f"{name} resolved to {fn.__module__}; app_tools has shadowed the richer "
        f"window_tools implementation — check the order in load_builtin()"
    )


def test_the_collision_is_real_and_not_imagined(config):
    """If the overlap ever disappears, the ordering comment should go too."""
    from jarvis.tools import app_tools, window_tools

    ctx = ToolContext(
        config=config, security=SecurityGate(config.security),
        bus=None, memory=None, extra={},
    )
    app_names = {t.name for t in app_tools.build_tools(ctx)}
    window_names = {t.name for t in window_tools.build_tools(ctx)}

    assert set(COLLIDING) <= (app_names & window_names)


def test_a_broken_builtin_module_does_not_break_startup(config, monkeypatch):
    """One bad module must not take the whole registry down with it."""
    real_import = importlib.import_module

    def explode(name, *args, **kwargs):
        if name == "jarvis.tools.input_tools":
            raise RuntimeError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", explode)

    ctx = ToolContext(
        config=config, security=SecurityGate(config.security),
        bus=None, memory=None, extra={},
    )
    reg = ToolRegistry(ctx)
    reg.load_builtin()

    assert reg.has("read_file"), "a single broken module aborted the whole load"
    assert not reg.has("type_text")
