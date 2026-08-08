"""Registry, FunctionTool, security enforcement and generated-tool loading."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from jarvis.core.contracts import Tool, ToolResult, ToolSpec
from jarvis.core.events import EventBus, Events
from jarvis.tools.registry import (
    FunctionTool,
    ToolContext,
    ToolRegistry,
    _spec_from_callable,
    generated_module_name,
    safe_truncate,
    tool,
)


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    allowed: bool = True
    requires_confirmation: bool = False
    reason: Optional[str] = None


class FakeSecurity:
    def __init__(self, decision: Decision = None, allows_result: bool = True) -> None:
        self.decision = decision or Decision()
        self.allows_result = allows_result
        self.check_calls: list = []
        self.allows_calls: list = []

    def check_tool(self, spec, cleaned):
        self.check_calls.append((spec.name, dict(cleaned)))
        return self.decision

    def allows(self, decision) -> bool:
        self.allows_calls.append(decision)
        return self.allows_result

    def is_protected(self, path: str) -> bool:
        return False


class NoAllowsSecurity:
    """A duck-typed security object without ``.allows`` — must be treated as refusal."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def check_tool(self, spec, cleaned):
        return self.decision


class FakeConfig:
    def __init__(self, tools_dir_path):
        self._dir = tools_dir_path

    def tools_dir(self):
        return self._dir


def _mk_ctx(tmp_path, security=None, bus=None):
    tools_dir = tmp_path / "generated"
    tools_dir.mkdir(exist_ok=True)
    return ToolContext(
        config=FakeConfig(tools_dir),
        security=security if security is not None else FakeSecurity(),
        bus=bus,
    )


# --------------------------------------------------------------------------- #
#  FunctionTool schema derivation
# --------------------------------------------------------------------------- #
def test_function_tool_derives_spec_from_signature():
    def sample(a: str, b: int = 3, c: bool = False) -> str:
        """A short summary line.

        Long details below.
        """
        return f"{a}-{b}-{c}"

    ft = FunctionTool(sample, name="sample", description=None)
    spec = ft.spec
    assert spec.name == "sample"
    assert spec.description == "A short summary line."
    types = {p.name: (p.type, p.required, p.default) for p in spec.params}
    assert types["a"] == ("string", True, None)
    assert types["b"] == ("integer", False, 3)
    assert types["c"] == ("boolean", False, False)


def test_function_tool_optional_hint_is_unwrapped():
    def sample(a: Optional[int] = None) -> None:
        pass

    ft = FunctionTool(sample, name="sample")
    p = {p.name: p for p in ft.spec.params}
    assert p["a"].type == "integer"
    assert p["a"].required is False


def test_function_tool_detects_var_kwargs():
    def sample(a: str, **rest: Any):
        return rest

    _, accepts = _spec_from_callable(sample, name="s", description=None, dangerous=False)
    assert accepts is True


def test_tool_decorator_produces_function_tool():
    @tool(name="hi", description="says hi", dangerous=False)
    def hi(who: str) -> str:
        return f"hi, {who}"

    assert isinstance(hi, FunctionTool)
    assert hi.spec.name == "hi"
    result = hi.run(who="world")
    assert result.ok is True
    assert result.output == "hi, world"


# --------------------------------------------------------------------------- #
#  Registry basics
# --------------------------------------------------------------------------- #
def test_register_and_get(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    reg.register(add)
    assert reg.has("add")
    assert reg.get("add") is add
    assert "add" in reg.names()
    assert any(s.name == "add" for s in reg.list())
    assert reg.schemas()[0]["name"] == "add"


def test_register_duplicate_rejected_unless_replace(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="ping")
    def ping() -> str:
        return "pong"

    reg.register(ping)
    with pytest.raises(ValueError):
        reg.register(ping)
    reg.register(ping, replace=True)  # ok


def test_unregister(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="foo")
    def foo() -> None:
        pass

    reg.register(foo)
    assert reg.unregister("foo") is True
    assert reg.has("foo") is False
    assert reg.unregister("foo") is False


def test_describe_includes_dangerous_marker(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="safe")
    def safe() -> None:
        pass

    @tool(name="risky", dangerous=True)
    def risky() -> None:
        pass

    reg.register(safe)
    reg.register(risky)
    text = reg.describe()
    assert "safe" in text
    assert "risky" in text
    assert "[DANGEROUS]" in text


# --------------------------------------------------------------------------- #
#  run() validation
# --------------------------------------------------------------------------- #
def test_run_unknown_tool_returns_failure(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))
    r = reg.run("nope")
    assert r.ok is False
    assert "Unknown" in r.error


def test_run_unknown_kwarg_rejected(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="only_a")
    def only_a(a: str) -> str:
        return a

    reg.register(only_a)
    r = reg.run("only_a", a="x", extra="y")
    assert r.ok is False
    assert "extra" in r.error


def test_run_accepts_var_kwargs(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="anykw")
    def anykw(a: str, **rest: Any) -> dict:
        return {"a": a, "rest": rest}

    reg.register(anykw)
    r = reg.run("anykw", a="x", extra="y")
    assert r.ok is True
    assert r.output == {"a": "x", "rest": {"extra": "y"}}


def test_run_coerces_string_to_int(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    reg.register(add)
    r = reg.run("add", a="10", b="32")
    assert r.ok is True
    assert r.output == 42


def test_run_missing_required_param(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="need_a")
    def need_a(a: str) -> str:
        return a

    reg.register(need_a)
    r = reg.run("need_a")
    assert r.ok is False
    assert "required" in r.error


def test_run_defaults_are_supplied(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="def_test")
    def def_test(a: str = "hi") -> str:
        return a

    reg.register(def_test)
    r = reg.run("def_test")
    assert r.ok is True
    assert r.output == "hi"


# --------------------------------------------------------------------------- #
#  Security enforcement — the whole point of the exercise
# --------------------------------------------------------------------------- #
def test_security_denial_blocks_execution(tmp_path):
    called = {"ran": 0}

    @tool(name="noop")
    def noop() -> str:
        called["ran"] += 1
        return "ran"

    sec = FakeSecurity(decision=Decision(allowed=False, reason="nope"))
    reg = ToolRegistry(_mk_ctx(tmp_path, security=sec))
    reg.register(noop)
    r = reg.run("noop")
    assert r.ok is False
    assert "nope" in r.error
    assert called["ran"] == 0


def test_requires_confirmation_with_allows_false_blocks_execution(tmp_path):
    ran = {"count": 0}

    @tool(name="rm_stuff", dangerous=True)
    def rm_stuff() -> str:
        ran["count"] += 1
        return "boom"

    sec = FakeSecurity(
        decision=Decision(allowed=True, requires_confirmation=True),
        allows_result=False,
    )
    reg = ToolRegistry(_mk_ctx(tmp_path, security=sec))
    reg.register(rm_stuff)

    r = reg.run("rm_stuff")
    assert r.ok is False
    assert ran["count"] == 0, "side effect must not have run"
    assert sec.allows_calls, "confirmation resolver should have been consulted"


def test_requires_confirmation_with_allows_true_runs(tmp_path):
    ran = {"count": 0}

    @tool(name="confirmable", dangerous=True)
    def confirmable() -> str:
        ran["count"] += 1
        return "ok"

    sec = FakeSecurity(
        decision=Decision(allowed=True, requires_confirmation=True),
        allows_result=True,
    )
    reg = ToolRegistry(_mk_ctx(tmp_path, security=sec))
    reg.register(confirmable)

    r = reg.run("confirmable")
    assert r.ok is True
    assert ran["count"] == 1
    assert sec.allows_calls, "confirmation resolver should have been consulted"


def test_requires_confirmation_without_allows_method_refuses(tmp_path):
    ran = {"count": 0}

    @tool(name="risky", dangerous=True)
    def risky() -> str:
        ran["count"] += 1
        return "ok"

    sec = NoAllowsSecurity(Decision(allowed=True, requires_confirmation=True))
    reg = ToolRegistry(_mk_ctx(tmp_path, security=sec))
    reg.register(risky)

    r = reg.run("risky")
    assert r.ok is False
    assert ran["count"] == 0


def test_security_optional_when_missing_check_tool(tmp_path):
    class OnlyProtected:
        def is_protected(self, path: str) -> bool:
            return False

    reg = ToolRegistry(_mk_ctx(tmp_path, security=OnlyProtected()))

    @tool(name="hello")
    def hello() -> str:
        return "hi"

    reg.register(hello)
    r = reg.run("hello")
    assert r.ok is True


# --------------------------------------------------------------------------- #
#  Timeout & exceptions
# --------------------------------------------------------------------------- #
def test_tool_exception_becomes_failure(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="broken")
    def broken() -> None:
        raise RuntimeError("kaboom")

    reg.register(broken)
    r = reg.run("broken")
    assert r.ok is False
    assert "kaboom" in r.error
    assert "RuntimeError" in r.error


def test_timeout_returns_failure(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="slow")
    def slow() -> str:
        time.sleep(2.0)
        return "done"

    reg.register(slow)
    r = reg.run("slow", timeout=0.2)
    assert r.ok is False
    assert "timed out" in r.error.lower()


# --------------------------------------------------------------------------- #
#  History & bus events
# --------------------------------------------------------------------------- #
def test_history_records_calls(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    @tool(name="echo")
    def echo(s: str) -> str:
        return s

    reg.register(echo)
    reg.run("echo", s="a")
    reg.run("echo", s="b")
    assert len(reg.history) == 2
    assert reg.history[0]["name"] == "echo"
    assert reg.history[0]["ok"] is True
    assert reg.history[0]["kwargs"] == {"s": "a"}
    assert "duration" in reg.history[0]


def test_bus_events_fire(tmp_path):
    bus = EventBus()
    calls: list = []
    bus.subscribe(Events.TOOL_CALL, lambda p: calls.append(("call", p)))
    bus.subscribe(Events.TOOL_RESULT, lambda p: calls.append(("result", p)))
    reg = ToolRegistry(_mk_ctx(tmp_path, bus=bus))

    @tool(name="ping")
    def ping() -> str:
        return "pong"

    reg.register(ping)
    reg.run("ping")

    kinds = [c[0] for c in calls]
    assert kinds == ["call", "result"]
    assert calls[0][1]["name"] == "ping"
    assert calls[1][1]["ok"] is True


# --------------------------------------------------------------------------- #
#  Generated tools
# --------------------------------------------------------------------------- #
GEN_GOOD = '''\
from jarvis.core.contracts import ToolResult
from jarvis.tools.registry import FunctionTool


def _shout(text: str) -> str:
    return text.upper()


def build_tools(ctx):
    return [FunctionTool(_shout, name="shout", description="upper-case")]
'''

GEN_BROKEN = '''\
this is not valid python syntax @@@
'''


def test_load_generated_valid_and_broken(tmp_path):
    tools_dir = tmp_path / "gen"
    tools_dir.mkdir()
    (tools_dir / "good.py").write_text(GEN_GOOD, encoding="utf-8")
    (tools_dir / "bad.py").write_text(GEN_BROKEN, encoding="utf-8")

    reg = ToolRegistry(_mk_ctx(tmp_path))
    count = reg.load_generated(tools_dir)
    assert count == 1
    assert reg.has("shout")
    assert generated_module_name("good") in sys.modules
    assert generated_module_name("bad") not in sys.modules

    r = reg.run("shout", text="quiet")
    assert r.ok is True
    assert r.output == "QUIET"


def test_load_generated_empty_dir_returns_zero(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert reg.load_generated(empty) == 0


def test_load_generated_registers_replace(tmp_path):
    tools_dir = tmp_path / "gen"
    tools_dir.mkdir()
    (tools_dir / "good.py").write_text(GEN_GOOD, encoding="utf-8")

    reg = ToolRegistry(_mk_ctx(tmp_path))
    reg.load_generated(tools_dir)
    # A second load must not raise despite the same name being present.
    reg.load_generated(tools_dir)
    assert reg.has("shout")


# --------------------------------------------------------------------------- #
#  Non-Tool objects
# --------------------------------------------------------------------------- #
def test_register_rejects_non_tool(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))
    with pytest.raises(TypeError):
        reg.register(object())


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def test_safe_truncate_marks_cut_output():
    text = "a" * 500
    out = safe_truncate(text, 100)
    assert len(out) <= 100 + 60
    assert "truncated" in out
    assert safe_truncate("short", 100) == "short"


def test_run_never_raises_on_arbitrary_exception_types(tmp_path):
    reg = ToolRegistry(_mk_ctx(tmp_path))

    class WeirdError(BaseException):
        pass

    @tool(name="weird")
    def weird() -> None:
        raise Exception("plain")

    reg.register(weird)
    r = reg.run("weird")
    assert r.ok is False
