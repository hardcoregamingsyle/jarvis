"""Tool registry — discovery, dispatch, security enforcement.

Every capability the agent can invoke passes through ``ToolRegistry.run``.
That method is the single choke-point where argument validation, permission
checks, timeouts and bus events happen; concrete tools stay tiny.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from ..core.contracts import Tool, ToolParam, ToolResult, ToolSpec
from ..core.events import Events

log = logging.getLogger(__name__)


GENERATED_MODULE_PREFIX = "jarvis_generated"


def generated_module_name(stem: str) -> str:
    """Full module name a generated tool file will be imported under."""
    return f"{GENERATED_MODULE_PREFIX}.{stem}"


# --------------------------------------------------------------------------- #
#  Context passed to tool builders
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    """The shared resources a tool may reach for (config, security, bus, memory)."""

    config: Any
    security: Any
    bus: Any = None
    memory: Any = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Signature -> ToolSpec derivation
# --------------------------------------------------------------------------- #
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    tuple: "array",
    set: "array",
    bytes: "string",
    type(None): "string",
}


def _unwrap_optional(hint: Any) -> Any:
    """Return the first non-None arg of an Optional[X] / Union[X, None]."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]  # noqa: E721
        if args:
            return args[0]
    return hint


def _hint_to_json_type(hint: Any) -> str:
    hint = _unwrap_optional(hint)
    origin = typing.get_origin(hint)
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    return _PY_TO_JSON.get(hint, "string")


def _short_desc_from_docstring(doc: Optional[str]) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _spec_from_callable(
    fn: Callable[..., Any],
    *,
    name: Optional[str],
    description: Optional[str],
    dangerous: bool,
) -> "tuple[ToolSpec, bool]":
    """Build a ToolSpec by inspecting *fn*.  Returns (spec, accepts_var_kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = inspect.Signature()
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001
        hints = {}
    params: List[ToolParam] = []
    accepts_var_kwargs = False
    for pname, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if pname in ("self", "cls"):
            continue
        ptype = _hint_to_json_type(hints.get(pname, str))
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        params.append(
            ToolParam(
                name=pname,
                type=ptype,
                description="",
                required=required,
                default=default,
            )
        )
    spec = ToolSpec(
        name=name or fn.__name__,
        description=description or _short_desc_from_docstring(fn.__doc__),
        params=tuple(params),
        dangerous=dangerous,
    )
    return spec, accepts_var_kwargs


# --------------------------------------------------------------------------- #
#  FunctionTool
# --------------------------------------------------------------------------- #
class FunctionTool(Tool):
    """A :class:`Tool` that wraps a plain Python callable."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        spec: Optional[ToolSpec] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dangerous: bool = False,
    ) -> None:
        self._fn = fn
        if spec is None:
            built, accepts = _spec_from_callable(
                fn, name=name, description=description, dangerous=dangerous
            )
            self._spec = built
        else:
            _, accepts = _spec_from_callable(
                fn, name=spec.name, description=spec.description, dangerous=spec.dangerous
            )
            self._spec = spec
        self._accepts_kwargs = accepts

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, **kwargs: Any) -> ToolResult:
        result = self._fn(**kwargs)
        if isinstance(result, ToolResult):
            return result
        return ToolResult.success(output=result)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    dangerous: bool = False,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator that turns a function into a :class:`FunctionTool`."""

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(fn, name=name, description=description, dangerous=dangerous)

    return decorator


# --------------------------------------------------------------------------- #
#  Parameter coercion
# --------------------------------------------------------------------------- #
def _coerce(value: Any, ptype: str) -> Any:
    if value is None:
        return None
    if ptype == "string":
        return str(value)
    if ptype == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            return int(float(value))
        return int(value)
    if ptype == "number":
        return float(value)
    if ptype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if ptype == "array":
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]
    if ptype == "object":
        if isinstance(value, dict):
            return dict(value)
        raise TypeError("expected an object/dict")
    return value


def safe_truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters with a visible marker when cut."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - 20)
    return text[:keep] + f"\n... [truncated {len(text) - keep} chars]"


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """Discovery + dispatch for all tools available to the agent."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self._tools: dict = {}
        self._accepts_kwargs: dict = {}
        self.history: list = []
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------- #
    def register(self, tool_obj: Tool, *, replace: bool = False) -> None:
        """Add *tool_obj* to the registry."""
        if not isinstance(tool_obj, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool_obj).__name__}")
        name = tool_obj.name
        with self._lock:
            if name in self._tools and not replace:
                raise ValueError(f"Tool already registered: {name}")
            self._tools[name] = tool_obj
            self._accepts_kwargs[name] = bool(getattr(tool_obj, "_accepts_kwargs", False))

    def register_function(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dangerous: bool = False,
    ) -> FunctionTool:
        ft = FunctionTool(fn, name=name, description=description, dangerous=dangerous)
        self.register(ft, replace=True)
        return ft

    def unregister(self, name: str) -> bool:
        with self._lock:
            existed = name in self._tools
            self._tools.pop(name, None)
            self._accepts_kwargs.pop(name, None)
            return existed

    # -- inspection --------------------------------------------------------- #
    def get(self, name: str) -> Optional[Tool]:
        with self._lock:
            return self._tools.get(name)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._tools.keys())

    def list(self) -> List[ToolSpec]:
        """Return every registered tool's spec."""
        with self._lock:
            return [t.spec for t in self._tools.values()]

    def schemas(self) -> List[dict]:
        return [s.json_schema() for s in self.list()]

    def describe(self) -> str:
        """A compact human-readable catalogue suitable for a system prompt."""
        lines = []
        for spec in sorted(self.list(), key=lambda s: s.name):
            parts = []
            for p in spec.params:
                marker = "" if p.required else "?"
                parts.append(f"{p.name}{marker}: {p.type}")
            head = f"{spec.name}({', '.join(parts)})"
            if spec.dangerous:
                head += " [DANGEROUS]"
            desc = spec.description or ""
            lines.append(f"- {head}: {desc}".rstrip())
        return "\n".join(lines)

    # -- execution ---------------------------------------------------------- #
    def run(self, name, /, *, timeout=None, **kwargs):  # type: ignore[override]
        """The single execution path — validates, checks security, times out.

        ``name`` is positional-only so that tools can freely declare a parameter
        called ``name`` without colliding with the dispatch method's own arg.
        """
        tool_obj = self.get(name)
        if tool_obj is None:
            return ToolResult.failure(f"Unknown tool: {name!r}")

        spec = tool_obj.spec
        accepts_kwargs = self._accepts_kwargs.get(name, False)
        known = {p.name for p in spec.params}

        if not accepts_kwargs:
            unknown = sorted(k for k in kwargs if k not in known)
            if unknown:
                return ToolResult.failure(
                    f"{name}: unknown parameter(s): {', '.join(unknown)}"
                )

        cleaned: dict = {}
        for p in spec.params:
            if p.name in kwargs:
                try:
                    cleaned[p.name] = _coerce(kwargs[p.name], p.type)
                except (TypeError, ValueError) as exc:
                    return ToolResult.failure(f"{name}: bad value for {p.name!r}: {exc}")
                if p.enum and cleaned[p.name] not in list(p.enum):
                    return ToolResult.failure(
                        f"{name}: {p.name} must be one of {list(p.enum)}"
                    )
            elif p.required:
                return ToolResult.failure(f"{name}: missing required parameter {p.name!r}")
            else:
                cleaned[p.name] = p.default
        if accepts_kwargs:
            for k, v in kwargs.items():
                if k not in known:
                    cleaned[k] = v

        sec = getattr(self.ctx, "security", None)
        if sec is not None and hasattr(sec, "check_tool"):
            try:
                decision = sec.check_tool(spec, cleaned)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"{name}: security check errored: {exc}")
            allowed = bool(getattr(decision, "allowed", True))
            if not allowed:
                reason = getattr(decision, "reason", None) or "refused by security policy"
                return ToolResult.failure(f"{name}: {reason}")
            if bool(getattr(decision, "requires_confirmation", False)):
                if not hasattr(sec, "allows"):
                    return ToolResult.failure(
                        f"{name}: requires confirmation but no confirmation resolver is available"
                    )
                try:
                    confirmed = bool(sec.allows(decision))
                except Exception as exc:  # noqa: BLE001
                    return ToolResult.failure(
                        f"{name}: confirmation resolver errored: {exc}"
                    )
                if not confirmed:
                    return ToolResult.failure(
                        f"{name}: refused — confirmation not granted"
                    )

        bus = getattr(self.ctx, "bus", None)
        started = time.monotonic()
        if bus is not None:
            try:
                bus.emit(Events.TOOL_CALL, {"name": name, "kwargs": dict(cleaned)})
            except Exception:  # noqa: BLE001
                log.debug("bus.emit TOOL_CALL failed", exc_info=True)

        result = self._execute(tool_obj, cleaned, timeout)
        duration = time.monotonic() - started

        if not isinstance(result, ToolResult):
            result = ToolResult.success(output=result)

        entry = {
            "name": name,
            "kwargs": dict(cleaned),
            "ok": result.ok,
            "duration": duration,
            "error": result.error,
        }
        self.history.append(entry)

        if bus is not None:
            try:
                bus.emit(
                    Events.TOOL_RESULT,
                    {
                        "name": name,
                        "ok": result.ok,
                        "error": result.error,
                        "duration": duration,
                        "is_artifact": result.is_artifact,
                    },
                )
            except Exception:  # noqa: BLE001
                log.debug("bus.emit TOOL_RESULT failed", exc_info=True)

        return result

    def _execute(
        self,
        tool_obj: Tool,
        kwargs: dict,
        timeout: Optional[float],
    ) -> ToolResult:
        if timeout is None or timeout <= 0:
            try:
                return tool_obj.run(**kwargs)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"{type(exc).__name__}: {exc}")

        holder: dict = {"result": None, "exc": None}

        def worker() -> None:
            try:
                holder["result"] = tool_obj.run(**kwargs)
            except Exception as exc:  # noqa: BLE001
                holder["exc"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # We can't force-kill a thread; return a failure and let it drift.
            return ToolResult.failure(
                f"{tool_obj.name}: timed out after {timeout:.2f}s"
            )
        if holder["exc"] is not None:
            exc = holder["exc"]
            return ToolResult.failure(f"{type(exc).__name__}: {exc}")
        result = holder["result"]
        if not isinstance(result, ToolResult):
            result = ToolResult.success(output=result)
        return result

    # -- loading ------------------------------------------------------------ #
    def load_builtin(self) -> None:
        """Import every ``jarvis.tools.*`` module that exposes ``build_tools``."""
        # ORDER IS SIGNIFICANT. Registration uses replace=True, so where two
        # modules define the same tool name the later one wins.
        #
        # `app_tools` and `window_tools` both define `list_windows` and
        # `focus_window`. window_tools is listed afterwards deliberately: its
        # title matching tries exact, then case-insensitive substring, and
        # reports ambiguity rather than picking silently, and its focus_window
        # verifies with GetForegroundWindow that focus actually took — Windows
        # blocks focus stealing, so the naive call succeeds while doing nothing.
        modules = (
            "file_tools",
            "system_tools",
            "process_tools",
            "web_tools",
            "app_tools",
            "input_tools",
            "window_tools",
            "tool_maker",
        )
        for modname in modules:
            try:
                mod = importlib.import_module(f"jarvis.tools.{modname}")
            except Exception:  # noqa: BLE001
                log.warning("built-in tool module %s failed to import", modname, exc_info=True)
                continue
            builder = getattr(mod, "build_tools", None)
            if not callable(builder):
                continue
            try:
                produced = builder(self.ctx) or []
            except Exception:  # noqa: BLE001
                log.warning("build_tools() failed for %s", modname, exc_info=True)
                continue
            for t in produced:
                try:
                    self.register(t, replace=True)
                except Exception:  # noqa: BLE001
                    log.warning("could not register %r from %s", t, modname, exc_info=True)

    def load_generated(self, dir: Optional[Path] = None) -> int:
        """Import every ``*.py`` in the generated-tools directory."""
        directory = Path(dir) if dir is not None else self.ctx.config.tools_dir()
        try:
            files = sorted(Path(directory).glob("*.py"))
        except OSError:
            return 0
        count = 0
        for py in files:
            if py.name.startswith("_"):
                continue
            stem = py.stem
            mod_name = generated_module_name(stem)
            try:
                mod_spec = importlib.util.spec_from_file_location(mod_name, py)
                if mod_spec is None or mod_spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(mod_spec)
                sys.modules[mod_name] = mod  # visible during exec, so relative imports work
                mod_spec.loader.exec_module(mod)
            except Exception:  # noqa: BLE001
                log.warning("generated tool %s failed to load", py, exc_info=True)
                sys.modules.pop(mod_name, None)
                continue
            produced: List[Tool] = []
            tools_attr = getattr(mod, "TOOLS", None)
            if isinstance(tools_attr, Iterable):
                for t in tools_attr:
                    if isinstance(t, Tool):
                        produced.append(t)
            builder = getattr(mod, "build_tools", None)
            if callable(builder):
                try:
                    for t in builder(self.ctx) or []:
                        if isinstance(t, Tool):
                            produced.append(t)
                except Exception:  # noqa: BLE001
                    log.warning("build_tools() failed for %s", py, exc_info=True)
            for t in produced:
                try:
                    self.register(t, replace=True)
                    count += 1
                except Exception:  # noqa: BLE001
                    log.warning("could not register %r from %s", t, py, exc_info=True)
        return count


__all__ = [
    "ToolContext",
    "FunctionTool",
    "ToolRegistry",
    "tool",
    "safe_truncate",
    "GENERATED_MODULE_PREFIX",
    "generated_module_name",
]
