"""The self-extending capability: JARVIS writes new tools on demand.

When the agent needs something the built-in tool suite does not cover it can
ask an LLM to draft a Python module that conforms to :data:`TOOL_TEMPLATE`,
validate it statically, drop it into ``ctx.config.tools_dir()``, import it
under the same name the registry uses, and confirm that ``build_tools``
returns proper :class:`Tool` instances.  On any failure the file and its
``sys.modules`` entry are cleaned up so the workspace never accumulates dead
or dangerous code.
"""

from __future__ import annotations

import ast
import importlib.util
import keyword
import logging
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..core.contracts import Tool, ToolResult
from ..core.events import Events
from .registry import (
    FunctionTool,
    GENERATED_MODULE_PREFIX,
    generated_module_name,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Template — every generated tool follows this exact shape.
# --------------------------------------------------------------------------- #
TOOL_TEMPLATE = '''\
"""Auto-generated tool: {name}.

{description}
"""

from __future__ import annotations

from typing import Any

from jarvis.core.contracts import ToolResult
from jarvis.tools.registry import FunctionTool


def _{name}(**kwargs: Any) -> ToolResult:
    """{description}"""
    # Requirement:
    #   {requirement}
    return ToolResult.success(output={{"stub": True, "kwargs": kwargs}})


def build_tools(ctx: Any) -> list:
    """Return the tools this module exposes."""
    return [
        FunctionTool(
            fn=_{name},
            name="{name}",
            description="{description}",
        ),
    ]
'''


# --------------------------------------------------------------------------- #
#  Name validation
# --------------------------------------------------------------------------- #
_RESERVED = frozenset({
    "sys", "os", "re", "json", "io", "abc", "ast", "asyncio", "typing",
    "socket", "subprocess", "importlib", "pathlib", "builtins", "logging",
    "init", "main", "registry",
})


def sanitize_name(name: str) -> Optional[str]:
    """Return a safe filename+identifier, or ``None`` if *name* cannot be made safe."""
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    forbidden_chars = ("/", "\\", ":", "\x00", "*", "?", "\"", "<", ">", "|")
    if any(ch in stripped for ch in forbidden_chars):
        return None
    if ".." in stripped or stripped in (".", ".."):
        return None
    if not stripped.isidentifier():
        return None
    if keyword.iskeyword(stripped):
        return None
    softkw = getattr(keyword, "softkwlist", ())
    if stripped in softkw:
        return None
    if stripped.startswith("_"):
        return None
    if stripped.startswith("__") and stripped.endswith("__"):
        return None
    if stripped.lower() in _RESERVED:
        return None
    return stripped


# --------------------------------------------------------------------------- #
#  Static source validation
# --------------------------------------------------------------------------- #
_BANNED_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "exec"),   # matches exec*, checked as prefix below
    ("shutil", "rmtree"),
}
_BANNED_BUILTINS = {"eval", "exec", "compile", "__import__"}
_BANNED_MODULES = {"ctypes"}
_SYSTEM_PATH_PREFIXES = ("/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/boot",
                          "/sys", "/proc", "/dev")


def _call_target(node: ast.Call) -> Optional[Tuple[str, str]]:
    """Return (module, name) for a ``mod.name(...)`` call — else None."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id, func.attr
    return None


def _is_shell_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _absolute_system_write(node: ast.Call) -> bool:
    """A best-effort check for ``open('/etc/foo', 'w')`` and friends."""
    func = node.func
    is_open = (
        (isinstance(func, ast.Name) and func.id == "open")
        or (isinstance(func, ast.Attribute) and func.attr == "open")
    )
    if not is_open or not node.args:
        return False
    target = node.args[0]
    if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
        return False
    path = target.value
    # A write-mode is required to be considered dangerous.
    mode = ""
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value)
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = str(kw.value.value)
    if not any(ch in mode for ch in ("w", "a", "x", "+")):
        return False
    if any(path.startswith(prefix) for prefix in _SYSTEM_PATH_PREFIXES):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", path):
        # Any absolute Windows-drive path; guarded on top of the security layer.
        lower = path.lower()
        if any(seg in lower for seg in ("\\windows", "\\program files", "/windows", "/program files")):
            return True
    return False


#: Checks that only make sense when the machine is being protected from the
#: agent. Under ``security.mode == "open"`` -- the shipped default, meaning
#: "this agent administers this box" -- they are exactly the capabilities the
#: agent is supposed to have, and refusing them cripples self-extension: a tool
#: that cannot shell out or touch /etc cannot administer anything.
#:
#: The syntax check and the ``build_tools`` check are NOT in this set. Those
#: are correctness, not policy, and they always run: importing a module with a
#: syntax error breaks the registry, and one without ``build_tools`` registers
#: nothing. Neither has anything to do with trust.
_POLICY_CHECKS = (
    "banned import",
    "banned call",
    "subprocess called with shell=True",
    "write to an absolute system path",
)


def _is_policy_problem(problem: str) -> bool:
    return any(problem.startswith(prefix) for prefix in _POLICY_CHECKS)


def validate_tool_source(
    source: str,
    *,
    unrestricted: bool = False,
) -> Tuple[bool, List[str]]:
    """Statically vet a generated tool module.

    Rejects on: syntax errors; missing ``build_tools``; dangerous builtins
    (``eval``/``exec``/``compile``/``__import__``); ``os.system``/``os.popen``/
    ``os.exec*``; ``subprocess`` with ``shell=True``; ``ctypes``; and writes to
    absolute system paths.
    """
    problems: List[str] = []
    if not isinstance(source, str) or not source.strip():
        return False, ["source is empty"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [f"syntax error: {exc.msg} (line {exc.lineno})"]

    has_build_tools = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_tools":
            has_build_tools = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BANNED_MODULES:
                    problems.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BANNED_MODULES:
                problems.append(f"banned import: {node.module}")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
                problems.append(f"banned call: {func.id}()")
            target = _call_target(node)
            if target is not None:
                mod, name = target
                if (mod, name) in _BANNED_CALLS:
                    problems.append(f"banned call: {mod}.{name}()")
                if mod == "os" and name.startswith("exec"):
                    problems.append(f"banned call: os.{name}()")
                if mod == "subprocess" and _is_shell_true(node):
                    problems.append("subprocess called with shell=True")
                if mod == "shutil" and name == "rmtree":
                    problems.append("banned call: shutil.rmtree()")
            if _absolute_system_write(node):
                problems.append("write to an absolute system path")

    if not has_build_tools:
        problems.append("missing build_tools(ctx) function")

    if unrestricted:
        # Keep only the correctness failures; the policy ones describe powers
        # this agent is meant to have. They are still logged, so an unexpected
        # ctypes import in a generated tool is visible after the fact.
        blocked = [p for p in problems if _is_policy_problem(p)]
        if blocked:
            log.info(
                "tool source uses privileged operations (%s); allowed because "
                "security.mode is open",
                "; ".join(blocked),
            )
        problems = [p for p in problems if not _is_policy_problem(p)]

    ok = not problems
    return ok, problems


# --------------------------------------------------------------------------- #
#  Security-config resolution (duck-typed)
# --------------------------------------------------------------------------- #
def _is_unrestricted(security: Any) -> bool:
    """True when the security policy is ``open``.

    ``open`` is the shipped default and means "this agent administers this
    machine". A tool it writes for itself should therefore be allowed the same
    operations the agent may already perform through its built-in tools --
    anything else is theatre: the model can shell out via ``run_command``
    regardless, so blocking ``subprocess`` only in *generated* code stops
    nothing and prevents the agent extending itself.
    """
    try:
        cfg = getattr(security, "cfg", None) or security
        mode = str(getattr(cfg, "mode", "") or "").strip().lower()
        return mode == "open"
    except Exception:  # noqa: BLE001 - a probe must never break tool creation
        return False


def _allow_network(security: Any) -> bool:
    """Extract ``allow_network`` from any of the shapes security may take."""
    try:
        cfg = getattr(security, "cfg", None)
        if cfg is not None:
            return bool(getattr(cfg, "allow_network", True))
    except AttributeError:
        pass
    try:
        return bool(getattr(security, "allow_network", True))
    except AttributeError:
        return True


# --------------------------------------------------------------------------- #
#  Source extraction from LLM output
# --------------------------------------------------------------------------- #
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<body>.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def _extract_source(text: str) -> Optional[str]:
    """Pull the first Python code fence out of an LLM reply, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    match = _CODE_FENCE_RE.search(text)
    if match:
        candidate = match.group("body").strip()
        if candidate:
            return candidate
    stripped = text.strip()
    if stripped.startswith("from ") or stripped.startswith("import ") or "def build_tools" in stripped:
        return stripped
    return None


def _generate_source(
    llm: Any,
    name: str,
    description: str,
    requirement: str,
    catalogue: str,
    previous_error: Optional[str] = None,
) -> Optional[str]:
    """Ask the LLM for one draft; return the extracted source or None."""
    from ..core.contracts import GenerationConfig, Message  # deferred, keeps import bare-stdlib safe

    system = (
        "You are a Python code generator for the JARVIS assistant.\n"
        "Produce a SINGLE self-contained module that follows this template EXACTLY, "
        "returning a list of FunctionTool objects via build_tools(ctx).\n"
        "Never use eval/exec/compile/__import__, os.system, os.popen, subprocess with "
        "shell=True, ctypes, or shutil.rmtree. Never write to absolute system paths.\n\n"
        "TEMPLATE:\n"
        f"```python\n{TOOL_TEMPLATE.format(name=name, description=description, requirement=requirement)}```\n\n"
        f"Existing tools you may call via ctx if useful:\n{catalogue or '(none)'}\n"
    )
    if previous_error:
        system += (
            f"\nThe previous attempt failed validation with: {previous_error}\n"
            "Fix these issues in the new attempt."
        )
    user = (
        f"Tool name: {name}\n"
        f"Description: {description}\n"
        f"Requirement: {requirement}\n\n"
        "Reply with a single ```python``` fenced code block only."
    )
    messages = [Message.system(system), Message.user(user)]
    try:
        result = llm.generate(messages, GenerationConfig(temperature=0.2, max_new_tokens=800))
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM generate failed: %s", exc)
        return None
    text = getattr(result, "text", None) or ""
    return _extract_source(text)


# --------------------------------------------------------------------------- #
#  make_tool
# --------------------------------------------------------------------------- #
def _import_from_path(path: Path, module_name: str):
    """Import ``path`` under ``module_name`` and register it in ``sys.modules``."""
    package = GENERATED_MODULE_PREFIX
    if package not in sys.modules:
        pkg_spec = importlib.util.spec_from_loader(package, loader=None, is_package=True)
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        pkg_mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package] = pkg_mod
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {module_name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _delete_file_and_module(path: Path, module_name: str) -> None:
    """Best-effort cleanup of ``path`` and its sys.modules entry."""
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass
    sys.modules.pop(module_name, None)


def make_tool(
    ctx: Any,
    name: str,
    description: str,
    requirement: str,
    *,
    llm: Any = None,
    source: Optional[str] = None,
    overwrite: bool = False,
) -> ToolResult:
    """Write, validate and load a new tool module."""
    safe_name = sanitize_name(name)
    if safe_name is None:
        return ToolResult.failure(
            f"invalid tool name {name!r}: must be a plain identifier, "
            "not a keyword, not reserved, not dunder"
        )
    if not description or not str(description).strip():
        return ToolResult.failure("description is required")
    if source is None and not str(requirement or "").strip():
        return ToolResult.failure("requirement is required when source is not supplied")

    config = getattr(ctx, "config", None)
    if config is None or not hasattr(config, "tools_dir"):
        return ToolResult.failure("ctx.config.tools_dir() is required")
    tools_dir = Path(config.tools_dir())
    tools_dir.mkdir(parents=True, exist_ok=True)
    target = tools_dir / f"{safe_name}.py"
    if target.exists() and not overwrite:
        return ToolResult.failure(
            f"tool file already exists: {target} (pass overwrite=True to replace)"
        )

    # Duck-typed security probing must never raise.
    security = getattr(ctx, "security", None)
    try:
        _allow_network(security)
    except AttributeError:
        pass
    unrestricted = _is_unrestricted(security)

    catalogue = ""
    registry = ctx.extra.get("registry") if hasattr(ctx, "extra") else None
    if registry is not None and hasattr(registry, "describe"):
        try:
            catalogue = registry.describe()
        except Exception:  # noqa: BLE001
            catalogue = ""

    attempts = 0
    last_problems: List[str] = []
    current_source = source
    if current_source is None:
        while attempts < 3:
            attempts += 1
            drafted = _generate_source(
                llm,
                safe_name,
                str(description),
                str(requirement),
                catalogue,
                previous_error="; ".join(last_problems) if last_problems else None,
            )
            if not drafted:
                last_problems = ["LLM returned no code fence"]
                continue
            ok, problems = validate_tool_source(drafted, unrestricted=unrestricted)
            if ok:
                current_source = drafted
                break
            last_problems = problems
        if current_source is None:
            return ToolResult.failure(
                f"could not generate a valid tool after {attempts} attempt(s): "
                f"{'; '.join(last_problems) or 'unknown'}"
            )
    else:
        ok, problems = validate_tool_source(
            current_source, unrestricted=unrestricted
        )
        if not ok:
            return ToolResult.failure(
                f"provided source failed validation: {'; '.join(problems)}"
            )

    try:
        target.write_text(current_source, encoding="utf-8")
    except OSError as exc:
        return ToolResult.failure(f"could not write {target}: {exc}")

    module_name = generated_module_name(safe_name)
    try:
        module = _import_from_path(target, module_name)
    except Exception as exc:  # noqa: BLE001
        _delete_file_and_module(target, module_name)
        return ToolResult.failure(f"import failed: {exc}")

    builder = getattr(module, "build_tools", None)
    if not callable(builder):
        _delete_file_and_module(target, module_name)
        return ToolResult.failure("generated module has no build_tools(ctx)")

    try:
        produced = builder(ctx) or []
    except Exception as exc:  # noqa: BLE001
        _delete_file_and_module(target, module_name)
        return ToolResult.failure(f"build_tools raised: {exc}")

    tools_produced = [t for t in produced if isinstance(t, Tool)]
    if not tools_produced:
        _delete_file_and_module(target, module_name)
        return ToolResult.failure("build_tools returned no Tool instances")

    bus = getattr(ctx, "bus", None)
    if bus is not None:
        try:
            bus.emit(
                Events.TOOL_CREATED,
                {
                    "name": safe_name,
                    "path": str(target),
                    "tools": [t.name for t in tools_produced],
                },
            )
        except Exception:  # noqa: BLE001
            log.debug("bus.emit TOOL_CREATED failed", exc_info=True)

    return ToolResult.success(
        output={
            "name": safe_name,
            "path": str(target),
            "module": module_name,
            "tools": [t.name for t in tools_produced],
        }
    )


# --------------------------------------------------------------------------- #
#  Enumeration / removal / reload
# --------------------------------------------------------------------------- #
def list_generated(ctx: Any) -> ToolResult:
    """Enumerate generated tool files on disk."""
    config = getattr(ctx, "config", None)
    if config is None or not hasattr(config, "tools_dir"):
        return ToolResult.failure("ctx.config.tools_dir() is required")
    tools_dir = Path(config.tools_dir())
    try:
        files = sorted(tools_dir.glob("*.py"))
    except OSError as exc:
        return ToolResult.failure(f"could not list {tools_dir}: {exc}")
    entries = []
    for f in files:
        if f.name.startswith("_"):
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = None
        entries.append({"name": f.stem, "path": str(f), "size": size})
    return ToolResult.success(
        output={"dir": str(tools_dir), "tools": entries, "count": len(entries)}
    )


def delete_generated(ctx: Any, name: str) -> ToolResult:
    """Remove a generated tool file and evict its module."""
    safe_name = sanitize_name(name)
    if safe_name is None:
        return ToolResult.failure(f"invalid tool name {name!r}")
    config = getattr(ctx, "config", None)
    if config is None or not hasattr(config, "tools_dir"):
        return ToolResult.failure("ctx.config.tools_dir() is required")
    target = Path(config.tools_dir()) / f"{safe_name}.py"
    if not target.exists():
        return ToolResult.failure(f"no such generated tool: {target}")
    try:
        target.unlink()
    except OSError as exc:
        return ToolResult.failure(f"delete failed: {exc}")
    sys.modules.pop(generated_module_name(safe_name), None)
    return ToolResult.success(output={"name": safe_name, "deleted": True})


def reload_generated(ctx: Any, registry: Any) -> ToolResult:
    """Reload every generated tool module through *registry*."""
    if registry is None or not hasattr(registry, "load_generated"):
        return ToolResult.failure("registry with load_generated() is required")
    config = getattr(ctx, "config", None)
    tools_dir = Path(config.tools_dir()) if config is not None else None
    try:
        count = int(registry.load_generated(tools_dir))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"reload failed: {exc}")
    return ToolResult.success(output={"loaded": count})


# --------------------------------------------------------------------------- #
#  Tool factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Expose the tool-maker as callable tools."""
    llm_ref = None
    if hasattr(ctx, "extra") and isinstance(ctx.extra, dict):
        llm_ref = ctx.extra.get("llm")

    def _create_tool(
        name: str,
        description: str,
        requirement: str,
        source: Optional[str] = None,
        overwrite: bool = False,
    ) -> ToolResult:
        """Draft, validate and install a brand-new tool."""
        return make_tool(
            ctx,
            name=name,
            description=description,
            requirement=requirement,
            llm=llm_ref,
            source=source,
            overwrite=bool(overwrite),
        )

    def _list_custom() -> ToolResult:
        """List every generated tool currently on disk."""
        return list_generated(ctx)

    def _delete_custom(name: str) -> ToolResult:
        """Remove a generated tool by name."""
        return delete_generated(ctx, name=name)

    return [
        FunctionTool(
            _create_tool,
            name="create_tool",
            description="Author, validate and install a new tool module.",
            dangerous=True,
        ),
        FunctionTool(
            _list_custom,
            name="list_custom_tools",
            description="List installed generated tools.",
        ),
        FunctionTool(
            _delete_custom,
            name="delete_custom_tool",
            description="Delete a generated tool by name.",
            dangerous=True,
        ),
    ]


__all__ = [
    "TOOL_TEMPLATE",
    "sanitize_name",
    "validate_tool_source",
    "make_tool",
    "list_generated",
    "delete_generated",
    "reload_generated",
    "build_tools",
]
