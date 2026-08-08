# Writing a JARVIS tool

This document is the specification for a JARVIS tool module. It is written to be
followed literally, by a person or by a language model. `jarvis/tools/tool_maker.py`
feeds a condensed version of it to the model at generation time; this is the full
version behind that.

If you only need the short form, jump to [the checklist](#checklist).

Related: [ARCHITECTURE.md](ARCHITECTURE.md) for how tools fit into the agent loop,
[TESTING.md](TESTING.md) for how to test one.

---

## 1. The contract

Four types, all from `jarvis.core.contracts`, plus one helper class from
`jarvis.tools.registry`.

### `ToolResult`

Every tool returns one. It is the *only* legal return value.

```python
@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: Optional[str] = None
    is_artifact: bool = False       # output is large/binary: summarise, do not speak
```

Construct it with the two static factories, never the constructor:

```python
ToolResult.success(output={"path": "/tmp/x", "bytes": 41})
ToolResult.failure("no such path: /tmp/x")
ToolResult.success(output=png_bytes, is_artifact=True)
```

`output` should be JSON-serialisable — a dict is best, because the agent sees it
rendered as text and keys give it something to reason about. `error` is a
sentence a human could act on, not a stack trace.

### `ToolParam`

One declared parameter. Field order matters if you pass positionally.

| field | type | default | meaning |
|---|---|---|---|
| `name` | `str` | — | parameter name, must match the function's |
| `type` | `str` | `"string"` | `string` \| `integer` \| `number` \| `boolean` \| `array` \| `object` |
| `description` | `str` | `""` | what the model reads when deciding what to pass |
| `required` | `bool` | `True` | if False, `default` is substituted when absent |
| `default` | `Any` | `None` | value passed when the caller omits it |
| `enum` | `Optional[Sequence]` | `None` | the registry rejects anything not in this list |

### `ToolSpec`

The schema the model sees.

```python
ToolSpec(name: str, description: str, params: Sequence[ToolParam] = (), dangerous: bool = False)
```

`spec.json_schema()` renders the OpenAI-style function schema. `dangerous=True`
is a *hint*, not a lock: in the shipped `open` security mode it changes nothing.
It only starts mattering if the owner opts into `guarded`/`readonly`. Mark
anything that deletes, kills, spends or executes.

### `Tool`

An abstract base class with two members: a `spec` property and
`run(**kwargs) -> ToolResult`. You will almost never subclass it — use
`FunctionTool`.

### `FunctionTool` — read this bit twice

```python
class FunctionTool(Tool):
    def __init__(
        self,
        fn: Callable[..., Any],          # <-- POSITIONAL-OR-KEYWORD, and it is called `fn`
        *,
        spec: Optional[ToolSpec] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dangerous: bool = False,
    ) -> None: ...
```

**The single most common failure in generated tool code is the first keyword.**
The parameter is `fn`. These are all `TypeError` at import time, and the module
is deleted:

```python
FunctionTool(func=_my_tool, name="my_tool")        # WRONG — no parameter called `func`
FunctionTool(function=_my_tool, name="my_tool")    # WRONG
FunctionTool(callable=_my_tool, name="my_tool")    # WRONG
FunctionTool(_my_tool, "my_tool", "does a thing")  # WRONG — name/description are keyword-only
```

Both of these are right:

```python
FunctionTool(fn=_my_tool, name="my_tool", description="Does a thing.")
FunctionTool(_my_tool, name="my_tool", description="Does a thing.")
```

Pass either `spec=` (full control, per-parameter descriptions) or
`name=`/`description=` (schema derived from the signature). If you pass `spec`,
`name` and `description` are ignored.

---

## 2. The module skeleton

`tool_maker.TOOL_TEMPLATE` is the exact shape the generator is told to produce.
Rendered for a tool named `weather_now`:

```python
"""Auto-generated tool: weather_now.

Report the current weather.
"""

from __future__ import annotations

from typing import Any

from jarvis.core.contracts import ToolResult
from jarvis.tools.registry import FunctionTool


def _weather_now(**kwargs: Any) -> ToolResult:
    """Report the current weather."""
    # Requirement:
    #   the requirement text passed to create_tool
    return ToolResult.success(output={"stub": True, "kwargs": kwargs})


def build_tools(ctx: Any) -> list:
    """Return the tools this module exposes."""
    return [
        FunctionTool(
            fn=_weather_now,
            name="weather_now",
            description="Report the current weather.",
        ),
    ]
```

That is the skeleton. Keep the shape; change one thing:

> **Replace `**kwargs: Any` with real named parameters.** A `**kwargs`-only
> function produces a spec with *no parameters at all*, so the model is told the
> tool takes nothing and will never pass it anything useful. The template uses
> `**kwargs` because a stub cannot know the signature yet. Your version must
> declare its arguments explicitly.

### Hard requirements on the module

| rule | why |
|---|---|
| First statement after the docstring is `from __future__ import annotations` | project-wide; keeps `Optional[...]` style annotations working on Python 3.9 |
| A module-level `def build_tools(ctx)` | the registry and `make_tool` both look for exactly this name |
| `build_tools` returns a list of `Tool` instances | `make_tool` deletes the module if the list contains none |
| Every third-party import is inside a function, in `try/except ImportError` | the package must import on bare stdlib Python |
| Every `open()` on text passes `encoding="utf-8"` | Windows defaults to cp1252 and mangles output |
| No `print()` | use `logging.getLogger(__name__)` |

`build_tools` must **not** be `async def`, must **not** be a `lambda`, and must
**not** be nested inside another function. The validator's check is an AST scan
for a module-visible `FunctionDef` named `build_tools`; `async def` and lambdas
fail it outright, and a nested one passes the scan but then fails at import with
*"generated module has no build_tools(ctx)"*.

A module may also expose a module-level `TOOLS` iterable of `Tool` instances;
`ToolRegistry.load_generated` collects those *in addition to* `build_tools(ctx)`.
Prefer `build_tools` — it gets `ctx`.

---

## 3. `build_tools(ctx)` and what `ctx` carries

`ctx` is a `jarvis.tools.registry.ToolContext`:

```python
@dataclass
class ToolContext:
    config: Any            # jarvis.core.config.Config
    security: Any          # jarvis.core.security.SecurityGate
    bus: Any = None        # jarvis.core.events.EventBus, or None
    memory: Any = None     # the MemoryStore, or None
    extra: dict = field(default_factory=dict)
```

| you want | reach for |
|---|---|
| a writable scratch directory | `ctx.config.home()`, `ctx.config.path("subdir", "file")` |
| the generated-tools directory | `ctx.config.tools_dir()` |
| downloaded models / voices / logs | `ctx.config.models_dir()`, `.voices_dir()`, `.logs_dir()` |
| any config value | `ctx.config.llm.model`, `ctx.config.agent.name`, … |
| to announce something | `ctx.bus.emit(Events.TOOL_RESULT, {...})` |
| to record a durable fact | `ctx.memory.add(MemoryRecord(...))` |
| the live LLM / registry | `ctx.extra.get("llm")`, `ctx.extra.get("registry")` |

**Everything on `ctx` may be `None` or absent.** `bus` and `memory` are `None` in
plenty of legitimate configurations, and `extra` is empty unless something put
something in it. Use `getattr(ctx, "bus", None)` and `ctx.extra.get(...)`, and
degrade rather than fail.

`build_tools` is called at import time, once, on a possibly half-built system.
Do no work in it: build closures and return them. Never raise from it — a raising
`build_tools` makes `make_tool` delete your module.

Closures are the idiomatic way to give a tool access to `ctx`:

```python
def build_tools(ctx: Any) -> List[Tool]:
    """Return the tools this module exposes."""
    home = ctx.config.home()

    def _save_note(text: str) -> ToolResult:
        """Append a line to the running notes file."""
        target = home / "notes.txt"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")
        return ToolResult.success(output={"path": str(target)})

    return [FunctionTool(_save_note, name="save_note",
                         description="Append a line to the running notes file.")]
```

---

## 4. How a spec is derived from your function

When you do **not** pass `spec=`, `registry._spec_from_callable` builds one:

- **Tool name** — `name=` if given, else the function's `__name__`.
  (`_weather_now` would become the tool name `_weather_now`, which is why the
  template always passes `name=`.)
- **Description** — `description=` if given, else **the first non-empty line of
  the docstring**. Nothing after that first line reaches the schema.
- **Parameters** — every parameter in the signature except `self`, `cls`,
  `*args` and `**kwargs`.
- **Parameter type** — from the type hint, mapped:
  `str→string`, `int→integer`, `float→number`, `bool→boolean`,
  `list`/`tuple`/`set`→`array`, `dict→object`, everything else → `string`.
  `Optional[X]` unwraps to `X`.
- **Required** — true when the parameter has no default.
- **Parameter description** — always `""`.

That last point is the reason to write an explicit `ToolSpec` for anything with
non-obvious arguments: derived specs give the model parameter *names and types*
but no prose. Compare:

```python
# derived: model sees largest_files(root: string, top: integer) with no per-arg help
FunctionTool(_largest_files, name="largest_files", description="Find big files.")

# explicit: model sees what each argument means and what values are sane
FunctionTool(_largest_files, spec=ToolSpec(
    name="largest_files",
    description="List the largest files under a directory, biggest first.",
    params=(
        ToolParam("root", "string", "Directory to scan. Absolute paths are safest.", True),
        ToolParam("top", "integer", "How many files to return, 1-200.", False, 10),
    ),
))
```

### Writing a docstring the model can act on

The catalogue the model sees is `ToolRegistry.describe()`, one line per tool:

```
- largest_files(root: string, top?: integer): List the largest files under a directory, biggest first.
```

That single line is the entire basis on which the model decides whether to call
your tool. So:

- **First line: what it does and when to use it.** Lead with the verb. Include
  the trigger — "Use to find what is consuming disk space" is worth more than a
  restatement of the name.
- **Be specific about the return.** "Returns absolute paths and sizes in bytes"
  stops the model from calling it twice to find out.
- **Do not** document arguments in the first line; put them in `ToolParam`
  descriptions, which is where they are actually rendered.
- **Do not** write "This tool …" or "A function that …". Space is scarce.
- Keep it under about 15 words. It is joined onto one line with the signature.

Bad: `"""A helper."""` → the model has no idea when to call it.
Good: `"""List the largest files under a directory, biggest first."""`

---

## 5. What the static validator rejects

`tool_maker.validate_tool_source(source) -> (ok, problems)` runs before the file
is written. Generated code is retried up to **three** times, each attempt fed the
previous attempt's problem list. Get these right and it passes first time.

| rejected | example | reason given |
|---|---|---|
| Empty source | `""` | `source is empty` |
| Any syntax error | — | `syntax error: … (line N)` |
| No module-level `build_tools` | `async def build_tools` , `build_tools = lambda ctx: []` | `missing build_tools(ctx) function` |
| `eval(...)` | | `banned call: eval()` |
| `exec(...)` | | `banned call: exec()` |
| `compile(...)` | | `banned call: compile()` |
| `__import__(...)` | | `banned call: __import__()` |
| `os.system(...)` | | `banned call: os.system()` |
| `os.popen(...)` | | `banned call: os.popen()` |
| `os.exec*(...)` | `os.execv`, `os.execve`, … | `banned call: os.execv()` |
| `shutil.rmtree(...)` | | `banned call: shutil.rmtree()` |
| `subprocess.*(..., shell=True)` | `subprocess.run("ls", shell=True)` | `subprocess called with shell=True` |
| `import ctypes` / `from ctypes import …` | | `banned import: ctypes` |
| Write-mode `open()` on an absolute system path | `open("/etc/hosts", "w")` | `write to an absolute system path` |

The system-path rule fires for a string literal starting with `/etc`, `/bin`,
`/sbin`, `/usr/bin`, `/usr/sbin`, `/boot`, `/sys`, `/proc`, `/dev`, or a Windows
drive path containing `\Windows` or `\Program Files` — **and only when the mode
string contains `w`, `a`, `x` or `+`**. Reading those paths is fine.
`open("C:\\Users\\me\\notes.txt", "w")` is fine.

Everything else is allowed, including `subprocess.run([...])` without a shell,
network calls, and writes anywhere else on the disk.

**Why these and not others.** This list is not a permission boundary — JARVIS
runs in `open` mode with no protected paths and a fully unrestricted
`run_command` tool, so nothing here withholds a capability. It is a *correctness*
filter on machine-written code: every entry is a construct that makes generated
code unreviewable (`eval`, `exec`, `__import__`, `ctypes`), unkillable
(`os.exec*` replaces the process image, taking JARVIS with it), or catastrophic
from a single hallucinated argument (`shutil.rmtree`, `shell=True` string
interpolation, clobbering `/etc`). The checks are shape-based AST matches, so
aliasing (`from subprocess import run`) slips past them. Do not do that — the
point is code you can read six months later, not a fence to climb.

If you genuinely need a shell, call the built-in `run_command` tool, which is
unrestricted and already handles timeouts and encodings.

---

## 6. Rules of the road

### Lazy imports, always

A tool module must import successfully on a Python with nothing installed —
`tests/test_import_hygiene.py` enforces this with an AST check *and* a
clean-subprocess import of the whole package.

```python
def _fetch(url: str) -> ToolResult:
    """Fetch a URL and return the response body."""
    try:
        import requests
    except ImportError:
        return ToolResult.failure("requests is not installed: pip install requests")
    ...
```

Stdlib is fine at module level, with one exception: OS-specific stdlib modules
(`winreg`, `winsound`, `msvcrt`, `fcntl`, `termios`, `pwd`) must be imported
inside an `IS_WINDOWS` / `IS_LINUX` branch, inside a function.

```python
from jarvis.core.platform_utils import IS_WINDOWS

def _read_run_key() -> ToolResult:
    """Read the current user's autostart entries."""
    if not IS_WINDOWS:
        return ToolResult.failure("Windows only")
    import winreg
    ...
```

### Return failures, never raise

`ToolRegistry.run` does catch exceptions and converts them to
`ToolResult.failure("TypeError: ...")`, but the message is a type name and the
model cannot act on it. Catch what you expect and say what went wrong:

```python
try:
    data = target.read_text(encoding="utf-8")
except FileNotFoundError:
    return ToolResult.failure(f"no such file: {target}")
except UnicodeDecodeError:
    return ToolResult.failure(f"{target} is not UTF-8 text; use read_file with binary=True")
except OSError as exc:
    return ToolResult.failure(f"could not read {target}: {exc}")
```

### Subprocesses: always a timeout

Prefer `platform_utils.run_command`, which enforces a timeout, decodes UTF-8 with
`errors="replace"`, and never raises:

```python
from jarvis.core.platform_utils import run_command

result = run_command(["git", "status", "--short"], timeout=15, cwd=repo)
if result.timed_out:
    return ToolResult.failure("git status timed out after 15s")
if not result.ok:
    return ToolResult.failure(f"git exited {result.returncode}: {result.stderr.strip()}")
return ToolResult.success(output={"status": result.stdout})
```

If you must use `subprocess` directly: pass `timeout=`, pass an argv list (never
`shell=True`), and kill *and wait for* the child on timeout —

```python
proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    out, err = proc.communicate(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.communicate()          # reap it; without this you leak a zombie
    return ToolResult.failure("command timed out after 20s")
```

### Cross-platform

Use `pathlib`, never string concatenation with `/` or `\`. Branch on
`platform_utils.IS_WINDOWS` / `IS_LINUX` / `IS_MAC`, never on `sys.platform`
directly. Locate binaries with `platform_utils.which(...)` and fail with the
install command when they are missing:

```python
if not which("xdotool"):
    return ToolResult.failure("xdotool is not installed: sudo apt install xdotool")
```

On the unsupported platform, return a failure that says so. Do not silently do
nothing.

### Resource management

Bound anything a language model can inflate. This is not a restriction on what
the owner may do — it is what stops one hallucinated argument from filling RAM
or spinning forever. The built-in `input_tools` do exactly this (`_MAX_PRESSES`,
`_MAX_TEXT_CHARS`, `_MAX_DURATION`), and so should yours:

- cap result counts and entries walked; report `"truncated": True` when you cut
- cap recursion depth explicitly; never recurse on model-supplied depth
- read files in chunks, or cap the bytes read
- give every loop a ceiling

Name the constants and comment them as resource management, so nobody later
mistakes them for a policy.

### Idempotence

The agent retries. Running your tool twice with the same arguments should be
safe and should produce the same answer. Creating something that already exists
is a success, not an error:

```python
target.mkdir(parents=True, exist_ok=True)
return ToolResult.success(output={"path": str(target), "created": not existed})
```

Report what actually changed (`"created": False`) rather than lying about having
done work.

### Types and style

Every module starts with `from __future__ import annotations`. Use
`typing.Optional[X]` / `typing.Union[...]`, never `X | None`, outside annotations.
Public functions get docstrings. Comments only where the code cannot speak.

---

## 7. Where tools live and how they load

| | |
|---|---|
| Directory | `config.tools_dir()` — `<data dir>/tools/` |
| Data dir, Windows | `%LOCALAPPDATA%\Jarvis\tools\` |
| Data dir, Linux | `~/.local/share/jarvis/tools/` |
| Override | `JARVIS_HOME=/some/path` → `/some/path/tools/` |
| Filename | `<tool_name>.py`, one module per `create_tool` call |
| Module name | `jarvis_generated.<stem>` (`GENERATED_MODULE_PREFIX` + stem) |

`ToolRegistry.load_generated(dir=None)` globs `*.py` in that directory, **skips
any file whose name starts with `_`**, imports each under
`jarvis_generated.<stem>`, collects `TOOLS` and `build_tools(ctx)`, registers
every `Tool` with `replace=True`, and returns how many it registered. A module
that fails to import is logged and skipped; it does not stop the others.

`jarvis.app.build()` calls it at startup, so anything in the directory is live on
the next run. `tool_maker.reload_generated(ctx, registry)` re-runs it without a
restart. Since registration uses `replace=True`, re-loading a changed file
overwrites the old tool — but the *module* is already in `sys.modules`, so edit
the file and restart if you want a clean reload.

Tool names are global. A generated tool named `read_file` replaces the built-in
one. Pick a name nothing else uses.

### Legal tool names (`sanitize_name`)

Rejected: anything that is not a plain Python identifier; Python keywords
(`class`, `import`) **and soft keywords** (`match`, `case`, `type`, `_`); names
starting with `_`; dunders; path characters `/ \ : * ? " < > |` and `..`; and
this reserved list —

`sys os re json io abc ast asyncio typing socket subprocess importlib pathlib builtins logging init main registry`

`weather_now` ✓  `Weather` ✓  `os` ✗  `match` ✗  `_hidden` ✗  `a-b` ✗

### The `create_tool` path, step by step

`create_tool(name, description, requirement, source=None, overwrite=False)`:

1. `sanitize_name(name)` → failure if unusable.
2. `description` required. `requirement` required when `source` is not supplied.
3. Target is `tools_dir()/<name>.py`; refuses if it exists unless `overwrite=True`.
4. With no `source`: up to 3 LLM attempts. Each is validated; failures are fed
   back into the next prompt. With a `source`: validated once, no retry.
5. Source written UTF-8, imported as `jarvis_generated.<name>`.
6. `build_tools(ctx)` is called and must return at least one `Tool`.
7. `Events.TOOL_CREATED` is emitted on the bus.

**Any failure after step 5 deletes the file and evicts the `sys.modules` entry.**
A broken tool never survives on disk. `list_custom_tools()` and
`delete_custom_tool(name)` manage what did survive.

---

## 8. A complete worked example

`disk_report.py`. This module has been run through `validate_tool_source`,
`make_tool`, `ToolRegistry.load_generated` and `registry.run(...)` — it passes
all four, and the schemas and results below are its real output.

```python
"""Auto-generated tool: disk_report.

Report the largest files under a directory, and list mounted volumes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from jarvis.core.contracts import Tool, ToolParam, ToolResult, ToolSpec
from jarvis.tools.registry import FunctionTool

# Resource management, not permission: a scan of "/" would otherwise walk the
# whole machine and hand the model a megabyte of JSON.  Both ceilings are
# generous and neither restricts *which* directory may be scanned.
_MAX_ENTRIES_SCANNED = 200_000
_MAX_RESULTS = 200


def _largest_files(
    root: str,
    top: int = 10,
    min_mb: float = 0.0,
    include_hidden: bool = False,
) -> ToolResult:
    """List the largest files under a directory, biggest first.

    Walks the tree beneath root and returns the top N files by size, each with
    its absolute path and size in bytes and megabytes. Use this to find what is
    consuming disk space. Unreadable directories are skipped and counted rather
    than aborting the scan.

    Args:
        root: Directory to scan. Absolute paths are safest.
        top: How many files to return, 1-200.
        min_mb: Ignore files smaller than this many megabytes.
        include_hidden: Include dot-files and dot-directories.
    """
    try:
        base = Path(root).expanduser()
    except (TypeError, ValueError) as exc:
        return ToolResult.failure(f"bad root {root!r}: {exc}")
    if not base.is_dir():
        return ToolResult.failure(f"not a directory: {base}")

    try:
        limit = max(1, min(int(top), _MAX_RESULTS))
    except (TypeError, ValueError):
        limit = 10
    try:
        floor_bytes = max(0.0, float(min_mb)) * 1024 * 1024
    except (TypeError, ValueError):
        floor_bytes = 0.0

    found: List[tuple] = []
    scanned = 0
    skipped = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(str(base), onerror=lambda _e: None):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            scanned += 1
            if scanned > _MAX_ENTRIES_SCANNED:
                truncated = True
                break
            full = Path(dirpath) / filename
            try:
                size = full.stat().st_size
            except OSError:
                skipped += 1
                continue
            if size >= floor_bytes:
                found.append((size, str(full)))
        if truncated:
            break

    found.sort(reverse=True)
    rows = [
        {"path": path, "bytes": size, "mb": round(size / 1048576, 2)}
        for size, path in found[:limit]
    ]
    return ToolResult.success(
        output={
            "root": str(base),
            "files": rows,
            "scanned": scanned,
            "unreadable": skipped,
            "truncated": truncated,
        }
    )


def _mounted_volumes() -> ToolResult:
    """List mounted volumes with their free and total space.

    Reports every drive or mount point the operating system exposes, with
    capacity in gigabytes. Falls back to the current filesystem alone when
    psutil is not installed.
    """
    import shutil

    partitions: List[str] = []
    source = "stdlib"
    try:
        import psutil  # noqa: F401
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        source = "psutil"
        try:
            partitions = [p.mountpoint for p in psutil.disk_partitions(all=False)]
        except Exception:  # noqa: BLE001
            partitions = []
            source = "stdlib"
    if not partitions:
        partitions = [os.path.abspath(os.sep)]

    volumes = []
    for mount in partitions:
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        volumes.append(
            {
                "mount": mount,
                "total_gb": round(usage.total / 1073741824, 1),
                "free_gb": round(usage.free / 1073741824, 1),
                "used_pct": round(100.0 * usage.used / usage.total, 1) if usage.total else 0.0,
            }
        )
    if not volumes:
        return ToolResult.failure("no readable volumes found")
    return ToolResult.success(output={"source": source, "volumes": volumes})


_LARGEST_FILES_SPEC = ToolSpec(
    name="largest_files",
    description=(
        "List the largest files under a directory, biggest first. Use to find "
        "what is consuming disk space."
    ),
    params=(
        ToolParam("root", "string", "Directory to scan. Absolute paths are safest.", True),
        ToolParam("top", "integer", "How many files to return, 1-200.", False, 10),
        ToolParam("min_mb", "number", "Ignore files smaller than this many megabytes.", False, 0.0),
        ToolParam("include_hidden", "boolean", "Include dot-files and dot-directories.", False, False),
    ),
)


def build_tools(ctx: Any) -> List[Tool]:
    """Return the tools this module exposes."""
    return [
        FunctionTool(_largest_files, spec=_LARGEST_FILES_SPEC),
        FunctionTool(
            _mounted_volumes,
            name="mounted_volumes",
            description="List mounted volumes with their free and total space.",
        ),
    ]
```

### What the model is shown

```json
{
  "name": "largest_files",
  "description": "List the largest files under a directory, biggest first. Use to find what is consuming disk space.",
  "parameters": {
    "type": "object",
    "properties": {
      "root": {"type": "string",  "description": "Directory to scan. Absolute paths are safest."},
      "top": {"type": "integer", "description": "How many files to return, 1-200."},
      "min_mb": {"type": "number", "description": "Ignore files smaller than this many megabytes."},
      "include_hidden": {"type": "boolean", "description": "Include dot-files and dot-directories."}
    },
    "required": ["root"]
  }
}
```

Note `mounted_volumes` renders with `"properties": {}` and `"required": []` —
correct, because it takes no arguments. Had it been written `**kwargs`-style, it
would render identically while silently accepting anything, which is the trap
described in §2.

### What actually happens when it runs

```
registry.run("largest_files", root="<dir>", top=2)
  -> ok  {"root": "...", "files": [{"path": ".../big.bin", "bytes": 5000, "mb": 0.0}, ...],
          "scanned": 2, "unreadable": 0, "truncated": false}

registry.run("mounted_volumes")
  -> ok  {"source": "psutil", "volumes": [ ... 4 entries ... ]}

registry.run("largest_files", root="<missing>")
  -> failed  "not a directory: <missing>"

registry.run("largest_files", root="<dir>", bogus=1)
  -> failed  "largest_files: unknown parameter(s): bogus"

registry.run("largest_files")
  -> failed  "largest_files: missing required parameter 'root'"
```

The last two come from `ToolRegistry.run` before your function is entered: it
rejects unknown parameters (unless the function declares `**kwargs`), coerces
each value to the declared type, checks `enum` membership, fills in defaults for
omitted optional parameters, and fails on a missing required one. You do not need
to re-check any of that — but do validate *semantics* (does the directory exist,
is the number sane), which the registry cannot know.

### Installing it

```python
result = registry.run(
    "create_tool",
    name="disk_report",
    description="Report the largest files under a directory, and list mounted volumes.",
    requirement="Find what is using disk space and how much room is left.",
    source=open("disk_report.py", encoding="utf-8").read(),   # omit to have the LLM write it
)
```

Or just drop the file into `<data dir>/tools/disk_report.py` and restart.

---

## Checklist

Follow this literally.

1. [ ] Module docstring, then `from __future__ import annotations`.
2. [ ] Import `ToolResult` (and `Tool`, `ToolSpec`, `ToolParam` if using an explicit
       spec) from `jarvis.core.contracts`; `FunctionTool` from `jarvis.tools.registry`.
3. [ ] Third-party imports **inside functions**, in `try/except ImportError`,
       returning `ToolResult.failure("<pkg> is not installed: pip install <pkg>")`.
4. [ ] Each tool is a module-level function with **explicit named parameters and
       type hints** — not `**kwargs`.
5. [ ] Every parameter that is not required has a default in the signature.
6. [ ] Docstring first line: verb, what it does, when to use it. Under ~15 words.
7. [ ] Return `ToolResult.success(output={...})` or `ToolResult.failure("why")`.
       Never raise. Never return a bare value.
8. [ ] Subprocess calls use `run_command(..., timeout=N)`, or pass `timeout=` and
       kill+wait the child.
9. [ ] Text `open()` calls pass `encoding="utf-8"`.
10. [ ] Cap counts, depths and sizes; report `"truncated": true` when you cut.
11. [ ] Branch on `IS_WINDOWS`/`IS_LINUX` from `jarvis.core.platform_utils`; return a
        clear failure on the platform you do not support.
12. [ ] No `eval`, `exec`, `compile`, `__import__`, `os.system`, `os.popen`,
        `os.exec*`, `shutil.rmtree`, `ctypes`, `shell=True`, or write-mode
        `open()` on `/etc`, `/bin`, `C:\Windows`, …
13. [ ] No `print()`. Use `logging.getLogger(__name__)`.
14. [ ] Module-level `def build_tools(ctx) -> list` — not async, not a lambda, not
        nested — returning `FunctionTool(...)` instances.
15. [ ] **`FunctionTool(fn=..., name=..., description=...)`** — the first parameter
        is `fn`. Not `func`. Not `function`.
16. [ ] Tool name is a plain identifier, not a keyword or soft keyword, not
        starting with `_`, not in the reserved list, and not already taken.
