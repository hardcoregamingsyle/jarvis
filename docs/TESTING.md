# Testing

Read the section on destructive tests (§5) before you write one. It is there
because a test in this repository once destroyed part of the developer's user
profile, and the mistake it made is one any competent engineer would make.

---

## 1. Running the suite

```bash
cd C:\Users\Hp\Desktop\JARVIS      # or the repo root on Linux
python -m pytest tests -q
```

At the time of writing that is **1607 tests in ~68 seconds** on the dev machine
(Windows 11, Python 3.14.6) — a number that grows weekly, so treat the command as
the source of truth rather than the figure. Nothing is skipped by default; nothing
needs a network, a model, an audio device or a GPU.

```bash
python -m pytest tests/test_agent_loop.py -q          # one file
python -m pytest tests -q -k "wake_word"              # by name
python -m pytest tests -q -x                          # stop on first failure
python -m pytest tests -q --lf                        # last failures only
python -m pytest tests -q -m "not slow"               # skip slow tests
python -m pytest tests --collect-only -q              # what exists, without running
```

`pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "-q --strict-markers"`.
`--strict-markers` means an unregistered `@pytest.mark.foo` is an **error**, not a
silent no-op. Three markers are registered: `slow`, `requires_audio`,
`requires_linux`.

Adding a dependency? Run the suite with and without it installed. Half the point of
the design is that it works both ways.

---

## 2. Hermetic, and why that matters

Every test runs against a throwaway home. `tests/conftest.py:30`:

```python
@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "jarvis_data"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JARVIS_HOME", str(home))
    monkeypatch.setenv("JARVIS_CONFIG_DIR", str(home / "config"))
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)   # no stray real config
    yield home
```

It is `autouse=True` and session-wide by construction: **you cannot opt out and you
do not have to remember it.** Every path helper in `jarvis/core/platform_utils.py`
consults `JARVIS_HOME` / `JARVIS_CONFIG_DIR` first, so `Config.home()`,
`db_file()`, `tools_dir()`, `voices_dir()`, `models_dir()` and `logs_dir()` all
land inside `tmp_path`. Deleting `JARVIS_CONFIG` matters too — without it, a
developer with a real `config.yaml` would get different test results from CI.

The fixture is deliberately **not** called `jarvis_home`: several test modules
define their own fixture at that name, and two fixtures racing to create the same
directory is a pointless source of flakes.

What "hermetic" buys, concretely:

| Constraint | How it is achieved | Why it matters |
|---|---|---|
| No network | No test performs real HTTP. HTTP backends are tested against fakes; `PiperTTS.ensure_voice(download=False)` never touches the network by default (`speech/tts.py:357`) | The suite runs on a plane, in CI, behind a firewall, and cannot be broken by someone else's outage |
| No models | Every LLM test uses `ScriptedLLM` or `StubBackend` | 20 GB of weights is not a test dependency |
| No audio device | Recorders and players are monkeypatched; `requires_audio` marks anything that would need one | A headless Linux box has no microphone |
| No GPU | Nothing imports `torch` at module level, and the hygiene test proves it | The production target has no CUDA |
| No real filesystem outside `tmp_path` | `isolated_home` plus the destructive-test rule in §5 | Because it went wrong once |

---

## 3. The fixtures

Defined in `tests/conftest.py`.

| Fixture | Type | What it gives you |
|---|---|---|
| `isolated_home` | autouse, `Path` | The temporary data directory; also usable as a value |
| `config` | `Config` | Defaults loaded with `use_env=False`, `data_dir` pinned to `isolated_home`, `tts.enabled = False` |
| `security` | `SecurityGate` | Over `config.security` (the shipped **open** defaults), auditing into the temp logs dir |
| `bus` | `EventBus` | A fresh bus — never the process-wide `get_bus()` |
| `scripted_llm` | **class**, not an instance | `ScriptedLLM`; you instantiate it |
| `fake_registry` | **class**, not an instance | `FakeRegistry`; you instantiate it |

The last two being classes is deliberate: almost every test needs a *differently
configured* one, so the fixture hands you the constructor.

### `ScriptedLLM`

A real `LLMBackend` whose replies come from a list.

```python
def test_tool_call_then_answer(scripted_llm, fake_registry):
    from jarvis.agent.protocol import format_tool_call
    from jarvis.agent.subagent import run_agent_loop
    from jarvis.core.contracts import Message, ToolResult

    registry = fake_registry({"read_file": ToolResult.success("file contents here")})
    llm = scripted_llm([
        format_tool_call("read_file", {"path": "a.txt"}),   # reply 1: a tool call
        "The file says hello, Sir.",                        # reply 2: the answer
    ])

    messages = [Message.system("s"), Message.user("read a.txt")]
    turn = run_agent_loop(llm, registry, messages, max_iterations=5)

    assert turn.text == "The file says hello, Sir."
    assert turn.used_tools == ["read_file"]
    assert registry.calls == [("read_file", {"path": "a.txt"})]
```

Properties worth knowing:

* **The script is a queue.** Each `generate()` pops the front. Once exhausted it
  returns `default` (`"Understood, Sir."`) forever — so a test that under-specifies
  its script fails on a clear assertion instead of an `IndexError`.
* `llm.calls` is a list of the message lists it was handed; `llm.configs` the
  `GenerationConfig` for each call.
* `llm.last_prompt` joins the most recent prompt's message contents into one
  string, which is how you assert on prompt *content*:
  ```python
  agent.chat("what is my name?")
  assert "Relevant recollections" in llm.last_prompt
  ```
* Use `format_tool_call(name, args)` from `jarvis/agent/protocol.py` to script a
  tool call. Never hand-write the `<tool_call>` block — if the wire format changes,
  `format_tool_call` changes with it and your test follows.
* You can script a *live* script: `booted.llm.script = [...]` mid-test, because
  `script` is a plain attribute.

### `FakeRegistry`

A `ToolRegistry` stand-in with the four methods the agent loop actually uses —
`names()`, `has()`, `describe()`, `run()` — plus `register_function()` so an
`Orchestrator` can attach its meta-tools.

```python
registry = fake_registry({
    "read_file": ToolResult.success("contents"),   # fixed result
    "broken":    ToolResult.failure("boom"),       # fixed failure
    "echo":      lambda **kw: ToolResult.success(kw),   # callable: gets the kwargs
})
...
assert registry.calls == [("read_file", {"path": "a.txt"})]
```

Use `FakeRegistry` for agent-loop and orchestrator tests. Use a **real**
`ToolRegistry` when the thing under test is validation, coercion, security
consultation or timeouts — those live in `ToolRegistry.run()` and a fake does not
have them. `tests/test_registry.py` and `tests/test_file_tools.py` build real ones.

### The end-to-end fixture

`tests/test_integration.py:26 booted` is the pattern for whole-system tests: run the
**real** `app.build()` so the wiring under test is the production wiring, then swap
the stub LLM for a scripted one.

```python
config.llm.backend = "stub"
config.tts.engine = "null"
config.stt.engine = "stub"
subsystems = app_module.build(config, configure_logging=False)

llm = scripted_llm()
subsystems.llm = llm
subsystems.orchestrator.llm = llm
subsystems.registry.ctx.extra["llm"] = llm      # tool_maker reads it from here
yield subsystems
app_module.shutdown(subsystems)
```

`configure_logging=False` matters — otherwise every test reconfigures the root
logger and the output becomes unreadable.

---

## 4. The import-hygiene test

`tests/test_import_hygiene.py`. Three tests, and it will fail you.

### `test_no_heavy_module_level_imports`

Parametrised over every `.py` under `jarvis/`. It **AST-parses** the file (never
imports it) and collects top-level import names, then intersects with
`FORBIDDEN_AT_MODULE_LEVEL`:

* `HEAVY` — ~30 large, slow, compiled or hardware-dependent third-party packages:
  `torch`, `numpy`, `transformers`, `airllm`, `sentence_transformers`,
  `faster_whisper`, `sounddevice`, `psutil`, `requests`, `yaml`, `mss`, `PIL`,
  `keyboard`, `win32api`, …
* `WINDOWS_ONLY_STDLIB` — `winreg`, `winsound`, `msvcrt`, `_winapi`. These cannot be
  blocked at runtime (the stdlib itself pulls in `msvcrt` via `subprocess`) but must
  never appear at module level, because that breaks Linux outright.

Imports nested inside a **`try:` with an `ImportError` or `Exception` handler** are
recognised as deliberate optionals and allowed (`module_level_imports()`,
`test_import_hygiene.py:45`).

### `test_annotated_files_declare_future_annotations`

If a file uses any annotation, it must have `from __future__ import annotations`.
That is what keeps `dict[str, int]` and friends legal on Python 3.9.

### `test_the_whole_package_imports_in_a_clean_subprocess`

The one that catches what the AST check cannot. It launches a **fresh interpreter**,
installs a `sys.meta_path` hook that raises `ImportError` for every `HEAVY` package,
then `pkgutil.walk_packages` imports every module in `jarvis`. Any failure is
reported with the module name and exception.

The subprocess is the point: inside pytest, `torch` is often already in
`sys.modules` from another test, so an accidental top-level import would succeed and
the regression would ship.

### Satisfying it when you add a dependency

```python
# WRONG — breaks the boot promise and fails the hygiene test
import numpy as np

def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples))))
```

```python
# RIGHT — lazy, handled, degrades
def rms(samples):
    """Root-mean-square level of a sample buffer."""
    try:
        import numpy as np
    except ImportError:
        n = len(samples)
        if not n:
            return 0.0
        return (sum(float(s) * float(s) for s in samples) / n) ** 0.5
    return float(np.sqrt(np.mean(np.square(samples))))
```

Checklist for a new third-party dependency:

1. Import it **inside** the function that needs it, in `try/except ImportError`.
2. Either provide a pure-stdlib fallback, or return
   `ToolResult.failure("... install with `pip install X`")` — an install hint, not a
   bare traceback.
3. If the class is an engine or backend, `is_available()` must return `False`
   (never raise) when the package is absent.
4. Add it to `requirements.txt` or `requirements-full.txt` and to the right
   `[project.optional-dependencies]` group in `pyproject.toml`.
5. Add it to a `cmd_doctor` group in `jarvis/cli.py:89` so `jarvis doctor` reports
   it with a purpose and an install line.
6. If it is genuinely heavy, add its import name to `HEAVY` in
   `tests/test_import_hygiene.py` — that is not a punishment, it is how the
   subprocess test learns to block it.
7. Run the suite **without** the package installed too.

OS-specific stdlib goes behind a platform branch, never at module level:

```python
def _winreg():
    """The winreg module, or None anywhere it does not exist."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg
    except ImportError as exc:
        log.debug("winreg unavailable: %s", exc)
        return None
    return winreg
```

(`jarvis/win/autostart.py:73` — copy this shape.)

---

## 5. How to write a safe test for a destructive operation

### The incident

An earlier version of `tests/test_file_tools.py` contained a test that called
`delete_path()` on `os.path.expanduser("~")` with `recursive=True` and asserted that
the call was refused.

The test's safety depended entirely on the feature it was testing.

When the guard it relied on was relaxed, the test stopped being a test and became a
live `shutil.rmtree` of the real user profile. It emptied six directories under
`C:\Users\Hp` before aborting on a locked file. **None of it was recoverable.**

That is the whole failure mode in one sentence: *a test that proves a guard works by
firing a real weapon at a real target is only as safe as the guard, and the guard is
the thing you are not sure about.*

### The rule

> **Never name a real filesystem root, home directory, or working directory as the
> target of a delete, move or overwrite in a test. Redirect the guard's *notion* of
> those places into `tmp_path` instead.**

Written that way, a regressed guard deletes a temporary directory and the assertion
**fails**. Written the old way, it deletes your home directory and the assertion
**passes**.

### The pattern

The fixed tests are in `tests/test_file_tools.py:241-361`, under a comment block
that records the incident so nobody re-derives the old shape. Copy them.

**Redirect the guard's idea of "home":**

```python
def test_delete_refuses_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "pretend_home"
    fake_home.mkdir()
    canary = fake_home / "documents.txt"
    canary.write_text("irreplaceable", encoding="utf-8")

    # file_tools asks os.path.expanduser("~"); make it answer with our stand-in.
    monkeypatch.setattr(
        file_tools.os.path, "expanduser", lambda p: str(fake_home) if p == "~" else p
    )

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path=str(fake_home), recursive=True)

    assert r.ok is False
    assert "home" in r.error.lower()
    assert canary.exists(), "the guard let the home directory be deleted"
    assert canary.read_text(encoding="utf-8") == "irreplaceable"
```

**Redirect the guard's idea of "filesystem root" and of drive-relative paths:**

```python
def test_delete_refuses_drive_relative(tmp_path, monkeypatch):
    fake_root = tmp_path / "pretend_root"
    fake_root.mkdir()
    canary = fake_root / "keep.txt"
    canary.write_text("safe", encoding="utf-8")

    monkeypatch.setattr(file_tools, "resolve_path", lambda raw: fake_root)
    monkeypatch.setattr(file_tools, "is_filesystem_root", lambda p: Path(p) == fake_root)

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path="C:", recursive=True)

    assert r.ok is False
    assert "root" in r.error.lower()
    assert canary.exists(), "the guard let a root-shaped delete through"
```

**Use the real CWD only via `monkeypatch.chdir(tmp_path/...)`,** so "the working
directory" is a temporary one:

```python
def test_delete_refuses_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)            # the real cwd is now inside tmp_path
    marker = workdir / "marker.txt"
    marker.write_text("x", encoding="utf-8")

    r = _mk_registry(tmp_path).run("delete_path", path=str(workdir), recursive=True)
    assert r.ok is False
    assert workdir.exists() and marker.exists()
```

### The four elements

Every destructive test in this repo has all four. If yours is missing one, it is not
safe yet.

1. **A fake target inside `tmp_path`** — `pretend_home`, `pretend_root`, `work`.
2. **A redirected guard** — `monkeypatch.setattr` on `expanduser`,
   `is_filesystem_root`, `resolve_path`, or `monkeypatch.chdir`. Patch the name **as
   the module under test sees it** (`file_tools.resolve_path`, not
   `platform_utils.resolve_path`).
3. **A canary file** with distinctive content, asserted to still exist *and* still
   contain what it contained. Existence alone does not prove the tree survived.
4. **An assertion on the refusal itself** — `r.ok is False` and something specific
   in `r.error`.

### Things that are still forbidden, whatever the guard says

* `Path.home()`, `os.path.expanduser("~")`, `Path.cwd()`, `"/"`, `"C:\\"`, `"C:"`,
  `os.environ["USERPROFILE"]`, `%APPDATA%` — as a **destructive target**, ever. Read
  them if you must; never delete, move onto, or overwrite them.
* `shutil.rmtree` on anything not derived from `tmp_path`.
* Tests that write outside `tmp_path`. `isolated_home` covers JARVIS's own paths; it
  does not cover a path you construct yourself.
* Asserting a refusal by observing that nothing was destroyed *without* a canary —
  an operation that failed for an unrelated reason looks identical.

### Sanity check before you commit a destructive test

Temporarily invert the guard (make it return "allowed") and run the test. It must
fail with the canary intact in `tmp_path`. If instead something outside `tmp_path`
disappears, the test is the old shape and must not be committed.

---

## 6. No vacuous assertions

A vacuous assertion is one that cannot fail, or that would still pass if the code
under test did nothing.

```python
# VACUOUS — the tautology
assert result is not None
assert isinstance(result, ToolResult)
assert result.ok in (True, False)
assert len(names) >= 0

# VACUOUS — asserts the mock, not the code
registry = fake_registry({"read_file": ToolResult.success("x")})
assert registry.run("read_file").output == "x"      # you configured that

# VACUOUS — a smoke test wearing an assertion's clothes
turn = run_agent_loop(llm, registry, messages)
assert turn                                          # AgentTurn is always truthy

# VACUOUS — passes whether or not the guard exists
r = reg.run("delete_path", path=str(target))
assert r is not None
```

```python
# REAL — pins the value, the side effect, and the call
w = reg.run("write_file", path=str(target), content="hello world")
assert w.ok is True
assert target.read_text(encoding="utf-8") == "hello world"

r = reg.run("read_file", path=str(target))
assert r.output["content"] == "hello world"
assert r.output["bytes_read"] == len("hello world")
assert r.output["truncated"] is False

# REAL — pins the failure mode, not just "it failed"
r = reg.run("edit_file", path=str(target), old="world", new="there")
assert r.ok is False
assert "not found" in r.error

# REAL — pins the interaction
assert registry.calls == [("read_file", {"path": "a.txt"})]
assert turn.used_tools == ["read_file"]
assert messages[-1].role.value == "assistant"
```

Rules of thumb:

* Prefer `is True` / `is False` over truthiness for `ToolResult.ok` — it catches a
  function that returned a non-empty string where a bool was expected.
* Assert on the **error text**, not just `ok is False`. There are many ways to fail
  and you care about one.
* Assert on the **side effect** (file content, database row, `registry.calls`), not
  only the return value.
* When you assert a count, assert the identity too: `names == ["a.txt"]` beats
  `len(names) == 1`.
* A test with no assertion at all is a smoke test. That is a legitimate thing to
  write — it catches import errors and crashes — but mark it as such in the name
  (`test_..._does_not_raise`) so nobody mistakes it for coverage.
* `pytest.raises` needs a `match=`: `pytest.raises(ValueError, match="drive-relative")`.

---

## 7. Where the existing tests live

| File | Covers |
|---|---|
| `test_core.py` | config load/merge/env overrides, events, platform utils |
| `test_security.py`, `test_security_policy.py` | `SecurityGate` classification, and the registry actually honouring a refusal |
| `test_llm.py` | backend factory, chat formatting, thinking-block stripping, context trimming |
| `test_memory.py`, `test_context.py` | store, hybrid search, embedders, rolling summary, persistence-failure reporting |
| `test_protocol.py` | every malformed `<tool_call>` shape the parser tolerates |
| `test_agent_loop.py` | `run_agent_loop` and `SubAgent` |
| `test_task_manager.py` | spawn, progress, cancel, timeout, report queue |
| `test_registry.py` | validation, coercion, security consultation, timeouts, generated-tool loading |
| `test_file_tools.py` | file tools **and the delete guards** — read the comment block |
| `test_system_tools.py`, `test_process_tools.py`, `test_web_tools.py`, `test_input_tools.py`, `test_window_tools.py` | the tool modules |
| `test_tool_maker.py` | name sanitisation, source validation, the write/import/rollback cycle |
| `test_stt.py`, `test_tts.py`, `test_tts_format.py`, `test_audio_io.py`, `test_windows_speech.py` | the speech layer |
| `test_voice.py` | wake-word gating, barge-in, the loop's state machine |
| `test_win_integration.py` | tray, hotkeys, autostart, notifications — including their unavailable-on-Linux behaviour |
| `test_integration.py` | boot the whole system and drive a real conversation |
| `test_import_hygiene.py` | the structural rules in §4 |
