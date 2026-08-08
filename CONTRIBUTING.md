# Contributing to JARVIS

This is a personal assistant that runs with full control of the machine it is
installed on. That shapes almost every rule below: the code has to be readable by
someone deciding whether to trust it, it has to import on a machine with nothing
installed, and it has to work identically on Windows and Linux.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before your first change, and
[docs/TESTING.md](docs/TESTING.md) before your first test. If you are adding a
tool, [docs/TOOL_AUTHORING.md](docs/TOOL_AUTHORING.md) is the specification.

---

## Environment

Python 3.9 or newer. Development happens on Windows 11 / Python 3.14; the
production target is Linux on a CPU-only i5 laptop. Both must work.

```bash
git clone https://github.com/hardcoregamingsyle/jarvis
cd jarvis

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

pip install -e ".[dev]"
```

`pip install -e ".[dev]"` gives you pytest and ruff and nothing else — which is
deliberate. **The package must import and the suite must pass with zero optional
dependencies installed.** Install extras (`.[speech]`, `.[control]`, `.[memory]`,
`.[config]`) only when you are working on those subsystems, and run the suite at
least once without them before you push.

`jarvis doctor` reports what is present and what each missing package unlocks.

---

## Running the tests

```bash
python -m pytest tests -q                       # everything — 1,602 tests, ~70s
python -m pytest tests/test_registry.py -q      # one file
python -m pytest tests -q -k "delete"           # one topic
python -m pytest tests -q -x --lf               # stop at the first failure, rerun last failures
```

Every test runs against a throwaway `JARVIS_HOME` — `tests/conftest.py` sets it
via an autouse `isolated_home` fixture, so nothing can touch your real data
directory, memory database or generated tools. Fixtures available to you:
`isolated_home`, `config`, `security`, `bus`, `scripted_llm`, `fake_registry`.

The suite must stay hermetic:

| never | instead |
|---|---|
| Network calls | monkeypatch `urllib.request.urlopen` / the backend's `_get`/`_post` |
| A real model | the `scripted_llm` fixture |
| An audio device | monkeypatch `sounddevice`; the `requires_audio` marker exists but is not run in CI |
| GPU | nothing in the suite may need CUDA |
| Your real filesystem | `tmp_path` |
| `time.sleep` for synchronisation | events, or monkeypatch the clock |

`tests/test_import_hygiene.py` is the one that will catch you. It runs an AST
check over every module *and* imports the whole package in a clean subprocess
with the optional dependencies blocked. If you put `import numpy` at module
level, it fails.

---

## The hard rules

### 1. Every third-party import is lazy

Inside the function, in `try/except ImportError`, returning a useful failure.

```python
def transcribe(self, audio, sample_rate: int = 16000) -> Transcript:
    """Transcribe a mono float32 buffer."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return Transcript(text="")
```

OS-specific *stdlib* modules — `winreg`, `winsound`, `msvcrt`, `fcntl`,
`termios`, `pwd` — go inside an `IS_WINDOWS` / `IS_LINUX` branch, inside a
function. Import them at module level and the package stops importing on the
other platform.

Every backend, engine and integration exposes `is_available()` that returns
`False` when its dependencies are absent, and **never raises**. A probe that
throws breaks `jarvis doctor`, which is the one command people run when
everything else is broken.

### 2. No safety rails

The owner has explicitly removed them. `security.mode` defaults to `"open"`,
`protected_paths` is `[]`, `dangerous_patterns` is `[]`. Do not add confirmation
prompts, allowlists, blocklists, or "are you sure?" paths. Do not quietly refuse
something because it looks dangerous. If a mode called `open` secretly refused
things, the setting would be a lie.

The policy *engine* stays — `guarded` and `readonly` exist for anyone who wants
them, and your code should keep routing through `ctx.security` so those modes
keep working. But the shipped default permits everything, and new code must not
change that.

**Resource limits are different and are wanted.** Caps that prevent unbounded
recursion, unbounded memory, unbounded output or a runaway respawn loop are good
engineering, not permission. Frame and comment them that way:

```python
# Resource management, not permission: a model will happily ask for 99999 of
# anything, and an unbounded repeat pins a core until the process is killed.
_MAX_PRESSES = 50
```

Anything that stops the owner doing something they asked for is a rail. Anything
that stops one hallucinated integer from taking the machine down is a limit.

### 3. Cross-platform, always

- `pathlib`, never string paths joined with `/` or `\`.
- Branch on `IS_WINDOWS` / `IS_LINUX` / `IS_MAC` from
  `jarvis.core.platform_utils` — never on `sys.platform` directly. That module is
  the single place allowed to know which OS this is.
- Every `subprocess` call passes a `timeout`, and kills **and waits for** the
  child on timeout. `platform_utils.run_command` already does both; prefer it.
- Every text `open()` passes `encoding="utf-8"`. Windows defaults to cp1252.
- Locate binaries with `platform_utils.which(...)`; on the platform you do not
  support, return a failure that says so rather than silently doing nothing.

### 4. Types and style

- Every module starts with `from __future__ import annotations`.
- `typing.Optional[X]` / `typing.Union[X, Y]`, never `X | None`, outside
  annotations — the target is Python 3.9.
- Public classes and functions get docstrings. Comments only where the code
  cannot speak for itself.
- No `print()` in library code. `logging.getLogger(__name__)`.
- Public methods return the contract's failure type (`ToolResult.failure(...)`,
  an empty `Transcript`, `None`) rather than raising.
- `ruff check .` before you push. Line length 100, target py39.

### 5. No vacuous assertions

A test that cannot fail is worse than no test: it costs the same to run and buys
a false sense of coverage.

```python
# vacuous — passes whatever the code does
assert result is not None
assert isinstance(result, ToolResult)
assert len(names) >= 0

# real — pins the behaviour
assert result.ok is False
assert "missing required parameter 'root'" in result.error
assert names == ["largest_files", "mounted_volumes"]
```

Assert on values, not on types. Assert the error *message* when you assert a
failure. If the assertion would still pass with the feature deleted, it is not
testing the feature.

---

## The destructive-test rule

**Never write a test that names a real filesystem root, home directory, or
working directory as the target of a delete, move or overwrite. Not even to
assert that it is refused.**

This is not caution in the abstract. An earlier version of `tests/test_file_tools.py`
called `delete_path()` on `os.path.expanduser("~")` with `recursive=True` and
asserted the call was refused — a test whose safety depended entirely on the
feature it was testing. When that guard was relaxed, the test stopped being a
test and became a live `shutil.rmtree` of the real user profile. It emptied six
directories under `C:\Users\Hp` before aborting on a locked file, and none of it
was recoverable.

Written the safe way, a regressed guard deletes a temporary directory and the
assertion **fails**. Written the old way, it deletes your home directory and the
assertion **passes**.

The technique is to redirect the guard's *notion* of those places into
`tmp_path`, so the real ones are never named:

```python
def test_delete_refuses_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "pretend_home"
    fake_home.mkdir()
    canary = fake_home / "documents.txt"
    canary.write_text("irreplaceable", encoding="utf-8")

    monkeypatch.setattr(
        file_tools.os.path, "expanduser", lambda p: str(fake_home) if p == "~" else p
    )

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path=str(fake_home), recursive=True)

    assert r.ok is False
    assert canary.exists(), "the guard let a home-directory delete through"
```

The canary file is the point: it turns "the guard returned a failure" into "the
data is still there".

See the comment block above the whole-tree refusal tests in
`tests/test_file_tools.py`, and
[docs/TESTING.md §5](docs/TESTING.md#5-how-to-write-a-safe-test-for-a-destructive-operation)
for the full technique. The same rule applies to `move_path`, `write_file` and
anything else that can clobber — and to any shell command a test constructs.

---

## Commits and pull requests

One logical change per commit. Present-tense imperative subject, under ~72
characters, no trailing full stop:

```
Add vLLM backend with OpenAI-compatible streaming
Fix drive-relative path resolution in delete_path
Document Hugging Face token precedence in README
```

The body explains *why*, not *what* — the diff already says what. If the change
fixes something subtle, say what the failure looked like; that is what someone
reading `git log` in a year needs.

Before opening a pull request:

1. `python -m pytest tests -q` passes — the **whole** suite, not just your file.
2. `ruff check .` is clean.
3. The suite still passes in a venv with no optional dependencies.
4. `jarvis doctor` runs without a traceback.
5. New behaviour has a test that fails without your change.
6. Documentation that names a symbol, flag or path you changed is updated.

Say plainly in the PR description what you could **not** verify. "Written and
unit-tested on Windows against a faked `systemctl`; never run against a real
systemd" is a useful sentence. Silence about it is not.

---

## Adding a dependency

The bar is high: every dependency is something that can fail to build on the
target laptop, and the core package deliberately has **zero** required
dependencies. Prefer the standard library. `jarvis/llm/ollama_backend.py` talks
HTTP with `urllib` rather than pulling in `requests`, and that is the house
style.

If you do add one, update **all** of these:

| file | what to add |
|---|---|
| `pyproject.toml` | the package in the right `[project.optional-dependencies]` group — never in `dependencies` |
| `requirements.txt` | if it belongs to the `lean` profile, with a comment saying what it buys |
| `requirements-full.txt` | if it belongs to the `full` profile, with the download size if it is large |
| `jarvis/cli.py` → `cmd_doctor` | a row in the right group: `(module_name, "what it unlocks", "pip install <name>")` |
| the code | a lazy `try/except ImportError` import and an `is_available()` that returns `False` without it |
| `install.sh` | any **system** package it needs (`portaudio19-dev`, `ffmpeg`, …) in the per-distro lists |
| `README.md` | only if it changes what a user has to do |

Platform-conditional dependencies use environment markers, so `pip` skips them
elsewhere rather than failing:

```toml
"pywin32>=306; sys_platform == 'win32'",
"vllm>=0.6.0; sys_platform == 'linux'",
"piper-tts>=1.2.0; python_version < '3.14'",
```

Then confirm the package still imports without the new dependency:

```bash
python -m pytest tests/test_import_hygiene.py -q
```

---

## Adding a model

`jarvis/llm/models.py` holds the registry of known Hugging Face repos and their
short aliases. Adding an entry there is a data change, not a code change — see
[docs/MODELS.md](docs/MODELS.md) for the fields and what each one means.

Nothing in the codebase should hardcode a model name. `config.llm.model` is the
single source of truth.

---

## Licence

MIT. By contributing you agree your work is released under it.
