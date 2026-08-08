# Handover

You are new here. This document is the fastest honest path to being useful.

Read this, then ARCHITECTURE.md. Everything else can wait until you need it.

---

## 1. What JARVIS is, in five sentences

JARVIS is a local, voice-driven personal assistant that runs entirely on one
machine and has full, unrestricted control of it — filesystem, shell, processes,
applications, windows, clipboard, screen. It talks to a local language model
through a pluggable backend (a vLLM or Ollama server, in-process transformers, or
disk-paged AirLLM), listens with offline Whisper, and answers in a polished British
voice. Everything it is told is written to a local SQLite store that is never
pruned, and every reply is assembled from a hybrid keyword-plus-vector recall over
that store. Anything slow is handed to a background subagent so the conversation
never blocks, and the subagent's report is folded into the next turn. When a
capability it needs does not exist, it can write, validate and load a new Python
tool for itself at runtime.

### What it is not

* **Not a cloud product.** No accounts, no telemetry, no server component. The only
  network traffic is what you explicitly configure: an LLM endpoint, `edge-tts` if
  you choose that voice, model downloads, and whatever the web tools are asked to
  fetch.
* **Not sandboxed, and not trying to be.** The security layer ships disabled by
  owner decision (§5). Treat a running JARVIS the way you would treat an unattended
  root shell.
* **Not multi-user.** One machine, one owner, one memory database.
* **Not a framework.** It is one assistant. The contracts are clean enough to swap
  parts, but nothing here is packaged for reuse.
* **Not a document RAG system.** Memory holds the conversation, facts, summaries and
  task reports — not your files. Files are read through tools, on demand.
* **Not finished.** See §3.

---

## 2. Current state — what genuinely works

**Verified by running it, on this machine, today:**

| Claim | How it was verified |
|---|---|
| The full test suite passes | `python -m pytest tests -q` → 1607 passed in ~68 s, Windows 11 / Python 3.14.6. The count grows weekly; run the command rather than trusting the number |
| The whole package imports on bare stdlib Python | `tests/test_import_hygiene.py` boots a clean subprocess with ~30 heavy packages blocked by a `sys.meta_path` hook and imports every module |
| 72 built-in tools register; the orchestrator adds 7 meta-tools (79 total) | `create_registry(cfg, SecurityGate(cfg.security)).names()` — the count moves, run it |
| The system boots end-to-end and holds a conversation | `tests/test_integration.py` runs the real `app.build()` and drives turns through a scripted model |
| Memory persists, searches and survives a restart | `tests/test_memory.py`, `tests/test_context.py` |
| The tool-call parser survives every malformed shape a small model produces | `tests/test_protocol.py` |
| The delete guards refuse roots, home and the working directory | `tests/test_file_tools.py:241-361` — read the comment block there |

The repository's own `requirements.txt` states that the voice pipeline was tested
end to end on the Windows dev machine — British speech synthesised, played, and
transcribed back. That is the owner's measurement, not one taken here.

### What is written but UNVERIFIED

Stated plainly, because a confident sentence about untested code is worse than no
sentence:

* **Every Linux path.** The production target is Linux; the code has, as far as any
  evidence in this repository shows, only ever been *run* on Windows. That covers:
  the XDG data/config directories, `install.sh`, `wmctrl`/`xdotool` window control,
  `xdotool` input synthesis, `notify-send`, `amixer`, `espeak-ng`, `xdg-open`, the
  `.desktop` autostart entry, and the `keyboard` package's root requirement. It also
  covers the whole of the `jarvis/linux/` package — the systemd **user** unit
  (`service.py`), desktop notifications and window control (`desktop.py`), and
  PipeWire/PulseAudio/ALSA detection (`audio.py`). All of it is unit-tested against
  faked `systemctl`, `pactl` and `wmctrl`; the unit text, argument lists and control
  flow are verified, real systemd behaviour is not. Treat the first
  `jarvis.linux.service.status()` on the laptop as the acceptance test.
* **No real Qwen3 generation has ever run.** Every LLM test uses `ScriptedLLM` or
  `StubBackend`. Nothing in the suite loads weights, contacts Ollama, or contacts
  vLLM. The backends are tested for their *plumbing* — request shape, response
  parsing, retries, streaming, availability probes — not for producing text. The
  first real generation is a bring-up exercise.
* **vLLM has never been contacted.** `VLLMBackend` and `OpenAICompatBackend` are
  tested against fakes. The wire format is the standard OpenAI one, so it should
  work; "should" is doing real work in that sentence.
* **Piper on Python 3.14.** There is no `piper-tts` wheel, so the requirement is
  pinned `python_version < "3.14"`. On the dev machine Piper is therefore absent and
  the voice falls back to edge-tts or SAPI.
* **Window and input control under Wayland.** `window_tools` shells out to `wmctrl`
  and `xdotool`; `input_tools` prefers `pyautogui` and falls back to `xdotool`. All
  of those are X11 tools.
* **Running as a service.** `jarvis/linux/service.py` generates and controls a
  systemd **user** unit, but it has only ever been driven against a faked
  `systemctl`. See OPERATIONS.md §8.

### What is deliberately absent

Not oversights — decisions:

| Absent | Why |
|---|---|
| Safety rails | Owner decision. `mode="open"`, `protected_paths=[]`, `dangerous_patterns=[]`. The engine is intact and one config line away |
| A GUI | A tray icon and hotkeys, nothing more. This is a voice assistant |
| Database migrations | The schema has never changed. `SCHEMA_VERSION` is written and never read — the hook exists for the first change |
| Memory pruning | The "forgets nothing" guarantee. `memory.prune` exists in the config and is read by nothing |
| Configurable agent-tree limits | The limits themselves exist (`max_depth=3`, `max_total_tasks=64`); exposing them as `AgentConfig` fields is simply not done yet — see §5 |
| An ANN index for vector recall | A full scan is correct at personal-assistant scale and has no dependencies |
| Telemetry, analytics, crash reporting | Local assistant |
| Authentication | One machine, one owner |
| Version control | There is no `.git` directory. See §5 |

---

## 3. The twenty-minute orientation

Read these ten in this order. Skim the bodies; read every docstring. The whole set
is about 5,000 lines, and most of it explains itself.

| # | File | What it teaches |
|---|---|---|
| 1 | `jarvis/core/contracts.py` (~350 lines) | The entire vocabulary. `Message`, `LLMBackend`, `Tool`/`ToolSpec`/`ToolResult`, `MemoryStore`, `Task`. Zero dependencies. Once you know these, every other file is legible |
| 2 | `jarvis/core/config.py` | Every knob that exists, with the reasoning in the field comments. Also the merge order: defaults → file → `JARVIS_*` env. Note that unknown keys are silently dropped |
| 3 | `jarvis/app.py` (~200 lines — the shortest useful read here) | How the whole thing is wired, in one screen. Each subsystem in its own try/except; `Subsystems.status()` is what `jarvis doctor` prints |
| 4 | `jarvis/agent/subagent.py` | `run_agent_loop()` — the reason-act loop, and the single most important function in the project. Also `SubAgent` |
| 5 | `jarvis/agent/protocol.py` (~300 lines) | How model output becomes an action. Read `parse_tool_calls` and note how forgiving it is, and why |
| 6 | `jarvis/agent/orchestrator.py` | The thing you actually talk to. Turn handling, delegation into the task tree, report collection, and the seven meta-tools registered in `_register_meta_tools()` |
| 7 | `jarvis/tools/registry.py` | The single execution path. `run()` is where validation, coercion, security and timeouts live. `FunctionTool` shows how a plain function becomes a tool |
| 8 | `jarvis/memory/store.py` + `jarvis/memory/context.py` | Durability and recall. Read `search()` for the hybrid scoring, then `build()` for how a prompt is assembled |
| 9 | `jarvis/core/platform_utils.py` + `jarvis/core/security.py` | The only module allowed to know about Windows vs Linux; and the policy engine that ships switched off. Read `resolve_path()`'s docstring — it explains a real disaster |
| 10 | `tests/conftest.py` + `tests/test_integration.py` | How to test anything here. `isolated_home`, `ScriptedLLM`, `FakeRegistry`, and the `booted` fixture |

If Linux is your target, add an eleventh: `jarvis/linux/__init__.py` and the
docstrings of `service.py`, `desktop.py` and `audio.py`. They are the most
Linux-specific thinking in the project and each one leads with what will actually
go wrong (lingering, Wayland, PortAudio).

Then run, in this order:

```bash
python -m pytest tests -q      # everything green?
jarvis doctor                  # what is installed, what is missing
jarvis config                  # the effective configuration and the data dir
jarvis tools                   # what it can actually do
jarvis chat                    # talk to it (needs a model — see OPERATIONS.md §3)
```

Companion documents: **ARCHITECTURE.md** (how it fits together and why),
**OPERATIONS.md** (running it), **TESTING.md** (writing tests — including the
destructive-test rule), **TROUBLESHOOTING.md** (symptom → cause → fix),
**docs/TOOL_AUTHORING.md** and **docs/MODELS.md** (written by others; the
authoritative references for those two topics).

---

## 4. How to add a capability

### A new tool

The full guide is `docs/TOOL_AUTHORING.md`. The short checklist:

1. **File.** Add a function to an existing module in `jarvis/tools/`, or create a
   new module there exposing `build_tools(ctx) -> List[Tool]`.
2. **Contract.** The function returns `ToolResult.success(...)` or
   `ToolResult.failure("...")`. It does not raise. Type-hint every parameter — the
   hints become the JSON schema (`_spec_from_callable`, `registry.py:97`). The first
   non-blank docstring line becomes the description the model reads, so write it for
   the model.
3. **Register.** Wrap with `FunctionTool(fn, name="...", description="...",
   dangerous=...)` and return it from `build_tools`.
4. **Wire.** If it is a **new module**, add its name to the `modules` tuple in
   `ToolRegistry.load_builtin()` (`registry.py:466`) — otherwise it is never loaded.
   Position matters: registration uses `replace=True`, so on a name collision the
   later module wins. Read the comment above that tuple before inserting into it.
5. **Laziness.** Any third-party import goes inside the function, in
   `try/except ImportError`, returning a failure with an install hint. OS-specific
   code goes behind `IS_WINDOWS` / `IS_LINUX`.
6. **Subprocess.** Every call passes a `timeout` and kills+waits the child on
   timeout. Prefer `platform_utils.run_command()`, which does this for you.
7. **Test.** New file or new tests in the matching `tests/test_*_tools.py`. Build a
   **real** `ToolRegistry` so validation and security are exercised. If it deletes,
   moves or overwrites anything, read TESTING.md §5 first — that is not optional.
8. **Verify.** `python -m pytest tests -q` and `jarvis tools | grep your_tool`.

### A new LLM backend

1. **File.** `jarvis/llm/<name>_backend.py`.
2. **Base.** Subclass `OpenAICompatBackend` if the server speaks
   `/v1/chat/completions` — you get transport, retries, streaming and the in-flight
   cap for free (`VLLMBackend` is ~60 lines this way). Otherwise subclass `BaseLLM`.
3. **Contract.** Implement `name`, `is_available()` (never raises, returns `False`
   when unavailable), `_do_load()`, and `generate(messages, config) -> LLMResult`.
   Override `stream()` if the server streams; the base yields whitespace-chunked
   `generate()` output otherwise.
4. **Housekeeping.** Merge the caller's config with `self._gen_config(config)`. Run
   the output through `strip_thinking()` and `apply_stop_strings()` — Qwen3 emits
   `<think>` blocks and the loop must not see them.
5. **Register.** Add to `BACKENDS` and, if it should be auto-selected, to
   `AUTO_PROBE_ORDER` in `jarvis/llm/__init__.py`. Position matters: a backend whose
   `is_available()` cannot fail must go last (see ARCHITECTURE.md §7). Add an entry
   to `_INSTALL_HINTS`.
6. **Surface.** Add its dependency to a `cmd_doctor` group in `jarvis/cli.py:89` and
   to the `--backend` help text; add an extra in `pyproject.toml`; document it in
   `config.example.yaml`. New config fields must be added to `LLMConfig` **and** the
   example file — a field only in the example does nothing.
7. **Test.** In `tests/test_llm.py`. Fake the transport (monkeypatch
   `urllib.request.urlopen`, or the library's client object). **No network.** Assert
   the request shape, the response parsing, availability-when-absent, and the
   fallback path when `allow_fallback` is true.

### A new speech engine

1. **File.** A class in `jarvis/speech/stt.py` or `jarvis/speech/tts.py` (or a new
   module for something large, as `windows_speech.py` does).
2. **Contract.** `STTEngine`: `name`, `is_available()`, `transcribe(audio,
   sample_rate) -> Transcript`. `TTSEngine`: `name`, `is_available()`,
   `synthesize(text) -> bytes` (16-bit PCM WAV), and `speak(text)` if it can play
   directly. Return an empty `Transcript` rather than raising on failure.
3. **Register.** STT: add to `_ENGINE_CLASSES` and `_AUTO_ORDER`
   (`speech/stt.py:382`). TTS: add a branch to `_make()` and a slot in
   `_ENGINE_ORDER` (`speech/tts.py:861`). Order is quality-descending — the first
   available one wins.
4. **Config.** Add fields to `STTConfig` / `TTSConfig`, and document them in
   `config.example.yaml`.
5. **Voice.** For TTS, run text through `british_polish()` (`speech/tts.py:224`) so
   numbers, times and abbreviations are spoken naturally.
6. **Test.** In `tests/test_stt.py` / `tests/test_tts.py`. Monkeypatch the engine's
   backend module; **never open a real audio device**. Assert: unavailable when the
   dependency is missing, the factory picks it when configured, the factory skips it
   when unavailable, and synthesis output is WAV-shaped.
7. **Verify.** `jarvis doctor` should list it; `jarvis say -o out.wav` should
   produce audio.

---

## 5. Known sharp edges

Honest list. None of these is hidden in the code; several are documented in the
source but easy to miss.

**The security layer ships disabled, by owner request.** `mode="open"`,
`protected_paths=[]`, `dangerous_patterns=[]`, `unattended_policy="allow"`. In open
mode every check short-circuits to "allowed, no confirmation"
(`core/security.py:343`). The engine — guarded and readonly modes, destructive
command-shape detection, confirmation callbacks — is complete and tested; it is
simply not switched on. Do not add prompts, allowlists or blocklists back; that
decision has been made. The one thing that survives is `delete_path`'s refusal of
filesystem roots, home, and the working directory and its ancestors, and the
argument there is "this cannot succeed", not "you may not" — read the docstring at
`tools/file_tools.py:447`.

**There is no version control.** `git rev-parse` fails: there is no `.git`
directory in the working tree. The project was rebuilt after an earlier copy was
destroyed mid-development, and the history did not survive. Consequence: there is
no blame, no bisect, and no way to recover a file you overwrite. **Initialise a
repository and commit before you change anything.**

**`.gitignore` will silently exclude the tool package.** It contains bare
`tools/`, `models/`, `logs/` and `data/` entries with no leading slash. Git matches
those at any depth, so `jarvis/tools/` — eight modules and the registry — would be
excluded from a first commit, along with anything else matching. Check before you
trust your first commit:

```bash
git init && git add -A
git check-ignore -v jarvis/tools/registry.py     # should print nothing
git status --short | grep "jarvis/tools"          # should list the files
```

The fix is to anchor the patterns to the data directory (`/data/tools/`) or to the
repo root (`/tools/`).

**Piper has no Python 3.14 wheel.** `requirements.txt` pins
`piper-tts>=1.2.0 ; python_version < "3.14"`. The dev machine runs 3.14.6, so the
offline British voice is unavailable there and the stack falls back to edge-tts
(needs internet, returns MP3) or Windows SAPI (which on that machine has only en-US
voices installed). On the Linux box, use Python 3.11 or 3.12 if you want Piper.

**Global hotkeys need administrator rights on Windows and do not work under
Wayland.** The `keyboard` package installs a low-level hook that, against an
elevated foreground window or on a locked-down machine, quietly never starts —
leaving the application convinced it owns a hotkey that will never fire.
`_KeyboardBackend` probes the hook after registration and reports the failure rather
than lying (`win/hotkey.py:285`). On Linux, `keyboard` reads `/dev/input` and raises
at *import* time unless the process is root. Under Wayland there is no global-hotkey
mechanism a normal client can use at all — use the compositor's own keybinding to
launch a command instead.

**Window and input control on Linux is X11-only.** `window_tools` shells out to
`wmctrl` and `xdotool`; `input_tools` prefers `pyautogui` (itself X11-based) and
falls back to `xdotool`. Under a Wayland session these either fail outright or see
only XWayland clients. `jarvis/linux/desktop.py` *detects* the session type and
`ydotool`, and refuses honestly with `is_wayland()` / `_wayland_failure()`, but the
tool layer has no Wayland branch and does not use `ydotool`. Log into an X11 session
if you need window control.

**The desktop-integration packages are not wired into the CLI.** `jarvis/win/`
(tray, hotkeys, autostart, toasts) and `jarvis/linux/` (systemd user service,
notifications, XDG autostart, audio diagnosis) are complete and tested, but nothing
in `cli.py`, `app.py` or `voice.py` imports either — verify with
`grep -rn "jarvis.win\|jarvis.linux" jarvis/cli.py jarvis/app.py jarvis/voice.py`.
So there is no tray icon in a running JARVIS, no hotkey does anything, and
`jarvis doctor` does not report either platform's integration. Drive them from
Python until someone wires them up; that is a contained, high-value first task.

**vLLM is Linux-only.** No supported Windows build. On Windows, run it under WSL2 or
point `llm.vllm_host` at another machine; the client is pure HTTP and does not care
(`llm/vllm_backend.py:15`).

**The agent-tree limits are constants you cannot configure.** `TaskManager` bounds
delegation properly — `max_depth = 3`, `max_total_tasks = 64`, one worker pool per
depth level — and a refused spawn returns an already-failed `Task` whose `error`
explains the limit to the model. But `Orchestrator.__init__` reads them with
`getattr(config.agent, "max_agent_depth", ...)` / `getattr(config.agent,
"max_total_tasks", ...)`, and **neither is a field on `AgentConfig`**, so the
`getattr` default always wins. Adding the two fields is a five-line change.

**Cancellation and task timeouts are cooperative.** Both are only observed when the
runner calls `progress()` (`agent/task_manager.py:356`). A task blocked inside one
long tool call cannot be cancelled or timed out. Relatedly,
`ToolRegistry.run(timeout=...)` joins a daemon thread — Python cannot kill a thread,
so a wedged tool is *abandoned*, not stopped.

**`config.example.yaml` documents settings that do not exist.** `VoiceConfig` has
four fields: `wake_words`, `require_wake_word`, `allow_interrupt`, `greeting`. The
example file additionally documents `mode`, `interrupt_margin`,
`follow_up_seconds`, `continuous_timeout`, `preroll_seconds`, `min_speech_seconds`,
`acknowledge`, and a whole `windows:` section. `_apply_mapping()` skips keys that
are not dataclass fields, so **setting any of them does nothing** — including
`JARVIS_VOICE_MODE=continuous`. `VoiceLoop` reads them with `getattr(..., default)`
and always gets the default, so continuous and push-to-talk modes are currently
unreachable through configuration. Fix by adding the fields, not by deleting the
docs. `config.example.yaml` also advertises a `jarvis calibrate` command that does
not exist in `cli.py`.

**The current user turn is duplicated in every prompt.** `Orchestrator.chat()` calls
`context.add_user(text)` (which appends to the live window) and then
`context.build(text)` (which emits the live window *and* appends a fresh user
message). Harmless to correctness, wasteful of context, confusing when reading a
captured prompt. Reproduction in ARCHITECTURE.md §8.

**The `remember` tool always reports an empty id.** `ContextManager.remember_fact()`
returns `Optional[str]` (the record id), but `Orchestrator._register_meta_tools`
does `getattr(record, "id", "")` on it, which is always `""` for a string
(`agent/orchestrator.py:440`). The fact is stored correctly; only the reported id is
wrong.

**The rolling summary never uses the LLM in production.** `app.py:89` builds the
`ContextManager` without an `llm=`, so `maybe_summarize()` always takes the
extractive fallback — line-truncation, not summarisation. The LLM path exists and is
tested; it is simply not wired.

**Two tool names are resolved by load order, not by design-time uniqueness.**
`app_tools` and `window_tools` both define `focus_window` and `list_windows`;
`window_tools` is listed later in `load_builtin()` so its versions win. That is
deliberate and commented, but it means inserting a module into the middle of that
tuple can silently change which implementation runs.

**A tool call is announced on the bus twice** — once by `run_agent_loop` before
dispatch, once by `ToolRegistry.run` around execution — with different payload
shapes. Anything counting tool calls will double-count.

**`jarvis/__init__.py` says `__version__ = "1.0.0"`; `pyproject.toml` says
`1.1.0`.** Pick one.

**Stray file.** `_probe.txt` in the repo root contains the words "survival probe"
and appears to be a leftover from recovering the project. Nothing imports it.

---

## 6. Glossary

Project-specific terms, in the sense this codebase uses them.

| Term | Meaning here |
|---|---|
| **Agent loop** / **reason-act loop** | `run_agent_loop()` in `agent/subagent.py`. Generate → parse tool calls → run them → feed results back → repeat until the model answers without calling a tool |
| **Turn** | One user input and everything that happened answering it. Represented by `AgentTurn` (text, messages, tool calls, results, iterations, truncated) |
| **Iteration** | One pass through the agent loop: one generation plus any tools it requested. Bounded by `agent.max_tool_iterations` |
| **Orchestrator** | The main, interactive agent (`agent/orchestrator.py`). The thing you talk to. Holds a lock for the duration of a turn |
| **Subagent** | An autonomous worker pursuing one goal on a background thread, whose final message is a *report* rather than a spoken reply (`SubAgent`) |
| **Task** | A unit of delegated work managed by `TaskManager`. States: `pending`, `running`, `done`, `failed`, `cancelled` |
| **Report** | A finished task's outcome, queued until announced. Consumed atomically by `take_reports()`, injected into the next turn as a system message, and persisted as memory kind `task` |
| **Meta-tool** | One of the seven tools the Orchestrator registers on itself: `spawn_task`, `task_tree`, `list_tasks`, `task_status`, `cancel_task`, `remember`, `recall` |
| **Tool spec** | `ToolSpec` — the model-visible description of a tool: name, description, typed parameters, `dangerous` flag. Derived automatically from a function's signature and docstring |
| **Generated tool** | A Python module JARVIS wrote for itself at runtime, living in `<data dir>/tools/`, imported under `jarvis_generated.<stem>` |
| **Tool maker** | `tools/tool_maker.py` — the machinery that drafts, statically validates, writes, imports and rolls back a generated tool |
| **Registry** | `ToolRegistry` — discovery plus the single execution path for every tool call |
| **Context manager** | `memory/context.py` `ContextManager` — the live conversation window plus recall and summarisation. Nothing to do with Python's `with` statement |
| **Live window** | The last `keep_recent_turns` messages held verbatim in memory (`ContextManager._live`) |
| **Rolling summary** | The compressed form of everything that fell out of the live window; persisted as memory kind `summary` |
| **Recollections** | Search hits injected into the prompt as a `Relevant recollections:` system block |
| **Kind** | The category on a `MemoryRecord`: `conversation`, `fact`, `summary`, `task` |
| **Hybrid search** | Recall scoring `0.55 × cosine + 0.35 × keyword + 0.10 × recency`, over a candidate pool drawn only from keyword or vector matches |
| **Security gate** | `SecurityGate` — the policy oracle. Answers `check_path`, `check_command`, `check_tool` with a `Decision`. Ships in `open` mode, which always says yes |
| **Decision** | `SecurityGate`'s verdict: `allowed`, `reason`, `requires_confirmation`, `destructive`. `bool(decision)` is true only when allowed *and* needing no confirmation |
| **Audit log** | `<data dir>/logs/audit.jsonl`. A record of decisions; never a restriction |
| **Open mode** | The shipped security mode. Every check short-circuits to allowed |
| **Barge-in** | Interrupting JARVIS by speaking while it is talking. Playback is cut and the rest of the reply discarded (`voice.py:560`) |
| **Wake word** | A word that must open an utterance (after optional filler) for it to be treated as addressed to JARVIS. Merely mentioning the name does not count |
| **Follow-up window** | The seconds after a reply during which no wake word is needed |
| **Doctor** | `jarvis doctor` — the dependency and subsystem report. The first command to run when anything is wrong |
| **Data dir / home** | `Config.home()`. Where the database, generated tools, voices, models and logs live. Overridable with `JARVIS_HOME` |
| **`is_available()`** | The universal capability probe. Must return `False` — never raise — when dependencies are absent |
| **Lazy import** | Every third-party import lives inside the function that needs it, in `try/except ImportError`, so the package imports on bare stdlib Python. Enforced by `tests/test_import_hygiene.py` |
| **Hermetic** | The test-suite property: no network, no models, no audio device, no GPU, nothing written outside `tmp_path` |
| **Scripted LLM** | `tests/conftest.py` `ScriptedLLM` — a deterministic `LLMBackend` whose replies come from a list |
| **Drive-relative path** | A Windows path like `C:` or `C:foo` with no separator. It means "the current directory on that drive", **not** the drive root. Resolving one naively once destroyed this project's source tree; `platform_utils.resolve_path()` anchors it to the drive root instead |
| **Canary** | A file with distinctive content placed inside a fake target in a destructive test, asserted to survive. See TESTING.md §5 |
