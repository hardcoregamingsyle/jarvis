# Troubleshooting

Symptom → cause → fix. Every row ends in a command you can run.

`jarvis doctor` is the right first move for almost everything: it prints the host,
the data directory, a present/absent line for every optional dependency with its
install command, and then actually boots each subsystem and reports what came up.

A few of these commands are long. They are written as one-liners so you can paste
them; on Windows use the `python -c "..."` form shown, in PowerShell or Git Bash.

---

## Quick reference

| # | Symptom | Most likely cause | First command |
|---|---|---|---|
| 1 | "JARVIS could not start" | No LLM backend, or memory failed to open | `jarvis doctor` |
| 2 | No microphone found | `sounddevice` missing, or no device / no PortAudio | `python -c "from jarvis.core.config import load_config as l; from jarvis.speech.audio_io import AudioRecorder as R; r=R(l().stt); print(r.is_available()); print(r.list_devices())"` |
| 3 | STT returns empty | `NullSTT` selected — no real engine installed | `python -c "from jarvis.core.config import load_config as l; from jarvis.speech.stt import create_stt, available_stt_engines as a; c=l().stt; print(create_stt(c).name, [e.name for e in a(c)])"` |
| 4 | The voice is silent | `NullTTS` selected, or `tts.enabled` is false, or no playback backend | `jarvis say "testing one two three"` |
| 5 | Wrong / US accent | An en-US system voice won the auto-probe | `python -c "from jarvis.core.config import load_config as l; from jarvis.speech.tts import available_tts_engines as a, create_tts; c=l().tts; print(create_tts(c).name, a(c))"` |
| 6 | Replies take minutes | AirLLM selected, or a 30B model on CPU, or `max_new_tokens` too high | `python -c "from jarvis.core.config import load_config as l; from jarvis.llm import create_llm, available_backends; c=l().llm; print(create_llm(c).name, available_backends(c))"` |
| 7 | The model will not download | No disk, no network, or a bad repo id | `python -c "from jarvis.core.config import load_config as l; from jarvis.llm import models as m; print(m.check_access(m.resolve_config(l()), m.hf_token(l())))"` |
| 8 | A gated HF repo returns 401 | No token, or the licence was never accepted | `python -c "from jarvis.core.config import load_config as l; from jarvis.llm.models import hf_token; print(bool(hf_token(l())))"` |
| 9 | vLLM will not start, or answers 503 | Not Linux, weights still loading, or wrong base URL | `python -c "from jarvis.core.config import load_config as l; from jarvis.llm.vllm_backend import VLLMBackend; import json; print(json.dumps(VLLMBackend(l().llm).health(), indent=2))"` |
| 10 | Ollama not found | Daemon not running, or a different host/port | `curl -s http://127.0.0.1:11434/api/tags` |
| 11 | Tools fail with permission errors | The OS refused, not JARVIS — the gate is open by default | `python -c "from jarvis.core.config import load_config as l; c=l().security; print(c.mode, c.allow_shell, c.allow_file_write, c.protected_paths)"` |
| 12 | Tray / hotkey does nothing | No backend, hook never started, or needs admin | `python -c "import json; from jarvis.win import is_windows_integration_available as w; print(json.dumps(w(), indent=2, default=str))"` |
| 13 | Wayland window control does nothing | `wmctrl`/`xdotool` are X11 tools; the tool layer has no Wayland path | `python -c "import json; from jarvis.linux import desktop; print(json.dumps(desktop.is_available(), indent=2, default=str))"` |
| 14 | A generated tool will not load | Import error in the module, or it is not registered | `python -c "import logging; logging.basicConfig(level=logging.DEBUG); from jarvis.core.config import load_config as l; from jarvis.core.security import SecurityGate as S; from jarvis.tools import create_registry; c=l(); r=create_registry(c, S(c.security)); print('loaded', r.load_generated())"` |
| 15 | Memory not persisting | Write failures, or two databases in different directories | `jarvis memory --stats` |
| 16 | Tests fail after adding a dependency | A module-level heavy import | `python -m pytest tests/test_import_hygiene.py -q` |

---

## The installer downloads the wrong model

**Symptom.** You upgraded and re-ran `./install.sh`, but it pulls `qwen3.6:27b`
when this release defaults to `qwen3.8:27b`.

**Cause.** `config.yaml` is yours and is gitignored, so an upgrade never
rewrites your model choice. An install predating the new default still pins the
old tag, and the installer honours it.

**Fix.**

```bash
./install.sh --model qwen3.8:27b     # switch and pull
ollama rm qwen3.6:27b                # reclaim ~18 GB
```

The installer warns about this mismatch rather than proceeding silently:

```
config.yaml pins llm.ollama_model: qwen3.6:27b
This release defaults to qwen3.8:27b. Keeping your setting.
To switch:  ./install.sh --model qwen3.8:27b
```

Confirm what is actually loaded with `jarvis selftest`, which prints the model
id next to the backend.


## 1. "JARVIS could not start"

`cli.py` prints this when `Subsystems.orchestrator is None`, which happens only when
`app.build()` could not produce **both** an LLM backend and a memory context
(`jarvis/app.py:155`). Everything else degrades silently — no microphone, no voice,
no tools — but those two are required.

```bash
jarvis doctor
```

Read the `Subsystems` block at the bottom:

* `llm  unavailable` → no backend was created at all. Rare: `StubBackend` is always
  available, so this usually means an exception inside `create_llm`. Run with
  `-v` for the traceback: `jarvis -v doctor`.
* `llm  stub` → JARVIS started, but with the canned-response backend. It will talk
  nonsense politely. Go to §6 and §10.
* `memory  unavailable` → the SQLite file could not be opened. Check the path and
  its permissions:
  ```bash
  python -c "from jarvis.core.config import load_config as l; c=l(); print(c.db_file(), c.db_file().parent.exists())"
  ```
* `memory  sqlite (WRITE FAILURES - see log)` → see §15.

If `doctor` itself explodes, the config file is the usual culprit — a YAML syntax
error, or a JSON file with a trailing comma:

```bash
python -c "from jarvis.core.config import default_config_paths as p; [print(x, x.exists()) for x in p()]"
python -c "from jarvis.core.config import load_config; load_config(); print('config OK')"
```

---

## 2. No microphone found

`jarvis` (bare) falls back to text mode when `AudioRecorder(cfg.stt).is_available()`
is false, and `jarvis voice` prints "No usable microphone or speech-to-text engine
was found."

`AudioRecorder.is_available()` (`speech/audio_io.py:464`) requires two things: that
`sounddevice` imports, **and** that `sd.query_devices()` succeeds. A backend that
imports but finds no driver counts as unavailable.

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.speech.audio_io import AudioRecorder as R; r=R(l().stt); print('available:', r.is_available()); [print(d) for d in r.list_devices()]"
```

| Result | Fix |
|---|---|
| `sounddevice` not installed | `pip install sounddevice numpy` |
| Linux, import fails on `libportaudio` | `sudo apt-get install -y portaudio19-dev` (or `dnf install portaudio-devel`, `pacman -S portaudio`) |
| Device list empty on Linux | Get the full diagnosis, including the exact package command for this distro: `python -c "import json; from jarvis.linux import audio; print(json.dumps(audio.check(), indent=2, default=str))"` |
| Device list empty on Windows | Settings → Privacy → Microphone → allow desktop apps |
| Wrong device chosen | Set `stt.input_device` to the index printed above, then `jarvis config --write` |
| Running headless / over SSH | There is no device. Use `jarvis chat` |

---

## 3. STT returns empty transcripts

You speak, the loop records, and nothing happens. Almost always: no real engine is
installed, so `create_stt` fell through to `NullSTT`, whose `transcribe()` returns an
empty `Transcript` by design (`speech/stt.py:364`).

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.speech.stt import create_stt, available_stt_engines as a; c=l().stt; print('selected:', create_stt(c).name); print('available:', [e.name for e in a(c)])"
```

If it prints `selected: null`:

```bash
pip install faster-whisper          # the recommended engine
```

The auto-probe order is `faster-whisper` → `whisper` → `vosk` → `windows` → `null`
(`speech/stt.py:394`). `windows` uses the recogniser built into Windows: no
download, noticeably less accurate, and it exists so voice works on a machine with
nothing installed.

If a real engine is selected and transcripts are *still* empty:

| Cause | Check |
|---|---|
| First run is downloading the Whisper model | Watch the log: `tail -f "$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/jarvis.log"` |
| Silence threshold too high — nothing is ever captured | Raise the mic gain, or lower `stt.silence_threshold` (default `0.015`; `config.example.yaml` reports a measured room floor of ~0.007) |
| Silence threshold too low — it records the room forever | Raise it |
| Wrong language | `stt.language: en`, `stt.model: small.en` — the `.en` models are English-only |
| Audio arriving as the wrong dtype/rate | Round-trip a known file: `python -c "from jarvis.core.config import load_config as l; from jarvis.speech.stt import create_stt; print(create_stt(l().stt).transcribe_file('sample.wav').text)"` |

Produce that `sample.wav` with `jarvis say -o sample.wav "the quick brown fox"` and
you have an end-to-end voice check that needs no microphone.

---

## 4. The voice is silent

```bash
jarvis say "testing one two three"
```

It prints the selected engine before speaking. Three distinct failures hide here.

**a) `Voice engine: null`.** No TTS engine was usable, so `NullTTS` was selected;
it synthesises 0.15 s of silence and logs the text at INFO
(`speech/tts.py:286`).

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.speech.tts import available_tts_engines as a; print(a(l().tts))"
pip install edge-tts        # online, best British voice
pip install piper-tts       # offline — needs Python < 3.14 (no wheel for 3.14)
```

**b) A real engine is selected but nothing is heard.** Synthesis works, playback does
not.

```bash
jarvis say -o /tmp/check.wav "testing"     # writes a file instead of playing
```

If the file exists and plays in another player, the problem is the playback path:
`pip install sounddevice`, check `tts.output_device`, and on Linux check that
PulseAudio/PipeWire has a sink (`pactl list short sinks`).

**c) The voice is disabled.** `tts.enabled: false`, or `--no-speech` was passed, or
`config` fixture-style defaults are in force. `create_tts` returns `NullTTS`
immediately when `enabled` is false (`speech/tts.py:903`).

```bash
python -c "from jarvis.core.config import load_config as l; print(l().tts.enabled, l().tts.engine)"
```

**Note on edge-tts:** it returns MP3, not WAV. `jarvis say -o out.wav` detects this
and writes `out.mp3` instead, telling you so. Install `ffmpeg` (or `pip install
pydub`) for WAV output. Playback and transcription both handle MP3 already.

---

## 5. Wrong accent — it sounds American

The TTS auto-probe is `piper → edge → sapi → pyttsx3 → espeak → null`
(`speech/tts.py:861`). If Piper and edge-tts are both unavailable, you land on the
operating system's own voices, and on a default Windows install those are
`Microsoft David` and `Microsoft Zira` — both en-US.

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.speech.tts import available_tts_engines as a, create_tts; c=l().tts; print('selected:', create_tts(c).name); print('available:', a(c))"
```

Fixes, best first:

```yaml
tts:
  engine: edge                    # needs internet; closest to the films
  edge_voice: en-GB-RyanNeural    # or en-GB-ThomasNeural, en-GB-SoniaNeural
```

```yaml
tts:
  engine: piper                   # offline; needs Python < 3.14
  piper_voice: en_GB-alan-medium  # male RP
```

```bash
jarvis setup                      # downloads the Piper voice into <data dir>/voices
```

If you must use the system voice, install a British one — Windows: Settings → Time &
language → Speech → Add voices → English (United Kingdom) — and leave
`tts.sapi_voice_hint: United Kingdom`, which is what `select_voice()` matches on
(`speech/windows_speech.py:218`). On Linux, `espeak-ng` accepts an English (RP)
variant but is robotic; edge-tts is the better answer.

Audition each candidate:

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.speech.tts import _make; c=l().tts; e=_make('edge', c, None); print(e.is_available()); e.speak('Good evening, Sir.')"
```

---

## 6. Replies take minutes

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.llm import create_llm, available_backends; c=l().llm; print('selected:', create_llm(c).name); print('available:', available_backends(c)); print('model:', c.model, '| ollama:', c.ollama_model)"
```

| Selected | Expected speed on an i5-10210U, CPU only | What to do |
|---|---|---|
| `airllm` | **0.02–0.1 tok/s** (10–50 s *per token*) | This is the cause. Switch backends |
| `ollama` + a 30B model | 4–8 tok/s | Fine for background work; switch to a 4B model for conversation |
| `ollama` + `qwen3:4b-instruct-2507-q4_K_M` | 15–25 tok/s | This is the target |
| `transformers` on CPU | <1 tok/s | Small models only, or move to a GPU |

The fix is almost always Ollama with a small model:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

```yaml
llm:
  backend: ollama
  ollama_model: qwen3:4b-instruct-2507-q4_K_M
  max_new_tokens: 256      # spoken replies are two or three sentences
  context_tokens: 4096
```

Other contributors, in descending order of impact:

* **`max_new_tokens: 512`** — every token costs. 256 halves the worst case.
* **`agent.max_tool_iterations: 8`** — each iteration is a *full generation*. A
  turn that uses six tools is seven generations. Lower it to 4–6 on slow hardware.
* **Prompt length** — `memory.recall_k: 8` injects eight full records. Lower it.
* **Concurrency** — four subagents on a serialising backend queue behind each other.
  Lower `agent.max_concurrent_tasks`, or move to vLLM / `OLLAMA_NUM_PARALLEL>1`.

Confirm where the time actually goes:

```bash
jarvis -v ask "say hello" 2>&1 | tail -40
```

---

## 7. The model will not download

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.llm import models as m; c=l(); print(m.check_access(m.resolve_config(c), m.hf_token(c)))"
```

`check_access` makes one cheap API call before the 30 GB one and returns exactly one
of `ok`, `gated`, `needs_token`, `not_found`, `offline` (`llm/models.py:752`).

| `status` | Meaning | Fix |
|---|---|---|
| `not_found` | Typo, or a private repo invisible to you | Check `llm.model` against the catalogue: `python -c "from jarvis.llm.models import all_models; [print(s.id) for s in all_models()]"` |
| `offline` | huggingface.co unreachable | `curl -sSf https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507 >/dev/null && echo reachable` |
| `needs_token` / `gated` | See §8 | |
| `ok` but the download still fails | Disk, or a proxy | `df -h "$(python -c 'from jarvis.core.config import load_config as l; print(l().models_dir())')"` |

Space matters more than people expect. A 30B model is 17–60 GB depending on
quantisation, and AirLLM additionally caches per-layer shards:

```bash
python -c "from jarvis.llm.models import estimate_footprint, resolve_config; from jarvis.core.config import load_config as l; print(estimate_footprint(resolve_config(l())))"
python -c "from jarvis.llm.models import local_models; [print(e) for e in local_models()]"
```

For **Ollama** models this is not JARVIS's download at all — Ollama fetches them:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama list
```

For the **Piper voice**, `jarvis setup` fetches from `rhasspy/piper-voices` on
Hugging Face; `ensure_voice(download=False)` never touches the network
(`speech/tts.py:357`):

```bash
jarvis setup                 # download
jarvis setup --no-download   # check only
```

---

## 8. A gated Hugging Face repo returns 401

Llama, Gemma and anything behind a licence click-through are *gated*: the repo
exists, but you must accept terms while signed in, and then present a token.

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.llm.models import hf_token; t=hf_token(l()); print('token found:', bool(t))"
```

`hf_token()` (`llm/models.py:696`) takes the first non-empty of:

1. `llm.hf_token` in the config (or `JARVIS_LLM_HF_TOKEN`)
2. `HF_TOKEN`
3. `HUGGING_FACE_HUB_TOKEN`
4. the file written by `huggingface-cli login`

A blank config value cannot shadow a good environment variable, and the function
returns `None` rather than `""` — an empty bearer token produces a baffling 401.

```bash
# 1. Accept the licence in a browser, signed in, on the model's page.
python -c "from jarvis.core.config import load_config as l; from jarvis.llm.models import resolve_config; print('https://huggingface.co/' + resolve_config(l()).id)"

# 2. Provide a token (read scope is enough).
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx          # Linux
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"        # PowerShell
#   or persistently:  huggingface-cli login

# 3. Re-check.
python -c "from jarvis.core.config import load_config as l; from jarvis.llm import models as m; c=l(); print(m.check_access(m.resolve_config(c), m.hf_token(c)))"
```

`status: gated` after supplying a token means the licence has not been accepted for
*that account*. `status: needs_token` means no token reached the call at all.

Never put the token in `config.yaml` if the repository will ever be shared —
`.gitignore` excludes `config.yaml` and `hf_token*`, but an environment variable is
safer. `models.redact()` keeps tokens out of log lines and returned dicts.

---

## 9. vLLM refuses to start, or answers 503

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.llm.vllm_backend import VLLMBackend; import json; print(json.dumps(VLLMBackend(l().llm).health(), indent=2))"
```

`health()` never raises; an unreachable server comes back as
`{"reachable": false, "error": ...}` (`llm/vllm_backend.py:172`).

| Observation | Cause | Fix |
|---|---|---|
| `reachable: false` on Windows | **vLLM has no supported Windows build** | Run it in WSL2, or point `llm.vllm_host` at a Linux machine. The client works fine from Windows |
| `reachable: false`, connection refused | Server not running | Print the exact argv and run it: see below |
| HTTP 503 for the first minute or two | Weights still loading — vLLM answers 503 until ready | Wait. The client already retries with capped exponential backoff and jitter (`llm/openai_compat.py:364`) |
| HTTP 401 | Server started with `--api-key` | Set `llm.api_key`, or `JARVIS_LLM_API_KEY` |
| HTTP 404 on `/models` | Base URL missing the `/v1` suffix | `llm.vllm_host: http://127.0.0.1:8000/v1` |
| `reachable: true` but `model` is empty | Config model id ≠ served model id | Use the id from `health()["models"]` |
| Server OOMs at startup | `--max-model-len` or `--max-num-seqs` too large for the box | Lower `llm.context_tokens` and `llm.max_concurrent_requests` |

Get the exact command for your config:

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.llm.vllm_backend import server_command; print(' '.join(server_command(l().llm)))"
```

Then, on the Linux machine:

```bash
pip install "vllm>=0.6.0"
# ...paste the argv printed above, and watch it until it stops logging.
curl -s http://127.0.0.1:8000/v1/models
```

JARVIS **never launches the server itself**. That is deliberate: the server outlives
any single JARVIS process and belongs under your own supervision.

> Untested: no vLLM server has been run against this codebase. Treat first contact
> as bring-up.

---

## 10. Ollama not found

```bash
curl -s http://127.0.0.1:11434/api/tags
python -c "from jarvis.core.config import load_config as l; from jarvis.llm.ollama_backend import OllamaBackend as O; b=O(l().llm); print('available:', b.is_available()); print(b.list_models())"
```

`is_available()` is a 1.5-second `GET /api/tags` (`llm/ollama_backend.py:58`). If it
fails, auto-selection moves on and you may silently end up on `transformers`,
`airllm`, or `stub`.

| Cause | Fix |
|---|---|
| Not installed | Linux: `curl -fsSL https://ollama.com/install.sh \| sh` · Windows: `winget install Ollama.Ollama` |
| Installed, not running | `ollama serve` (Linux service: `systemctl --user status ollama`) |
| Bound elsewhere | Set `llm.ollama_host`; check with `ss -ltnp \| grep 11434` (Linux) or `netstat -ano \| findstr 11434` (Windows) |
| Running, but the model is not pulled | `ollama list`, then `ollama pull qwen3:4b-instruct-2507-q4_K_M` |
| `ollama_model` names a tag that does not exist | Use a name printed by `ollama list` verbatim |
| Replies ignore earlier context | `llm.context_tokens` exceeds the model's window; Ollama truncates | Lower it, and confirm with `ollama show <model>` |

Note that `create_llm` with `allow_fallback: true` (the default) **silently** falls
back when a named backend is unavailable. If you want a hard failure instead — which
is right on a production box — set `llm.allow_fallback: false` and you get a
`RuntimeError` with an install hint (`llm/__init__.py:128`).

---

## 11. Tools fail with permission errors

First, rule out JARVIS. The shipped policy allows everything:

```bash
python -c "from jarvis.core.config import load_config as l; c=l().security; print('mode:', c.mode); print('shell:', c.allow_shell, 'write:', c.allow_file_write); print('protected:', c.protected_paths); print('patterns:', c.dangerous_patterns)"
```

Expected: `mode: open`, both `True`, both lists empty. In that state
`SecurityGate` returns "allowed, no confirmation" for every check
(`core/security.py:343`), so a permission error is coming from the **operating
system**, not from JARVIS.

| Error text | Real cause | Fix |
|---|---|---|
| `PermissionError: [Errno 13]` / `[WinError 5]` | OS file permissions | `ls -l <path>` / `icacls <path>` |
| `refusing to recursively delete the filesystem root` (or home, or the working directory) | `delete_path`'s four whole-tree guards (`tools/file_tools.py:447`) | Delete the specific children, or use `run_command`. This guard is not a policy — those deletes cannot succeed, only half-succeed |
| `is a directory; pass recursive=True` | Working as designed | Pass `recursive=True` |
| `... is not available` from a process tool | `psutil` missing | `pip install psutil` |
| Linux: cannot kill or inspect another user's process | Not root | `sudo` the whole JARVIS process, or accept the limit |
| Windows: cannot touch a service or another session | Not elevated | Run the terminal as Administrator |
| `readonly mode: ...` or `refused — confirmation not granted` | Someone enabled a restrictive mode | `JARVIS_SECURITY_MODE=open jarvis chat` to confirm, then fix the config |

The audit log records every decision. If a refusal came from JARVIS, it is in there:

```bash
tail -20 "$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/audit.jsonl"
```

If that file has no refusal for the failing call, the refusal was the OS's.

---

## 12. The tray icon or hotkey does nothing

```bash
python -c "import json; from jarvis.win import is_windows_integration_available as w; print(json.dumps(w(), indent=2, default=str))"
```

The report includes `hotkeys`, `hotkey_backend`, `hotkey_error`, `tray`,
`tray_error`, `notifications`, `autostart` (`jarvis/win/__init__.py:49`). Every probe
is wrapped so a diagnosis never itself crashes.

| Field | Meaning | Fix |
|---|---|---|
| `hotkey_backend: none` | Neither the `keyboard` package nor the ctypes `RegisterHotKey` fallback is usable | `pip install keyboard` |
| `hotkey_error` mentioning the hook not starting | On Windows the `keyboard` package's low-level hook silently fails against an elevated foreground window or a locked-down machine. Registration is probed afterwards and reported rather than lied about (`win/hotkey.py:285`) | Run the terminal as Administrator |
| Linux, `keyboard: ImportError` at import | The package reads `/dev/input` and refuses unless root | Run as root (not recommended), or bind the key in your desktop environment to run `jarvis` |
| Wayland | No global-hotkey mechanism exists for an ordinary client | Use the compositor's own keybinding settings |
| `tray: False` | `pystray` and/or `Pillow` missing | `pip install pystray Pillow` |
| Linux tray missing | No StatusNotifier host in the panel | GNOME needs the AppIndicator extension |
| Hotkey fires but nothing happens | The callback raised, or the dispatcher queue is full | Check `jarvis.log`; callbacks run on a private worker thread, never on the hook thread |

Note that hotkeys are *not* wired into the CLI today — `jarvis/win/` provides
`HotkeyManager`, `voice_loop_callbacks()` and `default_hotkeys()`, but `cli.py` never
constructs them. There is nothing for the key to do unless you write it.

```bash
python -c "from jarvis.win import default_hotkeys, DEFAULT_TOGGLE_COMBO, DEFAULT_PUSH_TO_TALK_COMBO; print(default_hotkeys(), DEFAULT_TOGGLE_COMBO, DEFAULT_PUSH_TO_TALK_COMBO)"
```

---

## 13. Wayland: window control does nothing

```bash
python -c "import json; from jarvis.linux import desktop; print(json.dumps(desktop.is_available(), indent=2, default=str))"
python -c "from jarvis.tools.window_tools import is_available; print(is_available())"
```

That first command is the authoritative answer. `jarvis/linux/desktop.py` reports
`session` (`x11` / `wayland` / unknown), which of `wmctrl` / `xdotool` / `ydotool`
are present, whether `window_control` and `global_hotkeys` are actually possible,
and an `advice` list naming the next command.

`tools/window_tools.py` on Linux shells out to `wmctrl` and `xdotool`
(`window_tools.py:203, 246, 257`); `input_tools` prefers `pyautogui` and falls back
to `xdotool`. **All of those are X11 tools**, and `tools/window_tools.py` has no
Wayland branch.

| `XDG_SESSION_TYPE` | Behaviour |
|---|---|
| `x11` | Works, once `wmctrl` and `xdotool` are installed |
| `wayland` | `wmctrl` and `xdotool` see only XWayland clients, and usually nothing at all. Focus, move, resize and synthesised input will fail or silently no-op |
| unset / `tty` | No display server — window tools are meaningless |

Options, in order of practicality:

1. **Log into an X11 session.** GDM/SDDM offer it on the login screen. This is the
   only thing that makes the existing code work as written.
2. Install the tools if you are already on X11:
   ```bash
   sudo apt-get install -y wmctrl xdotool
   ```
3. Use `ydotool` (Wayland-capable, needs a `uinput` daemon and permissions) or
   compositor-specific IPC (`swaymsg`, `gdbus` to GNOME Shell). `jarvis.linux.desktop`
   *detects* `ydotool` and reports it; **the tool layer does not use it**. Wiring it
   in means a new branch in `window_tools._all_windows()` and friends, plus tests —
   see HANDOVER.md §4.
4. For global hotkeys under Wayland there is no mechanism an ordinary client can
   use. `desktop.global_hotkeys()` says so rather than pretending:
   ```bash
   python -c "from jarvis.linux import desktop; r=desktop.global_hotkeys(); print(r.ok, r.output or r.error)"
   ```
   Bind the key in your compositor's own settings to run `jarvis` instead.

Confirm the window tools are actually registered — a module missing from
`ToolRegistry.load_builtin()`'s tuple is silently absent rather than an error:

```bash
jarvis tools | grep -E "focus_window|list_windows|move_window"
```

---

## 14. A generated tool will not load

Generated tools live in `<data dir>/tools/*.py` and are imported by
`ToolRegistry.load_generated()` under `jarvis_generated.<stem>`. A module that fails
to import is **logged and skipped**, not fatal (`tools/registry.py:516`) — so the
symptom is a tool quietly missing, not a crash.

```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG); from jarvis.core.config import load_config as l; from jarvis.core.security import SecurityGate as S; from jarvis.tools import create_registry; c=l(); r=create_registry(c, S(c.security)); print('generated loaded:', r.load_generated()); print(r.names())"
```

The DEBUG traceback names the file and the exception.

| Cause | Fix |
|---|---|
| Syntax error or bad import in the generated module | Open the file and fix it; it is ordinary Python: `python -c "from jarvis.core.config import load_config as l; print(l().tools_dir())"` |
| No `build_tools(ctx)` and no `TOOLS` list | The loader takes tools from either; add one |
| `build_tools` returned something that is not a `Tool` | Wrap functions in `FunctionTool(...)` |
| Filename starts with `_` | Skipped by design |
| Written against an older contract after a rollback | Delete it and let JARVIS write it again |
| It imports a package that is no longer installed | Install it, or make the import lazy inside the function |

Creation-time failures are different — `make_tool()` validates statically before
writing and **rolls back the file and the `sys.modules` entry** on any failure
(`tools/tool_maker.py:343`). Common rejections: `eval`/`exec`/`compile`/
`__import__`, `os.system`/`os.popen`/`os.exec*`, `subprocess(..., shell=True)`,
`ctypes`, `shutil.rmtree`, writes to absolute system paths, or a missing
`build_tools`.

```bash
python -c "from jarvis.tools.tool_maker import validate_tool_source; import sys; print(validate_tool_source(open(sys.argv[1], encoding='utf-8').read()))" <path-to-tool.py>
```

Manage them from the CLI side:

```bash
jarvis tools                       # what is actually registered
python -c "from jarvis.core.config import load_config as l; print(sorted(p.name for p in l().tools_dir().glob('*.py')))"
```

---

## 15. Memory not persisting

```bash
jarvis memory --stats
```

That prints `total`, `by_kind`, `embeddings`, `fts5`, `db_path`, `embedder`,
`embed_dim`. Three distinct failures:

**a) Write failures.** `ContextManager` retries a failed write once, records it in
`failed_writes`, and flips `persistence_healthy` to `False`; `jarvis doctor` then
reports `memory  sqlite (WRITE FAILURES - see log)` (`jarvis/app.py:60`). The whole
point is that a broken store must not look like a healthy one.

```bash
jarvis doctor | grep -i memory
grep -i "persist" "$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/jarvis.log" | tail -20
```

Usually a full disk, a read-only directory, or the database file locked by another
process. Check:

```bash
python -c "from jarvis.core.config import load_config as l; c=l(); p=c.db_file(); print(p, p.exists(), p.parent.exists()); import os; print('writable:', os.access(p.parent, os.W_OK))"
df -h "$(python -c 'from jarvis.core.config import load_config as l; print(l().home())')"
```

**b) Two databases.** `db_path` empty means `create_memory()` uses
`data_dir()/memory.db`, while `Config.db_file()` may resolve elsewhere if
`data_dir` or `JARVIS_HOME` is set. `app.build()` and `cli.cmd_memory()` both pin
`cfg.memory.db_path = str(cfg.db_file())` to prevent this — anything constructing a
store by hand can still get it wrong.

```bash
python -c "from jarvis.core.config import load_config as l; from jarvis.core.platform_utils import data_dir; c=l(); print('db_file():', c.db_file()); print('data_dir():', data_dir()); print('configured:', repr(c.memory.db_path))"
find "$HOME" -name "memory.db" 2>/dev/null          # Linux: are there several?
```

**c) It is persisting; recall is just not surfacing it.** Records are stored but
`search()` scores them below `memory.recall_min_score` (0.15), or only
`recall_k` (8) make it into the prompt. Check directly:

```bash
jarvis memory "the thing you told it"
jarvis memory -n 20                 # the 20 most recent records
```

If `fts5: false` in the stats, your SQLite build has no FTS5 and keyword search is
running on a `LIKE` fallback — functional, but weaker. If `embeddings` is far below
`total`, no embedder is configured: `pip install sentence-transformers`, or set
`memory.embedder: hash` for the zero-dependency embedder.

Never delete `memory.db-wal` by hand while JARVIS is running: it holds committed
data. Stop the process first, or export instead:

```bash
jarvis memory --export ~/jarvis-memory-backup.jsonl
```

---

## 16. Tests fail after adding a dependency

```bash
python -m pytest tests/test_import_hygiene.py -q
```

Three failure shapes:

**a) `... imports ['numpy'] at module level`.** Move the import inside the function
that needs it, wrapped in `try/except ImportError`:

```python
def rms(samples):
    try:
        import numpy as np
    except ImportError:
        n = len(samples)
        return (sum(float(s) * float(s) for s in samples) / n) ** 0.5 if n else 0.0
    return float(np.sqrt(np.mean(np.square(samples))))
```

**b) `the package does not import with heavy dependencies blocked`.** The clean
subprocess found a module that cannot be imported without your new package. The
failure text names the module and the exception. Reproduce it exactly:

```bash
python -c "
import sys
class B:
    def find_spec(self, n, p=None, t=None):
        if n.split('.')[0] in {'numpy','torch','yourpkg'}: raise ImportError(n)
sys.meta_path.insert(0, B())
import importlib, pkgutil, jarvis
for i in pkgutil.walk_packages(jarvis.__path__, 'jarvis.'):
    try: importlib.import_module(i.name)
    except Exception as e: print(i.name, type(e).__name__, e)
"
```

Add the new package's import name to `HEAVY` in `tests/test_import_hygiene.py` so
the block list covers it in future.

**c) `uses annotations without 'from __future__ import annotations'`.** Add that
line at the top of the module. It is what keeps the code valid on Python 3.9.

Then check the rest:

```bash
python -m pytest tests -q
pip uninstall -y <newpkg> && python -m pytest tests -q && pip install <newpkg>
```

That last line is the one people skip. The suite must pass **without** the new
dependency, because that is the state of a fresh machine.

Finally, register the dependency where it is discoverable — `requirements.txt` or
`requirements-full.txt`, the right `[project.optional-dependencies]` group in
`pyproject.toml`, and a `cmd_doctor` group in `jarvis/cli.py:89` so
`jarvis doctor` names it with a purpose and an install line.

---

## When none of the above helps

```bash
jarvis -v doctor 2>&1 | tee jarvis-doctor.txt
tail -200 "$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/jarvis.log"
python -c "import json; from jarvis.core.config import load_config; print(json.dumps(load_config().to_dict(), indent=2))"
python -c "from jarvis.core.platform_utils import system_summary; print(system_summary())"
python -m pytest tests -q
```

Those five outputs — the doctor report, the tail of the log, the effective config,
the host summary, and whether the suite still passes — are enough to diagnose almost
anything in this codebase without access to the machine.
