# Operations

Running JARVIS day to day, on Windows and on Linux. Linux is the production
target; Windows remains fully supported and is where the code was developed.

Everything here was read out of the source. Where a figure is a measurement taken
elsewhere in the repo it is cited as such; where something is untested it says so.

---

## 1. First run

```bash
# Windows
.\install.ps1                 # -Profile min|lean|full
# Linux
./install.sh                  # --min | --lean | --full   (default: lean)
./install.sh --help           # the authoritative flag list
```

Both scripts create `.venv`, install the chosen profile, run `python -m jarvis
setup` (which creates the data directories and optionally downloads the Piper
voice), and write a launcher — `jarvis.bat` on Windows, `./jarvis` on Linux.

`install.sh` additionally identifies the distribution, checks for `ffmpeg`,
`espeak-ng` and `libportaudio`, and **prints** the right `apt-get` / `dnf` /
`pacman` line for whatever is missing. It does not install them: nothing in that
script runs `sudo`, and nothing outside the project directory is touched. Its own
header documents the extra flags — `--no-voice`, `--venv PATH`, `--service`
(install and enable the systemd user unit), `--vllm`, `--model ID`. Read
`head -20 install.sh` before running it; it changes more often than this document.

Then, always:

```bash
jarvis doctor
```

`cmd_doctor` (`jarvis/cli.py:80`) prints the host summary, the data directory, a
present/absent line for every optional dependency grouped by what it unlocks,
then actually **boots every subsystem** and prints `Subsystems.status()`. It is
the single most useful command in the project and the correct first move whenever
anything misbehaves.

### The full command set

| Command | What it does |
|---|---|
| `jarvis` | voice mode if `AudioRecorder(cfg.stt).is_available()`, else text chat (`cli.py:443`) |
| `jarvis chat` | interactive text conversation; wires an interactive confirmation prompt into the security gate |
| `jarvis voice` | hands-free loop |
| `jarvis ask "..."` | one question, one answer, exit |
| `jarvis say ["..."] [-o FILE]` | audition the voice, or write an audio file |
| `jarvis doctor` | dependency + subsystem report |
| `jarvis tools` | list every registered tool with its parameters |
| `jarvis memory [QUERY] [-n N] [--stats] [--export PATH]` | inspect long-term memory |
| `jarvis config [--write [PATH]]` | print or write the effective configuration |
| `jarvis setup [--no-download]` | create the directories, fetch the voice model, verify |

Global flags: `-c/--config PATH`, `-v/--verbose` (DEBUG logging), `--model ID`,
`--backend NAME`, `--no-speech`.

---

## 2. The data directory

`jarvis config` prints `data_dir` at the bottom of its JSON; `jarvis doctor` prints
it near the top. Resolved by `jarvis/core/platform_utils.py:107 data_dir()`:

| Platform | Data directory | Config directory |
|---|---|---|
| Windows | `%LOCALAPPDATA%\Jarvis` | `%APPDATA%\Jarvis` |
| Linux | `$XDG_DATA_HOME/jarvis`, else `~/.local/share/jarvis` | `$XDG_CONFIG_HOME/jarvis`, else `~/.config/jarvis` |
| macOS | `~/Library/Application Support/Jarvis` | same |

Both are overridable: `JARVIS_HOME` and `JARVIS_CONFIG_DIR`. `config.data_dir`
(the config field) overrides `JARVIS_HOME` for everything routed through
`Config.home()`.

Layout under the data directory (`Config.path()` and friends, `core/config.py:229`):

```
<data dir>/
  memory.db            SQLite store  — precious
  memory.db-wal        WAL sidecar   — part of the database
  memory.db-shm        shared-memory index — regenerated, safe to lose
  tools/               generated tool modules (*.py) — precious
  voices/              downloaded Piper .onnx + .onnx.json — replaceable
  models/              AirLLM layer-shard cache (or llm.layer_shards_dir) — replaceable
  logs/
    jarvis.log         rotating, 5 MB × 5 backups (core/logging_setup.py:98)
    audit.jsonl        security decisions, append-only
```

Config file resolution (`core/config.py:405 default_config_paths()`), **ascending**
priority — later files overwrite earlier ones, then environment variables win over
all of them:

1. `<config dir>/config.yaml`, `.yml`, `.json`
2. `./config.yaml`, `.yml`, `.json` (the current working directory)
3. `$JARVIS_CONFIG` if set

Environment overrides are `JARVIS_<SECTION>_<FIELD>`, e.g.
`JARVIS_LLM_BACKEND=vllm`, `JARVIS_SECURITY_MODE=guarded`, `JARVIS_LOG_LEVEL=DEBUG`.

> A key that is not a field on the corresponding dataclass is **silently ignored**
> (`core/config.py:316 _apply_mapping()` skips anything failing `hasattr`). If a
> setting appears to do nothing, that is the first thing to check:
> `python -c "import json; from jarvis.core.config import load_config; print(json.dumps(load_config().to_dict(), indent=2))"`
> shows exactly what was actually absorbed. See ARCHITECTURE.md §8 for the known
> cases where `config.example.yaml` documents fields that do not exist.

---

## 3. Serving the model

This is the decision that determines whether JARVIS is usable. Five backends are
registered; check yours with:

```bash
python -c "from jarvis.llm import BACKENDS, AUTO_PROBE_ORDER; print(sorted(BACKENDS)); print(AUTO_PROBE_ORDER)"
```

### Measured throughput

These figures come from `README.md` and `requirements-full.txt`, measured on the
target class of machine — i5-10210U, 32 GB, CPU only, no usable CUDA. They are the
project's own numbers, not vendor claims.

| Backend | Model | Throughput | Fits in RAM? |
|---|---|---|---|
| AirLLM | Qwen3-30B-A3B | **~0.02–0.1 tok/s** (10–50 *seconds* per token) | No — streamed from disk |
| Ollama | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | **~15–25 tok/s** | Yes |
| Ollama | `qwen3:30b-a3b-instruct-2507-q4_K_M` (19 GB) | **~4–8 tok/s** | Yes |
| Ollama | `qwen3:32b-q4_K_M` (20 GB, dense) | ~1–2.5 tok/s | Yes |
| transformers | Qwen3-4B on CPU | <1 tok/s | Tight |

The uncomfortable conclusion, stated plainly in `requirements-full.txt`: **AirLLM's
purpose is running models that do not fit in RAM, and on a 32 GB machine the 30B
MoE already fits** as a 19 GB quantised Ollama model — roughly a hundredfold speed
difference for the same weights.

### Which one, and when

| Situation | Use | Why |
|---|---|---|
| Interactive voice on the CPU-only laptop | **Ollama** + `qwen3:4b-instruct-2507-q4_K_M` | 15–25 tok/s is the only setting that feels like conversation |
| Background subagents, quality matters more than latency | **Ollama** + `qwen3:30b-a3b-instruct-2507-q4_K_M` | 4–8 tok/s is fine for work nobody is waiting on |
| A tree of agents running concurrently | **vLLM** (or Ollama with `OLLAMA_NUM_PARALLEL>1`) | continuous batching: N agents cost far less than N × one agent |
| A machine with a real GPU | **transformers** + `Qwen/Qwen3-8B` or larger, or vLLM | weights resident in VRAM |
| Model genuinely too large for RAM (70B+ dense) | **AirLLM** | this is the one case where it wins |
| Tests, or "does it boot at all" | **stub** | zero dependencies, always available |

### Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows
winget install Ollama.Ollama

ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve                                   # if it is not already running
```

```yaml
llm:
  backend: ollama
  ollama_host: http://127.0.0.1:11434
  ollama_model: qwen3:4b-instruct-2507-q4_K_M
  context_tokens: 8192
```

`OllamaBackend` always sends `num_ctx` explicitly (`llm/ollama_backend.py:70`)
because Ollama otherwise defaults to a 4096-token window and **silently truncates**
the prompt. If replies start ignoring earlier context, check `context_tokens`
against the model's real limit before blaming the model.

Verify:

```bash
curl -s http://127.0.0.1:11434/api/tags
python -c "from jarvis.core.config import load_config; from jarvis.llm.ollama_backend import OllamaBackend; b=OllamaBackend(load_config().llm); print(b.is_available(), b.list_models())"
```

### vLLM

vLLM is the right answer for the agent tree, and the reason is throughput under
concurrency rather than single-request latency: one resident copy of the weights
serves many simultaneous requests through continuous batching, so a main agent plus
two layers of subagents costs far less than running them one after another.

**vLLM is Linux-only.** There is no supported Windows build. On Windows, run the
server under WSL2 or point `llm.vllm_host` at another machine — `VLLMBackend` is a
pure HTTP client and works fine from Windows either way
(`jarvis/llm/vllm_backend.py:13-27`).

```bash
pip install "vllm>=0.6.0"          # Linux
```

The exact argv for a server matching your config is generated for you:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.llm.vllm_backend import server_command
print(' '.join(server_command(load_config().llm)))
"
```

It produces `python -m vllm.entrypoints.openai.api_server --model <model> --host
<h> --port <p> --max-model-len <context_tokens> --max-num-seqs <batch>`, plus
`--device`, `--download-dir` and `--api-key` when the config asks for them
(`vllm_backend.py:92`). **Nothing in JARVIS ever launches that process** — you run
it, under systemd or a terminal multiplexer, and JARVIS connects to it.

```yaml
llm:
  backend: vllm
  vllm_host: http://127.0.0.1:8000/v1     # note the /v1 suffix
  api_key: ""                             # only if you started vLLM with --api-key
  max_concurrent_requests: 8
```

Health check, which never raises:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.llm.vllm_backend import VLLMBackend
import json; print(json.dumps(VLLMBackend(load_config().llm).health(), indent=2))
"
```

`503` from vLLM means "still loading weights" — the client retries with capped
exponential backoff and jitter, which is exactly the case where a fleet of agents
all start at once (`llm/openai_compat.py:364 _open_with_retries()`).

> **Unverified.** No vLLM server has ever been run against this codebase by the
> author of these docs. The client is tested against fakes; the wire format is the
> standard OpenAI one. Treat the first real vLLM run as a bring-up exercise, not
> as a regression.

### Any other OpenAI-compatible server

`openai-compat` is the same client pointed elsewhere: llama.cpp's `llama-server`,
LM Studio, text-generation-inference, a hosted endpoint, or **Ollama's own `/v1`
shim** on `http://127.0.0.1:11434/v1`. That last one is worth knowing: it gets you
Ollama's convenience with the concurrency-aware client and the in-flight cap.

### transformers

In-process weights via `AutoModelForCausalLM`. Viable on the target laptop only for
small Qwen3 checkpoints (~4B and below). `TransformersBackend` picks the device
(`cuda` → `mps` → `cpu`), picks bfloat16 on CPU/CUDA and float16 on MPS, aliases a
missing pad token to EOS, and runs real token streaming on a worker thread with an
explicit timeout so a dead worker cannot block the consumer forever
(`llm/transformers_backend.py:179`).

### AirLLM

Streams one transformer layer at a time from disk. Two version-specific gotchas are
worked around in `llm/airllm_backend.py`: `AutoModel.from_pretrained` defaults to
`device='cuda:0'` and crashes on a CPU-only machine, and `max_seq_len` defaults to
512 and silently caps the context. Both are always passed. Loading retries with
progressively fewer kwargs so an older `airllm` that does not understand
`compression` or `layer_shards_saving_path` still loads.

Use it only for models that genuinely do not fit in RAM.

### The concurrency story, concretely

An agent tree issues many simultaneous generation calls. What each backend does
with that:

| Backend | Concurrent behaviour |
|---|---|
| vLLM / openai-compat | Continuous batching server-side. Client-side, `llm.max_concurrent_requests` (default 8) is a `threading.Semaphore` held for the whole request including a streamed response. `0` = unlimited |
| Ollama | Serialises by default. Set `OLLAMA_NUM_PARALLEL` (and `OLLAMA_MAX_LOADED_MODELS`) in the daemon's environment to get parallelism |
| transformers | One `generate()` at a time in practice; `torch` will happily oversubscribe the CPU and make everything slower |
| AirLLM | Effectively serial, and each request is already 10–50 s/token |

`llm.max_concurrent_requests` is deliberately framed as **resource management, not
permission**: it stops a runaway tree from opening a thousand sockets and exhausting
file descriptors. It is enforced only by the HTTP backends; Ollama, transformers and
AirLLM ignore it.

---

## 4. Backup and restore

### What is precious

| Path | Why | Replaceable? |
|---|---|---|
| `<data dir>/memory.db` (+ `-wal`) | Every conversation, fact, summary and task report ever recorded. Nothing recreates it | **No** |
| `<data dir>/tools/*.py` | Tools JARVIS wrote for itself. Regenerating them means a different model on a different day writing different code | **No** |
| `<config dir>/config.yaml` (or `./config.yaml`) | Your settings | No, but small |

### What is disposable

`<data dir>/models/` (AirLLM shard cache — re-downloaded), `<data dir>/voices/`
(re-downloaded by `jarvis setup`), `<data dir>/logs/` (diagnostic history only,
though `audit.jsonl` may be worth keeping), `memory.db-shm`, `.venv`, `__pycache__`,
the Hugging Face cache.

### Backing up

The database is opened in WAL mode, so **do not copy `memory.db` alone from a
running system** — the WAL may hold committed data. Two safe routes:

```bash
# 1. Logical export — portable, diffable, survives a schema change.
jarvis memory --export ~/jarvis-backup/memory-$(date +%F).jsonl

# 2. Physical copy — stop JARVIS first, then take all three files.
cp "$(python -c 'from jarvis.core.config import load_config as l; print(l().db_file())')"* \
   ~/jarvis-backup/
```

A complete backup, JARVIS not running:

```bash
DATA=$(python -c "from jarvis.core.config import load_config as l; print(l().home())")
tar czf jarvis-backup-$(date +%F).tar.gz \
    -C "$DATA" memory.db memory.db-wal tools \
    -C "$(python -c 'from jarvis.core.platform_utils import config_dir; print(config_dir())')" .
```

PowerShell equivalent:

```powershell
$Data = python -c "from jarvis.core.config import load_config as l; print(l().home())"
Compress-Archive -Path "$Data\memory.db","$Data\memory.db-wal","$Data\tools" `
                 -DestinationPath "jarvis-backup-$(Get-Date -f yyyy-MM-dd).zip"
```

### Restoring

```bash
# Logical: import into a fresh or existing database. Idempotent by record id,
# so re-importing the same file twice does not duplicate anything.
python -c "
from jarvis.core.config import load_config
from jarvis.memory import create_memory
cfg = load_config(); cfg.memory.db_path = str(cfg.db_file())
s = create_memory(cfg.memory); print(s.import_jsonl('memory-2026-08-08.jsonl')); s.close()
"

# Physical: stop JARVIS, restore all of memory.db / -wal / tools/, start again.
```

`import_jsonl` re-adds through `add()`, which is first-write-wins by id
(`memory/store.py:200`), so restoring over a live database merges rather than
overwrites. Note that embeddings are **not** in the JSONL export — they are
recomputed on import from whatever embedder is configured at that moment. If you
restore with a different `memory.embedder`, keyword search still works and vector
recall re-indexes; the scores will not match the old ones.

Sanity-check a restore:

```bash
jarvis memory --stats
```

---

## 5. Upgrading, and rolling back

There is no migration framework. `SCHEMA_VERSION = 1` (`memory/store.py:40`) is
written once into a `schema_version` table and **nothing reads it back** — schema
creation is `CREATE TABLE IF NOT EXISTS`. That is fine today because the schema has
never changed; it means a future schema change needs a migration path written by
hand, and the version row is the hook for it.

Upgrade procedure:

```bash
# 1. Back up first — memory.db and tools/ are the irreplaceable parts.
jarvis memory --export ~/jarvis-backup/pre-upgrade.jsonl

# 2. Note what you are on.
git rev-parse HEAD > ~/jarvis-backup/version.txt
pip freeze > ~/jarvis-backup/requirements-frozen.txt

# 3. Update the code and reinstall in place.
git pull
pip install -e .            # or: pip install -r requirements.txt

# 4. Prove it still works before trusting it.
python -m pytest tests -q
jarvis doctor
jarvis ask "what operating system am I running?"
```

Rolling back:

```bash
git checkout "$(cat ~/jarvis-backup/version.txt)"
pip install -e .
```

Two rollback hazards, both real:

* **Generated tools are written against the contracts of the version that wrote
  them.** A tool in `<data dir>/tools/` that imports something the old code does not
  have will fail to load. `ToolRegistry.load_generated()` logs a warning and skips
  it rather than failing the boot (`tools/registry.py:516`), so the symptom is "a
  tool quietly disappeared", not a crash. Check the log.
* **A newer database opened by older code** is fine today (the schema has not
  changed) but is exactly what the unused `schema_version` row is there to protect
  against later. Restore from the JSONL export if in doubt.

Environment pinning is worth doing on the Linux box: `pip freeze` before every
upgrade, so a bad `transformers` or `faster-whisper` release can be reverted
independently of JARVIS itself.

---

## 6. Resource management on a small box

None of these are permission controls. They exist so that unbounded recursion or an
unbounded fan-out cannot take the machine down.

| Setting | Default | Effect | Turn it down when |
|---|---|---|---|
| `agent.max_concurrent_tasks` | 4 | Worker threads **per depth level**. Extra tasks queue. Thread ceiling is `(max_depth + 1) × this` = 16 | Fewer than 4 physical cores, or the model is the bottleneck (it usually is) — try 2 |
| `max_depth` (`DEFAULT_MAX_DEPTH`) | 3 | How far delegation may recurse; roots are depth 0. Beyond it, `spawn` returns an already-failed task explaining the limit | A small model over-delegates — 1 or 2 keeps the tree flat. **Not a config field yet**, see below |
| `max_total_tasks` (`DEFAULT_MAX_TOTAL_TASKS`) | 64 | Tasks tracked at once, after reaping announced leaves | Memory pressure. **Not a config field yet** |
| `agent.max_tool_iterations` | 8 | Reason-act steps for the main agent | Replies take too long; each iteration is a full generation |
| (derived) subagent iterations | `max(4, max_tool_iterations * 2)` = 16 | Steps a subagent gets | Lower `max_tool_iterations`; there is no separate knob |
| `agent.subagent_timeout` | 900 s | Wall clock per task | A task that has run 15 minutes on this hardware is stuck |
| `llm.max_concurrent_requests` | 8 | In-flight HTTP generation calls (vLLM / openai-compat only) | CPU-only: 2–4. `0` disables the cap |
| `llm.max_new_tokens` | 512 | Cap per reply | Spoken replies are 2–3 sentences; 256 is plenty and halves the wait |
| `llm.context_tokens` | 8192 | Prompt window; sent to Ollama as `num_ctx` | RAM pressure — the KV cache scales with it |
| `memory.recall_k` | 8 | Recollections injected per turn | Prompts are long; each recollection is a full stored record |
| `memory.keep_recent_turns` | 8 | Live turns kept verbatim | Same |
| `memory.summarize_after_turns` | 20 | When the window is compressed | Lower it to keep prompts short |
| `stt.model` | `small.en` | Whisper size | `tiny.en` if transcription lags; `small.en` if accuracy matters more than latency |
| `stt.compute_type` | `int8` | Whisper quantisation | Already the fast setting; leave it |

A conservative CPU-only Linux profile:

```yaml
llm:
  backend: ollama
  ollama_model: qwen3:4b-instruct-2507-q4_K_M
  max_new_tokens: 256
  context_tokens: 4096
  max_concurrent_requests: 2
agent:
  max_concurrent_tasks: 2
  max_tool_iterations: 6
memory:
  recall_k: 5
```

`max_depth` and `max_total_tasks` are read through
`getattr(config.agent, "max_agent_depth", ...)` and
`getattr(config.agent, "max_total_tasks", ...)`, but **neither is a field on
`AgentConfig`**, so today they are effectively constants. To change them, either add
the fields or construct the `TaskManager` yourself and pass it to `Orchestrator`.
Check the live values:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.agent.task_manager import TaskManager
tm = TaskManager(max_workers=load_config().agent.max_concurrent_tasks)
print(tm.stats()); tm.shutdown(wait=False)
"
```

### What is *not* bounded

Documented honestly so nobody assumes protection that is not there:

* **A tool blocked inside one call cannot be cancelled or timed out.** Cancellation
  and the task timeout are both checked inside `progress()`, which only runs
  between steps (`agent/task_manager.py:356`).
* **`ToolRegistry.run(timeout=...)` abandons rather than kills.** Python cannot kill
  a thread; the call returns a timeout failure while the thread keeps running
  (`tools/registry.py:452`).
* **Memory grows without bound.** There is no pruning, no TTL and no vacuum. `prune`
  exists in the config and is never read. Expect the database to grow roughly with
  total conversation text; watch it with `jarvis memory --stats`.
* **Search is a full scan.** Vector recall compares against every stored embedding
  on every query. Fine at tens of thousands of records; it is not an ANN index.

Practical monitoring on the Linux box:

```bash
watch -n5 'jarvis memory --stats; du -sh "$(python -c "from jarvis.core.config import load_config as l; print(l().home())")"'
tail -f "$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/jarvis.log"
```

---

## 7. The audit log

Location: `<data dir>/logs/audit.jsonl` — `app.build()` passes
`cfg.logs_dir() / "audit.jsonl"` explicitly (`jarvis/app.py:147`); a `SecurityGate`
constructed without an `audit_path` falls back to `data_dir()/logs/audit.jsonl`
(`core/security.py:664`). Controlled by `security.audit_log` (default `true`).

Format: one JSON object per line.

```json
{"ts": 1786000000.0, "action": "tool", "detail": "run_command (open mode)", "allowed": true}
```

| Field | Meaning |
|---|---|
| `ts` | `time.time()` float |
| `action` | `"tool"`, `"unattended-approved"`, `"unattended-denied"`, or whatever a caller passes |
| `detail` | Human-readable subject — the tool name, or the confirmation message |
| `allowed` | The verdict |

What actually gets written in the shipped configuration: `check_tool` audits every
call in open mode (`security.py:556`), and `confirm()` audits unattended
approvals and denials. `check_path` and `check_command` do not write audit records
themselves — they return a `Decision`, and it is `check_tool` that journals.

**It is a record, not a restriction.** `audit()` never blocks an action, never
raises, and a single write failure disables the sink for the rest of the process
rather than poisoning every subsequent check with log spam
(`core/security.py:666`). Nothing rotates it — unlike `jarvis.log`, which rotates
at 5 MB × 5 backups — so on a long-lived Linux install, rotate it yourself:

```bash
# /etc/logrotate.d/jarvis
/home/YOU/.local/share/jarvis/logs/audit.jsonl {
    weekly
    rotate 12
    compress
    missingok
    notifempty
    copytruncate
}
```

Reading it:

```bash
AUDIT="$(python -c 'from jarvis.core.config import load_config as l; print(l().logs_dir())')/audit.jsonl"

tail -f "$AUDIT"                                            # live
python - "$AUDIT" <<'EOF'                                   # refusals only
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if not r["allowed"]:
        print(r["ts"], r["action"], r["detail"])
EOF
```

If you want the audit trail to be *complete* rather than policy-shaped, note that
`ToolRegistry.history` (`tools/registry.py:403`) records every dispatch — name,
cleaned kwargs, ok, duration, error — in memory for the life of the process. That
is the richer record; it simply is not persisted today.

---

## 8. Running as a service on Linux

`jarvis/linux/` provides this: `service` (systemd **user** unit), `desktop`
(notifications, XDG autostart, window control, an honest Wayland answer), and
`audio` (PipeWire/PulseAudio/ALSA detection with the right install command for the
distro). It is **not wired into the CLI** — `grep -n linux jarvis/cli.py` returns
nothing — so drive it from Python for now.

One diagnostic covers all three:

```bash
python -c "import json; from jarvis.linux import is_linux_integration_available as w; print(json.dumps(w(), indent=2, default=str))"
```

### Why a *user* unit and never a system one

A system unit runs as root outside any login session: no `XDG_RUNTIME_DIR`, no
PipeWire/PulseAudio socket, therefore no microphone and no speakers — the exact
opposite of what a voice assistant needs. Everything in `jarvis/linux/service.py`
writes to `~/.config/systemd/user/` and talks to `systemctl --user`. It never runs
`sudo`.

### Installing

```bash
python -c "
from jarvis.linux import service
import json
print(json.dumps(service.install().output, indent=2))
"
```

`install()` writes `~/.config/systemd/user/jarvis.service` **and runs
`systemctl --user daemon-reload`** — systemd caches unit files, so a new unit is
invisible without it, and the reload is therefore part of installing rather than a
step to forget. It refuses a relative `ExecStart` (systemd rejects those) and
returns the unit path plus the next steps. It does not start or enable anything.

```bash
systemctl --user enable --now jarvis.service
loginctl enable-linger "$USER"        # see below — this is the one that bites
```

### Lingering: the failure everybody hits

A user manager starts when you log in and is torn down when your last session ends.
Close the lid, log out, or reboot to the login screen and your "always on" assistant
is simply gone, **with no error logged anywhere**. `loginctl enable-linger $USER`
tells systemd to start your user manager at boot and keep it after logout.

`service.status()` detects the current state with `loginctl show-user` and leads its
`advice` list with this warning when lingering is off. The module deliberately never
runs the command for you — it is the one step that changes system state outside your
home directory.

The unit is installed `WantedBy=default.target`, not `graphical-session.target`,
precisely so that lingering is sufficient: `default.target` is reached whenever the
user manager starts, desktop login or not.

### Operating it

```bash
python -c "import json; from jarvis.linux import service; print(json.dumps(service.status(), indent=2, default=str))"

systemctl --user status jarvis.service
systemctl --user restart jarvis.service
journalctl --user -u jarvis.service -n 50 -f
```

Equivalent Python wrappers exist — `service.enable()`, `disable()`, `start()`,
`stop()`, `restart()`, `logs(lines=50)`, `uninstall()` — each returning a
`ToolResult`. `logs()` is capped at `MAX_LOG_LINES = 5000`, framed as resource
management: an unbounded journal dump would pin megabytes in memory and, worse, be
fed to a language model.

### Ordering and environment

The generated unit is ordered `After=` the sound services. Two things you will
likely want to add by hand in `unit_text()` or a drop-in:

* `Environment=JARVIS_HOME=...` if you are not using the default data directory.
* Ordering after your model server (`After=ollama.service`, or your own vLLM unit),
  since JARVIS starts fine without it but will fall back to the stub backend.

### Session autostart (the other mechanism)

`jarvis/linux/desktop.py` writes an XDG `.desktop` entry into
`$XDG_CONFIG_HOME/autostart` — that is a *desktop session* autostart, not a service:
it starts when you log into a graphical session and dies with it. Use it when you
want JARVIS present only while you are at the desk; use the systemd user unit plus
lingering when you want it always on.

```bash
python -c "from jarvis.linux import desktop; print(desktop.autostart_path(), desktop.autostart_is_enabled())"
```

Note that `jarvis/win/autostart.py` also writes a `.desktop` file at the same path
when it runs on Linux — deliberately the same filename, so the two mechanisms cannot
produce two competing entries.

### Audio on the Linux box

```bash
python -c "import json; from jarvis.linux import audio; print(json.dumps(audio.check(), indent=2, default=str))"
python -c "from jarvis.linux import audio; print(audio.backend(), audio.default_input(), audio.default_output())"
python -c "from jarvis.linux import audio, install_command; print(install_command(['portaudio']))"
```

`audio.check()` reports the sound server in use, input/output devices, whether
PortAudio is installed, and whether your user is in the `audio` group — and gives the
exact package-manager command for the missing pieces on this distro.

> **Unverified.** `jarvis/linux/` is written and unit-tested on the Windows
> development box against a faked `systemctl`, faked `pactl` and faked `wmctrl`. The
> unit text, argument lists and control flow are verified; the behaviour of real
> systemd, real PipeWire and a real window manager is not. Treat the first
> `service.status()` on the Linux laptop as the acceptance test.
