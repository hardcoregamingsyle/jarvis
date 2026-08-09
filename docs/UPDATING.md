# Updating JARVIS

> **"Will running the installer again update all of it?"**
>
> **Yes.** `./install.sh` pulls the code, upgrades every Python package,
> upgrades Ollama itself, re-checks the model weights against the registry,
> installs anything missing, and prints a per-component summary of what moved.
>
> The one thing still checked for presence rather than freshness is the
> **speech models**, and that is correct rather than a shortcut: Piper voices
> and Whisper checkpoints are versioned *in their filenames*
> (`en_GB-alan-medium`, `base.en`), so a file that is present is by definition
> the right one. Model weights are different — an Ollama tag like
> `qwen3.6:27b` is a moving pointer, re-aimed upstream when the model is
> re-quantised or fixed — which is why those are re-checked properly.

```bash
cd /path/to/jarvis
./install.sh
```

Nothing else is needed. The run ends with a summary like this:

```
==> Summary
    Repository         updated          a1b2c3d4..e5f6a7b8
    System libs        already current  all present
    Python packages    updated          lean profile, --upgrade
    config.yaml        updated          llm.backend=ollama, llm.ollama_model=qwen3.6:27b
    Ollama runtime     already current  /home/hp/.local/share/jarvis/ollama
    Ollama server      already current  already listening
    Main model         already current  qwen3.6:27b
    Fast model         already current  qwen3:4b-instruct-2507-q4_K_M
    Speech assets      already current  voice + transcription
    Launcher           already current  /home/hp/.local/bin/jarvis
```

`updated` / `already current` / `skipped` / `failed` are decided from the
`action` each `jarvis.runtime` call reports — `present`, `current`, `skipped`,
`external` and `already-running` mean nothing moved; `pulled`, `fetched`,
`downloaded`, `installed`, `upgraded` and `written` mean it did. Two rows are
less precise than they look: **`config.yaml` always says `updated`**, because
the file is rewritten on every run whether or not the keys changed, and a
freshly downloaded Piper voice can be reported as `already current` (see
[INSTALL.md § Known gaps](INSTALL.md#known-gaps-in-this-release)).

---

## Contents

- [What a re-run updates](#what-a-re-run-updates)
- [What a re-run deliberately leaves alone](#what-a-re-run-deliberately-leaves-alone)
- [The local-changes case](#the-local-changes-case)
- [Checking what is currently installed](#checking-what-is-currently-installed)
- [Rolling back](#rolling-back)
- [Before a risky update](#before-a-risky-update)

---

## What a re-run updates

| Component | Updated by a re-run? | How | How to force it |
|---|---|---|---|
| **The code** | **Yes**, when the tree is clean | `git pull --ff-only` on the current branch. If `install.sh` itself changed, the installer `exec`s the new copy with your original arguments | `git fetch && git reset --hard @{upstream}` — **discards local work** |
| **Python packages** | **Yes** | `pip install --upgrade -e .` then `pip install --upgrade -r requirements.txt`. The `--upgrade` is the whole difference: without it pip is satisfied by whatever is already installed | `.venv/bin/pip install -U <package>` |
| **Ollama itself** | **Yes**, when JARVIS installed it | `install_plan()` compares `ollama --version` against the latest GitHub release; `upgrade` downloads the new tarball and swaps the tree | `python -c "from jarvis.runtime import ollama; print(ollama.install(force=True))"` — but read the warning below first |
| **Ollama, system-wide copy** | **No, by design** | A binary on `PATH` outside JARVIS's tree reports `action="external"`; it belongs to your package manager or to you | `sudo apt-get upgrade ollama`, or however you installed it |
| **Model weights** | **Yes, on an update run** | `ensure_model(refresh=True)` re-issues the pull. Ollama compares digests and transfers only changed layers, so a current model costs a manifest. Reported as `updated` when bytes moved, `current` when none did | `ollama pull qwen3.6:27b`, or `./install.sh --no-update` to skip the check |
| **A model you added by hand** | **No** | The installer only ever touches `llm.ollama_model` and the small interactive tag | `ollama pull <tag>` |
| **The Piper voice** | **No, presence only** | Both `en_GB-alan-medium.onnx` and `.onnx.json` present ⇒ nothing happens | `rm ~/.local/share/jarvis/voices/en_GB-alan-medium.onnx*` then re-run |
| **The Whisper model** | **No, presence only** | A cache directory containing `model.bin`, `config.json` and `tokenizer.json` ⇒ nothing happens | `rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en` then re-run |
| **`jarvis.service`** | **Yes, with `--service`** | `jarvis.linux.service.install()` rewrites the unit and runs `systemctl --user daemon-reload` | `./install.sh --service` |
| **`jarvis-ollama.service`** | **Yes, with `--service`** | `install_service()` re-renders the unit from your config and rewrites `~/.config/systemd/user/jarvis-ollama.service` only when the text differs, then `daemon-reload` and `enable --now`. Edits you made by hand are overwritten — the unit's own header comment says so | `./install.sh --service` |
| **`config.yaml`** | **Three keys only** | `llm.backend`, `llm.ollama_model`, and `llm.model` when `--model` names a Hugging Face repo. Edited line by line — comments, ordering and every other setting survive | `./install.sh --model <id>` |
| **System packages** | **Never** | Detected and reported; installing one needs root | Run the printed `sudo` line yourself |

> **`install(force=True)` switches off both safety checks**, not just the
> "already current" one. It will replace `~/.local/share/jarvis/ollama` even
> when the directory carries no `.jarvis-managed` marker, and it will install a
> second, JARVIS-managed copy alongside a system-wide `ollama` that
> `install_plan()` had reported as `external`. Use it only on a tree you know is
> JARVIS's own; `./install.sh` never passes it.

### What re-checking the weights actually costs

Very little, when they are current. `ensure_model(refresh=True)` re-issues
`POST /api/pull`; Ollama fetches the manifest, compares digests against the
blobs already on disk, and transfers only the layers that differ. For an
unchanged model that is a few kilobytes. The outcome is reported honestly —
`updated` when bytes actually moved, `current` when none did — rather than
being inferred from the tag merely existing.

When the tag *has* moved, you get a real download. That is the point, but it is
worth knowing about on a metered connection: it is the same bytes `ollama pull`
would fetch, it streams visible progress, and it resumes if interrupted. If you
would rather that never happen unasked:

```bash
./install.sh --no-update
```

which checks presence only and changes no versions of anything.

Speech models stay presence-checked because their versions live in their
filenames: `en_GB-alan-medium.onnx` present *is* `en_GB-alan-medium` current.
There is nothing a freshness check could tell you that the filename does not.

### Updating without changing versions

```bash
./install.sh --no-update
```

This is the repair mode: no `git pull`, no `--upgrade` on pip, and Ollama is
left at whatever version is installed. Anything **missing** is still installed —
a deleted venv is rebuilt, an absent model is pulled. Use it when you need the
install fixed but the versions frozen, for instance after a bad `transformers`
release.

---

## What a re-run deliberately leaves alone

These are your data, not build artefacts. The installer never reads, moves,
rewrites or deletes any of them.

| Path | What it is | Why it is untouchable |
|---|---|---|
| `~/.local/share/jarvis/memory.db` (+ `-wal`, `-shm`) | Every conversation JARVIS has ever had | `memory.prune` is `false` by design: recall spans the whole history. There is no second copy anywhere |
| `~/.local/share/jarvis/tools/` | Python modules JARVIS wrote for itself with `create_tool` | Generated, validated and registered at runtime. An installer that regenerated them would be inventing capabilities the assistant never decided it needed |
| `~/.local/share/jarvis/logs/` | `jarvis.log`, `audit.jsonl` | `audit.jsonl` is an append-only record of security decisions. Rotating it is a logrotate job, not an install step |
| `config.yaml`, apart from three `llm` keys | Your settings | The file is edited line by line inside the `llm:` block. Comments, key order, and every other section survive verbatim |
| `~/.ollama/models/` beyond the configured tags | Models you pulled yourself | Nothing is ever removed from the weight cache |
| `~/.local/bin/*` that JARVIS did not create | Somebody else's commands | Names are only replaced when they are a symlink into JARVIS's own tree, or carry JARVIS's copy marker. Anything else is reported as a conflict and left in place |
| `~/.local/share/jarvis/ollama/` without `.jarvis-managed` | A directory JARVIS did not create | The marker file is the whole check. Without it the install refuses and tells you to move the directory aside |

If you want a clean slate for one of them, delete it yourself and re-run. That
asymmetry is deliberate: creating is safe to automate, destroying is not.

---

## The local-changes case

```
==> Updating the checkout
    tracked files have local changes, so nothing was pulled.
    Commit them, or set them aside yourself, and re-run — 'git status'
    lists them. The installer will not discard your work to update.
```

The test is `git status --porcelain --untracked-files=no`: a modification to a
**tracked** file stops the pull. Untracked files do *not* — a stray note in the
directory is no reason to refuse every update, and git itself refuses a
fast-forward that would overwrite one. The rest of the install proceeds normally
— packages, runtime, models and services are all still updated — but the code
stays exactly as you left it.

The installer will not run `git stash`, `git checkout --`, `git reset --hard` or
`git clean`. Every one of those can destroy work that exists nowhere else, and
an installer is the wrong program to be making that decision at 2 a.m. Resolve
it yourself:

```bash
git status                       # see what is actually modified
git diff                         # read it

git stash push -m "wip"          # set it aside, recoverable with `git stash pop`
# or
git checkout -b my-changes && git commit -am "wip"
# or, if you are certain it is disposable:
git checkout -- .

./install.sh
```

Four other conditions skip the pull, each reported by name and none of them an
error:

| Reported | Meaning |
|---|---|
| `not a git checkout` | You obtained the source another way; update it the same way |
| `detached HEAD` | You are on a tag or a specific commit. Check out a branch to resume updating |
| `no upstream for <branch>` | The branch tracks nothing. `git branch --set-upstream-to=origin/main` |
| `git pull --ff-only returned N` | Offline, or the branch has diverged and needs a real merge |

### The re-exec

When the pull moves HEAD, `install.sh` may itself have been rewritten. Bash
reads a script incrementally, so continuing would execute a spliced mixture of
the old and new files. The installer therefore re-executes the new copy with the
arguments you originally gave it:

```
    updated main  a1b2c3d4..e5f6a7b8
    the installer itself changed — restarting it
```

The restart happens at most once per run, and the summary still reports the pull.

---

## Checking what is currently installed

```bash
# The code
git -C /path/to/jarvis log -1 --oneline
git -C /path/to/jarvis describe --tags --always

# Python packages
.venv/bin/pip list --outdated
jarvis doctor

# Ollama and the weights
ollama --version
ollama list
curl -s http://127.0.0.1:11434/api/tags

# Everything the runtime knows about, in one object
python -c "
import json
from jarvis.core.config import load_config
from jarvis import runtime
print(json.dumps(runtime.status(load_config()), indent=2, default=str))
"
```

`jarvis doctor` is the Python-side report: every optional dependency, whether it
is importable, what it unlocks, and the exact `pip install` for each gap, plus
the state of each subsystem and the data directory. It does **not** yet cover the
Ollama runtime or the model weights.

`runtime.status()` covers those. It is read-only — no downloads, no daemons
started, no files written — and each component reports itself even when the
others are broken, which matters because the machine you run it on is usually
the one where something is wrong.

```jsonc
{
  "ready": false,                       // true only when all four are in place
  "missing": ["model"],                 // which of ollama / model / voice / stt
  "ollama": {
    "installed": true,
    "version": "0.5.7",
    "managed": true,                    // false ⇒ a system-wide copy; not ours to update
    "runtime_dir": "/home/hp/.local/share/jarvis/ollama",
    "serving": true,
    "model": "qwen3.6:27b",
    "model_present": false,
    "models_cache_dir": "/home/hp/.ollama/models"
  },
  "voice": { "present": true,  "model_path": "…/voices/en_GB-alan-medium.onnx" },
  "stt":   { "present": true,  "repo": "Systran/faster-whisper-base.en" },
  "espeak": { "present": false }        // advisory only; Piper is the default voice
}
```

> `unit_installed` in that output refers to `jarvis-ollama.service` — the name
> `jarvis.runtime.ollama` manages and the name `./install.sh --service` writes.
> Some of the installer's console output still calls it "ollama.service"; the
> file on disk, and the name `systemctl --user` wants, is
> `jarvis-ollama.service`.

---

## Rolling back

### The code

```bash
cd /path/to/jarvis
git tag -l                          # or: git log --oneline -20
git checkout v1.1.0                 # a tag, or a commit sha
./install.sh --no-update            # reinstall in place, change no versions
```

A detached HEAD makes the pull skip on every subsequent run, so a checked-out
tag stays pinned until you `git checkout main` again. That is the intended
behaviour, not a bug — pinning by tag is the supported way to hold a version.

Two rollback hazards, both real:

- **Generated tools are written against the contracts of the version that wrote
  them.** A module in `~/.local/share/jarvis/tools/` that imports something the
  older code does not have will fail to load. The registry logs a warning and
  skips it rather than failing the boot, so the symptom is "a tool quietly
  disappeared", not a crash. Check `~/.local/share/jarvis/logs/jarvis.log`.
- **A newer database opened by older code** is fine today — the schema has never
  changed — but `SCHEMA_VERSION` exists precisely so that stops being true one
  day. If in doubt, restore from a JSONL export.

### Python packages

```bash
.venv/bin/pip install "transformers==4.57.0"     # one package
.venv/bin/pip install -r ~/jarvis-backup/requirements-frozen.txt   # the whole set
```

Freezing before each update is worth the two seconds — it lets a bad
`faster-whisper` or `transformers` release be reverted independently of JARVIS
itself. See [Before a risky update](#before-a-risky-update).

### An older model

Nothing is deleted from the weight cache, so the previous model is still there
if you pulled it:

```bash
ollama list
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M    # if it is not
```

Then point `config.yaml` at it:

```yaml
llm:
  backend: ollama
  ollama_model: qwen3:30b-a3b-instruct-2507-q4_K_M
```

or, for one shell:

```bash
export JARVIS_LLM_OLLAMA_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
jarvis chat
```

There is no `jarvis model` subcommand, and the global `--model` flag sets
`llm.model` rather than `llm.ollama_model` — with an Ollama tag already in the
config, `jarvis --model … chat` will not change which model answers. Use the
environment variable or edit the file.

To reclaim the space:

```bash
ollama rm qwen3.6:27b
```

### An older Ollama

The installer always targets the latest GitHub release, so rolling back means
placing the older tarball yourself:

```bash
systemctl --user stop jarvis-ollama.service

VERSION=0.5.7                       # the release you want
curl -LO "https://github.com/ollama/ollama/releases/download/v${VERSION}/ollama-linux-amd64.tgz"

rm -rf ~/.local/share/jarvis/ollama
mkdir -p ~/.local/share/jarvis/ollama
tar -C ~/.local/share/jarvis/ollama -xzf ollama-linux-amd64.tgz
echo "ollama ${VERSION} placed by hand" > ~/.local/share/jarvis/ollama/.jarvis-managed

systemctl --user start jarvis-ollama.service
ollama --version
```

The `.jarvis-managed` marker matters: without it the next `./install.sh` refuses
to touch the directory at all, because an unmarked tree is assumed to belong to
somebody else. With it, the next run will happily upgrade you again — so pass
`--no-update` while you want to stay pinned.

---

## Before a risky update

The two-minute version of [OPERATIONS.md § 5](OPERATIONS.md):

```bash
mkdir -p ~/jarvis-backup

# 1. The irreplaceable parts
jarvis memory --export ~/jarvis-backup/pre-update.jsonl
cp -r ~/.local/share/jarvis/tools ~/jarvis-backup/tools

# 2. What you are on now
git -C /path/to/jarvis rev-parse HEAD > ~/jarvis-backup/version.txt
.venv/bin/pip freeze                > ~/jarvis-backup/requirements-frozen.txt
ollama list                         > ~/jarvis-backup/models.txt
ollama --version                    > ~/jarvis-backup/ollama-version.txt

# 3. Update
./install.sh

# 4. Prove it still works before trusting it
python -m pytest tests -q
jarvis doctor
jarvis ask "what operating system am I running?"
```

Restoring the code is `git checkout "$(cat ~/jarvis-backup/version.txt)"` and
`./install.sh --no-update`. Restoring the packages is
`.venv/bin/pip install -r ~/jarvis-backup/requirements-frozen.txt`. The memory
export goes back in through `MemoryStore.import_jsonl()`, which is first-write-wins
by record id — so replaying the same file twice does not duplicate anything, and
restoring over a live database merges rather than overwrites. The exact command,
and the caveat that embeddings are recomputed rather than restored, are in
[OPERATIONS.md § 4](OPERATIONS.md).

---

## See also

- [INSTALL.md](INSTALL.md) — the full installation reference, and what is written where
- [OPERATIONS.md](OPERATIONS.md) — backups, the service, resource limits, the audit log
- [MODELS.md](MODELS.md) — choosing a model, quantisation, context versus RAM
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — when something does not work
