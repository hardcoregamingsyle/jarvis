# Installing JARVIS on Linux

```bash
git clone https://github.com/hardcoregamingsyle/jarvis
cd jarvis
./install.sh
```

That is the whole installation. One command produces a JARVIS that can actually
answer you: the Python package, the inference runtime (Ollama), the model
weights, the British voice and the speech-to-text model. Nothing is left as a
printed instruction for you to run afterwards.

Re-running the same command updates all of it. That is documented separately in
[UPDATING.md](UPDATING.md).

**The installer never runs `sudo`.** Where a system package is genuinely needed
it prints the exact command and leaves it to you. Ollama is installed from its
official release tarball into your home directory rather than through
`curl … | sh`, which requires root. See [No sudo](#no-sudo-and-the-packages-you-must-install-yourself)
and [Rootless Ollama](#rootless-ollama).

Budget **about 20 GB of free disk** for a default install and an hour on a slow
connection. Every large download is sized and space-checked *before* it starts.

---

## Contents

- [What `./install.sh` does, in order](#what-installsh-does-in-order)
- [Profiles and flags](#profiles-and-flags)
- [What gets written, and where](#what-gets-written-and-where)
- [No sudo, and the packages you must install yourself](#no-sudo-and-the-packages-you-must-install-yourself)
- [Rootless Ollama](#rootless-ollama)
- [Choosing the model](#choosing-the-model)
- [Running it as a service](#running-it-as-a-service)
- [Air-gapped and metered connections](#air-gapped-and-metered-connections)
- [Verifying the install](#verifying-the-install)
- [Uninstalling](#uninstalling)
- [Known gaps in this release](#known-gaps-in-this-release)

---

## What `./install.sh` does, in order

Each stage records one line in a summary table printed at the end, so the run
tells you what state you are in rather than making you read a wall of pip
output. The summary is printed even if a later stage dies.

| # | Stage | Disk cost | Refuses below |
|---|---|---|---|
| 1 | **Update the checkout** — `git pull --ff-only` | — | — |
| 2 | **Identify the distribution** — apt / dnf / pacman / zypper | — | — |
| 3 | **Find Python** — 3.9+, preferring 3.12 → 3.11 → 3.10 → 3.13 → `python3` | — | — |
| 4 | **Check system libraries** — prints the one `sudo` line you must run | — | — |
| 5 | **Detect the session type** — warns on Wayland | — | — |
| 6 | **Create the virtualenv** (`.venv`), guard against a venv with no pip | ~0.03 GB | — |
| 7 | **Upgrade `pip`, `setuptools`, `wheel`** | — | — |
| 8 | **Install the profile** — `pip install -e .` plus the profile's requirements | ~1 GB (lean) | 12 GB with `--full` |
| 9 | **Write the model into `config.yaml`** — `llm.backend`, `llm.ollama_model` | — | — |
| 10 | **Print the download plan** — every size, before anything downloads | — | — |
| 11 | **Install the Ollama runtime**, rootless | ~1.4 GB compressed (amd64), read from the GitHub release | 3 GB in the shell, then the real check: 4× the download (6× for a zstd asset) + 200 MB |
| 12 | **Start the Ollama server** — a pull goes through the running daemon | — | — |
| 13 | **Pull the main model** (default `qwen3.8:27b`) | ~18 GB | 21 GB |
| 14 | **Pull the small interactive model** (`qwen3:4b-instruct-2507-q4_K_M`) | ~2.4 GB | 4 GB |
| 15 | **Fetch the speech models** — Piper voice + faster-whisper | ~0.2 GB | 2 GB |
| 16 | **vLLM** — only with `--vllm`, only on Linux with an NVIDIA GPU | several GB | 10 GB |
| 17 | **Locate the launcher** — the console script `pip install -e .` produced | — | — |
| 18 | **Put `jarvis` on your PATH** — `~/.local/bin/jarvis` and `~/.local/bin/JARVIS` | — | — |
| 19 | **Install the systemd user services** — only with `--service` | — | — |
| 20 | **Verify** — prints the runtime status, then the summary | — | — |

Notes on the stages that surprise people:

**Stage 1 re-executes the installer.** `git pull --ff-only` may rewrite
`install.sh` itself, and bash reads a script incrementally — continuing would
run a spliced mixture of both versions. When the pull moves HEAD, the installer
`exec`s the new copy with your original arguments. You will see
`the installer itself changed — restarting it`.

The pull is deliberately timid. It refuses, and changes nothing, when the
working tree is dirty, when HEAD is detached, when the branch has no upstream,
or when `--ff-only` would need a merge. It will not discard uncommitted work to
make an update succeed.

**Stage 4 never installs anything.** It probes for `ffmpeg`, `espeak-ng`,
`wmctrl`, `xdotool`, `notify-send`, `pactl`, `aplay`, PortAudio and
`python3-venv`, prints the exact package-manager line for whatever is missing,
and carries on. JARVIS installs either way; the affected features degrade rather
than crash.

**Stage 9 runs before any download**, so the pull in stage 13 and JARVIS at
runtime cannot disagree about which model this machine has. It sets
`llm.backend: ollama` and `llm.ollama_model: <tag>` inside the `llm:` block of
`config.yaml`, editing line by line — comments, ordering and every other setting
survive. `config.yaml` is created from `config.example.yaml` if it does not
exist.

**Stage 10 is the disk-space contract.** Sizes and free space are printed
before a byte moves, and each download refuses with a number rather than dying
at 90%. The size of a model comes from `jarvis.llm.models.estimate_footprint()`;
the size of the Ollama archive is read from the GitHub release API.

There are two checks per download, and the second is the one that counts. The
shell prints a cheap `df`-based estimate from a fixed table, then
`jarvis.runtime` re-checks against the real asset size before opening the
connection. The two do not always agree — the shell still quotes 3 GB for the
Ollama runtime, while the runtime's own rule (4× the download, 6× when it must
decompress zstd through an intermediate `.tar`, plus 200 MB) works out nearer
9 GB for a 1.4 GB asset. When they disagree the stricter one wins, and it
refuses with the actual numbers.

**Stage 12 exists because a model pull goes through the daemon**, not through
the binary. `POST /api/pull` needs something listening on
`http://127.0.0.1:11434`, and without it stages 13 and 14 would both fail with
`no-server`. If the daemon is already up — because you ran `--service` on an
earlier run, or started it yourself — the stage reports `already current` and
does nothing.

> Stage 12 uses `systemctl --user start jarvis-ollama.service` when that unit
> already exists, and otherwise spawns a detached `ollama serve` that the
> installer never stops. On the **first** `--service` run the unit does not
> exist yet at stage 12, so port 11434 is still held by that spawned process
> when stage 19 enables the unit, and the unit can fail to bind. It comes up
> correctly on the next reboot, and every later run takes the systemd path; to
> fix it immediately:
> `pkill -f "ollama serve" && systemctl --user restart jarvis-ollama.service`.

**Downloads resume.** The Ollama tarball is fetched with an HTTP `Range` request
against the partial file left by an interrupted run; `ollama pull` resumes from
the blobs already on disk. Nothing already current is re-fetched.

---

## Profiles and flags

```
./install.sh [FLAGS]
```

| Flag | What it changes |
|---|---|
| *(none)* | Install or update everything. Equivalent to `--update --lean`. |
| `--update` | The default, said out loud: `git pull`, `pip install --upgrade`, refresh the runtime and weights. |
| `--no-update` | Repair only. No `git pull`, no `--upgrade` on pip, no version bumps. Missing pieces are still installed. |
| `--lean` | **Default profile.** Voice, memory, machine control, config (~1 GB). |
| `--min` | The Python package only. No runtime, no weights, no speech models. |
| `--full` | Adds `torch`, `transformers`, `accelerate`, `safetensors`, `airllm`, `sentence-transformers`. Many GB; refuses below 12 GB free. |
| `--no-voice` | Skip the speech-model downloads (Piper voice, faster-whisper). |
| `--no-model` | Skip stages 11–14 entirely: no Ollama, no server, no weights. The speech stage still runs, but it is called with `skip=["ollama", "model"]`, so it downloads nothing large. |
| `--only-main-model` | Do not also fetch the small interactive model. |
| `--model ID` | Use `ID` as the main model. Accepts a Hugging Face repo id (`Qwen/Qwen3-4B-Instruct-2507`) or an Ollama tag (`qwen3:4b-instruct-2507-q4_K_M`). |
| `--venv PATH` | Put the virtualenv somewhere else. An absolute path is honoured as given. |
| `--service` | Also write and enable the systemd **user** units for Ollama and JARVIS. |
| `--vllm` | Set up the vLLM path. Linux-only, and only useful with an NVIDIA GPU. |
| `--no-link` | Do not create `~/.local/bin/jarvis`. |
| `-h`, `--help` | Print the header block of the script. |

`--model` decides two config keys. A value containing a colon is treated as an
Ollama tag and written to `llm.ollama_model` only; a value without one is a
Hugging Face repo id and is written to `llm.model` **and** resolved through the
catalogue into `llm.ollama_model`.

```bash
./install.sh --model qwen3:4b-instruct-2507-q4_K_M   # small model as the main one
./install.sh --model Qwen/Qwen3-30B-A3B-Instruct-2507 # the MoE, by repo id
./install.sh --full --service                         # everything, running at boot
./install.sh --no-update                              # fix a broken venv, change no versions
```

The installer also honours the usual XDG variables — `XDG_DATA_HOME`,
`XDG_CONFIG_HOME`, `XDG_BIN_HOME`, `XDG_CACHE_HOME` — plus `OLLAMA_MODELS` for
relocating the weight cache and `JARVIS_HOME` for relocating JARVIS's own data
directory.

---

## What gets written, and where

**This is the complete list.** Nothing outside the checkout and these paths is
created, modified or deleted, and no file the installer did not itself create is
ever overwritten — `~/.local/bin/jarvis` and `~/.local/bin/ollama` are left
untouched if something else already owns those names, and the managed Ollama
tree carries a `.jarvis-managed` marker file that must be present before a
re-run will replace it.

### Inside the checkout

| Path | What |
|---|---|
| `.venv/` | The virtualenv, including the `jarvis` console script (`--venv PATH` moves it) |
| `config.yaml` | Created from `config.example.yaml` on first run; only `llm.backend`, `llm.ollama_model` and (with `--model`) `llm.model` are rewritten |

### Outside the checkout

| Path | What | Written by |
|---|---|---|
| `~/.local/bin/jarvis` | Symlink to `.venv/bin/jarvis` | `install.sh` (`--no-link` to skip) |
| `~/.local/bin/JARVIS` | The same symlink under the capitalised name | `install.sh` |
| `~/.local/bin/ollama` | Symlink to the managed Ollama binary | `jarvis.runtime.ollama.link_binary()` |
| `~/.local/share/jarvis/ollama/` | The unpacked Ollama release, plus `.jarvis-managed` | `jarvis.runtime.ollama.install()` |
| `~/.local/share/jarvis/downloads/` | The release archive, kept so an interrupted download resumes. Named per version, and **never pruned** — after a few Ollama upgrades this is worth a look, at roughly 1.4 GB each. Deleting anything in here is always safe | `jarvis.runtime.ollama` |
| `~/.local/share/jarvis/voices/` | `en_GB-alan-medium.onnx` and `.onnx.json` | `jarvis.speech.tts.PiperTTS` |
| `~/.local/share/jarvis/` | `memory.db`, `tools/`, `models/`, `logs/` — **your data** | JARVIS at runtime |
| `~/.ollama/models/` | Model weight blobs and manifests | `ollama pull` |
| `~/.cache/huggingface/hub/` | The faster-whisper CTranslate2 model | `faster_whisper` |
| `~/.config/systemd/user/jarvis-ollama.service` | Ollama user unit. Named so it cannot collide with the `ollama.service` the official root installer writes | `jarvis.runtime.ollama.install_service()` (`--service`) |
| `~/.config/systemd/user/jarvis.service` | JARVIS user unit | `jarvis.linux.service` (`--service`) |

Every one of those paths follows the XDG variable that governs it, so
`XDG_DATA_HOME`, `XDG_BIN_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`,
`OLLAMA_MODELS` and `HF_HOME` all relocate the corresponding directory.

The installer's header comment says the Ollama tree goes to
`~/.local/share/ollama`. It does not: `jarvis.runtime.ollama.runtime_dir()`
resolves to `<data dir>/ollama`, which is `~/.local/share/jarvis/ollama`. The
table above is the real layout. Confirm it on your own machine with:

```bash
python -c "from jarvis.runtime import ollama; print(ollama.runtime_dir())"
```

---

## No sudo, and the packages you must install yourself

Installing a system package writes outside your home directory and needs root.
That is the one thing the installer will not do on your behalf — not out of
squeamishness, but because a script that silently acquires root to make its own
run succeed is a script you cannot audit by reading its output. So the installer
detects what is missing, prints the exact command for *your* distribution, and
carries on installing everything it can do without privilege.

Everything, in one line:

```bash
# Debian / Ubuntu
sudo apt-get install -y python3-venv portaudio19-dev ffmpeg espeak-ng \
    pulseaudio-utils alsa-utils libnotify-bin wmctrl xdotool ydotool

# Fedora / RHEL
sudo dnf install -y portaudio-devel ffmpeg-free espeak-ng \
    pulseaudio-utils alsa-utils libnotify wmctrl xdotool ydotool

# Arch
sudo pacman -S --needed portaudio ffmpeg espeak-ng libpulse \
    alsa-utils libnotify wmctrl xdotool ydotool

# openSUSE
sudo zypper install -y portaudio-devel ffmpeg espeak-ng \
    pulseaudio-utils alsa-utils libnotify-tools wmctrl xdotool ydotool
```

| Package | What you lose without it |
|---|---|
| `python3-venv` | The virtualenv itself. On Debian and Ubuntu `ensurepip` is a separate package, and without it `python3 -m venv` creates a tree with **no pip** — the installer detects exactly this and says so |
| `portaudio*` | The microphone. `import sounddevice` fails |
| `ffmpeg` | WAV conversion of edge-tts MP3 output (playback still works) |
| `espeak-ng` | The last-resort offline voice. Piper is the default, so this is not fatal |
| `pulseaudio-utils` (`pactl`) | Audio device enumeration and switching |
| `alsa-utils` | The ALSA playback fallback |
| `libnotify` (`notify-send`) | Desktop notifications when a background task finishes |
| `wmctrl`, `xdotool` | Window listing, focus, move, snap; keyboard and mouse injection — **X11 only** |
| `ydotool` | Keyboard and mouse injection under Wayland. Not probed for by the installer; install it only if you are on Wayland |

> Fedora ships `ffmpeg-free` in its own repositories. The full build needs RPM
> Fusion, which is not a third-party repository an installer should enable on
> your behalf.

If you are on Wayland, read the Wayland section of the [README](../README.md#the-wayland-caveat--read-this-one-too)
before assuming window control works. The installer prints the same warning when
it detects `XDG_SESSION_TYPE=wayland`.

---

## Rootless Ollama

Ollama is the inference runtime JARVIS talks to. It publishes an
OpenAI-compatible API on `/v1`, so `jarvis/llm/openai_compat.py` and
`jarvis/llm/vllm_backend.py` reach it with no adapter at all.

The official installer — `curl -fsSL https://ollama.com/install.sh | sh` —
**cannot be used here.** It needs root: it writes `/usr/local/bin/ollama`,
creates an `ollama` system user, and drops a system-wide systemd unit. So
`jarvis/runtime/ollama.py` does the same job without root.

| Step | What happens |
|---|---|
| **Discover** | `shutil.which("ollama")` first. A system-wide or self-installed Ollama on `PATH` wins and nothing is duplicated — the plan reports `action="external"` |
| **Resolve** | `GET https://api.github.com/repos/ollama/ollama/releases/latest` for the version, and the release asset's own byte size for the space check |
| **Check space** | Four times the download, plus twice again for the intermediate `.tar` when the asset is zstd-compressed, plus 200 MB of margin. The archive, the unpacked tree and the version being replaced all coexist for a moment |
| **Download** | `https://github.com/ollama/ollama/releases/download/v<version>/ollama-linux-<arch><suffix>` into `~/.local/share/jarvis/downloads/`, resuming a partial file with a `Range` request. A size mismatch deletes the partial file so a retry starts clean |
| **Extract** | Into a staging directory beside the target, with path-traversal filtering (`filter="data"` semantics, written out by hand for Python 3.9–3.11) |
| **Mark** | Write `.jarvis-managed` into the tree. Without this marker a later run will refuse to replace the directory |
| **Move** | Swap the staged tree into `~/.local/share/jarvis/ollama` |
| **Link** | `~/.local/bin/ollama` → the extracted binary. If something already owns that name and JARVIS did not create it, the conflict is reported and the file is left alone |

Only `x86_64`/`amd64` and `aarch64`/`arm64` are supported, because those are the
only Linux tarballs Ollama publishes. On anything else the plan returns
`action="unsupported"` and tells you to install Ollama from your distribution.

**`<suffix>` is not fixed.** Ollama moved its Linux assets from `.tgz` to
zstd-compressed tar, so the asset name is taken from the live release metadata
whenever `api.github.com` is reachable; `.tar.zst`, `.tgz` and `.tar.gz` are
tried in that order otherwise. Which one you get decides whether you need zstd:

| Your Python | What is needed |
|---|---|
| 3.14 or newer | Nothing — `tarfile` reads `.tar.zst` natively |
| 3.9 – 3.13 | `pip install zstandard` **(no root)**, or the `zstd`/`unzstd` command from your distribution's `zstd` package |

The installer detects all three and tells you which you have. Compression is
sniffed from the file's magic bytes, not from its name, so an upstream rename
cannot silently pick the wrong decompressor.

### The user service

`./install.sh --service` calls `jarvis.runtime.ollama.install_service()`, which
writes `~/.config/systemd/user/jarvis-ollama.service` and runs
`systemctl --user daemon-reload` followed by
`systemctl --user enable --now jarvis-ollama.service`.

The name matters: `ollama.service` is what the official *root* installer calls
its system unit, and using it here would make `systemctl status ollama`
ambiguous about which of the two you meant. Some of the installer's own console
output still says "ollama.service" — the file on disk is
`jarvis-ollama.service`, and that is the name `systemctl --user` wants.

It is a **user** unit, never a system one, for the same reason as
`jarvis.service`: a system unit runs as root outside any login session, so it has
no `XDG_RUNTIME_DIR`, no PipeWire or PulseAudio socket, and therefore no
microphone and no speakers. A user unit inherits all three.

The unit sets four environment variables that matter on this hardware:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `llm.max_concurrent_requests`, clamped to 1–32. The shipped default is `8`; a configured `0` ("unlimited") also becomes `8` | JARVIS runs a *tree* of subagents. Ollama's default of one in-flight request per model serialises the whole tree behind its slowest branch |
| `OLLAMA_MAX_LOADED_MODELS` | `1` for a model ≥ 6 GB, else `2` | Two 16 GB models in 32 GB of RAM means swapping, and a swapping language model is indistinguishable from a hang |
| `OLLAMA_KEEP_ALIVE` | `30m` | Re-reading 16 GB of weights from a laptop disk between utterances is a minute of silence |
| `OLLAMA_HOST` | from `llm.ollama_host`, falling back to `$OLLAMA_HOST` then `127.0.0.1:11434` | Keeps the server and the client on the same address |

Preview exactly what will be written, without writing it:

```bash
python -c "
from jarvis.core.config import load_config
from jarvis.runtime import ollama
print(ollama.service_unit_text(load_config()))
"
```

`StartLimitIntervalSec` and `StartLimitBurst` are in `[Unit]`, not `[Service]`.
systemd moved them in v230 and **silently ignores** them under `[Service]`, which
turns a crash loop into an infinite one.

### `loginctl enable-linger`

```bash
loginctl enable-linger "$USER"
```

A systemd **user manager** starts when you log in and is torn down when your last
session ends. Close the lid, log out, or reboot to the login screen and both
`jarvis-ollama.service` and `jarvis.service` are simply gone — with no error
logged anywhere. This is the single most common "it stopped working" report.

`enable-linger` tells systemd to start your user manager at boot and keep it
running after logout. Both units are installed `WantedBy=default.target` rather
than `graphical-session.target` precisely so that lingering is sufficient:
`default.target` is reached whenever the user manager runs, desktop login or not.

The installer prints the command and will not run it for you. It is the one step
that changes system state outside your home directory.

---

## Choosing the model

The default is `qwen3.8:27b`. It is the most capable thing that fits 32 GB at
Q4, and on a CPU-only i5-class laptop it is also the slowest thing you could
point a microphone at.

| Ollama tag | Download | Params | Active per token | CPU throughput | Thinks first? |
|---|---|---|---|---|---|
| `qwen3.8:27b` *(default)* | ~18 GB | 27B dense | **27B** | **~0.5-1 tok/s** | **Yes**, by default |
| `qwen3:30b-a3b-instruct-2507-q4_K_M` | ~18.3 GB | 30.5B MoE | **~3.3B** | ~4–8 tok/s | No |
| `qwen3:4b-instruct-2507-q4_K_M` | ~2.4 GB | 4B dense | 4B | ~15–25 tok/s | No |

**Active parameters per token is the number that decides latency, not total
parameters.** A dense model reads every weight for every token it produces, so
27B dense means 27 billion multiply-accumulates against RAM bandwidth per token
— and RAM bandwidth, not the CPU, is the wall on this machine. A
mixture-of-experts model of the same size routes each token through a small
fraction of its weights: `qwen3:30b-a3b` holds 30.5B parameters but touches only
~3.3B per token. That is why it is four to eight times faster than the 27B dense
model while being *larger* on disk.

On top of that, Qwen3.8 has thinking mode on by default and emits hundreds of
`<think>` tokens before it says anything. At ~1 tok/s that is minutes of silence
before the first word of the answer.

**The recommendation for this machine:**

- **Live voice → `qwen3:4b-instruct-2507-q4_K_M`.** 15–25 tok/s is the only
  setting that feels like a conversation. This is why the installer fetches it
  alongside the main model.
- **Background subagents → `qwen3.8:27b` or `qwen3:30b-a3b`.** Nobody is waiting
  on a background task, so quality is worth the wait. If you want one model for
  everything and can spare the extra 2 GB, `qwen3:30b-a3b` is the better single
  choice: near-27B quality at conversational-ish speed.

### Setting it at install time

```bash
./install.sh --model qwen3:4b-instruct-2507-q4_K_M
./install.sh --model qwen3:30b-a3b-instruct-2507-q4_K_M
```

### Changing it afterwards

There is no `jarvis model` subcommand. Use one of these three:

```bash
# 1. config.yaml — permanent
#    llm:
#      backend: ollama
#      ollama_model: qwen3:4b-instruct-2507-q4_K_M

# 2. environment — for one shell
export JARVIS_LLM_OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M
jarvis chat

# 3. pull another model without touching config
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama list
```

The global `--model` flag on the `jarvis` command sets `llm.model`, not
`llm.ollama_model`, and `OllamaBackend` reads `cfg.ollama_model or cfg.model` —
so with an Ollama tag already in `config.yaml`, `jarvis --model … chat` will
**not** change which model answers. Use `JARVIS_LLM_OLLAMA_MODEL` for that.

Full reasoning on quantisation, context versus RAM, and adding a model the
catalogue has never heard of is in [MODELS.md](MODELS.md).

---

## Running it as a service

```bash
./install.sh --service
loginctl enable-linger "$USER"

systemctl --user status jarvis-ollama.service
systemctl --user status jarvis.service
journalctl --user -u jarvis.service -f
```

`--service` installs two units: `jarvis-ollama.service` first, because JARVIS is
useless without something to think with, then `jarvis.service`. Both are user
units under `~/.config/systemd/user/`. Neither uses `sudo`.

If `jarvis-ollama.service` shows as `failed` immediately after the install, it
is almost certainly the port clash described under
[stage 12](#what-installsh-does-in-order): the installer's own detached
`ollama serve` is still holding 11434. `pkill -f "ollama serve"` then
`systemctl --user restart jarvis-ollama.service`.

Day-to-day operation, log rotation, backups and the Wayland/X11 story are in
[OPERATIONS.md](OPERATIONS.md).

---

## Air-gapped and metered connections

Nothing here needs the installer to be online at the moment you run it, but the
pieces have to arrive somehow. Fetch them on a connected machine and place them
by hand; every stage then reports "already present" and downloads nothing.

### 1. The Python packages

```bash
# connected machine, same Python version and same architecture
pip download -r requirements.txt -d jarvis-wheels/

# target machine
./install.sh --no-model --no-voice --no-update
.venv/bin/pip install --no-index --find-links jarvis-wheels/ -r requirements.txt
```

### 2. Ollama

The installer needs `api.github.com` to discover the latest version, so on an
offline box do the extraction yourself. The marker file is what stops a later
re-run from treating the directory as somebody else's and refusing to touch it.

Check the release page for the asset name first — current releases publish
`ollama-linux-amd64.tar.zst`, older ones `ollama-linux-amd64.tgz`.

```bash
# connected machine
curl -LO https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst

# target machine
mkdir -p ~/.local/share/jarvis/ollama
tar -C ~/.local/share/jarvis/ollama --zstd -xf ollama-linux-amd64.tar.zst   # -xzf for a .tgz
echo "installed by hand" > ~/.local/share/jarvis/ollama/.jarvis-managed
mkdir -p ~/.local/bin
ln -s ~/.local/share/jarvis/ollama/bin/ollama ~/.local/bin/ollama
ollama --version
```

On a *metered* rather than air-gapped connection, drop the archive into
`~/.local/share/jarvis/downloads/` as
`ollama-<version>-<asset-name-from-the-release>` instead and let the installer
verify and unpack it — the name must match exactly, and the size is checked
against the release before use.

### 3. The model weights

```bash
# connected machine
ollama pull qwen3.8:27b
ollama pull qwen3:4b-instruct-2507-q4_K_M
tar -C ~/.ollama -czf ollama-models.tgz models

# target machine
mkdir -p ~/.ollama
tar -C ~/.ollama -xzf ollama-models.tgz
ollama list
```

Or keep the cache on an external disk and point `OLLAMA_MODELS` at it — the
installer, the runtime module and `ollama` itself all honour that variable.

### 4. The British voice

Two files, straight from Hugging Face, into `~/.local/share/jarvis/voices/`:

```bash
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium
mkdir -p ~/.local/share/jarvis/voices
curl -L -o ~/.local/share/jarvis/voices/en_GB-alan-medium.onnx      "$BASE/en_GB-alan-medium.onnx"
curl -L -o ~/.local/share/jarvis/voices/en_GB-alan-medium.onnx.json "$BASE/en_GB-alan-medium.onnx.json"
```

Both files must be present — the status check requires the `.onnx` *and* the
`.onnx.json`.

### 5. The speech-to-text model

`small.en` is fetched into the Hugging Face hub cache the first time a
`WhisperModel` is constructed. Copy the cache directory across:

```bash
# connected machine
huggingface-cli download Systran/faster-whisper-small.en
tar -C ~/.cache/huggingface/hub -czf whisper-small-en.tgz models--Systran--faster-whisper-small.en

# target machine
mkdir -p ~/.cache/huggingface/hub
tar -C ~/.cache/huggingface/hub -xzf whisper-small-en.tgz
```

`models--guillaumekln--faster-whisper-small.en` is also recognised — that is where
the same models lived before Systran took over publishing them. A directory
counts as usable only when `model.bin`, `config.json` and `tokenizer.json` are
all present, so an interrupted copy is reported as missing rather than as a
model that cannot load.

### Skipping downloads deliberately

```bash
./install.sh --min                    # package only: no runtime, no weights, no speech
./install.sh --no-model --no-voice    # skip everything large
./install.sh --only-main-model        # skip the small model
./install.sh --no-update              # change no versions; install only what is missing
```

---

## Verifying the install

```bash
jarvis doctor          # every optional dependency, and the pip line for each gap
jarvis say             # auditions the voice — proves TTS and audio output
jarvis chat            # text conversation — proves the model, memory and tools
jarvis voice           # hands-free — proves the microphone, STT and the wake word
```

`jarvis doctor` covers the Python side: which optional packages are importable,
which subsystems built, where the data directory is. It does **not** yet report
the Ollama runtime or the model weights. For those:

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags

python -c "
import json
from jarvis.core.config import load_config
from jarvis import runtime
print(json.dumps(runtime.status(load_config()), indent=2, default=str))
"
```

`runtime.status()` is read-only — no downloads, no daemons started, no files
written — and reports each component even when the others are broken. Its
`ready` key is `True` only when Ollama is installed, the configured model is
pulled, the Piper voice is on disk and the Whisper model is cached; `missing`
names whichever of those is not.

If `jarvis` is "command not found" immediately after a successful install, the
symlink exists but `~/.local/bin` was not on `PATH` when you logged in. Debian
and Ubuntu add it from `~/.profile` only if the directory already existed at
login:

```bash
export PATH="$HOME/.local/bin:$PATH"     # this shell
exec $SHELL -l                           # or log out and back in
```

Both `jarvis` and `JARVIS` are linked, because Linux filenames are
case-sensitive and the program's name is written in capitals everywhere.

More failure modes, with causes, are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Uninstalling

In order, most reversible first. **Read the last step before running any of it.**

```bash
# 1. Stop and remove the services
systemctl --user disable --now jarvis.service jarvis-ollama.service
rm -f ~/.config/systemd/user/jarvis.service \
      ~/.config/systemd/user/jarvis-ollama.service
pkill -f "ollama serve"              # anything the installer spawned directly
systemctl --user daemon-reload
loginctl disable-linger "$USER"      # only if you enabled it for JARVIS

# 2. Remove the commands from your PATH
rm -f ~/.local/bin/jarvis ~/.local/bin/JARVIS ~/.local/bin/ollama

# 3. Remove the Ollama runtime and its downloads
rm -rf ~/.local/share/jarvis/ollama ~/.local/share/jarvis/downloads

# 4. Remove the model weights (~19 GB)
rm -rf ~/.ollama

# 5. Remove the speech-to-text model
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-small.en \
       ~/.cache/huggingface/hub/models--guillaumekln--faster-whisper-small.en

# 6. Remove the config, if you ever ran `jarvis config --write`
rm -rf ~/.config/jarvis
```

**Step 7 destroys your data. Back it up first.**
`~/.local/share/jarvis/` holds `memory.db` — every conversation JARVIS has ever
had, which it is configured never to prune — and `tools/`, the Python modules
JARVIS wrote for itself. Neither can be reconstructed.

```bash
# 7. Everything JARVIS remembers and everything it built — back these up first
jarvis memory --export ~/jarvis-final-export.jsonl
cp -r ~/.local/share/jarvis/tools ~/jarvis-tools-backup

rm -rf ~/.local/share/jarvis        # everything above must be saved first
rm -rf /path/to/your/jarvis         # the checkout, including .venv and config.yaml
```

Nothing else was ever written. There are no entries in `/usr`, `/etc`,
`/var`, `/opt`, no system service, no system user, and no package-manager state
— which is the payoff for the no-sudo rule. The system packages from
[the sudo section](#no-sudo-and-the-packages-you-must-install-yourself) are the
one exception, and you installed those yourself; remove them the same way if you
want to.

---

## Known gaps in this release

Documented rather than glossed over, because finding them yourself at 16 GB into
a download is worse.

| Gap | Effect | Work around it |
|---|---|---|
| Stage 12 spawns a detached `ollama serve` when nothing is listening and no unit exists yet, and the installer never stops it | On the **first** `--service` run the spawned process still holds port 11434 when stage 19 enables `jarvis-ollama.service`, so the unit can fail to bind. Correct after the next reboot, and later runs take the systemd path | `pkill -f "ollama serve"` then `systemctl --user restart jarvis-ollama.service` |
| The `--service` stage prints "ollama.service" in its console output and in the summary detail | Cosmetic only — the file written is `~/.config/systemd/user/jarvis-ollama.service`, which is also the name `systemctl --user` and `runtime.status()` use | `ls ~/.config/systemd/user/` |
| A Piper voice that was just downloaded is reported as `already current`. `ensure_piper_voice()` reports the download in a `downloaded` key, but the installer classifies the stage by `action`, which that function does not set | Reporting only — the voice is genuinely fetched | `runtime.status()`, whose `voice.present` is authoritative |
| The header comment at the top of `install.sh` names `~/.local/share/ollama` as the runtime directory | Cosmetic only — the real path is `~/.local/share/jarvis/ollama` | `python -c "from jarvis.runtime import ollama; print(ollama.runtime_dir())"` |

---

## See also

- [UPDATING.md](UPDATING.md) — what a re-run actually updates, and how to roll back
- [MODELS.md](MODELS.md) — the catalogue, quantisation, context versus RAM
- [OPERATIONS.md](OPERATIONS.md) — running it day to day, backups, the service
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — when something does not work
