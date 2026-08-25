# JARVIS

A local voice assistant that runs entirely on your own machine. It listens
through the microphone, thinks with a language model you host yourself, speaks
back in a British voice, remembers every conversation in a local database, and
has real control of the computer it runs on — files, shell, processes, windows,
applications, keyboard and mouse. When a capability it needs does not exist, it
writes the tool, validates it, and loads it. No cloud service is involved unless
you point it at one.

Runs on **Linux** and **Windows**. With no third-party packages installed at all
it still boots; each subsystem degrades rather than failing.

---

## What it does

| | |
|---|---|
| **Listens** | Offline speech-to-text — faster-whisper (`base.en` by default), Whisper, Vosk, or the Windows recogniser. Wake-word gating on the transcript, no second model |
| **Speaks** | British RP — Piper `en_GB-alan-medium` offline, edge-tts `en-GB-RyanNeural` online, or the OS voice. Barge-in: start talking and it stops mid-sentence |
| **Thinks** | Any Hugging Face model, through vLLM, Ollama, any OpenAI-compatible server, transformers, or AirLLM. Default `Qwen/Qwen3.6-27B` |
| **Remembers** | SQLite plus vector search. `memory.prune` is `false` — nothing is ever deleted, and recall spans every past conversation |
| **Acts** | 72 built-in tools: files, shell, processes, services, apps, windows, keyboard, mouse, clipboard, screenshots, network |
| **Delegates** | Long jobs become background subagents with their own tool access; the main agent answers you immediately and relays reports when they land |
| **Extends itself** | `create_tool` writes a Python module, statically validates it, imports it, and registers it — permanently |
| **Degrades honestly** | Every backend has an `is_available()` that returns `False` instead of raising. `jarvis doctor` names every gap and the exact `pip install` for it |

---

## Install on Windows

```powershell
git clone https://github.com/hardcoregamingsyle/jarvis
cd jarvis
.\install.ps1
```

It finds a Python 3.9+ interpreter, creates `.venv`, installs the profile,
downloads the British voice model, and writes `jarvis.bat`.

| flag | meaning |
|---|---|
| `-Profile lean` | **default.** Voice, memory, machine control, Windows integration (~1 GB) |
| `-Profile min` | the package only — nothing optional |
| `-Profile full` | adds torch + transformers + AirLLM. Several gigabytes; read [Serving and concurrency](#serving-and-concurrency) first |
| `-NoVoiceDownload` | skip fetching the Piper voice |
| `-VenvPath <path>` | put the virtualenv somewhere else |

**Nothing in the installer needs Administrator.** It writes only inside the
checkout and your user profile. Specifically:

| action | admin? |
|---|---|
| `install.ps1`, the venv, `jarvis.bat` | no |
| `jarvis` — files, shell, processes, windows, screenshots | no (beyond whatever your own account can do) |
| Autostart at login | no — it writes `HKCU\...\CurrentVersion\Run`, never `HKLM` |
| `winget install Ollama.Ollama` | yes, once, for the installer's own UAC prompt |
| Killing another user's process, editing `C:\Windows` | yes — run JARVIS from an elevated shell if you want that |

Then:

```powershell
.\jarvis.bat doctor
```

If PowerShell refuses to run the script, it is the execution policy, not JARVIS:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Install on Linux

This is the production target: a dedicated laptop, CPU-only, running JARVIS with
full system control.

```bash
git clone https://github.com/hardcoregamingsyle/jarvis
cd jarvis
./install.sh
```

**That one command installs everything, including the model.** Not just the
Python package: the inference runtime (Ollama), the model weights, the British
voice and the speech-to-text model. Nothing is left as a printed instruction to
run afterwards. Budget about **20 GB of disk** and an hour on a slow line.

**Running it again updates all of it** — `git pull`, `pip install --upgrade`, a
newer Ollama if one has been released, and the model weights re-checked against
the registry. It ends with a summary saying, per component, what moved and what
was already current.

Re-checking the weights is cheap: Ollama compares digests and transfers only the
layers that changed, so a model that is already current costs a manifest fetch
and nothing more. `--no-update` checks presence only and changes nothing. See
[docs/UPDATING.md](docs/UPDATING.md) for what each component does on a re-run.

| flag | meaning |
|---|---|
| `--update` | **default.** Pull, upgrade packages, refresh the runtime |
| `--no-update` | repair only: install what is missing, change no versions |
| `--lean` | **default profile.** Voice, memory, machine control |
| `--min` | the Python package only — no runtime, no weights |
| `--full` | adds torch + transformers + AirLLM |
| `--no-voice` | skip the speech-model downloads |
| `--no-model` | skip the Ollama and model-weight stages |
| `--only-main-model` | do not also fetch the small interactive model |
| `--model ID` | use `ID` — a Hugging Face repo id or an Ollama tag — as the main model |
| `--venv PATH` | put the virtualenv somewhere else |
| `--service` | install and enable the systemd **user** services |
| `--vllm` | set up the vLLM path (Linux only; see below) |
| `--no-link` | do not symlink `~/.local/bin/jarvis` |

It reads `/etc/os-release`, detects `apt`/`dnf`/`pacman`/`zypper`, probes for
`ffmpeg`, `espeak-ng`, `wmctrl`, `xdotool`, `notify-send`, `pactl`, `aplay`,
PortAudio and `python3-venv`, and prints the exact install command for whatever
is missing — with a line explaining what each one buys. It also detects a
Wayland session and warns before you find out the hard way.

**The installer never calls `sudo`.** Installing system packages is the one step
that touches the machine outside your home directory, so it stays yours to run.
Ollama is installed rootless too: the official `curl … | sh` needs root, so
JARVIS fetches the official release tarball from GitHub, unpacks it under
`~/.local/share/jarvis/ollama`, links `~/.local/bin/ollama`, and runs it as a
systemd **user** service.

Every large download reports its size and the free space on the target disk
*before* it starts, refuses rather than dying at 90%, resumes if interrupted,
and is skipped when it is already current.

Outside the checkout it writes only `~/.local/bin` symlinks,
`~/.local/share/jarvis`, `~/.ollama`, the Hugging Face cache, and — with
`--service` — `~/.config/systemd/user`. Nothing else, and never over a file it
did not create.

After it finishes, **`jarvis` works from anywhere**: the installer symlinks
`~/.local/bin/jarvis` and, because Linux filenames are case-sensitive,
`~/.local/bin/JARVIS` as well.

```bash
jarvis doctor
```

> Full detail — every stage with its disk cost, the complete list of paths
> written, the per-distro system packages, the rootless Ollama mechanics,
> air-gapped installation and uninstall — is in
> **[docs/INSTALL.md](docs/INSTALL.md)**.
> What a re-run actually updates, and how to roll back, is in
> **[docs/UPDATING.md](docs/UPDATING.md)**.

### System packages

`jarvis/linux/audio.py` knows the package name for each capability on each
distro, and `jarvis doctor` prints the right line for the machine it is on.
Everything, in one command:

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

| package | what you lose without it |
|---|---|
| `portaudio*` | the microphone — `sounddevice` will not build or open a device |
| `ffmpeg` | WAV conversion of edge-tts MP3 output (playback still works) |
| `espeak-ng` | the last-resort offline voice |
| `pulseaudio-utils` (`pactl`) | device enumeration and the audio diagnosis in `jarvis doctor` |
| `alsa-utils` | the ALSA fallback path |
| `libnotify` (`notify-send`) | desktop notifications when a background task finishes |
| `wmctrl`, `xdotool` | window listing, focus, move, snap; keyboard and mouse injection (**X11 only**) |
| `ydotool` | keyboard and mouse injection under Wayland |

> Fedora ships `ffmpeg-free` in its own repositories; the full build needs RPM
> Fusion, which is not something an installer should enable on your behalf.

### Run it as a systemd **user** service

```bash
./install.sh --service          # writes both units, reloads systemd, enables them
systemctl --user status jarvis-ollama.service   # the model server
systemctl --user status jarvis.service          # the assistant
journalctl --user -u jarvis.service -f
```

`--service` installs two user units: `jarvis-ollama.service` first, because
JARVIS is useless without something to think with, then `jarvis.service` — the
prefix keeps it clear of the `ollama.service` the official root installer
writes. The Ollama unit
sets `OLLAMA_NUM_PARALLEL` (from `llm.max_concurrent_requests`),
`OLLAMA_MAX_LOADED_MODELS` and `OLLAMA_KEEP_ALIVE=30m` — concurrency matters
because JARVIS runs a tree of subagents, and keep-alive matters because
re-reading 16 GB of weights between utterances is a minute of silence.

Equivalently, from Python:

```python
from jarvis.linux import service
service.install()        # writes the unit + daemon-reload
service.enable()         # WantedBy=default.target
service.start()
service.status()         # includes whether lingering is enabled
```

`jarvis/linux/service.py` writes `~/.config/systemd/user/jarvis.service` and
talks to `systemctl --user`. It is **never** a system service and never uses
`sudo`, and that is not squeamishness: a system unit runs as root outside any
login session, so it has no `XDG_RUNTIME_DIR`, no PipeWire/PulseAudio socket,
and therefore no microphone and no speakers. A user unit gets all three for
free.

The unit uses `Restart=always` with `StartLimitIntervalSec=120` /
`StartLimitBurst=5`, so a crash loop (a bad model path, say) gives up and shows
as `failed` instead of burning the CPU forever. It installs
`WantedBy=default.target`, not `graphical-session.target`.

#### The linger caveat — read this one

A systemd **user manager** starts when you log in and is torn down when your
last session ends. Close the lid, log out, or reboot to the login screen and
your "always on" assistant is simply gone, with no error anywhere. This is the
single most common "it stopped working" report.

```bash
loginctl enable-linger $USER
```

That tells systemd to start your user manager at boot and keep it running after
logout. `service.status()` detects whether it is set and says so plainly. JARVIS
will **not** run it for you — it is the one step that changes system state
outside your home directory.

#### The Wayland caveat — read this one too

`wmctrl` and `xdotool` are X11 clients. Under Wayland (the default on current
Fedora and Ubuntu GNOME) they see only XWayland windows — usually none of them —
and window activation either fails or **silently does nothing**. A tool that
reports success after doing nothing is worse than one that refuses, so every
window operation checks the session type first and returns an explicit failure
naming Wayland.

| capability | X11 | Wayland |
|---|---|---|
| List / focus / move / snap / close windows | works | **not possible** for an ordinary client |
| Type text, click, move the mouse | works (`xdotool`) | works via `ydotool` (needs the `ydotoold` daemon and membership of the `input` group) |
| Global hotkeys | works | **not possible** — no client may grab a combination globally, by design |
| Screenshots | works | works |
| Voice, memory, files, shell, processes, apps | works | works |

Three real ways forward:

1. **Log out and pick "GNOME on Xorg"** at the login screen. Everything then
   behaves as on any X11 desktop. On a dedicated assistant machine this is the
   pragmatic answer.
2. **`ydotool`** for input injection under Wayland. It can type and click; it
   cannot enumerate or focus windows, because Wayland gives no client that
   information.
3. **A GNOME Shell extension** exposing window control over D-Bus (the "Window
   Calls" family), driven with `gdbus` — the only route to real window
   management on a stock Wayland GNOME session.

For global hotkeys under Wayland, let the compositor own the binding: add a
custom shortcut in GNOME Settings that runs a JARVIS command.

If the service starts before you log in, X11 helpers will have no `DISPLAY`.
From a desktop session:

```bash
systemctl --user import-environment DISPLAY XAUTHORITY WAYLAND_DISPLAY
```

---

## Quick start

Run these four in order. Each proves a different half of the system.

```bash
jarvis doctor          # what is installed, what is missing, the exact fix for each
jarvis say             # auditions the voice — proves TTS and audio output
jarvis chat            # text conversation — proves the model, memory and tools
jarvis voice           # hands-free — proves the microphone, STT and the wake word
```

| command | proves |
|---|---|
| `doctor` | Every optional dependency, every subsystem, the data directory. Start here whenever anything misbehaves — it prints the `pip install` line for each gap |
| `say` | The TTS engine was selected and audio reaches your speakers. It prints which engine won. `null` means none was usable |
| `chat` | An LLM backend answered, memory opened, and the tool registry loaded. The banner shows the model, backend, voice and tool count |
| `voice` | A microphone was found, speech-to-text works, and the wake word gates correctly. Say **"Jarvis"**; after a reply there is a 15-second window to follow up without repeating it |

The rest:

```bash
jarvis                        # voice if a microphone exists, else text
jarvis ask "how much disk space is left?"
jarvis tools                  # every registered tool with its signature
jarvis memory "wifi"          # search long-term memory
jarvis memory --stats
jarvis config                 # the effective config after files + environment
jarvis config --write         # write it to disk
jarvis setup                  # download the voice model, verify the install
```

Global flags: `-c/--config PATH`, `-v/--verbose`, `--model ID`,
`--backend NAME`, `--no-speech`.

---

## Choosing and changing the model

**Every model is a Hugging Face repo id in one config field: `llm.model`.**
Nothing in the codebase hardcodes a model name. Changing which brain JARVIS uses
is a one-line change, including for a repo published after this README was
written.

### Three ways to change it

```yaml
# 1. config.yaml
llm:
  model: Qwen/Qwen3-4B-Instruct-2507
```

```bash
# 2. environment — JARVIS_<SECTION>_<FIELD>
JARVIS_LLM_MODEL=Qwen/Qwen3-8B jarvis chat
JARVIS_LLM_BACKEND=vllm jarvis chat

# 3. command line, for one run
jarvis --model Qwen/Qwen3-8B --backend ollama chat
```

Precedence, later wins: **dataclass defaults → config file → environment → CLI
flag.**

### The catalogue

`jarvis/llm/models.py` keeps short aliases for curated models. It is a
convenience, **never a whitelist** — an unrecognised repo id is synthesised into
a working spec, because that is how every new model starts life.

| alias | repo id | params | active/token | ctx | ~q4 |
|---|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.6B | dense | 32k | 0.5 GB |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7B | dense | 32k | 1.1 GB |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | 4B | dense | 256k | 2.5 GB |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8.2B | dense | 32k | 4.9 GB |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 14.8B | dense | 32k | 8.9 GB |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 32.8B | dense | 32k | 19.7 GB |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30.5B | **3.3B** | 256k | 18.3 GB |
| `qwen3-coder-30b-a3b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 30.5B | 3.3B | 256k | 18.3 GB |
| `qwen3-235b-a22b` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235B | 22B | 256k | 141 GB |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | dense | 128k | 4.9 GB — **gated** |
| **`qwen3.6-27b`** | `Qwen/Qwen3.6-27B` | 27B | dense | **256k** | 16.1 GB |
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` | — | — | — | **not released** |

The default is **`qwen3.6-27b`** (`Qwen/Qwen3.6-27B`): a dense 27B
vision-language model with a 262,144-token native context, extensible to about
1M with YaRN. It is the most capable thing that still fits 32 GB at Q4 (~16 GB),
and it accepts images and video as well as text. It needs `transformers >= 4.57`
for the `Qwen3_5` architecture, and it **thinks by default**, emitting a
`<think>…</think>` block before every answer.

Be aware of the trade-off before pointing a microphone at it. Dense means all
27B parameters are read for every token, where `qwen3-30b-a3b` activates only
~3.3B. On a CPU-only laptop that is roughly **1 tok/s against 4–8**, and the
default thinking block multiplies the wait. For live conversation on such a
machine, set `llm.model: Qwen/Qwen3-4B-Instruct-2507` and leave Qwen3.6 to
background subagents — or serve it from a GPU box over vLLM and point
`llm.vllm_host` at it. `jarvis model recommend` will tell you the same thing:
it ranks by *active* parameters, not total.

The mixture-of-experts alternative, `qwen3-30b-a3b`, gives 30B of knowledge for
~3B of arithmetic per token. On a CPU-only 32 GB laptop that shape is roughly
ten times faster than the dense 32B for the same download size. Full reasoning, the
quantisation arithmetic and the context-vs-RAM tables are in
[docs/MODELS.md](docs/MODELS.md).

### About `Qwen3.8-27B`

**It is not released. No such repository exists on Hugging Face.** It is listed
in `KNOWN_MODELS` with `exists=False` and placeholder figures, on purpose, so
that asking for it produces a straight answer instead of a 404 or a "did you
mean" guess:

```
Qwen3.8 27B (unreleased) (Qwen/Qwen3.8-27B) is not released yet: no such model
exists, so it cannot be selected. ... When it is published, set exists=True for
the 'qwen3.8-27b' entry in jarvis/llm/models.py and it becomes selectable
immediately.
```

When it ships, that is the whole migration: flip `exists=True`, correct the
numbers, set `llm.model`. One line of config for users, one line of code here.

### Pinning a revision

`llm.model` names a repo; `llm.model_revision` pins *which commit of it*. Empty
means "whatever `main` is today", which is fine until an upstream re-upload
changes the weights beneath a working deployment.

```yaml
llm:
  model: Qwen/Qwen3.6-27B
  model_revision: "9a1b2c3..."     # commit SHA, branch or tag
```

Pin it on the production machine. Leave it empty while experimenting.

### `trust_remote_code`

```yaml
llm:
  trust_remote_code: false      # the default
```

Some repos ship their own modelling code — a new architecture that predates its
`transformers` release, or a custom attention kernel. Those need
`trust_remote_code: true`, which executes Python from the repo at load time.

It is off by default as a *correctness* default, not a restriction: with it off,
an unsupported architecture fails loudly instead of quietly running someone
else's code. Every model in the catalogue loads without it. Turn it on when the
load error tells you to, for repos you would run a script from.

---

## Hugging Face tokens and gated models

Most Qwen repos are public and need no token at all. You need one for **gated**
repos (Llama, Gemma, and anything behind a licence click-through) and for your
own **private** repos.

### Why a repo 401s

Two different problems produce a similar-looking failure, and JARVIS
distinguishes them:

| `check_access` status | what actually happened | fix |
|---|---|---|
| `needs_token` | HTTP 401, no token was sent. The repo requires authentication | create a token, set it |
| `gated` | HTTP 403, or 401 **with** a token — the repo exists, you are authenticated, but the licence has not been accepted by *this account* | open the model page, click through the licence, then retry |
| `not_found` | HTTP 404 — no such repo. A typo, or private and invisible without a token | check the spelling; if private, set a token |
| `offline` | huggingface.co could not be reached | fine if the weights are already downloaded |
| `ok` | reachable and downloadable | — |

The distinction matters because a token does not grant access to a gated repo —
**accepting the licence with the same account does.** Adding a token to an
un-accepted licence just turns a 401 into a 401 with a different message.

### Get a token

[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) →
**New token** → **Read** scope. Read is enough; JARVIS never writes to the Hub.

### Where JARVIS looks for it

`models.hf_token()` checks four places, first non-empty wins:

| # | source | set it with |
|---|---|---|
| 1 | `llm.hf_token` in the config | `config.yaml`, or `JARVIS_LLM_HF_TOKEN=...` |
| 2 | `HF_TOKEN` | the ecosystem-standard variable |
| 3 | `HUGGING_FACE_HUB_TOKEN` | the older ecosystem variable |
| 4 | the CLI cache | `huggingface-cli login` — written to `~/.cache/huggingface/token` (or `$HF_HOME/token`, `$XDG_CACHE_HOME/huggingface/token`, `$HF_TOKEN_PATH`) |

A blank or whitespace-only value at any level counts as absent, so an empty
config field cannot shadow a perfectly good environment variable.

### Setting it on Windows

```powershell
# this session only
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

# permanently, for your user (takes effect in NEW shells)
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxx"
```

`setx` truncates at 1024 characters and does not affect the current shell — open
a new terminal afterwards.

### Setting it on Linux

```bash
# this session only
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# permanently, for interactive shells
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc

# or let the official CLI cache it
huggingface-cli login
```

**A systemd service does not read `~/.bashrc`.** For the user unit, either add
it to the unit and reload:

```ini
[Service]
Environment=HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

```bash
systemctl --user daemon-reload && systemctl --user restart jarvis.service
```

…or, better, keep it out of the unit file and put it in an environment file only
you can read:

```bash
install -m 600 /dev/null ~/.config/jarvis/env
echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx' >> ~/.config/jarvis/env
# then in the [Service] section:  EnvironmentFile=%h/.config/jarvis/env
```

`huggingface-cli login` also works for the service, since the cached token file
lives in your home directory and a user unit runs as you.

### Never commit it

`config.yaml`, `config.yml`, `config.json`, `.env`, `*.token` and `hf_token*`
are all in `.gitignore` — the tracked file is `config.example.yaml`, which
contains no secrets. JARVIS additionally **redacts tokens from its own output**:
`models.redact()` masks anything shaped like `hf_...`, `api_org_...`, a
`Bearer ...` header, or a `token=`/`api_key=` query parameter, and it is applied
to every error string and every URL those modules hand back. The token is never
logged, never placed in a returned dict, and never interpolated into an
exception message.

### Verify access before a 30 GB download

One cheap API call first:

```python
from jarvis.llm import models

spec = models.resolve("meta-llama/Llama-3.1-8B-Instruct")
print(models.check_access(spec, models.hf_token()))
# {'status': 'gated', 'ok': False, 'repo': '...', 'message': 'This repo is gated:
#  accept the licence at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.'}
```

And to see what would be downloaded and whether it fits:

```python
print(models.estimate_footprint(spec, "q4"))
# {'download_gb': 4.92, 'kv_cache_gb': 1.21, 'ram_gb': 7.22, ...}
```

---

## Serving and concurrency

Which backend runs the model matters more than which model you pick.

Measured on the reference machine — **i5-10210U class, 4c/8t, 32 GB, no CUDA:**

| backend | what it is | throughput | concurrency | platform |
|---|---|---|---|---|
| **Ollama** | local daemon, GGUF quantised weights | `qwen3:4b-instruct-2507-q4_K_M` ~15–25 tok/s · `qwen3:30b-a3b-instruct-2507-q4_K_M` ~4–8 tok/s · `qwen3:32b-q4_K_M` ~1–2.5 tok/s | serialised by default; `OLLAMA_NUM_PARALLEL>1` batches | Linux, Windows, macOS |
| **vLLM** | continuous-batching inference server | per-token on CPU is not better than Ollama and was not measured on this class of hardware | **the point of it** — one resident copy of the weights serves many simultaneous requests | **Linux only** |
| **openai-compat** | client for llama.cpp / LM Studio / TGI / a remote box | whatever that server does | whatever that server does | anywhere |
| **transformers** | weights loaded in-process | Qwen3-4B on CPU <1 tok/s | none — one request at a time | anywhere |
| **AirLLM** | streams one layer at a time from disk | **~0.02–0.1 tok/s — 10 to 50 *seconds* per token** | none | anywhere |

### Why an agent tree needs a batching server

JARVIS is not one agent. The main agent spawns subagents, which spawn their own,
and all of them call the model concurrently. Ollama's default configuration and
AirLLM both effectively serialise that traffic, so a tree of ten agents finishes
no sooner than ten agents run one after another.

Continuous batching changes the arithmetic. One resident copy of the weights
serves many requests at once, and the expensive part — reading 18 GB of weights
for every token — is amortised across the whole batch. Ten agents cost far less
than ten times one agent. **Throughput under concurrency is what a tree needs;
single-request latency is not.**

`llm.max_concurrent_requests` (default `8`) caps in-flight generation calls on
the client side. That is resource management, not policy: on a box with four
cores, unbounded parallel requests turn a slow answer into no answer. `0`
disables the cap.

### The Ollama knob

Ollama serves one request at a time unless you tell it otherwise:

```bash
# Linux, systemd-managed Ollama
sudo systemctl edit ollama.service
#   [Service]
#   Environment="OLLAMA_NUM_PARALLEL=4"
#   Environment="OLLAMA_MAX_LOADED_MODELS=1"
sudo systemctl restart ollama

# Windows / any shell, before starting the daemon
$env:OLLAMA_NUM_PARALLEL = "4"
ollama serve
```

Each parallel slot needs its own KV cache, so raising it costs RAM (see the
context tables in [docs/MODELS.md](docs/MODELS.md)). Four is a sensible start on
32 GB with the 30B-A3B model, whose cache is unusually small.

Ollama also exposes an OpenAI-compatible `/v1` endpoint, so you can point the
`openai-compat` backend at `http://127.0.0.1:11434/v1` and the rest of JARVIS
cannot tell the difference.

### The exact vLLM launch command

Installing it first, because this is where the target machine bites:

```bash
# With an NVIDIA GPU — the published wheels apply:
pip install vllm

# CPU-only — the published wheels do NOT apply; it must be built:
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
git clone https://github.com/vllm-project/vllm.git
cd vllm && VLLM_TARGET_DEVICE=cpu pip install -e .
```

`./install.sh --vllm` does exactly this decision for you: it installs the wheel
when `nvidia-smi` is present, and otherwise installs nothing and prints the
source-build instructions — a multi-gigabyte download that cannot run is not a
favour.

Then start the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --max-num-seqs 8
```

That is exactly what `jarvis.llm.vllm_backend.server_command(cfg)` produces from
the default config — `--max-model-len` mirrors `llm.context_tokens` and
`--max-num-seqs` mirrors `llm.max_concurrent_requests`. Add `--device cpu` when
`llm.device` is `cpu`, `--download-dir` when `llm.layer_shards_dir` is set, and
`--api-key` when `llm.api_key` is. `vllm serve <model> ...` with the same flags
is equivalent. **JARVIS never launches this process; it only ever connects.**

Then:

```yaml
llm:
  backend: vllm
  vllm_host: http://127.0.0.1:8000/v1
  api_key: ""            # matches the server's --api-key, empty if unauthenticated
```

**Frank advice for the reference laptop:** vLLM's value is batching concurrent
requests, and it shows that value most clearly on a GPU. On a CPU-only i5 it
will not beat Ollama with a small quantised model on single-request latency, and
it is considerably more work to keep running. Reach for it when the agent tree is
genuinely deep and busy, or when you have a GPU. Otherwise Ollama with
`OLLAMA_NUM_PARALLEL=4` is the pragmatic answer, and it speaks the same `/v1`
API if you want to keep your options open.

### vLLM on Windows

There is no supported Windows build of vLLM. Two options, and the JARVIS side is
a pure HTTP client either way:

- **WSL2** — run the server inside the WSL distribution, leave
  `vllm_host: http://127.0.0.1:8000/v1`; WSL2 forwards localhost.
- **A remote host** — point `vllm_host` at the Linux box:
  `http://192.168.1.50:8000/v1`, and start the server with `--host 0.0.0.0`.

### Backend selection

```yaml
llm:
  backend: auto        # auto | vllm | ollama | openai-compat | transformers | airllm | stub
```

`auto` probes **vllm → ollama → openai-compat → transformers → airllm** and
takes the first that answers. The order is deliberate: a vLLM server is never up
by accident, and AirLLM is last because its availability check is only an import
test — probing it earlier would cheerfully select the disk-paged backend at tens
of seconds per token while a perfectly good server was running.

With `allow_fallback: true` (the default) a named-but-unavailable backend falls
through to `auto` rather than failing. Set it to `false` on a production box when
you want a missing server to be loud.

**AirLLM's honest place:** it exists to run a model that does not fit in RAM. On
a 32 GB machine, Qwen3-30B-A3B quantised to ~18 GB *does* fit, and Ollama runs
the same weights roughly a hundred times faster. Use AirLLM for a 70B+ dense
model you cannot otherwise load, and for nothing else.

### Router + task: two models instead of one

By default there is one backend and `spawn_task` reuses it — unchanged from
how JARVIS has always worked. Setting `llm.task_*` in the config splits that
into two models with two different jobs:

- **`llm.model`/`llm.ollama_model`** becomes the **router** — small, fast,
  holds the live conversation and speaks. Its own token stream drives live
  speech (see [Barge-in](#barge-in) below), so its latency *is* the
  assistant's latency: `qwen3:4b-instruct-2507-q4_K_M` (~15–25 tok/s on CPU)
  belongs here, not a dense 27B.
- **`llm.task_backend`/`llm.task_model`/`llm.task_base_url`** becomes what
  `spawn_task` — JARVIS's existing background-delegation tool — actually
  dispatches to. Nothing new to learn: the router already has "call
  `spawn_task` for anything substantial" in its instructions, so pointing
  that tool at a heavier model is the entire mechanism. The router keeps
  talking while it works, and relays the report through speech when it lands.

```yaml
llm:
  model: Qwen/Qwen3-4B-Instruct-2507        # the router
  ollama_model: qwen3:4b-instruct-2507-q4_K_M

  task_backend: openai-compat                # what spawn_task uses instead
  task_base_url: http://<host>:8080/v1        # e.g. a llama.cpp llama-server
```

`task_base_url` can point anywhere an OpenAI-compatible server answers —
including a separately hosted MoE far larger than would ever fit locally, or
this same machine's Ollama running the dense 27B (`task_backend: ollama`,
`task_ollama_model: qwen3.6:27b`) — since the router no longer has to be that
model too. See `config.example.yaml` for the full worked example.

---

## Hardware & compatibility

```bash
jarvis hardware
```

JARVIS auto-detects what it is running on — CPU-only, NVIDIA (CUDA), AMD
(ROCm), Apple Silicon (MPS), or Google TPU (detected, but not yet accelerated
by any backend here — falls back to CPU) — and reports the backend, device and
model pair it would recommend. `config.yaml`'s new `hardware:` section
overrides auto-detection when it gets a machine wrong (a VM with GPU
passthrough `nvidia-smi` cannot see, a container reporting the host's full RAM
instead of its cgroup limit) or when you just want to plan for hardware you do
not have in front of you. See [docs/HARDWARE.md](docs/HARDWARE.md) for the full
compatibility matrix, the 32 GB RAM / 8 GB VRAM worked example, and the
ROCm/CUDA device-string fact that is easy to get backwards.

---

## Voice

```bash
jarvis voice
```

### Modes

| mode | behaviour |
|---|---|
| **`wake`** (default) | Acts only on an utterance that *opens* with a wake word — optionally after filler ("um, Jarvis, …"). Merely mentioning the name ("the jarvis project") does not trigger it. After each reply there is a follow-up window in which the wake word is not needed |
| **`continuous`** | Open conversation, no wake word, until a stretch of silence re-arms it — so a room left alone does not start answering the television |
| **`push`** | Push-to-talk. Nothing is recorded until the hotkey is pressed |

Wake-word detection runs on the *transcript*, not with a separate keyword
spotter. On a CPU-only laptop a small Whisper model is cheap enough to run on
every utterance, and this avoids a second model, a second set of weights and an
extra dependency.

> All three modes are implemented and tested in `jarvis/voice.py`, but the
> `voice.mode` key is not yet a field on `VoiceConfig`, so the config loader
> currently drops it and `wake` is what you get. See
> [docs/HANDOVER.md §5](docs/HANDOVER.md#5-known-sharp-edges) — this README does
> not pretend otherwise.

### Barge-in

While JARVIS is speaking, the level monitor keeps watching the microphone. The
moment you start talking, playback is cut and the rest of the reply is dropped.
It is gated on a loudness margin over the measured ambient floor, because the
one thing barge-in must never do is interrupt JARVIS because it heard JARVIS
through the speakers. Raise the margin if it interrupts itself; lower it if
barge-in feels unresponsive.

```yaml
voice:
  wake_words: [jarvis, hey jarvis]
  require_wake_word: true
  allow_interrupt: true
  greeting: "Good day. All systems are online and at your disposal."
```

### Live speech

`tts.streaming: true` (the default) speaks the reply live, sentence by
sentence, as the model generates it — not after the whole reply exists. Two
threads pipeline the work: one synthesizes each sentence to audio as fast as
the engine allows, the other plays finished sentences in order, so sentence
two is already being synthesized while sentence one is still sounding. The
model's own tool-call syntax (`<tool_call>...</tool_call>`) is recognised and
never spoken — only the prose around it is.

```yaml
tts:
  streaming: true
  stream_min_chars: 12     # merge a sentence shorter than this into the next
  stream_max_buffer: 220   # speak on a whitespace boundary if punctuation
                            # never arrives, rather than staying silent
```

Set `streaming: false` to fall back to the previous whole-utterance
behaviour. Barge-in works identically either way: interrupting drops whatever
is queued or mid-synthesis, not just what is currently sounding.

### The British voices

Engine auto-probe order: **piper → edge → sapi → pyttsx3 → espeak → null.**

| engine | voice | offline | notes |
|---|---|---|---|
| **piper** | `en_GB-alan-medium` | yes | Male RP, neural, CPU-friendly. The default. Also `en_GB-northern_english_male-medium`, `en_GB-jenny_dioco-medium` |
| **edge** | `en-GB-RyanNeural` | no | The most convincing British voice available. Also `en-GB-ThomasNeural`, `en-GB-SoniaNeural`. Returns MP3 |
| **sapi** | system voices | yes | **Windows, zero download.** See below |
| **pyttsx3** | system voices | yes | Cross-platform wrapper; picks a UK voice by `sapi_voice_hint` |
| **espeak** | robotic | yes | Last resort on Linux. It works, and it sounds like it |

```yaml
tts:
  engine: edge
  edge_voice: en-GB-RyanNeural
```

Audition any of them with `jarvis say`. It prints which engine was selected.

> edge-tts returns MP3. JARVIS plays it and transcribes it fine; install
> `ffmpeg` (or `pydub`) if you want `jarvis say -o out.wav` to write real WAV.
> Without a converter it writes `out.mp3` and tells you it did, rather than
> writing an MP3 named `.wav`.

### The zero-download Windows path

Set `tts.engine: sapi` and `stt.engine: windows` and JARVIS talks and listens
using only what ships with Windows — no Piper model, no Whisper weights, no
network. It exists so voice works on a machine with nothing installed.

The trade-offs are real: the Windows recogniser is markedly less accurate than
Whisper (which is why `windows` sits below the real engines in the auto-probe
order, and above `null` only because it actually transcribes), and a stock
Windows install has only US voices (David, Zira) — so it will not sound British
until you add one under **Settings → Time & language → Speech → Add voices**.

---

## Machine control

**72 tools are registered at startup**, including the 7 agent meta-tools the
orchestrator adds. `jarvis tools` lists every one with its signature.

| area | tools |
|---|---|
| **Files** | `read_file` `write_file` `edit_file` `list_dir` `find_files` `search_text` `file_info` `make_dir` `copy_path` `move_path` `delete_path` |
| **Shell & system** | `run_command` `system_info` `disk_usage` `battery_status` `get_env` `set_env` `power_action` `screen_resolution` `take_screenshot` `notify` `volume_set` `volume_mute` `clipboard_get` `clipboard_set` |
| **Processes & services** | `list_processes` `find_process` `process_info` `process_tree` `kill_process` `start_process` `top_consumers` `cpu_memory_snapshot` `list_services` `service_action` |
| **Applications** | `launch_app` `close_app` `open_file_with` `open_folder` `media_control` |
| **Windows** | `list_windows` `focus_window` `active_window` `minimize_window` `maximize_window` `restore_window` `move_window` `snap_window` `close_window` `set_always_on_top` `window_text` |
| **Keyboard & mouse** | `type_text` `press_key` `hotkey` `key_names` `mouse_move` `mouse_click` `mouse_drag` `mouse_scroll` `mouse_position` `screen_size` `clipboard_paste_text` |
| **Network** | `http_get` `http_post_json` `fetch_page_text` `download_file` `open_url` `web_search` `check_internet` |
| **Self-extension** | `create_tool` `list_custom_tools` `delete_custom_tool` |
| **Agent meta-tools** | `spawn_task` `task_tree` `list_tasks` `task_status` `cancel_task` `remember` `recall` |

Concretely: it can read and rewrite your files, run any shell command, kill any
process it has rights to, install packages, start and stop services, launch and
close applications, type into whatever window has focus, drive the mouse,
rearrange your desktop, take screenshots, fetch web pages, and reboot the
machine.

`list_windows` and `focus_window` are defined by both `app_tools` and
`window_tools`. `load_builtin()` registers `window_tools` last, and registration
replaces by name, so the richer implementation wins: it matches titles exactly
before falling back to a case-insensitive substring, reports ambiguity instead
of guessing, and verifies with `GetForegroundWindow` that focus actually took —
Windows blocks focus stealing, so the naive call reports success while doing
nothing.

Window and input control need `pywin32`/`pyautogui` on Windows and
`wmctrl`/`xdotool` (X11) or `ydotool` (Wayland) on Linux; everything else is
stdlib or `psutil`. Anything missing degrades to a clear failure naming the
package, never a silent no-op.

---

## Access policy

**JARVIS ships with no restrictions.**

```yaml
security:
  mode: open
  protected_paths: []
  dangerous_patterns: []
```

`mode: open` means every path check, every command check and every tool check
returns "allowed, no confirmation". There are no protected directories, no
blocked commands, no allowlist, and no confirmation prompts. It will run
`rm -rf`, it will delete your files, it will `format` a drive, and it will not
ask first. In `open` mode the destructive-shape detectors are not even
consulted — a mode called `open` that secretly refused things would make the
setting a lie.

This is deliberate. It is the owner's explicit choice for a dedicated machine
that exists to be driven by an assistant. The policy engine is fully present and
fully tested; the shipped configuration simply tells it nothing to enforce.

Two things still hold, and neither is a permission rule:

- **The audit log stays on.** Every decision is appended to
  `<data dir>/logs/audit.jsonl`. A log is a record of what happened, not a
  restriction on what may happen — it never refuses anything.
- **`delete_path` refuses four whole-tree targets:** a filesystem root, the home
  directory, the current working directory, and any ancestor of it. Not as
  policy — a recursive delete of those *cannot do what asking for it implies*.
  It destroys the running environment partway through and aborts on the first
  locked file, so the result is never "deleted" but always "half-deleted,
  unrecoverable". Each is reachable another way (delete the children, or use the
  unrestricted shell), so nothing is withheld. Windows drive-relative paths like
  `C:` are anchored to the drive root before any check, because `Path("C:")`
  resolves to *the current directory*, not the root — a confusion that destroyed
  part of this repository once.

### Turning restrictions on

If you want them, three lines of YAML:

```yaml
security:
  mode: guarded                          # or: readonly
  protected_paths: ["/etc", "/boot", "C:\\Windows"]
  dangerous_patterns: ["format", "diskpart", "mkfs"]
```

| mode | behaviour |
|---|---|
| `open` | **default.** Everything allowed, nothing prompts |
| `guarded` | Commands matching `dangerous_patterns` or a known destructive shape ask for confirmation; writes under `protected_paths` are refused |
| `readonly` | Mutation refused outright — inspection commands only |

In `guarded` mode, `jarvis chat` and `jarvis voice` wire the confirmation
callback to an interactive prompt. With nothing to ask (a headless run),
`security.unattended_policy` decides: `allow` (default) proceeds and audits,
`deny` refuses. Both lists are empty by default, so even these modes start
permissive until you populate them.

Also available, independent of mode: `allow_shell`, `allow_file_write`,
`allow_process_control`, `allow_network`. These are capability switches, not
mode policy — `open` does not override them.

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit. Every value there is the
default, so delete anything you do not want to change. JSON works if PyYAML is
not installed.

Searched in ascending priority: the user config directory, then the current
directory, then `$JARVIS_CONFIG`. Then environment variables
`JARVIS_<SECTION>_<FIELD>` override everything:

```bash
JARVIS_LLM_BACKEND=vllm
JARVIS_LLM_MODEL=Qwen/Qwen3-8B
JARVIS_TTS_ENGINE=edge
JARVIS_SECURITY_MODE=guarded
JARVIS_LOG_LEVEL=DEBUG
JARVIS_HOME=/srv/jarvis          # relocate the whole data directory
```

`jarvis config` prints the effective configuration after everything is merged.

Data lives in `%LOCALAPPDATA%\Jarvis\` on Windows and
`~/.local/share/jarvis/` on Linux: `memory.db`, `tools/`, `voices/`, `models/`,
`logs/`.

---

## Project layout

```
jarvis/
  core/         contracts, config, events, logging, security, platform layer
  llm/          models catalogue + vllm | ollama | openai-compat | transformers | airllm | stub
  speech/       audio_io, stt (whisper/vosk/windows), tts (piper/edge/sapi/pyttsx3/espeak)
  memory/       sqlite store, embeddings, dynamic context manager
  tools/        registry + file/system/process/web/app/input/window tools + tool_maker
  agent/        prompts, tool-call protocol, agent loop, subagents, task manager
  win/          tray icon, global hotkeys, autostart, toast notifications
  linux/        systemd user service, desktop/Wayland integration, audio detection
  app.py        boots and wires every subsystem
  voice.py      the hands-free loop
  cli.py        command-line entry point
tests/          no network, no GPU, no models, no audio device required
docs/           architecture, operations, models, tool authoring, testing, troubleshooting
```

---

## Documentation

| | |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | The complete installation reference: every stage and its disk cost, every flag, every path written, rootless Ollama, air-gapped installs, uninstall |
| [docs/UPDATING.md](docs/UPDATING.md) | What re-running the installer updates, what it deliberately leaves alone, and how to roll back |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pieces fit: the agent loop, the task tree, the event bus |
| [docs/MODELS.md](docs/MODELS.md) | Qwen3 family, dense vs MoE, quantisation, context vs RAM, adding a model |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Auto-detection, the compatibility matrix, the ROCm/CUDA device-string fact, TPU's detected-not-accelerated gap |
| [docs/TOOL_AUTHORING.md](docs/TOOL_AUTHORING.md) | The specification for writing a tool — for humans and for JARVIS itself |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Running it day to day: the service, logs, backups |
| [docs/TESTING.md](docs/TESTING.md) | How the suite is built and how to add to it |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | When something does not work |
| [docs/HANDOVER.md](docs/HANDOVER.md) | State of the project, what is verified and what is not |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Environment, the hard rules, the destructive-test rule |

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

The suite is hermetic: every test runs against a temporary `JARVIS_HOME`, uses a
scripted language model, and never touches the network, an audio device, a GPU,
or your real data. It passes on a bare Python with zero optional dependencies —
`tests/test_import_hygiene.py` enforces that by re-importing the whole package in
a clean subprocess with every heavy dependency blocked.

---

## Licence

MIT. See [LICENSE](LICENSE).
