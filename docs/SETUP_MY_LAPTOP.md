# Setting up on an i5-10210U / 32 GB / XFCE / X11 laptop

Written for this exact machine:

| | |
|---|---|
| CPU | Intel i5-10210U @ 1.60 GHz — **4 cores / 8 threads** |
| GPU | Intel CometLake-U GT2 + Radeon R5 M230 / R7 M260DX / 520 / 610 |
| RAM | 32.6 GB |
| Desktop | XFCE on **X11** |

**Read this first: you are running on the CPU.** The Radeon R5 M230 is a GCN 1.0
part that ROCm has never supported, and the Intel iGPU has no usable inference
path here either. JARVIS now detects this and says so rather than pretending
otherwise. Everything below assumes 4 cores and no accelerator, because that is
what you have.

The good news is that X11 is the *right* display server for this — window
control, global hotkeys and input injection all work properly. Wayland users
have to give some of that up; you do not.

---

## 1. System packages

The installer never runs `sudo`. This is the one command it cannot run for you:

```bash
sudo apt-get install -y python3-venv portaudio19-dev ffmpeg espeak-ng \
    pulseaudio-utils alsa-utils libnotify-bin wmctrl xdotool
```

*(Adjust for your distribution — `dnf`, `pacman` and `zypper` equivalents are in
[INSTALL.md](INSTALL.md#no-sudo-and-the-packages-you-must-install-yourself).)*

Why each one matters on your setup:

| Package | Without it |
|---|---|
| `python3-venv` | On Debian/Ubuntu `python3 -m venv` silently creates a tree **with no pip** |
| `portaudio19-dev` | **No microphone.** `import sounddevice` fails outright |
| `espeak-ng` | Loses the fallback voice (Piper is the default, so not fatal) |
| `pulseaudio-utils` | No audio device enumeration or switching |
| `wmctrl`, `xdotool` | No window control, no keyboard/mouse injection — **these work for you because you are on X11** |
| `ffmpeg` | edge-tts MP3 → WAV conversion |

Do **not** install `ydotool`. It is the Wayland input path; on X11 `xdotool` is
the correct and better-supported choice.

---

## 2. Install

```bash
git clone https://github.com/hardcoregamingsyle/jarvis
cd jarvis
./install.sh
```

Budget **~25 GB of disk** and an hour on a slow connection. Every download is
size-checked against free space *before* it starts, and resumes if interrupted.

It installs the Python package, Ollama (rootless, into your home directory —
no `curl | sh`, no system service), and pulls:

| Model | Size | Role |
|---|---|---|
| `qwen3.8:27b` | ~18 GB | **The brain.** Reasoning, tool calls, decisions |
| `qwen3:1.7b` | ~1.1 GB | **The voice.** Phrases answers for speech |
| `qwen3:4b-instruct-2507` | ~2.4 GB | Fast general-purpose alternative |
| Piper `en_GB-alan-medium` | ~60 MB | The British voice |
| faster-whisper `small.en` | ~500 MB | Transcription |

### If 25 GB is too much

Skip the 27B and run the mixture-of-experts model instead. On your CPU it is
**4–8× faster** for a similar download, because it activates only ~3B
parameters per token:

```bash
./install.sh --model qwen3:30b-a3b-instruct-2507-q4_K_M
```

Honestly, for a 4-core laptop this is the configuration I would choose. The 27B
is more capable, but you will feel every one of those parameters.

---

## 3. Verify

```bash
jarvis selftest
```

This is the command to run whenever anything misbehaves. It executes every
stage for real and names the broken link:

```
Hardware
  OK    CPU                          4 cores / 8 threads, 32.6 GB RAM
  WARN  Accelerator                  none usable (Radeon R5 M230) - running on CPU

Language model
  OK    Backend                      ollama (Qwen/Qwen3.8-27B)
  OK    Generation                   14.2s, ~0.8 tok/s: 'Systems nominal, Sir.'
  WARN  Generation speed             very slow for interactive use - the voice
                                     model below is what keeps replies immediate
  OK    Voice model                  0.4s: 'It is half past four, Sir.'

Speech
  OK    Speech to text               faster-whisper (small.en)
  OK    Microphone                   2 input device(s)
  OK    Text to speech               piper, 68840 bytes in 0.3s

Full turn
  OK    Agent turn                   15.1s: 'Systems nominal, Sir.'
```

The `WARN` lines above are **expected and correct** on your hardware. They are
not failures — they are the tool being honest about a CPU-only machine.

Then audition the voice and have a conversation:

```bash
jarvis say            # proves TTS and audio output
jarvis chat           # text conversation — proves model, memory, tools
jarvis voice          # hands-free — say "Jarvis"
```

---

## 4. What you will actually experience

This is the part most setup guides lie about, so here are real numbers for
4 cores at Q4:

| | Qwen3.8-27B (dense) | Qwen3-30B-A3B (MoE) |
|---|---|---|
| Speed | ~0.5–1 tok/s | ~4–8 tok/s |
| 40-word reply | **2–4 minutes** | ~15–30 seconds |

The two-model split is what makes the first column survivable. Per turn:

1. You stop speaking. The transcript is gated on the wake word.
2. **Under a second** — "One moment, Sir." So the pause reads as thought, not a crash.
3. The 27B works. Takes as long as it takes.
4. Its answer goes to the 1.7B, which phrases it in well under a second.
5. Piper speaks **sentence by sentence** — audio starts on the first full stop.

To an onlooker it is immediate and continuous. That was the design goal.

If the wait still bothers you, swap the brain — one line, no reinstall:

```yaml
llm:
  model: Qwen/Qwen3-30B-A3B-Instruct-2507
  ollama_model: qwen3:30b-a3b-instruct-2507-q4_K_M
```

---

## 5. Tuning for 4 cores

Edit `config.yaml` in the checkout. Everything below is optional.

**If transcription lags** — the default `small.en` is the accuracy sweet spot,
but `base.en` is ~3× cheaper:

```yaml
stt:
  model: base.en
```

**If it cuts you off mid-sentence** — raise the silence window:

```yaml
stt:
  silence_duration: 1.5     # default 1.2
```

**If it triggers on background noise, or misses you** — measure your room. The
default assumes a quiet one:

```yaml
stt:
  silence_threshold: 0.02   # raise if it triggers on noise, lower if it misses you
```

**If subagents make everything crawl** — 4 cores do not go far:

```yaml
agent:
  max_concurrent_tasks: 2   # default 4
llm:
  max_concurrent_requests: 2
```

**Thinking mode.** Leave `llm.thinking: auto` alone. Setting it to `on` will
make the 27B spend its entire token budget reasoning and return nothing — that
was the original bug.

---

## 6. Autostart on XFCE

```bash
./install.sh --service
systemctl --user enable --now jarvis.service
journalctl --user -u jarvis -f      # watch it
```

One caveat specific to a laptop: a user service stops when you log out unless
lingering is enabled.

```bash
sudo loginctl enable-linger "$USER"
```

Alternatively, XFCE's own **Settings → Session and Startup → Application
Autostart** works fine and needs no root.

---

## 7. When something breaks

```bash
jarvis selftest       # which stage is broken
jarvis doctor         # what is missing, and the exact install line
tail -f ~/.local/share/jarvis/logs/jarvis.log
```

Common cases on this hardware:

| Symptom | Cause |
|---|---|
| "No usable microphone" | `portaudio19-dev` missing, or you are not in the `audio` group on a bare-ALSA setup |
| No answer at all | Ollama not running — `ollama serve`. `selftest` says this explicitly |
| Replies take minutes | Expected for the dense 27B. Switch to the MoE model above |
| Voice sounds robotic | Piper voice did not download; `jarvis setup` fetches it |
| Wake word never fires | Check `jarvis selftest` transcription. `stt.initial_prompt` already biases "Jarvis" |

---

## Two honest caveats

**I could not test this on real hardware.** The fixes were verified against a
stand-in Ollama daemon and synthetic audio in a sandbox with no GPU, no
microphone and no model weights. The logic is covered by 2408 passing tests;
the *experience* on your laptop is not something I can promise from here.
`jarvis selftest` is there precisely so you can check it in one command.

**The tok/s figures are estimates** from the model's architecture and your CPU,
not measurements. `jarvis selftest` prints your real numbers — trust those over
this document.
