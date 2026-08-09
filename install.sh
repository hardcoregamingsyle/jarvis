#!/usr/bin/env bash
#
# JARVIS installer for Linux (and, with fewer features, macOS).
#
#   ./install.sh [--lean]        default: voice, memory, machine control
#   ./install.sh --full          adds torch + transformers + AirLLM (many GB)
#   ./install.sh --min           the package only
#   ./install.sh --no-voice      skip the voice-model download
#   ./install.sh --venv PATH     use a different virtualenv directory
#   ./install.sh --service       install + enable the systemd *user* service
#   ./install.sh --vllm          set up the vLLM path (Linux-only, separate install)
#   ./install.sh --model ID      write ID into config.yaml as the model to use
#   ./install.sh --no-link       do not symlink ~/.local/bin/jarvis
#
# Nothing here deletes, moves or overwrites anything, and nothing is run with
# sudo — where a system package is needed the exact command is printed for you
# to run yourself.  The only thing written outside this directory is a single
# symlink at ~/.local/bin/jarvis so that `jarvis` works as a bare command; it
# is never created over the top of an existing file.  Pass --no-link to skip it.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROFILE="lean"
DOWNLOAD_VOICE=1
VENV=".venv"
WANT_SERVICE=0
WANT_VLLM=0
WANT_LINK=1
MODEL_ID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --full) PROFILE="full" ;;
        --min) PROFILE="min" ;;
        --lean) PROFILE="lean" ;;
        --no-voice) DOWNLOAD_VOICE=0 ;;
        --service) WANT_SERVICE=1 ;;
        --vllm) WANT_VLLM=1 ;;
        --no-link) WANT_LINK=0 ;;
        --model)
            if [ $# -lt 2 ]; then echo "--model requires a model id" >&2; exit 1; fi
            MODEL_ID="$2"; shift
            ;;
        --venv)
            if [ $# -lt 2 ]; then echo "--venv requires a path" >&2; exit 1; fi
            VENV="$2"; shift
            ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; DIM='\033[2m'; RESET='\033[0m'
step() { printf "\n${CYAN}==> %s${RESET}\n" "$1"; }
ok()   { printf "    ${GREEN}%s${RESET}\n" "$1"; }
warn() { printf "    ${YELLOW}%s${RESET}\n" "$1"; }
info() { printf "    %s\n" "$1"; }

OS="$(uname -s 2>/dev/null || echo unknown)"

cat <<'BANNER'
    ___  _   _____   _   _ ___ ___
   |_ _|/_\ | _ \ \ / /_| / __/ __|
    | |/ _ \|   /\ V / | \__ \__ \
   |___/_/ \_\_|_\ \_/  |_|___/___/   installer for Linux
BANNER

# --------------------------------------------------------------------------- #
#  Distribution
# --------------------------------------------------------------------------- #
step "Identifying the distribution"

PM=""
PM_INSTALL=""
if command -v apt-get >/dev/null 2>&1; then
    PM="apt";    PM_INSTALL="sudo apt-get install -y"
elif command -v dnf >/dev/null 2>&1; then
    PM="dnf";    PM_INSTALL="sudo dnf install -y"
elif command -v pacman >/dev/null 2>&1; then
    PM="pacman"; PM_INSTALL="sudo pacman -S --needed"
elif command -v zypper >/dev/null 2>&1; then
    PM="zypper"; PM_INSTALL="sudo zypper install -y"
fi

DISTRO_NAME=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    # The trailing '|| true' matters: under `set -e` a failed source here would
    # abort the whole installer over a cosmetic banner line.
    DISTRO_NAME="$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}" || true)"
fi

if [ -n "$PM" ]; then
    ok "${DISTRO_NAME:-$OS} (package manager: $PM)"
else
    warn "${DISTRO_NAME:-$OS} — no apt/dnf/pacman/zypper found."
    warn "System package names will be described generically."
fi

# Map a capability to this distribution's package name.  An empty answer means
# "already part of the base install here", and nothing is printed for it.
pkg_for() {
    case "$1:$PM" in
        portaudio:apt)    echo "portaudio19-dev" ;;
        portaudio:dnf)    echo "portaudio-devel" ;;
        portaudio:pacman) echo "portaudio" ;;
        portaudio:zypper) echo "portaudio-devel" ;;

        ffmpeg:dnf)       echo "ffmpeg-free" ;;   # full ffmpeg needs RPM Fusion
        ffmpeg:*)         echo "ffmpeg" ;;

        espeak:*)         echo "espeak-ng" ;;
        wmctrl:*)         echo "wmctrl" ;;
        xdotool:*)        echo "xdotool" ;;

        libnotify:apt)    echo "libnotify-bin" ;;
        libnotify:zypper) echo "libnotify-tools" ;;
        libnotify:dnf)    echo "libnotify" ;;
        libnotify:pacman) echo "libnotify" ;;

        pactl:pacman)     echo "libpulse" ;;
        pactl:apt|pactl:dnf|pactl:zypper) echo "pulseaudio-utils" ;;

        alsa:*)           echo "alsa-utils" ;;

        venv:apt)         echo "python3-venv" ;;
        venv:*)           echo "" ;;             # bundled with python3 elsewhere
        *)                echo "" ;;
    esac
}

# --------------------------------------------------------------------------- #
#  Python
# --------------------------------------------------------------------------- #
step "Checking Python"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            ok "$("$candidate" --version 2>&1)  ($candidate)"
            break
        fi
    fi
done
[ -n "$PYTHON" ] || { echo "Python 3.9+ is required." >&2; exit 1; }

# --------------------------------------------------------------------------- #
#  System libraries
# --------------------------------------------------------------------------- #
step "Checking system libraries"
NEEDS=""
add_need() { NEEDS="$NEEDS $1"; }

command -v ffmpeg      >/dev/null 2>&1 || add_need ffmpeg
command -v espeak-ng   >/dev/null 2>&1 || command -v espeak >/dev/null 2>&1 || add_need espeak
command -v wmctrl      >/dev/null 2>&1 || add_need wmctrl
command -v xdotool     >/dev/null 2>&1 || add_need xdotool
command -v notify-send >/dev/null 2>&1 || add_need libnotify
command -v pactl       >/dev/null 2>&1 || command -v pw-cli >/dev/null 2>&1 || add_need pactl
command -v aplay       >/dev/null 2>&1 || add_need alsa
"$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1 || add_need venv

if ldconfig -p 2>/dev/null | grep -q libportaudio; then
    :
elif [ -e /usr/lib/x86_64-linux-gnu/libportaudio.so.2 ] || [ -e /usr/lib64/libportaudio.so.2 ]; then
    :
else
    add_need portaudio
fi

if [ -n "${NEEDS// /}" ]; then
    warn "Missing or not on PATH:$NEEDS"
    PACKAGES=""
    for need in $NEEDS; do
        name="$(pkg_for "$need")"
        [ -n "$name" ] && PACKAGES="$PACKAGES $name"
    done
    if [ -n "$PM" ] && [ -n "${PACKAGES// /}" ]; then
        info ""
        info "Run this yourself (the installer never calls sudo):"
        printf "        ${YELLOW}%s%s${RESET}\n" "$PM_INSTALL" "$PACKAGES"
    else
        info "Install the equivalents of:$NEEDS"
    fi
    info ""
    info "What each one is for:"
    info "  portaudio  microphone capture — 'import sounddevice' fails without it"
    info "  ffmpeg     audio format conversion"
    info "  espeak-ng  last-resort offline voice"
    info "  wmctrl     listing and focusing windows (X11 only, see below)"
    info "  xdotool    keyboard/mouse control and window focus (X11 only)"
    info "  libnotify  notify-send, for desktop notifications"
    info "  pactl      audio device listing and switching"
    info "  python3-venv  the virtual environment this installer creates"
    warn "JARVIS installs regardless; the affected features degrade, they do not crash."
else
    ok "all present"
fi

# --------------------------------------------------------------------------- #
#  Session type (Wayland warning)
# --------------------------------------------------------------------------- #
SESSION="${XDG_SESSION_TYPE:-}"
if [ -z "$SESSION" ]; then
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then SESSION="wayland"
    elif [ -n "${DISPLAY:-}" ]; then SESSION="x11"
    else SESSION="unknown"; fi
fi
if [ "$SESSION" = "wayland" ]; then
    step "Session type: wayland"
    warn "wmctrl and xdotool are X11 clients. Under Wayland they see only XWayland"
    warn "windows and window focus silently does nothing, so JARVIS refuses those"
    warn "operations instead of pretending. Global hotkeys cannot be grabbed at all."
    info "Options: log in with 'GNOME on Xorg'; or install ydotool for input"
    info "injection; or bind a Custom Shortcut in GNOME Settings -> Keyboard to"
    info "'$VENV/bin/jarvis voice' so the compositor owns the hotkey."
else
    step "Session type: $SESSION"
fi

# --------------------------------------------------------------------------- #
#  Virtual environment
# --------------------------------------------------------------------------- #
step "Creating the virtual environment ($VENV)"

# `--venv /abs/path` must not be pasted onto $ROOT.
case "$VENV" in
    /*) VENV_DIR="$VENV" ;;
    ~*) VENV_DIR="${VENV/#\~/$HOME}" ;;
    *)  VENV_DIR="$ROOT/$VENV" ;;
esac

# The package name that supplies ensurepip, which is versioned on Debian and
# Ubuntu ("python3.12-venv") as often as it is generic ("python3-venv").
venv_package_hint() {
    local ver
    ver="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 3)"
    case "$PM" in
        apt) printf 'sudo apt-get install -y python3-venv || sudo apt-get install -y python%s-venv' "$ver" ;;
        dnf) printf 'sudo dnf install -y python3-libs' ;;
        *)   printf 'install your distribution'\''s python venv/ensurepip package' ;;
    esac
}

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR" || {
        echo "venv creation failed. Try: $(venv_package_hint)" >&2
        exit 1
    }
    ok "created"
else
    ok "already present, reusing it"
fi

VPY="$VENV_DIR/bin/python"
[ -x "$VPY" ] || { echo "Interpreter missing at $VPY" >&2; exit 1; }

# A venv WITHOUT pip is the single most common Linux failure here, and the
# error it produces on its own — "No module named pip" — sends people off to
# install pip3, which is a different thing entirely and does not help.
#
# On Debian and Ubuntu, `python3 -m venv` is split across packages: without
# python3-venv there is no ensurepip, and CPython creates bin/python and
# pyvenv.cfg *before* it tries to install pip. So the tree exists, `[ -d ]` and
# `[ -x ]` both pass, and on any re-run the broken venv is cheerfully reused.
# Detect it here and say the useful thing.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
    printf "\n${YELLOW}The virtual environment has no pip.${RESET}\n" >&2
    cat >&2 <<EOF

  $VENV_DIR was created without pip. This is not the same problem as a
  missing pip3: a system pip3 cannot supply one to a new virtual environment.
  Debian and Ubuntu ship the 'ensurepip' module in a separate package.

  Fix it with:

      $(venv_package_hint)
      rm -r "$VENV_DIR"
      ./install.sh

EOF
    exit 1
fi
ok "pip is present"

step "Upgrading pip"
"$VPY" -m pip install --upgrade pip setuptools wheel --quiet
ok "done"

# --------------------------------------------------------------------------- #
#  Dependencies
# --------------------------------------------------------------------------- #
step "Installing the '$PROFILE' profile"
set +e
case "$PROFILE" in
    min)  "$VPY" -m pip install -e . ;;
    lean) "$VPY" -m pip install -e . && "$VPY" -m pip install -r requirements.txt ;;
    full)
        "$VPY" -m pip install -e .
        warn "This downloads torch and friends — several gigabytes."
        "$VPY" -m pip install -r requirements-full.txt
        ;;
esac
STATUS=$?
set -e
if [ $STATUS -ne 0 ]; then
    warn "Some packages failed. JARVIS will still run in degraded mode."
    warn "Run '\$JARVIS_BIN doctor' to see exactly what is missing."
fi

# --------------------------------------------------------------------------- #
#  vLLM
# --------------------------------------------------------------------------- #
if [ "$WANT_VLLM" -eq 1 ]; then
    step "vLLM"
    if [ "$OS" != "Linux" ]; then
        warn "vLLM is Linux-only. On Windows, run it inside WSL2 or on another host"
        warn "and point llm.vllm_host at it."
    elif command -v nvidia-smi >/dev/null 2>&1; then
        ok "NVIDIA GPU detected — installing the prebuilt CUDA wheel."
        set +e
        "$VPY" -m pip install vllm
        VSTATUS=$?
        set -e
        if [ $VSTATUS -eq 0 ]; then
            ok "vLLM installed."
            info "Serve a model, then set llm.backend: vllm in config.yaml:"
            info "  $VENV/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192"
        else
            warn "vLLM install failed; see the output above."
        fi
    else
        warn "No NVIDIA GPU found, so the published vLLM wheels do not apply and"
        warn "nothing was installed — a multi-gigabyte download that cannot run is"
        warn "not a favour."
        info ""
        info "vLLM on a CPU-only machine has to be built from source:"
        info "  ${PM_INSTALL:-<your package manager> install} gcc-12 g++-12 libnuma-dev"
        info "  git clone https://github.com/vllm-project/vllm.git"
        info "  cd vllm && VLLM_TARGET_DEVICE=cpu $VPY -m pip install -e ."
        info ""
        info "Frank advice for an i5 CPU-only laptop: vLLM's value is batching many"
        info "concurrent requests on a GPU. On CPU it will not beat Ollama with a"
        info "small quantised model, and it is far more work to keep running:"
        info "  curl -fsSL https://ollama.com/install.sh | sh"
        info "  ollama pull qwen3:4b-instruct-2507-q4_K_M"
        info "  then set llm.backend: ollama in config.yaml"
    fi
fi

# --------------------------------------------------------------------------- #
#  Voice model
# --------------------------------------------------------------------------- #
if [ "$DOWNLOAD_VOICE" -eq 1 ] && [ "$PROFILE" != "min" ]; then
    step "Fetching the British voice model"
    "$VPY" -m jarvis setup || warn "voice download failed; edge-tts or espeak will be used"
else
    step "Skipping the voice download"
    "$VPY" -m jarvis setup --no-download || true
fi

# --------------------------------------------------------------------------- #
#  Launcher
# --------------------------------------------------------------------------- #
step "Locating the launcher"

# NO hand-written launcher here, and the reason matters.
#
# This used to redirect into "$ROOT/jarvis" — which on Linux names the
# package DIRECTORY, so the redirect hit EISDIR ("Is a directory") and, under
# `set -euo pipefail`, aborted the installer after every package had already
# been downloaded. Windows never saw it: install.ps1 writes "jarvis.bat".
#
# It was also unnecessary. pyproject.toml declares
# [project.scripts] jarvis = "jarvis.cli:main", so `pip install -e .` above has
# already produced a console script in the venv.
JARVIS_BIN="$VENV_DIR/bin/jarvis"
if [ -x "$JARVIS_BIN" ]; then
    ok "$JARVIS_BIN"
else
    warn "console script not found at $JARVIS_BIN"
    info "Use '$VPY -m jarvis' instead; both entry points are equivalent."
    JARVIS_BIN="$VPY -m jarvis"
fi

# --------------------------------------------------------------------------- #
#  Put `jarvis` on the PATH
#
#  The console script lives inside the virtualenv, so without this the obvious
#  thing to type — `jarvis doctor`, which is what the README says everywhere —
#  answers "command not found" on a successful install.  A symlink in
#  ~/.local/bin is the fix: it is the freedesktop-blessed location for
#  per-user executables, needs no sudo, and is already on PATH on Debian,
#  Ubuntu, Fedora and Arch.
# --------------------------------------------------------------------------- #
JARVIS_ON_PATH=0
if [ "$WANT_LINK" -eq 1 ] && [ -x "$VENV_DIR/bin/jarvis" ]; then
    step "Putting 'jarvis' on your PATH"
    LOCAL_BIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
    LINK="$LOCAL_BIN/jarvis"

    mkdir -p "$LOCAL_BIN"

    # Link one name. Returns 0 when the name ends up pointing at our console
    # script, 1 otherwise.  -ef compares device and inode through the symlink,
    # so a re-run recognises its own link regardless of how the path was
    # spelled; comparing readlink output as a string breaks on a relative link.
    link_one() {
        local target="$1" link="$2"
        if [ -e "$link" ] && [ "$link" -ef "$target" ]; then
            ok "already linked: $link"
            return 0
        elif [ -e "$link" ] || [ -L "$link" ]; then
            # Never clobber. Another tool owning this name is the user's business.
            warn "$link already exists and is not ours — leaving it untouched."
            return 1
        elif ln -s "$target" "$link" 2>/dev/null; then
            ok "linked $link -> $target"
            return 0
        fi
        warn "could not create $link"
        return 1
    }

    if link_one "$VENV_DIR/bin/jarvis" "$LINK"; then
        JARVIS_ON_PATH=1
    else
        info "Run JARVIS directly instead: $VENV_DIR/bin/jarvis"
    fi

    # Linux filenames are case-sensitive, so `JARVIS` is a different command
    # from `jarvis` and would otherwise be "command not found" for a program
    # whose name is written in capitals everywhere. Link both spellings.
    link_one "$VENV_DIR/bin/jarvis" "$LOCAL_BIN/JARVIS" >/dev/null 2>&1 || true

    if [ "$JARVIS_ON_PATH" -eq 1 ]; then
        case ":${PATH:-}:" in
            *":$LOCAL_BIN:"*)
                ok "$LOCAL_BIN is on your PATH — 'jarvis' will work"
                ;;
            *)
                # Debian and Ubuntu add ~/.local/bin from ~/.profile only when
                # the directory already existed at login. If this run created
                # it, PATH will not pick it up until the next login — which is
                # exactly when "it installed fine but the command is missing"
                # gets blamed on the installer.
                JARVIS_ON_PATH=0
                warn "$LOCAL_BIN is not on your PATH in this shell."
                warn "Most distributions add it at login, but only if it already"
                warn "existed — and this run may have just created it."
                info "For this shell:   export PATH=\"$LOCAL_BIN:\$PATH\""
                info "Permanently:      log out and back in, or run: exec \$SHELL -l"
                ;;
        esac
    fi
fi

# --------------------------------------------------------------------------- #
#  Model selection
# --------------------------------------------------------------------------- #
if [ -n "$MODEL_ID" ]; then
    step "Recording the model in config.yaml"
    if [ ! -f "$ROOT/config.yaml" ] && [ -f "$ROOT/config.example.yaml" ]; then
        cp "$ROOT/config.example.yaml" "$ROOT/config.yaml"
        ok "config.yaml created from config.example.yaml"
    fi
    JARVIS_SET_MODEL="$MODEL_ID" JARVIS_CONFIG_FILE="$ROOT/config.yaml" "$VPY" - <<'PYEOF'
"""Set the model id in config.yaml without needing PyYAML.

A tag containing ':' (qwen3:4b-...) is an Ollama tag and belongs in
llm.ollama_model; anything else is a Hugging Face repo id and belongs in
llm.model.  The file is edited line by line inside the llm: block so comments
and every other setting survive untouched.
"""
import os
import re
from pathlib import Path

path = Path(os.environ["JARVIS_CONFIG_FILE"])
model = os.environ["JARVIS_SET_MODEL"].strip()
key = "ollama_model" if ":" in model else "model"

lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else ["llm:"]
out, in_llm, replaced = [], False, False
for line in lines:
    if re.match(r"^llm:\s*$", line):
        in_llm = True
        out.append(line)
        continue
    if in_llm and line and not line[0].isspace() and not line.startswith("#"):
        if not replaced:
            out.append(f"  {key}: {model}")
            replaced = True
        in_llm = False
    if in_llm and re.match(r"^\s+%s:\s" % key, line):
        out.append(f"  {key}: {model}")
        replaced = True
        continue
    out.append(line)

if not replaced:
    if not any(re.match(r"^llm:\s*$", ln) for ln in out):
        out.append("llm:")
    out.append(f"  {key}: {model}")

path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("    llm.%s: %s" % (key, model))
PYEOF
    ok "config.yaml updated"
fi

# --------------------------------------------------------------------------- #
#  systemd user service
# --------------------------------------------------------------------------- #
if [ "$WANT_SERVICE" -eq 1 ]; then
    step "Installing the systemd user service"
    if [ "$OS" != "Linux" ]; then
        warn "systemd user services are Linux-only; skipping on $OS."
    elif ! command -v systemctl >/dev/null 2>&1; then
        warn "systemctl not found — this system does not run systemd."
        info "Use the XDG autostart entry instead:"
        info "  $VPY -c 'from jarvis.linux import desktop; print(desktop.autostart_enable())'"
    else
        set +e
        "$VPY" - <<'PYEOF'
from jarvis.linux import service

result = service.install()
if not result.ok:
    print("    install failed: %s" % result.error)
    raise SystemExit(1)
print("    unit: %s" % result.output["path"])
print("    exec: %s" % result.output["command"])

enabled = service.enable()
print("    enable: %s" % ("ok" if enabled.ok else enabled.error))

status = service.status()
if status.get("linger") is True:
    print("    lingering: already enabled")
else:
    print("    lingering: NOT enabled")
PYEOF
        SVC_STATUS=$?
        set -e
        if [ $SVC_STATUS -ne 0 ]; then
            warn "The service could not be installed; see the message above."
        else
            ok "installed and enabled"
            info "Start it now:   systemctl --user start jarvis.service"
            info "Watch it:       journalctl --user -u jarvis.service -f"
            info ""
            warn "A user service is torn down when your last session ends, so JARVIS"
            warn "would stop when you log out or close the lid — silently, with no"
            warn "error anywhere. To keep it running with nobody logged in, run:"
            printf "        ${YELLOW}loginctl enable-linger %s${RESET}\n" "${USER:-$(id -un)}"
        fi
    fi
fi

# --------------------------------------------------------------------------- #
#  Done
# --------------------------------------------------------------------------- #
# Print the invocation the user can actually type, not the one we wish worked.
if [ "$JARVIS_ON_PATH" -eq 1 ]; then
    RUN_AS="jarvis"
    PATH_NOTE=""
else
    RUN_AS="$JARVIS_BIN"
    PATH_NOTE="
  'jarvis' on its own is not on your PATH yet. Either:
      export PATH=\"\$HOME/.local/bin:\$PATH\"     # add to ~/.bashrc to persist
      source $VENV_DIR/bin/activate           # or activate the virtualenv"
fi

printf "\n${GREEN}==> Installed.${RESET}\n"
cat <<EOF

${DIM}  Try it:
      $RUN_AS doctor        check what is installed
      $RUN_AS say           audition the British voice
      $RUN_AS chat          talk to it by keyboard
      $RUN_AS voice         hands-free
$PATH_NOTE

  A local language model is required for conversation. The quickest route:
      curl -fsSL https://ollama.com/install.sh | sh
      ollama pull qwen3:4b-instruct-2507-q4_K_M
  then set  llm.backend: ollama  in config.yaml.

  Run it in the background:
      ./install.sh --service
      loginctl enable-linger \$USER
${RESET}
EOF
