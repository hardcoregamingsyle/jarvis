#!/usr/bin/env bash
#
# JARVIS installer AND updater for Linux (and, with fewer features, macOS).
#
# One command installs everything JARVIS needs in order to actually answer you:
# the Python package, the inference runtime (Ollama), the language model
# weights, and the speech models for talking and listening.  Running it a
# second time updates all of the same things and prints a per-component
# summary saying what moved, what was already current, and what was skipped.
#
#   ./install.sh                     install or update everything (the default)
#   ./install.sh --update            the same thing, said out loud
#   ./install.sh --no-update         repair only: no git pull, no version bumps
#   ./install.sh --lean              default profile: voice, memory, machine control
#   ./install.sh --full              adds torch + transformers + AirLLM (many GB)
#   ./install.sh --min               the Python package only: no runtime, no weights
#   ./install.sh --no-voice          skip the speech model downloads
#   ./install.sh --no-model          skip Ollama and every model download
#   ./install.sh --only-main-model   do not also fetch the small interactive model
#   ./install.sh --model ID          use ID (HF repo id or Ollama tag) as the main model
#   ./install.sh --venv PATH         use a different virtualenv directory
#   ./install.sh --service           install + enable the systemd *user* services
#   ./install.sh --vllm              set up the vLLM path (Linux-only, separate install)
#   ./install.sh --no-link           do not symlink ~/.local/bin/jarvis
#   ./install.sh --help              print this text
#
# Nothing here is run with sudo.  Where a system package is needed the exact
# command is printed for you to run yourself, and Ollama is installed from its
# official release tarball into your home directory rather than through the
# curl|sh installer, which requires root and writes a system-wide service.
#
# Nothing is deleted, and no file this installer did not create is overwritten.
# Outside this directory it writes only:
#
#     ~/.local/bin/jarvis        symlink to the console script (--no-link to skip)
#     ~/.local/bin/JARVIS        the same link under the capitalised name
#     ~/.local/bin/ollama        symlink to the Ollama binary below
#     ~/.local/share/jarvis      voice and transcription models, and the Ollama
#                                runtime itself in ~/.local/share/jarvis/ollama
#     ~/.ollama                  the model weight cache
#     ~/.config/systemd/user     user service units (--service only)
#
# Each of those is created only if absent, or replaced only when this installer
# is the thing that created it.  config.yaml in this directory is edited in
# place: the llm keys below are rewritten and every other setting, comment and
# blank line is preserved byte for byte.
#
# Every large download reports its size and the free space on the target disk
# BEFORE it starts, refuses to start when it would not fit, resumes if it was
# interrupted, and is skipped entirely when it is already current.
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
WANT_UPDATE=1
WANT_MODEL=1
WANT_FAST_MODEL=1
MODEL_ID=""

# The small model is not a downgrade, it is half the architecture: the main
# (dense, 27B) model does the thinking and the tool work, and this one turns
# the result into the sentence that is actually spoken. See llm.voice_model in
# config.example.yaml. On a CPU-only laptop the main model runs at roughly
# 0.5-1 tok/s, so without the split there is no conversation to be had.
FAST_TAG="qwen3:4b-instruct-2507-q4_K_M"
# Smaller still, and the default llm.voice_model: purely a mouth, so 1.7B is
# ample and it starts speaking in well under a second.
VOICE_TAG="qwen3:1.7b"
FALLBACK_MAIN_TAG="qwen3.8:27b"

# Print the comment block at the top of this file. Deriving the help from the
# header means the two can never drift apart, which a hard-coded line range did.
usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

# Saved before the parse loop consumes them: if `git pull` brings in a new
# version of this script we re-exec it with the arguments we were given.
ORIGINAL_ARGS=("$@")

while [ $# -gt 0 ]; do
    case "$1" in
        --full) PROFILE="full" ;;
        --min) PROFILE="min" ;;
        --lean) PROFILE="lean" ;;
        --no-voice) DOWNLOAD_VOICE=0 ;;
        --service) WANT_SERVICE=1 ;;
        --vllm) WANT_VLLM=1 ;;
        --no-link) WANT_LINK=0 ;;
        --update) WANT_UPDATE=1 ;;
        --no-update) WANT_UPDATE=0 ;;
        --no-model) WANT_MODEL=0 ;;
        --only-main-model) WANT_FAST_MODEL=0 ;;
        --model)
            if [ $# -lt 2 ]; then echo "--model requires a model id" >&2; exit 1; fi
            MODEL_ID="$2"; shift
            ;;
        --venv)
            if [ $# -lt 2 ]; then echo "--venv requires a path" >&2; exit 1; fi
            VENV="$2"; shift
            ;;
        -h|--help) usage; exit 0 ;;
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

# --------------------------------------------------------------------------- #
#  The summary
#
#  The question this installer has to answer out loud is "did re-running it
#  update everything?".  Every section therefore records one line, and the run
#  ends by printing them together: updated / installed / already current /
#  skipped (why) / failed (why).  Nothing else in the output is a substitute:
#  a wall of pip and download chatter does not tell you what state you are in.
# --------------------------------------------------------------------------- #
SUMMARY=""
SUMMARY_PRINTED=0

record() {
    # component | state | detail   (details must not contain a pipe)
    SUMMARY="${SUMMARY}${1}|${2}|${3:-}
"
}

print_summary() {
    if [ "$SUMMARY_PRINTED" -eq 1 ] || [ -z "$SUMMARY" ]; then
        return 0
    fi
    SUMMARY_PRINTED=1
    printf "\n${CYAN}==> Summary${RESET}\n"
    printf "%s" "$SUMMARY" | while IFS='|' read -r comp state detail; do
        [ -n "$comp" ] || continue
        colour="$RESET"
        case "$state" in
            updated|installed)  colour="$GREEN" ;;
            "already current")  colour="$GREEN" ;;
            skipped)            colour="$DIM" ;;
            failed)             colour="$YELLOW" ;;
        esac
        printf "    %-18s %b%-16s%b %s\n" "$comp" "$colour" "$state" "$RESET" "$detail"
    done
}

# Print it even if a later section dies, so a half-finished run still says
# where it got to.
trap print_summary EXIT

# --------------------------------------------------------------------------- #
#  Disk space
#
#  A 16 GB model that runs out of room at 90% is a wasted afternoon and a
#  confusing error.  Everything large is therefore preceded by a check that
#  prints the size and the free space and refuses rather than starting.
# --------------------------------------------------------------------------- #

# Free space in decimal GB (10^9), matching how model sizes are quoted.
# Prints nothing at all when it cannot tell, which callers treat as "unknown".
free_gb() {
    local probe="$1" parent
    while [ ! -d "$probe" ]; do
        parent="$(dirname "$probe")"
        if [ "$parent" = "$probe" ]; then break; fi
        probe="$parent"
    done
    if [ ! -d "$probe" ]; then return 0; fi
    df -Pk "$probe" 2>/dev/null | awk 'NR == 2 { printf "%d", int($4 * 1024 / 1000000000) }'
}

# Download size in GB for an Ollama tag. The figures come from
# jarvis.llm.models.KNOWN_MODELS; this table is the fallback used when the
# package cannot be imported yet, and errs upwards for anything unrecognised.
model_size_gb() {
    case "$1" in
        *0.6b*)                       echo 1 ;;
        *1.7b*)                       echo 2 ;;
        *:4b*|*4b-*)                  echo 3 ;;
        *:8b*|*8b-*)                  echo 5 ;;
        *:14b*)                       echo 9 ;;
        *30b-a3b*|*30b*)              echo 19 ;;
        *27b*)                        echo 17 ;;
        *:32b*)                       echo 20 ;;
        *:70b*)                       echo 42 ;;
        *235b*)                       echo 140 ;;
        *)                            echo 20 ;;
    esac
}

# require_disk_space DIR NEED_GB LABEL -> 0 if it fits, 1 if it does not.
# Unknown free space is not treated as failure: refusing to install because df
# is unavailable would be worse than trying.
require_disk_space() {
    local dir="$1" need="$2" label="$3" free
    free="$(free_gb "$dir")"
    if [ -z "$free" ]; then
        warn "could not read free space for $dir — continuing without the check"
        return 0
    fi
    info "$label: needs ~${need} GB, ${free} GB free on $dir"
    if [ "$free" -lt "$need" ]; then
        warn "NOT ENOUGH DISK SPACE for $label."
        warn "Needs about ${need} GB including headroom; ${dir} has ${free} GB free."
        return 1
    fi
    return 0
}

# --------------------------------------------------------------------------- #
#  Bridge to jarvis.runtime
#
#  jarvis/runtime/ owns the rootless Ollama install, the model pulls and the
#  speech assets.  This function is the only way the shell talks to it, and a
#  checkout that predates that package degrades to printing the manual
#  commands rather than failing.
#
#  Exit codes:  0 changed   1 failed   3 already current   4 skipped   5 absent
# --------------------------------------------------------------------------- #
RT_CHANGED=0; RT_FAILED=1; RT_CURRENT=3; RT_SKIPPED=4; RT_ABSENT=5

runtime_py() {
    JARVIS_RT_ACTION="$1" \
    JARVIS_RT_ARG="${2:-}" \
    JARVIS_RT_UPDATE="$WANT_UPDATE" \
    JARVIS_RT_CONFIG="$ROOT/config.yaml" \
        "$VPY" - <<'PYEOF'
"""Installer -> jarvis.runtime bridge.

Bound to the real jarvis.runtime API rather than guessing at it, so a rename
on that side shows up here as a loud FAILED line naming the function instead
of a silent wrong call.  Progress goes to stderr so that the value-returning
actions can still be captured with command substitution.  Nothing raises:
every failure becomes an exit code the shell turns into one summary line.
"""
from __future__ import annotations

import os
import sys
import time

CHANGED, FAILED, CURRENT, SKIPPED, ABSENT = 0, 1, 3, 4, 5

ACTION = os.environ.get("JARVIS_RT_ACTION", "")
ARG = os.environ.get("JARVIS_RT_ARG", "").strip()
UPDATE = os.environ.get("JARVIS_RT_UPDATE", "0") == "1"
CONFIG_PATH = os.environ.get("JARVIS_RT_CONFIG", "")

#: Actions reported by jarvis.runtime that mean bytes actually moved.
FETCHED = ("pulled", "fetched", "downloaded", "installed", "upgraded", "written")
#: ...and the ones that mean it was already there.
UNCHANGED = ("present", "current", "skipped", "already-running", "external")

#: True while the last thing written to stderr was an unterminated \r
#: progress line, so say() knows to start its own line cleanly rather than
#: overwriting the tail of a progress bar.
_PROGRESS_LINE_OPEN = [False]


def say(text):
    if _PROGRESS_LINE_OPEN[0]:
        sys.stderr.write("\n")
        _PROGRESS_LINE_OPEN[0] = False
    sys.stderr.write("    %s\n" % text)


def make_progress():
    """A throttled stderr progress printer for a multi-gigabyte transfer.

    Without this, ensure_model()/install()/provision_all() ran in complete
    silence for as long as the download took -- easily tens of minutes for a
    16 GB model on an ordinary connection -- and a run that was working
    perfectly looked identical to one that had hung after "checking space
    available". jarvis.runtime already reports progress through the
    `progress` callback every action below now accepts; nothing was ever
    passing one in.

    Throttled to roughly one update per second per (phase, tag) so a fast
    local link does not flood the terminal with one line per network chunk.
    """
    is_tty = sys.stderr.isatty()
    last = {}

    def emit(event):
        phase = str(event.get("phase") or "progress")
        tag = str(event.get("tag") or "")
        key = phase + ":" + tag
        message = event.get("message")
        completed = event.get("completed")
        total = event.get("total")
        percent = event.get("percent")
        if percent is None and isinstance(completed, (int, float)) \
                and isinstance(total, (int, float)) and total:
            percent = round(completed * 100.0 / total, 1)
        done = percent is not None and percent >= 100

        now = time.monotonic()
        prev_t, prev_p = last.get(key, (0.0, None))
        first = key not in last
        # A message ("pulling X (~16 GB)") always shows; numeric progress is
        # throttled by time and by a minimum percent movement.
        if not first and not message and not done:
            if now - prev_t < 1.0 or (prev_p is not None and percent is not None
                                      and abs(percent - prev_p) < 1):
                return
        last[key] = (now, percent)

        label = phase if not tag else "%s %s" % (phase, tag)
        if message:
            line = "%-16s %s" % (label, message)
        elif percent is not None:
            have_gb = (completed or 0) / 1e9
            want_gb = (total or 0) / 1e9
            line = ("%-16s %5.1f%%  (%.2f / %.2f GB)" % (label, percent, have_gb, want_gb)
                    if want_gb else "%-16s %5.1f%%" % (label, percent))
        elif isinstance(completed, (int, float)) and completed:
            line = "%-16s %.2f GB" % (label, completed / 1e9)
        else:
            return

        if is_tty and not message:
            sys.stderr.write("\r    " + line + " " * 8)
            _PROGRESS_LINE_OPEN[0] = True
            if done:
                sys.stderr.write("\n")
                _PROGRESS_LINE_OPEN[0] = False
        else:
            if _PROGRESS_LINE_OPEN[0]:
                sys.stderr.write("\n")
                _PROGRESS_LINE_OPEN[0] = False
            sys.stderr.write("    " + line + "\n")
        sys.stderr.flush()

    return emit


PROGRESS = make_progress()


def load_cfg():
    try:
        from jarvis.core.config import load_config
    except Exception as exc:
        say("config unavailable: %s" % exc)
        return None
    try:
        if CONFIG_PATH and os.path.exists(CONFIG_PATH):
            return load_config(CONFIG_PATH)
        return load_config()
    except Exception as exc:
        say("could not read config.yaml: %s" % exc)
        return None


def runtime():
    try:
        import jarvis.runtime as rt
        return rt
    except Exception as exc:
        say("jarvis.runtime is not available here (%s)" % exc)
        return None


def verdict(result, *, changed_when_ok=True):
    """(exit code, reason) from the {ok, action, reason} dicts jarvis.runtime returns."""
    if not isinstance(result, dict):
        return (CHANGED if result else FAILED), ""
    reason = str(result.get("reason") or "")
    if not result.get("ok"):
        return FAILED, reason
    action = str(result.get("action") or "")
    if action in UNCHANGED:
        return CURRENT, reason
    if action in FETCHED:
        return CHANGED, reason
    return (CHANGED if changed_when_ok else CURRENT), reason


# --------------------------------------------------------------------------- #
#  Actions that print a value
# --------------------------------------------------------------------------- #
def action_configured_tag():
    cfg = load_cfg()
    tag = ""
    if cfg is not None:
        tag = str(getattr(getattr(cfg, "llm", None), "ollama_model", "") or "")
    sys.stdout.write(tag.strip() + "\n")
    return CHANGED if tag.strip() else FAILED


def action_resolve_tag():
    """Map a Hugging Face repo id onto the Ollama tag that serves it."""
    if ":" in ARG:
        sys.stdout.write(ARG + "\n")
        return CHANGED
    try:
        from jarvis.llm import models
        spec = models.resolve(ARG)
    except Exception as exc:
        say("no Ollama tag known for %s (%s)" % (ARG, exc))
        return FAILED
    tag = str(getattr(spec, "ollama_tag", "") or "")
    sys.stdout.write(tag + "\n")
    return CHANGED if tag else FAILED


def action_model_size():
    """Download size in whole GB, rounded up, for the disk-space check."""
    try:
        from jarvis.llm import models
        figures = models.estimate_footprint(ARG)
        size = float(figures.get("download_gb") or 0.0)
    except Exception:
        return FAILED
    if size <= 0:
        return FAILED
    sys.stdout.write("%d\n" % int(size + 0.999))
    return CHANGED


# --------------------------------------------------------------------------- #
#  Actions that change the machine
# --------------------------------------------------------------------------- #
def action_ollama():
    rt = runtime()
    if rt is None:
        return ABSENT
    cfg = load_cfg()
    try:
        plan = rt.ollama.install_plan(cfg)
    except Exception as exc:
        say("install_plan failed: %s" % exc)
        plan = {}
    if plan:
        say("plan: %s" % (plan.get("action") or "?"))
        if plan.get("url"):
            say("from: %s" % plan["url"])
            say("size: %s GB -> %s" % (plan.get("size_gb"), plan.get("target_dir")))
        if plan.get("reason"):
            say(plan["reason"])
        # --no-update promises no version bumps, and that has to include this one.
        if plan.get("action") == "upgrade" and not UPDATE:
            say("a newer ollama exists; left alone because --no-update was given")
            return CURRENT
    try:
        result = rt.ollama.install(cfg, progress=PROGRESS)
    except Exception as exc:
        say("ollama.install failed: %s" % exc)
        return FAILED
    code, reason = verdict(result, changed_when_ok=bool(result.get("installed_now")))
    if reason:
        say(reason)
    return code


def action_serve():
    """A pull needs a server; ensure_model reports 'no-server' without one."""
    rt = runtime()
    if rt is None:
        return ABSENT
    cfg = load_cfg()
    try:
        result = rt.ollama.start_server(cfg)
    except Exception as exc:
        say("start_server failed: %s" % exc)
        return FAILED
    code, reason = verdict(result)
    if reason:
        say(reason)
    return code


def action_model():
    if not ARG:
        return SKIPPED
    rt = runtime()
    if rt is None:
        return ABSENT
    cfg = load_cfg()
    try:
        # On an update run, re-check the tag against the registry rather than
        # merely confirming that some copy of it exists.  An Ollama tag is a
        # moving target -- qwen3.8:27b is re-pointed upstream when the model is
        # re-quantised -- so "present" and "current" are different claims.
        # Ollama compares digests and transfers only changed layers, so this is
        # a manifest fetch when the model is already up to date.
        result = rt.ollama.ensure_model(cfg, tag=ARG, refresh=UPDATE, progress=PROGRESS)
    except Exception as exc:
        say("ensure_model failed: %s" % exc)
        return FAILED
    code, reason = verdict(result)
    if reason:
        say(reason)
    return code


def action_assets():
    """Voice and transcription only — the runtime and weights are pulled above."""
    rt = runtime()
    if rt is None:
        return ABSENT
    cfg = load_cfg()
    try:
        result = rt.assets.provision_all(cfg, skip=["ollama", "model"], progress=PROGRESS)
    except Exception as exc:
        say("provision_all failed: %s" % exc)
        return FAILED
    for line in result.get("summary") or []:
        if not line.startswith("[ok] ollama") and not line.startswith("[ok] model"):
            say(line)
    components = result.get("components") or {}
    interesting = [components.get(name) or {} for name in ("voice", "stt")]
    if not result.get("ok"):
        return FAILED
    if any(str(entry.get("action") or "") in FETCHED for entry in interesting):
        return CHANGED
    return CURRENT


def action_ollama_service():
    """Write and enable the Ollama systemd user unit (jarvis.runtime names it)."""
    rt = runtime()
    if rt is None:
        return ABSENT
    cfg = load_cfg()
    try:
        result = rt.ollama.install_service(cfg)
    except Exception as exc:
        say("install_service failed: %s" % exc)
        return FAILED
    if not result.get("ok"):
        say(str(result.get("reason") or "the unit could not be written"))
        return FAILED
    say(str(result.get("reason") or ""))
    say("path: %s" % result.get("path"))
    # ok=True only means the unit file is on disk.  install_service reports a
    # failed `systemctl --user enable --now` in "enabled", and a summary line
    # saying "enabled" when it is not is exactly the kind of optimistic
    # constant this installer is supposed to have stopped printing.
    if not result.get("enabled"):
        return FAILED
    return CHANGED if result.get("written") else CURRENT


def action_status():
    rt = runtime()
    if rt is None:
        return ABSENT
    try:
        result = rt.status(load_cfg())
    except Exception as exc:
        say("status unavailable: %s" % exc)
        return FAILED
    say("ready: %s" % ("yes" if result.get("ready") else "no"))
    missing = result.get("missing") or []
    if missing:
        say("missing: %s" % ", ".join(str(item) for item in missing))
    runtime_info = result.get("ollama") or {}
    if isinstance(runtime_info, dict):
        say("ollama  installed=%s serving=%s model=%s present=%s" % (
            bool(runtime_info.get("installed")),
            bool(runtime_info.get("serving")),
            runtime_info.get("model"),
            bool(runtime_info.get("model_present")),
        ))
    for name in ("voice", "stt"):
        entry = result.get(name) or {}
        if isinstance(entry, dict):
            say("%-7s %s" % (name, entry.get("reason") or entry.get("summary") or ""))
    return CHANGED if result.get("ready") else CURRENT


ACTIONS = {
    "configured-tag": action_configured_tag,
    "resolve-tag": action_resolve_tag,
    "model-size": action_model_size,
    "ollama": action_ollama,
    "serve": action_serve,
    "model": action_model,
    "assets": action_assets,
    "ollama-service": action_ollama_service,
    "status": action_status,
}

handler = ACTIONS.get(ACTION)
if handler is None:
    say("unknown runtime action: %s" % ACTION)
    raise SystemExit(FAILED)
try:
    raise SystemExit(handler())
except SystemExit:
    raise
except Exception as exc:            # never let the bridge take the installer down
    say("%s: %s" % (ACTION, exc))
    raise SystemExit(FAILED)
PYEOF
}

# runtime_py, but for the actions that print a value; empty on any failure.
runtime_capture() {
    local out
    set +e
    out="$(runtime_py "$1" "${2:-}" 2>/dev/null)"
    set -e
    printf "%s" "$out" | tr -d '\r\n'
}

cat <<'BANNER'
    ___  _   _____   _   _ ___ ___
   |_ _|/_\ | _ \ \ / /_| / __/ __|
    | |/ _ \|   /\ V / | \__ \__ \
   |___/_/ \_\_|_\ \_/  |_|___/___/   installer for Linux
BANNER

# --------------------------------------------------------------------------- #
#  The checkout itself
#
#  This runs first so the rest of the install is the code you just fetched.
#  It is deliberately timid: a working tree with local edits, a detached HEAD
#  or a branch with no upstream is reported and left completely alone.  This
#  installer never throws away work you have not committed.
# --------------------------------------------------------------------------- #
step "Updating the checkout"

PULLED_RANGE="${JARVIS_INSTALL_PULLED:-}"
if [ -n "$PULLED_RANGE" ]; then
    ok "already pulled this run: $PULLED_RANGE"
    record "Repository" "updated" "$PULLED_RANGE"
elif [ "$WANT_UPDATE" -eq 0 ]; then
    info "skipped (--no-update)"
    record "Repository" "skipped" "--no-update"
elif ! command -v git >/dev/null 2>&1; then
    info "git is not installed — nothing to pull"
    record "Repository" "skipped" "git not installed"
elif ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    info "not a git checkout — update it however you obtained it"
    record "Repository" "skipped" "not a git checkout"
elif [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    # Tracked files have been edited. Untracked ones are deliberately ignored:
    # a stray note in the directory is no reason to refuse an update, and git
    # itself refuses a fast-forward that would overwrite one.
    warn "tracked files have local changes, so nothing was pulled."
    warn "Commit them, or set them aside yourself, and re-run — 'git status'"
    warn "lists them. The installer will not discard your work to update."
    record "Repository" "skipped" "local changes present"
elif ! BRANCH="$(git -C "$ROOT" symbolic-ref --quiet --short HEAD 2>/dev/null)"; then
    info "detached HEAD — check out a branch to be able to update"
    record "Repository" "skipped" "detached HEAD"
elif ! git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
    info "branch '$BRANCH' tracks nothing upstream"
    record "Repository" "skipped" "no upstream for $BRANCH"
else
    BEFORE_SHA="$(git -C "$ROOT" rev-parse HEAD)"
    set +e
    git -C "$ROOT" pull --ff-only
    PULL_STATUS=$?
    set -e
    AFTER_SHA="$(git -C "$ROOT" rev-parse HEAD)"
    if [ $PULL_STATUS -ne 0 ]; then
        warn "git pull --ff-only failed (offline, or the branch has diverged)."
        warn "Nothing was changed. Reconcile it yourself and re-run."
        record "Repository" "failed" "git pull --ff-only returned $PULL_STATUS"
    elif [ "$BEFORE_SHA" = "$AFTER_SHA" ]; then
        ok "already up to date ($BRANCH)"
        record "Repository" "already current" "$BRANCH"
    else
        RANGE="${BEFORE_SHA:0:8}..${AFTER_SHA:0:8}"
        ok "updated $BRANCH  $RANGE"
        record "Repository" "updated" "$RANGE"
        # bash reads a script incrementally, so continuing to run an install.sh
        # that git has just rewritten underneath us would execute a spliced
        # mixture of both versions. Start the new one instead.
        if [ "${JARVIS_INSTALL_REEXEC:-0}" != "1" ]; then
            ok "the installer itself changed — restarting it"
            export JARVIS_INSTALL_REEXEC=1
            export JARVIS_INSTALL_PULLED="$RANGE"
            exec "${BASH:-bash}" "$0" ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}
        fi
    fi
fi

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
    record "System libs" "skipped" "missing:$NEEDS (printed the command)"
else
    ok "all present"
    record "System libs" "already current" "all present"
fi

# --------------------------------------------------------------------------- #
#  Hardware detection (report only)
#
#  Advisory: prints what jarvis.core.hardware / jarvis.llm.planner see on this
#  machine so the download-plan numbers further down have context. Uses
#  $PYTHON here, not $VPY -- this runs before the virtual environment exists.
#  That works because every third-party import inside those two modules is
#  lazy (see jarvis/core/hardware.py and jarvis/llm/planner.py): `python -m
#  jarvis hardware` runs fine against a bare interpreter with nothing
#  pip-installed at all. This step is pure reporting -- it never sets or reads
#  MAIN_TAG/FAST_TAG and has no bearing on the download-plan or
#  model-selection logic further down, which are computed independently. A
#  checkout that predates this feature, or any other failure, degrades to a
#  one-line note rather than aborting the install -- the same tolerance
#  already given to jarvis.runtime being absent.
# --------------------------------------------------------------------------- #
step "Detecting hardware"
HARDWARE_REPORT="$("$PYTHON" -m jarvis hardware 2>/dev/null)" || true
if [ -n "$HARDWARE_REPORT" ]; then
    while IFS= read -r hw_line; do
        info "$hw_line"
    done <<< "$HARDWARE_REPORT"
else
    warn "hardware detection is not available in this checkout yet; continuing"
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
      delete the directory $VENV_DIR
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
#
#  --upgrade is the whole difference between "installed" and "updated": a plain
#  `pip install -r requirements.txt` is satisfied by whatever is already there
#  and silently leaves every pinned dependency at the version it found. That is
#  why re-running the old installer never moved anything.
# --------------------------------------------------------------------------- #
PIP_ARGS=(--disable-pip-version-check)
if [ "$WANT_UPDATE" -eq 1 ]; then
    PIP_ARGS+=(--upgrade)
fi

step "Installing the '$PROFILE' profile"
info "pip arguments: ${PIP_ARGS[*]}"
set +e
case "$PROFILE" in
    min)
        "$VPY" -m pip install "${PIP_ARGS[@]}" -e .
        ;;
    lean)
        "$VPY" -m pip install "${PIP_ARGS[@]}" -e . \
            && "$VPY" -m pip install "${PIP_ARGS[@]}" -r requirements.txt
        ;;
    full)
        "$VPY" -m pip install "${PIP_ARGS[@]}" -e .
        warn "This downloads torch and friends — several gigabytes."
        require_disk_space "$ROOT" 12 "the full profile (torch and friends)" \
            && "$VPY" -m pip install "${PIP_ARGS[@]}" -r requirements-full.txt
        ;;
esac
STATUS=$?
set -e
if [ $STATUS -ne 0 ]; then
    warn "Some packages failed. JARVIS will still run in degraded mode."
    warn "Run '\$JARVIS_BIN doctor' to see exactly what is missing."
    record "Python packages" "failed" "$PROFILE profile — see the output above"
elif [ "$WANT_UPDATE" -eq 1 ]; then
    record "Python packages" "updated" "$PROFILE profile, --upgrade"
else
    record "Python packages" "installed" "$PROFILE profile, no version bumps"
fi

# --------------------------------------------------------------------------- #
#  Model selection -> config.yaml
#
#  Written before anything is downloaded so that the pull below and JARVIS at
#  runtime agree about which model this machine has.
# --------------------------------------------------------------------------- #
MAIN_TAG=""
if [ "$WANT_MODEL" -eq 1 ] && [ "$PROFILE" != "min" ]; then
    if [ -n "$MODEL_ID" ]; then
        MAIN_TAG="$(runtime_capture resolve-tag "$MODEL_ID")"
        [ -n "$MAIN_TAG" ] || MAIN_TAG="$MODEL_ID"
    else
        MAIN_TAG="$(runtime_capture configured-tag)"
        [ -n "$MAIN_TAG" ] || MAIN_TAG="$FALLBACK_MAIN_TAG"
    fi
fi

if [ -n "$MODEL_ID" ] || [ -n "$MAIN_TAG" ]; then
    step "Recording the model in config.yaml"
    if [ ! -f "$ROOT/config.yaml" ] && [ -f "$ROOT/config.example.yaml" ]; then
        cp "$ROOT/config.example.yaml" "$ROOT/config.yaml"
        ok "config.yaml created from config.example.yaml"
    fi

    SET_LINES="backend=ollama"
    if [ -n "$MAIN_TAG" ]; then
        SET_LINES="$SET_LINES
ollama_model=$MAIN_TAG"
    fi
    if [ -n "$MODEL_ID" ] && [ "${MODEL_ID#*:}" = "$MODEL_ID" ]; then
        # No colon, so it is a Hugging Face repo id rather than an Ollama tag.
        SET_LINES="$SET_LINES
model=$MODEL_ID"
    fi
    # Record the voice model too. Pulling it without pointing the config at it
    # would leave the two-model split silently switched off, which is exactly
    # the "capable but unusably slow" configuration it exists to avoid.
    if [ "$WANT_MODEL" -eq 1 ] && [ -n "$VOICE_TAG" ]; then
        SET_LINES="$SET_LINES
voice_model=$VOICE_TAG"
    fi

    set +e
    JARVIS_SET_MODEL="$SET_LINES" JARVIS_CONFIG_FILE="$ROOT/config.yaml" "$VPY" - <<'PYEOF'
"""Set keys inside the llm: block of config.yaml without needing PyYAML.

The file is edited line by line so comments, ordering and every other setting
survive untouched; only the keys named in JARVIS_SET_MODEL are rewritten, and
any that were not already present are appended to the end of the block.

Three rules, each of which this got wrong once:

* **Match the block's own indentation.**  Appending two-space keys to a block
  written with four produced a file PyYAML refuses to parse, and load_config
  does not catch YAMLError — so a perfectly ordinary formatting choice turned
  every later `jarvis` command into a traceback, with no backup of the file
  that holds the api_key.
* **Do not overturn a deliberate choice.**  `backend` is only moved off "auto"
  (or an empty value, or the one we would write anyway).  Somebody who set
  `backend: vllm` — which the --vllm section of this installer tells them to
  do — must not have it silently reset on the next run.
* **Say whether anything actually changed**, so the summary is not a constant.

Exit: 0 rewritten, 3 already correct, 1 refused.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

CHANGED, FAILED, CURRENT = 0, 1, 3

#: Values of a key that this installer is allowed to overwrite.  Anything else
#: was chosen by the owner on purpose and is left exactly as it is.
OVERWRITABLE = {"backend": {"", "auto", "ollama"}}

path = Path(os.environ["JARVIS_CONFIG_FILE"])
wanted = []
for raw in os.environ.get("JARVIS_SET_MODEL", "").splitlines():
    raw = raw.strip()
    if not raw or "=" not in raw:
        continue
    key, _, value = raw.partition("=")
    wanted.append((key.strip(), value.strip()))
if not wanted:
    raise SystemExit(CURRENT)

by_key = dict(wanted)
try:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
except OSError as exc:
    print("    could not read %s: %s" % (path, exc))
    raise SystemExit(FAILED)
lines = original.splitlines()

out = []
done = set()
kept = []                    # keys deliberately left at the owner's value
in_llm = False
seen_llm = False
indent = ["  "]              # the block's own indentation, learned from it
indent_known = [False]


def flush(target):
    for key, value in wanted:
        if key not in done:
            target.append("%s%s: %s" % (indent[0], key, value))
            done.add(key)


for line in lines:
    if re.match(r"^llm:\s*$", line):
        in_llm = True
        seen_llm = True
        out.append(line)
        continue
    if in_llm and line and not line[0].isspace() and not line.startswith("#"):
        flush(out)
        in_llm = False
    if in_llm:
        match = re.match(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*)\s*:(.*)$", line)
        if match:
            # The first key inside the block fixes the indentation for every
            # key we append, and tells us which lines are direct children of
            # llm: rather than of some nested mapping under it.
            if not indent_known[0]:
                indent[0] = match.group(1)
                indent_known[0] = True
            key = match.group(2)
            if (match.group(1) == indent[0]
                    and key in by_key and key not in done):
                current = match.group(3).strip()
                allowed = OVERWRITABLE.get(key)
                if allowed is not None and current not in allowed:
                    done.add(key)
                    kept.append((key, current))
                    out.append(line)
                    continue
                out.append("%s%s: %s" % (match.group(1), key, by_key[key]))
                done.add(key)
                continue
    out.append(line)

if in_llm:
    flush(out)
if not seen_llm:
    out.append("llm:")
    indent[0] = "  "
    flush(out)

updated = "\n".join(out) + "\n"
if updated == original:
    for key, value in wanted:
        if key not in dict(kept):
            print("    llm.%s: %s (already set)" % (key, value))
    raise SystemExit(CURRENT)

# Belt and braces: never replace a file that parses with one that does not.
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None
if yaml is not None:
    try:
        yaml.safe_load(original)
    except Exception:
        pass                 # it was already unparseable; we cannot make it worse
    else:
        try:
            yaml.safe_load(updated)
        except Exception as exc:
            print("    refusing to write %s: the result would not parse (%s)"
                  % (path, exc.__class__.__name__))
            raise SystemExit(FAILED)

try:
    path.write_text(updated, encoding="utf-8")
except OSError as exc:
    print("    could not write %s: %s" % (path, exc))
    raise SystemExit(FAILED)
for key, value in wanted:
    if key in dict(kept):
        continue
    print("    llm.%s: %s" % (key, value))
for key, current in kept:
    print("    llm.%s: left at '%s' — you set it deliberately" % (key, current))
raise SystemExit(CHANGED)
PYEOF
    CFG_STATUS=$?
    set -e
    CFG_DETAIL="llm.backend=ollama${MAIN_TAG:+, llm.ollama_model=$MAIN_TAG}"
    case "$CFG_STATUS" in
        0) ok "config.yaml updated";        record "config.yaml" "updated" "$CFG_DETAIL" ;;
        3) ok "config.yaml already correct"; record "config.yaml" "already current" "$CFG_DETAIL" ;;
        *) warn "config.yaml was left alone; see the message above"
           record "config.yaml" "failed" "could not record the model" ;;
    esac
fi

# --------------------------------------------------------------------------- #
#  Inference runtime and model weights
#
#  This is the part that used to be a printed instruction. Ollama is fetched
#  from its official GitHub release tarball and unpacked under ~/.local, run as
#  a systemd *user* service: no root, no /usr/local, no system unit.
# --------------------------------------------------------------------------- #
OLLAMA_DATA="${OLLAMA_MODELS:-$HOME/.ollama}"
XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
OLLAMA_RUNTIME_GB=3          # tarball plus the unpacked libraries
SPEECH_ASSETS_GB=2           # piper voice ~60 MB, faster-whisper small.en ~500 MB

if [ "$WANT_MODEL" -eq 0 ]; then
    step "Skipping the inference runtime and all weights (--no-model)"
    record "Ollama runtime" "skipped" "--no-model"
    record "Main model" "skipped" "--no-model"
    record "Fast model" "skipped" "--no-model"
elif [ "$PROFILE" = "min" ]; then
    step "Skipping the inference runtime and all weights (--min)"
    record "Ollama runtime" "skipped" "--min profile"
    record "Main model" "skipped" "--min profile"
    record "Fast model" "skipped" "--min profile"
else
    # ---- what is about to happen, in numbers, before anything downloads ---- #
    MAIN_GB="$(runtime_capture model-size "$MAIN_TAG")"
    case "$MAIN_GB" in ''|*[!0-9]*) MAIN_GB="$(model_size_gb "$MAIN_TAG")" ;; esac
    FAST_GB="$(runtime_capture model-size "$FAST_TAG")"
    case "$FAST_GB" in ''|*[!0-9]*) FAST_GB="$(model_size_gb "$FAST_TAG")" ;; esac

    if [ "$MAIN_TAG" = "$FAST_TAG" ]; then
        WANT_FAST_MODEL=0
    fi

    step "Download plan"
    info "Ollama runtime                   ~${OLLAMA_RUNTIME_GB} GB   -> $XDG_DATA/jarvis/ollama"
    info "$MAIN_TAG   ~${MAIN_GB} GB  -> $OLLAMA_DATA"
    info "    the capable one: dense, thinks before answering, roughly 1 tok/s"
    info "    on a CPU-only laptop. Use it for work you will wait for."
    if [ "$WANT_FAST_MODEL" -eq 1 ]; then
        info "$FAST_TAG   ~${FAST_GB} GB   -> $OLLAMA_DATA"
        info "    the interactive one: answers at conversation speed, and is what"
        info "    makes voice usable on this hardware. --only-main-model skips it."
    else
        info "small interactive model: skipped"
    fi
    info "Downloads resume if interrupted, and existing layers are not re-fetched."

    # ---- runtime ---------------------------------------------------------- #
    step "Installing the Ollama runtime (rootless, no sudo)"
    EXISTING_OLLAMA="$(command -v ollama 2>/dev/null || true)"
    if [ -n "$EXISTING_OLLAMA" ]; then
        info "found an existing ollama at $EXISTING_OLLAMA"
    fi
    if require_disk_space "$XDG_DATA" "$OLLAMA_RUNTIME_GB" "the Ollama runtime"; then
        set +e
        runtime_py ollama
        RT_STATUS=$?
        set -e
    else
        RT_STATUS=$RT_SKIPPED
    fi
    case "$RT_STATUS" in
        "$RT_CHANGED") ok "Ollama installed"; record "Ollama runtime" "updated" "$XDG_DATA/ollama" ;;
        "$RT_CURRENT") ok "Ollama already current"; record "Ollama runtime" "already current" "${EXISTING_OLLAMA:-$XDG_DATA/ollama}" ;;
        "$RT_SKIPPED") warn "Ollama install skipped"; record "Ollama runtime" "skipped" "not enough disk, or the runtime declined" ;;
        "$RT_ABSENT")
            warn "jarvis.runtime is not in this checkout, so Ollama was not installed."
            info "Install it yourself without root:"
            info "  https://github.com/ollama/ollama/releases  ->  ollama-linux-amd64.tgz"
            info "  mkdir -p ~/.local/share/ollama && tar -C ~/.local/share/ollama -xzf ollama-linux-amd64.tgz"
            info "  ln -s ~/.local/share/ollama/bin/ollama ~/.local/bin/ollama"
            record "Ollama runtime" "failed" "jarvis.runtime missing — manual steps printed"
            ;;
        *) warn "Ollama install failed"; record "Ollama runtime" "failed" "see the output above" ;;
    esac

    # ---- the server ------------------------------------------------------- #
    # A pull goes through the running daemon: with nothing listening,
    # jarvis.runtime reports "no-server" and every model below would fail with
    # the same confusing message.
    step "Starting the Ollama server"
    set +e
    runtime_py serve
    SERVE_STATUS=$?
    set -e
    case "$SERVE_STATUS" in
        "$RT_CHANGED") ok "started"; record "Ollama server" "updated" "started for the pulls" ;;
        "$RT_CURRENT") ok "already running"; record "Ollama server" "already current" "already listening" ;;
        "$RT_ABSENT")  record "Ollama server" "skipped" "jarvis.runtime missing" ;;
        *)             warn "the server did not start"; record "Ollama server" "failed" "see the output above" ;;
    esac

if [ "$SERVE_STATUS" -eq "$RT_FAILED" ]; then
    warn "No Ollama server, so no model can be pulled. Start one with"
    warn "'ollama serve' (or ./install.sh --service) and re-run."
    record "Main model" "skipped" "no Ollama server"
    record "Fast model" "skipped" "no Ollama server"
else

    # ---- main model ------------------------------------------------------- #
    step "Fetching the main model: $MAIN_TAG"
    if require_disk_space "$OLLAMA_DATA" "$((MAIN_GB + 2))" "$MAIN_TAG"; then
        set +e
        runtime_py model "$MAIN_TAG"
        MODEL_STATUS=$?
        set -e
    else
        MODEL_STATUS=$RT_SKIPPED
        warn "Skipping $MAIN_TAG. Free some space and re-run, or pass"
        warn "--model $FAST_TAG to use the small one as your main model."
    fi
    case "$MODEL_STATUS" in
        "$RT_CHANGED") ok "$MAIN_TAG is present"; record "Main model" "updated" "$MAIN_TAG (~${MAIN_GB} GB)" ;;
        "$RT_CURRENT") ok "$MAIN_TAG already current"; record "Main model" "already current" "$MAIN_TAG" ;;
        "$RT_SKIPPED") record "Main model" "skipped" "$MAIN_TAG needs ~$((MAIN_GB + 2)) GB" ;;
        "$RT_ABSENT")
            info "Pull it yourself once ollama is running:"
            info "  ollama pull $MAIN_TAG"
            record "Main model" "failed" "jarvis.runtime missing — command printed"
            ;;
        *) record "Main model" "failed" "ollama pull $MAIN_TAG did not complete" ;;
    esac

    # ---- small interactive model ------------------------------------------ #
    if [ "$WANT_FAST_MODEL" -eq 0 ]; then
        record "Fast model" "skipped" "--only-main-model, or it is the main model"
    else
        step "Fetching the small interactive model: $FAST_TAG"
        if require_disk_space "$OLLAMA_DATA" "$((FAST_GB + 1))" "$FAST_TAG"; then
            set +e
            runtime_py model "$FAST_TAG"
            FAST_STATUS=$?
            set -e
        else
            FAST_STATUS=$RT_SKIPPED
        fi
        case "$FAST_STATUS" in
            "$RT_CHANGED") ok "$FAST_TAG is present"; record "Fast model" "updated" "$FAST_TAG (~${FAST_GB} GB)" ;;
            "$RT_CURRENT") ok "$FAST_TAG already current"; record "Fast model" "already current" "$FAST_TAG" ;;
            "$RT_SKIPPED") record "Fast model" "skipped" "not enough disk for ~$((FAST_GB + 1)) GB" ;;
            "$RT_ABSENT")
                info "  ollama pull $FAST_TAG"
                record "Fast model" "failed" "jarvis.runtime missing — command printed"
                ;;
            *) record "Fast model" "failed" "ollama pull $FAST_TAG did not complete" ;;
        esac
    fi

    # ---- the voice model --------------------------------------------------- #
    # Small enough to be nearly free, and it is what makes a dense 27B on a CPU
    # feel like a conversation rather than a batch job: it speaks the answer
    # while the big model is still working. Skipped when it is already one of
    # the two tags above.
    if [ "$VOICE_TAG" = "$MAIN_TAG" ] || [ "$VOICE_TAG" = "$FAST_TAG" ]; then
        record "Voice model" "skipped" "$VOICE_TAG is already being pulled"
    else
        VOICE_GB="$(runtime_capture model-size "$VOICE_TAG")"
        case "$VOICE_GB" in ''|*[!0-9]*) VOICE_GB="$(model_size_gb "$VOICE_TAG")" ;; esac
        step "Fetching the voice model: $VOICE_TAG"
        if require_disk_space "$OLLAMA_DATA" "$((VOICE_GB + 1))" "$VOICE_TAG"; then
            set +e
            runtime_py model "$VOICE_TAG"
            VOICE_STATUS=$?
            set -e
        else
            VOICE_STATUS=$RT_SKIPPED
        fi
        case "$VOICE_STATUS" in
            "$RT_CHANGED") ok "$VOICE_TAG is present"; record "Voice model" "updated" "$VOICE_TAG (~${VOICE_GB} GB)" ;;
            "$RT_CURRENT") ok "$VOICE_TAG already current"; record "Voice model" "already current" "$VOICE_TAG" ;;
            "$RT_SKIPPED") record "Voice model" "skipped" "not enough disk for ~$((VOICE_GB + 1)) GB" ;;
            "$RT_ABSENT")
                info "  ollama pull $VOICE_TAG"
                record "Voice model" "failed" "jarvis.runtime missing — command printed"
                ;;
            *) record "Voice model" "failed" "ollama pull $VOICE_TAG did not complete" ;;
        esac
    fi

fi          # end of "is there a server to pull with?"
fi          # end of "should anything be downloaded at all?"

# --------------------------------------------------------------------------- #
#  vLLM
# --------------------------------------------------------------------------- #
if [ "$WANT_VLLM" -eq 1 ]; then
    step "vLLM"
    if [ "$OS" != "Linux" ]; then
        warn "vLLM is Linux-only. On Windows, run it inside WSL2 or on another host"
        warn "and point llm.vllm_host at it."
        record "vLLM" "skipped" "not Linux"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        ok "NVIDIA GPU detected — installing the prebuilt CUDA wheel."
        set +e
        require_disk_space "$ROOT" 10 "the vLLM CUDA wheel" \
            && "$VPY" -m pip install vllm "${PIP_ARGS[@]}"
        VSTATUS=$?
        set -e
        if [ $VSTATUS -eq 0 ]; then
            ok "vLLM installed."
            info "Serve a model, then set llm.backend: vllm in config.yaml:"
            info "  $VENV/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192"
            record "vLLM" "updated" "CUDA wheel"
        else
            warn "vLLM install failed; see the output above."
            record "vLLM" "failed" "pip exit $VSTATUS"
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
        info "concurrent requests on a GPU. On CPU it will not beat the Ollama setup"
        info "this installer has already built for you, and it is far more work to"
        info "keep running."
        record "vLLM" "skipped" "no NVIDIA GPU; source build printed"
    fi
fi

# --------------------------------------------------------------------------- #
#  Voice model, and the transcription model that hears you
# --------------------------------------------------------------------------- #
if [ "$DOWNLOAD_VOICE" -eq 1 ] && [ "$PROFILE" != "min" ]; then
    step "Fetching the speech models (voice out, transcription in)"
    if require_disk_space "$XDG_DATA" "$SPEECH_ASSETS_GB" "the speech models"; then
        set +e
        runtime_py assets
        ASSET_STATUS=$?
        set -e
    else
        ASSET_STATUS=$RT_SKIPPED
    fi
    if [ "$ASSET_STATUS" -eq "$RT_ABSENT" ]; then
        # Older checkout: fall back to what has always worked.
        set +e
        "$VPY" -m jarvis setup
        ASSET_STATUS=$?
        set -e
    fi
    case "$ASSET_STATUS" in
        "$RT_CHANGED") ok "speech models ready"; record "Speech assets" "updated" "voice + transcription" ;;
        "$RT_CURRENT") ok "speech models already current"; record "Speech assets" "already current" "voice + transcription" ;;
        "$RT_SKIPPED") record "Speech assets" "skipped" "not enough disk space" ;;
        *)
            warn "the speech download did not finish; edge-tts or espeak will be used"
            record "Speech assets" "failed" "see the output above"
            ;;
    esac
else
    step "Skipping the speech downloads"
    "$VPY" -m jarvis setup --no-download || true
    record "Speech assets" "skipped" "--no-voice or --min"
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
        LINK_ACTION="created"
        if [ -e "$link" ] && [ "$link" -ef "$target" ]; then
            ok "already linked: $link"
            LINK_ACTION="unchanged"
            return 0
        elif [ -L "$link" ] && [ ! -e "$link" ]; then
            # A symlink pointing at nothing: the virtualenv it named was moved,
            # rebuilt elsewhere, or --venv changed. `-e` is false for a broken
            # link, so without this branch it fell through to "belongs to
            # something else" below — which is untrue, unhelpful, and left
            # `jarvis` permanently broken with no hint of why. Say what it is;
            # removing a name this installer found in place is still the
            # owner's decision, not ours.
            warn "$link is a symlink that points at nothing:"
            warn "    $(readlink "$link" 2>/dev/null)"
            warn "Its target has moved or been rebuilt. Nothing here removes a"
            warn "name it found in place, so 'jarvis' stays broken until you do:"
            warn "    rm $link  &&  ./install.sh"
            return 1
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

    LINK_ACTION=""
    if link_one "$VENV_DIR/bin/jarvis" "$LINK"; then
        JARVIS_ON_PATH=1
        case "$LINK_ACTION" in
            unchanged) record "Launcher" "already current" "$LINK" ;;
            *)         record "Launcher" "installed" "$LINK" ;;
        esac
    else
        info "Run JARVIS directly instead: $VENV_DIR/bin/jarvis"
        record "Launcher" "skipped" "$LINK belongs to something else"
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
else
    record "Launcher" "skipped" "--no-link, or no console script"
fi

# --------------------------------------------------------------------------- #
#  systemd user services
# --------------------------------------------------------------------------- #
if [ "$WANT_SERVICE" -eq 1 ]; then
    step "Installing the systemd user services"
    if [ "$OS" != "Linux" ]; then
        warn "systemd user services are Linux-only; skipping on $OS."
        record "Service" "skipped" "not Linux"
    elif ! command -v systemctl >/dev/null 2>&1; then
        warn "systemctl not found — this system does not run systemd."
        info "Use the XDG autostart entry instead:"
        info "  $VPY -c 'from jarvis.linux import desktop; print(desktop.autostart_enable())'"
        record "Service" "skipped" "no systemd"
    else
        # Ollama first: JARVIS is useless without something to think with, and
        # the model server has to be up before the assistant starts.
        # jarvis.runtime writes the unit and runs `systemctl --user daemon-reload`
        # and `enable --now` itself, so nothing is repeated here.
        if [ "$WANT_MODEL" -eq 1 ] && [ "$PROFILE" != "min" ]; then
            set +e
            runtime_py ollama-service
            OSVC_STATUS=$?
            set -e
            case "$OSVC_STATUS" in
                "$RT_CHANGED")
                    ok "the Ollama user unit was written and enabled"
                    record "Ollama service" "updated" "unit in ~/.config/systemd/user"
                    ;;
                "$RT_CURRENT")
                    ok "the Ollama user unit is already current"
                    record "Ollama service" "already current" "unit in ~/.config/systemd/user"
                    ;;
                "$RT_ABSENT")
                    record "Ollama service" "skipped" "jarvis.runtime missing"
                    ;;
                *)
                    warn "the Ollama user unit was not installed or not enabled;"
                    warn "start it by hand with: ollama serve"
                    record "Ollama service" "failed" "see the output above"
                    ;;
            esac
        fi

        set +e
        "$VPY" - <<'PYEOF'
"""Exit: 0 installed and enabled, 1 not installed, 2 installed but not enabled."""
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

# A unit that exists but was never enabled does not start at login, and
# reporting that as "enabled" is a lie the owner only discovers at the worst
# possible moment.
if not enabled.ok:
    raise SystemExit(2)
PYEOF
        SVC_STATUS=$?
        set -e
        if [ $SVC_STATUS -eq 2 ]; then
            warn "jarvis.service was written but could not be enabled."
            info "Enable it yourself: systemctl --user enable --now jarvis.service"
            record "Service" "failed" "unit written, systemctl --user enable failed"
        elif [ $SVC_STATUS -ne 0 ]; then
            warn "The service could not be installed; see the message above."
            record "Service" "failed" "jarvis.service"
        else
            ok "installed and enabled"
            info "Start it now:   systemctl --user start jarvis.service"
            info "Watch it:       journalctl --user -u jarvis.service -f"
            info ""
            warn "A user service is torn down when your last session ends, so JARVIS"
            warn "would stop when you log out or close the lid — silently, with no"
            warn "error anywhere. To keep it running with nobody logged in, run:"
            printf "        ${YELLOW}loginctl enable-linger %s${RESET}\n" "${USER:-$(id -un)}"
            record "Service" "updated" "jarvis.service enabled"
        fi
    fi
fi

# --------------------------------------------------------------------------- #
#  Done
# --------------------------------------------------------------------------- #
step "Verifying"
set +e
runtime_py status
set -e

print_summary

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

if [ -n "$MAIN_TAG" ] && [ "$WANT_FAST_MODEL" -eq 1 ]; then
    MODEL_NOTE="
  Two models are installed and they are for different jobs:
      $FAST_TAG   conversation and voice — answers immediately
      $MAIN_TAG   hard questions — far more capable, far slower
  config.yaml points at $MAIN_TAG. To use the quick one instead, either
  edit llm.ollama_model in config.yaml or set it for one run:
      JARVIS_LLM_OLLAMA_MODEL=$FAST_TAG $RUN_AS voice"
else
    MODEL_NOTE="
  Model in config.yaml: ${MAIN_TAG:-none installed}"
fi

# Be exact about what a re-run does to the weights.  An update run re-checks
# each tag against the registry; Ollama compares digests and transfers only the
# layers that changed, so a model that is already current costs a manifest.
if [ -n "$MAIN_TAG" ] && [ "$WANT_UPDATE" = "1" ]; then
    WEIGHTS_NOTE="
  Weights were re-checked against the registry, and only changed layers were
  transferred. Run with --no-update to check presence only and change nothing."
elif [ -n "$MAIN_TAG" ]; then
    WEIGHTS_NOTE="
  Weights already on disk were not re-checked this run (--no-update). To move
  one to a newer build, ask for it by name:
      ollama pull $MAIN_TAG"
else
    WEIGHTS_NOTE=""
fi

printf "\n${GREEN}==> Done.${RESET}\n"
cat <<EOF

${DIM}  Try it:
      $RUN_AS doctor        check what is installed
      $RUN_AS say           audition the British voice
      $RUN_AS chat          talk to it by keyboard
      $RUN_AS voice         hands-free
$PATH_NOTE
$MODEL_NOTE

  Re-run this installer at any time to update everything it installed:
      ./install.sh                 pull the repo, upgrade the Python packages,
                                   upgrade Ollama, and fetch anything missing
      ./install.sh --no-update     repair without changing any versions
$WEIGHTS_NOTE

  Run it in the background:
      ./install.sh --service
      loginctl enable-linger \$USER
${RESET}
EOF
