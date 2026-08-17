#!/usr/bin/env bash
#
# push.sh — push this branch and open a PR against main.
#
# Written because the agent sandbox that produced these commits cannot reach
# github.com: TCP connects, then the TLS handshake is terminated mid-flight
# (`gnutls_handshake() failed`, `unexpected eof while reading`) for
# github.com specifically while other hosts such as pypi.org negotiate fine.
# SSH on port 22 is closed the same way. That is an egress filter outside the
# sandbox, so no script, credential or transport run *there* can push.
#
# Run this on YOUR machine, where the network is not filtered. It needs no
# token: it uses whatever git credentials you already have.
#
#   ./push.sh                 push the branch, then open a PR
#   ./push.sh --no-pr         push only
#   ./push.sh --merge         push, open a PR, and merge it into main
#
# It is deliberately cautious: it refuses to run with a dirty tree, shows you
# exactly what will be pushed, and asks before doing anything irreversible.

set -euo pipefail

BRANCH="arena/01a00aea-jarvis"
BASE="main"
REMOTE="origin"
WANT_PR=1
WANT_MERGE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no-pr)  WANT_PR=0 ;;
        --merge)  WANT_MERGE=1 ;;
        --branch) BRANCH="$2"; shift ;;
        --base)   BASE="$2"; shift ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\033[38;5;39m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[38;5;214m!!!\033[0m %s\n' "$*"; }
die()  { printf '\033[38;5;203mxxx\033[0m %s\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")"

# --- sanity ---------------------------------------------------------------- #
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"

CURRENT="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT" != "$BRANCH" ]; then
    warn "on '$CURRENT', expected '$BRANCH'"
    git show-ref --verify --quiet "refs/heads/$BRANCH" \
        || die "branch '$BRANCH' does not exist locally"
    read -r -p "Switch to $BRANCH? [y/N] " reply
    [ "$reply" = "y" ] || die "aborted"
    git checkout "$BRANCH"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "the working tree has uncommitted changes:"
    git status --short
    die "commit or stash them first — this script pushes commits, not edits"
fi

# --- reachability ---------------------------------------------------------- #
# The exact failure the sandbox hits. Checking it here turns a confusing
# TLS error into a sentence that says what is wrong.
say "checking that github.com is reachable..."
if ! curl -sS -m 15 -o /dev/null https://github.com 2>/dev/null; then
    die "cannot reach github.com over TLS from this machine either.
    Check a proxy, VPN or firewall. The sandbox that wrote these commits
    fails here too, which is why this script exists."
fi

# --- show what will be pushed ---------------------------------------------- #
git fetch "$REMOTE" "$BASE" --quiet 2>/dev/null || true

# The local copy of $REMOTE/$BASE may be stale (the sandbox could not fetch),
# so fall back to "everything not yet on the remote branch" when there is no
# usable merge base to diff against.
RANGE="$REMOTE/$BASE..HEAD"
if ! git merge-base "$REMOTE/$BASE" HEAD >/dev/null 2>&1; then
    warn "no merge base with $REMOTE/$BASE (stale ref); showing recent commits"
    RANGE="HEAD~5..HEAD"
fi
COUNT="$(git rev-list --count "$RANGE" 2>/dev/null || echo 0)"
if [ "$COUNT" = "0" ]; then
    say "nothing to push — $BRANCH is already contained in $REMOTE/$BASE"
    exit 0
fi

say "$COUNT commit(s) will be pushed to $REMOTE/$BRANCH:"
git --no-pager log --oneline "$RANGE"
echo
say "files changed:"
git --no-pager diff --stat "$RANGE" 2>/dev/null | tail -20 || true
echo

read -r -p "Push these to $REMOTE/$BRANCH? [y/N] " reply
[ "$reply" = "y" ] || die "aborted"

# --- push ------------------------------------------------------------------ #
say "pushing..."
git push --set-upstream "$REMOTE" "$BRANCH"
say "pushed."

# --- pull request ----------------------------------------------------------- #
[ "$WANT_PR" -eq 1 ] || { say "done (--no-pr)"; exit 0; }

if ! command -v gh >/dev/null 2>&1; then
    warn "the GitHub CLI (gh) is not installed, so no PR was opened."
    warn "Open one here:"
    warn "  https://github.com/hardcoregamingsyle/jarvis/compare/$BASE...$BRANCH"
    exit 0
fi

if ! gh auth status >/dev/null 2>&1; then
    warn "gh is installed but not logged in. Run:  gh auth login"
    warn "Or open the PR here:"
    warn "  https://github.com/hardcoregamingsyle/jarvis/compare/$BASE...$BRANCH"
    exit 0
fi

EXISTING="$(gh pr list --head "$BRANCH" --state open --json url \
            --jq '.[0].url' 2>/dev/null || true)"
if [ -n "$EXISTING" ]; then
    say "a pull request is already open: $EXISTING"
    say "the commits above have been added to it."
else
    say "opening a pull request..."
    gh pr create --base "$BASE" --head "$BRANCH" --fill
fi

# --- merge ------------------------------------------------------------------ #
if [ "$WANT_MERGE" -eq 1 ]; then
    warn "about to merge into $BASE."
    read -r -p "Merge now? [y/N] " reply
    if [ "$reply" = "y" ]; then
        gh pr merge --merge
        say "merged into $BASE."
    else
        say "left the pull request open."
    fi
fi

say "done."
