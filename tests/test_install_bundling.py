"""install.sh must produce a working JARVIS in one command — and update it in one.

Two questions decide whether this installer is honest, and both used to have the
wrong answer:

* **Does one command give you something that can think?**  It used to install
  Python packages and then *print* the Ollama and model commands for the owner
  to run by hand, which meant a "successful" install could not answer a single
  question.  The runtime, the weights and the speech models are now fetched by
  the installer itself — rootlessly, because the standing promise is that it
  never calls sudo.
* **Does running it again update everything?**  It used to `pip install` without
  ``--upgrade`` (so every dependency stayed at whatever version was already
  there), never pull the checkout, and never touch the model.  A re-run was
  close to a no-op that looked like a success.

These tests are static analysis of the script plus behavioural runs of the shell
functions that can be executed in isolation.  Nothing here downloads anything,
runs the installer end to end, or touches a real home directory: the git tests
build their own repositories inside ``tmp_path`` and point ``$ROOT`` at them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"

#: Seconds. Generous: these spawn git and bash on a laptop, not a build farm.
TIMEOUT = 120

#: Every state a summary line is allowed to report. Anything else is either
#: vague ("done") or a claim the installer cannot back up.
ALLOWED_STATES = {"updated", "installed", "already current", "skipped", "failed"}

#: The components the owner asked about by name. A summary that omits one of
#: these does not answer "will running it again update all of it?".
REQUIRED_COMPONENTS = {
    "Repository",
    "Python packages",
    "Ollama runtime",
    "Main model",
    "Fast model",
    "Speech assets",
    "Launcher",
}

#: Lines that mean the installer would destroy work the owner had not committed.
#: `--ff-only` is the only acceptable way for this script to move HEAD.
DESTRUCTIVE_GIT = ("reset --hard", "checkout --", "clean -fd", "clean -f", "stash")


def _find_bash() -> str:
    """Locate bash, including when it is not on a Windows PATH.

    Worth the effort: these tests are the only thing standing between a broken
    installer and the Linux box, and skipping them because the test runner was
    launched from PowerShell rather than Git Bash would silently remove that
    protection exactly when it is needed.
    """
    found = shutil.which("bash")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        "/bin/bash",
        "/usr/bin/bash",
    ):
        if Path(candidate).exists():
            return candidate
    return ""


def _find_git() -> str:
    found = shutil.which("git")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files\Git\mingw64\bin\git.exe",
        "/usr/bin/git",
    ):
        if Path(candidate).exists():
            return candidate
    return ""


# --------------------------------------------------------------------------- #
#  Fixtures and extraction helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def source() -> str:
    if not SCRIPT.exists():
        pytest.skip("install.sh not present")
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lines(source: str) -> List[str]:
    return source.splitlines()


@pytest.fixture(scope="module")
def code_lines(source: str) -> List[str]:
    """Non-comment, non-blank lines. Comments may legitimately quote old bugs."""
    return [
        stripped
        for stripped in (raw.strip() for raw in source.splitlines())
        if stripped and not stripped.startswith("#")
    ]


@pytest.fixture(scope="module")
def header(source: str) -> str:
    """The comment block the --help output is generated from."""
    return source.split("set -euo pipefail")[0]


@pytest.fixture(scope="module")
def bash() -> str:
    found = _find_bash()
    if not found:
        pytest.skip("bash unavailable")
    return found


def _extract_function(source: str, name: str) -> str:
    """Pull one shell function out of the script so it can be run alone.

    Relies on the closing brace sitting in column 0, which is the style the
    whole script uses; a function that breaks it fails here loudly rather than
    silently dropping out of the behavioural tests.
    """
    body = source.splitlines()
    start = None
    for index, line in enumerate(body):
        if re.match(r"^%s\(\)\s*\{" % re.escape(name), line):
            start = index
            break
    assert start is not None, f"no function {name}() in install.sh"
    for index in range(start + 1, len(body)):
        if body[index] == "}":
            return "\n".join(body[start:index + 1])
    raise AssertionError(f"{name}() is never closed at column 0")


def _extract_block(source: str, marker: str) -> str:
    """Lines from the one containing `marker` up to the next banner comment."""
    body = source.splitlines()
    start = None
    for index, line in enumerate(body):
        if marker in line:
            start = index
            break
    assert start is not None, f"marker not found in install.sh: {marker!r}"
    for index in range(start + 1, len(body)):
        if body[index].startswith("# ---"):
            return "\n".join(body[start:index])
    return "\n".join(body[start:])


def _extract_span(source: str, first: str, last: str) -> str:
    """Lines from the one equal to `first` through the next one equal to `last`."""
    body = [line.rstrip() for line in source.splitlines()]
    try:
        start = body.index(first)
    except ValueError:
        raise AssertionError(f"line not found in install.sh: {first!r}")
    for index in range(start + 1, len(body)):
        if body[index].strip() == last:
            return "\n".join(body[start:index + 1])
    raise AssertionError(f"no {last!r} after {first!r}")


STUBS = "\n".join([
    "set -euo pipefail",
    "CYAN=''; GREEN=''; YELLOW=''; DIM=''; RESET=''",
    'step() { printf "STEP %s\\n" "$1"; }',
    'ok()   { printf "OK %s\\n" "$1"; }',
    'warn() { printf "WARN %s\\n" "$1"; }',
    'info() { printf "INFO %s\\n" "$1"; }',
    'record() { printf "RECORD %s|%s|%s\\n" "$1" "$2" "${3:-}"; }',
])


def _run(bash: str, snippet: str, *, env=None, argv=(), cwd=None) -> subprocess.CompletedProcess:
    environ = {"PATH": os.environ.get("PATH", ""), "HOME": str(cwd or ROOT)}
    if env:
        environ.update(env)
    return subprocess.run(
        [bash, "-c", snippet, "install.sh", *argv],
        env=environ, capture_output=True, text=True, timeout=TIMEOUT, cwd=cwd,
    )


def _strip_quoted(line: str) -> str:
    """Blank out quoted segments so printed advice is not read as a command.

    The installer's whole approach is to *print* the sudo line for the user to
    run themselves, so `printf 'sudo apt-get install ...'` is correct and must
    not be flagged. Only an unquoted `sudo` is an invocation.
    """
    out, quote = [], None
    for char in line:
        if quote:
            out.append(" ")
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def _record_calls(source: str) -> List[Tuple[str, str]]:
    return re.findall(r'record\s+"([^"]+)"\s+"([^"]+)"', source)


# --------------------------------------------------------------------------- #
#  The embedded Python
#
#  install.sh carries three `python - <<PYEOF` programs, and they are where the
#  bundling actually happens.  Greping the shell for the line that calls them
#  proves nothing about what they do, so they are extracted and executed here
#  against stub packages built in tmp_path.  Nothing below imports the real
#  jarvis package, touches a real home directory, or reaches the network.
# --------------------------------------------------------------------------- #
def _embedded_python(source: str, needle: str) -> str:
    """The `python - <<'PYEOF'` program containing `needle`."""
    blocks = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", source, re.S)
    matches = [block for block in blocks if needle in block]
    assert len(matches) == 1, (
        f"expected exactly one embedded Python block containing {needle!r}, "
        f"found {len(matches)} of {len(blocks)}"
    )
    return matches[0]


def _run_embedded(program: str, tmp_path: Path, *, env=None, extra_path=None):
    """Run one embedded program the way install.sh does: source on stdin."""
    script = tmp_path / "embedded.py"
    script.write_text(program, encoding="utf-8")
    environ = dict(os.environ)
    environ["PYTHONIOENCODING"] = "utf-8"
    environ["PYTHONPATH"] = str(extra_path) if extra_path else ""
    if env:
        environ.update(env)
    return subprocess.run(
        [sys.executable, str(script)],
        env=environ, cwd=str(tmp_path), capture_output=True, text=True, timeout=TIMEOUT,
    )


def _stub_runtime(tmp_path: Path, body: str) -> Path:
    """A minimal fake `jarvis` package exposing only what the bridge calls."""
    root = tmp_path / "stub"
    (root / "jarvis" / "core").mkdir(parents=True, exist_ok=True)
    (root / "jarvis" / "runtime").mkdir(parents=True, exist_ok=True)
    (root / "jarvis" / "__init__.py").write_text("", encoding="utf-8")
    (root / "jarvis" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (root / "jarvis" / "core" / "config.py").write_text(
        "class _LLM:\n"
        "    ollama_model = 'stub:tag'\n"
        "class _Cfg:\n"
        "    llm = _LLM()\n"
        "def load_config(path=None, **kw):\n"
        "    return _Cfg()\n",
        encoding="utf-8",
    )
    (root / "jarvis" / "runtime" / "__init__.py").write_text(body, encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
#  It still has to be a valid script
# --------------------------------------------------------------------------- #
def test_the_script_is_syntactically_valid(bash):
    result = subprocess.run(
        [bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=TIMEOUT
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


# --------------------------------------------------------------------------- #
#  --update, and a plain re-run that does the same thing
# --------------------------------------------------------------------------- #
def test_update_is_parsed_and_documented(source, header):
    assert re.search(r'^\s*--update\)\s*WANT_UPDATE=1', source, re.M), \
        "--update is not parsed"
    assert re.search(r'^\s*--no-update\)\s*WANT_UPDATE=0', source, re.M), \
        "--no-update is not parsed"
    assert "--update" in header, "--update is not in the header help"
    assert "--no-update" in header, "--no-update is not in the header help"


def test_updating_is_the_default_so_a_plain_rerun_updates(source):
    """The owner's question was about re-running the installer, not a flag."""
    assert re.search(r'^WANT_UPDATE=1\s*$', source, re.M), (
        "WANT_UPDATE must default to 1: the owner asked whether re-running the "
        "installer updates everything, and the answer has to be yes without "
        "remembering a flag"
    )


@pytest.mark.parametrize("argv,expected", [
    ((), {"UPDATE": "1", "MODEL": "1", "FAST": "1", "PROFILE": "lean"}),
    (("--update",), {"UPDATE": "1"}),
    (("--no-update",), {"UPDATE": "0"}),
    (("--no-model",), {"MODEL": "0"}),
    (("--only-main-model",), {"FAST": "0"}),
    (("--no-model", "--only-main-model"), {"MODEL": "0", "FAST": "0"}),
    (("--min",), {"PROFILE": "min"}),
    (("--full", "--no-voice"), {"PROFILE": "full", "VOICE": "0"}),
    (("--model", "qwen3:4b"), {"MODELID": "qwen3:4b"}),
    (("--venv", "/opt/v"), {"VENV": "/opt/v"}),
    (("--no-link", "--service", "--vllm"), {"LINK": "0", "SERVICE": "1", "VLLM": "1"}),
])
def test_the_flag_parser_sets_what_the_help_promises(bash, source, argv, expected):
    """Run the real argument loop rather than trusting it by inspection."""
    snippet = "\n".join([
        "set -euo pipefail",
        _extract_span(source, 'PROFILE="lean"', 'MODEL_ID=""'),
        'usage() { echo HELP; }',
        'ORIGINAL_ARGS=("$@")',
        _extract_span(source, "while [ $# -gt 0 ]; do", "done"),
        'echo "PROFILE=$PROFILE UPDATE=$WANT_UPDATE MODEL=$WANT_MODEL '
        'FAST=$WANT_FAST_MODEL VOICE=$DOWNLOAD_VOICE LINK=$WANT_LINK '
        'SERVICE=$WANT_SERVICE VLLM=$WANT_VLLM VENV=$VENV MODELID=$MODEL_ID"',
    ])
    result = _run(bash, snippet, argv=argv)
    assert result.returncode == 0, result.stderr
    parsed = dict(pair.split("=", 1) for pair in result.stdout.split())
    for key, value in expected.items():
        assert parsed[key] == value, f"{argv} -> {key}={parsed[key]!r}, wanted {value!r}"


def test_an_unknown_flag_is_rejected_rather_than_ignored(bash, source):
    snippet = "\n".join([
        "set -euo pipefail",
        _extract_span(source, 'PROFILE="lean"', 'MODEL_ID=""'),
        'usage() { echo HELP; }',
        'ORIGINAL_ARGS=("$@")',
        _extract_span(source, "while [ $# -gt 0 ]; do", "done"),
        'echo REACHED',
    ])
    result = _run(bash, snippet, argv=("--upgrade-everything",))
    assert result.returncode == 1
    assert "REACHED" not in result.stdout
    assert "Unknown option" in result.stderr


def test_help_is_generated_from_the_header_and_lists_every_flag(bash, source):
    """A hard-coded line range rots; deriving --help from the header cannot."""
    snippet = _extract_function(source, "usage") + '\nusage\n'
    result = _run(bash, snippet, argv=())
    assert result.returncode == 0, result.stderr
    printed = result.stdout
    parsed = set(re.findall(r'^\s*(--[a-z-]+)\)', source, re.M))
    parsed |= set(re.findall(r'\|(--[a-z-]+)\)', source))
    missing = {flag for flag in parsed if flag not in printed}
    assert not missing, f"--help does not mention: {missing}"


def test_help_lists_every_flag_the_parser_accepts(source, header):
    parsed = set(re.findall(r'^\s*(--[a-z-]+)\)', source, re.M))
    parsed |= set(re.findall(r'\|(--[a-z-]+)\)', source))
    documented = set(re.findall(r'(--[a-z-]+)', header))
    undocumented = {flag for flag in parsed if flag not in documented}
    assert not undocumented, f"flags accepted but not in the header help: {undocumented}"


# --------------------------------------------------------------------------- #
#  git: fast-forward or nothing
# --------------------------------------------------------------------------- #
def test_the_pull_is_fast_forward_only(code_lines):
    # Quoted text is stripped first so that a printed message mentioning a
    # pull is not mistaken for one being run.
    pulls = [
        line for line in code_lines
        if re.search(r'\bgit\b[^\n]*\bpull\b', _strip_quoted(line))
    ]
    assert pulls, "install.sh never pulls, so a re-run cannot pick up new code"
    for pull in pulls:
        assert "--ff-only" in pull, (
            f"a pull without --ff-only can create a merge commit or fail "
            f"halfway through: {pull!r}"
        )


@pytest.mark.parametrize("pattern", DESTRUCTIVE_GIT)
def test_no_git_command_can_discard_local_work(source, pattern):
    """Not even in a comment: nobody should be able to copy one out of here."""
    assert pattern not in source, (
        f"install.sh mentions `git {pattern}`. The installer must never discard "
        f"or hide uncommitted work to make an update succeed."
    )


def test_a_dirty_tree_is_reported_rather_than_forced(source):
    block = _extract_block(source, 'step "Updating the checkout"')
    assert "status --porcelain" in block, "the working tree is never inspected"
    assert re.search(r'record "Repository" "skipped"', block), \
        "a skipped pull must still appear in the summary"


@pytest.fixture
def repo_pair(tmp_path: Path):
    """An upstream repository and a clone of it, both inside tmp_path.

    Everything these tests touch lives under tmp_path; no real checkout, home
    directory or working directory is named anywhere in them.
    """
    git = _find_git()
    if not git:
        pytest.skip("git unavailable")
    upstream = tmp_path / "upstream"
    work = tmp_path / "work"

    def run(*args, cwd=None):
        result = subprocess.run(
            [git, *args], cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        assert result.returncode == 0, f"git {args}: {result.stderr}"
        return result

    def commit(text: str, message: str):
        (upstream / "tracked.txt").write_text(text, encoding="utf-8")
        run("-C", str(upstream), "add", "-A")
        run("-C", str(upstream), "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-qm", message)

    run("init", "-q", "-b", "main", str(upstream))
    commit("one\n", "one")
    run("clone", "-q", str(upstream), str(work))
    return upstream, work, commit


def _run_git_block(bash: str, source: str, work: Path, **env) -> subprocess.CompletedProcess:
    snippet = STUBS + "\n" + _extract_block(source, 'step "Updating the checkout"')
    environ = {
        "ROOT": str(work),
        "WANT_UPDATE": "1",
        # Stops the re-exec branch from replacing the test's shell; the branch
        # itself is checked statically below.
        "JARVIS_INSTALL_REEXEC": "1",
    }
    environ.update(env)
    return _run(bash, snippet, env=environ, cwd=work)


def test_an_up_to_date_checkout_reports_already_current(bash, source, repo_pair):
    _, work, _ = repo_pair
    result = _run_git_block(bash, source, work)
    assert result.returncode == 0, result.stderr
    assert "RECORD Repository|already current|" in result.stdout


def test_a_clean_checkout_behind_upstream_is_fast_forwarded(bash, source, repo_pair):
    _, work, commit = repo_pair
    commit("two\n", "two")
    result = _run_git_block(bash, source, work)
    assert result.returncode == 0, result.stderr
    assert "RECORD Repository|updated|" in result.stdout
    assert (work / "tracked.txt").read_text(encoding="utf-8") == "two\n"


def test_local_edits_stop_the_pull_and_survive_it(bash, source, repo_pair):
    """The one failure mode that would be unforgivable: losing the owner's work."""
    _, work, commit = repo_pair
    commit("two\n", "two")
    (work / "tracked.txt").write_text("MINE — not committed\n", encoding="utf-8")

    result = _run_git_block(bash, source, work)

    assert result.returncode == 0, result.stderr
    assert "RECORD Repository|skipped|local changes present" in result.stdout
    assert (work / "tracked.txt").read_text(encoding="utf-8") == "MINE — not committed\n"


def test_an_untracked_file_does_not_block_the_update(bash, source, repo_pair):
    """A stray note in the directory is not a reason to refuse every update."""
    _, work, commit = repo_pair
    commit("two\n", "two")
    (work / "notes.txt").write_text("shopping list\n", encoding="utf-8")

    result = _run_git_block(bash, source, work)

    assert "RECORD Repository|updated|" in result.stdout
    assert (work / "notes.txt").exists(), "an untracked file was removed"


def test_a_detached_head_is_skipped_with_a_reason(bash, source, repo_pair):
    git = _find_git()
    _, work, commit = repo_pair
    commit("two\n", "two")
    subprocess.run([git, "-C", str(work), "checkout", "-q", "--detach", "HEAD"],
                   capture_output=True, text=True, timeout=TIMEOUT)
    result = _run_git_block(bash, source, work)
    assert "RECORD Repository|skipped|detached HEAD" in result.stdout


def test_a_directory_that_is_not_a_checkout_is_skipped(bash, source, tmp_path):
    plain = tmp_path / "tarball-install"
    plain.mkdir()
    result = _run_git_block(bash, source, plain)
    assert "RECORD Repository|skipped|not a git checkout" in result.stdout


def test_no_update_skips_the_pull_entirely(bash, source, repo_pair):
    _, work, commit = repo_pair
    commit("two\n", "two")
    result = _run_git_block(bash, source, work, WANT_UPDATE="0")
    assert "RECORD Repository|skipped|--no-update" in result.stdout
    assert (work / "tracked.txt").read_text(encoding="utf-8") == "one\n"


def test_a_pull_that_changed_the_installer_restarts_it(source):
    """bash reads a script incrementally.

    Continuing to run an install.sh that git has just rewritten underneath the
    interpreter executes a spliced mixture of the old and new files, which is
    the kind of bug that only appears on the one machine that was mid-update.
    """
    block = _extract_block(source, 'step "Updating the checkout"')
    assert "JARVIS_INSTALL_REEXEC" in block, "no guard against re-exec looping"
    assert re.search(r'exec\s+"\$\{BASH:-bash\}"\s+"\$0"', block), (
        "after pulling a new install.sh the script must exec the new copy "
        "rather than carry on reading the old one"
    )
    assert "ORIGINAL_ARGS" in source, "the re-exec must pass the original flags on"


# --------------------------------------------------------------------------- #
#  pip: --upgrade is the whole difference between install and update
# --------------------------------------------------------------------------- #
def test_the_requirements_install_can_actually_upgrade(source, code_lines):
    installs = [
        line for line in code_lines
        if "pip install" in line and "-r requirements" in line
    ]
    assert installs, "install.sh never installs the requirements files"
    for line in installs:
        assert "--upgrade" in line or '"${PIP_ARGS[@]}"' in line, (
            f"plain `pip install -r ...` is satisfied by whatever is already "
            f"installed, so a re-run moves nothing: {line.strip()!r}"
        )
    assert re.search(r'PIP_ARGS\+=\(--upgrade\)', source), (
        "nothing ever adds --upgrade to the pip arguments"
    )


def test_pip_arguments_carry_upgrade_by_default(bash, source):
    """The default path — a bare re-run — must upgrade; --no-update must not."""
    block = _extract_span(source, "PIP_ARGS=(--disable-pip-version-check)", "fi")
    snippet = "set -euo pipefail\n" + block + '\necho "ARGS=${PIP_ARGS[*]}"'

    updating = _run(bash, snippet, env={"WANT_UPDATE": "1"})
    assert "--upgrade" in updating.stdout, updating.stdout

    frozen = _run(bash, snippet, env={"WANT_UPDATE": "0"})
    assert "--upgrade" not in frozen.stdout, (
        "--no-update promises no version bumps, so it must not pass --upgrade"
    )


def test_the_editable_install_is_still_there(source):
    assert re.search(r'pip install [^\n]*-e \.', source), (
        "the package itself must still be installed in editable mode"
    )


# --------------------------------------------------------------------------- #
#  Bundling: the runtime and the weights arrive with the installer
# --------------------------------------------------------------------------- #
def test_the_runtime_and_weights_are_installed_not_merely_suggested(source):
    assert "runtime_py ollama" in source, (
        "install.sh must install the inference runtime itself; printing the "
        "commands is what made a 'successful' install unable to answer anything"
    )
    assert re.search(r'runtime_py model "\$MAIN_TAG"', source), "no main-model pull"
    assert re.search(r'runtime_py model "\$FAST_TAG"', source), "no fast-model pull"
    assert "runtime_py assets" in source, "no speech-asset provisioning"


def test_the_root_requiring_ollama_installer_is_not_used_or_recommended(code_lines):
    """`curl | sh` writes /usr/local/bin and a system unit, and needs root."""
    joined = "\n".join(code_lines)
    assert "ollama.com/install.sh" not in joined, (
        "the official Ollama installer needs root and a system-wide systemd "
        "unit, which breaks the no-sudo promise; install the release tarball "
        "under ~/.local instead"
    )


def test_an_existing_system_ollama_is_detected_rather_than_duplicated(source):
    assert re.search(r'command -v ollama', source), (
        "a system-wide ollama already on PATH should be used, not shadowed by "
        "a second copy in the home directory"
    )


def test_the_small_interactive_model_is_pulled_too(source):
    assert 'FAST_TAG="qwen3:4b-instruct-2507-q4_K_M"' in source, (
        "the default model is dense and thinks before answering — roughly "
        "1 tok/s on this CPU — so voice needs a small model as well"
    )
    assert "--only-main-model" in source


def test_the_output_says_which_model_is_for_which_job(source):
    assert re.search(r'MODEL_NOTE=', source), "no closing note about the two models"
    for phrase in ("interactive", "capable"):
        assert phrase in source, f"the output never explains the {phrase} model"


def test_no_model_skips_every_weight_download(source):
    block = _extract_span(
        source,
        'if [ "$WANT_MODEL" -eq 0 ]; then',
        'elif [ "$PROFILE" = "min" ]; then',
    )
    assert "runtime_py model" not in block
    for component in ("Ollama runtime", "Main model", "Fast model"):
        assert f'record "{component}" "skipped" "--no-model"' in block, (
            f"--no-model must still report {component} in the summary"
        )


def test_config_yaml_is_wired_to_what_was_installed(source):
    assert "backend=ollama" in source, (
        "installing Ollama and leaving llm.backend on its old value means the "
        "install worked and JARVIS still cannot find a model"
    )
    assert re.search(r'ollama_model=\$MAIN_TAG', source), \
        "llm.ollama_model is never set to the tag that was pulled"


def test_the_ollama_user_service_is_installed_with_service(source):
    block = _extract_block(source, 'step "Installing the systemd user services"')
    assert "runtime_py ollama-service" in block, (
        "--service installs jarvis.service but not the model server it needs"
    )
    assert "systemctl --user" in block
    assert "systemctl --system" not in source, "a system unit would need root"


# --------------------------------------------------------------------------- #
#  Disk space, before anything large moves
# --------------------------------------------------------------------------- #
#: Every line that starts a download measured in gigabytes, and whether the
#: marker is the whole line ("exact") or part of it ("within").  `runtime_py
#: ollama` has to be exact or it also matches `runtime_py ollama-service`,
#: which writes a unit file and downloads nothing.
LARGE_DOWNLOADS = (
    ("runtime_py ollama", "exact"),
    ('runtime_py model "$MAIN_TAG"', "exact"),
    ('runtime_py model "$FAST_TAG"', "exact"),
    ("runtime_py assets", "exact"),
    ("-r requirements-full.txt", "within"),
    ('pip install vllm "${PIP_ARGS[@]}"', "within"),
)


@pytest.mark.parametrize("marker,mode", LARGE_DOWNLOADS)
def test_a_disk_check_immediately_precedes_every_large_download(lines, marker, mode):
    """Ordering, not mere presence: a check after the download is worthless."""
    hits = [
        i for i, line in enumerate(lines)
        if not line.strip().startswith("#")
        and (line.strip() == marker if mode == "exact" else marker in line)
    ]
    assert hits, f"no line starting the download {marker!r}"
    for index in hits:
        window = lines[max(0, index - 12):index]
        assert any("require_disk_space" in line for line in window), (
            f"line {index + 1} starts a multi-gigabyte download with no disk "
            f"check in the twelve lines above it: {lines[index].strip()!r}"
        )


def test_sizes_and_destinations_are_printed_before_the_downloads(lines):
    plan = next(i for i, line in enumerate(lines) if 'step "Download plan"' in line)
    first_pull = next(
        i for i, line in enumerate(lines) if line.strip() == 'runtime_py ollama'
    )
    assert plan < first_pull, "the plan must be printed before anything downloads"
    window = "\n".join(lines[plan:first_pull])
    assert "GB" in window and "->" in window, (
        "the plan must state sizes and destinations, not just names"
    )


def test_free_space_is_measured_against_the_real_filesystem(bash, source, tmp_path):
    """free_gb must return a plausible number for a directory that exists."""
    snippet = _extract_function(source, "free_gb") + '\nfree_gb "$TARGET"\necho\n'
    result = _run(bash, snippet, env={"TARGET": str(tmp_path)}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    reported = result.stdout.strip()
    if not reported:
        pytest.skip("df is unavailable in this environment")
    assert reported.isdigit()
    assert 0 <= int(reported) < 1_000_000


def test_free_space_of_a_path_that_does_not_exist_yet_uses_its_parent(bash, source, tmp_path):
    """~/.ollama usually does not exist on the first run; df would fail on it."""
    snippet = _extract_function(source, "free_gb") + '\nfree_gb "$TARGET"\necho\n'
    missing = tmp_path / "not" / "created" / "yet"
    result = _run(bash, snippet, env={"TARGET": str(missing)}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    reported = result.stdout.strip()
    if not reported:
        pytest.skip("df is unavailable in this environment")
    assert reported.isdigit()


@pytest.mark.parametrize("tag,low,high", [
    ("qwen3.6:27b", 14, 20),
    ("qwen3:4b-instruct-2507-q4_K_M", 2, 5),
    ("qwen3:30b-a3b-instruct-2507-q4_K_M", 17, 24),
    ("something-nobody-has-heard-of", 15, 40),
])
def test_the_fallback_size_table_is_in_the_right_ballpark(bash, source, tag, low, high):
    snippet = _extract_function(source, "model_size_gb") + f'\nmodel_size_gb "{tag}"\n'
    result = _run(bash, snippet)
    assert result.returncode == 0, result.stderr
    size = int(result.stdout.strip())
    assert low <= size <= high, f"{tag} estimated at {size} GB"


def _disk_snippet(source: str, free: str) -> str:
    """require_disk_space with free_gb replaced by a fixed answer."""
    return "\n".join([
        STUBS,
        _extract_function(source, "free_gb"),
        _extract_function(source, "require_disk_space"),
        'free_gb() { printf "%s" "%s"; }' % ("%s", free),
    ])


def test_a_download_that_does_not_fit_is_refused_with_the_numbers(bash, source):
    snippet = _disk_snippet(source, "4") + '\n' + (
        'if require_disk_space /somewhere 17 "the main model"; then '
        'echo PROCEED; else echo REFUSED; fi'
    )
    result = _run(bash, snippet)
    assert "REFUSED" in result.stdout
    assert "17" in result.stdout and "4" in result.stdout, (
        "refusing without saying how much is needed and how much there is "
        "leaves the owner with nothing to act on"
    )


def test_a_download_that_fits_proceeds(bash, source):
    snippet = _disk_snippet(source, "40") + '\n' + (
        'if require_disk_space /somewhere 17 "the main model"; then '
        'echo PROCEED; else echo REFUSED; fi'
    )
    result = _run(bash, snippet)
    assert "PROCEED" in result.stdout


def test_unknown_free_space_does_not_block_the_install(bash, source):
    """Refusing because df is missing would be worse than trying."""
    snippet = _disk_snippet(source, "") + '\n' + (
        'if require_disk_space /somewhere 17 "the main model"; then '
        'echo PROCEED; else echo REFUSED; fi'
    )
    result = _run(bash, snippet)
    assert "PROCEED" in result.stdout


# --------------------------------------------------------------------------- #
#  The summary
# --------------------------------------------------------------------------- #
def test_every_component_reports_a_line(source):
    reported = {component for component, _ in _record_calls(source)}
    missing = REQUIRED_COMPONENTS - reported
    assert not missing, f"these components never appear in the summary: {missing}"


def test_every_reported_state_is_one_of_the_honest_five(source):
    states = {state for _, state in _record_calls(source)}
    unexpected = states - ALLOWED_STATES
    assert not unexpected, (
        f"summary states outside the agreed vocabulary {sorted(ALLOWED_STATES)}: "
        f"{unexpected}"
    )


def test_the_summary_is_printed_and_survives_an_early_exit(source):
    assert "print_summary" in source
    assert re.search(r'^\s*trap print_summary EXIT', source, re.M), (
        "a run that dies half way must still say which components it got to"
    )


def test_the_summary_prints_each_component_once(bash, source):
    snippet = "\n".join([
        "set -euo pipefail",
        "CYAN=''; GREEN=''; YELLOW=''; DIM=''; RESET=''",
        'SUMMARY=""',
        "SUMMARY_PRINTED=0",
        _extract_function(source, "record"),
        _extract_function(source, "print_summary"),
        'record "Main model" "updated" "qwen3.6:27b"',
        'record "Fast model" "skipped" "--only-main-model"',
        'record "Repository" "already current" "main"',
        "print_summary",
        "print_summary",
    ])
    result = _run(bash, snippet)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.count("Main model") == 1, "the summary printed twice"
    assert "updated" in out and "skipped" in out and "already current" in out
    assert "qwen3.6:27b" in out and "--only-main-model" in out


def test_an_empty_summary_prints_nothing(bash, source):
    snippet = "\n".join([
        "set -euo pipefail",
        "CYAN=''; GREEN=''; YELLOW=''; DIM=''; RESET=''",
        'SUMMARY=""',
        "SUMMARY_PRINTED=0",
        _extract_function(source, "print_summary"),
        "print_summary",
        "echo END",
    ])
    result = _run(bash, snippet)
    assert result.stdout.strip() == "END", (
        "an installer that failed during argument parsing should not print an "
        "empty summary table"
    )


# --------------------------------------------------------------------------- #
#  Standing promises
# --------------------------------------------------------------------------- #
def test_the_installer_never_calls_sudo(code_lines):
    """It prints sudo commands for the user to run; it must not run them."""
    offenders = [
        line for line in code_lines
        if re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted(line))
    ]
    assert not offenders, f"install.sh appears to invoke sudo: {offenders}"


def test_the_sudo_detector_would_actually_catch_a_real_call():
    """A guard that cannot fail is not a guard."""
    assert re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted("sudo apt-get install -y x"))
    assert re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted("foo && sudo tar -xzf x.tgz"))
    assert not re.search(
        r'(^|[;&|(]\s*)sudo\s',
        _strip_quoted("""printf 'sudo apt-get install -y python3-venv'"""),
    )


@pytest.mark.parametrize("pattern", ["rm -rf", "rm -fr", "rmdir /s"])
def test_nothing_is_deleted(source, pattern):
    assert pattern not in source, (
        f"install.sh contains {pattern!r}; it is allowed to create and update, "
        f"never to delete"
    )


def test_the_header_lists_every_place_written_outside_the_repository(header):
    """The promise has to enumerate the new destinations, not just the symlink."""
    for destination in (
        "~/.local/bin/jarvis",
        "~/.local/share",
        "~/.ollama",
        "~/.config/systemd/user",
    ):
        assert destination in header, (
            f"the installer now writes {destination} but the header does not "
            f"say so"
        )


def test_the_header_answers_the_question_that_was_asked(header):
    lowered = header.lower()
    assert "updat" in lowered, "the header never mentions updating"
    assert "sudo" in lowered, "the no-sudo promise disappeared from the header"


def test_the_runtime_bridge_degrades_instead_of_failing(source):
    """A checkout without jarvis/runtime/ must still install what it can."""
    assert re.search(r'RT_ABSENT=5', source), "no 'module absent' exit code"
    assert source.count('"$RT_ABSENT"') >= 3, (
        "each runtime step must handle jarvis.runtime being missing by "
        "printing the manual commands, not by aborting the install"
    )
    assert "ollama pull $MAIN_TAG" in source or 'ollama pull $MAIN_TAG' in source, (
        "the degraded path should still tell the owner what to run"
    )


# --------------------------------------------------------------------------- #
#  config.yaml is edited in place, and that edit must never break the file
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config_writer(source: str) -> str:
    return _embedded_python(source, "JARVIS_SET_MODEL")


def _write_config(writer: str, tmp_path: Path, text, keys="backend=ollama\nollama_model=t:1"):
    target = tmp_path / "config.yaml"
    if text is not None:
        target.write_text(text, encoding="utf-8")
    result = _run_embedded(writer, tmp_path, env={
        "JARVIS_CONFIG_FILE": str(target),
        "JARVIS_SET_MODEL": keys,
    })
    return result, target


# YAML forbids tabs in indentation, so only space widths are worth covering.
@pytest.mark.parametrize("indent", ["  ", "   ", "    ", "        "])
def test_the_config_edit_matches_the_indentation_already_in_the_file(
        config_writer, tmp_path, indent):
    """Appending two-space keys to a four-space block produced invalid YAML.

    ``jarvis.core.config._load_file`` does not catch ``yaml.YAMLError``, so the
    installer would have left config.yaml — the file holding the api_key, with
    no backup anywhere — in a state that made every later ``jarvis`` command a
    traceback, over nothing more than a formatting preference.
    """
    yaml = pytest.importorskip("yaml")
    original = "llm:\n%sbackend: auto\n%smodel: X\nstt:\n  engine: whisper\n" % (
        indent, indent)
    result, target = _write_config(config_writer, tmp_path, original)
    assert result.returncode == 0, result.stdout + result.stderr

    written = target.read_text(encoding="utf-8")
    parsed = yaml.safe_load(written)
    assert parsed["llm"]["ollama_model"] == "t:1", written
    assert parsed["llm"]["backend"] == "ollama", written
    assert parsed["stt"]["engine"] == "whisper", "another section was damaged"
    for line in written.splitlines():
        if re.match(r"^\s+\w+:", line) and "engine" not in line:
            assert line.startswith(indent) and not line.startswith(indent + " "), (
                f"mixed indentation inside llm:, which YAML rejects: {line!r}"
            )


def test_a_backend_the_owner_chose_is_not_silently_reset(config_writer, tmp_path):
    """The --vllm section tells the owner to set `backend: vllm` by hand.

    Rewriting it to `ollama` on the next re-run would undo that instruction
    without saying so, and the installer would be fighting its own advice.
    """
    result, target = _write_config(
        config_writer, tmp_path, "llm:\n  backend: vllm\n  model: X\n")
    assert result.returncode == 0, result.stdout + result.stderr
    written = target.read_text(encoding="utf-8")
    assert "backend: vllm" in written, "a deliberate backend choice was overwritten"
    assert "ollama_model: t:1" in written, "the model tag was not recorded"
    assert "vllm" in result.stdout, "the output does not say the value was kept"


def test_an_unset_backend_is_pointed_at_the_runtime_that_was_installed(
        config_writer, tmp_path):
    """The guard above must not stop a fresh install from being wired up."""
    result, target = _write_config(
        config_writer, tmp_path, "llm:\n  backend: auto\n  model: X\n")
    assert result.returncode == 0
    assert "backend: ollama" in target.read_text(encoding="utf-8")


def test_the_second_run_reports_already_current_rather_than_updated(
        config_writer, tmp_path):
    """A summary that says "updated" every time answers nobody's question."""
    original = "llm:\n  backend: auto\n  model: X\n"
    first, target = _write_config(config_writer, tmp_path, original)
    assert first.returncode == 0, "the first run should have changed the file"
    after_first = target.read_text(encoding="utf-8")

    second, _ = _write_config(config_writer, tmp_path, None)
    assert second.returncode == 3, (
        "an unchanged config.yaml must report 'already current', not 'updated'"
    )
    assert target.read_text(encoding="utf-8") == after_first


def test_the_config_edit_preserves_comments_and_every_other_setting(
        config_writer, tmp_path):
    original = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    result, target = _write_config(
        config_writer, tmp_path, original,
        keys="backend=ollama\nollama_model=qwen3.6:27b")
    assert result.returncode in (0, 3), result.stdout + result.stderr
    written = target.read_text(encoding="utf-8")
    before = original.splitlines()
    after = written.splitlines()
    assert len(before) == len(after), "the edit added or removed lines"
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert all("backend" in after[i] or "ollama_model" in after[i] for i in differing), (
        f"lines changed that should not have: {[after[i] for i in differing]}"
    )


def test_a_config_edit_that_fails_does_not_abort_the_whole_install(source):
    """Every other Python call is wrapped; this one used to abort under set -e."""
    block = _extract_block(source, 'step "Recording the model in config.yaml"')
    assert "set +e" in block and "CFG_STATUS=$?" in block, (
        "a config.yaml that cannot be written must not kill an install that "
        "has already downloaded the packages"
    )
    assert 'record "config.yaml" "failed"' in block


# --------------------------------------------------------------------------- #
#  The bridge into jarvis.runtime, executed rather than greped
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bridge(source: str) -> str:
    return _embedded_python(source, "JARVIS_RT_ACTION")


#: A stub jarvis.runtime whose every answer comes from the environment, so one
#: package serves every branch below.
STUB_RUNTIME = """
import json, os


def _answer(name, default):
    return json.loads(os.environ.get(name, default))


class _Ollama:
    def install_plan(self, cfg=None):
        return _answer("STUB_PLAN", '{"action": "install", "reason": "planned"}')

    def install(self, cfg=None, **kw):
        return _answer("STUB_INSTALL",
                       '{"ok": true, "action": "installed", "installed_now": true}')

    def start_server(self, cfg=None, **kw):
        return _answer("STUB_SERVE", '{"ok": true, "action": "already-running"}')

    def ensure_model(self, cfg=None, *, tag=None, **kw):
        if tag is None:
            raise AssertionError("the tag was passed positionally: cfg=%r" % (cfg,))
        return _answer("STUB_MODEL", '{"ok": true, "action": "present"}')

    def install_service(self, cfg=None, **kw):
        return _answer("STUB_SVC",
                       '{"ok": true, "written": true, "enabled": true, "path": "/u"}')


ollama = _Ollama()


class _Assets:
    def provision_all(self, cfg=None, **kw):
        return _answer("STUB_ASSETS", '{"ok": true, "summary": [], "components": '
                                      '{"voice": {"action": "present"}, '
                                      '"stt": {"action": "present"}}}')


assets = _Assets()


def status(cfg=None):
    return {"ready": False, "missing": ["model"], "ollama": {"installed": True}}
"""

#: The exit codes install.sh maps onto summary lines.
CHANGED, FAILED, CURRENT, SKIPPED, ABSENT = 0, 1, 3, 4, 5


def _bridge(bridge_src: str, tmp_path: Path, action: str, arg: str = "", **stub):
    stubs = _stub_runtime(tmp_path, STUB_RUNTIME)
    env = {"JARVIS_RT_ACTION": action, "JARVIS_RT_ARG": arg,
           "JARVIS_RT_UPDATE": stub.pop("update", "1"), "JARVIS_RT_CONFIG": ""}
    env.update(stub)
    return _run_embedded(bridge_src, tmp_path, env=env, extra_path=stubs)


@pytest.mark.parametrize("action,stub,expected", [
    ("ollama", {}, CHANGED),
    ("ollama", {"STUB_INSTALL": '{"ok": true, "action": "current"}'}, CURRENT),
    ("ollama", {"STUB_INSTALL": '{"ok": false, "reason": "no disk"}'}, FAILED),
    ("serve", {}, CURRENT),
    ("serve", {"STUB_SERVE": '{"ok": true, "action": "spawned"}'}, CHANGED),
    ("serve", {"STUB_SERVE": '{"ok": false, "reason": "no binary"}'}, FAILED),
    ("assets", {"STUB_ASSETS": '{"ok": true, "summary": [], "components": '
                               '{"voice": {"action": "downloaded"}}}'}, CHANGED),
    ("assets", {"STUB_ASSETS": '{"ok": false, "summary": []}'}, FAILED),
])
def test_the_bridge_maps_runtime_results_onto_the_right_summary_line(
        bridge, tmp_path, action, stub, expected):
    result = _bridge(bridge, tmp_path, action, **stub)
    assert result.returncode == expected, (
        f"{action} with {stub} exited {result.returncode}, wanted {expected}\n"
        f"{result.stdout}{result.stderr}"
    )


def test_the_model_pull_passes_the_tag_by_keyword(bridge, tmp_path):
    """`ensure_model(cfg=None, *, tag=None)` — a positional binds it to cfg.

    That mistake fails silently: the wrong model, or none, with a cheerful
    summary line. The stub asserts on it, so this test is the alarm.
    """
    result = _bridge(bridge, tmp_path, "model", "qwen3.6:27b")
    assert result.returncode == CURRENT, result.stdout + result.stderr
    assert "AssertionError" not in result.stderr


def test_a_unit_that_could_not_be_enabled_is_not_reported_as_enabled(bridge, tmp_path):
    """install_service returns ok=True for a unit it merely wrote to disk.

    A unit that was never enabled does not start at login, and the owner finds
    out the next time they reboot rather than now.
    """
    result = _bridge(
        bridge, tmp_path, "ollama-service",
        STUB_SVC='{"ok": true, "written": true, "enabled": false, "path": "/u",'
                 ' "reason": "systemctl --user enable failed"}')
    assert result.returncode == FAILED, (
        "a written-but-not-enabled unit must not be reported as a success"
    )
    assert "enable failed" in result.stderr


def test_a_unit_that_was_enabled_is_reported_as_installed(bridge, tmp_path):
    result = _bridge(bridge, tmp_path, "ollama-service")
    assert result.returncode == CHANGED, result.stdout + result.stderr
    result = _bridge(
        bridge, tmp_path, "ollama-service",
        STUB_SVC='{"ok": true, "written": false, "enabled": true, "path": "/u"}')
    assert result.returncode == CURRENT, "an unchanged unit is not an update"


def test_no_update_does_not_bump_ollama(bridge, tmp_path):
    """--no-update promises no version bumps, and ollama is a version."""
    upgrade = '{"action": "upgrade", "reason": "newer exists"}'
    frozen = _bridge(bridge, tmp_path, "ollama", update="0", STUB_PLAN=upgrade)
    assert frozen.returncode == CURRENT, frozen.stdout + frozen.stderr
    assert "--no-update" in frozen.stderr

    moving = _bridge(bridge, tmp_path, "ollama", update="1", STUB_PLAN=upgrade)
    assert moving.returncode == CHANGED, moving.stdout + moving.stderr


def test_a_checkout_without_jarvis_runtime_reports_absent_not_failure(bridge, tmp_path):
    (tmp_path / "empty").mkdir()
    result = _run_embedded(
        bridge, tmp_path,
        env={"JARVIS_RT_ACTION": "ollama", "JARVIS_RT_ARG": "",
             "JARVIS_RT_UPDATE": "1", "JARVIS_RT_CONFIG": ""},
        extra_path=tmp_path / "empty",
    )
    assert result.returncode == ABSENT, (
        "an older checkout must fall through to the printed manual commands"
    )


def test_an_unknown_bridge_action_fails_loudly(bridge, tmp_path):
    result = _bridge(bridge, tmp_path, "definitely-not-an-action")
    assert result.returncode == FAILED
    assert "unknown runtime action" in result.stderr


# --------------------------------------------------------------------------- #
#  The launcher symlink
# --------------------------------------------------------------------------- #
def _extract_nested_function(source: str, name: str) -> str:
    """Pull out a function defined inside an `if`, dedented so it can be run."""
    body = source.splitlines()
    start = next(i for i, line in enumerate(body)
                 if re.match(r"^\s*%s\(\)\s*\{" % re.escape(name), line))
    pad = len(body[start]) - len(body[start].lstrip())
    closing = " " * pad + "}"
    end = next(i for i in range(start + 1, len(body)) if body[i] == closing)
    return "\n".join(line[pad:] if line.startswith(" " * pad) else line
                     for line in body[start:end + 1])


@pytest.fixture
def symlinks(tmp_path: Path, bash: str):
    """Skip where the filesystem or the OS will not make a symlink."""
    probe = tmp_path / "probe"
    probe.write_text("x", encoding="utf-8")
    made = subprocess.run(
        [bash, "-c", 'ln -s "$1" "$2"', "sh", str(probe), str(tmp_path / "probe-link")],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if made.returncode != 0 or not (tmp_path / "probe-link").is_symlink():
        pytest.skip("this filesystem does not support symlinks")
    return True


def test_a_stale_link_is_diagnosed_rather_than_blamed_on_another_tool(
        bash, source, tmp_path, symlinks):
    """A dangling symlink is invisible to `-e`, so it looked like someone else's.

    Rebuilding the virtualenv somewhere else leaves ~/.local/bin/jarvis pointing
    at nothing. The old code fell through to "already exists and is not ours",
    which is untrue and gives the owner nothing to act on: `jarvis` stays
    "command not found" for ever with no hint of why. The installer still does
    not remove a name it found in place — it says exactly what to type.
    """
    new_venv = tmp_path / "venv" / "bin"
    new_venv.mkdir(parents=True)
    console = new_venv / "jarvis"
    console.write_text("#!/bin/sh\n", encoding="utf-8")
    console.chmod(0o755)

    snippet = "\n".join([
        STUBS,
        _extract_nested_function(source, "link_one"),
        'ln -s "$TMP/gone/bin/jarvis" "$TMP/bin/jarvis"',
        'link_one "$TARGET" "$TMP/bin/jarvis" && echo LINKED || echo REFUSED',
    ])
    (tmp_path / "bin").mkdir()
    result = _run(bash, snippet, env={"TMP": str(tmp_path), "TARGET": str(console)},
                  cwd=tmp_path)
    out = result.stdout
    assert "REFUSED" in out, out + result.stderr
    assert "points at nothing" in out, (
        f"a broken link of our own must be named as such, not disowned: {out!r}"
    )
    assert "is not ours" not in out, "the misleading message is still printed"
    assert "rm " in out, "the owner is not told how to clear it"
    assert (tmp_path / "bin" / "jarvis").is_symlink(), (
        "the installer removed a name it found in place"
    )


def test_a_name_belonging_to_something_else_is_still_never_touched(
        bash, source, tmp_path):
    """The repair above must not become a licence to overwrite."""
    (tmp_path / "bin").mkdir()
    intruder = tmp_path / "bin" / "jarvis"
    intruder.write_text("#!/bin/sh\necho another tool\n", encoding="utf-8")
    target = tmp_path / "venv" / "bin"
    target.mkdir(parents=True)
    (target / "jarvis").write_text("#!/bin/sh\n", encoding="utf-8")

    snippet = "\n".join([
        STUBS,
        _extract_nested_function(source, "link_one"),
        'link_one "$TARGET" "$TMP/bin/jarvis" && echo LINKED || echo REFUSED',
    ])
    result = _run(bash, snippet,
                  env={"TMP": str(tmp_path), "TARGET": str(target / "jarvis")},
                  cwd=tmp_path)
    assert "REFUSED" in result.stdout, result.stdout + result.stderr
    assert intruder.read_text(encoding="utf-8") == "#!/bin/sh\necho another tool\n"


def test_speech_assets_fall_back_to_the_old_setup_command(source):
    block = _extract_block(source, 'step "Fetching the speech models')
    assert "-m jarvis setup" in block, (
        "when jarvis.runtime is absent the voice download must fall back to "
        "the behaviour that has always worked"
    )


# --------------------------------------------------------------------------- #
#  Progress during a multi-gigabyte download
#
#  install(), ensure_model() and provision_all() all accept a `progress`
#  callback and report through it as bytes move -- but nothing in the bridge
#  ever passed one, so a real download produced ZERO terminal output for
#  however long it took. A 16 GB pull on an ordinary connection is easily
#  twenty minutes of total silence after "checking space available", which is
#  indistinguishable from a hang. These tests prove the callback is real, is
#  actually wired to the three calls that can be large, and turns into
#  visible, throttled output rather than either silence or a flood.
# --------------------------------------------------------------------------- #
def test_the_bridge_defines_a_progress_printer(bridge):
    assert "def make_progress" in bridge
    assert re.search(r"^PROGRESS\s*=\s*make_progress\(\)", bridge, re.M), (
        "make_progress() is defined but never instantiated at module scope"
    )


@pytest.mark.parametrize("call,keyword", [
    ("rt.ollama.install(cfg", "progress=PROGRESS"),
    ("rt.ollama.ensure_model(cfg", "progress=PROGRESS"),
    ("rt.assets.provision_all(cfg", "progress=PROGRESS"),
])
def test_every_large_download_is_given_the_progress_callback(bridge, call, keyword):
    line = next((l for l in bridge.splitlines() if call in l), None)
    assert line is not None, f"call site {call!r} not found in the bridge"
    assert keyword in line, (
        f"{line.strip()!r} does not pass {keyword} -- this download will run "
        f"in total silence"
    )


#: A stub whose install/ensure_model/provision_all actually DRIVE the caller's
#: progress function with realistic events, so the bridge's own printer can be
#: exercised for real rather than merely proven present.
STUB_RUNTIME_WITH_LIVE_PROGRESS = (
    "def _feed(progress):\n"
    "    if progress is None:\n"
    "        raise AssertionError('no progress callback was passed')\n"
    "    if not callable(progress):\n"
    "        raise AssertionError('progress is not callable: %r' % (progress,))\n"
    "    progress({'phase': 'pull', 'tag': 'm:1', 'message': 'pulling m:1 (~16 GB)'})\n"
    "    for i in range(0, 101, 25):\n"
    "        progress({'phase': 'pull', 'tag': 'm:1',\n"
    "                  'completed': i * 160_000_000, 'total': 16_000_000_000,\n"
    "                  'percent': float(i)})\n"
    "\n"
    "\n"
    "class _Ollama:\n"
    "    def install_plan(self, cfg=None):\n"
    "        return {'action': 'install', 'reason': 'planned'}\n"
    "\n"
    "    def install(self, cfg=None, progress=None, **kw):\n"
    "        _feed(progress)\n"
    "        return {'ok': True, 'action': 'installed', 'installed_now': True}\n"
    "\n"
    "    def start_server(self, cfg=None, **kw):\n"
    "        return {'ok': True, 'action': 'already-running'}\n"
    "\n"
    "    def ensure_model(self, cfg=None, *, tag=None, progress=None, **kw):\n"
    "        _feed(progress)\n"
    "        return {'ok': True, 'action': 'pulled', 'reason': 'm:1 pulled (success).'}\n"
    "\n"
    "    def install_service(self, cfg=None, **kw):\n"
    "        return {'ok': True, 'written': True, 'enabled': True, 'path': '/u'}\n"
    "\n"
    "\n"
    "ollama = _Ollama()\n"
    "\n"
    "\n"
    "class _Assets:\n"
    "    def provision_all(self, cfg=None, progress=None, **kw):\n"
    "        _feed(progress)\n"
    "        return {'ok': True, 'summary': [], 'components':\n"
    "                {'voice': {'action': 'present'}, 'stt': {'action': 'present'}}}\n"
    "\n"
    "\n"
    "assets = _Assets()\n"
    "\n"
    "\n"
    "def status(cfg=None):\n"
    "    return {'ready': False, 'missing': [], 'ollama': {}}\n"
)


@pytest.mark.parametrize("action,arg", [
    ("ollama", ""),
    ("model", "m:1"),
    ("assets", ""),
])
def test_progress_reaches_the_callback_and_becomes_visible_output(
        bridge, tmp_path, action, arg):
    """End to end: a stub that behaves like a real multi-gigabyte pull.

    If this callback were still missing, the stub's own AssertionError
    ("no progress callback was passed") would surface as a FAILED exit code --
    so a regression here is caught even without inspecting stderr. The stderr
    assertions on top of that prove the output is not just present but
    actually informative: a percentage and a GB figure the owner can watch.
    """
    stubs = _stub_runtime(tmp_path, STUB_RUNTIME_WITH_LIVE_PROGRESS)
    env = {"JARVIS_RT_ACTION": action, "JARVIS_RT_ARG": arg,
           "JARVIS_RT_UPDATE": "1", "JARVIS_RT_CONFIG": ""}
    result = _run_embedded(bridge, tmp_path, env=env, extra_path=stubs)

    # FAILED (1) is exactly the code the stub's own "no progress callback was
    # passed" AssertionError produces, via the bridge's catch-all handler --
    # so this alone would catch the callback going missing again. The exact
    # non-failure code differs by action (assets reports CURRENT when its
    # components are merely "present", not freshly fetched), which is correct
    # and not what this test is about.
    assert result.returncode != FAILED, (
        f"the stub's own progress-callback assertion failed:\n{result.stderr}"
    )
    assert "16.00 GB" in result.stderr or "16 GB" in result.stderr, (
        f"no size information reached the terminal:\n{result.stderr!r}"
    )
    assert "%" in result.stderr, f"no percentage reached the terminal:\n{result.stderr!r}"


def test_progress_output_is_throttled_not_flooded(bridge, tmp_path):
    """A fast local link must not turn into one printed line per chunk."""
    stub = (
        "class _Ollama:\n"
        "    def install_plan(self, cfg=None):\n"
        "        return {'action': 'install', 'reason': 'planned'}\n"
        "\n"
        "    def install(self, cfg=None, progress=None, **kw):\n"
        "        for i in range(500):\n"
        "            progress({'phase': 'download', 'completed': i * 1_000_000,\n"
        "                      'total': 500_000_000})\n"
        "        return {'ok': True, 'action': 'installed', 'installed_now': True}\n"
        "\n"
        "    def start_server(self, cfg=None, **kw):\n"
        "        return {'ok': True, 'action': 'already-running'}\n"
        "\n"
        "    def ensure_model(self, cfg=None, *, tag=None, progress=None, **kw):\n"
        "        return {'ok': True, 'action': 'present'}\n"
        "\n"
        "    def install_service(self, cfg=None, **kw):\n"
        "        return {'ok': True, 'written': True, 'enabled': True, 'path': '/u'}\n"
        "\n"
        "\n"
        "ollama = _Ollama()\n"
        "\n"
        "\n"
        "class _Assets:\n"
        "    def provision_all(self, cfg=None, progress=None, **kw):\n"
        "        return {'ok': True, 'summary': [], 'components': {}}\n"
        "\n"
        "\n"
        "assets = _Assets()\n"
        "\n"
        "\n"
        "def status(cfg=None):\n"
        "    return {'ready': False, 'missing': [], 'ollama': {}}\n"
    )
    stubs = _stub_runtime(tmp_path, stub)
    env = {"JARVIS_RT_ACTION": "ollama", "JARVIS_RT_ARG": "",
           "JARVIS_RT_UPDATE": "1", "JARVIS_RT_CONFIG": ""}
    result = _run_embedded(bridge, tmp_path, env=env, extra_path=stubs)

    assert result.returncode == CHANGED, result.stderr
    lines = [l for l in result.stderr.splitlines() if l.strip()]
    assert len(lines) < 20, (
        f"500 events produced {len(lines)} printed lines -- the throttle is "
        f"not throttling:\n{result.stderr[:2000]}"
    )
    assert len(lines) >= 1, "500 real progress events produced no output at all"


# --------------------------------------------------------------------------- #
#  The progress printer's own logic, unit-tested in-process
# --------------------------------------------------------------------------- #
def _bridge_definitions(bridge_src: str) -> dict:
    """Exec everything up to the dispatcher, so make_progress()/say() can be
    called directly without going through a subprocess or triggering an
    actual runtime action."""
    cut = bridge_src.index("handler = ACTIONS.get(ACTION)")
    namespace: dict = {"__name__": "bridge_defs"}
    exec(compile(bridge_src[:cut], "<bridge>", "exec"), namespace)
    return namespace


class _FakeStream:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty
        self.chunks: List[str] = []

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def test_make_progress_throttles_by_time_and_by_percent_movement(bridge, monkeypatch):
    ns = _bridge_definitions(bridge)
    fake = _FakeStream(is_tty=False)
    monkeypatch.setattr(ns["sys"], "stderr", fake)

    clock = [0.0]
    monkeypatch.setattr(ns["time"], "monotonic", lambda: clock[0])

    emit = ns["make_progress"]()
    for i in range(0, 40, 2):
        emit({"phase": "pull", "tag": "m", "completed": i * 1_000_000, "total": 100_000_000})
    printed_before = fake.text.count("\n")

    clock[0] += 1.5
    emit({"phase": "pull", "tag": "m", "completed": 60_000_000, "total": 100_000_000})
    printed_after = fake.text.count("\n")

    assert printed_before == 1, (
        f"a burst of same-second events should print once (the first), got "
        f"{printed_before} lines"
    )
    assert printed_after == 2, "a new event after the throttle window did not print"


def test_make_progress_always_shows_a_message_event(bridge, monkeypatch):
    """"pulling X (~16 GB)" must appear even right after a numeric update --
    it is rare and informative, never spam."""
    ns = _bridge_definitions(bridge)
    fake = _FakeStream(is_tty=False)
    monkeypatch.setattr(ns["sys"], "stderr", fake)
    clock = [0.0]
    monkeypatch.setattr(ns["time"], "monotonic", lambda: clock[0])

    emit = ns["make_progress"]()
    emit({"phase": "model", "message": "pulling m:1 (~16 GB)"})
    emit({"phase": "model", "message": "pulling m:1 (~16 GB)"})

    assert fake.text.count("pulling m:1") == 2


def test_say_starts_a_fresh_line_after_an_open_tty_progress_bar(bridge, monkeypatch):
    """On a real terminal, progress overwrites itself with \\r. A say() call
    arriving mid-transfer must not get smeared onto the tail of that line."""
    ns = _bridge_definitions(bridge)
    fake = _FakeStream(is_tty=True)
    monkeypatch.setattr(ns["sys"], "stderr", fake)

    emit = ns["make_progress"]()
    emit({"phase": "pull", "tag": "m", "completed": 500, "total": 1000, "percent": 50.0})
    assert "\r" in fake.text and not fake.text.endswith("\n"), (
        "setup assumption failed: the progress line should be open (no trailing \\n)"
    )

    ns["say"]("interrupting mid-transfer")
    assert fake.text.endswith("\n    interrupting mid-transfer\n"), (
        f"say() did not cleanly close the open progress line:\n{fake.text!r}"
    )


def test_make_progress_never_raises_on_a_malformed_event(bridge, monkeypatch):
    """_notify() in jarvis.runtime already swallows a broken callback, but the
    printer itself should not depend on that safety net."""
    ns = _bridge_definitions(bridge)
    fake = _FakeStream(is_tty=False)
    monkeypatch.setattr(ns["sys"], "stderr", fake)
    emit = ns["make_progress"]()

    for garbage in ({}, {"phase": None}, {"completed": "not a number"},
                    {"total": object()}, {"percent": float("nan")}):
        emit(garbage)  # must not raise
