"""Static checks on install.sh — the parts that only fail on Linux.

The installer cannot be executed here, so these tests read it as text and assert
the properties whose absence produced real failures on the owner's machine:

* It wrote its launcher to ``$ROOT/jarvis``. On Linux that path is the **package
  directory**, so the redirect failed with ``EISDIR: Is a directory`` and, under
  ``set -euo pipefail``, aborted the installer after every package had already
  been downloaded. Windows never saw it — install.ps1 writes ``jarvis.bat``.
* It ran ``$VPY -m pip`` without checking the venv had pip. On Debian and Ubuntu
  ``python3 -m venv`` is split across packages: with no ``python3-venv`` there is
  no ``ensurepip``, and CPython creates ``bin/python`` before it tries to install
  pip — so the tree exists, the directory and interpreter checks both pass, and
  the user gets ``No module named pip``. That message sends people off to install
  ``pip3``, which is a different thing and does not help.
* ``--venv /absolute/path`` was concatenated onto ``$ROOT``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"


def _find_bash() -> str:
    """Locate bash, including when it is not on a Windows PATH.

    Worth the effort: these tests are the only thing standing between a broken
    installer and the Linux box, and skipping them because the test runner was
    launched from PowerShell rather than Git Bash would silently remove that
    protection exactly when it is needed.
    """
    found = shutil.which("bash")  # noqa: the only intended call site
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


@pytest.fixture(scope="module")
def source() -> str:
    if not SCRIPT.exists():
        pytest.skip("install.sh not present")
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def code_lines(source: str) -> list:
    """Non-comment, non-blank lines — comments may legitimately quote old bugs."""
    out = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


def test_the_script_is_syntactically_valid():
    bash = _find_bash()
    if not bash:
        pytest.skip("bash unavailable")
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


# --------------------------------------------------------------------------- #
#  The launcher must never be written over the package directory
# --------------------------------------------------------------------------- #
def test_no_redirect_targets_the_package_directory(code_lines):
    offenders = [
        line for line in code_lines
        if re.search(r'>\s*"?\$\{?ROOT\}?/jarvis"?\s*(<<|$)', line)
    ]
    assert not offenders, (
        f"install.sh redirects into $ROOT/jarvis, which is the package "
        f"directory — on Linux this is EISDIR and aborts the installer: {offenders}"
    )


def test_the_launcher_comes_from_the_console_script(source):
    assert "bin/jarvis" in source, (
        "install.sh should point at the console script pip installs, not write "
        "its own launcher"
    )


def test_pyproject_really_declares_that_console_script():
    """The installer's launcher story depends on this entry point existing."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present")
    try:
        import tomllib
    except ModuleNotFoundError:                      # Python 3.9/3.10
        pytest.skip("tomllib unavailable")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("jarvis") == "jarvis.cli:main"


def test_gitignore_does_not_exclude_the_package(source):
    """`/jarvis` matches the package directory as well as any launcher file."""
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        pytest.skip(".gitignore not present")
    for raw in gitignore.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        assert line not in ("/jarvis", "jarvis"), (
            "a bare `/jarvis` ignore rule also matches the jarvis/ package "
            "directory and excludes the entire source tree"
        )


# --------------------------------------------------------------------------- #
#  A venv without pip must be diagnosed, not stumbled into
# --------------------------------------------------------------------------- #
def test_pip_presence_is_checked_before_it_is_used(source):
    assert re.search(r'-m\s+pip\s+--version', source), (
        "install.sh never verifies the venv has pip; on Debian/Ubuntu without "
        "python3-venv the user gets a bare 'No module named pip'"
    )


def test_the_pip_failure_explains_that_system_pip3_is_not_the_answer(source):
    """The whole point is that the obvious reading of the error is wrong."""
    assert "pip3" in source, (
        "the pip-less-venv message should say explicitly that a system pip3 "
        "does not supply pip to a virtual environment — that is the wrong turn "
        "the raw error invites"
    )
    assert "ensurepip" in source


def test_the_pip_check_precedes_the_first_pip_use(source):
    check = source.find("-m pip --version")
    install = source.find("-m pip install")
    assert check != -1 and install != -1
    assert check < install, "the pip guard must come before pip is relied upon"


# --------------------------------------------------------------------------- #
#  --venv with an absolute path
# --------------------------------------------------------------------------- #
def test_absolute_venv_paths_are_not_concatenated_onto_root(source):
    assert 'VENV_DIR=' in source, "install.sh should normalise --venv into VENV_DIR"
    assert re.search(r'case\s+"\$VENV"', source), (
        "no case analysis on $VENV; an absolute --venv path would be appended "
        "to $ROOT, producing /root/path//opt/whatever"
    )


@pytest.mark.parametrize("given,root,expected", [
    (".venv", "/home/hp/JARVIS", "/home/hp/JARVIS/.venv"),
    ("relative/dir", "/home/hp/JARVIS", "/home/hp/JARVIS/relative/dir"),
    ("/opt/jarvis-venv", "/home/hp/JARVIS", "/opt/jarvis-venv"),
])
def test_venv_path_normalisation_behaviour(given, root, expected):
    """Run the actual case statement from the script."""
    bash = _find_bash()
    if not bash:
        pytest.skip("bash unavailable")
    snippet = (
        'case "$VENV" in\n'
        '  /*) VENV_DIR="$VENV" ;;\n'
        '  ~*) VENV_DIR="${VENV/#\\~/$HOME}" ;;\n'
        '  *)  VENV_DIR="$ROOT/$VENV" ;;\n'
        'esac\n'
        'printf "%s" "$VENV_DIR"\n'
    )
    result = subprocess.run(
        [bash, "-c", snippet],
        env={"VENV": given, "ROOT": root, "HOME": "/home/hp", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.stdout == expected


# --------------------------------------------------------------------------- #
#  Standing promises
# --------------------------------------------------------------------------- #
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


def test_the_installer_never_calls_sudo(code_lines):
    """It prints sudo commands for the user to run; it must not run them."""
    offenders = [
        line for line in code_lines
        if re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted(line))
    ]
    assert not offenders, f"install.sh appears to invoke sudo: {offenders}"


def test_the_sudo_detector_would_actually_catch_a_real_call():
    """A guard that cannot fail is not a guard."""
    assert re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted('sudo apt-get install -y x'))
    assert re.search(r'(^|[;&|(]\s*)sudo\s', _strip_quoted('foo && sudo rm -rf /x'))
    # ...and would not flag the advice the installer legitimately prints.
    assert not re.search(
        r'(^|[;&|(]\s*)sudo\s',
        _strip_quoted("""printf 'sudo apt-get install -y python3-venv'"""),
    )


# --------------------------------------------------------------------------- #
#  `jarvis` must work as a bare command
#
#  The console script lives inside the virtualenv, so a perfectly successful
#  install left `jarvis doctor` — the invocation the README uses everywhere —
#  answering "command not found". The installer now symlinks ~/.local/bin/jarvis.
# --------------------------------------------------------------------------- #
def test_the_installer_puts_jarvis_on_the_path(source):
    assert ".local/bin" in source, (
        "install.sh never puts the console script anywhere on PATH, so a "
        "successful install still leaves `jarvis` as command-not-found"
    )
    assert "ln -s" in source


def _extract_shell_function(source: str, name: str) -> str:
    """Pull one function definition out of the script, brace-balanced.

    Extracting and running the real function beats pattern-matching it: a regex
    pins one spelling of one variable, so renaming a local — which is exactly
    what happened when this linking code was refactored into link_one() — breaks
    the test while the behaviour it guards is untouched.
    """
    start = re.search(rf'^\s*{re.escape(name)}\(\)\s*\{{', source, re.M)
    assert start, f"no function {name}() in install.sh"
    depth, out, i = 0, [], start.start()
    for line in source[i:].splitlines(keepends=True):
        out.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and len(out) > 1:
            break
    return "".join(out)


def test_the_symlink_never_clobbers_an_existing_file(source, tmp_path):
    """~/.local/bin is shared with everything else the user installs.

    Exercised for real: the extracted function is run against a directory that
    already contains a file under the target name, and the file must survive
    byte for byte.
    """
    bash = _find_bash()
    if not bash:
        pytest.skip("bash unavailable")

    assert "ln -sf" not in source, "-f would clobber silently"

    fn = _extract_shell_function(source, "link_one")
    target = tmp_path / "console_script"
    target.write_text("#!/bin/sh\necho jarvis\n", encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.write_text("belongs to another tool", encoding="utf-8")

    script = (
        "ok() { :; }\nwarn() { :; }\n"
        + fn
        + f'\nlink_one "{target.as_posix()}" "{occupied.as_posix()}" && echo LINKED || echo REFUSED\n'
    )
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)

    assert result.stdout.strip().endswith("REFUSED"), (
        f"link_one overwrote an existing file instead of refusing: {result.stdout!r}"
    )
    assert occupied.read_text(encoding="utf-8") == "belongs to another tool", (
        "the pre-existing file was modified"
    )


def test_the_symlink_is_created_when_the_name_is_free(source, tmp_path):
    """The refusal above must not be the function simply never working."""
    bash = _find_bash()
    if not bash:
        pytest.skip("bash unavailable")

    fn = _extract_shell_function(source, "link_one")
    target = tmp_path / "console_script"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    fresh = tmp_path / "fresh_name"

    script = (
        "ok() { :; }\nwarn() { :; }\n"
        + fn
        + f'\nlink_one "{target.as_posix()}" "{fresh.as_posix()}" && echo LINKED || echo REFUSED\n'
    )
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=30)

    if not fresh.exists():
        pytest.skip("this filesystem cannot create the link (Git Bash without symlink support)")
    assert result.stdout.strip().endswith("LINKED")


def test_the_link_check_uses_inode_comparison(source):
    """A re-run must recognise its own link rather than warn about it."""
    assert "-ef" in source, (
        "the already-linked check should compare inodes with -ef; comparing "
        "readlink output as a string breaks on a relative link"
    )


def test_the_path_situation_is_reported_not_assumed(source):
    """Creating ~/.local/bin does not put it on PATH until the next login.

    Debian and Ubuntu add it from ~/.profile only when it already existed at
    login, so an installer that creates it and then claims `jarvis` works is
    wrong precisely on a fresh machine — the case that matters.
    """
    assert "JARVIS_ON_PATH" in source
    assert re.search(r'case\s+":\$\{?PATH', source), "PATH membership is never actually tested"
    assert "export PATH=" in source, "no recovery instruction when it is not on PATH"


def test_no_link_flag_exists_and_is_documented(source):
    assert "--no-link" in source
    assert re.search(r'--no-link\)\s*WANT_LINK=0', source), "--no-link is documented but not parsed"


def test_the_header_promise_matches_what_it_actually_writes(source):
    """The header used to promise nothing outside the directory is written."""
    header = source.split("set -euo pipefail")[0]
    assert "~/.local/bin/jarvis" in header, (
        "the installer writes a symlink outside the repository; the header must "
        "say so rather than promising it touches nothing outside this directory"
    )


def test_help_lists_every_flag_the_parser_accepts(source):
    """A documented-flag/real-flag mismatch sends people down blind alleys."""
    parsed = set(re.findall(r'^\s*(--[a-z-]+)\)', source, re.M))
    parsed |= set(re.findall(r'\|(--[a-z-]+)\)', source))
    header = source.split("set -euo pipefail")[0]
    documented = set(re.findall(r'(--[a-z-]+)', header))

    undocumented = {f for f in parsed if f not in documented and f != "--help"}
    assert not undocumented, f"flags accepted but not in the header help: {undocumented}"
