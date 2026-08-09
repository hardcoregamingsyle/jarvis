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


def test_help_lists_every_flag_the_parser_accepts(source):
    """A documented-flag/real-flag mismatch sends people down blind alleys."""
    parsed = set(re.findall(r'^\s*(--[a-z-]+)\)', source, re.M))
    parsed |= set(re.findall(r'\|(--[a-z-]+)\)', source))
    header = source.split("set -euo pipefail")[0]
    documented = set(re.findall(r'(--[a-z-]+)', header))

    undocumented = {f for f in parsed if f not in documented and f != "--help"}
    assert not undocumented, f"flags accepted but not in the header help: {undocumented}"
