"""Properties of the *repository* that only fail on a different OS.

This project is written on Windows and deployed on Linux, so a whole class of
defect is invisible where it is created:

* ``install.sh`` was committed with mode ``100644``. On Windows nothing notices
  — the filesystem has no executable bit to lose. On Linux ``./install.sh``
  answers ``bash: permission denied``, and the file looks perfectly normal.
* A shell script committed with CRLF endings makes the kernel search for an
  interpreter named ``bash\\r``, and the error — "bad interpreter: No such file
  or directory" — names a file that obviously exists.

Both are checked here, from Windows, against what git actually stores. Skipped
when git is unavailable or this is not a checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent

# Scripts that must be directly runnable as ./<name> on Linux.
EXECUTABLE_SCRIPTS = ("install.sh",)


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable or not a checkout: {result.stderr.strip()[:120]}")
    return result.stdout


@pytest.fixture(scope="module")
def tracked_modes() -> dict:
    """Map of tracked path -> git index mode ("100644" / "100755")."""
    modes = {}
    for line in _git("ls-files", "-s").splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 1 and path:
            modes[path.strip()] = parts[0]
    if not modes:
        pytest.skip("no tracked files reported")
    return modes


@pytest.mark.parametrize("script", EXECUTABLE_SCRIPTS)
def test_shell_scripts_are_committed_executable(script, tracked_modes):
    """Mode 100644 means `./install.sh` is 'permission denied' on Linux."""
    if script not in tracked_modes:
        pytest.skip(f"{script} is not tracked")
    assert tracked_modes[script] == "100755", (
        f"{script} is committed as mode {tracked_modes[script]}, so it will not "
        f"run as ./{script} on Linux. Fix with:\n"
        f"    git update-index --chmod=+x {script}"
    )


def test_every_tracked_shell_script_is_executable(tracked_modes):
    """Catches the next .sh someone adds, not just the one that broke."""
    offenders = [
        path for path, mode in tracked_modes.items()
        if path.endswith(".sh") and mode != "100755"
    ]
    assert not offenders, (
        f"shell scripts committed without the executable bit: {offenders}. "
        f"Run: git update-index --chmod=+x " + " ".join(offenders)
    )


@pytest.mark.parametrize("script", EXECUTABLE_SCRIPTS)
def test_shell_scripts_have_unix_line_endings_in_the_repository(script):
    """A CR in the shebang makes Linux hunt for an interpreter named 'bash\\r'."""
    blob = subprocess.run(
        ("git", "cat-file", "-p", f"HEAD:{script}"),
        cwd=ROOT, capture_output=True, timeout=30,
    )
    if blob.returncode != 0:
        pytest.skip(f"{script} not in HEAD yet")
    content = blob.stdout

    assert content, f"{script} is empty in HEAD"
    assert b"\r\n" not in content, (
        f"{script} is stored with CRLF line endings. On Linux this yields "
        f"'bad interpreter: No such file or directory'. .gitattributes should "
        f"pin it to eol=lf."
    )
    assert content.startswith(b"#!"), f"{script} has no shebang"
    first_line = content.split(b"\n", 1)[0]
    assert b"\r" not in first_line, "carriage return in the shebang line"


def test_gitattributes_pins_shell_scripts_to_lf():
    """Without this, a checkout with core.autocrlf=true recreates the bug."""
    attrs = ROOT / ".gitattributes"
    assert attrs.exists(), (
        ".gitattributes is missing; shell scripts will pick up CRLF on a "
        "Windows checkout and stop working on Linux"
    )
    text = attrs.read_text(encoding="utf-8")
    assert "*.sh" in text and "eol=lf" in text


def test_the_package_itself_is_tracked(tracked_modes):
    """A .gitignore rule once excluded the entire jarvis/ package.

    `/jarvis` was meant to ignore the launcher script install.sh generates, but
    the pattern matches the package directory too — staging a repository of
    tests and documentation with no source code in it.
    """
    package_files = [p for p in tracked_modes if p.startswith("jarvis/") and p.endswith(".py")]
    assert len(package_files) > 20, (
        f"only {len(package_files)} package modules are tracked — check that "
        f".gitignore is not excluding jarvis/"
    )
    for essential in ("jarvis/__init__.py", "jarvis/cli.py", "jarvis/app.py"):
        assert essential in tracked_modes, f"{essential} is not tracked"


def test_no_secrets_or_local_state_is_tracked(tracked_modes):
    """config.yaml may hold an api_key; memory.db holds every conversation."""
    forbidden = []
    for path in tracked_modes:
        name = path.lower()
        if name.endswith((".db", ".log", ".onnx", ".gguf")):
            forbidden.append(path)
        elif Path(name).name in ("config.yaml", "config.yml", "config.json", ".env"):
            forbidden.append(path)
        elif "audit" in name and name.endswith(".jsonl"):
            forbidden.append(path)
    assert not forbidden, f"these should never be committed: {forbidden}"
