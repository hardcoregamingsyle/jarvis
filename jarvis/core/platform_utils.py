"""Cross-platform helpers.  The one place that knows about Windows vs Linux.

Nothing else in the codebase should branch on ``sys.platform`` directly; import
the helpers from here so behaviour stays consistent and testable.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"


def os_name() -> str:
    if IS_WINDOWS:
        return "windows"
    if IS_MAC:
        return "macos"
    if IS_LINUX:
        return "linux"
    return sys.platform


# --------------------------------------------------------------------------- #
#  Path safety
# --------------------------------------------------------------------------- #
def is_drive_relative(raw: str) -> bool:
    """True for Windows paths like ``C:`` or ``C:foo`` that omit the separator.

    These are *drive-relative*: ``C:`` means "the current directory on drive C",
    not the drive root.  Resolving one silently yields the process's working
    directory, so a destructive operation on ``"C:"`` would hit the CWD instead
    of the root the caller obviously meant.  Every path that reaches a delete,
    move or overwrite must be screened for this.
    """
    if not raw or not isinstance(raw, str):
        return False
    text = raw.strip()
    if len(text) < 2 or text[1] != ":":
        return False
    if not text[0].isalpha():
        return False
    # "C:" alone, or "C:foo" — anything not followed by a separator.
    return len(text) == 2 or text[2] not in ("\\", "/")


def resolve_path(raw: str, *, strict: bool = False) -> Path:
    """Expand and resolve a user-supplied path, anchoring drive-relative forms.

    The trap: ``Path("C:").resolve()`` does not return the drive root, it
    silently returns the process working directory, so an operation aimed at
    ``C:\\`` lands on whatever folder JARVIS happens to be running in — which is
    how an earlier version of this repository deleted itself.  This function
    therefore anchors any drive-relative input to that drive's root instead:
    ``"C:"`` becomes ``C:\\`` and ``"C:foo"`` becomes ``C:\\foo``, never
    anything derived from the CWD.

    Pass ``strict=True`` to restore the old behaviour of raising ``ValueError``
    on a drive-relative path rather than interpreting it.
    """
    if is_drive_relative(raw):
        text = raw.strip()
        if strict:
            raise ValueError(
                f"{raw!r} is a drive-relative path: on Windows it means 'the current "
                f"directory on that drive', not the drive root. Use an explicit path "
                f"such as {text[0]}:\\ instead."
            )
        # Build the absolute form by hand.  Anything routed through resolve()
        # here would re-introduce the working directory this branch exists to
        # avoid.
        root = f"{text[0]}:\\"
        remainder = text[2:].lstrip("\\/")
        return Path(os.path.normpath(root + remainder) if remainder else root)
    return Path(raw).expanduser().resolve()


def is_filesystem_root(path: Path) -> bool:
    """True when ``path`` is a drive root (``C:\\``) or the POSIX root (``/``)."""
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False
    return resolved.parent == resolved


# --------------------------------------------------------------------------- #
#  Standard directories (no external "platformdirs" dependency)
# --------------------------------------------------------------------------- #
APP_NAME = "Jarvis"


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def data_dir() -> Path:
    """Where JARVIS keeps its database, models, logs, generated tools."""
    override = os.environ.get("JARVIS_HOME")
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(_home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if IS_MAC:
        return _home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
    return Path(base) / APP_NAME.lower()


def config_dir() -> Path:
    override = os.environ.get("JARVIS_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        base = os.environ.get("APPDATA") or str(_home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if IS_MAC:
        return _home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(_home() / ".config")
    return Path(base) / APP_NAME.lower()


def cache_dir() -> Path:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(_home() / "AppData" / "Local")
        return Path(base) / APP_NAME / "Cache"
    if IS_MAC:
        return _home() / "Library" / "Caches" / APP_NAME
    base = os.environ.get("XDG_CACHE_HOME") or str(_home() / ".cache")
    return Path(base) / APP_NAME.lower()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
#  Command execution
# --------------------------------------------------------------------------- #
@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def default_shell() -> list:
    """The argv prefix used to run a shell command string."""
    if IS_WINDOWS:
        # PowerShell if present, else cmd.exe.
        pwsh = which("pwsh") or which("powershell")
        if pwsh:
            return [pwsh, "-NoProfile", "-NonInteractive", "-Command"]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c"]
    shell = os.environ.get("SHELL", "/bin/sh")
    return [shell, "-lc"]


def run_command(
    command,
    *,
    timeout: float = 60.0,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    shell: bool = False,
) -> CommandResult:
    """Run a command robustly and never raise on a non-zero exit.

    ``shell=True`` runs ``command`` (a string) through the platform's shell.
    ``shell=False`` expects an argv list (or a string that will be wrapped by
    the platform shell prefix) and is the safer default.
    """
    if shell:
        if not isinstance(command, str):
            command = subprocess.list2cmdline(list(command))
        argv: Sequence[str] = [*default_shell(), command]
    elif isinstance(command, str):
        argv = [*default_shell(), command]
    else:
        argv = list(command)

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            encoding="utf-8",
            errors="replace",
        )
        return CommandResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return CommandResult(124, out, err + f"\n[timed out after {timeout}s]", timed_out=True)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))


def open_path(target: str) -> CommandResult:
    """Open a file/folder/URL with the OS default handler."""
    if IS_WINDOWS:
        try:
            os.startfile(target)  # type: ignore[attr-defined]
            return CommandResult(0, "", "")
        except OSError as exc:
            return CommandResult(1, "", str(exc))
    opener = "open" if IS_MAC else "xdg-open"
    return run_command([opener, target], timeout=15)


def system_summary() -> dict:
    """A quick, dependency-free description of the host."""
    return {
        "os": os_name(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "hostname": platform.node(),
    }


__all__ = [
    "IS_WINDOWS", "IS_LINUX", "IS_MAC", "os_name",
    "is_drive_relative", "resolve_path", "is_filesystem_root",
    "data_dir", "config_dir", "cache_dir", "ensure_dir",
    "CommandResult", "which", "default_shell", "run_command",
    "open_path", "system_summary",
]
