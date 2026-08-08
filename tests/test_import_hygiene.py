"""Structural guarantees about the package itself.

The promise that JARVIS boots on any machine — degraded, but running — rests
entirely on one rule: no heavy third-party package may be imported at module
level.  That rule is easy to break by accident and impossible to notice on a
developer machine where everything happens to be installed, so it is enforced
here by parsing the source rather than by importing it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "jarvis"

# Third-party packages that are large, slow to import, compiled, or
# hardware-dependent.  These get blocked outright in the subprocess check.
HEAVY = {
    "torch", "numpy", "scipy", "pandas", "transformers", "accelerate",
    "safetensors", "airllm", "sentence_transformers", "faster_whisper",
    "whisper", "vosk", "sounddevice", "simpleaudio", "pyaudio", "edge_tts",
    "pyttsx3", "piper", "piper_tts", "psutil", "requests", "httpx", "aiohttp",
    "bs4", "lxml", "mss", "PIL", "Pillow", "cv2", "pydub", "yaml",
    "sklearn", "matplotlib", "win32api", "win32com", "pywintypes",
    "pycaw", "keyboard", "wmi",
}

# Windows-only *standard library* modules.  They must never be imported at
# module level (that would break Linux), but they cannot be blocked at runtime
# because the stdlib itself imports them — `subprocess` pulls in msvcrt.
WINDOWS_ONLY_STDLIB = {"winreg", "winsound", "msvcrt", "_winapi"}

FORBIDDEN_AT_MODULE_LEVEL = HEAVY | WINDOWS_ONLY_STDLIB


def python_files() -> list:
    return sorted(PACKAGE.rglob("*.py"))


def module_level_imports(tree: ast.AST) -> set:
    """Top-level import names only — nested ones are lazy and therefore fine."""
    names: set = set()
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # A try/except ImportError import is a deliberate, handled optional.
            handled = any(
                (isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "Exception"))
                or (
                    isinstance(h.type, ast.Tuple)
                    and any(getattr(e, "id", "") in ("ImportError", "Exception")
                            for e in h.type.elts)
                )
                for h in node.handlers
            )
            if handled:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    for alias in inner.names:
                        names.add(alias.name.split(".")[0])
                elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                    names.add(inner.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.name))
def test_no_heavy_module_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = module_level_imports(tree) & FORBIDDEN_AT_MODULE_LEVEL
    assert not offenders, (
        f"{path.relative_to(PACKAGE.parent)} imports {sorted(offenders)} at module level. "
        "Move it inside the function that needs it, wrapped in try/except ImportError, "
        "so JARVIS still starts on a machine where it is absent (or not Windows)."
    )


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.name))
def test_annotated_files_declare_future_annotations(path: Path):
    """`from __future__ import annotations` is what keeps us Python 3.9-compatible."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    has_future = any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__"
        and any(a.name == "annotations" for a in node.names)
        for node in tree.body
    )
    uses_annotations = any(
        isinstance(node, ast.AnnAssign) for node in ast.walk(tree)
    ) or any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.returns or any(a.annotation for a in node.args.args))
        for node in ast.walk(tree)
    )
    if uses_annotations:
        assert has_future, (
            f"{path.name} uses annotations without `from __future__ import annotations`; "
            "modern syntax such as `dict[str, int]` would fail on Python 3.9."
        )


def test_the_whole_package_imports_in_a_clean_subprocess():
    """Import every module in a fresh interpreter with heavy packages blocked.

    Running in a subprocess is what makes this meaningful: inside pytest the
    heavy packages are often already in ``sys.modules`` from another test, so an
    accidental top-level import would succeed and the regression would ship.
    """
    blocked = sorted(HEAVY)
    program = f"""
import sys

BLOCKED = set({blocked!r})


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError("blocked for the import-hygiene test: " + name)
        return None


sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name.split('.')[0] in BLOCKED:
        del sys.modules[name]

import importlib
import pkgutil

import jarvis

failures = []
for info in pkgutil.walk_packages(jarvis.__path__, prefix="jarvis."):
    try:
        importlib.import_module(info.name)
    except Exception as exc:
        failures.append(f"{{info.name}}: {{type(exc).__name__}}: {{exc}}")

if failures:
    print("FAILURES")
    print("\\n".join(failures))
    sys.exit(1)
print("ALL MODULES IMPORTED CLEANLY")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE.parent),
        timeout=180,
    )
    assert result.returncode == 0, (
        "the package does not import with heavy dependencies blocked:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "ALL MODULES IMPORTED CLEANLY" in result.stdout
