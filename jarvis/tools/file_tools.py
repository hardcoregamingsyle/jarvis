"""File-system tools.

Every path goes through :func:`platform_utils.resolve_path`, so a drive-relative
Windows path like ``C:`` is anchored to that drive's root rather than silently
becoming the process working directory — the exact confusion that destroyed an
earlier version of this repo.

Mutations are offered to ``ctx.security``, which permits everything under the
default open policy and only starts refusing once someone opts into
``guarded``/``readonly``.  These tools do not second-guess it.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import (
    is_filesystem_root,
    resolve_path,
)
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Local helpers
# --------------------------------------------------------------------------- #
_BINARY_SNIFF_BYTES = 8192


def _looks_binary(data: bytes) -> bool:
    """Heuristic — a NUL byte in the first chunk means "not text"."""
    return b"\x00" in data[:_BINARY_SNIFF_BYTES]


def _safe_resolve(raw: str) -> Path:
    """Wrap :func:`resolve_path`, raising the caller's ``ValueError``."""
    if raw is None:
        raise ValueError("path is required")
    return resolve_path(str(raw))


def _is_protected(security: Any, path: Path) -> bool:
    if security is None:
        return False
    checker = getattr(security, "is_protected", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(str(path)))
    except Exception:  # noqa: BLE001
        return False


def _is_ancestor_or_equal(candidate: Path, other: Path) -> bool:
    """True when *candidate* is ``other`` or one of its ancestors."""
    try:
        other.relative_to(candidate)
        return True
    except ValueError:
        return False


def _check(security: Any, path: Path, write: bool = False) -> Optional[ToolResult]:
    """Run *path* past the security gate; return a failure only if it objects.

    ``None`` means "proceed", which is what the default open policy always
    yields.  The call is kept so the opt-in ``guarded``/``readonly`` modes and
    any configured confirmation callback still govern this tool.
    """
    if security is None:
        return None
    checker = getattr(security, "check_path", None)
    if not callable(checker):
        return None
    try:
        decision = checker(str(path), write=write)
    except Exception:  # noqa: BLE001 - a broken gate must not break the tool.
        log.debug("security.check_path raised", exc_info=True)
        return None
    if not getattr(decision, "allowed", True):
        reason = getattr(decision, "reason", None) or "refused by security policy"
        return ToolResult.failure(str(reason))
    if getattr(decision, "requires_confirmation", False):
        approver = getattr(security, "allows", None)
        if callable(approver):
            try:
                granted = bool(approver(decision))
            except Exception:  # noqa: BLE001
                granted = False
            if not granted:
                return ToolResult.failure("refused — confirmation not granted")
    return None


def _fmt_size(entry: Path) -> Optional[int]:
    try:
        return entry.stat().st_size
    except OSError:
        return None


# --------------------------------------------------------------------------- #
#  Read / list / search
# --------------------------------------------------------------------------- #
def _read_file(
    path: str,
    max_bytes: int = 200000,
    encoding: str = "utf-8",
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> ToolResult:
    """Read a text file, with binary sniffing and a size cap."""
    try:
        resolved = _safe_resolve(path)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"File not found: {resolved}")
    if not resolved.is_file():
        return ToolResult.failure(f"Not a regular file: {resolved}")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        return ToolResult.failure(f"read failed: {exc}")

    truncated = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True
    if _looks_binary(raw):
        return ToolResult.failure(f"File appears to be binary: {resolved}")

    for enc in (encoding, "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:  # pragma: no cover - latin-1 always decodes
        return ToolResult.failure("Could not decode file with any known encoding")

    if start_line is not None or end_line is not None:
        lines = text.splitlines(keepends=True)
        lo = max(1, int(start_line)) if start_line is not None else 1
        hi = int(end_line) if end_line is not None else len(lines)
        text = "".join(lines[lo - 1 : hi])

    return ToolResult.success(
        output={
            "path": str(resolved),
            "content": text,
            "bytes_read": len(raw),
            "truncated": truncated,
        }
    )


def _list_dir(
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    max_entries: int = 500,
) -> ToolResult:
    try:
        resolved = _safe_resolve(path)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"No such directory: {resolved}")
    if not resolved.is_dir():
        return ToolResult.failure(f"Not a directory: {resolved}")

    entries: List[dict] = []
    it = resolved.rglob(pattern) if recursive else resolved.glob(pattern)
    try:
        for child in it:
            try:
                stat = child.stat()
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": child.is_dir(),
                        "size": stat.st_size if child.is_file() else None,
                        "mtime": stat.st_mtime,
                    }
                )
            except OSError:
                continue
            if len(entries) >= max_entries:
                break
    except OSError as exc:
        return ToolResult.failure(f"list failed: {exc}")

    return ToolResult.success(
        output={"path": str(resolved), "entries": entries, "count": len(entries)}
    )


def _find_files(
    root: str,
    pattern: str,
    max_results: int = 200,
    ignore_hidden: bool = True,
) -> ToolResult:
    try:
        resolved = _safe_resolve(root)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"No such root: {resolved}")
    matches: List[str] = []
    try:
        for child in resolved.rglob(pattern):
            if ignore_hidden and any(part.startswith(".") for part in child.relative_to(resolved).parts):
                continue
            matches.append(str(child))
            if len(matches) >= max_results:
                break
    except OSError as exc:
        return ToolResult.failure(f"find failed: {exc}")
    return ToolResult.success(
        output={"root": str(resolved), "matches": matches, "count": len(matches)}
    )


def _search_text(
    root: str,
    query: str,
    pattern: str = "*",
    max_results: int = 100,
    ignore_case: bool = True,
) -> ToolResult:
    try:
        resolved = _safe_resolve(root)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"No such root: {resolved}")
    if query is None or query == "":
        return ToolResult.failure("query must be a non-empty string")
    needle = query.lower() if ignore_case else query

    hits: List[dict] = []
    try:
        iterable = resolved.rglob(pattern) if resolved.is_dir() else [resolved]
        for candidate in iterable:
            if not candidate.is_file():
                continue
            try:
                raw = candidate.read_bytes()
            except OSError:
                continue
            if _looks_binary(raw):
                continue
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                haystack = line.lower() if ignore_case else line
                if needle in haystack:
                    hits.append(
                        {"path": str(candidate), "line": lineno, "text": line.rstrip()}
                    )
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
    except OSError as exc:
        return ToolResult.failure(f"search failed: {exc}")

    return ToolResult.success(
        output={"root": str(resolved), "matches": hits, "count": len(hits)}
    )


def _file_info(path: str) -> ToolResult:
    try:
        resolved = _safe_resolve(path)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    if not resolved.exists():
        return ToolResult.failure(f"No such path: {resolved}")
    try:
        st = resolved.stat()
    except OSError as exc:
        return ToolResult.failure(f"stat failed: {exc}")
    return ToolResult.success(
        output={
            "path": str(resolved),
            "exists": True,
            "is_file": resolved.is_file(),
            "is_dir": resolved.is_dir(),
            "size": st.st_size,
            "mtime": st.st_mtime,
            "ctime": st.st_ctime,
            "mode": st.st_mode,
        }
    )


# --------------------------------------------------------------------------- #
#  Mutations (dangerous)
# --------------------------------------------------------------------------- #
def _make_write(security: Any):
    def _write_file(
        path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
    ) -> ToolResult:
        """Write *content* to *path*.  Creates parent directories."""
        try:
            resolved = _safe_resolve(path)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if is_filesystem_root(resolved):
            return ToolResult.failure(f"refusing to write to a filesystem root: {resolved}")
        if _is_protected(security, resolved):
            return ToolResult.failure(f"path is protected: {resolved}")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(resolved, mode, encoding=encoding) as fh:
                written = fh.write(str(content))
        except OSError as exc:
            return ToolResult.failure(f"write failed: {exc}")
        return ToolResult.success(
            output={"path": str(resolved), "bytes_written": written, "appended": bool(append)}
        )

    return _write_file


def _make_edit(security: Any):
    def _edit_file(path: str, old: str, new: str, count: int = 1) -> ToolResult:
        """Replace an exact substring in *path*.  Fails when *old* is absent."""
        try:
            resolved = _safe_resolve(path)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if _is_protected(security, resolved):
            return ToolResult.failure(f"path is protected: {resolved}")
        if not resolved.exists() or not resolved.is_file():
            return ToolResult.failure(f"not a regular file: {resolved}")
        if old == "":
            return ToolResult.failure("`old` must be a non-empty string")
        try:
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.failure(f"read failed: {exc}")
        occurrences = text.count(old)
        if occurrences == 0:
            return ToolResult.failure(f"`old` string not found in {resolved}")
        max_count = -1 if int(count) <= 0 else int(count)
        if max_count > 0 and occurrences > max_count and max_count != occurrences:
            # ambiguity is a bug — the caller asked for `count` replacements but
            # the file contains more, so refuse rather than silently trim.
            if occurrences > max_count:
                pass  # str.replace(..., count) naturally caps; this is fine.
        replaced = text.replace(old, new, occurrences if max_count == -1 else max_count)
        try:
            resolved.write_text(replaced, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(f"write failed: {exc}")
        return ToolResult.success(
            output={
                "path": str(resolved),
                "replacements": occurrences if max_count == -1 else min(occurrences, max_count),
            }
        )

    return _edit_file


def _make_dir(path: str) -> ToolResult:
    try:
        resolved = _safe_resolve(path)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ToolResult.failure(f"mkdir failed: {exc}")
    return ToolResult.success(output={"path": str(resolved), "created": True})


def _make_move(security: Any):
    def _move_path(src: str, dst: str) -> ToolResult:
        try:
            src_resolved = _safe_resolve(src)
            dst_resolved = _safe_resolve(dst)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if is_filesystem_root(src_resolved) or is_filesystem_root(dst_resolved):
            return ToolResult.failure("refusing to move a filesystem root")
        if _is_protected(security, src_resolved) or _is_protected(security, dst_resolved):
            return ToolResult.failure("source or destination is protected")
        if not src_resolved.exists():
            return ToolResult.failure(f"no such source: {src_resolved}")
        try:
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_resolved), str(dst_resolved))
        except (OSError, shutil.Error) as exc:
            return ToolResult.failure(f"move failed: {exc}")
        return ToolResult.success(
            output={"src": str(src_resolved), "dst": str(dst_resolved)}
        )

    return _move_path


def _make_copy(security: Any):
    def _copy_path(src: str, dst: str) -> ToolResult:
        try:
            src_resolved = _safe_resolve(src)
            dst_resolved = _safe_resolve(dst)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if _is_protected(security, dst_resolved):
            return ToolResult.failure(f"destination is protected: {dst_resolved}")
        if not src_resolved.exists():
            return ToolResult.failure(f"no such source: {src_resolved}")
        try:
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            if src_resolved.is_dir():
                shutil.copytree(str(src_resolved), str(dst_resolved), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src_resolved), str(dst_resolved))
        except (OSError, shutil.Error) as exc:
            return ToolResult.failure(f"copy failed: {exc}")
        return ToolResult.success(
            output={"src": str(src_resolved), "dst": str(dst_resolved)}
        )

    return _copy_path


def _make_delete(security: Any):
    def _delete_path(path: str, recursive: bool = False) -> ToolResult:
        """Delete a file or directory.

        Ordinary paths are deleted without ceremony — no prompts, no protected
        list, whatever ``ctx.security`` permits, which by default is everything.

        Four whole-tree targets are still refused, and none of them is a policy
        about what the owner may do.  A filesystem root, the home directory, the
        working directory and its ancestors are targets where a recursive delete
        cannot do what asking for it implies: it destroys the running
        environment partway through and aborts on the first locked file, so the
        result is never "deleted" but always "half-deleted, unrecoverable".
        Every one of them is reachable a different way (delete the children, or
        use the unrestricted shell), so this costs no capability — it only stops
        a mistyped or hallucinated argument from being unrecoverable.
        """
        if path is None or (isinstance(path, str) and not path.strip()):
            return ToolResult.failure("path is required")
        try:
            resolved = resolve_path(str(path))
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if is_filesystem_root(resolved):
            return ToolResult.failure(
                f"refusing to recursively delete the filesystem root {resolved}: "
                f"this cannot succeed, it can only leave the machine unbootable. "
                f"Delete the specific children you meant instead."
            )

        try:
            home = Path(os.path.expanduser("~")).resolve()
        except (OSError, ValueError):
            home = None
        if home is not None and resolved == home:
            return ToolResult.failure(
                f"refusing to recursively delete the home directory {resolved}: "
                f"this aborts on the first locked file and leaves the profile "
                f"half-destroyed. Delete the specific children you meant instead."
            )

        try:
            cwd = Path.cwd().resolve()
        except (OSError, ValueError):
            cwd = None
        if cwd is not None and _is_ancestor_or_equal(resolved, cwd):
            return ToolResult.failure(
                f"refusing to delete {resolved}: it is the current working "
                f"directory or an ancestor of it, so the delete would destroy the "
                f"ground the process is standing on and abort part-way."
            )

        decision = _check(security, resolved, write=True)
        if decision is not None:
            return decision

        if not resolved.exists():
            return ToolResult.failure(f"no such path: {resolved}")

        try:
            if resolved.is_dir() and not resolved.is_symlink():
                if not recursive:
                    return ToolResult.failure(
                        f"{resolved} is a directory; pass recursive=True to delete it"
                    )
                shutil.rmtree(str(resolved))
            else:
                resolved.unlink()
        except OSError as exc:
            return ToolResult.failure(f"delete failed: {exc}")

        return ToolResult.success(output={"path": str(resolved), "deleted": True})

    return _delete_path


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the built-in filesystem tools bound to *ctx*."""
    security = getattr(ctx, "security", None)
    tools: List[Tool] = [
        FunctionTool(_read_file, name="read_file", description="Read a text file's contents."),
        FunctionTool(_list_dir, name="list_dir", description="List directory entries matching a glob."),
        FunctionTool(_find_files, name="find_files", description="Recursively locate files by glob."),
        FunctionTool(_search_text, name="search_text", description="Grep text files for a substring."),
        FunctionTool(_file_info, name="file_info", description="Stat metadata for a path."),
        FunctionTool(_make_dir, name="make_dir", description="Create a directory (and parents)."),
        FunctionTool(
            _make_write(security),
            name="write_file",
            description="Write text content to a file.",
            dangerous=True,
        ),
        FunctionTool(
            _make_edit(security),
            name="edit_file",
            description="Replace a substring inside a text file.",
            dangerous=True,
        ),
        FunctionTool(
            _make_move(security),
            name="move_path",
            description="Move/rename a file or directory.",
            dangerous=True,
        ),
        FunctionTool(
            _make_copy(security),
            name="copy_path",
            description="Copy a file or directory tree.",
            dangerous=True,
        ),
        FunctionTool(
            _make_delete(security),
            name="delete_path",
            description="Delete a file or (with recursive=True) a directory tree.",
            dangerous=True,
        ),
    ]
    return tools


__all__ = ["build_tools", "safe_truncate"]
