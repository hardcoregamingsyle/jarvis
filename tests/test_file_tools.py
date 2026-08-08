"""File-tool round-trips, edge cases, and the delete_path safety rules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from jarvis.core.contracts import ToolResult
from jarvis.tools import file_tools
from jarvis.tools.file_tools import build_tools
from jarvis.tools.registry import ToolContext, ToolRegistry


# --------------------------------------------------------------------------- #
#  Fake security — no protected paths, no confirmation friction.
# --------------------------------------------------------------------------- #
class OpenSecurity:
    def __init__(self, protected: Optional[list] = None) -> None:
        self._protected = [str(Path(p).resolve()) for p in (protected or [])]

    def is_protected(self, path: str) -> bool:
        target = str(Path(path).resolve())
        return any(target == p or target.startswith(p + os.sep) for p in self._protected)

    def check_tool(self, spec, cleaned):
        class D:
            allowed = True
            requires_confirmation = False

        return D()

    def allows(self, decision) -> bool:
        return True


class FakeConfig:
    def __init__(self, dir_path):
        self._dir = dir_path

    def tools_dir(self):
        return self._dir


def _mk_registry(tmp_path, protected: Optional[list] = None):
    ctx = ToolContext(
        config=FakeConfig(tmp_path / "gen"),
        security=OpenSecurity(protected=protected),
    )
    reg = ToolRegistry(ctx)
    for t in build_tools(ctx):
        reg.register(t, replace=True)
    return reg


# --------------------------------------------------------------------------- #
#  Round trips
# --------------------------------------------------------------------------- #
def test_write_and_read_round_trip(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "hello.txt"

    w = reg.run("write_file", path=str(target), content="hello world")
    assert w.ok is True
    assert target.read_text(encoding="utf-8") == "hello world"

    r = reg.run("read_file", path=str(target))
    assert r.ok is True
    assert r.output["content"] == "hello world"
    assert r.output["bytes_read"] == len("hello world")
    assert r.output["truncated"] is False


def test_write_append_mode(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "log.txt"
    reg.run("write_file", path=str(target), content="one\n")
    reg.run("write_file", path=str(target), content="two\n", append=True)
    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


def test_edit_file_replaces_substring(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "e.txt"
    target.write_text("foo bar foo", encoding="utf-8")
    r = reg.run("edit_file", path=str(target), old="foo", new="baz", count=1)
    assert r.ok is True
    assert target.read_text(encoding="utf-8") == "baz bar foo"


def test_edit_file_missing_old_fails(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "e.txt"
    target.write_text("hello", encoding="utf-8")
    r = reg.run("edit_file", path=str(target), old="world", new="there")
    assert r.ok is False
    assert "not found" in r.error


def test_edit_file_replace_all_with_count_zero(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "e.txt"
    target.write_text("a a a a", encoding="utf-8")
    r = reg.run("edit_file", path=str(target), old="a", new="b", count=0)
    assert r.ok is True
    assert target.read_text(encoding="utf-8") == "b b b b"


# --------------------------------------------------------------------------- #
#  Listing, finding, searching
# --------------------------------------------------------------------------- #
def test_list_dir_returns_entries(tmp_path):
    reg = _mk_registry(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    r = reg.run("list_dir", path=str(tmp_path), pattern="*.txt")
    assert r.ok is True
    names = [e["name"] for e in r.output["entries"]]
    assert names == ["a.txt"]
    assert r.output["count"] == 1


def test_find_files_recursive(tmp_path):
    reg = _mk_registry(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "needle.py").write_text("x", encoding="utf-8")
    (tmp_path / "other.py").write_text("y", encoding="utf-8")
    r = reg.run("find_files", root=str(tmp_path), pattern="*.py")
    assert r.ok is True
    matches = set(Path(m).name for m in r.output["matches"])
    assert "needle.py" in matches
    assert "other.py" in matches


def test_find_files_ignore_hidden(tmp_path):
    reg = _mk_registry(tmp_path)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("x", encoding="utf-8")
    (tmp_path / "visible.py").write_text("y", encoding="utf-8")
    r = reg.run("find_files", root=str(tmp_path), pattern="*.py", ignore_hidden=True)
    assert r.ok is True
    matches = [Path(m).name for m in r.output["matches"]]
    assert "visible.py" in matches
    assert "secret.py" not in matches


def test_search_text_finds_matches(tmp_path):
    reg = _mk_registry(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\nbeta\nGAMMA\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("no dice", encoding="utf-8")
    r = reg.run("search_text", root=str(tmp_path), query="gamma", ignore_case=True)
    assert r.ok is True
    assert r.output["count"] == 1
    hit = r.output["matches"][0]
    assert hit["line"] == 3
    assert "GAMMA" in hit["text"]


def test_search_text_skips_binaries(tmp_path):
    reg = _mk_registry(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"needle\x00\x00needle")
    r = reg.run("search_text", root=str(tmp_path), query="needle")
    assert r.ok is True
    assert r.output["count"] == 0


# --------------------------------------------------------------------------- #
#  Info / mkdir / move / copy
# --------------------------------------------------------------------------- #
def test_file_info_reports_metadata(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("abcdef", encoding="utf-8")
    r = reg.run("file_info", path=str(target))
    assert r.ok is True
    assert r.output["is_file"] is True
    assert r.output["size"] == 6


def test_file_info_missing_path_fails(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("file_info", path=str(tmp_path / "nope"))
    assert r.ok is False


def test_make_dir_creates_nested(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "a" / "b" / "c"
    r = reg.run("make_dir", path=str(target))
    assert r.ok is True
    assert target.is_dir()


def test_move_and_copy(tmp_path):
    reg = _mk_registry(tmp_path)
    src = tmp_path / "src.txt"
    src.write_text("moving", encoding="utf-8")

    dst = tmp_path / "dst.txt"
    r = reg.run("copy_path", src=str(src), dst=str(dst))
    assert r.ok is True
    assert dst.read_text(encoding="utf-8") == "moving"

    moved = tmp_path / "final.txt"
    r = reg.run("move_path", src=str(dst), dst=str(moved))
    assert r.ok is True
    assert moved.exists() and not dst.exists()


# --------------------------------------------------------------------------- #
#  Delete round-trip and safety
# --------------------------------------------------------------------------- #
def test_delete_file(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "gone.txt"
    target.write_text("x", encoding="utf-8")
    r = reg.run("delete_path", path=str(target))
    assert r.ok is True
    assert not target.exists()


def test_delete_dir_requires_recursive(tmp_path):
    reg = _mk_registry(tmp_path)
    d = tmp_path / "d"
    d.mkdir()
    (d / "f").write_text("x", encoding="utf-8")

    r = reg.run("delete_path", path=str(d))
    assert r.ok is False
    assert d.exists()

    r = reg.run("delete_path", path=str(d), recursive=True)
    assert r.ok is True
    assert not d.exists()


# --------------------------------------------------------------------------- #
#  Whole-tree refusals
#
#  These tests NEVER name a real root, home directory or working directory as a
#  delete target. They redirect the guard's *notion* of those places into
#  tmp_path instead.
#
#  This is not fastidiousness. The earlier version of this file called
#  delete_path() on os.path.expanduser("~") with recursive=True and asserted the
#  call was refused — a test whose safety depended entirely on the feature it was
#  testing. The moment that guard was relaxed the test stopped being a test and
#  became a live shutil.rmtree of the real user profile. It emptied six
#  directories under C:\Users\Hp before aborting on a locked file, and none of it
#  was recoverable.
#
#  Written the way they are below, a regressed guard deletes a temporary
#  directory and the assertion FAILS. Written the old way, it deletes your home
#  directory and the assertion PASSES. Keep them pointed at fakes.
# --------------------------------------------------------------------------- #
def test_delete_refuses_drive_relative(tmp_path, monkeypatch):
    """"C:" must never be treated as "the current directory on drive C"."""
    fake_root = tmp_path / "pretend_root"
    fake_root.mkdir()
    canary = fake_root / "keep.txt"
    canary.write_text("safe", encoding="utf-8")

    # resolve_path maps "C:" to the drive root; make the guard treat our
    # stand-in as that root so nothing real is ever the target.
    monkeypatch.setattr(file_tools, "resolve_path", lambda raw: fake_root)
    monkeypatch.setattr(file_tools, "is_filesystem_root", lambda p: Path(p) == fake_root)

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path="C:", recursive=True)

    assert r.ok is False
    assert "root" in r.error.lower()
    assert canary.exists(), "the guard let a root-shaped delete through"


def test_delete_refuses_filesystem_root(tmp_path, monkeypatch):
    fake_root = tmp_path / "pretend_root"
    fake_root.mkdir()
    canary = fake_root / "keep.txt"
    canary.write_text("safe", encoding="utf-8")

    monkeypatch.setattr(file_tools, "is_filesystem_root", lambda p: Path(p) == fake_root)

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path=str(fake_root), recursive=True)

    assert r.ok is False
    assert "root" in r.error.lower()
    assert canary.exists()


def test_protected_paths_no_longer_block_deletes_by_default(tmp_path):
    """Path protection is a rail, and the shipped config has it switched off.

    ``protected_paths`` is empty by default, so nothing is protected. The
    security gate is still consulted, which is what keeps the opt-in modes
    working — see tests/test_security_policy.py for the guarded-mode case.
    """
    victim = tmp_path / "keepme"
    victim.mkdir()
    (victim / "important.txt").write_text("x", encoding="utf-8")

    reg = _mk_registry(tmp_path)          # default config: nothing protected
    r = reg.run("delete_path", path=str(victim), recursive=True)

    assert r.ok, r.error
    assert not victim.exists()


def test_delete_refuses_cwd(tmp_path, monkeypatch):
    reg = _mk_registry(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    marker = workdir / "marker.txt"
    marker.write_text("x", encoding="utf-8")
    r = reg.run("delete_path", path=str(workdir), recursive=True)
    assert r.ok is False
    assert "working directory" in r.error.lower() or "ancestor" in r.error.lower()
    assert workdir.exists()
    assert marker.exists()


def test_delete_refuses_cwd_ancestor(tmp_path, monkeypatch):
    reg = _mk_registry(tmp_path)
    workdir = tmp_path / "outer" / "inner"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    marker = tmp_path / "outer" / "marker.txt"
    marker.write_text("x", encoding="utf-8")
    r = reg.run("delete_path", path=str(tmp_path / "outer"), recursive=True)
    assert r.ok is False
    assert marker.exists()


def test_delete_refuses_home(tmp_path, monkeypatch):
    """The guard's idea of "home" is redirected — the real profile is never named.

    See the note above: the previous version of this test pointed a recursive
    delete at the actual user profile and destroyed part of it.
    """
    fake_home = tmp_path / "pretend_home"
    fake_home.mkdir()
    canary = fake_home / "documents.txt"
    canary.write_text("irreplaceable", encoding="utf-8")

    monkeypatch.setattr(
        file_tools.os.path, "expanduser", lambda p: str(fake_home) if p == "~" else p
    )

    reg = _mk_registry(tmp_path)
    r = reg.run("delete_path", path=str(fake_home), recursive=True)

    assert r.ok is False
    assert "home" in r.error.lower()
    assert canary.exists(), "the guard let the home directory be deleted"
    assert canary.read_text(encoding="utf-8") == "irreplaceable"


# --------------------------------------------------------------------------- #
#  Unicode & max_bytes truncation
# --------------------------------------------------------------------------- #
def test_unicode_names_and_content(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "üñíçødé.txt"
    payload = "こんにちは 世界 🌐"
    w = reg.run("write_file", path=str(target), content=payload)
    assert w.ok is True
    r = reg.run("read_file", path=str(target))
    assert r.ok is True
    assert r.output["content"] == payload


def test_read_max_bytes_truncation(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "big.txt"
    target.write_text("x" * 5000, encoding="utf-8")
    r = reg.run("read_file", path=str(target), max_bytes=100)
    assert r.ok is True
    assert r.output["truncated"] is True
    assert len(r.output["content"]) == 100


def test_read_binary_file_refused(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "bin.dat"
    target.write_bytes(b"\x00\x01\x02\x03")
    r = reg.run("read_file", path=str(target))
    assert r.ok is False
    assert "binary" in r.error.lower()


def test_read_missing_file(tmp_path):
    reg = _mk_registry(tmp_path)
    r = reg.run("read_file", path=str(tmp_path / "no.txt"))
    assert r.ok is False


def test_read_line_range(tmp_path):
    reg = _mk_registry(tmp_path)
    target = tmp_path / "lines.txt"
    target.write_bytes(b"a\nb\nc\nd\ne\n")
    r = reg.run("read_file", path=str(target), start_line=2, end_line=4)
    assert r.ok is True
    assert r.output["content"] == "b\nc\nd\n"
