"""Tests for :mod:`jarvis.tools.process_tools`.

These tests never kill a real process and never mutate system state — the
parser is unit-tested directly, and functions that would signal a process are
exercised only for their rejection paths.
"""

from __future__ import annotations

import os

import pytest

from jarvis.tools.process_tools import (
    _kill_process,
    _list_processes,
    _process_info,
    parse_ps_output,
    parse_tasklist_csv,
)


# --------------------------------------------------------------------------- #
#  CSV / ps parser fixtures
# --------------------------------------------------------------------------- #
_TASKLIST_SAMPLE = (
    '"System Idle Process","0","Services","0","24 K"\n'
    '"System","4","Services","0","144 K"\n'
    '"Google Chrome, Inc.exe","1234","Console","1","512,000 K"\n'
    '"my program with spaces.exe","5678","Console","1","1,024 K"\n'
    '"malformed line, only three, fields"\n'
)

_PS_SAMPLE = (
    "    1 systemd           0.1  0.5 root     Ss\n"
    "  200 my program        1.2  3.4 alice    S\n"
    " 3210 spaced comm name  0.0  0.1 bob      R+\n"
    "  bad-line-that-should-be-skipped\n"
    "  400 nginx             0.0  2.0 www-data S\n"
)


class TestParsers:

    def test_tasklist_parses_names_with_commas(self):
        rows = parse_tasklist_csv(_TASKLIST_SAMPLE)
        names = [r["name"] for r in rows]
        assert "Google Chrome, Inc.exe" in names
        assert "my program with spaces.exe" in names

    def test_tasklist_parses_memory_bytes(self):
        rows = parse_tasklist_csv(_TASKLIST_SAMPLE)
        by_name = {r["name"]: r for r in rows}
        # "512,000 K" -> 512000 * 1024
        assert by_name["Google Chrome, Inc.exe"]["memory"] == 512000 * 1024
        assert by_name["my program with spaces.exe"]["memory"] == 1024 * 1024

    def test_tasklist_skips_malformed_rows(self):
        rows = parse_tasklist_csv(_TASKLIST_SAMPLE)
        # Malformed 3-field row must not appear.
        assert all(len(r) > 0 for r in rows)
        assert len(rows) == 4

    def test_tasklist_pids_are_ints(self):
        rows = parse_tasklist_csv(_TASKLIST_SAMPLE)
        for row in rows:
            assert isinstance(row["pid"], int)
        assert {r["pid"] for r in rows} == {0, 4, 1234, 5678}

    def test_tasklist_empty_returns_empty(self):
        assert parse_tasklist_csv("") == []

    def test_ps_parses_multi_word_command(self):
        rows = parse_ps_output(_PS_SAMPLE)
        by_pid = {r["pid"]: r for r in rows}
        assert by_pid[200]["name"] == "my program"
        assert by_pid[3210]["name"] == "spaced comm name"
        assert by_pid[200]["cpu_percent"] == pytest.approx(1.2)
        assert by_pid[200]["memory_percent"] == pytest.approx(3.4)
        assert by_pid[200]["user"] == "alice"
        assert by_pid[200]["status"] == "S"

    def test_ps_skips_malformed(self):
        rows = parse_ps_output(_PS_SAMPLE)
        pids = {r["pid"] for r in rows}
        assert pids == {1, 200, 3210, 400}

    def test_ps_handles_empty(self):
        assert parse_ps_output("") == []
        assert parse_ps_output("\n\n") == []


# --------------------------------------------------------------------------- #
#  list_processes
# --------------------------------------------------------------------------- #
class TestListProcesses:

    def test_lists_current_pid(self):
        result = _list_processes(filter=None, sort_by="memory", limit=5000)
        assert result.ok is True
        pids = {p.get("pid") for p in result.output["processes"]}
        assert os.getpid() in pids, (
            f"current pid {os.getpid()} must appear in the process list; "
            f"got {len(pids)} pids"
        )

    def test_filter_narrows_results(self):
        # Filter for our own pid — must return exactly that process (may also
        # return others whose name matches the digits, but our pid MUST be in).
        result = _list_processes(filter=str(os.getpid()), sort_by="memory", limit=100)
        assert result.ok is True
        pids = {p.get("pid") for p in result.output["processes"]}
        assert os.getpid() in pids

    def test_limit_is_respected(self):
        result = _list_processes(filter=None, sort_by="memory", limit=3)
        assert result.ok is True
        assert len(result.output["processes"]) <= 3


# --------------------------------------------------------------------------- #
#  kill_process — refusal paths only, never a real kill
# --------------------------------------------------------------------------- #
class TestKillProcessRefusals:

    @pytest.mark.parametrize("pid", [0, 1, 4])
    def test_refuses_reserved_pids(self, pid):
        result = _kill_process(pid, force=False)
        assert result.ok is False
        assert "protected" in (result.error or "").lower()

    def test_refuses_own_pid(self):
        result = _kill_process(os.getpid(), force=True)
        assert result.ok is False
        assert "protected" in (result.error or "").lower()

    def test_refuses_missing_target(self):
        result = _kill_process("", force=False)
        assert result.ok is False

    def test_refuses_impossible_name(self):
        result = _kill_process(
            "definitely-not-a-real-process-abcxyz-1234567890", force=False
        )
        assert result.ok is False
        assert "matched" in (result.error or "") or "no" in (result.error or "").lower()


# --------------------------------------------------------------------------- #
#  process_info edge cases
# --------------------------------------------------------------------------- #
class TestProcessInfo:

    def test_bad_pid_type(self):
        result = _process_info("not-a-number")
        assert result.ok is False

    def test_negative_pid(self):
        result = _process_info(-1)
        assert result.ok is False

    def test_impossible_pid_fails_cleanly(self):
        # 2**30 is astronomically unlikely to exist.
        result = _process_info(2**30)
        assert result.ok is False
        assert result.error, "impossible pid must include an error message"


# --------------------------------------------------------------------------- #
#  build_tools smoke test
# --------------------------------------------------------------------------- #
class TestBuildTools:

    def test_registers_expected_tools(self):
        from jarvis.tools.process_tools import build_tools

        class _Ctx:
            security = None
            config = None
            bus = None

        produced = build_tools(_Ctx())
        names = {t.name for t in produced}
        for expected in (
            "list_processes",
            "find_process",
            "process_info",
            "kill_process",
            "start_process",
            "list_services",
            "service_action",
            "cpu_memory_snapshot",
            "top_consumers",
            "process_tree",
        ):
            assert expected in names, f"missing tool: {expected}"

    def test_kill_and_service_are_dangerous(self):
        from jarvis.tools.process_tools import build_tools

        class _Ctx:
            security = None
            config = None
            bus = None

        tools = {t.name: t for t in build_tools(_Ctx())}
        assert tools["kill_process"].spec.dangerous is True
        assert tools["start_process"].spec.dangerous is True
        assert tools["service_action"].spec.dangerous is True
        assert tools["list_processes"].spec.dangerous is False
