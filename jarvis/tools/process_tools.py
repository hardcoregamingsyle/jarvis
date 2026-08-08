"""Process- and service-management tools.

Uses :mod:`psutil` when it is installed for rich, cross-platform data.  When it
is not, falls back to the platform's built-in CLI (``tasklist`` on Windows,
``ps`` on Linux) and a parser that is pure, testable and quote-aware.  Nothing
here imports a heavy dependency at module scope; everything is deferred into
the function that needs it.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import (
    IS_WINDOWS,
    run_command,
    which,
)
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
_PROCESS_TIMEOUT = 15.0
_SERVICE_TIMEOUT = 20.0
_SERVICE_STOP_TIMEOUT = 30.0
_KILL_GRACE = 3.0

# Windows / Linux kernel/init PIDs that must never be signalled.
_PROTECTED_PIDS = frozenset({0, 1, 4})


# --------------------------------------------------------------------------- #
#  Pure parsers — unit-tested in isolation
# --------------------------------------------------------------------------- #
def parse_tasklist_csv(text: str) -> List[Dict[str, Any]]:
    """Parse the output of ``tasklist /FO CSV /NH`` into a list of dicts.

    Windows process names may contain spaces and commas (e.g. ``"Google
    Chrome, Inc."``) so the parser must be quote-aware — ``csv`` handles that.
    Memory values look like ``"12,345 K"`` which is parsed into bytes.
    """
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows
    reader = csv.reader(io.StringIO(text))
    for fields in reader:
        if len(fields) < 5:
            continue
        name = fields[0].strip()
        pid_raw = fields[1].strip()
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        session = fields[2].strip() if len(fields) > 2 else ""
        try:
            session_id = int(fields[3].strip()) if len(fields) > 3 else None
        except (TypeError, ValueError):
            session_id = None
        mem_raw = fields[4].strip() if len(fields) > 4 else ""
        mem_bytes = _parse_windows_mem(mem_raw)
        rows.append(
            {
                "name": name,
                "pid": pid,
                "session": session,
                "session_id": session_id,
                "memory": mem_bytes,
                "memory_raw": mem_raw,
                "cpu_percent": None,
                "user": None,
                "status": None,
            }
        )
    return rows


def _parse_windows_mem(raw: str) -> Optional[int]:
    """Turn ``"12,345 K"`` / ``"1 234 K"`` into bytes."""
    if not raw:
        return None
    text = raw.replace(",", "").replace("\xa0", " ").strip()
    parts = text.split()
    if not parts:
        return None
    try:
        value = int(parts[0])
    except (TypeError, ValueError):
        return None
    unit = parts[1].upper() if len(parts) > 1 else "K"
    multiplier = {"B": 1, "K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}.get(
        unit[:1], 1024
    )
    return value * multiplier


def parse_ps_output(text: str) -> List[Dict[str, Any]]:
    """Parse ``ps -eo pid,comm,pcpu,pmem,user,stat --no-headers`` output.

    ``comm`` can contain spaces on some systems, but ps typically truncates it
    to the executable's basename.  We split on runs of whitespace and treat the
    LAST 4 columns as the fixed-width fields (cpu, mem, user, stat), joining
    everything in between as the command name.  That handles the odd case
    where a comm has a space without misclassifying it.
    """
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        pid_raw = parts[0]
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        stat = parts[-1]
        user = parts[-2]
        mem_raw = parts[-3]
        cpu_raw = parts[-4]
        try:
            cpu = float(cpu_raw)
        except (TypeError, ValueError):
            cpu = None
        try:
            mem_pct = float(mem_raw)
        except (TypeError, ValueError):
            mem_pct = None
        name = " ".join(parts[1:-4]).strip() or parts[1]
        rows.append(
            {
                "pid": pid,
                "name": name,
                "cpu_percent": cpu,
                "memory_percent": mem_pct,
                "memory": None,
                "user": user,
                "status": stat,
            }
        )
    return rows


def _sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    key = (sort_by or "memory").lower()
    if key in ("mem", "memory"):
        return sorted(
            rows,
            key=lambda r: (r.get("memory") or r.get("memory_percent") or 0),
            reverse=True,
        )
    if key in ("cpu", "cpu_percent"):
        return sorted(rows, key=lambda r: (r.get("cpu_percent") or 0), reverse=True)
    if key in ("pid",):
        return sorted(rows, key=lambda r: r.get("pid") or 0)
    if key in ("name",):
        return sorted(rows, key=lambda r: (r.get("name") or "").lower())
    return rows


# --------------------------------------------------------------------------- #
#  Enumeration
# --------------------------------------------------------------------------- #
def _list_via_psutil(filter_term: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Return psutil-enriched rows, or ``None`` when psutil is unavailable."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    rows: List[Dict[str, Any]] = []
    needle = (filter_term or "").lower()
    for proc in psutil.process_iter(attrs=("pid", "name", "username", "status")):
        info = proc.info or {}
        name = str(info.get("name") or "")
        if needle and needle not in name.lower() and needle not in str(info.get("pid") or ""):
            continue
        try:
            cpu = proc.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            cpu = None
        try:
            mem = proc.memory_info().rss
        except Exception:  # noqa: BLE001
            mem = None
        rows.append(
            {
                "pid": int(info.get("pid") or 0),
                "name": name,
                "cpu_percent": cpu,
                "memory": mem,
                "memory_percent": None,
                "user": info.get("username"),
                "status": info.get("status"),
            }
        )
    return rows


def _list_via_cli(filter_term: Optional[str]) -> List[Dict[str, Any]]:
    """Enumerate processes with the platform's built-in CLI tool."""
    if IS_WINDOWS:
        result = run_command(["tasklist", "/FO", "CSV", "/NH"], timeout=_PROCESS_TIMEOUT)
        if not result.ok:
            return []
        rows = parse_tasklist_csv(result.stdout)
    else:
        argv = ["ps", "-eo", "pid,comm,pcpu,pmem,user,stat", "--no-headers"]
        result = run_command(argv, timeout=_PROCESS_TIMEOUT)
        if not result.ok:
            argv = ["ps", "-eo", "pid,comm,pcpu,pmem,user,stat"]
            result = run_command(argv, timeout=_PROCESS_TIMEOUT)
            if not result.ok:
                return []
            lines = result.stdout.splitlines()
            if lines and lines[0].lstrip().lower().startswith("pid"):
                result_text = "\n".join(lines[1:])
            else:
                result_text = result.stdout
            rows = parse_ps_output(result_text)
        else:
            rows = parse_ps_output(result.stdout)
    if filter_term:
        needle = filter_term.lower()
        rows = [
            r
            for r in rows
            if needle in (r.get("name") or "").lower()
            or needle == str(r.get("pid"))
        ]
    return rows


def _list_processes(
    filter: Optional[str] = None,
    sort_by: str = "memory",
    limit: int = 30,
) -> ToolResult:
    """List processes, filtered and sorted."""
    rows = _list_via_psutil(filter)
    if rows is None:
        rows = _list_via_cli(filter)
    rows = _sort_rows(rows, sort_by)
    try:
        cap = int(limit)
    except (TypeError, ValueError):
        cap = 30
    if cap > 0:
        rows = rows[:cap]
    return ToolResult.success(
        output={"processes": rows, "count": len(rows), "sort_by": sort_by}
    )


def _find_process(name: str) -> ToolResult:
    """Return every process whose name contains ``name`` (case-insensitive)."""
    if not name or not str(name).strip():
        return ToolResult.failure("name is required")
    return _list_processes(filter=str(name), sort_by="memory", limit=200)


def _process_info(pid: Union[int, str]) -> ToolResult:
    """Detailed info about a single process."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return ToolResult.failure(f"pid must be an integer, got {pid!r}")
    if pid_int < 0:
        return ToolResult.failure("pid must be non-negative")
    try:
        import psutil  # type: ignore
    except ImportError:
        rows = _list_via_cli(None)
        for row in rows:
            if row.get("pid") == pid_int:
                return ToolResult.success(output=row)
        return ToolResult.failure(f"no process with pid {pid_int}")
    try:
        proc = psutil.Process(pid_int)
    except Exception as exc:  # noqa: BLE001 - psutil.NoSuchProcess / AccessDenied
        return ToolResult.failure(f"process_info({pid_int}) failed: {exc}")
    try:
        with proc.oneshot():
            data = {
                "pid": proc.pid,
                "name": proc.name(),
                "status": proc.status(),
                "cpu_percent": proc.cpu_percent(interval=None),
                "memory": proc.memory_info().rss,
                "user": proc.username() if hasattr(proc, "username") else None,
                "create_time": proc.create_time(),
                "num_threads": proc.num_threads(),
                "ppid": proc.ppid(),
                "cmdline": list(proc.cmdline() or ()),
                "exe": proc.exe() if hasattr(proc, "exe") else None,
            }
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"process_info({pid_int}) failed: {exc}")
    return ToolResult.success(output=data)


# --------------------------------------------------------------------------- #
#  Kill / start
# --------------------------------------------------------------------------- #
def _self_and_parent_pids() -> List[int]:
    pids = [os.getpid()]
    try:
        ppid = os.getppid()
        if ppid and ppid not in pids:
            pids.append(ppid)
    except OSError:
        pass
    return pids


def _resolve_pids(target: Union[int, str]) -> Tuple[List[int], Optional[str]]:
    """Turn a name or pid into a list of pids."""
    try:
        pid = int(target)
        return [pid], None
    except (TypeError, ValueError):
        pass
    name = str(target).strip()
    if not name:
        return [], "empty target"
    try:
        import psutil  # type: ignore
    except ImportError:
        rows = _list_via_cli(name)
        return [r["pid"] for r in rows if r.get("pid")], None
    pids: List[int] = []
    for proc in psutil.process_iter(attrs=("pid", "name")):
        pname = str((proc.info or {}).get("name") or "")
        if name.lower() in pname.lower():
            pids.append(int(proc.info.get("pid")))
    return pids, None


def _kill_via_taskkill(pid: int, force: bool) -> Tuple[bool, str]:
    argv = ["taskkill", "/PID", str(pid)]
    if force:
        argv.append("/F")
    result = run_command(argv, timeout=_PROCESS_TIMEOUT)
    if result.ok:
        return True, result.stdout.strip() or "ok"
    return False, safe_truncate((result.stderr or result.stdout).strip(), 300)


def _kill_via_signal(pid: int, force: bool) -> Tuple[bool, str]:
    if IS_WINDOWS:
        return _kill_via_taskkill(pid, force)
    import signal as _signal
    try:
        os.kill(pid, _signal.SIGKILL if force else _signal.SIGTERM)
    except ProcessLookupError:
        return False, f"no such process {pid}"
    except PermissionError as exc:
        return False, f"permission denied: {exc}"
    except OSError as exc:
        return False, f"os.kill failed: {exc}"
    if force:
        return True, f"SIGKILL sent to {pid}"
    deadline = time.monotonic() + _KILL_GRACE
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True, f"SIGTERM accepted by {pid}"
        except OSError:
            break
        time.sleep(0.1)
    try:
        os.kill(pid, _signal.SIGKILL)
        return True, f"SIGKILL after SIGTERM ({pid})"
    except ProcessLookupError:
        return True, f"exited after SIGTERM ({pid})"
    except OSError as exc:
        return False, f"SIGKILL failed: {exc}"


def _kill_process(pid_or_name: Union[int, str], force: bool = False) -> ToolResult:
    """Terminate a process by pid or name."""
    if pid_or_name is None or (isinstance(pid_or_name, str) and not pid_or_name.strip()):
        return ToolResult.failure("pid_or_name is required")
    pids, err = _resolve_pids(pid_or_name)
    if err:
        return ToolResult.failure(err)
    if not pids:
        return ToolResult.failure(f"no process matched {pid_or_name!r}")

    reserved = set(_PROTECTED_PIDS) | set(_self_and_parent_pids())
    protected = [p for p in pids if p in reserved]
    if protected:
        return ToolResult.failure(
            f"refusing to kill protected pid(s): {sorted(protected)}"
        )

    killed: List[Dict[str, Any]] = []
    for pid in pids:
        ok, detail = _kill_via_signal(pid, bool(force))
        killed.append({"pid": pid, "ok": ok, "detail": detail})
    all_ok = all(entry["ok"] for entry in killed)
    output = {"killed": killed, "count": len(killed), "force": bool(force)}
    if not all_ok:
        return ToolResult(False, output=output, error="one or more kills failed")
    return ToolResult.success(output=output)


def _start_process(
    command: str,
    cwd: Optional[str] = None,
    detached: bool = True,
) -> ToolResult:
    """Launch a program in the background (or attached)."""
    if not command or not str(command).strip():
        return ToolResult.failure("command is required")
    text = str(command)
    if IS_WINDOWS:
        argv = text if isinstance(text, str) else list(text)
    else:
        try:
            argv = shlex.split(text)
        except ValueError as exc:
            return ToolResult.failure(f"could not parse command: {exc}")
        if not argv:
            return ToolResult.failure("command parsed to nothing")

    kwargs: Dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if IS_WINDOWS:
        kwargs["shell"] = True
        if detached:
            flags = 0
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            kwargs["creationflags"] = flags
    else:
        if detached:
            kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except (OSError, ValueError) as exc:
        return ToolResult.failure(f"start_process failed: {exc}")
    return ToolResult.success(
        output={
            "pid": proc.pid,
            "command": text,
            "detached": bool(detached),
            "cwd": str(cwd) if cwd else None,
        }
    )


# --------------------------------------------------------------------------- #
#  Services
# --------------------------------------------------------------------------- #
def _list_services() -> ToolResult:
    """Enumerate services (Windows SC / Linux systemctl)."""
    if IS_WINDOWS:
        result = run_command(["sc", "query", "state=", "all"], timeout=_SERVICE_TIMEOUT)
        if not result.ok:
            return ToolResult.failure(
                f"sc query failed: {safe_truncate(result.stderr, 300)}"
            )
        services: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for raw in result.stdout.splitlines():
            line = raw.strip()
            if not line:
                if current:
                    services.append(current)
                    current = {}
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "service_name":
                if current:
                    services.append(current)
                current = {"name": value}
            elif key == "display_name":
                current["display_name"] = value
            elif key == "state":
                parts = value.split()
                current["state"] = parts[-1] if parts else value
        if current:
            services.append(current)
        return ToolResult.success(output={"services": services, "count": len(services)})

    systemctl = which("systemctl")
    if not systemctl:
        return ToolResult.failure("systemctl is not available on this system")
    result = run_command(
        [systemctl, "list-units", "--type=service", "--all", "--no-pager", "--no-legend"],
        timeout=_SERVICE_TIMEOUT,
    )
    if not result.ok:
        return ToolResult.failure(
            f"systemctl failed: {safe_truncate(result.stderr, 300)}"
        )
    services = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        services.append(
            {
                "name": parts[0],
                "load": parts[1],
                "active": parts[2],
                "sub": parts[3],
                "description": parts[4] if len(parts) > 4 else "",
            }
        )
    return ToolResult.success(output={"services": services, "count": len(services)})


def _sc_wait_stopped(name: str, timeout: float) -> Tuple[bool, str]:
    """Poll ``sc query`` until the service reports STOPPED (or timeout)."""
    deadline = time.monotonic() + max(1.0, float(timeout))
    last_state = "unknown"
    while time.monotonic() < deadline:
        query = run_command(["sc", "query", name], timeout=_SERVICE_TIMEOUT)
        if not query.ok:
            return False, f"sc query failed: {safe_truncate(query.stderr, 200)}"
        for raw in query.stdout.splitlines():
            line = raw.strip().lower()
            if line.startswith("state"):
                _, _, tail = line.partition(":")
                tokens = tail.strip().split()
                if tokens:
                    last_state = tokens[-1]
                if "stopped" in tail:
                    return True, "stopped"
                break
        time.sleep(0.5)
    return False, f"still {last_state} after {timeout:.0f}s"


def _service_action(name: str, action: str) -> ToolResult:
    """Start/stop/restart a service."""
    if not name or not str(name).strip():
        return ToolResult.failure("name is required")
    act = str(action or "").strip().lower()
    if act not in ("start", "stop", "restart", "status"):
        return ToolResult.failure(
            f"unsupported action {action!r}; use start/stop/restart/status"
        )

    if IS_WINDOWS:
        return _service_action_windows(str(name), act)

    systemctl = which("systemctl")
    if not systemctl:
        return ToolResult.failure("systemctl is not available on this system")
    result = run_command([systemctl, act, str(name)], timeout=_SERVICE_TIMEOUT)
    return ToolResult.success(
        output={
            "name": str(name),
            "action": act,
            "returncode": result.returncode,
            "stdout": safe_truncate(result.stdout, 2000),
            "stderr": safe_truncate(result.stderr, 2000),
            "ok": result.ok,
        }
    )


def _service_action_windows(name: str, action: str) -> ToolResult:
    if action == "status":
        result = run_command(["sc", "query", name], timeout=_SERVICE_TIMEOUT)
        return ToolResult.success(
            output={
                "name": name,
                "action": action,
                "returncode": result.returncode,
                "stdout": safe_truncate(result.stdout, 2000),
                "stderr": safe_truncate(result.stderr, 2000),
                "ok": result.ok,
            }
        )

    if action == "restart":
        stop = run_command(["sc", "stop", name], timeout=_SERVICE_TIMEOUT)
        stopped, detail = _sc_wait_stopped(name, _SERVICE_STOP_TIMEOUT)
        if not stopped:
            return ToolResult.failure(
                f"restart aborted: service {name} did not stop ({detail}); "
                f"sc stop said: {safe_truncate(stop.stdout, 200)}"
            )
        start = run_command(["sc", "start", name], timeout=_SERVICE_TIMEOUT)
        return ToolResult.success(
            output={
                "name": name,
                "action": "restart",
                "stop_ok": True,
                "start_ok": start.ok,
                "stdout": safe_truncate(stop.stdout + "\n" + start.stdout, 2000),
                "stderr": safe_truncate(stop.stderr + "\n" + start.stderr, 2000),
                "ok": start.ok,
            }
        )

    argv = ["sc", action, name]
    result = run_command(argv, timeout=_SERVICE_TIMEOUT)
    return ToolResult.success(
        output={
            "name": name,
            "action": action,
            "returncode": result.returncode,
            "stdout": safe_truncate(result.stdout, 2000),
            "stderr": safe_truncate(result.stderr, 2000),
            "ok": result.ok,
        }
    )


# --------------------------------------------------------------------------- #
#  Snapshots
# --------------------------------------------------------------------------- #
def _cpu_memory_snapshot() -> ToolResult:
    """CPU + memory in one shot."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return ToolResult.success(
            output={
                "cpu_count": os.cpu_count() or 1,
                "cpu_percent": None,
                "memory": None,
                "note": "psutil not installed",
            }
        )
    try:
        cpu = psutil.cpu_percent(interval=0.0)
    except Exception:  # noqa: BLE001
        cpu = None
    try:
        vm = psutil.virtual_memory()
        mem = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }
    except Exception:  # noqa: BLE001
        mem = None
    return ToolResult.success(
        output={
            "cpu_count": os.cpu_count() or 1,
            "cpu_percent": cpu,
            "memory": mem,
        }
    )


def _top_consumers(by: str = "cpu", n: int = 5) -> ToolResult:
    """The top N processes by ``cpu`` or ``memory``."""
    try:
        count = int(n)
    except (TypeError, ValueError):
        count = 5
    if count <= 0:
        count = 5
    return _list_processes(filter=None, sort_by=str(by or "cpu"), limit=count)


def _process_tree(pid: Union[int, str]) -> ToolResult:
    """Descendants of ``pid`` as a flat list (requires psutil)."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return ToolResult.failure(f"pid must be an integer, got {pid!r}")
    try:
        import psutil  # type: ignore
    except ImportError:
        return ToolResult.failure("psutil is required for process_tree")
    try:
        proc = psutil.Process(pid_int)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"process_tree({pid_int}) failed: {exc}")
    try:
        children = proc.children(recursive=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"process_tree({pid_int}) failed: {exc}")
    tree: List[Dict[str, Any]] = []
    for child in children:
        try:
            tree.append(
                {
                    "pid": child.pid,
                    "name": child.name(),
                    "ppid": child.ppid(),
                    "status": child.status(),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return ToolResult.success(
        output={"pid": pid_int, "children": tree, "count": len(tree)}
    )


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the built-in process tools bound to *ctx*."""
    del ctx  # ctx unused — process tools go through the registry-level security check.
    tools: List[Tool] = [
        FunctionTool(
            _list_processes,
            name="list_processes",
            description="Enumerate running processes (filter, sort, limit).",
        ),
        FunctionTool(
            _find_process,
            name="find_process",
            description="Find processes whose name contains a substring.",
        ),
        FunctionTool(
            _process_info,
            name="process_info",
            description="Detailed information about a single process id.",
        ),
        FunctionTool(
            _kill_process,
            name="kill_process",
            description="Terminate a process by pid or name.",
            dangerous=True,
        ),
        FunctionTool(
            _start_process,
            name="start_process",
            description="Launch a program.",
            dangerous=True,
        ),
        FunctionTool(
            _list_services,
            name="list_services",
            description="List system services (Windows SC / systemd).",
        ),
        FunctionTool(
            _service_action,
            name="service_action",
            description="Start/stop/restart/status a system service.",
            dangerous=True,
        ),
        FunctionTool(
            _cpu_memory_snapshot,
            name="cpu_memory_snapshot",
            description="Current CPU and memory utilisation.",
        ),
        FunctionTool(
            _top_consumers,
            name="top_consumers",
            description="Top N processes by cpu or memory.",
        ),
        FunctionTool(
            _process_tree,
            name="process_tree",
            description="Descendants of a process id (requires psutil).",
        ),
    ]
    return tools


__all__ = [
    "build_tools",
    "parse_tasklist_csv",
    "parse_ps_output",
]
