"""System-level tools: shell, hardware, clipboard, notifications, power.

These are the "access to the entire PC" layer.  Every heavy dependency
(``psutil``, ``mss``, ``PIL``) is imported lazily inside the function that
needs it, so importing this module on a stock Python has no side effects.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import (
    IS_LINUX,
    IS_MAC,
    IS_WINDOWS,
    resolve_path,
    run_command,
    system_summary,
    which,
)
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Local helpers
# --------------------------------------------------------------------------- #
_OUTPUT_LIMIT = 20000


def _kill_proc(proc: "subprocess.Popen") -> None:
    """Best-effort kill + wait — a hung child holding a pipe becomes a zombie."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=2)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _run_with_input(argv: List[str], data: str, timeout: float = 5.0) -> ToolResult:
    """Run *argv* with *data* piped to stdin, killing the child on timeout."""
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, FileNotFoundError) as exc:
        return ToolResult.failure(f"could not launch {argv[0]}: {exc}")
    try:
        _, stderr = proc.communicate(input=data, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_proc(proc)
        return ToolResult.failure(f"{argv[0]}: timed out after {timeout}s")
    if proc.returncode != 0:
        return ToolResult.failure(
            f"{argv[0]} exited {proc.returncode}: {safe_truncate(stderr or '', 500)}"
        )
    return ToolResult.success(output={"bytes": len(data)})


# --------------------------------------------------------------------------- #
#  System info / hardware
# --------------------------------------------------------------------------- #
def _system_info() -> ToolResult:
    """A quick, dependency-free snapshot of the host — enriched by psutil when available."""
    info = dict(system_summary())
    info["cpu_count"] = os.cpu_count() or 1
    try:
        total, used, free = shutil.disk_usage(str(Path.home()))
        info["home_disk"] = {"total": total, "used": used, "free": free}
    except OSError:
        info["home_disk"] = None

    try:
        import psutil  # type: ignore
    except ImportError:
        info["psutil"] = False
        return ToolResult.success(output=info)

    try:
        info["cpu_percent"] = psutil.cpu_percent(interval=0.0)
    except Exception:  # noqa: BLE001
        info["cpu_percent"] = None
    try:
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "percent": vm.percent,
        }
    except Exception:  # noqa: BLE001
        info["memory"] = None
    try:
        info["boot_time"] = psutil.boot_time()
    except Exception:  # noqa: BLE001
        info["boot_time"] = None
    try:
        battery = psutil.sensors_battery()
        info["battery"] = (
            {"percent": battery.percent, "plugged": battery.power_plugged}
            if battery
            else None
        )
    except Exception:  # noqa: BLE001
        info["battery"] = None
    info["psutil"] = True
    return ToolResult.success(output=info)


def _disk_usage(path: Optional[str] = None) -> ToolResult:
    """Per-mount disk usage.  Uses ``psutil`` if present, else stdlib."""
    if path:
        try:
            resolved = resolve_path(str(path))
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        try:
            total, used, free = shutil.disk_usage(str(resolved))
        except OSError as exc:
            return ToolResult.failure(f"disk_usage failed: {exc}")
        return ToolResult.success(
            output={"path": str(resolved), "total": total, "used": used, "free": free}
        )

    mounts: List[dict] = []
    try:
        import psutil  # type: ignore

        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                mounts.append(
                    {
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                    }
                )
            except OSError:
                continue
    except ImportError:
        if IS_WINDOWS:
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                root = f"{letter}:\\"
                if not os.path.exists(root):
                    continue
                try:
                    total, used, free = shutil.disk_usage(root)
                    mounts.append(
                        {"device": root, "mountpoint": root, "total": total, "used": used, "free": free}
                    )
                except OSError:
                    continue
        else:
            for root in ("/", "/home", "/tmp", "/var"):
                if not os.path.exists(root):
                    continue
                try:
                    total, used, free = shutil.disk_usage(root)
                    mounts.append(
                        {"device": root, "mountpoint": root, "total": total, "used": used, "free": free}
                    )
                except OSError:
                    continue

    return ToolResult.success(output={"mounts": mounts, "count": len(mounts)})


def _battery_status() -> ToolResult:
    try:
        import psutil  # type: ignore
    except ImportError:
        return ToolResult.failure("psutil is not installed; battery status is unavailable")
    try:
        battery = psutil.sensors_battery()
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"battery query failed: {exc}")
    if battery is None:
        return ToolResult.failure("no battery detected on this system")
    return ToolResult.success(
        output={
            "percent": battery.percent,
            "plugged": bool(battery.power_plugged),
            "secsleft": battery.secsleft,
        }
    )


def _screen_resolution() -> ToolResult:
    try:
        import tkinter  # noqa: F401 — checking availability

        root = tkinter.Tk()
        try:
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
        finally:
            root.destroy()
        return ToolResult.success(output={"width": int(width), "height": int(height)})
    except Exception:  # noqa: BLE001
        pass

    if IS_LINUX and which("xrandr"):
        result = run_command(["xrandr"], timeout=5.0)
        if result.ok:
            for line in result.stdout.splitlines():
                if "*" in line:
                    tokens = line.split()
                    if tokens:
                        parts = tokens[0].split("x")
                        if len(parts) == 2 and all(p.isdigit() for p in parts):
                            return ToolResult.success(
                                output={"width": int(parts[0]), "height": int(parts[1])}
                            )
        return ToolResult.failure("could not parse xrandr output")

    return ToolResult.failure("screen resolution query is not available on this platform")


# --------------------------------------------------------------------------- #
#  Environment
# --------------------------------------------------------------------------- #
def _get_env(name: str) -> ToolResult:
    if not name:
        return ToolResult.failure("name is required")
    value = os.environ.get(str(name))
    if value is None:
        return ToolResult.failure(f"environment variable not set: {name}")
    return ToolResult.success(output={"name": str(name), "value": value})


def _set_env(name: str, value: str) -> ToolResult:
    if not name:
        return ToolResult.failure("name is required")
    os.environ[str(name)] = "" if value is None else str(value)
    return ToolResult.success(output={"name": str(name), "value": os.environ[str(name)]})


# --------------------------------------------------------------------------- #
#  Shell
# --------------------------------------------------------------------------- #
def _make_run_command(security: Any, default_timeout: float):
    def _run(
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        """Run a shell command through the platform default shell."""
        if not command or not str(command).strip():
            return ToolResult.failure("command is required")
        checker = getattr(security, "check_command", None) if security is not None else None
        if callable(checker):
            try:
                decision = checker(str(command))
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"security check_command errored: {exc}")
            if not bool(getattr(decision, "allowed", True)):
                reason = getattr(decision, "reason", None) or "refused by security policy"
                return ToolResult.failure(f"run_command: {reason}")
            if bool(getattr(decision, "requires_confirmation", False)):
                allows = getattr(security, "allows", None)
                if not callable(allows) or not bool(allows(decision)):
                    return ToolResult.failure("run_command: requires confirmation")
        actual_timeout = float(timeout) if timeout else float(default_timeout)
        cwd_arg = str(cwd) if cwd else None
        result = run_command(str(command), timeout=actual_timeout, cwd=cwd_arg, shell=True)
        return ToolResult.success(
            output={
                "command": str(command),
                "returncode": result.returncode,
                "stdout": safe_truncate(result.stdout, _OUTPUT_LIMIT),
                "stderr": safe_truncate(result.stderr, _OUTPUT_LIMIT),
                "timed_out": result.timed_out,
                "ok": result.ok,
            }
        )

    return _run


# --------------------------------------------------------------------------- #
#  Clipboard
# --------------------------------------------------------------------------- #
def _clipboard_get() -> ToolResult:
    if IS_WINDOWS:
        argv = ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"]
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, FileNotFoundError) as exc:
            return ToolResult.failure(f"could not launch PowerShell: {exc}")
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            _kill_proc(proc)
            return ToolResult.failure("clipboard read timed out")
        if proc.returncode != 0:
            return ToolResult.failure(
                f"Get-Clipboard exited {proc.returncode}: {safe_truncate(stderr, 500)}"
            )
        return ToolResult.success(output={"text": stdout.rstrip("\r\n")})

    if IS_LINUX:
        for exe, args in (
            ("wl-paste", ["--no-newline"]),
            ("xclip", ["-selection", "clipboard", "-o"]),
            ("xsel", ["--clipboard", "--output"]),
        ):
            if which(exe):
                argv = [exe, *args]
                try:
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                except (OSError, FileNotFoundError) as exc:
                    return ToolResult.failure(f"could not launch {exe}: {exc}")
                try:
                    stdout, stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    _kill_proc(proc)
                    return ToolResult.failure(f"{exe}: timed out")
                if proc.returncode != 0:
                    return ToolResult.failure(
                        f"{exe} exited {proc.returncode}: {safe_truncate(stderr, 500)}"
                    )
                return ToolResult.success(output={"text": stdout})
        return ToolResult.failure("no clipboard tool found (install xclip, xsel, or wl-clipboard)")

    if IS_MAC:
        if not which("pbpaste"):
            return ToolResult.failure("pbpaste is unavailable")
        result = run_command(["pbpaste"], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"pbpaste failed: {safe_truncate(result.stderr, 500)}")
        return ToolResult.success(output={"text": result.stdout})

    return ToolResult.failure("clipboard access is not supported on this platform")


def _clipboard_set(text: str) -> ToolResult:
    payload = "" if text is None else str(text)
    if IS_WINDOWS:
        argv = ["powershell", "-NoProfile", "-Command", "$Input | Set-Clipboard"]
        return _run_with_input(argv, payload, timeout=5.0)
    if IS_LINUX:
        for exe, args in (
            ("wl-copy", []),
            ("xclip", ["-selection", "clipboard"]),
            ("xsel", ["--clipboard", "--input"]),
        ):
            if which(exe):
                return _run_with_input([exe, *args], payload, timeout=5.0)
        return ToolResult.failure("no clipboard tool found (install xclip, xsel, or wl-clipboard)")
    if IS_MAC:
        if not which("pbcopy"):
            return ToolResult.failure("pbcopy is unavailable")
        return _run_with_input(["pbcopy"], payload, timeout=5.0)
    return ToolResult.failure("clipboard access is not supported on this platform")


# --------------------------------------------------------------------------- #
#  Notifications / audio / display
# --------------------------------------------------------------------------- #
def _notify(title: str, message: str) -> ToolResult:
    if IS_WINDOWS:
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            f"$n = New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon = [System.Drawing.SystemIcons]::Information;"
            "$n.Visible = $true;"
            f"$n.ShowBalloonTip(5000, '{str(title).replace(chr(39), chr(39)*2)}', "
            f"'{str(message).replace(chr(39), chr(39)*2)}', 'Info');"
            "Start-Sleep -Seconds 5;"
        )
        result = run_command(
            ["powershell", "-NoProfile", "-Command", script], timeout=10.0
        )
        if not result.ok:
            return ToolResult.failure(f"notify failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"title": str(title), "message": str(message)})
    if IS_LINUX:
        if not which("notify-send"):
            return ToolResult.failure("notify-send is not installed")
        result = run_command(["notify-send", str(title), str(message)], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"notify-send failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"title": str(title), "message": str(message)})
    if IS_MAC:
        if not which("osascript"):
            return ToolResult.failure("osascript is unavailable")
        applescript = (
            f'display notification "{str(message).replace(chr(34), chr(92)+chr(34))}" '
            f'with title "{str(title).replace(chr(34), chr(92)+chr(34))}"'
        )
        result = run_command(["osascript", "-e", applescript], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"osascript failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"title": str(title), "message": str(message)})
    return ToolResult.failure("notifications are not supported on this platform")


def _volume_set(level: float) -> ToolResult:
    try:
        pct = max(0.0, min(100.0, float(level)))
    except (TypeError, ValueError):
        return ToolResult.failure("level must be a number in 0..100")
    if IS_LINUX and which("amixer"):
        result = run_command(["amixer", "-q", "sset", "Master", f"{pct:.0f}%"], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"amixer failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"level": pct})
    if IS_MAC and which("osascript"):
        result = run_command(
            ["osascript", "-e", f"set volume output volume {int(pct)}"], timeout=5.0
        )
        if not result.ok:
            return ToolResult.failure(f"osascript failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"level": pct})
    if IS_WINDOWS and which("nircmd"):
        raw = int((pct / 100.0) * 65535)
        result = run_command(["nircmd", "setsysvolume", str(raw)], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"nircmd failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"level": pct})
    return ToolResult.failure("no volume tool available on this system")


def _volume_mute(flag: bool) -> ToolResult:
    want_mute = bool(flag)
    if IS_LINUX and which("amixer"):
        state = "mute" if want_mute else "unmute"
        result = run_command(["amixer", "-q", "sset", "Master", state], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"amixer failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"mute": want_mute})
    if IS_MAC and which("osascript"):
        cmd = f"set volume output muted {'true' if want_mute else 'false'}"
        result = run_command(["osascript", "-e", cmd], timeout=5.0)
        if not result.ok:
            return ToolResult.failure(f"osascript failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"mute": want_mute})
    if IS_WINDOWS and which("nircmd"):
        result = run_command(
            ["nircmd", "mutesysvolume", "1" if want_mute else "0"], timeout=5.0
        )
        if not result.ok:
            return ToolResult.failure(f"nircmd failed: {safe_truncate(result.stderr, 300)}")
        return ToolResult.success(output={"mute": want_mute})
    return ToolResult.failure("no volume tool available on this system")


def _take_screenshot(path: Optional[str] = None) -> ToolResult:
    try:
        import mss  # type: ignore
    except ImportError:
        return ToolResult.failure("mss is not installed; cannot take a screenshot")
    if path:
        try:
            target = resolve_path(str(path))
        except ValueError as exc:
            return ToolResult.failure(str(exc))
    else:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(prefix="jarvis-shot-", suffix=".png", delete=False) as fh:
            target = Path(fh.name)
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            data = sct.grab(monitor)
        try:
            import mss.tools as mss_tools  # type: ignore

            mss_tools.to_png(data.rgb, data.size, output=str(target))
        except Exception:
            try:
                from PIL import Image  # type: ignore

                img = Image.frombytes("RGB", data.size, data.rgb)
                img.save(str(target))
            except ImportError:
                return ToolResult.failure(
                    "cannot save screenshot: neither mss.tools nor Pillow succeeded"
                )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"screenshot failed: {exc}")
    return ToolResult.success(
        output={"path": str(target), "size": target.stat().st_size if target.exists() else 0},
        is_artifact=True,
    )


# --------------------------------------------------------------------------- #
#  Power
# --------------------------------------------------------------------------- #
def _make_power(security: Any):
    def _power_action(action: str) -> ToolResult:
        if not action:
            return ToolResult.failure("action is required")
        act = str(action).strip().lower()
        checker = getattr(security, "check_command", None) if security is not None else None
        if callable(checker):
            try:
                decision = checker(f"power:{act}")
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"security check_command errored: {exc}")
            if not bool(getattr(decision, "allowed", True)):
                return ToolResult.failure(
                    f"power_action: {getattr(decision, 'reason', None) or 'refused by security policy'}"
                )
            if bool(getattr(decision, "requires_confirmation", False)):
                allows = getattr(security, "allows", None)
                if not callable(allows) or not bool(allows(decision)):
                    return ToolResult.failure("power_action: requires confirmation")

        argv: Optional[List[str]] = None
        if IS_WINDOWS:
            mapping = {
                "shutdown": ["shutdown", "/s", "/t", "0"],
                "restart": ["shutdown", "/r", "/t", "0"],
                "reboot": ["shutdown", "/r", "/t", "0"],
                "logoff": ["shutdown", "/l"],
                "logout": ["shutdown", "/l"],
                "lock": ["rundll32.exe", "user32.dll,LockWorkStation"],
                "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            }
            argv = mapping.get(act)
        elif IS_MAC:
            mapping = {
                "shutdown": ["osascript", "-e", 'tell app "System Events" to shut down'],
                "restart": ["osascript", "-e", 'tell app "System Events" to restart'],
                "reboot": ["osascript", "-e", 'tell app "System Events" to restart'],
                "sleep": ["pmset", "sleepnow"],
                "lock": ["pmset", "displaysleepnow"],
                "logoff": ["osascript", "-e", 'tell app "System Events" to log out'],
                "logout": ["osascript", "-e", 'tell app "System Events" to log out'],
            }
            argv = mapping.get(act)
        else:
            mapping = {
                "shutdown": ["systemctl", "poweroff"],
                "restart": ["systemctl", "reboot"],
                "reboot": ["systemctl", "reboot"],
                "sleep": ["systemctl", "suspend"],
                "logoff": ["loginctl", "terminate-user", os.environ.get("USER", "")],
                "logout": ["loginctl", "terminate-user", os.environ.get("USER", "")],
                "lock": ["loginctl", "lock-session"],
            }
            argv = mapping.get(act)

        if argv is None:
            return ToolResult.failure(
                f"power action {action!r} is not supported on {platform.system()}"
            )
        if not which(argv[0]):
            return ToolResult.failure(
                f"required command {argv[0]!r} is not available on this system"
            )
        result = run_command(argv, timeout=15.0)
        return ToolResult.success(
            output={
                "action": act,
                "returncode": result.returncode,
                "ok": result.ok,
                "stdout": safe_truncate(result.stdout, 2000),
                "stderr": safe_truncate(result.stderr, 2000),
            }
        )

    return _power_action


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the built-in system tools bound to *ctx*."""
    security = getattr(ctx, "security", None)
    config = getattr(ctx, "config", None)
    default_timeout = 120.0
    try:
        default_timeout = float(config.security.command_timeout)
    except Exception:  # noqa: BLE001
        pass

    tools: List[Tool] = [
        FunctionTool(_system_info, name="system_info", description="Basic host and hardware info."),
        FunctionTool(_disk_usage, name="disk_usage", description="Per-mount disk usage."),
        FunctionTool(_battery_status, name="battery_status", description="Battery percentage / plugged state."),
        FunctionTool(_screen_resolution, name="screen_resolution", description="Primary display resolution."),
        FunctionTool(_get_env, name="get_env", description="Read an environment variable."),
        FunctionTool(_set_env, name="set_env", description="Set an environment variable for this process."),
        FunctionTool(_clipboard_get, name="clipboard_get", description="Read the system clipboard as text."),
        FunctionTool(_clipboard_set, name="clipboard_set", description="Write text to the system clipboard."),
        FunctionTool(_notify, name="notify", description="Show a system notification."),
        FunctionTool(_volume_set, name="volume_set", description="Set the master audio volume (0..100)."),
        FunctionTool(_volume_mute, name="volume_mute", description="Mute or unmute audio."),
        FunctionTool(_take_screenshot, name="take_screenshot", description="Capture the primary display."),
        FunctionTool(
            _make_run_command(security, default_timeout),
            name="run_command",
            description="Run a shell command through the platform default shell.",
            dangerous=True,
        ),
        FunctionTool(
            _make_power(security),
            name="power_action",
            description="Shutdown/restart/sleep/lock/logoff the machine.",
            dangerous=True,
        ),
    ]
    return tools


__all__ = ["build_tools"]
