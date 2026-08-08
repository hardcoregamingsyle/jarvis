"""Linux audio: which sound server is actually running, and what is wrong with it.

Linux audio is three stacked systems and JARVIS has to know which one it is
talking to:

* **PipeWire** — the default on current Fedora and Ubuntu.  It ships a
  PulseAudio-compatible shim, so ``pactl`` works and *reports itself as
  PulseAudio*; the only reliable tell is the ``Server Name`` line, which reads
  ``PulseAudio (on PipeWire 1.x)``.  Detecting this correctly matters because
  the advice differs.
* **PulseAudio** — older installs, still common on Debian stable.
* **bare ALSA** — servers, minimal installs, and anything where the sound
  server died.  This is the case where the ``audio`` group actually matters.

:func:`check` is the diagnosis behind ``jarvis doctor`` on Linux.  It reports
what it found, what is missing, and the exact package-manager command to fix
each gap — no guessing at the distro, no "install the audio libraries".

This module also owns the small distro/package table (:func:`package_manager`,
:func:`install_command`) because audio is its largest consumer;
:mod:`jarvis.linux.desktop` imports it for the wmctrl/xdotool/libnotify advice.

Honesty note: written and tested on a Windows development box against faked
``pactl``/``aplay`` output captured from documentation and memory, not from the
target laptop.  The parsers are exercised against realistic text; the exact
output of the versions on that machine is not verified.  If a device list comes
back empty on Linux, dump ``pactl list short sources`` and compare.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import platform_utils
from ..core.contracts import ToolResult

log = logging.getLogger(__name__)

_QUERY_TIMEOUT = 10.0

#: Devices returned by one listing call.  Purely resource management: a
#: misbehaving server can enumerate thousands of virtual nodes, and that list
#: has to fit in a prompt.
MAX_DEVICES = 200


# --------------------------------------------------------------------------- #
#  Distro package advice
# --------------------------------------------------------------------------- #
#: ``manager -> (probe binary, install command prefix)``.  ``apt-get`` rather
#: than ``apt`` because ``apt`` also exists on a few non-Debian systems, while
#: ``apt-get`` is a reliable Debian/Ubuntu tell.
PACKAGE_MANAGERS = (
    ("apt", "apt-get", "sudo apt-get install -y"),
    ("dnf", "dnf", "sudo dnf install -y"),
    ("pacman", "pacman", "sudo pacman -S --needed"),
    ("zypper", "zypper", "sudo zypper install -y"),
)

#: ``capability -> {manager: package name}``.  ``None`` means "already part of
#: the base python3 package on that distro" — printing a package name that does
#: not exist is worse than saying nothing.
PACKAGES: Dict[str, Dict[str, Optional[str]]] = {
    "portaudio": {
        "apt": "portaudio19-dev",
        "dnf": "portaudio-devel",
        "pacman": "portaudio",
        "zypper": "portaudio-devel",
    },
    "ffmpeg": {
        "apt": "ffmpeg",
        # Fedora's own repositories carry ffmpeg-free; the full build needs
        # RPM Fusion, which is not something an installer should enable for you.
        "dnf": "ffmpeg-free",
        "pacman": "ffmpeg",
        "zypper": "ffmpeg",
    },
    "espeak-ng": {
        "apt": "espeak-ng",
        "dnf": "espeak-ng",
        "pacman": "espeak-ng",
        "zypper": "espeak-ng",
    },
    "pactl": {
        "apt": "pulseaudio-utils",
        "dnf": "pulseaudio-utils",
        "pacman": "libpulse",
        "zypper": "pulseaudio-utils",
    },
    "alsa-utils": {
        "apt": "alsa-utils",
        "dnf": "alsa-utils",
        "pacman": "alsa-utils",
        "zypper": "alsa-utils",
    },
    "wmctrl": {
        "apt": "wmctrl",
        "dnf": "wmctrl",
        "pacman": "wmctrl",
        "zypper": "wmctrl",
    },
    "xdotool": {
        "apt": "xdotool",
        "dnf": "xdotool",
        "pacman": "xdotool",
        "zypper": "xdotool",
    },
    "ydotool": {
        "apt": "ydotool",
        "dnf": "ydotool",
        "pacman": "ydotool",
        "zypper": "ydotool",
    },
    "libnotify": {
        "apt": "libnotify-bin",
        "dnf": "libnotify",
        "pacman": "libnotify",
        "zypper": "libnotify-tools",
    },
    "python-venv": {
        "apt": "python3-venv",
        "dnf": None,
        "pacman": None,
        "zypper": None,
    },
}


def package_manager() -> Optional[str]:
    """Which package manager this machine uses: ``apt``/``dnf``/``pacman``/``zypper``.

    ``None`` on anything else (or on Windows), which callers must render as
    "install the equivalent of X for your distribution" rather than a wrong
    command.
    """
    if not platform_utils.IS_LINUX:
        return None
    try:
        for name, probe, _prefix in PACKAGE_MANAGERS:
            if platform_utils.which(probe):
                return name
    except Exception as exc:  # noqa: BLE001
        log.debug("package-manager probe failed: %s", exc)
    return None


def package_for(capability: str, manager: Optional[str] = None) -> Optional[str]:
    """The package providing ``capability`` on ``manager``, or ``None``."""
    table = PACKAGES.get(capability)
    if not table:
        return None
    return table.get(manager or package_manager() or "")


def install_command(capabilities: List[str], manager: Optional[str] = None) -> Optional[str]:
    """The exact one-line command that installs ``capabilities``.

    Returns ``None`` when the package manager is unknown or every requested
    capability is already bundled with the distro's python3, so callers can
    stay silent instead of printing a command that would fail.
    """
    name = manager or package_manager()
    if not name:
        return None
    prefix = next((p for m, _probe, p in PACKAGE_MANAGERS if m == name), None)
    if not prefix:
        return None
    packages = []
    for capability in capabilities:
        package = package_for(capability, name)
        if package and package not in packages:
            packages.append(package)
    if not packages:
        return None
    return f"{prefix} {' '.join(packages)}"


# --------------------------------------------------------------------------- #
#  Backend detection
# --------------------------------------------------------------------------- #
def _run(argv: List[str]):
    return platform_utils.run_command(argv, timeout=_QUERY_TIMEOUT)


def _pactl_info() -> Optional[str]:
    """``pactl info`` output, or ``None`` when pactl is absent or fails."""
    pactl = platform_utils.which("pactl")
    if not pactl:
        return None
    result = _run([pactl, "info"])
    return result.stdout if result.ok else None


def backend() -> str:
    """The sound server in use: ``pipewire``, ``pulseaudio``, ``alsa`` or ``none``.

    PipeWire is distinguished from PulseAudio by the ``Server Name`` line, not
    by the presence of ``pactl`` — PipeWire's compatibility layer answers
    ``pactl`` perfectly well and would otherwise be misreported as PulseAudio,
    sending the owner to the wrong config files.
    """
    if not platform_utils.IS_LINUX:
        return "none"
    try:
        info = _pactl_info()
        if info:
            for line in info.splitlines():
                if line.lower().startswith("server name:"):
                    return "pipewire" if "pipewire" in line.lower() else "pulseaudio"
            return "pipewire" if "pipewire" in info.lower() else "pulseaudio"

        pw_cli = platform_utils.which("pw-cli")
        if pw_cli and _run([pw_cli, "info", "0"]).ok:
            return "pipewire"

        aplay = platform_utils.which("aplay")
        if aplay:
            result = _run([aplay, "-l"])
            if result.ok and "card" in result.stdout.lower():
                return "alsa"
    except Exception as exc:  # noqa: BLE001 - detection never raises
        log.debug("audio backend probe failed: %s", exc)
    return "none"


# --------------------------------------------------------------------------- #
#  Device enumeration
# --------------------------------------------------------------------------- #
_ALSA_CARD_RE = re.compile(
    r"^card\s+(?P<card>\d+):\s+(?P<cardid>\S+)\s+\[(?P<cardname>[^]]*)\],\s+"
    r"device\s+(?P<device>\d+):\s+(?P<devid>.*?)\s*\[(?P<devname>[^]]*)\]"
)


def _parse_pactl_short(text: str, *, kind: str) -> List[Dict[str, Any]]:
    """Parse ``pactl list short sources|sinks`` (tab-separated, one per line)."""
    devices: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        name = fields[1].strip()
        entry: Dict[str, Any] = {
            "index": fields[0].strip(),
            "name": name,
            "description": name,
            "driver": fields[2].strip() if len(fields) > 2 else "",
            "spec": fields[3].strip() if len(fields) > 3 else "",
            "state": fields[4].strip() if len(fields) > 4 else "",
            "kind": kind,
            # A ".monitor" source is a loopback of an output, not a microphone.
            # Kept in the list (it is genuinely recordable) but flagged, so
            # check() does not count it as a usable input.
            "monitor": name.endswith(".monitor"),
        }
        devices.append(entry)
        if len(devices) >= MAX_DEVICES:
            break
    return devices


def _parse_alsa_list(text: str, *, kind: str) -> List[Dict[str, Any]]:
    """Parse ``aplay -l`` / ``arecord -l`` into the same device dicts."""
    devices: List[Dict[str, Any]] = []
    for line in text.splitlines():
        match = _ALSA_CARD_RE.match(line.strip())
        if not match:
            continue
        card = match.group("card")
        device = match.group("device")
        devices.append(
            {
                "index": f"hw:{card},{device}",
                "name": f"hw:{card},{device}",
                "description": f"{match.group('cardname')} — {match.group('devname')}",
                "driver": "alsa",
                "spec": "",
                "state": "",
                "kind": kind,
                "monitor": False,
            }
        )
        if len(devices) >= MAX_DEVICES:
            break
    return devices


def _list_devices(*, kind: str) -> List[Dict[str, Any]]:
    if not platform_utils.IS_LINUX:
        return []
    pactl_noun = "sources" if kind == "input" else "sinks"
    alsa_tool = "arecord" if kind == "input" else "aplay"
    try:
        pactl = platform_utils.which("pactl")
        if pactl:
            result = _run([pactl, "list", "short", pactl_noun])
            if result.ok:
                devices = _parse_pactl_short(result.stdout, kind=kind)
                if devices:
                    return devices

        tool = platform_utils.which(alsa_tool)
        if tool:
            result = _run([tool, "-l"])
            if result.ok:
                return _parse_alsa_list(result.stdout, kind=kind)
    except Exception as exc:  # noqa: BLE001 - enumeration never raises
        log.debug("%s enumeration failed: %s", kind, exc)
    return []


def list_input_devices() -> List[Dict[str, Any]]:
    """Recording devices, newest API first (``pactl``), then ``arecord -l``.

    Returns ``[]`` — never raises — when no sound server answers.  Entries with
    ``monitor=True`` are output loopbacks, not microphones.
    """
    return _list_devices(kind="input")


def list_output_devices() -> List[Dict[str, Any]]:
    """Playback devices.  Returns ``[]`` when no sound server answers."""
    return _list_devices(kind="output")


def _default_device(*, kind: str) -> Optional[str]:
    if not platform_utils.IS_LINUX:
        return None
    pactl = platform_utils.which("pactl")
    if not pactl:
        return None
    verb = "get-default-source" if kind == "input" else "get-default-sink"
    label = "Default Source:" if kind == "input" else "Default Sink:"
    try:
        result = _run([pactl, verb])
        if result.ok and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
        # get-default-source/sink arrived in PulseAudio 15; older servers only
        # report it inside `pactl info`.
        info = _pactl_info() or ""
        for line in info.splitlines():
            if line.strip().startswith(label):
                return line.split(":", 1)[1].strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("default %s probe failed: %s", kind, exc)
    return None


def default_input() -> Optional[str]:
    """The default recording device's name, or ``None``."""
    return _default_device(kind="input")


def default_output() -> Optional[str]:
    """The default playback device's name, or ``None``."""
    return _default_device(kind="output")


def set_default_input(name: str) -> ToolResult:
    """Make ``name`` the default recording device for this user.

    A per-user PulseAudio/PipeWire setting; nothing system-wide is touched.
    Returns a failure rather than raising when ``pactl`` is absent or the name
    is unknown, and the failure lists the names that *do* exist.
    """
    target = str(name or "").strip()
    if not target:
        return ToolResult.failure("no device name given")
    if not platform_utils.IS_LINUX:
        return ToolResult.failure(
            f"set_default_input is Linux-only; this host is {platform_utils.os_name()}"
        )
    pactl = platform_utils.which("pactl")
    if not pactl:
        hint = install_command(["pactl"])
        return ToolResult.failure(
            "pactl is not installed, so the default input cannot be changed."
            + (f" Install it with: {hint}" if hint else "")
        )
    try:
        result = _run([pactl, "set-default-source", target])
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"pactl failed: {exc}")
    if not result.ok:
        known = [d["name"] for d in list_input_devices()]
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return ToolResult.failure(
            f"could not set the default source to {target!r}: {detail}. "
            f"Known sources: {', '.join(known) if known else 'none'}"
        )
    return ToolResult.success({"default_input": target})


# --------------------------------------------------------------------------- #
#  Diagnosis
# --------------------------------------------------------------------------- #
#: Where distributions put the PortAudio shared object, for the case where
#: ``ldconfig`` itself is missing (minimal images drop it).
_PORTAUDIO_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libportaudio.so.2",
    "/usr/lib/aarch64-linux-gnu/libportaudio.so.2",
    "/usr/lib64/libportaudio.so.2",
    "/usr/lib/libportaudio.so.2",
    "/usr/local/lib/libportaudio.so.2",
)


def portaudio_installed() -> bool:
    """True when ``libportaudio2`` is present, which is what ``sounddevice`` needs.

    ``pip install sounddevice`` succeeds without it and then fails at import
    with ``OSError: PortAudio library not found`` — a message that sends most
    people looking for a Python problem.  Probed with ``ldconfig -p`` first,
    then by looking for the shared object directly.
    """
    if not platform_utils.IS_LINUX:
        return False
    try:
        ldconfig = platform_utils.which("ldconfig")
        if ldconfig:
            result = _run([ldconfig, "-p"])
            if result.ok and "libportaudio" in result.stdout:
                return True
            if result.ok:
                return False
        return any(Path(p).exists() for p in _PORTAUDIO_CANDIDATES)
    except Exception as exc:  # noqa: BLE001
        log.debug("portaudio probe failed: %s", exc)
        return False


def in_audio_group() -> Optional[bool]:
    """Whether this user is in the ``audio`` group; ``None`` when undeterminable.

    Worth a caveat: on a PipeWire or PulseAudio box this group is mostly
    historical.  The sound server runs as you and you reach it through a socket
    in ``$XDG_RUNTIME_DIR``, so membership changes nothing.  It matters when
    something opens ``/dev/snd/*`` directly — bare ALSA, or PortAudio falling
    back to ALSA because the server is not running.  :func:`check` therefore
    only raises it as a problem in that case.
    """
    if not platform_utils.IS_LINUX:
        return None
    try:
        id_exe = platform_utils.which("id")
        if id_exe:
            result = _run([id_exe, "-nG"])
            if result.ok:
                return "audio" in result.stdout.split()
        # No coreutils `id`: ask the C library directly.  `grp` exists only on
        # POSIX, hence the guarded import.
        try:
            import grp  # noqa: PLC0415 - POSIX-only stdlib, must stay lazy
        except ImportError:
            return None
        gids = set(os.getgroups())
        entry = grp.getgrnam("audio")
        return entry.gr_gid in gids
    except Exception as exc:  # noqa: BLE001
        log.debug("audio group probe failed: %s", exc)
        return None


def is_available() -> bool:
    """True when some usable sound system was detected.  Never raises."""
    return backend() != "none"


def check() -> Dict[str, Any]:
    """Full audio diagnosis for ``jarvis doctor`` on Linux.

    Always returns a dict; never raises, even with nothing installed.  The
    ``problems`` list is written for a human, and every entry has a matching
    entry in ``fixes`` giving the exact command for *this* distribution.  On a
    non-Linux host the report says so and stops rather than inventing findings.
    """
    manager = package_manager()
    report: Dict[str, Any] = {
        "platform": platform_utils.os_name(),
        "backend": "none",
        "tools": {},
        "inputs": [],
        "outputs": [],
        "microphones": 0,
        "default_input": None,
        "default_output": None,
        "audio_group": None,
        "portaudio": False,
        "package_manager": manager,
        "problems": [],
        "fixes": [],
        "ok": False,
    }
    problems: List[str] = report["problems"]
    fixes: List[str] = report["fixes"]

    if not platform_utils.IS_LINUX:
        problems.append(
            f"this host is {platform_utils.os_name()}, not Linux; "
            "jarvis.linux.audio reports nothing here by design"
        )
        return report

    try:
        report["tools"] = {
            name: bool(platform_utils.which(name))
            for name in ("pactl", "pw-cli", "aplay", "arecord", "ffmpeg", "espeak-ng")
        }
        report["backend"] = backend()
        report["inputs"] = list_input_devices()
        report["outputs"] = list_output_devices()
        report["microphones"] = sum(1 for d in report["inputs"] if not d.get("monitor"))
        report["default_input"] = default_input()
        report["default_output"] = default_output()
        report["audio_group"] = in_audio_group()
        report["portaudio"] = portaudio_installed()

        missing: List[str] = []

        if report["backend"] == "none":
            problems.append(
                "no sound server answered: neither pactl/pw-cli nor aplay found a device. "
                "If you are on a desktop, check 'systemctl --user status pipewire'."
            )
            missing.extend(["pactl", "alsa-utils"])
        if not report["tools"].get("pactl") and report["backend"] in ("pipewire", "pulseaudio"):
            problems.append(
                "pactl is missing, so JARVIS cannot list or switch audio devices."
            )
            missing.append("pactl")
        if report["microphones"] == 0:
            problems.append(
                "no capture device found (monitor/loopback sources do not count). "
                "Check that the microphone is not muted in the sound settings."
            )
        if not report["portaudio"]:
            problems.append(
                "libportaudio2 is missing — 'import sounddevice' will fail with "
                "'PortAudio library not found', which looks like a Python problem "
                "but is not."
            )
            missing.append("portaudio")
        if not report["tools"].get("ffmpeg"):
            problems.append("ffmpeg is missing; some audio conversion paths need it.")
            missing.append("ffmpeg")
        if not report["tools"].get("espeak-ng"):
            problems.append(
                "espeak-ng is missing; it is the last-resort offline voice when "
                "Piper and edge-tts are unavailable."
            )
            missing.append("espeak-ng")
        if report["backend"] == "alsa" and report["audio_group"] is False:
            problems.append(
                "using bare ALSA and this user is not in the 'audio' group, so "
                "/dev/snd may not be readable. (On PipeWire/PulseAudio this group "
                "is irrelevant — the socket in $XDG_RUNTIME_DIR is what matters.)"
            )
            fixes.append(f"sudo usermod -aG audio {os.environ.get('USER', '$USER')}")
            fixes.append("# then log out and back in for the new group to take effect")

        command = install_command(missing, manager)
        if command:
            fixes.insert(0, command)
        elif missing and not manager:
            fixes.insert(
                0,
                "install the equivalent of "
                + ", ".join(sorted(set(missing)))
                + " with your distribution's package manager "
                "(apt/dnf/pacman/zypper were not found)",
            )

        report["ok"] = not problems
    except Exception as exc:  # noqa: BLE001 - a diagnosis never crashes the caller
        log.debug("audio check failed", exc_info=True)
        problems.append(f"the audio check itself failed: {type(exc).__name__}: {exc}")

    return report


__all__ = [
    "PACKAGE_MANAGERS",
    "PACKAGES",
    "MAX_DEVICES",
    "package_manager",
    "package_for",
    "install_command",
    "backend",
    "list_input_devices",
    "list_output_devices",
    "default_input",
    "default_output",
    "set_default_input",
    "portaudio_installed",
    "in_audio_group",
    "is_available",
    "check",
]
