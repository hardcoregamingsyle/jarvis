"""Rootless Ollama: install it, serve it, and keep the weights current.

Ollama is the inference runtime JARVIS actually talks to — it publishes an
OpenAI-compatible API on ``/v1``, so :mod:`jarvis.llm.openai_compat` and
:mod:`jarvis.llm.vllm_backend` reach it with no adapter at all.  The problem is
getting it onto the machine.

**Why not the official installer.**  ``curl -fsSL https://ollama.com/install.sh
| sh`` needs root: it writes ``/usr/local/bin/ollama``, creates an ``ollama``
system user and drops a system-wide systemd unit.  The JARVIS installer's
standing promise is that it never runs ``sudo`` — where a system package is
needed it prints the command and lets the owner run it.  So this module does the
same job without root: fetch the official release tarball from GitHub, extract
it under ``$XDG_DATA_HOME/jarvis/ollama``, symlink ``~/.local/bin/ollama``, and
run it as a systemd **user** service.  Same binary, same releases, no root.

If a system-wide ``ollama`` is already on ``PATH`` — because the owner ran the
official installer, or their distribution packages it — that one is used and
nothing is duplicated.  :func:`install_plan` reports that as ``action="external"``.

**Everything here is safe to call twice.**  No function in this module raises
into its caller: each returns a dict describing what actually happened, with
``ok`` and a human-readable ``reason``.  A network outage, a missing binary and
a full disk are all normal conditions, not exceptions.

**Nothing outside the repository is clobbered.**  The managed directory carries
a ``.jarvis-managed`` marker; a directory without one is never replaced.  A
pre-existing ``~/.local/bin/ollama`` that JARVIS did not create is never
overwritten — :func:`install` reports the conflict and leaves it alone.

Developed on Windows against faked downloads and a faked systemd; the target is
the CPU-only Linux laptop.  The URLs, the control flow and the unit text are
unit-tested, the behaviour of a real ollama daemon is not.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from ..core import platform_utils

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
#: Upstream project.  Releases are published here as static tarballs, which is
#: what makes a rootless install possible at all.
GITHUB_REPO = "ollama/ollama"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_DOWNLOAD = (
    "https://github.com/" + GITHUB_REPO
    + "/releases/download/v{version}/ollama-linux-{arch}{suffix}"
)

#: Archive suffixes upstream has published, most recent first.
#:
#: Ollama moved its Linux releases from ``.tgz`` to zstd-compressed tar, so a
#: hardcoded extension 404s against a current release. The asset is therefore
#: chosen from the live release metadata whenever the API is reachable and this
#: list is only the fallback ordering — the next rename costs a re-run, not a
#: code change.
ASSET_SUFFIXES = (".tar.zst", ".tgz", ".tar.gz")
DEFAULT_ASSET_SUFFIX = ASSET_SUFFIXES[0]

#: zstd's frame magic. Sniffed from the file rather than trusted from the name,
#: because the name is upstream's to change.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

#: The binary is named ``ollama`` on every platform Ollama ships a tarball for.
BIN_NAME = "ollama"

#: Dropped inside the managed directory at install time.  Its absence is what
#: stops :func:`install` from replacing a directory somebody else put there.
MANAGED_MARKER = ".jarvis-managed"

#: systemd *user* unit.  Deliberately not ``ollama.service``: that is the name
#: the official root installer uses for its system unit, and colliding with it
#: would make ``systemctl --user status ollama`` ambiguous.
UNIT_NAME = "jarvis-ollama.service"

#: First line of any unit this module writes.  Its absence marks the file as
#: someone else's, and :func:`install_service` then refuses to overwrite it —
#: the same ownership check :func:`link_binary` applies to ~/.local/bin.
UNIT_MARKER = "# Written by JARVIS (jarvis.runtime.ollama)."

DEFAULT_HOST = "http://127.0.0.1:11434"

#: Short, because being offline is a normal state and must not stall a CLI.
API_TIMEOUT = 10.0
#: A model pull moves gigabytes; the read loop needs a socket timeout that
#: tolerates a slow mirror without hanging for ever on a dead one.
PULL_TIMEOUT = 900.0
DOWNLOAD_TIMEOUT = 900.0
#: Bytes per read.  Large enough to keep the loop cheap, small enough that
#: progress callbacks stay responsive on a slow link.
CHUNK = 1024 * 1024

#: A tarball needs room for the archive *and* the extracted tree, and the old
#: version stays on disk until the new one is in place.  Four times the
#: download plus a margin is deliberately pessimistic: refusing an install that
#: would just have fitted costs a re-run, running out of space at 90% costs the
#: afternoon.  (The amd64 asset is ~1.4 GB compressed.)
_INSTALL_SPACE_FACTOR = 4.0
#: Decompressing zstd through an intermediate ``.tar`` means the archive, the
#: tar and the unpacked tree all exist at once.
_INSTALL_SPACE_INTERMEDIATE = 2.0
_INSTALL_SPACE_MARGIN = 200_000_000
#: Model weights are written once, but ollama unpacks blobs alongside the
#: download, and a laptop with zero bytes free stops being a laptop.
_MODEL_SPACE_FACTOR = 1.15
_MODEL_SPACE_MARGIN = 2_000_000_000

_GB = 1_000_000_000.0

#: Machines Ollama publishes Linux tarballs for.
_ARCH_MAP = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "x64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


# --------------------------------------------------------------------------- #
#  Small indirections, so tests can replace the outside world
# --------------------------------------------------------------------------- #
def _which(name: str) -> Optional[str]:
    """``shutil.which``, via platform_utils.  Patched in tests."""
    return platform_utils.which(name)


def _urlopen(request: Any, timeout: float = API_TIMEOUT):
    """The single network entry point of this module.  Patched in tests."""
    return urllib.request.urlopen(request, timeout=timeout)  # nosec B310 - fixed https hosts


def _notify(progress: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
    """Hand an event to the caller's progress callback, swallowing its mistakes.

    A broken progress printer must not abort a 16 GB download that is otherwise
    going fine.
    """
    if progress is None:
        return
    try:
        progress(event)
    except Exception as exc:  # noqa: BLE001 - the callback is the caller's problem
        log.debug("progress callback raised %s", type(exc).__name__)


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
def home_dir() -> Path:
    return Path(os.path.expanduser("~"))


def local_bin_dir() -> Path:
    """``$XDG_BIN_HOME`` or ``~/.local/bin`` — where the symlink goes."""
    override = (os.environ.get("XDG_BIN_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return home_dir() / ".local" / "bin"


def runtime_dir(cfg: Any = None) -> Path:
    """Where JARVIS keeps its own copy of Ollama.

    ``<data>/ollama``, and ``<data>`` on Linux is ``$XDG_DATA_HOME/jarvis``.
    Honouring ``cfg.home()`` (and therefore ``JARVIS_HOME``) is what lets the
    test-suite point the whole thing at a temporary directory.
    """
    if cfg is not None:
        try:
            return Path(cfg.home()) / "ollama"
        except (AttributeError, OSError, TypeError, ValueError):
            log.debug("cfg.home() unusable; falling back to the platform data dir")
    return platform_utils.data_dir() / "ollama"


def download_dir(cfg: Any = None) -> Path:
    """Partial downloads live beside the target, on the same filesystem.

    Same filesystem matters twice: the free-space check is meaningful, and the
    finished file can be moved into place without a copy.
    """
    return runtime_dir(cfg).parent / "downloads"


def managed_binary(cfg: Any = None) -> Optional[Path]:
    """The binary inside our managed tree, if the tree holds one."""
    root = runtime_dir(cfg)
    for candidate in (root / "bin" / BIN_NAME, root / BIN_NAME):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def unit_dir() -> Path:
    """``$XDG_CONFIG_HOME/systemd/user`` — user units only, never system ones."""
    base = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    root = Path(base).expanduser() if base else (home_dir() / ".config")
    return root / "systemd" / "user"


def unit_path() -> Path:
    return unit_dir() / UNIT_NAME


def models_cache_dir() -> Path:
    """Where ollama itself stores blobs — ``$OLLAMA_MODELS`` or ``~/.ollama/models``."""
    override = (os.environ.get("OLLAMA_MODELS") or "").strip()
    if override:
        return Path(override).expanduser()
    return home_dir() / ".ollama" / "models"


# --------------------------------------------------------------------------- #
#  Disk space
# --------------------------------------------------------------------------- #
def _existing_ancestor(path: Path) -> Optional[Path]:
    """The nearest ancestor of ``path`` that exists, for a free-space query."""
    try:
        current = Path(path)
    except (TypeError, ValueError):
        return None
    seen = 0
    while seen < 64:
        try:
            if current.exists():
                return current
        except OSError:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
        seen += 1
    return None


def free_bytes(path: Path) -> int:
    """Bytes free on the filesystem holding ``path``; ``-1`` when unknowable.

    ``path`` need not exist yet — the query walks up to the nearest ancestor
    that does, which is the whole point when the target directory is what we
    are about to create.
    """
    anchor = _existing_ancestor(path)
    if anchor is None:
        return -1
    try:
        return int(shutil.disk_usage(str(anchor)).free)
    except (OSError, ValueError):
        log.debug("disk_usage failed for %s", anchor)
        return -1


def _space_check(target: Path, required: int) -> Dict[str, Any]:
    """Free space at ``target`` against ``required``, with the shortfall in GB."""
    available = free_bytes(target)
    known = available >= 0
    shortfall = max(0, required - available) if known else 0
    return {
        "path": str(target),
        "required_bytes": int(required),
        "required_gb": round(required / _GB, 2),
        "free_bytes": int(available),
        "free_gb": round(available / _GB, 2) if known else None,
        "space_known": known,
        # Unknown free space is not a reason to refuse: some filesystems simply
        # do not answer, and blocking a valid install on that would be worse
        # than the risk it guards against.
        "enough_space": (not known) or available >= required,
        "shortfall_bytes": int(shortfall),
        "shortfall_gb": round(shortfall / _GB, 2),
    }


def _short_message(what: str, check: Dict[str, Any]) -> str:
    return (
        f"Not enough disk space for {what}: {check['required_gb']} GB needed at "
        f"{check['path']}, {check['free_gb']} GB free, "
        f"{check['shortfall_gb']} GB short."
    )


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #
def is_installed(cfg: Any = None) -> Optional[Path]:
    """The ``ollama`` binary this machine would actually run, or ``None``.

    ``PATH`` first: a system-wide install (or one the owner made themselves)
    wins over ours, because duplicating it would leave two versions to keep in
    step.  Then ``~/.local/bin`` — which may not be on ``PATH`` in a
    non-interactive shell even though it is in the login one — and finally the
    managed tree.
    """
    found = _which(BIN_NAME)
    if found:
        return Path(found)
    candidates: List[Optional[Path]] = [local_bin_dir() / BIN_NAME, managed_binary(cfg)]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def is_managed(cfg: Any = None) -> bool:
    """True when the managed tree was written by this module."""
    try:
        return (runtime_dir(cfg) / MANAGED_MARKER).is_file()
    except OSError:
        return False


# ``ollama --version`` prints "ollama version is 0.5.7", and older builds print
# "ollama version 0.5.7"; a client that cannot reach a server prepends a warning
# line first, so the version is not necessarily on line one.
_VERSION_RE = re.compile(
    r"version\s+(?:is\s+)?v?(\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.]+)?)", re.IGNORECASE
)
_BARE_VERSION_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.]+)?)$")


def parse_version(text: Any) -> Optional[str]:
    """Pull a version out of ``ollama --version`` output; ``None`` for garbage.

    Requires the word "version" (or output that is nothing *but* a version
    number) so that arbitrary text containing a decimal number does not get
    mistaken for a release.
    """
    if not text:
        return None
    body = str(text)
    matches = _VERSION_RE.findall(body)
    if matches:
        return matches[-1]
    stripped = body.strip()
    if stripped:
        bare = _BARE_VERSION_RE.match(stripped)
        if bare:
            return bare.group(1)
    return None


def installed_version(cfg: Any = None, *, timeout: float = 20.0) -> Optional[str]:
    """The installed Ollama's version, or ``None`` if it cannot be determined.

    ``None`` covers every failure identically — not installed, not executable,
    hung, or printing something this build has never seen — because every one of
    them means the same thing to the caller: we do not know, so treat it as
    stale.
    """
    binary = is_installed(cfg)
    if binary is None:
        return None
    try:
        result = platform_utils.run_command([str(binary), "--version"], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a version probe never takes the app down
        log.debug("ollama --version failed: %s", type(exc).__name__)
        return None
    # The exit code is not consulted: some builds exit non-zero when they cannot
    # reach a running server, while still printing the client version we want.
    return parse_version((result.stdout or "") + "\n" + (result.stderr or ""))


def target_arch(machine: Optional[str] = None) -> Optional[str]:
    """Ollama's name for this CPU (``amd64``/``arm64``), or ``None`` if unsupported."""
    raw = str(machine if machine is not None else platform.machine() or "").strip().lower()
    return _ARCH_MAP.get(raw)


# --------------------------------------------------------------------------- #
#  Releases
# --------------------------------------------------------------------------- #
def _request(url: str, *, method: str = "GET", data: Optional[bytes] = None,
             headers: Optional[Dict[str, str]] = None) -> Any:
    merged = {"User-Agent": "jarvis-installer/1.0", "Accept": "*/*"}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, data=data, headers=merged, method=method)


def _release_json(*, timeout: float = API_TIMEOUT) -> Optional[Dict[str, Any]]:
    """The GitHub "latest release" object, or ``None``.

    Offline is the expected case on the target machine as often as not, so
    every failure collapses to ``None`` and the caller carries on with whatever
    is already installed.
    """
    try:
        with _urlopen(_request(GITHUB_API_LATEST), timeout=timeout) as response:
            raw = response.read(2_000_000)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        log.debug("GitHub release lookup failed: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - never propagate a lookup failure
        log.debug("GitHub release lookup failed: %s", type(exc).__name__)
        return None
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def latest_release(*, timeout: float = API_TIMEOUT) -> Optional[str]:
    """Latest published Ollama version (``"0.5.7"``), or ``None`` when offline."""
    data = _release_json(timeout=timeout)
    if not data:
        return None
    tag = str(data.get("tag_name") or data.get("name") or "").strip()
    if not tag:
        return None
    return parse_version(tag) or (tag[1:] if tag.startswith("v") else tag) or None


def release_asset_url(version: str, arch: Optional[str] = None,
                      suffix: str = DEFAULT_ASSET_SUFFIX) -> str:
    """URL of the official Linux archive for ``version`` on ``arch``.

    Pure string construction — no network, so it is usable in a plan that has to
    render while offline.  When the release API *is* reachable the real asset
    is taken from its metadata instead; this is the offline fallback.
    """
    clean = str(version or "").strip().lstrip("vV")
    resolved = arch or target_arch() or "amd64"
    return GITHUB_DOWNLOAD.format(version=clean, arch=resolved, suffix=suffix)


# --------------------------------------------------------------------------- #
#  zstd
# --------------------------------------------------------------------------- #
def _stdlib_zstd() -> bool:
    """True when this interpreter's ``tarfile`` reads ``.tar.zst`` by itself.

    Python 3.14 gained ``compression.zstd`` and with it transparent
    ``tarfile`` support.  On 3.9-3.13 there is none, which matters because the
    target laptop's interpreter is not this one.
    """
    try:
        import compression.zstd  # noqa: F401
        return True
    except ImportError:
        return False


def zstd_support() -> Optional[str]:
    """How this machine can decompress zstd: ``stdlib``, ``zstandard``, ``cli``, or ``None``.

    Three ways because the target is a CPU-only Linux laptop whose Python
    version is not ours to choose: 3.14 does it natively, older Pythons need
    either the ``zstandard`` wheel or the ``zstd`` command, and the last of
    those is a system package we may only *recommend* — installing it needs
    root, and this installer never runs ``sudo``.
    """
    if _stdlib_zstd():
        return "stdlib"
    try:
        import zstandard  # noqa: F401
        return "zstandard"
    except ImportError:
        pass
    if _which("zstd") or _which("unzstd"):
        return "cli"
    return None


ZSTD_HINT = (
    "this Python cannot read .tar.zst archives. Fix it with any one of: "
    "`pip install zstandard` (no root needed), Python 3.14 or newer, or "
    "`sudo apt-get install -y zstd` (run it yourself; this installer never uses sudo)."
)


def _is_zstd(path: Path) -> bool:
    """Sniff the zstd frame magic rather than trusting the file name."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == _ZSTD_MAGIC
    except OSError:
        return False


def _suffix_of(name: str) -> str:
    for suffix in ASSET_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def _can_extract(name: str) -> bool:
    """Whether this machine can unpack an asset with this name."""
    suffix = _suffix_of(name)
    if suffix in (".tgz", ".tar.gz"):
        return True
    if suffix == ".tar.zst":
        return zstd_support() is not None
    return False


def _pick_asset(release: Optional[Dict[str, Any]], arch: str,
                version: str) -> Dict[str, Any]:
    """Choose the plain Linux archive for ``arch`` from the release metadata.

    Exact-name matching is what excludes the variant builds published
    alongside it: ``ollama-linux-amd64-rocm.tar.zst`` is a gigabyte of AMD GPU
    runtime, and ``-mlx`` and ``-jetpack`` are equally wrong for a CPU-only
    laptop.  Preference goes to a format this machine can actually unpack, so a
    Python without zstd still installs from the ``.tgz`` when upstream offers
    one.
    """
    catalogue: Dict[str, Dict[str, Any]] = {}
    for asset in ((release or {}).get("assets") or []):
        if isinstance(asset, dict) and asset.get("name"):
            catalogue[str(asset["name"])] = asset

    candidates: List[Dict[str, Any]] = []
    for suffix in ASSET_SUFFIXES:
        name = f"ollama-linux-{arch}{suffix}"
        entry = catalogue.get(name)
        if entry is None:
            continue
        try:
            size = int(entry.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        candidates.append({
            "name": name,
            "url": str(entry.get("browser_download_url") or "")
                   or release_asset_url(version, arch, suffix),
            "size": size,
            "extractable": _can_extract(name),
        })

    if not candidates:
        # Offline, or a release whose asset list we could not read: fall back to
        # the current naming convention and let the download report the 404.
        name = f"ollama-linux-{arch}{DEFAULT_ASSET_SUFFIX}"
        return {
            "name": name,
            "url": release_asset_url(version, arch, DEFAULT_ASSET_SUFFIX),
            "size": 0,
            "extractable": _can_extract(name),
        }
    for candidate in candidates:
        if candidate["extractable"]:
            return candidate
    return candidates[0]


def _asset_size(url: str, *, timeout: float = API_TIMEOUT) -> int:
    """Byte size of an asset via a HEAD request; ``0`` when not known.

    ``0`` is reported as such rather than guessed — a made-up size would make
    the free-space check lie.
    """
    try:
        with _urlopen(_request(url, method="HEAD"), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else 0
    except Exception as exc:  # noqa: BLE001 - a size probe never raises
        log.debug("HEAD failed: %s", type(exc).__name__)
        return 0


# --------------------------------------------------------------------------- #
#  Install plan
# --------------------------------------------------------------------------- #
def install_plan(cfg: Any = None, *, timeout: float = API_TIMEOUT) -> Dict[str, Any]:
    """What :func:`install` *would* do.  No side effects whatsoever.

    ``action`` is exactly one of:

    ``external``     ollama is already on ``PATH`` outside our tree — use it
    ``current``      our copy is the latest release
    ``upgrade``      our copy is older than the latest release
    ``install``      not installed here at all
    ``unknown``      installed, but the latest release could not be looked up
    ``unsupported``  no official tarball exists for this CPU
    """
    arch = target_arch()
    binary = is_installed(cfg)
    managed = is_managed(cfg)
    current = installed_version(cfg) if binary is not None else None
    target = runtime_dir(cfg)
    link = local_bin_dir() / BIN_NAME

    plan: Dict[str, Any] = {
        "ok": True,
        "action": "install",
        "arch": arch or str(platform.machine() or "unknown"),
        "installed": current,
        "installed_path": str(binary) if binary else None,
        "managed": managed,
        "latest": None,
        "url": "",
        "asset": "",
        "extractable": True,
        "zstd": zstd_support(),
        "size_bytes": 0,
        "size_gb": 0.0,
        "target_dir": str(target),
        "bin_link": str(link),
        "unit_path": str(unit_path()),
        "reason": "",
    }

    if arch is None:
        plan.update({
            "ok": False,
            "action": "unsupported",
            "reason": (
                f"Ollama publishes Linux tarballs for x86_64 and aarch64 only; this "
                f"machine reports {platform.machine() or 'an unknown architecture'}. "
                "Install ollama from your distribution instead."
            ),
        })
        plan.update(_space_check(target, 0))
        return plan

    # A binary on PATH that is not the one we manage belongs to the system (or
    # to the owner).  Installing a second copy would leave two versions and a
    # symlink war over ~/.local/bin.
    #
    # This deliberately does NOT also require "no managed tree exists".  Both
    # can be true at once — the owner installs a distro package after JARVIS
    # already provisioned one — and the external binary still wins, because it
    # is what PATH resolves and therefore what actually runs.  Gating on
    # `not managed` skipped this branch in exactly that case and fell through to
    # the release comparison, which then reported the EXTERNAL binary's version
    # as the state of the MANAGED install: "already the latest release" while a
    # stale tree sat in the data directory, unused and never upgraded.
    if binary is not None and _is_external(binary, cfg):
        reason = (
            f"ollama {current or '(version unknown)'} is already installed at "
            f"{binary}; JARVIS will use it rather than install a second copy."
        )
        if managed:
            reason += (
                f" The copy JARVIS installed under {target} is now unused and "
                f"can be deleted."
            )
        plan.update({"action": "external", "reason": reason})
        plan.update(_space_check(target, 0))
        return plan

    release = _release_json(timeout=timeout)
    latest = None
    if release:
        tag = str(release.get("tag_name") or "").strip()
        latest = parse_version(tag) or (tag.lstrip("vV") or None)
    plan["latest"] = latest

    if latest is None:
        plan["url"] = release_asset_url(current or "", arch) if current else ""
        if current:
            plan.update({
                "action": "unknown",
                "reason": (
                    f"ollama {current} is installed, but github.com could not be "
                    "reached to check for a newer release. Nothing to do while offline."
                ),
            })
        else:
            plan.update({
                "ok": False,
                "action": "unknown",
                "reason": (
                    "ollama is not installed and github.com could not be reached to "
                    "find a release. Reconnect and re-run the installer."
                ),
            })
        plan.update(_space_check(target, 0))
        return plan

    asset = _pick_asset(release, arch, latest)
    size = asset["size"] or _asset_size(asset["url"], timeout=timeout)
    plan["url"] = asset["url"]
    plan["asset"] = asset["name"]
    plan["extractable"] = asset["extractable"]
    plan["zstd"] = zstd_support()
    plan["size_bytes"] = size
    plan["size_gb"] = round(size / _GB, 2)

    # An intermediate .tar is needed when zstd has to be shelled out to, so the
    # archive, the tar and the unpacked tree are all on disk at once.
    factor = _INSTALL_SPACE_FACTOR
    if _suffix_of(asset["name"]) == ".tar.zst" and plan["zstd"] not in ("stdlib", None):
        factor += _INSTALL_SPACE_INTERMEDIATE
    # An unknown asset size still gets the margin checked: a disk with nothing
    # free must be refused up front rather than part-way through a write.
    required = (int(size * factor) if size else 0) + _INSTALL_SPACE_MARGIN
    plan.update(_space_check(target, required))

    if current and _same_version(current, latest):
        plan.update({
            "action": "current",
            "reason": f"ollama {current} is already the latest release.",
        })
        return plan
    if current:
        plan.update({
            "action": "upgrade",
            "reason": f"ollama {current} is installed; {latest} is available.",
        })
    else:
        plan.update({
            "action": "install",
            "reason": f"ollama is not installed; {latest} will be downloaded.",
        })
    if not plan["extractable"]:
        plan["ok"] = False
        plan["reason"] = (
            f"the only ollama {latest} asset for this machine is {asset['name']}, and "
            + ZSTD_HINT
        )
        return plan
    if not plan["enough_space"]:
        plan["ok"] = False
        plan["reason"] = _short_message(f"ollama {latest}", plan)
    return plan


def _same_version(left: Any, right: Any) -> bool:
    return str(left or "").strip().lstrip("vV") == str(right or "").strip().lstrip("vV")


def _is_external(binary: Path, cfg: Any = None) -> bool:
    """True when ``binary`` lives outside the tree and link this module owns."""
    try:
        resolved = Path(binary).resolve()
    except (OSError, ValueError):
        return True
    for owned in (runtime_dir(cfg), local_bin_dir()):
        try:
            resolved.relative_to(Path(owned).resolve())
            return False
        except (ValueError, OSError):
            continue
    return True


# --------------------------------------------------------------------------- #
#  Download
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, *, expected_size: int = 0,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None,
              timeout: float = DOWNLOAD_TIMEOUT) -> Dict[str, Any]:
    """Fetch ``url`` to ``dest``, resuming a partial file when the server allows.

    A half-downloaded tarball is kept, not deleted: re-running the installer on
    a flaky connection should continue rather than start the gigabyte again.
    Resumption is via a ``Range`` request; a server that ignores it answers 200
    and the file is rewritten from zero, which is correct if wasteful.
    """
    result: Dict[str, Any] = {
        "ok": False, "url": url, "path": str(dest), "bytes": 0,
        "total": int(expected_size or 0), "resumed": False, "reason": "",
    }
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["reason"] = f"could not create {dest.parent}: {exc}"
        return result

    have = 0
    try:
        if dest.is_file():
            have = dest.stat().st_size
    except OSError:
        have = 0

    if expected_size and have == expected_size:
        result.update({"ok": True, "bytes": have, "reason": "already downloaded"})
        _notify(progress, {"phase": "download", "completed": have,
                           "total": expected_size, "message": "already downloaded"})
        return result
    if expected_size and have > expected_size:
        # Longer than the asset: a truncated *previous* asset, or a mirror that
        # lied. Start over rather than reason about it.
        try:
            dest.unlink()
        except OSError:
            pass
        have = 0

    headers: Dict[str, str] = {}
    if have > 0:
        headers["Range"] = f"bytes={have}-"

    mode = "ab" if have > 0 else "wb"
    written = have
    try:
        with _urlopen(_request(url, headers=headers), timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 200)
            if have > 0 and status != 206:
                # Range refused: restart from the beginning.
                mode = "wb"
                written = 0
            else:
                result["resumed"] = have > 0
            total = result["total"]
            length = response.headers.get("Content-Length") if hasattr(response, "headers") else None
            if length:
                try:
                    total = int(length) + (written if status == 206 else 0)
                except (TypeError, ValueError):
                    pass
            result["total"] = total or result["total"]

            with open(dest, mode) as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    _notify(progress, {
                        "phase": "download",
                        "completed": written,
                        "total": result["total"],
                        "percent": round(written * 100.0 / result["total"], 1)
                        if result["total"] else None,
                        "message": "downloading ollama",
                    })
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        result["bytes"] = written
        result["reason"] = f"download failed: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - a download failure is a result, not a crash
        result["bytes"] = written
        result["reason"] = f"download failed: {type(exc).__name__}: {exc}"
        return result

    result["bytes"] = written
    if expected_size and written != expected_size:
        # A wrong-sized archive would fail to extract anyway, and leaving it in
        # place would poison every retry with a resume that can never complete.
        try:
            dest.unlink()
        except OSError:
            pass
        result["reason"] = (
            f"downloaded {written} bytes but the release says {expected_size}; "
            "the partial file was removed so a retry starts clean."
        )
        return result
    result["ok"] = True
    return result


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
def _safe_members(archive: Any, dest: Path) -> Iterator[Any]:
    """Members that stay inside ``dest``.

    Python 3.12 grew ``filter="data"`` for exactly this; on 3.9-3.11 the check
    has to be written out, because ``extractall`` there will happily follow an
    absolute path or a ``../`` out of the destination.
    """
    root = os.path.realpath(str(dest))
    for member in archive.getmembers():
        name = member.name or ""
        if name.startswith("/") or os.path.isabs(name) or ".." in Path(name).parts:
            log.warning("skipping unsafe tar entry %r", name)
            continue
        if member.issym() or member.islnk():
            target = os.path.realpath(os.path.join(root, os.path.dirname(name), member.linkname or ""))
            if not (target == root or target.startswith(root + os.sep)):
                log.warning("skipping tar link %r pointing outside the target", name)
                continue
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            continue
        yield member


def _discard(path: Path) -> None:
    """Delete a file that must not be reused, without caring if it is gone."""
    try:
        path.unlink()
    except OSError:
        log.debug("could not remove %s", path)


def _decompress_zstd(source: Path, target: Path, mechanism: str) -> Dict[str, Any]:
    """Turn a ``.tar.zst`` into a plain ``.tar`` at ``target``."""
    result: Dict[str, Any] = {"ok": False, "reason": ""}
    if mechanism == "zstandard":
        try:
            import zstandard
            decompressor = zstandard.ZstdDecompressor()
            with open(source, "rb") as raw, open(target, "wb") as plain:
                decompressor.copy_stream(raw, plain)
            result["ok"] = True
            return result
        except Exception as exc:  # noqa: BLE001 - a decompression failure is a result
            result["reason"] = f"zstandard could not decompress {source.name}: {exc}"
            return result
    if mechanism == "cli":
        exe = _which("zstd") or _which("unzstd")
        if not exe:  # pragma: no cover - zstd_support() already found one
            result["reason"] = "the zstd command disappeared between check and use."
            return result
        run = platform_utils.run_command(
            [exe, "-d", "-f", "-o", str(target), str(source)], timeout=600.0
        )
        if run.ok and target.is_file():
            result["ok"] = True
            return result
        result["reason"] = (
            f"zstd failed to decompress {source.name}: "
            + ((run.stderr or run.stdout or "").strip() or f"exit {run.returncode}")
        )
        return result
    result["reason"] = f"unknown zstd mechanism {mechanism!r}."
    return result


def _extract(archive_path: Path, dest: Path) -> Dict[str, Any]:
    """Unpack ``archive_path`` into ``dest`` (which must not already exist).

    Handles gzip and zstd.  ``tarfile`` reads ``.tar.zst`` directly on Python
    3.14+; on anything older the archive is decompressed to a temporary ``.tar``
    first, because the target laptop's interpreter is not necessarily ours.
    """
    result: Dict[str, Any] = {"ok": False, "path": str(dest), "reason": ""}
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        result["reason"] = f"could not create {dest}: {exc}"
        return result

    source = archive_path
    intermediate: Optional[Path] = None
    if _is_zstd(archive_path) and not _stdlib_zstd():
        mechanism = zstd_support()
        if mechanism is None:
            shutil.rmtree(str(dest), ignore_errors=True)
            result["reason"] = f"could not extract {archive_path.name}: {ZSTD_HINT}"
            return result
        intermediate = archive_path.with_name(archive_path.name + ".tar")
        unpacked = _decompress_zstd(archive_path, intermediate, mechanism)
        if not unpacked["ok"]:
            _discard(intermediate)
            shutil.rmtree(str(dest), ignore_errors=True)
            result["reason"] = f"could not extract {archive_path.name}: {unpacked['reason']}"
            return result
        source = intermediate

    try:
        with tarfile.open(str(source), "r:*") as archive:
            if hasattr(tarfile, "data_filter"):
                archive.extractall(path=str(dest), filter="data")  # nosec B202 - filtered
            else:  # pragma: no cover - Python < 3.12
                archive.extractall(path=str(dest),
                                   members=_safe_members(archive, dest))  # nosec B202
    except Exception as exc:  # noqa: BLE001 - a corrupt archive is a result
        # Leave nothing half-unpacked: a partial tree would make the next run
        # think ollama is installed.
        shutil.rmtree(str(dest), ignore_errors=True)
        result["reason"] = f"could not extract {archive_path.name}: {exc}"
        return result
    finally:
        if intermediate is not None:
            _discard(intermediate)
    result["ok"] = True
    return result


def _find_binary(root: Path) -> Optional[Path]:
    """Locate ``ollama`` in a freshly extracted tree."""
    for candidate in (root / "bin" / BIN_NAME, root / BIN_NAME):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    try:
        for found in root.rglob(BIN_NAME):
            if found.is_file():
                return found
    except OSError:
        pass
    return None


def _install_dir(staged: Path, dest: Path) -> Dict[str, Any]:
    """Move ``staged`` into place as ``dest``, keeping the old copy until it lands."""
    result: Dict[str, Any] = {"ok": False, "reason": ""}
    backup: Optional[Path] = None
    try:
        if dest.exists():
            backup = dest.with_name(f"{dest.name}.old-{os.getpid()}")
            shutil.rmtree(str(backup), ignore_errors=True)
            os.replace(str(dest), str(backup))
        os.replace(str(staged), str(dest))
    except OSError as exc:
        if backup is not None and not dest.exists():
            try:
                os.replace(str(backup), str(dest))
            except OSError:
                log.error("could not restore the previous ollama at %s", dest)
        result["reason"] = f"could not move the new install into {dest}: {exc}"
        return result
    if backup is not None:
        shutil.rmtree(str(backup), ignore_errors=True)
    result["ok"] = True
    return result


# --------------------------------------------------------------------------- #
#  The ~/.local/bin link
# --------------------------------------------------------------------------- #
def _link_marker(link: Path) -> Path:
    """Sidecar that records a *copy* we made, so a re-run may replace it."""
    return link.with_name(f".{link.name}.jarvis-managed")


def link_binary(binary: Path, *, link: Optional[Path] = None,
                cfg: Any = None) -> Dict[str, Any]:
    """Put ``ollama`` on ``PATH`` at ``~/.local/bin/ollama``.

    Refuses, loudly and without touching anything, if something is already at
    that path that JARVIS did not create.  That is the whole safety story for
    writing outside the repository: we replace our own symlink and our own
    recorded copy, and nothing else, ever.
    """
    target = Path(link) if link is not None else (local_bin_dir() / BIN_NAME)
    result: Dict[str, Any] = {
        "ok": False, "path": str(target), "mode": "none",
        "target": str(binary), "reason": "",
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["reason"] = f"could not create {target.parent}: {exc}"
        return result

    owned = False
    exists = False
    try:
        exists = target.exists() or target.is_symlink()
        if target.is_symlink():
            pointed = Path(os.path.realpath(str(target)))
            owned = not _is_external(pointed, cfg)
        elif exists:
            owned = _link_marker(target).is_file()
    except OSError as exc:
        result["reason"] = f"could not inspect {target}: {exc}"
        return result

    if exists and not owned:
        result.update({
            "mode": "conflict",
            "reason": (
                f"{target} already exists and was not created by JARVIS, so it has "
                "been left untouched. Remove it yourself, or add "
                f"{binary.parent} to PATH instead."
            ),
        })
        return result

    if exists:
        try:
            target.unlink()
        except OSError as exc:
            result["reason"] = f"could not replace {target}: {exc}"
            return result

    try:
        target.symlink_to(binary)
        result.update({"ok": True, "mode": "symlink"})
        # A symlink is self-describing; drop any stale copy marker.
        marker = _link_marker(target)
        if marker.is_file():
            marker.unlink()
        return result
    except (OSError, NotImplementedError, AttributeError) as exc:
        # Windows without developer mode, or a filesystem with no symlinks.
        log.debug("symlink %s -> %s failed (%s); copying instead", target, binary, exc)

    try:
        shutil.copy2(str(binary), str(target))
        _link_marker(target).write_text(
            f"copied from {binary}\n", encoding="utf-8"
        )
        try:
            os.chmod(str(target), 0o755)
        except OSError:
            pass
        result.update({"ok": True, "mode": "copy"})
    except OSError as exc:
        result["reason"] = f"could not place {target}: {exc}"
    return result


# --------------------------------------------------------------------------- #
#  Install
# --------------------------------------------------------------------------- #
def install(cfg: Any = None, *, progress: Optional[Callable[[Dict[str, Any]], None]] = None,
            force: bool = False, timeout: float = DOWNLOAD_TIMEOUT) -> Dict[str, Any]:
    """Install or upgrade the managed, rootless copy of Ollama.

    Idempotent: with ollama already current this returns immediately without
    touching the network.  Never raises — a failure at any stage comes back as
    ``ok=False`` with a ``reason``, and the filesystem is left in the state it
    was in before, so a retry is never poisoned by the attempt that failed.
    """
    plan = install_plan(cfg, timeout=min(timeout, API_TIMEOUT))
    action = plan["action"]
    result: Dict[str, Any] = dict(plan)
    result["installed_now"] = False
    result["link"] = None

    if action in ("external", "current") and not force:
        result["ok"] = True
        result["action"] = "skipped" if action == "external" else "current"
        _notify(progress, {"phase": "ollama", "message": plan["reason"]})
        return result
    if action == "unsupported":
        return result
    if action == "unknown" and not plan.get("latest"):
        return result
    if not plan.get("url"):
        result["ok"] = False
        result["reason"] = result["reason"] or "no download URL could be determined."
        return result
    if not plan.get("extractable", True):
        result["ok"] = False
        return result
    if not plan.get("enough_space", True):
        result["ok"] = False
        result["reason"] = _short_message(f"ollama {plan.get('latest')}", plan)
        return result

    version = str(plan.get("latest") or "")
    arch = plan.get("arch") or "amd64"

    # Checked before the download, not after: whether the target belongs to
    # somebody else is knowable from the filesystem alone, and finding out
    # afterwards costs a gigabyte and a half for a refusal we could have made
    # at second zero.
    target = runtime_dir(cfg)
    if target.exists() and not is_managed(cfg) and not force:
        result["ok"] = False
        result["action"] = "conflict"
        result["reason"] = (
            f"{target} exists but carries no {MANAGED_MARKER} marker, so JARVIS did "
            "not create it and will not replace it. Move it aside and re-run."
        )
        _notify(progress, {"phase": "ollama", "message": result["reason"]})
        return result

    # The version is in the file name so a stale partial from a previous
    # release is never resumed into the current one.
    asset_name = str(plan.get("asset") or f"ollama-linux-{arch}{DEFAULT_ASSET_SUFFIX}")
    archive = download_dir(cfg) / f"ollama-{version}-{asset_name}"

    _notify(progress, {"phase": "download", "completed": 0,
                       "total": plan.get("size_bytes", 0),
                       "message": f"downloading ollama {version} ({plan.get('size_gb')} GB)"})
    fetched = _download(plan["url"], archive, expected_size=int(plan.get("size_bytes") or 0),
                        progress=progress, timeout=timeout)
    result["download"] = fetched
    if not fetched["ok"]:
        result["ok"] = False
        result["reason"] = fetched["reason"]
        return result

    staged = target.with_name(f"{target.name}.new-{os.getpid()}")
    shutil.rmtree(str(staged), ignore_errors=True)
    _notify(progress, {"phase": "extract", "message": f"unpacking ollama {version}"})
    extracted = _extract(archive, staged)
    if not extracted["ok"]:
        # The archive is the right *length* or it would not have got here, so a
        # retry would resume nothing, re-extract the same corruption and fail
        # identically for ever. Delete it: re-downloading is the only way out.
        _discard(archive)
        result["ok"] = False
        result["reason"] = extracted["reason"] + " The archive was deleted so a retry re-fetches it."
        return result

    binary = _find_binary(staged)
    if binary is None:
        shutil.rmtree(str(staged), ignore_errors=True)
        _discard(archive)
        result["ok"] = False
        result["reason"] = (
            f"the {archive.name} tarball contained no '{BIN_NAME}' binary; the "
            "download may be a different asset than expected. The archive was "
            "deleted so a retry re-fetches it."
        )
        return result

    try:
        os.chmod(str(binary), 0o755)
    except OSError:
        log.debug("could not chmod %s", binary)
    try:
        (staged / MANAGED_MARKER).write_text(
            f"ollama {version} installed by jarvis.runtime.ollama\n", encoding="utf-8"
        )
    except OSError as exc:
        shutil.rmtree(str(staged), ignore_errors=True)
        result["ok"] = False
        result["reason"] = f"could not write the {MANAGED_MARKER} marker: {exc}"
        return result

    moved = _install_dir(staged, target)
    if not moved["ok"]:
        shutil.rmtree(str(staged), ignore_errors=True)
        result["ok"] = False
        result["reason"] = moved["reason"]
        return result

    final = managed_binary(cfg)
    if final is None:  # pragma: no cover - only if the move raced with something
        result["ok"] = False
        result["reason"] = f"ollama was unpacked but no binary is present under {target}."
        return result

    # The tarball has done its job.  Keeping it would leave 1.4 GB per release
    # accumulating in <data>/downloads for ever — nothing ever reads it again,
    # because the next run sees the installed version and returns "current"
    # before it looks at the download directory.
    _discard(archive)

    linked = link_binary(final, cfg=cfg)
    result["link"] = linked
    result.update({
        "ok": True,
        "installed_now": True,
        "installed": version,
        "installed_path": str(final),
        "managed": True,
        "reason": f"ollama {version} installed at {final}."
        + ("" if linked["ok"] else f" {linked['reason']}"),
    })
    _notify(progress, {"phase": "ollama", "message": result["reason"]})
    return result


# --------------------------------------------------------------------------- #
#  The server
# --------------------------------------------------------------------------- #
def host_url(cfg: Any = None) -> str:
    """Base URL of the Ollama API, scheme included."""
    raw = ""
    if cfg is not None:
        raw = str(getattr(getattr(cfg, "llm", None), "ollama_host", "") or "").strip()
    if not raw:
        raw = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        raw = DEFAULT_HOST
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def host_authority(cfg_or_url: Any = None) -> str:
    """``host:port`` — what ``OLLAMA_HOST`` wants, without the scheme."""
    url = cfg_or_url if isinstance(cfg_or_url, str) else host_url(cfg_or_url)
    if "://" not in url:
        url = "http://" + url
    parts = urllib.parse.urlsplit(url)
    return parts.netloc or "127.0.0.1:11434"


def _api_get(host: str, path: str, *, timeout: float = API_TIMEOUT) -> Dict[str, Any]:
    """GET a JSON endpoint.  Always returns a dict; never raises."""
    url = host.rstrip("/") + path
    try:
        with _urlopen(_request(url), timeout=timeout) as response:
            raw = response.read(8_000_000)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        return {"ok": False, "data": None, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return {"ok": True, "data": json.loads(text), "error": ""}
    except (ValueError, TypeError) as exc:
        return {"ok": False, "data": None, "error": f"malformed JSON from {url}: {exc}"}


def is_serving(host: Any = None, *, timeout: float = 3.0) -> bool:
    """True when an Ollama daemon answers on ``host``."""
    url = host if isinstance(host, str) else host_url(host)
    if "://" not in url:
        url = "http://" + url
    return bool(_api_get(url.rstrip("/"), "/api/tags", timeout=timeout).get("ok"))


#: The daemon this process started, if any.  Kept so :func:`stop_server` can
#: terminate and *reap* it rather than orphaning a 16 GB-resident process.
_SERVER: Optional[Any] = None


def start_server(cfg: Any = None, *, wait: float = 20.0,
                 timeout: float = 20.0) -> Dict[str, Any]:
    """Start ``ollama serve`` — via systemd when the unit is installed, else directly.

    The direct path is the one deliberate long-lived child in this codebase: a
    server that exits when the installer does would be useless.  It is tracked
    in ``_SERVER`` so :func:`stop_server` can kill *and wait for* it.
    """
    url = host_url(cfg)
    result: Dict[str, Any] = {"ok": False, "action": "none", "host": url, "reason": ""}

    if is_serving(url):
        result.update({"ok": True, "action": "already-running",
                       "reason": f"ollama is already serving on {url}."})
        return result

    systemctl = _which("systemctl") if platform_utils.IS_LINUX else None
    if systemctl and unit_path().is_file():
        started = platform_utils.run_command(
            [systemctl, "--user", "start", UNIT_NAME], timeout=timeout
        )
        if started.ok and _wait_until_serving(url, wait):
            result.update({"ok": True, "action": "systemd",
                           "reason": f"started {UNIT_NAME} via systemctl --user."})
            return result
        result["reason"] = (started.stderr or started.stdout or "").strip()

    binary = is_installed(cfg)
    if binary is None:
        result["reason"] = "ollama is not installed; run the installer first."
        return result

    env = dict(os.environ)
    env.setdefault("OLLAMA_HOST", host_authority(url))
    for key, value in _server_environment(cfg).items():
        env.setdefault(key, value)

    global _SERVER
    try:
        kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "env": env,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        _SERVER = subprocess.Popen([str(binary), "serve"], **kwargs)  # noqa: S603
    except (OSError, ValueError) as exc:
        result["reason"] = f"could not start {binary}: {exc}"
        return result

    if _wait_until_serving(url, wait):
        result.update({"ok": True, "action": "spawned",
                       "pid": getattr(_SERVER, "pid", None),
                       "reason": f"ollama serve started on {url}."})
        return result
    result["action"] = "spawned"
    result["pid"] = getattr(_SERVER, "pid", None)
    result["reason"] = (
        f"ollama serve was started but {url} did not answer within {wait:g}s. "
        "Check `journalctl --user -u " + UNIT_NAME + "` or run `ollama serve` by hand."
    )
    return result


def _wait_until_serving(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if is_serving(url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def stop_server(cfg: Any = None, *, timeout: float = 20.0) -> Dict[str, Any]:
    """Stop the daemon this process started, or the systemd user unit."""
    global _SERVER
    result: Dict[str, Any] = {"ok": False, "action": "none", "reason": ""}

    proc = _SERVER
    if proc is not None:
        _SERVER = None
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout)
            else:
                proc.wait(timeout=timeout)
            result.update({"ok": True, "action": "terminated",
                           "reason": "the ollama server started by JARVIS was stopped."})
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result["reason"] = f"could not stop the ollama process: {exc}"
            return result

    systemctl = _which("systemctl") if platform_utils.IS_LINUX else None
    if systemctl and unit_path().is_file():
        stopped = platform_utils.run_command(
            [systemctl, "--user", "stop", UNIT_NAME], timeout=timeout
        )
        result.update({
            "ok": bool(stopped.ok),
            "action": "systemd",
            "reason": (stopped.stderr or stopped.stdout or "").strip()
            or f"stopped {UNIT_NAME}.",
        })
        return result

    result["reason"] = "no ollama server started by JARVIS is running."
    return result


# --------------------------------------------------------------------------- #
#  The systemd user unit
# --------------------------------------------------------------------------- #
def _parallel_requests(cfg: Any = None) -> int:
    """How many generations Ollama may run at once.

    JARVIS runs a *tree* of subagents, and Ollama's default of one in-flight
    request per model serialises the whole tree behind its slowest branch.
    Mirrors ``llm.max_concurrent_requests`` so the two cannot drift; ``0``
    there means unlimited, which for a server setting has to become a number.
    """
    raw = getattr(getattr(cfg, "llm", None), "max_concurrent_requests", 4)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4
    if value <= 0:
        value = 8
    return max(1, min(value, 32))


def _max_loaded_models(cfg: Any = None) -> int:
    """How many models may sit in RAM at once.

    One, whenever the model is large: the default is a 27B at Q4, which is
    ~16 GB of the target machine's 32 GB. Holding two would swap, and swapping
    a language model is indistinguishable from a hang.
    """
    try:
        from ..llm import models as model_catalogue
    except ImportError:  # pragma: no cover - first-party module
        return 1
    tag = model_tag(cfg)
    try:
        spec = model_catalogue.resolve(tag, allow_unreleased=True)
        size = float(spec.quantised_size_gb or 0.0)
    except Exception:  # noqa: BLE001 - an unknown model just means "be careful"
        return 1
    if size and size < 6.0:
        return 2
    return 1


def _server_environment(cfg: Any = None) -> Dict[str, str]:
    """The environment that makes Ollama usable for a tree of subagents."""
    return {
        "OLLAMA_HOST": host_authority(cfg),
        "OLLAMA_NUM_PARALLEL": str(_parallel_requests(cfg)),
        "OLLAMA_MAX_LOADED_MODELS": str(_max_loaded_models(cfg)),
        # Re-reading 16 GB of weights from a laptop disk between utterances is a
        # minute of silence in a voice assistant.
        "OLLAMA_KEEP_ALIVE": "30m",
    }


def service_unit_text(cfg: Any = None, *, binary: Optional[Path] = None) -> str:
    """Render the systemd **user** unit for ``ollama serve``.

    A user unit, never a system one, for the same reason as
    :mod:`jarvis.linux.service`: no root, and the unit inherits the login
    session's environment.

    ``StartLimitIntervalSec``/``StartLimitBurst`` are in ``[Unit]``.  systemd
    moved them there in v230 and *silently ignores* them under ``[Service]``, so
    a unit that puts them in the wrong section respawns a crash-looping daemon
    for ever while looking perfectly correct.
    """
    exe = Path(binary) if binary is not None else (is_installed(cfg) or
                                                   (runtime_dir(cfg) / "bin" / BIN_NAME))
    env = _server_environment(cfg)
    return "\n".join([
        UNIT_MARKER,
        "# Re-running the installer rewrites this file, so hand edits are lost.",
        "# To change it durably, add a drop-in instead — that survives updates:",
        "#     systemctl --user edit jarvis-ollama.service",
        "# Delete this first line and the installer will treat the unit as",
        "# yours and refuse to touch it.",
        "[Unit]",
        "Description=Ollama inference server for JARVIS",
        "Documentation=https://github.com/ollama/ollama",
        "After=network-online.target",
        "Wants=network-online.target",
        "# StartLimit* belong to [Unit]: systemd moved them out of [Service] in",
        "# v230 and ignores them there without a word, which turns a crash loop",
        "# into an infinite one.",
        "StartLimitIntervalSec=120",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        # as_posix(), not str(): systemd parses POSIX paths, and a unit rendered
        # on a Windows development box would otherwise carry backslashes.
        f'ExecStart="{Path(exe).as_posix()}" serve',
        "Restart=always",
        "RestartSec=5",
        "TimeoutStopSec=30",
        "# A tree of subagents issues concurrent requests; Ollama's default of one",
        "# in-flight request per model would serialise the whole tree.",
        f"Environment=OLLAMA_NUM_PARALLEL={env['OLLAMA_NUM_PARALLEL']}",
        f"Environment=OLLAMA_MAX_LOADED_MODELS={env['OLLAMA_MAX_LOADED_MODELS']}",
        f"Environment=OLLAMA_KEEP_ALIVE={env['OLLAMA_KEEP_ALIVE']}",
        f"Environment=OLLAMA_HOST={env['OLLAMA_HOST']}",
        "",
        "[Install]",
        "# default.target, not graphical-session.target: with lingering enabled",
        "# (loginctl enable-linger $USER) the server survives logout.",
        "WantedBy=default.target",
        "",
    ])


def install_service(cfg: Any = None, *, enable: bool = True,
                    timeout: float = 20.0) -> Dict[str, Any]:
    """Write (and optionally enable) the systemd user unit.  Idempotent."""
    result: Dict[str, Any] = {"ok": False, "path": str(unit_path()),
                              "written": False, "enabled": False, "reason": ""}
    text = service_unit_text(cfg)
    path = unit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if existing and not existing.lstrip().startswith(UNIT_MARKER):
            # Someone else's unit under our filename. Overwriting it would be
            # the one thing this installer promises never to do, and a systemd
            # unit is precisely the kind of file a person tunes by hand and
            # then forgets about.
            result.update({
                "ok": False,
                "action": "conflict",
                "reason": (
                    f"{path} exists but was not written by JARVIS, so it was "
                    f"left untouched. Move it aside and re-run, or delete the "
                    f"unit if you no longer want it."
                ),
            })
            return result
        if existing != text:
            path.write_text(text, encoding="utf-8")
            result["written"] = True
    except OSError as exc:
        result["reason"] = f"could not write {path}: {exc}"
        return result
    result["ok"] = True
    result["reason"] = f"unit {'written' if result['written'] else 'already current'} at {path}."

    if not enable:
        return result
    systemctl = _which("systemctl") if platform_utils.IS_LINUX else None
    if not systemctl:
        result["reason"] += " systemctl is unavailable, so it was not enabled."
        return result
    reload_result = platform_utils.run_command(
        [systemctl, "--user", "daemon-reload"], timeout=timeout
    )
    if not reload_result.ok:
        result["reason"] += " systemctl --user daemon-reload failed."
        return result
    enabled = platform_utils.run_command(
        [systemctl, "--user", "enable", "--now", UNIT_NAME], timeout=timeout
    )
    result["enabled"] = bool(enabled.ok)
    if not enabled.ok:
        result["reason"] += (
            " systemctl --user enable failed: "
            + ((enabled.stderr or enabled.stdout or "").strip() or "no output")
        )
    return result


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #
def normalise_tag(tag: Any) -> str:
    """``qwen3.8`` -> ``qwen3.8:latest``; Ollama's own default, made explicit."""
    text = str(tag or "").strip()
    if not text:
        return ""
    return text if ":" in text else f"{text}:latest"


def model_tag(cfg: Any = None) -> str:
    """The Ollama tag JARVIS should be running."""
    llm = getattr(cfg, "llm", None)
    tag = str(getattr(llm, "ollama_model", "") or "").strip()
    if tag:
        return tag
    try:
        from ..llm import models as model_catalogue
        spec = model_catalogue.resolve(
            str(getattr(llm, "model", "") or ""), allow_unreleased=True
        )
        if spec.ollama_tag:
            return spec.ollama_tag
    except Exception:  # noqa: BLE001 - fall through to the catalogue default
        log.debug("could not derive an ollama tag from llm.model")
    try:
        from ..llm import models as model_catalogue
        return model_catalogue.KNOWN_MODELS[model_catalogue.DEFAULT_ALIAS].ollama_tag
    except Exception:  # noqa: BLE001
        # Last resort only, for a catalogue too broken to read. Kept in step
        # with LLMConfig.ollama_model by test_default_model.py, because a stale
        # literal here silently pulls the wrong (previous) model.
        return "qwen3.8:27b"


def list_models(host: Any = None, *, timeout: float = API_TIMEOUT) -> List[Dict[str, Any]]:
    """Everything ``GET /api/tags`` reports; ``[]`` when the server is unreachable."""
    url = host if isinstance(host, str) else host_url(host)
    if "://" not in url:
        url = "http://" + url
    answer = _api_get(url.rstrip("/"), "/api/tags", timeout=timeout)
    if not answer["ok"] or not isinstance(answer["data"], dict):
        return []
    entries = answer["data"].get("models")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _model_size_bytes(tag: str) -> int:
    """Expected download size for ``tag``, from the catalogue; ``0`` if unknown."""
    try:
        from ..llm import models as model_catalogue
        spec = model_catalogue.resolve(tag, allow_unreleased=True)
        return int(float(spec.quantised_size_gb or 0.0) * _GB)
    except Exception:  # noqa: BLE001 - an unknown tag simply has no known size
        return 0


def pull_plan(tag: Any = None, host: Any = None, *,
              cfg: Any = None, timeout: float = API_TIMEOUT) -> Dict[str, Any]:
    """Whether ``tag`` is present, how big it is, and whether it will fit.

    Reports the size *before* anything starts downloading, which is the entire
    point: 16 GB that does not fit is worth knowing about at second zero.
    """
    wanted = normalise_tag(tag or model_tag(cfg))
    url = host if isinstance(host, str) else host_url(host if host is not None else cfg)
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")

    answer = _api_get(url, "/api/tags", timeout=timeout)
    cache = models_cache_dir()
    plan: Dict[str, Any] = {
        "ok": True,
        "action": "pull",
        "tag": wanted,
        "host": url,
        "server": bool(answer["ok"]),
        "present": False,
        "present_size_bytes": 0,
        "size_bytes": _model_size_bytes(wanted),
        "cache_dir": str(cache),
        "reason": "",
    }
    plan["size_gb"] = round(plan["size_bytes"] / _GB, 2)

    required = (
        int(plan["size_bytes"] * _MODEL_SPACE_FACTOR) + _MODEL_SPACE_MARGIN
        if plan["size_bytes"] else 0
    )
    plan.update(_space_check(cache, required))
    plan["tag"] = wanted  # _space_check does not touch it, but be explicit

    if not answer["ok"]:
        plan.update({
            "ok": False,
            "action": "no-server",
            "reason": (
                f"no Ollama server answered on {url} ({answer['error']}). "
                "Start it with `ollama serve`, or install the systemd user service."
            ),
        })
        return plan

    for entry in list_models(url, timeout=timeout):
        name = normalise_tag(entry.get("name") or entry.get("model") or "")
        if name == wanted:
            try:
                plan["present_size_bytes"] = int(entry.get("size") or 0)
            except (TypeError, ValueError):
                plan["present_size_bytes"] = 0
            plan.update({
                "action": "present",
                "present": True,
                "reason": f"{wanted} is already pulled "
                          f"({round(plan['present_size_bytes'] / _GB, 2)} GB).",
            })
            return plan

    if not plan["enough_space"]:
        plan["ok"] = False
        plan["reason"] = _short_message(wanted, plan)
        return plan
    plan["reason"] = (
        f"{wanted} is not present; pulling it will download about "
        f"{plan['size_gb']} GB."
        if plan["size_bytes"] else
        f"{wanted} is not present and its size is not in the catalogue."
    )
    return plan


class NDJSONStream:
    """Incremental newline-delimited JSON reader for Ollama's streaming APIs.

    Ollama emits one JSON object per line, but chunk boundaries land wherever
    the network decides — routinely mid-object, and routinely mid-UTF-8-
    character.  Anything after the last newline is held back until the rest of
    it arrives, and the bytes are decoded incrementally so a split multi-byte
    character is not turned into two replacement characters.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._decoder: Any = None

    def _decode(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            return str(chunk)
        if self._decoder is None:
            import codecs
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        return self._decoder.decode(bytes(chunk))

    def feed(self, chunk: Any) -> List[Dict[str, Any]]:
        """Add a chunk; return the objects that are now complete."""
        self._buffer += self._decode(chunk)
        out: List[Dict[str, Any]] = []
        while True:
            index = self._buffer.find("\n")
            if index < 0:
                break
            line = self._buffer[:index]
            self._buffer = self._buffer[index + 1:]
            obj = _json_object(line)
            if obj is not None:
                out.append(obj)
        return out

    def flush(self) -> List[Dict[str, Any]]:
        """Objects held in a trailing line with no newline after it."""
        line, self._buffer = self._buffer, ""
        obj = _json_object(line)
        return [obj] if obj is not None else []


def _json_object(line: str) -> Optional[Dict[str, Any]]:
    text = (line or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        log.debug("ignoring unparsable NDJSON line: %.80s", text)
        return None
    return value if isinstance(value, dict) else None


def pull(tag: Any, *, host: Any = None, cfg: Any = None,
         progress: Optional[Callable[[Dict[str, Any]], None]] = None,
         timeout: float = PULL_TIMEOUT) -> Dict[str, Any]:
    """Stream ``POST /api/pull`` and report progress.  Never raises.

    Ollama's pull is itself resumable — an interrupted download continues from
    the blobs already on disk — so a failed call here costs a retry, not the
    16 GB.
    """
    wanted = normalise_tag(tag or model_tag(cfg))
    url = host if isinstance(host, str) else host_url(host if host is not None else cfg)
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")

    result: Dict[str, Any] = {
        "ok": False, "tag": wanted, "host": url, "status": "",
        "completed": 0, "total": 0, "events": 0, "error": "", "reason": "",
        # Peak bytes transferred for any single layer. Ollama reports progress
        # per layer and emits no ``completed`` at all for layers it already
        # holds, so this stays 0 exactly when the pull moved no data — which is
        # how a refresh tells "already current" from "actually updated".
        "downloaded": 0,
    }
    if not wanted:
        result["reason"] = "no model tag was given."
        return result

    body = json.dumps({"model": wanted, "stream": True}).encode("utf-8")
    request = _request(url + "/api/pull", method="POST", data=body,
                       headers={"Content-Type": "application/json"})
    stream = NDJSONStream()

    def absorb(objects: Sequence[Dict[str, Any]]) -> None:
        for obj in objects:
            result["events"] += 1
            error = obj.get("error")
            if error:
                result["error"] = str(error)
                continue
            status = obj.get("status")
            if status:
                result["status"] = str(status)
            for key in ("completed", "total"):
                if key in obj:
                    try:
                        result[key] = int(obj[key] or 0)
                    except (TypeError, ValueError):
                        pass
            if result["completed"] > result["downloaded"]:
                result["downloaded"] = result["completed"]
            _notify(progress, {
                "phase": "pull",
                "tag": wanted,
                "status": result["status"],
                "completed": result["completed"],
                "total": result["total"],
                "percent": round(result["completed"] * 100.0 / result["total"], 1)
                if result["total"] else None,
            })

    try:
        with _urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                absorb(stream.feed(chunk))
            absorb(stream.flush())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        result["reason"] = f"pull failed: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - a pull failure is a result
        result["reason"] = f"pull failed: {type(exc).__name__}: {exc}"
        return result

    if result["error"]:
        result["reason"] = f"ollama refused to pull {wanted}: {result['error']}"
        return result
    if result["status"].lower() != "success" and result["events"] == 0:
        result["reason"] = f"ollama returned no progress for {wanted}."
        return result
    result["ok"] = True
    result["reason"] = f"{wanted} pulled ({result['status'] or 'done'})."
    return result


def ensure_model(cfg: Any = None, *, tag: Any = None, host: Any = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                 timeout: float = PULL_TIMEOUT,
                 refresh: bool = False) -> Dict[str, Any]:
    """Make sure the configured model is pulled.  The one install.sh calls.

    With ``refresh=False`` this is idempotent by construction: the second call
    sees the tag in ``/api/tags`` and returns ``action="present"`` without a
    byte of network traffic beyond that one query.

    ``refresh=True`` re-issues the pull even when the tag is already present,
    which is what makes re-running the installer genuinely update the weights
    rather than merely confirm that *some* version of them exists. A tag is a
    moving target — ``qwen3.8:27b`` is re-pointed upstream when the model is
    re-quantised or fixed — so "present" and "current" are not the same claim.

    The cost of a redundant refresh is small: Ollama compares digests and
    re-downloads only the layers that changed, so a current model transfers
    nothing but the manifest. That is also how the outcome is classified —
    ``action="updated"`` when bytes actually moved, ``"current"`` when none did.
    """
    wanted = normalise_tag(tag or model_tag(cfg))
    plan = pull_plan(wanted, host, cfg=cfg)
    result: Dict[str, Any] = dict(plan)

    if plan["action"] == "present" and not refresh:
        result["ok"] = True
        _notify(progress, {"phase": "model", "message": plan["reason"]})
        return result
    if plan["action"] == "no-server" or not plan["ok"]:
        return result

    if plan["action"] == "present":
        # Already on disk, but asked to check upstream for a newer build.
        _notify(progress, {"phase": "model",
                           "message": f"checking {wanted} for updates"})
        refreshed = pull(wanted, host=plan["host"], cfg=cfg,
                         progress=progress, timeout=timeout)
        moved = int(refreshed.get("downloaded") or 0)
        if not refreshed["ok"]:
            # A refresh that cannot reach the registry is not a failure: the
            # model is still present and usable, which is what was asked for.
            result["ok"] = True
            result["action"] = "present"
            result["reason"] = (
                f"{wanted} is present; could not check for updates "
                f"({refreshed['reason']})"
            )
            return result
        result.update({
            "ok": True,
            "action": "updated" if moved else "current",
            "reason": (f"{wanted} updated ({moved / 1e9:.1f} GB transferred)."
                       if moved else f"{wanted} is already up to date."),
            "pull": refreshed,
        })
        return result

    _notify(progress, {"phase": "model",
                       "message": f"pulling {wanted} (~{plan['size_gb']} GB)"})
    pulled = pull(wanted, host=plan["host"], cfg=cfg, progress=progress, timeout=timeout)
    result.update({
        "ok": pulled["ok"],
        "action": "pulled" if pulled["ok"] else "failed",
        "reason": pulled["reason"],
        "pull": pulled,
    })
    return result


def status(cfg: Any = None, *, timeout: float = API_TIMEOUT) -> Dict[str, Any]:
    """Read-only picture of the runtime, for ``jarvis doctor`` and the installer."""
    binary = is_installed(cfg)
    url = host_url(cfg)
    serving = is_serving(url, timeout=min(timeout, 3.0))
    tag = normalise_tag(model_tag(cfg))
    present = False
    models: List[Dict[str, Any]] = []
    if serving:
        models = list_models(url, timeout=timeout)
        present = any(
            normalise_tag(entry.get("name") or entry.get("model") or "") == tag
            for entry in models
        )
    return {
        "installed": binary is not None,
        "path": str(binary) if binary else None,
        "version": installed_version(cfg) if binary else None,
        "managed": is_managed(cfg),
        "runtime_dir": str(runtime_dir(cfg)),
        "bin_link": str(local_bin_dir() / BIN_NAME),
        "host": url,
        "serving": serving,
        "unit_path": str(unit_path()),
        "unit_installed": unit_path().is_file(),
        "model": tag,
        "model_present": present,
        "models": [str(entry.get("name") or entry.get("model") or "") for entry in models],
        "models_cache_dir": str(models_cache_dir()),
    }


__all__ = [
    "GITHUB_REPO", "GITHUB_API_LATEST", "GITHUB_DOWNLOAD", "BIN_NAME",
    "MANAGED_MARKER", "UNIT_NAME", "DEFAULT_HOST", "NDJSONStream",
    "home_dir", "local_bin_dir", "runtime_dir", "download_dir", "managed_binary",
    "unit_dir", "unit_path", "models_cache_dir", "free_bytes",
    "is_installed", "is_managed", "installed_version", "parse_version",
    "target_arch", "latest_release", "release_asset_url",
    "install_plan", "install", "link_binary",
    "host_url", "host_authority", "is_serving", "start_server", "stop_server",
    "service_unit_text", "install_service",
    "normalise_tag", "model_tag", "list_models", "pull_plan", "pull",
    "ensure_model", "status",
]
