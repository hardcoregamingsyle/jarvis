"""Hardware detection: what this machine actually has, told honestly.

JARVIS is meant to run "everywhere" — a bare CPU laptop, an NVIDIA gaming
rig, an AMD ROCm box, a Google Cloud TPU VM, an Apple Silicon Mac — and the
one thing every one of those environments deserves is a detector that does
not lie to it. This module never claims an accelerator is usable when this
codebase cannot actually drive it (see the TPU note below), and it never
raises: probing hardware on a machine nobody has seen before means every
probe is optimistic and every failure is just "not detected".

Two facts worth internalising before touching this file:

* **AMD is a reporting/package-selection distinction, not a device-string
  one.** A ROCm build of PyTorch places tensors with ``torch.device("cuda")``
  — the exact same API NVIDIA uses. There is no ``torch.device("rocm")``.
  ``HardwareProfile.accelerator`` reports ``"rocm"`` for an AMD GPU purely so
  callers can pick the right *package* (e.g. Ollama's ROCm build) and show
  the right label; a backend that then needs an actual torch device for that
  profile must still pass ``"cuda"``. Get this backwards and AMD GPU usage
  breaks silently. See :func:`detect_amd` for the detection details.

* **A detected TPU is not a usable TPU, here.** Detection is real (env vars
  GCP/Colab set, plus a lazy import of ``torch_xla``/``jax``), but none of
  ``jarvis/llm/*_backend.py`` places a single tensor on an XLA device today —
  that needs PyTorch-XLA or JAX wired into a new backend, which does not
  exist in this codebase. :func:`detect` always attaches a note saying so
  and callers must fall back to CPU rather than pretend acceleration exists.

Every probe below is bounded (subprocess calls carry a short timeout),
logged rather than printed, and returns ``None``/a safe default instead of
raising — see the module-level rules in this project's contribution notes.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import platform_utils

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  The result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardwareProfile:
    """A best-effort, honest snapshot of this machine's compute resources.

    ``accelerator`` is one of ``"cuda"`` (NVIDIA), ``"rocm"`` (AMD),
    ``"mps"`` (Apple Silicon), ``"tpu"`` (Google — detected but NOT usable by
    any backend in this codebase yet, see the module docstring), or
    ``"none"``. It is a *reporting* label, not always a literal torch device
    string: for ``"rocm"`` specifically, the torch device to actually use is
    still ``"cuda"`` (a ROCm PyTorch build has no ``"rocm"`` device type).
    Callers that turn this into a torch device must special-case that.

    ``gpu_vendor`` is the separate "who makes this chip" fact used for
    reporting and *package* selection (e.g. installing Ollama's ROCm build
    for ``"amd"``): one of ``"nvidia"``, ``"amd"``, ``"apple"``, ``"google"``,
    or ``""`` when there is no accelerator.

    ``vram_gb`` is ``0.0`` both when there is genuinely no separate VRAM pool
    (Apple Silicon's unified memory) and when an accelerator was detected but
    its memory size could not be determined (a weak AMD signal with no
    ``rocm-smi``/torch available) — check ``notes`` to tell the two apart.
    """

    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    accelerator: str
    gpu_vendor: str
    gpu_name: str
    gpu_count: int
    vram_gb: float
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view for CLI output and JSON."""
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "ram_gb": self.ram_gb,
            "accelerator": self.accelerator,
            "gpu_vendor": self.gpu_vendor,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "vram_gb": self.vram_gb,
            "notes": list(self.notes),
        }


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    """Call ``fn()``, swallowing any exception and returning ``default`` instead.

    Every detector in this module is documented to never raise on its own,
    but :func:`detect` wraps each call anyway — a monkeypatched test, a
    future edit, or a genuinely bizarre host should degrade this module, not
    take the caller down with it.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a probe must never propagate
        logger.debug("%s raised %s: %s", getattr(fn, "__name__", fn), type(exc).__name__, exc)
        return default


# --------------------------------------------------------------------------- #
#  CPU / RAM
# --------------------------------------------------------------------------- #
def cpu_info() -> Tuple[int, int]:
    """Best-effort ``(physical cores, logical threads)`` for this machine.

    ``os.cpu_count()`` only ever reports logical processors, so it is used
    both as the threads figure and as the cores fallback — meaning
    hyperthreading/SMT is invisible without ``psutil``. When ``psutil`` is
    importable its physical count is preferred for ``cores``. Never raises;
    worst case returns ``(1, 1)``.
    """
    threads = 0
    try:
        threads = int(os.cpu_count() or 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("os.cpu_count() failed: %s", exc)
        threads = 0
    if threads <= 0:
        threads = 1

    cores = threads
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        logger.debug("importing psutil failed: %s", exc)
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
            if physical:
                cores = int(physical)
            if logical:
                threads = int(logical)
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil.cpu_count() failed: %s", exc)

    return max(cores, 1), max(threads, 1)


def _ram_gb_proc_meminfo(path: Any = Path("/proc/meminfo")) -> Optional[float]:
    """Parse ``MemTotal`` out of ``/proc/meminfo`` (Linux). ``path`` is
    overridable for tests."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kib = float(parts[1])
                except ValueError:
                    return None
                return round(kib * 1024.0 / 1e9, 2)
    return None


def _ram_gb_windows_ctypes() -> Optional[float]:
    """Total physical RAM via ``GlobalMemoryStatusEx`` — stdlib-only, Windows.

    The correct stdlib-only approach when ``psutil`` is absent: no extra
    dependency, no subprocess, just the Win32 API via ``ctypes``.
    """
    try:
        import ctypes
    except ImportError:  # pragma: no cover - ctypes is stdlib
        return None

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.ullTotalPhys / 1e9, 2)
    except (AttributeError, OSError, ValueError) as exc:
        logger.debug("GlobalMemoryStatusEx fallback failed: %s", exc)
        return None


def _ram_gb_sysctl() -> Optional[float]:
    """Total physical RAM via ``sysctl -n hw.memsize`` (macOS)."""
    try:
        res = platform_utils.run_command(["sysctl", "-n", "hw.memsize"], timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sysctl hw.memsize failed: %s", exc)
        return None
    if not res.ok:
        return None
    try:
        return round(float(res.stdout.strip()) / 1e9, 2)
    except ValueError:
        return None


def system_ram_gb() -> float:
    """Total system RAM in GB (10^9 bytes). ``0.0`` if it truly cannot be found.

    Prefers ``psutil`` when importable; otherwise falls back to a stdlib-only
    method per platform: ``/proc/meminfo`` on Linux, ``GlobalMemoryStatusEx``
    via ``ctypes`` on Windows, ``sysctl`` on macOS. Never raises.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        logger.debug("importing psutil failed: %s", exc)
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            total = getattr(psutil.virtual_memory(), "total", None)
            if total:
                return round(float(total) / 1e9, 2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil.virtual_memory() failed: %s", exc)

    try:
        if platform_utils.IS_LINUX:
            value = _ram_gb_proc_meminfo()
        elif platform_utils.IS_WINDOWS:
            value = _ram_gb_windows_ctypes()
        elif platform_utils.IS_MAC:
            value = _ram_gb_sysctl()
        else:
            value = None
    except Exception as exc:  # noqa: BLE001
        logger.debug("stdlib RAM fallback failed: %s", exc)
        value = None

    return float(value) if value else 0.0


# --------------------------------------------------------------------------- #
#  NVIDIA
# --------------------------------------------------------------------------- #
def _parse_nvidia_smi_csv(text: str) -> List[Tuple[str, float]]:
    """Robustly parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,
    noheader,nounits`` output: one ``name, vram_mib`` pair per GPU, tolerant
    of blank lines, extra whitespace, or a stray header line."""
    gpus: List[Tuple[str, float]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram_mib = float(parts[1])
        except ValueError:
            continue
        if not name:
            continue
        gpus.append((name, vram_mib))
    return gpus


def _detect_nvidia_smi() -> Optional[Dict[str, Any]]:
    if not platform_utils.which("nvidia-smi"):
        return None
    try:
        res = platform_utils.run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout=5.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi invocation failed: %s", exc)
        return None
    if not res.ok:
        return None
    gpus = _parse_nvidia_smi_csv(res.stdout)
    if not gpus:
        return None
    # Multiple cards: vram_gb is the SUM (aggregate capacity this machine
    # offers); count says how many there are. A caller placing one unsharded
    # model still needs to fit on a single card's worth, not the sum.
    total_vram_gb = sum(v for _, v in gpus) * 1_048_576.0 / 1e9  # MiB -> GB(1e9)
    return {"name": gpus[0][0], "vram_gb": round(total_vram_gb, 2), "count": len(gpus)}


def _detect_nvidia_torch() -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        # CRITICAL: a ROCm build of torch also answers True to
        # torch.cuda.is_available() (it reuses the CUDA API surface). Only
        # trust this path as NVIDIA when it is NOT a ROCm build — otherwise
        # an AMD GPU would be misreported as NVIDIA. detect_amd() owns the
        # ROCm case.
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        if hip_version:
            return None
        if not torch.cuda.is_available():
            return None
        count = int(torch.cuda.device_count())
        if count <= 0:
            return None
        total_bytes = 0
        name = ""
        for idx in range(count):
            try:
                props = torch.cuda.get_device_properties(idx)
                total_bytes += int(getattr(props, "total_memory", 0) or 0)
                if not name:
                    name = str(getattr(props, "name", "") or "")
            except Exception as exc:  # noqa: BLE001
                logger.debug("torch.cuda.get_device_properties(%s) failed: %s", idx, exc)
        if not name:
            name = "NVIDIA GPU (via torch)"
        return {"name": name, "vram_gb": round(total_bytes / 1e9, 2), "count": count}
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.cuda probe failed: %s", exc)
        return None


def detect_nvidia() -> Optional[Dict[str, Any]]:
    """Probe for an NVIDIA GPU, honestly.

    Tries ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,
    nounits`` first — present on both Windows and Linux once the NVIDIA
    driver is installed, independent of whether PyTorch/CUDA are. Falls back
    to a lazy-imported ``torch.cuda`` only when ``nvidia-smi`` is absent or
    unusable.

    Returns ``{"name", "vram_gb", "count"}`` or ``None``. Never raises.
    """
    result = _detect_nvidia_smi()
    if result is not None:
        return result
    return _detect_nvidia_torch()


# --------------------------------------------------------------------------- #
#  AMD / ROCm
# --------------------------------------------------------------------------- #
def _rocm_sysfs_present() -> bool:
    """Weak presence signal for a ROCm install: no VRAM number, just "is it here"."""
    for candidate in ("/sys/class/kfd/kfd", "/opt/rocm"):
        try:
            if Path(candidate).exists():
                return True
        except OSError:
            continue
    return False


# AMD GPU families that ROCm has never supported, or dropped long ago. A card
# from one of these is real hardware that genuinely cannot run a language
# model, and reporting it as an "accelerator" is worse than reporting nothing:
# it makes the planner size a model for VRAM that will never be used, and it
# tells the user they have acceleration they do not have.
#
# These are the low-end mobile GCN 1.0/2.0 parts still common in 2019-2020
# laptops (the R5 M230 in the target machine among them). ROCm requires GFX8+
# in practice, and Ollama's ROCm build refuses them outright.
_ROCM_UNSUPPORTED_MARKERS = (
    "r5 m2", "r7 m2", "r5 m3", "r7 m3",     # M230/M260/M330/M340/M360...
    "radeon 520", "radeon 530", "radeon 535",
    "radeon 610", "radeon 620", "radeon 625",
    "hd 8", "r5 graphics", "r4 graphics",
    "oland", "hainan", "mars", "exo", "sun ",
    "caicos", "turks", "cedar", "redwood", "juniper",
)


def rocm_supports(gpu_name: Any) -> bool:
    """False when ``gpu_name`` is an AMD part ROCm cannot drive.

    Conservative: an unrecognised card is assumed supported, because a false
    negative here would disable a GPU that works. Only the families that are
    definitively unsupported are listed.
    """
    name = str(gpu_name or "").strip().lower()
    if not name:
        return True
    return not any(marker in name for marker in _ROCM_UNSUPPORTED_MARKERS)


_ROCM_VRAM_PREFIX = "VRAM Total Memory (B)"


def _parse_rocm_vram_bytes(text: str) -> List[int]:
    """Extract each GPU's total VRAM in bytes from ``rocm-smi --showmeminfo
    vram`` output — one ``... VRAM Total Memory (B): N`` line per card.
    Deliberately line-oriented and tolerant of surrounding columns/whitespace
    because the exact layout has shifted across ROCm releases."""
    totals: List[int] = []
    for line in (text or "").splitlines():
        if _ROCM_VRAM_PREFIX not in line:
            continue
        tail = line.split(":")[-1].strip()
        try:
            totals.append(int(tail))
        except ValueError:
            continue
    return totals


def _detect_rocm_smi() -> Optional[Dict[str, Any]]:
    if not platform_utils.IS_LINUX:
        return None
    if not platform_utils.which("rocm-smi"):
        return None
    try:
        res = platform_utils.run_command(["rocm-smi", "--showmeminfo", "vram"], timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("rocm-smi invocation failed: %s", exc)
        return None
    if not res.ok:
        return None
    totals = _parse_rocm_vram_bytes(res.stdout)
    if not totals:
        return None
    return {
        "name": "AMD GPU (ROCm)",
        "vram_gb": round(sum(totals) / 1e9, 2),
        "count": len(totals),
    }


def _detect_rocm_sysfs() -> Optional[Dict[str, Any]]:
    if not platform_utils.IS_LINUX:
        return None
    if not _rocm_sysfs_present():
        return None
    return {"name": "AMD GPU (ROCm runtime present, VRAM unknown)", "vram_gb": 0.0, "count": 1}


def _detect_amd_torch() -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except ImportError:
        return _detect_rocm_sysfs()
    try:
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        if not hip_version:
            return _detect_rocm_sysfs()
        if not torch.cuda.is_available():
            return _detect_rocm_sysfs()
        count = int(torch.cuda.device_count())
        if count <= 0:
            return _detect_rocm_sysfs()
        total_bytes = 0
        name = ""
        for idx in range(count):
            try:
                props = torch.cuda.get_device_properties(idx)
                total_bytes += int(getattr(props, "total_memory", 0) or 0)
                if not name:
                    name = str(getattr(props, "name", "") or "")
            except Exception as exc:  # noqa: BLE001
                logger.debug("torch.cuda.get_device_properties(%s) failed: %s", idx, exc)
        if not name:
            name = "AMD GPU (ROCm, via torch)"
        return {"name": name, "vram_gb": round(total_bytes / 1e9, 2), "count": count}
    except Exception as exc:  # noqa: BLE001
        logger.debug("ROCm torch probe failed: %s", exc)
        return _detect_rocm_sysfs()


def detect_amd() -> Optional[Dict[str, Any]]:
    """Probe for an AMD GPU usable via ROCm, honestly.

    CRITICAL, and easy to get backwards: a ROCm build of PyTorch places
    tensors with ``torch.device("cuda")`` — the *same* API NVIDIA uses. There
    is no ``torch.device("rocm")``. So "AMD" here is a *reporting* and
    *package-selection* distinction only — once a caller has decided the
    vendor is AMD, the actual torch device string a ROCm-aware backend must
    pass is still ``"cuda"``. Writing ``torch.device(profile.accelerator)``
    naively when ``accelerator == "rocm"`` is a bug: that string does not
    exist as a torch device type and would raise.

    Detection order, first hit wins: ``rocm-smi --showmeminfo vram`` (Linux;
    gives real per-card VRAM), a lazy-imported torch checking
    ``torch.version.hip is not None`` (true only for a ROCm build — a CUDA or
    CPU-only build has ``hip is None``; also gives real VRAM via the same
    ``torch.cuda.*`` calls NVIDIA uses), then the weak presence-only signal
    of ``/sys/class/kfd/kfd`` or ``/opt/rocm`` existing (Linux; no VRAM
    number, just "ROCm is installed here").

    ROCm support for consumer cards is patchy to absent on Windows as of this
    codebase's knowledge, and even on Linux a hand-matched PyTorch/ROCm wheel
    is fragile. Ollama ships its own ROCm runtime, so **Ollama's ROCm build
    is the more reliable path for AMD** than depending on this project's
    PyTorch-based backends — recommend it in any AMD-facing docs or CLI
    output.

    Returns ``{"name", "vram_gb", "count"}`` or ``None``. Never raises.
    """
    result = _detect_rocm_smi()
    if result is not None:
        return result
    return _detect_amd_torch()


# --------------------------------------------------------------------------- #
#  Apple Silicon (MPS)
# --------------------------------------------------------------------------- #
def detect_apple() -> Optional[Dict[str, Any]]:
    """Probe for Apple Silicon GPU acceleration via Metal (MPS).

    ``torch.backends.mps.is_available()`` is the authoritative signal;
    ``platform.machine() == "arm64"`` is only used to pick a nicer label, as
    corroboration. Apple Silicon uses unified memory — there is no separate
    VRAM pool to query — so a detected result always reports ``vram_gb=0.0``
    by design, not as a stand-in for "unknown"; a caller sizing models for
    Apple Silicon should use :func:`system_ram_gb` instead.

    Returns ``{"name", "vram_gb", "count"}`` or ``None``. Never raises.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        available = bool(mps.is_available()) if mps is not None else False
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.backends.mps probe failed: %s", exc)
        return None
    if not available:
        return None

    try:
        arch = platform.machine()
    except Exception:  # noqa: BLE001
        arch = ""
    name = "Apple Silicon GPU (MPS)" if arch == "arm64" else "Apple GPU (MPS)"
    return {"name": name, "vram_gb": 0.0, "count": 1}


# --------------------------------------------------------------------------- #
#  Google TPU
# --------------------------------------------------------------------------- #
_TPU_ENV_VARS = ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID")


def detect_tpu() -> Optional[Dict[str, Any]]:
    """Detect a Google Cloud TPU environment/runtime, honestly.

    A TPU is essentially a cloud-only accelerator — GCP TPU VMs, Colab/Kaggle
    TPU runtimes — almost never present on a personal machine. Detection is
    real: the environment variables GCP/Colab set (``TPU_NAME``,
    ``COLAB_TPU_ADDR``, ``TPU_WORKER_ID``), plus a lazy import of
    ``torch_xla`` or a TPU-backed JAX. But detecting one is not the same as
    being able to use it: **no backend in this codebase
    (``jarvis/llm/*_backend.py``) places a single tensor on an XLA device
    today.** That would need PyTorch-XLA or JAX wired into a new backend,
    which does not exist here. :func:`detect` always attaches a note to this
    effect whenever a TPU is reported — never treat a non-``None`` result
    from this function as "acceleration will happen".

    Returns ``{"name", "vram_gb", "count"}`` or ``None``. Never raises.
    """
    env_hit: Optional[str] = None
    for var in _TPU_ENV_VARS:
        if os.environ.get(var):
            env_hit = var
            break

    xla_available = False
    try:
        import torch_xla  # type: ignore  # noqa: F401
        xla_available = True
    except ImportError:
        xla_available = False
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch_xla import raised: %s", exc)
        xla_available = False

    jax_tpu_count = 0
    if not xla_available:
        try:
            import jax  # type: ignore
            devices = jax.devices("tpu")
            jax_tpu_count = len(devices) if devices else 0
        except ImportError:
            jax_tpu_count = 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("jax TPU probe raised: %s", exc)
            jax_tpu_count = 0

    if env_hit is None and not xla_available and jax_tpu_count <= 0:
        return None

    if xla_available or jax_tpu_count > 0:
        name = "Google Cloud TPU"
    else:
        name = f"TPU environment detected ({env_hit})"
    count = jax_tpu_count if jax_tpu_count > 0 else 1
    return {"name": name, "vram_gb": 0.0, "count": count}


# --------------------------------------------------------------------------- #
#  Composition
# --------------------------------------------------------------------------- #
def detect() -> HardwareProfile:
    """Probe this machine and return a single, honest :class:`HardwareProfile`.

    Runs every detector in priority order — NVIDIA > AMD > Apple > TPU — and
    reports only the first real accelerator found; a machine with more than
    one signal present (e.g. a ROCm-built torch installed alongside a real
    NVIDIA card that ``nvidia-smi`` finds) reports only the
    highest-priority one, not both. Every probe is individually wrapped so a
    raising probe cannot take this function down with it, and every
    subprocess call a probe makes carries its own short timeout — so the
    whole pass stays bounded (on the order of the couple of subprocess calls
    involved, a few seconds each) even if every probe times out.

    Never raises.
    """
    cores, threads = _safe(cpu_info, (1, 1))
    if cores <= 0:
        cores = 1
    if threads <= 0:
        threads = 1
    ram = float(_safe(system_ram_gb, 0.0) or 0.0)

    nvidia = _safe(detect_nvidia, None)
    amd = _safe(detect_amd, None)
    apple = _safe(detect_apple, None)
    tpu = _safe(detect_tpu, None)

    notes: List[str] = []
    accelerator = "none"
    vendor = ""
    gpu_name = ""
    gpu_count = 0
    vram = 0.0

    if nvidia is not None:
        accelerator, vendor = "cuda", "nvidia"
        gpu_name = str(nvidia.get("name") or "")
        gpu_count = int(nvidia.get("count") or 0)
        vram = float(nvidia.get("vram_gb") or 0.0)
    elif amd is not None and not rocm_supports(amd.get("name")):
        # A real AMD GPU that ROCm cannot drive. Reporting it as an
        # accelerator would make the planner size a model against VRAM that
        # will never be used, and would promise the user acceleration the
        # machine cannot deliver. Report the card, plan for CPU.
        gpu_name = str(amd.get("name") or "")
        gpu_count = int(amd.get("count") or 0)
        notes.append(
            f"An AMD GPU ({gpu_name or 'unknown model'}) is present, but it "
            "belongs to a family ROCm does not support, so no local runtime "
            "can use it for inference. Planning for CPU only. This is a "
            "limitation of the card, not of the configuration -- the GPU is "
            "fine for display and video decode, just not for running a "
            "language model."
        )
    elif amd is not None:
        accelerator, vendor = "rocm", "amd"
        gpu_name = str(amd.get("name") or "")
        gpu_count = int(amd.get("count") or 0)
        vram = float(amd.get("vram_gb") or 0.0)
        notes.append(
            "AMD GPU detected via ROCm. A ROCm build of PyTorch places tensors "
            "with torch.device('cuda') -- the same API as NVIDIA -- so there is "
            "no torch.device('rocm'); any backend must map this to 'cuda'. "
            "Ollama's own ROCm build is the more reliable route than a "
            "hand-matched PyTorch/ROCm wheel, especially on Windows, where "
            "ROCm support for consumer cards is patchy to absent as of this "
            "writing."
        )
        if vram <= 0.0:
            notes.append(
                "VRAM size could not be determined for this AMD GPU (only its "
                "presence was detected) -- treat it as unknown, not zero."
            )
    elif apple is not None:
        accelerator, vendor = "mps", "apple"
        gpu_name = str(apple.get("name") or "")
        gpu_count = int(apple.get("count") or 0)
        vram = float(apple.get("vram_gb") or 0.0)
        notes.append(
            "Apple Silicon uses unified memory: there is no separate VRAM "
            "pool, so vram_gb is 0.0 by design here -- size models against "
            "system RAM instead."
        )
    elif tpu is not None:
        accelerator, vendor = "tpu", "google"
        gpu_name = str(tpu.get("name") or "")
        gpu_count = int(tpu.get("count") or 0)
        vram = float(tpu.get("vram_gb") or 0.0)
        notes.append(
            "A TPU was detected, but no backend in this codebase "
            "(jarvis/llm/*_backend.py) can place a tensor on it yet -- that "
            "needs PyTorch-XLA or JAX wired into a new backend, which does not "
            "exist here. Falling back to CPU for anything that actually runs."
        )
    else:
        notes.append("No GPU/accelerator detected; running on CPU only.")

    return HardwareProfile(
        cpu_cores=int(cores),
        cpu_threads=int(threads),
        ram_gb=round(ram, 2),
        accelerator=accelerator,
        gpu_vendor=vendor,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_gb=round(vram, 2),
        notes=tuple(notes),
    )


_VENDOR_LABELS = {"nvidia": "NVIDIA", "amd": "AMD", "apple": "Apple", "google": "Google"}


def summary(profile: HardwareProfile) -> str:
    """One human-readable line for ``jarvis doctor`` / ``jarvis hardware``.

    Examples: ``"NVIDIA RTX 3070 (8.0 GB VRAM), 32.0 GB RAM, 8 cores"`` or
    ``"CPU only, 32.0 GB RAM, 8 cores"``.
    """
    cores_bit = f"{profile.cpu_cores} core" + ("s" if profile.cpu_cores != 1 else "")
    ram_bit = f"{profile.ram_gb:g} GB RAM"

    if profile.accelerator == "none" or not profile.gpu_name:
        return f"CPU only, {ram_bit}, {cores_bit}"

    vendor_label = _VENDOR_LABELS.get(profile.gpu_vendor, profile.gpu_vendor.upper())
    name = profile.gpu_name
    if vendor_label and not name.upper().startswith(vendor_label.upper()):
        name = f"{vendor_label} {name}".strip()

    if profile.accelerator == "tpu":
        gpu_bit = f"{name} (not usable by any backend yet)"
    elif profile.accelerator == "mps":
        gpu_bit = f"{name} (unified memory)"
    elif profile.vram_gb > 0:
        gpu_bit = f"{name} ({profile.vram_gb:g} GB VRAM)"
    else:
        gpu_bit = f"{name} (VRAM unknown)"

    extra = f", {profile.gpu_count} GPUs" if profile.gpu_count > 1 else ""
    return f"{gpu_bit}{extra}, {ram_bit}, {cores_bit}"


__all__ = [
    "HardwareProfile",
    "cpu_info",
    "system_ram_gb",
    "detect_nvidia",
    "detect_amd",
    "detect_apple",
    "detect_tpu",
    "detect",
    "summary",
]
