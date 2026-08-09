"""Tests for jarvis.core.hardware — every detector, every branch, no real GPU needed.

Every probe in the module under test is designed to never raise and to be
cheap to fake: subprocess calls go through ``platform_utils.run_command``
(monkeypatched directly here) and every third-party import is lazy, so a fake
module dropped into ``sys.modules`` stands in for torch/psutil/torch_xla/jax
without needing the real package installed.
"""

from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

import pytest

from jarvis.core import hardware
from jarvis.core import platform_utils
from jarvis.core.platform_utils import CommandResult


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _fake_torch(*, cuda_available=False, hip=None, device_count=0, devices=None,
                mps_available=None):
    """Build a minimal fake ``torch`` module for injection into sys.modules."""
    mod = types.ModuleType("torch")
    devices = devices or []

    version_mod = types.SimpleNamespace(hip=hip)
    mod.version = version_mod  # type: ignore[attr-defined]

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def device_count():
            return device_count

        @staticmethod
        def get_device_properties(idx):
            name, total_memory = devices[idx]
            return types.SimpleNamespace(name=name, total_memory=total_memory)

    mod.cuda = _Cuda()  # type: ignore[attr-defined]

    if mps_available is not None:
        class _Mps:
            @staticmethod
            def is_available():
                return mps_available

        backends = types.SimpleNamespace(mps=_Mps())
        mod.backends = backends  # type: ignore[attr-defined]

    return mod


def _install_fake_module(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _block_module(monkeypatch, name):
    """Force ``import name`` to raise ImportError even if it is really installed."""
    monkeypatch.setitem(sys.modules, name, None)


# --------------------------------------------------------------------------- #
#  cpu_info
# --------------------------------------------------------------------------- #
def test_cpu_info_no_psutil_falls_back_to_os_cpu_count(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    cores, threads = hardware.cpu_info()
    assert cores == 8
    assert threads == 8


def test_cpu_info_with_psutil_prefers_physical_count(monkeypatch):
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.cpu_count = lambda logical=True: 8 if logical else 4  # type: ignore
    _install_fake_module(monkeypatch, "psutil", fake_psutil)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    cores, threads = hardware.cpu_info()
    assert cores == 4
    assert threads == 8


def test_cpu_info_os_cpu_count_none_never_returns_zero(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: None)
    cores, threads = hardware.cpu_info()
    assert cores == 1
    assert threads == 1


def test_cpu_info_os_cpu_count_raises_is_handled(monkeypatch):
    _block_module(monkeypatch, "psutil")

    def _boom():
        raise OSError("no cpus for you")

    monkeypatch.setattr(hardware.os, "cpu_count", _boom)
    cores, threads = hardware.cpu_info()
    assert cores == 1
    assert threads == 1


def test_cpu_info_psutil_raising_is_handled(monkeypatch):
    fake_psutil = types.ModuleType("psutil")

    def _boom(logical=True):
        raise RuntimeError("nope")

    fake_psutil.cpu_count = _boom  # type: ignore
    _install_fake_module(monkeypatch, "psutil", fake_psutil)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 4)
    cores, threads = hardware.cpu_info()
    assert cores == 4
    assert threads == 4


# --------------------------------------------------------------------------- #
#  system_ram_gb
# --------------------------------------------------------------------------- #
def test_system_ram_gb_uses_psutil_when_available(monkeypatch):
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.virtual_memory = lambda: types.SimpleNamespace(total=32_000_000_000)  # type: ignore
    _install_fake_module(monkeypatch, "psutil", fake_psutil)
    assert hardware.system_ram_gb() == 32.0


def test_ram_gb_proc_meminfo_parses_kib_to_gb(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    # 32 GiB in KiB, as /proc/meminfo reports it.
    meminfo.write_text("MemTotal:       33554432 kB\nMemFree: 1000 kB\n", encoding="utf-8")
    assert hardware._ram_gb_proc_meminfo(meminfo) == pytest.approx(34.36, abs=0.01)


def test_system_ram_gb_dispatches_to_linux_helper(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_MAC", False, raising=False)
    monkeypatch.setattr(hardware, "_ram_gb_proc_meminfo", lambda: 16.5)
    assert hardware.system_ram_gb() == 16.5


def test_system_ram_gb_dispatches_to_windows_helper(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(platform_utils, "IS_LINUX", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(platform_utils, "IS_MAC", False, raising=False)
    monkeypatch.setattr(hardware, "_ram_gb_windows_ctypes", lambda: 64.0)
    assert hardware.system_ram_gb() == 64.0


@pytest.mark.skipif(not platform_utils.IS_WINDOWS, reason="exercises the real Win32 API")
def test_ram_gb_windows_ctypes_returns_a_positive_value_on_real_windows():
    value = hardware._ram_gb_windows_ctypes()
    assert value is not None
    assert value > 0.0


def test_system_ram_gb_dispatches_to_mac_helper(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(platform_utils, "IS_LINUX", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_MAC", True, raising=False)
    monkeypatch.setattr(hardware, "_ram_gb_sysctl", lambda: 8.0)
    assert hardware.system_ram_gb() == 8.0


def test_ram_gb_sysctl_parses_run_command_output(monkeypatch):
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(0, "17179869184\n", ""),
    )
    assert hardware._ram_gb_sysctl() == pytest.approx(17.18, abs=0.01)


def test_ram_gb_sysctl_handles_failure(monkeypatch):
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(1, "", "no sysctl"),
    )
    assert hardware._ram_gb_sysctl() is None


def test_ram_gb_sysctl_handles_garbage_output(monkeypatch):
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(0, "not-a-number", ""),
    )
    assert hardware._ram_gb_sysctl() is None


def test_ram_gb_proc_meminfo_missing_file_returns_none(tmp_path: Path):
    assert hardware._ram_gb_proc_meminfo(tmp_path / "does-not-exist") is None


def test_ram_gb_proc_meminfo_malformed_returns_none(tmp_path: Path):
    bad = tmp_path / "meminfo"
    bad.write_text("MemTotal: notanumber kB\n", encoding="utf-8")
    assert hardware._ram_gb_proc_meminfo(bad) is None


def test_system_ram_gb_all_fallbacks_absent_returns_zero(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(platform_utils, "IS_LINUX", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", False, raising=False)
    monkeypatch.setattr(platform_utils, "IS_MAC", False, raising=False)
    assert hardware.system_ram_gb() == 0.0


def test_system_ram_gb_never_raises_even_if_helper_raises(monkeypatch):
    _block_module(monkeypatch, "psutil")
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(hardware, "_ram_gb_proc_meminfo", _boom)
    assert hardware.system_ram_gb() == 0.0


# --------------------------------------------------------------------------- #
#  detect_nvidia
# --------------------------------------------------------------------------- #
def test_detect_nvidia_single_gpu(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(0, "NVIDIA GeForce RTX 3070, 8192\n", ""),
    )
    result = hardware.detect_nvidia()
    assert result is not None
    assert result["name"] == "NVIDIA GeForce RTX 3070"
    assert result["count"] == 1
    assert result["vram_gb"] == pytest.approx(8.59, abs=0.01)


def test_detect_nvidia_multiple_gpus_sums_vram(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")
    stdout = "NVIDIA A100, 40960\nNVIDIA A100, 40960\n"
    monkeypatch.setattr(platform_utils, "run_command", lambda *a, **k: CommandResult(0, stdout, ""))
    result = hardware.detect_nvidia()
    assert result is not None
    assert result["count"] == 2
    assert result["name"] == "NVIDIA A100"
    assert result["vram_gb"] == pytest.approx(85.9, abs=0.1)


def test_detect_nvidia_malformed_output_falls_through(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(0, "garbage,,,\nnot,csv,either\n\n", ""),
    )
    _block_module(monkeypatch, "torch")
    assert hardware.detect_nvidia() is None


def test_detect_nvidia_empty_output_falls_through(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(platform_utils, "run_command", lambda *a, **k: CommandResult(0, "", ""))
    _block_module(monkeypatch, "torch")
    assert hardware.detect_nvidia() is None


def test_detect_nvidia_smi_absent_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    fake = _fake_torch(cuda_available=True, hip=None, device_count=1,
                       devices=[("NVIDIA RTX 4090", 24_000_000_000)])
    _install_fake_module(monkeypatch, "torch", fake)
    result = hardware.detect_nvidia()
    assert result is not None
    assert result["name"] == "NVIDIA RTX 4090"
    assert result["count"] == 1
    assert result["vram_gb"] == pytest.approx(24.0, abs=0.01)


def test_detect_nvidia_neither_smi_nor_torch(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    _block_module(monkeypatch, "torch")
    assert hardware.detect_nvidia() is None


def test_detect_nvidia_torch_fallback_rejects_rocm_build(monkeypatch):
    """The critical guard: a ROCm torch build also has cuda.is_available()==True,
    but must NOT be reported as NVIDIA."""
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    fake = _fake_torch(cuda_available=True, hip="5.7", device_count=1,
                       devices=[("AMD Radeon RX 7900", 20_000_000_000)])
    _install_fake_module(monkeypatch, "torch", fake)
    assert hardware.detect_nvidia() is None


def test_detect_nvidia_run_command_raises_is_handled(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")

    def _boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(platform_utils, "run_command", _boom)
    _block_module(monkeypatch, "torch")
    assert hardware.detect_nvidia() is None


def test_detect_nvidia_timed_out_result_is_treated_as_absent(monkeypatch):
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        platform_utils, "run_command",
        lambda *a, **k: CommandResult(124, "", "timed out", timed_out=True),
    )
    _block_module(monkeypatch, "torch")
    assert hardware.detect_nvidia() is None


# --------------------------------------------------------------------------- #
#  detect_amd
# --------------------------------------------------------------------------- #
def test_detect_amd_rocm_smi_single_gpu(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/rocm-smi")
    stdout = "GPU[0]      : VRAM Total Memory (B): 17179869184\n"
    monkeypatch.setattr(platform_utils, "run_command", lambda *a, **k: CommandResult(0, stdout, ""))
    result = hardware.detect_amd()
    assert result is not None
    assert result["count"] == 1
    assert result["vram_gb"] == pytest.approx(17.18, abs=0.01)


def test_detect_amd_rocm_smi_multiple_gpus(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/rocm-smi")
    stdout = (
        "GPU[0]      : VRAM Total Memory (B): 17179869184\n"
        "GPU[1]      : VRAM Total Memory (B): 17179869184\n"
    )
    monkeypatch.setattr(platform_utils, "run_command", lambda *a, **k: CommandResult(0, stdout, ""))
    result = hardware.detect_amd()
    assert result is not None
    assert result["count"] == 2
    assert result["vram_gb"] == pytest.approx(34.36, abs=0.01)


def test_detect_amd_rocm_smi_not_on_linux_is_skipped(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", False, raising=False)
    calls = []
    monkeypatch.setattr(platform_utils, "which", lambda name: calls.append(name) or None)
    _block_module(monkeypatch, "torch")
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: False)
    assert hardware.detect_amd() is None


def test_detect_amd_torch_hip_truthy_uses_cuda_api(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    fake = _fake_torch(cuda_available=True, hip="5.7", device_count=1,
                       devices=[("AMD Radeon RX 7900 XTX", 24_000_000_000)])
    _install_fake_module(monkeypatch, "torch", fake)
    result = hardware.detect_amd()
    assert result is not None
    assert result["name"] == "AMD Radeon RX 7900 XTX"
    assert result["vram_gb"] == pytest.approx(24.0, abs=0.01)


def test_detect_amd_torch_hip_falsy_falls_to_sysfs(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    fake = _fake_torch(cuda_available=True, hip=None, device_count=1,
                       devices=[("NVIDIA RTX", 8_000_000_000)])
    _install_fake_module(monkeypatch, "torch", fake)
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: False)
    assert hardware.detect_amd() is None


def test_detect_amd_torch_absent_falls_to_sysfs_present(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    _block_module(monkeypatch, "torch")
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: True)
    result = hardware.detect_amd()
    assert result is not None
    assert result["vram_gb"] == 0.0
    assert result["count"] == 1


def test_detect_amd_nothing_present_returns_none(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    _block_module(monkeypatch, "torch")
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: False)
    assert hardware.detect_amd() is None


def test_detect_amd_rocm_smi_malformed_output_falls_through(monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/rocm-smi")
    monkeypatch.setattr(platform_utils, "run_command", lambda *a, **k: CommandResult(0, "nothing useful\n", ""))
    _block_module(monkeypatch, "torch")
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: False)
    assert hardware.detect_amd() is None


def test_parse_rocm_vram_bytes_tolerant_of_whitespace_and_columns():
    text = "card0, foo,  VRAM Total Memory (B):   1234567890  \nunrelated line\n"
    assert hardware._parse_rocm_vram_bytes(text) == [1234567890]


# --------------------------------------------------------------------------- #
#  detect_apple
# --------------------------------------------------------------------------- #
def test_detect_apple_available(monkeypatch):
    fake = _fake_torch(mps_available=True)
    _install_fake_module(monkeypatch, "torch", fake)
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    result = hardware.detect_apple()
    assert result is not None
    assert "Apple Silicon" in result["name"]
    assert result["vram_gb"] == 0.0


def test_detect_apple_unavailable(monkeypatch):
    fake = _fake_torch(mps_available=False)
    _install_fake_module(monkeypatch, "torch", fake)
    assert hardware.detect_apple() is None


def test_detect_apple_torch_absent(monkeypatch):
    _block_module(monkeypatch, "torch")
    assert hardware.detect_apple() is None


def test_detect_apple_non_arm_label(monkeypatch):
    fake = _fake_torch(mps_available=True)
    _install_fake_module(monkeypatch, "torch", fake)
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    result = hardware.detect_apple()
    assert result is not None
    assert result["name"] == "Apple GPU (MPS)"


def test_detect_apple_no_backends_attribute(monkeypatch):
    fake = types.ModuleType("torch")
    _install_fake_module(monkeypatch, "torch", fake)
    assert hardware.detect_apple() is None


# --------------------------------------------------------------------------- #
#  detect_tpu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("var", ["TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"])
def test_detect_tpu_each_env_var(monkeypatch, var):
    for other in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(var, "some-value")
    _block_module(monkeypatch, "torch_xla")
    _block_module(monkeypatch, "jax")
    result = hardware.detect_tpu()
    assert result is not None
    assert var in result["name"]


def test_detect_tpu_no_env_no_libs_returns_none(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    _block_module(monkeypatch, "torch_xla")
    _block_module(monkeypatch, "jax")
    assert hardware.detect_tpu() is None


def test_detect_tpu_torch_xla_importable_without_env(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    fake_xla = types.ModuleType("torch_xla")
    _install_fake_module(monkeypatch, "torch_xla", fake_xla)
    _block_module(monkeypatch, "jax")
    result = hardware.detect_tpu()
    assert result is not None
    assert result["name"] == "Google Cloud TPU"


def test_detect_tpu_torch_xla_import_raises_non_import_error(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)

    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "torch_xla":
            raise RuntimeError("broken install")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    _block_module(monkeypatch, "jax")
    # Must not raise.
    result = hardware.detect_tpu()
    assert result is None


def test_detect_tpu_jax_with_tpu_backend(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    _block_module(monkeypatch, "torch_xla")
    fake_jax = types.ModuleType("jax")
    fake_jax.devices = lambda kind: ["tpu0", "tpu1"]  # type: ignore
    _install_fake_module(monkeypatch, "jax", fake_jax)
    result = hardware.detect_tpu()
    assert result is not None
    assert result["name"] == "Google Cloud TPU"
    assert result["count"] == 2


def test_detect_tpu_jax_without_tpu_backend_raises(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    _block_module(monkeypatch, "torch_xla")

    def _raise(kind):
        raise RuntimeError("Unknown backend tpu")

    fake_jax = types.ModuleType("jax")
    fake_jax.devices = _raise  # type: ignore
    _install_fake_module(monkeypatch, "jax", fake_jax)
    assert hardware.detect_tpu() is None


def test_detect_tpu_jax_without_tpu_backend_empty_list(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    _block_module(monkeypatch, "torch_xla")
    fake_jax = types.ModuleType("jax")
    fake_jax.devices = lambda kind: []  # type: ignore
    _install_fake_module(monkeypatch, "jax", fake_jax)
    assert hardware.detect_tpu() is None


def test_detect_tpu_env_var_present_but_no_libs_reports_weak_signal(monkeypatch):
    for var in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TPU_NAME", "grpc://10.0.0.1:8470")
    _block_module(monkeypatch, "torch_xla")
    _block_module(monkeypatch, "jax")
    result = hardware.detect_tpu()
    assert result is not None
    assert result["count"] == 1


# --------------------------------------------------------------------------- #
#  detect() composition
# --------------------------------------------------------------------------- #
def test_detect_no_signals_reports_cpu_only(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_info", lambda: (8, 16))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 32.0)
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "detect_apple", lambda: None)
    monkeypatch.setattr(hardware, "detect_tpu", lambda: None)

    profile = hardware.detect()
    assert profile.accelerator == "none"
    assert profile.gpu_vendor == ""
    assert profile.cpu_cores == 8
    assert profile.cpu_threads == 16
    assert profile.ram_gb == 32.0
    assert any("CPU only" in n for n in profile.notes)


def test_detect_priority_nvidia_wins_over_amd(monkeypatch):
    """Both an NVIDIA and an AMD signal fire simultaneously (e.g. a ROCm torch
    install alongside a real NVIDIA card nvidia-smi finds): NVIDIA must win,
    since detect() only ever reports the first accelerator by priority."""
    monkeypatch.setattr(hardware, "cpu_info", lambda: (4, 8))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 16.0)
    monkeypatch.setattr(
        hardware, "detect_nvidia",
        lambda: {"name": "NVIDIA RTX 3070", "vram_gb": 8.0, "count": 1},
    )
    monkeypatch.setattr(
        hardware, "detect_amd",
        lambda: {"name": "AMD GPU (ROCm)", "vram_gb": 16.0, "count": 1},
    )
    monkeypatch.setattr(hardware, "detect_apple", lambda: None)
    monkeypatch.setattr(hardware, "detect_tpu", lambda: None)

    profile = hardware.detect()
    assert profile.accelerator == "cuda"
    assert profile.gpu_vendor == "nvidia"
    assert profile.gpu_name == "NVIDIA RTX 3070"
    assert profile.vram_gb == 8.0


def test_detect_priority_amd_over_apple_and_tpu(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_info", lambda: (4, 8))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: None)
    monkeypatch.setattr(
        hardware, "detect_amd",
        lambda: {"name": "AMD GPU (ROCm)", "vram_gb": 16.0, "count": 1},
    )
    monkeypatch.setattr(
        hardware, "detect_apple",
        lambda: {"name": "Apple Silicon GPU (MPS)", "vram_gb": 0.0, "count": 1},
    )
    monkeypatch.setattr(
        hardware, "detect_tpu",
        lambda: {"name": "Google Cloud TPU", "vram_gb": 0.0, "count": 1},
    )
    profile = hardware.detect()
    assert profile.accelerator == "rocm"
    assert profile.gpu_vendor == "amd"
    assert any("torch.device('cuda')" in n for n in profile.notes)


def test_detect_tpu_reports_note_about_no_usable_backend(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_info", lambda: (4, 8))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "detect_apple", lambda: None)
    monkeypatch.setattr(
        hardware, "detect_tpu",
        lambda: {"name": "Google Cloud TPU", "vram_gb": 0.0, "count": 1},
    )
    profile = hardware.detect()
    assert profile.accelerator == "tpu"
    assert profile.gpu_vendor == "google"
    assert any("no backend" in n for n in profile.notes)


def test_detect_apple_reports_unified_memory_note(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_info", lambda: (4, 8))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "detect_amd", lambda: None)
    monkeypatch.setattr(
        hardware, "detect_apple",
        lambda: {"name": "Apple Silicon GPU (MPS)", "vram_gb": 0.0, "count": 1},
    )
    monkeypatch.setattr(hardware, "detect_tpu", lambda: None)
    profile = hardware.detect()
    assert profile.accelerator == "mps"
    assert profile.gpu_vendor == "apple"
    assert any("unified memory" in n for n in profile.notes)


def test_detect_never_raises_when_every_probe_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("simulated hardware probe failure")

    monkeypatch.setattr(hardware, "cpu_info", _boom)
    monkeypatch.setattr(hardware, "system_ram_gb", _boom)
    monkeypatch.setattr(hardware, "detect_nvidia", _boom)
    monkeypatch.setattr(hardware, "detect_amd", _boom)
    monkeypatch.setattr(hardware, "detect_apple", _boom)
    monkeypatch.setattr(hardware, "detect_tpu", _boom)

    profile = hardware.detect()
    assert isinstance(profile, hardware.HardwareProfile)
    assert profile.accelerator == "none"
    assert profile.cpu_cores >= 1
    assert profile.cpu_threads >= 1
    assert profile.ram_gb == 0.0


def test_detect_bounded_even_with_a_hanging_probe_simulated(monkeypatch):
    """A slow subprocess must not make detect() hang: run_command itself owns
    the timeout, so a monkeypatched 'slow' probe that instead returns a
    timed-out CommandResult immediately must not cause detect() to retry or
    block further."""
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(platform_utils, "IS_LINUX", True, raising=False)

    def _slow_but_bounded(*a, **k):
        # Simulates what run_command itself returns once its internal
        # subprocess timeout fires -- instant in the test, no real sleep.
        return CommandResult(124, "", "timed out", timed_out=True)

    monkeypatch.setattr(platform_utils, "run_command", _slow_but_bounded)
    _block_module(monkeypatch, "torch")
    monkeypatch.setattr(hardware, "_rocm_sysfs_present", lambda: False)

    start = time.monotonic()
    profile = hardware.detect()
    elapsed = time.monotonic() - start
    assert elapsed < 2.0
    assert isinstance(profile, hardware.HardwareProfile)


def test_detect_gpu_count_and_vram_types(monkeypatch):
    monkeypatch.setattr(hardware, "cpu_info", lambda: (4, 8))
    monkeypatch.setattr(hardware, "system_ram_gb", lambda: 16.0)
    monkeypatch.setattr(
        hardware, "detect_nvidia",
        lambda: {"name": "NVIDIA RTX 3070", "vram_gb": 8.0, "count": 1},
    )
    monkeypatch.setattr(hardware, "detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "detect_apple", lambda: None)
    monkeypatch.setattr(hardware, "detect_tpu", lambda: None)
    profile = hardware.detect()
    d = profile.to_dict()
    assert d["gpu_count"] == 1
    assert d["vram_gb"] == 8.0
    assert isinstance(d["notes"], list)


# --------------------------------------------------------------------------- #
#  summary()
# --------------------------------------------------------------------------- #
def test_summary_cpu_only():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="none",
        gpu_vendor="", gpu_name="", gpu_count=0, vram_gb=0.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "CPU only" in line
    assert "32 GB RAM" in line
    assert "8 core" in line


def test_summary_nvidia():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="RTX 3070", gpu_count=1, vram_gb=8.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "NVIDIA RTX 3070" in line
    assert "8 GB VRAM" in line


def test_summary_nvidia_name_already_prefixed_not_doubled():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="NVIDIA GeForce RTX 3070", gpu_count=1,
        vram_gb=8.0, notes=(),
    )
    line = hardware.summary(profile)
    assert line.count("NVIDIA") == 1


def test_summary_multi_gpu_mentions_count():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=64.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="A100", gpu_count=2, vram_gb=80.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "2 GPUs" in line


def test_summary_tpu_says_not_usable():
    profile = hardware.HardwareProfile(
        cpu_cores=4, cpu_threads=8, ram_gb=16.0, accelerator="tpu",
        gpu_vendor="google", gpu_name="Google Cloud TPU", gpu_count=1,
        vram_gb=0.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "not usable" in line


def test_summary_apple_unified_memory():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=8, ram_gb=16.0, accelerator="mps",
        gpu_vendor="apple", gpu_name="Apple Silicon GPU (MPS)", gpu_count=1,
        vram_gb=0.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "unified memory" in line


def test_summary_amd_vram_unknown():
    profile = hardware.HardwareProfile(
        cpu_cores=8, cpu_threads=8, ram_gb=16.0, accelerator="rocm",
        gpu_vendor="amd", gpu_name="AMD GPU (ROCm)", gpu_count=1,
        vram_gb=0.0, notes=(),
    )
    line = hardware.summary(profile)
    assert "VRAM unknown" in line


# --------------------------------------------------------------------------- #
#  Import hygiene: a direct, file-scoped check (in addition to the whole-
#  package check in tests/test_import_hygiene.py).
# --------------------------------------------------------------------------- #
def test_hardware_module_imports_cleanly_with_heavy_packages_blocked():
    program = """
import sys

BLOCKED = {"torch", "psutil", "numpy", "jax", "torch_xla"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError("blocked for the import-hygiene test: " + name)
        return None


sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name.split('.')[0] in BLOCKED:
        del sys.modules[name]

import jarvis.core.hardware as hardware

# And every public function must be *callable* without those packages, too.
hardware.cpu_info()
hardware.system_ram_gb()
hardware.detect_nvidia()
hardware.detect_amd()
hardware.detect_apple()
hardware.detect_tpu()
profile = hardware.detect()
hardware.summary(profile)

print("ALL GOOD")
"""
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ALL GOOD" in result.stdout
