"""The hardware planner: composes config + a hardware profile into a HardwarePlan.

No real hardware is ever probed here -- every scenario passes a synthetic
:class:`jarvis.core.hardware.HardwareProfile` via ``profile=`` (or
monkeypatches ``jarvis.core.hardware.detect``), exactly as the module is
designed to be tested.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from jarvis.core.config import Config
from jarvis.core.hardware import HardwareProfile
from jarvis.llm import models
from jarvis.llm.planner import HardwarePlan, plan


# --------------------------------------------------------------------------- #
#  Synthetic profiles -- the real HardwareProfile contract, never a live probe
# --------------------------------------------------------------------------- #
def cpu_only_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=4, cpu_threads=8, ram_gb=32.0, accelerator="none",
        gpu_vendor="", gpu_name="", gpu_count=0, vram_gb=0.0,
        notes=("No GPU/accelerator detected; running on CPU only.",),
    )


def nvidia_large_vram_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=16, cpu_threads=32, ram_gb=64.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="NVIDIA RTX 4090", gpu_count=1, vram_gb=24.0,
    )


def nvidia_small_vram_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="NVIDIA RTX 3050", gpu_count=1, vram_gb=8.0,
    )


def amd_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="rocm",
        gpu_vendor="amd", gpu_name="AMD Radeon RX 7900", gpu_count=1, vram_gb=20.0,
        notes=("AMD GPU detected via ROCm. A ROCm build of PyTorch places tensors "
               "with torch.device('cuda') -- the same API as NVIDIA.",),
    )


def apple_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=8, cpu_threads=8, ram_gb=32.0, accelerator="mps",
        gpu_vendor="apple", gpu_name="Apple Silicon GPU (MPS)", gpu_count=1, vram_gb=0.0,
        notes=("Apple Silicon uses unified memory: there is no separate VRAM pool.",),
    )


def tpu_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=4, cpu_threads=4, ram_gb=32.0, accelerator="tpu",
        gpu_vendor="google", gpu_name="Google Cloud TPU", gpu_count=1, vram_gb=0.0,
        notes=("A TPU was detected, but no backend in this codebase can place a "
               "tensor on it yet.",),
    )


ALL_PROFILES = [
    cpu_only_profile, nvidia_large_vram_profile, nvidia_small_vram_profile,
    amd_profile, apple_profile, tpu_profile,
]


@pytest.fixture
def cfg(config) -> Config:
    """Reuse the shared isolated-home Config fixture, hardware left at defaults."""
    return config


# --------------------------------------------------------------------------- #
#  plan() never raises
# --------------------------------------------------------------------------- #
def test_plan_never_raises_when_detect_is_monkeypatched_to_raise(monkeypatch, cfg):
    def _boom():
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr("jarvis.core.hardware.detect", _boom)
    result = plan(cfg)
    assert isinstance(result, HardwarePlan)
    assert result.device == "cpu"
    assert result.notes, "the detection failure must be recorded, not swallowed silently"


def test_plan_never_raises_when_hardware_module_is_missing(monkeypatch, cfg):
    """Simulate jarvis.core.hardware being unimportable entirely."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jarvis.core.hardware" or name.endswith(".hardware"):
            raise ImportError("simulated: no such module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = plan(cfg)
    assert isinstance(result, HardwarePlan)
    assert result.device == "cpu"


def test_plan_with_a_cfg_of_none_still_works():
    result = plan(None, profile=cpu_only_profile())
    assert isinstance(result, HardwarePlan)
    assert result.device == "cpu"


# --------------------------------------------------------------------------- #
#  Auto mode composes a synthetic profile correctly, per scenario
# --------------------------------------------------------------------------- #
def test_auto_cpu_only(cfg):
    result = plan(cfg, profile=cpu_only_profile())
    assert result.device == "cpu"
    assert result.accelerator == "none"
    assert result.fits_vram is None
    assert result.backend
    assert result.model_interactive
    assert result.model_background


def test_auto_nvidia_large_vram(cfg):
    result = plan(cfg, profile=nvidia_large_vram_profile())
    assert result.device == "cuda"
    assert result.accelerator == "cuda"
    assert result.fits_vram is not None


def test_auto_nvidia_small_vram(cfg):
    result = plan(cfg, profile=nvidia_small_vram_profile())
    assert result.device == "cuda"
    assert result.accelerator == "cuda"
    # A real, non-hand-waved footprint check backs this up.
    spec = models.lookup(result.model_interactive) or models.resolve(result.model_interactive)
    footprint = models.estimate_footprint(spec, "q4", vram_gb=8.0)
    assert footprint["fits_vram"] is True


def test_auto_amd_device_is_cuda_never_rocm(cfg):
    result = plan(cfg, profile=amd_profile())
    assert result.device == "cuda", "ROCm PyTorch uses torch.device('cuda'), not 'rocm'"
    assert result.device != "rocm"
    assert result.accelerator == "rocm"


def test_auto_amd_notes_mention_rocm_and_favour_ollama(cfg):
    result = plan(cfg, profile=amd_profile())
    joined = " ".join(result.notes).lower()
    assert "rocm" in joined
    assert "cuda" in joined  # explains the device-string overlap
    assert result.backend == "ollama"


def test_auto_apple_silicon(cfg):
    result = plan(cfg, profile=apple_profile())
    assert result.device == "mps"
    assert result.accelerator == "mps"
    joined = " ".join(result.notes).lower()
    assert "unified" in joined or "shared" in joined


def test_auto_tpu_always_falls_back_to_cpu_device(cfg):
    result = plan(cfg, profile=tpu_profile())
    assert result.device == "cpu"
    assert result.accelerator == "tpu"
    joined = " ".join(result.notes).lower()
    assert "tpu" in joined
    assert any(w in joined for w in ("no backend", "falling back", "cannot", "does not"))


def test_tpu_notes_are_never_silent():
    result = plan(None, profile=tpu_profile())
    assert result.notes, "a TPU-detected plan must always explain the CPU fallback"


def test_a_detected_tpu_gets_the_same_model_sizing_as_no_accelerator_at_all(cfg):
    """A TPU-flagged machine must not be sized as though it had real acceleration.

    device="cpu" is already asserted for TPU elsewhere, but has_gpu was being
    computed from the pre-mapping accelerator label ("tpu" != "cpu" -> True)
    instead of the resolved device -- so recommend() thought a real GPU was
    present and sized a bigger, CPU-unfriendly model as the *interactive*
    pick, directly contradicting the plan's own device="cpu". That is exactly
    the kind of silent overclaiming this whole feature exists to prevent; it
    just leaked through the sizing path instead of the device-string path.

    16 GB is chosen deliberately, not the module's 32 GB fixtures: at 32 GB the
    MoE model (low active parameters) already wins the "chat" pick regardless
    of has_gpu, so the bug is invisible there. At 16 GB the budget excludes
    the MoE model entirely and the interactive-CPU-active filter is the thing
    that decides between a small and a large dense model -- exactly the
    boundary the bug crossed (qwen3:4b vs. qwen3:14b, reproduced by hand
    against the real code before this test was written).
    """
    ram_gb = 16.0
    none_profile = replace(cpu_only_profile(), ram_gb=ram_gb)
    tpu_result_profile = replace(tpu_profile(), ram_gb=ram_gb)

    none_result = plan(cfg, profile=none_profile)
    tpu_result = plan(cfg, profile=tpu_result_profile)

    assert tpu_result.device == "cpu"
    assert tpu_result.model_interactive == none_result.model_interactive, (
        f"at {ram_gb} GB RAM, TPU picked {tpu_result.model_interactive!r} for "
        f"interactive use while no-accelerator picked "
        f"{none_result.model_interactive!r} -- the TPU plan is being sized as "
        f"though it had real acceleration"
    )
    assert tpu_result.quantisation == none_result.quantisation


# --------------------------------------------------------------------------- #
#  Manual mode
# --------------------------------------------------------------------------- #
def test_manual_mode_never_calls_detect(monkeypatch, cfg):
    calls = []
    monkeypatch.setattr(
        "jarvis.core.hardware.detect",
        lambda: calls.append(1) or cpu_only_profile(),
    )
    cfg.hardware.mode = "manual"
    cfg.hardware.accelerator = "cuda"
    cfg.hardware.vram_gb = 12.0
    cfg.hardware.ram_gb = 64.0
    cfg.hardware.gpu_count = 1

    result = plan(cfg)
    assert not calls, "manual mode with every field set must not call detect() at all"
    assert result.device == "cuda"
    assert result.accelerator == "cuda"


def test_manual_mode_honours_every_override(monkeypatch, cfg):
    monkeypatch.setattr(
        "jarvis.core.hardware.detect",
        lambda: (_ for _ in ()).throw(AssertionError("detect() must not be called")),
    )
    cfg.hardware.mode = "manual"
    cfg.hardware.accelerator = "rocm"
    cfg.hardware.vram_gb = 16.0
    cfg.hardware.ram_gb = 32.0
    cfg.hardware.gpu_count = 1

    result = plan(cfg)
    assert result.accelerator == "rocm"
    assert result.device == "cuda"


def test_manual_mode_partial_override_falls_back_to_detection_for_the_rest(monkeypatch, cfg):
    """Only accelerator is manually pinned; vram/ram/gpu_count stay at 'auto' (0)
    and must come from the profile passed to plan()."""
    cfg.hardware.mode = "manual"
    cfg.hardware.accelerator = "cuda"
    cfg.hardware.vram_gb = 0.0     # left at auto-detect sentinel
    cfg.hardware.ram_gb = 0.0      # left at auto-detect sentinel
    cfg.hardware.gpu_count = 0     # left at auto-detect sentinel

    result = plan(cfg, profile=nvidia_small_vram_profile())
    assert result.accelerator == "cuda"
    # vram came from the profile (8.0), not a manual value -- confirm the
    # interactive model was actually sized against it.
    assert result.fits_vram is True


def test_manual_mode_partial_override_uses_manual_value_when_present(cfg):
    """accelerator and vram_gb are manual; only ram_gb/gpu_count fall back."""
    cfg.hardware.mode = "manual"
    cfg.hardware.accelerator = "cuda"
    cfg.hardware.vram_gb = 4.0      # manual, deliberately small
    cfg.hardware.ram_gb = 0.0       # falls back to detection
    cfg.hardware.gpu_count = 0      # falls back to detection

    # Profile disagrees about VRAM (24 GB) -- the manual 4 GB must win.
    result = plan(cfg, profile=nvidia_large_vram_profile())
    assert result.accelerator == "cuda"
    spec = models.lookup(result.model_interactive) or models.resolve(result.model_interactive)
    footprint = models.estimate_footprint(spec, "q4", vram_gb=4.0)
    assert footprint["vram_headroom_gb"] > -4.0, (
        "the recommendation must respect the small manual VRAM figure, not the "
        "profile's larger one"
    )


def test_auto_mode_uses_manual_field_only_when_detection_leaves_it_undetermined(cfg):
    """Auto mode: detection determines everything except vram_gb (0.0, i.e.
    'no GPU or unknown'); the manual hardware.vram_gb is used as a hint only
    for that one field, while accelerator/ram_gb come straight from the
    profile, not the manual config."""
    cfg.hardware.mode = "auto"
    cfg.hardware.accelerator = "mps"       # must be ignored: profile says cuda
    cfg.hardware.ram_gb = 999.0            # must be ignored: profile says 32
    cfg.hardware.vram_gb = 6.0             # hint: profile's vram_gb is 0 (undetermined)

    undetermined_vram = HardwareProfile(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0, accelerator="cuda",
        gpu_vendor="nvidia", gpu_name="Mystery GPU", gpu_count=1, vram_gb=0.0,
    )
    result = plan(cfg, profile=undetermined_vram)
    assert result.accelerator == "cuda", "detected accelerator must win over the manual hint"
    joined = " ".join(result.notes).lower()
    assert "vram" in joined


# --------------------------------------------------------------------------- #
#  The owner's worked example: 32 GB RAM / 8 GB VRAM
# --------------------------------------------------------------------------- #
def test_32gb_ram_8gb_vram_differs_from_cpu_only(cfg):
    cpu_plan = plan(cfg, profile=cpu_only_profile())
    mixed_plan = plan(cfg, profile=nvidia_small_vram_profile())

    assert (
        cpu_plan.model_interactive != mixed_plan.model_interactive
        or cpu_plan.quantisation != mixed_plan.quantisation
    ), "a 32GB/8GB-VRAM machine must get a genuinely different plan than CPU-only"


def test_32gb_ram_8gb_vram_interactive_model_actually_fits(cfg):
    result = plan(cfg, profile=nvidia_small_vram_profile())
    spec = models.lookup(result.model_interactive) or models.resolve(result.model_interactive)
    footprint = models.estimate_footprint(spec, result.quantisation, vram_gb=8.0)
    assert footprint["fits_vram"] is True
    assert result.fits_vram is True


def test_32gb_ram_8gb_vram_background_model_is_at_least_as_capable(cfg):
    """The background pick may legitimately not fit in VRAM (partial offload
    is fine for a slower-is-acceptable pick) but should not be a strictly
    worse model than the interactive one."""
    result = plan(cfg, profile=nvidia_small_vram_profile())
    interactive_spec = models.lookup(result.model_interactive) or models.resolve(result.model_interactive)
    background_spec = models.lookup(result.model_background) or models.resolve(result.model_background)
    assert background_spec.params >= interactive_spec.params


# --------------------------------------------------------------------------- #
#  General sanity across every scenario
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile_factory", ALL_PROFILES)
def test_every_scenario_yields_a_well_formed_plan(cfg, profile_factory):
    result = plan(cfg, profile=profile_factory())
    assert isinstance(result, HardwarePlan)
    assert result.device in ("cuda", "mps", "cpu")
    assert result.accelerator in ("cuda", "rocm", "mps", "tpu", "none")
    assert result.backend
    assert result.model_interactive
    assert result.model_background
    assert result.quantisation
    assert isinstance(result.notes, tuple)


@pytest.mark.parametrize("profile_factory", ALL_PROFILES)
def test_recommended_models_are_real_and_ungated(cfg, profile_factory):
    result = plan(cfg, profile=profile_factory())
    for ref in (result.model_interactive, result.model_background):
        spec = models.lookup(ref) or models.resolve(ref)
        assert spec.exists
        assert not spec.gated


def test_plan_is_deterministic_for_the_same_inputs(cfg):
    a = plan(cfg, profile=nvidia_small_vram_profile())
    b = plan(cfg, profile=nvidia_small_vram_profile())
    assert a.model_interactive == b.model_interactive
    assert a.model_background == b.model_background
    assert a.device == b.device
