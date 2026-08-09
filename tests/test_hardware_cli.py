"""`jarvis hardware` — the CLI half of hardware detection and model planning.

``jarvis.core.hardware`` and ``jarvis.llm.planner`` are owned by other parts of
this feature and are imported lazily, inside ``cmd_hardware`` only. Most tests
below never depend on either module actually existing or being importable on
this machine: they install a throwaway stand-in module directly into
``sys.modules`` (and onto the real parent package's attribute, so a stale
cached attribute from an earlier import can never leak into a later test) and
tear it back down automatically via ``monkeypatch``. A handful of tests near
the bottom instead exercise the real modules end to end, to prove the CLI is
actually wired to the shapes those modules produce
(``HardwareProfile``/``detect()``/``summary(profile)`` and
``HardwarePlan``/``plan(cfg)``), not just to whatever shape a fake happened to
supply.

What is under test: which section headers ``cmd_hardware`` prints, how it
renders whatever the two modules hand back, how it distinguishes manual
overrides from auto-detected fields, and that a failure inside either module
comes back as a one-line message rather than a traceback.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import jarvis.core as jarvis_core_pkg
import jarvis.llm as jarvis_llm_pkg
from jarvis import cli


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def run(argv):
    """Parse `argv` and dispatch, the way `main()` does."""
    args = cli.build_parser().parse_args(argv)
    return int(args.func(args) or 0)


@pytest.fixture
def hardware_config(config, monkeypatch: pytest.MonkeyPatch):
    """Make `jarvis hardware` load the isolated test config, never a real one."""
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    return config


@pytest.fixture
def fake_hardware_module(monkeypatch: pytest.MonkeyPatch):
    """Install a throwaway `jarvis.core.hardware` for the duration of one test.

    Sets both `sys.modules["jarvis.core.hardware"]` and the attribute on the
    real `jarvis.core` package, because `from .core import hardware` in cli.py
    is satisfied by either one being present — and setting only the former
    would leave a stale attribute on `jarvis.core` for whatever test runs next
    if the real module is ever imported first in this same process.
    """
    module = types.ModuleType("jarvis.core.hardware")
    monkeypatch.setitem(sys.modules, "jarvis.core.hardware", module)
    monkeypatch.setattr(jarvis_core_pkg, "hardware", module, raising=False)
    return module


@pytest.fixture
def fake_planner_module(monkeypatch: pytest.MonkeyPatch):
    """Install a throwaway `jarvis.llm.planner` for the duration of one test."""
    module = types.ModuleType("jarvis.llm.planner")
    monkeypatch.setitem(sys.modules, "jarvis.llm.planner", module)
    monkeypatch.setattr(jarvis_llm_pkg, "planner", module, raising=False)
    return module


def _profile(**overrides):
    """A stand-in matching jarvis.core.hardware.HardwareProfile's shape."""
    fields = dict(
        cpu_cores=8, cpu_threads=16, ram_gb=32.0,
        accelerator="cuda", gpu_vendor="nvidia", gpu_name="RTX 3070",
        gpu_count=1, vram_gb=8.0, notes=(),
    )
    fields.update(overrides)

    def to_dict():
        d = dict(fields)
        d["notes"] = list(fields["notes"])
        return d

    ns = SimpleNamespace(**fields)
    ns.to_dict = to_dict
    return ns


def _plan(**overrides):
    """A stand-in matching jarvis.llm.planner.HardwarePlan's shape."""
    fields = dict(
        backend="ollama",
        device="cuda",
        accelerator="nvidia",
        model_interactive="qwen3-8b",
        model_background="qwen3-14b",
        quantisation="q4",
        fits_vram=True,
        notes=(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


# --------------------------------------------------------------------------- #
#  Registration and help
# --------------------------------------------------------------------------- #
def test_hardware_is_registered():
    args = cli.build_parser().parse_args(["hardware"])
    assert args.command == "hardware"
    assert args.func is cli.cmd_hardware


def test_hardware_appears_in_the_top_level_help():
    assert "hardware" in cli.build_parser().format_help()


def test_hardware_is_documented_in_the_module_docstring():
    assert "jarvis hardware" in (cli.__doc__ or "")


# --------------------------------------------------------------------------- #
#  The happy path (fake modules matching the real dataclass shapes)
# --------------------------------------------------------------------------- #
def test_prints_the_detected_profile_and_the_plan(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    profile = _profile(
        ram_gb=32.0, accelerator="cuda", gpu_vendor="nvidia",
        gpu_name="RTX 3070", vram_gb=8.0,
        notes=("no accelerator-specific caveats",),
    )
    fake_hardware_module.detect = lambda: profile
    fake_hardware_module.summary = lambda p: "NVIDIA RTX 3070 (8.0 GB VRAM), 32.0 GB RAM, 8 cores"
    fake_planner_module.plan = lambda cfg: _plan(notes=["note one", "note two"])

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Detected hardware" in out
    assert "NVIDIA RTX 3070 (8.0 GB VRAM), 32.0 GB RAM, 8 cores" in out
    assert "ram_gb" in out and "32.0" in out
    assert "gpu_name" in out and "RTX 3070" in out
    assert "vram_gb" in out and "8.0" in out
    assert "Detection notes" in out
    assert "no accelerator-specific caveats" in out

    assert "Plan" in out
    assert "backend" in out and "ollama" in out
    assert "device" in out and "cuda" in out
    assert "model (interactive)" in out and "qwen3-8b" in out
    assert "model (background)" in out and "qwen3-14b" in out
    assert "quantisation" in out and "q4" in out
    assert "fits VRAM" in out and "yes" in out

    assert "Notes" in out
    assert "- note one" in out
    assert "- note two" in out


def test_no_notes_means_no_notes_sections(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile(notes=())
    fake_hardware_module.summary = lambda p: "CPU only, 32.0 GB RAM, 8 cores"
    fake_planner_module.plan = lambda cfg: _plan(notes=[])

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Detection notes" not in out
    assert "\nNotes" not in out


def test_fits_vram_false_is_shown_as_no_not_unknown(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    fake_planner_module.plan = lambda cfg: _plan(fits_vram=False)

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    fits_line = next(line for line in out.splitlines() if line.strip().startswith("fits VRAM"))
    assert fits_line.strip() == "fits VRAM            no"


def test_fits_vram_none_is_shown_as_unknown(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    fake_planner_module.plan = lambda cfg: _plan(fits_vram=None)

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    fits_line = next(line for line in out.splitlines() if line.strip().startswith("fits VRAM"))
    assert "(unknown)" in fits_line


def test_a_plan_field_the_planner_did_not_set_is_shown_as_unknown(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    # A plan missing `quantisation` entirely (e.g. an older planner build).
    incomplete = SimpleNamespace(
        backend="ollama", device="cpu", accelerator="cpu",
        model_interactive="qwen3-4b", model_background="qwen3-8b",
        fits_vram=None, notes=(),
    )
    fake_planner_module.plan = lambda cfg: incomplete

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "quantisation" in out
    assert "(unknown)" in out


# --------------------------------------------------------------------------- #
#  Manual overrides vs auto-detected fields
# --------------------------------------------------------------------------- #
def test_auto_mode_never_shows_a_manual_overrides_section(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    hardware_config.hardware.mode = "auto"
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    fake_planner_module.plan = lambda cfg: _plan()

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Manual overrides" not in out


def test_manual_mode_reports_which_fields_are_overridden_and_which_stay_auto(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    hardware_config.hardware.mode = "manual"
    hardware_config.hardware.accelerator = "cuda"     # overridden
    hardware_config.hardware.vram_gb = 8.0             # overridden
    hardware_config.hardware.ram_gb = 0.0               # left auto
    hardware_config.hardware.gpu_count = 0               # left auto
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    fake_planner_module.plan = lambda cfg: _plan()

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Manual overrides (hardware.mode = manual)" in out
    assert "manual override -> cuda" in out
    assert "manual override -> 8.0" in out
    # ram_gb and gpu_count were left at their auto-detect sentinel (0). Search
    # only the overrides section: the detected-profile table above it has its
    # own, unrelated "ram_gb" line (the probed value, not the override state).
    overrides_section = out.split("Manual overrides (hardware.mode = manual)", 1)[1]
    ram_line = next(
        line for line in overrides_section.splitlines() if line.strip().startswith("ram_gb")
    )
    gpu_line = next(
        line for line in overrides_section.splitlines() if line.strip().startswith("gpu_count")
    )
    assert "auto (not overridden)" in ram_line
    assert "auto (not overridden)" in gpu_line


# --------------------------------------------------------------------------- #
#  Failures are reported, never raised
# --------------------------------------------------------------------------- #
def test_a_missing_hardware_module_is_reported_not_raised(
    hardware_config, monkeypatch: pytest.MonkeyPatch, capsys
):
    # Force the import to fail regardless of whether the real module exists in
    # this checkout or was already cached by an earlier test in this process.
    monkeypatch.delattr(jarvis_core_pkg, "hardware", raising=False)
    monkeypatch.setitem(sys.modules, "jarvis.core.hardware", None)

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "not available" in out
    assert "Traceback" not in out


def test_a_missing_planner_module_is_reported_not_raised(
    hardware_config, fake_hardware_module, monkeypatch: pytest.MonkeyPatch, capsys
):
    fake_hardware_module.detect = lambda: _profile()
    fake_hardware_module.summary = lambda p: "..."
    monkeypatch.delattr(jarvis_llm_pkg, "planner", raising=False)
    monkeypatch.setitem(sys.modules, "jarvis.llm.planner", None)

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "not available" in out
    assert "Traceback" not in out


def test_detection_raising_is_reported_not_raised(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    def boom():
        raise RuntimeError("nvidia-smi is not on PATH")

    fake_hardware_module.detect = boom
    fake_hardware_module.summary = lambda p: "unreachable"
    fake_planner_module.plan = lambda cfg: _plan()

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "detection failed" in out
    assert "nvidia-smi is not on PATH" in out
    assert "Traceback" not in out


def test_summary_raising_is_reported_not_raised(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile()

    def boom(p):
        raise RuntimeError("could not format the summary")

    fake_hardware_module.summary = boom
    fake_planner_module.plan = lambda cfg: _plan()

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "detection failed" in out
    assert "could not format the summary" in out
    assert "Traceback" not in out


def test_planning_raising_is_reported_not_raised_and_detection_still_shown(
    hardware_config, fake_hardware_module, fake_planner_module, capsys
):
    fake_hardware_module.detect = lambda: _profile(ram_gb=32.0)
    fake_hardware_module.summary = lambda p: "CPU only, 32.0 GB RAM, 8 cores"

    def boom(cfg):
        raise RuntimeError("no model catalogue entry fits")

    fake_planner_module.plan = boom

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Detected hardware" in out
    assert "CPU only, 32.0 GB RAM, 8 cores" in out
    assert "planning failed" in out
    assert "no model catalogue entry fits" in out
    assert "Traceback" not in out


# --------------------------------------------------------------------------- #
#  Integration: the real jarvis.core.hardware / jarvis.llm.planner
#
#  No monkeypatching of the probes themselves — this just proves cmd_hardware
#  is actually wired to what those modules really return on whatever machine
#  the suite happens to run on (a CPU-only dev box has no GPU, so this is
#  exactly the "RAM-only" path). Detailed per-accelerator behaviour is Part A
#  (tests/test_hardware.py) and Part C's (tests/test_planner.py) job, not this
#  file's.
# --------------------------------------------------------------------------- #
def test_end_to_end_against_the_real_modules_does_not_raise(hardware_config, capsys):
    pytest.importorskip("jarvis.core.hardware")
    pytest.importorskip("jarvis.llm.planner")

    rc = run(["hardware"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Detected hardware" in out
    assert "Plan" in out
    assert "backend" in out
    assert "Traceback" not in out
