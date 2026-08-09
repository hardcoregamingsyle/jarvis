"""``HardwareConfig`` — manual hardware overrides for auto-detection.

Mirrors the pattern in ``tests/test_config_plumbing.py`` (defaults, config-file
round-trip, environment overrides, and the "config.example.yaml documents
nothing Config lacks" cross-check), scoped to the new ``hardware:`` section
only. ``jarvis.core.hardware`` / ``jarvis.llm.planner`` are what actually
*read* these fields; this file only proves the plumbing between the config
file / environment and ``Config.hardware`` itself is sound.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.core.config import Config, HardwareConfig, load_config


# (field, default) exactly as HardwareConfig declares them.
HARDWARE_FIELDS = [
    ("mode", "auto"),
    ("accelerator", "auto"),
    ("vram_gb", 0.0),
    ("ram_gb", 0.0),
    ("gpu_count", 0),
]


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #
def test_config_has_a_hardware_section():
    assert isinstance(Config().hardware, HardwareConfig)


@pytest.mark.parametrize("name,expected", HARDWARE_FIELDS)
def test_hardware_config_defaults(name, expected):
    hw = HardwareConfig()
    assert hasattr(hw, name), f"HardwareConfig has no {name!r}"
    assert getattr(hw, name) == expected


def test_hardware_defaults_mean_fully_automatic():
    """0 / 'auto' everywhere -- nothing here overrides detection out of the box."""
    hw = HardwareConfig()
    assert hw.mode == "auto"
    assert hw.accelerator == "auto"
    assert hw.vram_gb == 0.0
    assert hw.ram_gb == 0.0
    assert hw.gpu_count == 0


# --------------------------------------------------------------------------- #
#  Config-file round-trip (JSON and YAML, matching the fallback parser)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,_default", HARDWARE_FIELDS)
def test_every_hardware_field_survives_a_json_config_file(tmp_path, name, _default):
    overrides = {
        "mode": "manual",
        "accelerator": "cuda",
        "vram_gb": 8.0,
        "ram_gb": 32.0,
        "gpu_count": 1,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hardware": {name: overrides[name]}}), encoding="utf-8")

    cfg = load_config(path, use_env=False)
    assert getattr(cfg.hardware, name) == overrides[name]


def test_hardware_section_survives_a_yaml_config_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({
            "hardware": {
                "mode": "manual",
                "accelerator": "rocm",
                "vram_gb": 16.0,
                "ram_gb": 64.0,
                "gpu_count": 2,
            }
        }),
        encoding="utf-8",
    )

    cfg = load_config(path, use_env=False)
    assert cfg.hardware.mode == "manual"
    assert cfg.hardware.accelerator == "rocm"
    assert cfg.hardware.vram_gb == pytest.approx(16.0)
    assert cfg.hardware.ram_gb == pytest.approx(64.0)
    assert cfg.hardware.gpu_count == 2


# --------------------------------------------------------------------------- #
#  Environment overrides -- the generic JARVIS_HARDWARE_<FIELD> mechanism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_name,env_value,attr,expected", [
    ("JARVIS_HARDWARE_MODE", "manual", "mode", "manual"),
    ("JARVIS_HARDWARE_ACCELERATOR", "cuda", "accelerator", "cuda"),
    ("JARVIS_HARDWARE_VRAM_GB", "8", "vram_gb", 8.0),
    ("JARVIS_HARDWARE_RAM_GB", "32", "ram_gb", 32.0),
    ("JARVIS_HARDWARE_GPU_COUNT", "2", "gpu_count", 2),
])
def test_hardware_fields_survive_the_environment(env_name, env_value, attr, expected):
    cfg = load_config(environ={env_name: env_value})
    assert getattr(cfg.hardware, attr) == expected


def test_numeric_hardware_settings_are_coerced_from_the_environment():
    cfg = load_config(environ={
        "JARVIS_HARDWARE_VRAM_GB": "8.5",
        "JARVIS_HARDWARE_GPU_COUNT": "1",
    })
    assert cfg.hardware.vram_gb == pytest.approx(8.5)
    assert isinstance(cfg.hardware.vram_gb, float)
    assert cfg.hardware.gpu_count == 1
    assert isinstance(cfg.hardware.gpu_count, int)


def test_hardware_env_overrides_apply_on_top_of_a_config_file(tmp_path):
    """Env beats file, matching every other section's precedence order."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hardware": {"vram_gb": 8.0}}), encoding="utf-8")

    cfg = load_config(path, environ={"JARVIS_HARDWARE_VRAM_GB": "24"})
    assert cfg.hardware.vram_gb == pytest.approx(24.0)


def test_hardware_untouched_env_leaves_defaults_alone():
    cfg = load_config(environ={"JARVIS_LOG_LEVEL": "DEBUG"})
    assert cfg.hardware.mode == "auto"
    assert cfg.hardware.accelerator == "auto"
    assert cfg.hardware.vram_gb == 0.0
    assert cfg.hardware.ram_gb == 0.0
    assert cfg.hardware.gpu_count == 0


# --------------------------------------------------------------------------- #
#  Serialisation
# --------------------------------------------------------------------------- #
def test_to_dict_includes_hardware_section():
    payload = Config().to_dict()
    assert "hardware" in payload
    assert payload["hardware"] == {
        "mode": "auto",
        "accelerator": "auto",
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "gpu_count": 0,
    }


def test_save_and_reload_round_trips_hardware_json(tmp_path):
    cfg = Config()
    cfg.hardware.mode = "manual"
    cfg.hardware.accelerator = "mps"
    cfg.hardware.vram_gb = 8.0
    cfg.hardware.ram_gb = 32.0
    cfg.hardware.gpu_count = 1

    saved_path = cfg.save(tmp_path / "config.json")
    reloaded = load_config(saved_path, use_env=False)

    assert reloaded.hardware.mode == "manual"
    assert reloaded.hardware.accelerator == "mps"
    assert reloaded.hardware.vram_gb == pytest.approx(8.0)
    assert reloaded.hardware.ram_gb == pytest.approx(32.0)
    assert reloaded.hardware.gpu_count == 1


def test_save_and_reload_round_trips_hardware_yaml(tmp_path):
    pytest.importorskip("yaml")
    cfg = Config()
    cfg.hardware.accelerator = "rocm"
    cfg.hardware.vram_gb = 16.0

    saved_path = cfg.save(tmp_path / "config.yaml")
    assert saved_path.suffix == ".yaml"
    reloaded = load_config(saved_path, use_env=False)

    assert reloaded.hardware.accelerator == "rocm"
    assert reloaded.hardware.vram_gb == pytest.approx(16.0)


# --------------------------------------------------------------------------- #
#  config.example.yaml must document exactly what Config declares
# --------------------------------------------------------------------------- #
def test_example_config_documents_a_hardware_section():
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    if not example.exists():
        pytest.skip("config.example.yaml not present")

    text = example.read_text(encoding="utf-8")
    assert "\nhardware:" in text or text.startswith("hardware:")


@pytest.mark.parametrize("name,_default", HARDWARE_FIELDS)
def test_example_config_hardware_fields_are_not_silently_dropped(tmp_path, name, _default):
    """Every key documented under hardware: in the example file must be a real field.

    Same failure mode as the voice: block bug documented in
    test_config_plumbing.py -- a documented-but-undeclared key is dropped in
    total silence by both the file loader and the environment-override pass.
    """
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    if not example.exists():
        pytest.skip("config.example.yaml not present")

    yaml = pytest.importorskip("yaml")
    documented = yaml.safe_load(example.read_text(encoding="utf-8"))
    hardware_block = documented.get("hardware") or {}
    assert name in hardware_block, (
        f"config.example.yaml's hardware: block does not document {name!r}"
    )

    # And it must actually reach Config.hardware, not merely exist in the file.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hardware": {name: hardware_block[name]}}), encoding="utf-8")
    cfg = load_config(path, use_env=False)
    assert getattr(cfg.hardware, name) == hardware_block[name]
