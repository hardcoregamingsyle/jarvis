"""Documented settings must actually reach the code that reads them.

``_apply_mapping`` and ``_env_overrides`` both skip keys the target dataclass
does not declare (``if not hasattr(target, key): continue``). That is the right
behaviour for junk keys, but it means a field documented in
``config.example.yaml`` and read by a module — yet missing from the dataclass —
is dropped in complete silence, from both the config file and the environment.

Seven ``voice:`` settings were in exactly that state: fully implemented in
``jarvis/voice.py``, documented in the example config, and unreachable. The
continuous and push-to-talk modes could not be selected at all.

These tests pin the dataclass to what the consuming code actually reads.
"""

from __future__ import annotations

import json

import pytest

from jarvis.core.config import Config, VoiceConfig, WindowsConfig, load_config
from jarvis.voice import MODE_CONTINUOUS, MODE_PUSH, MODE_WAKE, VoiceLoop


# (field, default) exactly as jarvis/voice.py falls back to when absent.
VOICE_FIELDS = [
    ("mode", "wake"),
    ("follow_up_seconds", 15.0),
    ("continuous_timeout", 120.0),
    ("interrupt_margin", 2.5),
    ("preroll_seconds", 0.5),
    ("min_speech_seconds", 0.3),
    ("acknowledge", ""),
]


@pytest.mark.parametrize("name,expected", VOICE_FIELDS)
def test_voice_config_declares_every_field_voice_py_reads(name, expected):
    voice = VoiceConfig()
    assert hasattr(voice, name), (
        f"VoiceConfig has no {name!r}; jarvis/voice.py reads it, so config files "
        f"and JARVIS_VOICE_{name.upper()} would be silently ignored"
    )
    assert getattr(voice, name) == expected, (
        f"VoiceConfig.{name} default disagrees with the fallback in voice.py"
    )


@pytest.mark.parametrize("name,_expected", VOICE_FIELDS)
def test_every_voice_field_survives_a_config_file(tmp_path, name, _expected):
    overrides = {
        "mode": "continuous", "follow_up_seconds": 42.0, "continuous_timeout": 7.5,
        "interrupt_margin": 9.0, "preroll_seconds": 1.25,
        "min_speech_seconds": 0.75, "acknowledge": "Yes?",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"voice": {name: overrides[name]}}), encoding="utf-8")

    cfg = load_config(path, use_env=False)
    assert getattr(cfg.voice, name) == overrides[name]


def test_voice_mode_survives_the_environment():
    cfg = load_config(environ={"JARVIS_VOICE_MODE": "continuous"})
    assert cfg.voice.mode == "continuous"


def test_numeric_voice_settings_are_coerced_from_the_environment():
    cfg = load_config(environ={
        "JARVIS_VOICE_PREROLL_SECONDS": "1.25",
        "JARVIS_VOICE_FOLLOW_UP_SECONDS": "30",
    })
    assert cfg.voice.preroll_seconds == pytest.approx(1.25)
    assert isinstance(cfg.voice.preroll_seconds, float)
    assert cfg.voice.follow_up_seconds == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
#  The setting has to reach the consumer, not merely the dataclass
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("configured,expected", [
    ("continuous", MODE_CONTINUOUS),
    ("push", MODE_PUSH),
    ("wake", MODE_WAKE),
    ("CONTINUOUS", MODE_CONTINUOUS),     # normalised
    ("nonsense", MODE_WAKE),             # unknown falls back
])
def test_voice_loop_honours_the_configured_mode(configured, expected):
    cfg = Config()
    cfg.voice.mode = configured
    loop = VoiceLoop(cfg, orchestrator=None, stt=None, recorder=None, calibrate=False)
    assert loop.mode == expected


def test_voice_loop_mode_from_the_environment_end_to_end():
    cfg = load_config(environ={"JARVIS_VOICE_MODE": "push"})
    loop = VoiceLoop(cfg, orchestrator=None, stt=None, recorder=None, calibrate=False)
    assert loop.mode == MODE_PUSH


# --------------------------------------------------------------------------- #
#  The windows: block in config.example.yaml needs a dataclass behind it
# --------------------------------------------------------------------------- #
def test_config_has_a_windows_section():
    assert isinstance(Config().windows, WindowsConfig)


@pytest.mark.parametrize("name,default", [
    ("tray_icon", True),
    ("hotkey_toggle", "ctrl+alt+j"),
    ("hotkey_push_to_talk", "ctrl+alt+space"),
    ("autostart", False),
    ("notifications", True),
])
def test_windows_config_fields(name, default):
    assert getattr(WindowsConfig(), name) == default


def test_windows_settings_survive_file_and_environment(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"windows": {"tray_icon": False}}), encoding="utf-8")
    assert load_config(path, use_env=False).windows.tray_icon is False

    cfg = load_config(environ={"JARVIS_WINDOWS_HOTKEY_TOGGLE": "ctrl+alt+k"})
    assert cfg.windows.hotkey_toggle == "ctrl+alt+k"


def test_the_example_config_documents_nothing_that_is_dropped():
    """Every section in config.example.yaml must exist on Config.

    A documented setting that does nothing is worse than an undocumented one:
    the user configures it, sees no effect, and has no way to tell whether the
    feature or the setting is broken.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    if not example.exists():
        pytest.skip("config.example.yaml not present")

    documented = set()
    for line in example.read_text(encoding="utf-8").splitlines():
        # Top-level keys only: no indentation, ends in a colon, not a comment.
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            key = line.split(":", 1)[0].strip()
            if key and key.isidentifier():
                documented.add(key)

    cfg = Config()
    missing = {k for k in documented if not hasattr(cfg, k)}
    assert not missing, f"config.example.yaml documents sections Config lacks: {sorted(missing)}"
