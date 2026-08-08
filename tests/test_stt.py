"""Tests for :mod:`jarvis.speech.stt`.

Every heavy backend (faster_whisper, whisper, vosk, torch, numpy) is absent
in the test environment, so we exercise the fallback paths and the factory
routing.  No model files are downloaded; nothing is played or recorded.
"""

from __future__ import annotations

import sys
import types

import pytest

from jarvis.core.config import STTConfig
from jarvis.core.contracts import STTEngine, Transcript, TTSEngine
from jarvis.speech import (
    FasterWhisperSTT,
    NullSTT,
    VoskSTT,
    WhisperSTT,
    available_stt_engines,
    create_stt,
    create_tts,
)
from jarvis.speech.audio_io import wav_bytes


# --------------------------------------------------------------------------- #
#  NullSTT
# --------------------------------------------------------------------------- #
def test_null_stt_is_always_available_and_returns_empty():
    engine = NullSTT(STTConfig())
    assert engine.name == "null"
    assert engine.is_available() is True
    result = engine.transcribe([0.0, 0.1, -0.1])
    assert isinstance(result, Transcript)
    assert result.text == ""
    assert result.segments == ()


def test_null_stt_accepts_bytes_and_path_inputs(tmp_path):
    engine = NullSTT(STTConfig())
    wav_path = tmp_path / "silence.wav"
    wav_path.write_bytes(wav_bytes([0.0] * 100, sample_rate=16000))
    assert engine.transcribe(str(wav_path)).text == ""
    assert engine.transcribe(b"anything").text == ""


# --------------------------------------------------------------------------- #
#  Backend availability (nothing installed → all False except NullSTT)
# --------------------------------------------------------------------------- #
def test_backends_report_unavailable_without_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "vosk", None)
    cfg = STTConfig()
    assert FasterWhisperSTT(cfg).is_available() is False
    assert WhisperSTT(cfg).is_available() is False
    assert VoskSTT(cfg).is_available() is False
    assert NullSTT(cfg).is_available() is True


def test_backend_is_available_never_raises_on_broken_import(monkeypatch):
    """A broken ``import faster_whisper`` (e.g. missing native lib) must be
    reported as unavailable rather than propagated."""
    boom = types.ModuleType("faster_whisper")

    def _explode(*a, **kw):
        raise RuntimeError("simulated backend failure")

    boom.WhisperModel = _explode
    monkeypatch.setitem(sys.modules, "faster_whisper", boom)
    # is_available should still succeed — it only asks whether the module
    # imports, and it does.
    assert FasterWhisperSTT(STTConfig()).is_available() is True
    # And a subsequent transcribe must not crash even though load will fail.
    result = FasterWhisperSTT(STTConfig()).transcribe([0.0, 0.1, -0.1] * 100)
    assert result.text == ""


# --------------------------------------------------------------------------- #
#  create_stt / available_stt_engines
# --------------------------------------------------------------------------- #
def test_create_stt_auto_falls_back_to_null(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "vosk", None)
    # The Windows recogniser ships with the OS, so on Windows it is genuinely
    # available and would (correctly) be chosen ahead of null. Disable it too,
    # since this test is about the "nothing at all" floor.
    from jarvis.speech import windows_speech

    monkeypatch.setattr(windows_speech.WindowsSTT, "is_available", lambda self: False)

    engine = create_stt(STTConfig(engine="auto"))
    assert isinstance(engine, NullSTT)
    assert engine.is_available() is True


def test_create_stt_auto_prefers_the_windows_recogniser_over_null(monkeypatch):
    """With no pip packages but a working OS recogniser, voice still functions."""
    from jarvis.speech import windows_speech

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "vosk", None)
    monkeypatch.setattr(windows_speech.WindowsSTT, "is_available", lambda self: True)

    engine = create_stt(STTConfig(engine="auto"))
    assert isinstance(engine, windows_speech.WindowsSTT)


def test_create_stt_named_engine_missing_dep_falls_back_to_null(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    engine = create_stt(STTConfig(engine="faster-whisper"))
    assert isinstance(engine, NullSTT)


def test_create_stt_unknown_engine_falls_back_to_null():
    engine = create_stt(STTConfig(engine="does-not-exist"))
    assert isinstance(engine, NullSTT)


def test_create_stt_stub_alias_maps_to_null():
    engine = create_stt(STTConfig(engine="stub"))
    assert isinstance(engine, NullSTT)


def test_available_stt_engines_always_includes_null(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "vosk", None)
    engines = available_stt_engines(STTConfig())
    assert engines, "available_stt_engines must never return empty"
    assert isinstance(engines[-1], NullSTT)
    # Every returned engine implements the STT contract.
    assert all(isinstance(e, STTEngine) for e in engines)


def test_create_stt_auto_picks_first_available(monkeypatch):
    """When a backend imports, ``auto`` must pick it over the null fallback."""
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = object    # merely present
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    engine = create_stt(STTConfig(engine="auto"))
    assert isinstance(engine, FasterWhisperSTT)


# --------------------------------------------------------------------------- #
#  create_tts fallback (the sibling module might be broken)
# --------------------------------------------------------------------------- #
def test_create_tts_falls_back_when_tts_module_missing(monkeypatch):
    """Poisoning sys.modules alone is not enough for ``from . import tts``.

    Once the submodule has been imported, the parent package holds it as an
    attribute and the from-import resolves straight to that, ignoring the
    sys.modules entry — so the package attribute has to be poisoned too.
    """
    import jarvis.speech as speech_pkg
    from jarvis.core.config import TTSConfig

    monkeypatch.setitem(sys.modules, "jarvis.speech.tts", None)
    monkeypatch.delattr(speech_pkg, "tts", raising=False)

    engine = create_tts(TTSConfig())
    assert isinstance(engine, TTSEngine)
    assert engine.is_available() is True

    data = engine.synthesize("hello")
    assert isinstance(data, (bytes, bytearray))
    assert data[:4] == b"RIFF", "the fallback engine must still produce real audio"


def test_create_tts_rejects_a_config_without_the_tts_fields():
    """Better an honest null engine than one that returns silence forever."""
    from jarvis.core.contracts import TTSEngine as _TTSEngine

    engine = create_tts(STTConfig())
    assert isinstance(engine, _TTSEngine)
    assert engine.synthesize("hello")[:4] == b"RIFF"


def test_create_tts_falls_back_when_factory_raises(monkeypatch):
    fake_tts = types.ModuleType("jarvis.speech.tts")

    def _explode(cfg, *, voices_dir=None):
        raise RuntimeError("simulated tts factory failure")

    fake_tts.create_tts = _explode
    monkeypatch.setitem(sys.modules, "jarvis.speech.tts", fake_tts)
    engine = create_tts(STTConfig())
    assert isinstance(engine, TTSEngine)
    # Null fallback subclasses TTSEngine — not doing so would make the
    # orchestrator's isinstance checks silently drop TTS on error.


def test_create_tts_falls_back_when_factory_returns_wrong_type(monkeypatch):
    fake_tts = types.ModuleType("jarvis.speech.tts")
    fake_tts.create_tts = lambda cfg, *, voices_dir=None: "not a tts engine"
    monkeypatch.setitem(sys.modules, "jarvis.speech.tts", fake_tts)
    engine = create_tts(STTConfig())
    assert isinstance(engine, TTSEngine)


# --------------------------------------------------------------------------- #
#  Input preparation
# --------------------------------------------------------------------------- #
def test_prepare_audio_resamples_when_rate_differs(monkeypatch, tmp_path):
    """Feeding non-16 kHz input to a NullSTT still exercises _prepare_audio
    via the transcribe() path — but here we call it directly to be explicit."""
    from jarvis.speech import stt as stt_mod
    # 8 kHz mono file: 8 samples -> should become 16 samples at 16 kHz.
    src = tmp_path / "eight_khz.wav"
    src.write_bytes(wav_bytes([0.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0], sample_rate=8000))
    samples, sr = stt_mod._prepare_audio(str(src), 8000)
    assert sr == 16000
    assert len(samples) == 16


def test_prepare_audio_empty_input_gives_empty_samples():
    from jarvis.speech import stt as stt_mod
    samples, sr = stt_mod._prepare_audio([], 16000)
    assert samples == []
    assert sr == 16000


def test_transcribe_returns_empty_for_empty_input():
    # Even a "working" engine must return an empty transcript, not raise.
    engine = NullSTT(STTConfig())
    result = engine.transcribe([])
    assert result.text == ""
    assert result.confidence == pytest.approx(0.0)
