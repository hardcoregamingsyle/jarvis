"""Audio-format handling for the voice.

These cover a class of bug invisible in a unit test of the engine alone: an
engine that returns MP3 while the contract promises WAV, and a ``speak()`` that
responds to an unexpected format by going silent.  For a voice assistant,
silence is the worst possible failure, so it is tested explicitly.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import jarvis.core.platform_utils as platform_utils
import jarvis.speech.tts as tts_mod
from jarvis.core.config import TTSConfig
from jarvis.speech.tts import EdgeTTS, NullTTS, create_tts, mp3_to_wav


# A minimal but structurally real MPEG audio frame header.
MP3_MAGIC = b"\xff\xf3\x64\xc4" + b"\x00" * 128


def stub_network(monkeypatch, engine, payload: bytes) -> None:
    """Replace the network round-trip with a fixed payload.

    ``_fetch_bytes`` is the seam between the engine's logic and edge-tts, so
    stubbing it exercises the real format-detection code without a coroutine,
    a socket, or an event loop.
    """
    monkeypatch.setattr(engine, "_fetch_bytes", lambda text: payload)


class FakeFFmpeg:
    """Stands in for ``subprocess.Popen`` running ffmpeg on stdin/stdout."""

    def __init__(self, stdout: bytes, returncode: int = 0):
        self._stdout = stdout
        self.returncode = returncode
        self.killed = False

    def communicate(self, input=None, timeout=None):  # noqa: A002 - Popen's name
        return self._stdout, b""

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


# --------------------------------------------------------------------------- #
#  mp3_to_wav
# --------------------------------------------------------------------------- #
def test_returns_none_without_a_converter(monkeypatch):
    """No decoder available must mean "keep the MP3", not an exception."""
    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setattr(platform_utils, "which", lambda name: None)
    assert mp3_to_wav(MP3_MAGIC) is None


def test_empty_input():
    assert mp3_to_wav(b"") is None


def test_uses_ffmpeg_when_present(monkeypatch):
    fake_wav = b"RIFF" + b"\x00" * 40

    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeFFmpeg(fake_wav))

    assert mp3_to_wav(MP3_MAGIC) == fake_wav


def test_rejects_ffmpeg_output_that_is_not_wav(monkeypatch):
    """A zero exit code is not proof that real audio came back."""
    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: FakeFFmpeg(b"not-a-riff-header"))

    assert mp3_to_wav(MP3_MAGIC) is None


def test_survives_a_broken_ffmpeg(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("ffmpeg vanished")

    monkeypatch.setitem(sys.modules, "pydub", None)
    monkeypatch.setattr(platform_utils, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "Popen", explode)

    assert mp3_to_wav(MP3_MAGIC) is None


# --------------------------------------------------------------------------- #
#  EdgeTTS format handling
# --------------------------------------------------------------------------- #
def test_format_is_decided_by_the_bytes_not_the_request(monkeypatch):
    """The service accepts an output_format kwarg and may silently ignore it."""
    engine = EdgeTTS(TTSConfig())
    stub_network(monkeypatch, engine, MP3_MAGIC)
    monkeypatch.setattr(tts_mod, "mp3_to_wav", lambda data, **kw: None)

    data = engine.synthesize("Good evening.")
    assert data == MP3_MAGIC
    assert engine.output_format == "mp3"


def test_reports_wav_when_conversion_succeeds(monkeypatch):
    engine = EdgeTTS(TTSConfig())
    converted = b"RIFF" + b"\x00" * 40
    stub_network(monkeypatch, engine, MP3_MAGIC)
    monkeypatch.setattr(tts_mod, "mp3_to_wav", lambda data, **kw: converted)

    assert engine.synthesize("Good evening.") == converted
    assert engine.output_format == "wav"


def test_speak_falls_back_to_the_os_rather_than_going_silent(monkeypatch):
    """Unplayable format must still reach the user's speakers somehow."""
    engine = EdgeTTS(TTSConfig())
    stub_network(monkeypatch, engine, MP3_MAGIC)
    monkeypatch.setattr(tts_mod, "mp3_to_wav", lambda data, **kw: None)

    played: list = []
    monkeypatch.setattr(
        tts_mod, "_play_via_os",
        lambda data, suffix=".wav": played.append((data, suffix)),
    )

    engine.speak("Good evening.")
    assert played, "speak() produced no audible output at all"
    assert played[0][1] == ".mp3"


def test_speak_on_empty_text_does_nothing(monkeypatch):
    engine = EdgeTTS(TTSConfig())
    called: list = []
    monkeypatch.setattr(tts_mod, "_play_via_os",
                        lambda data, suffix=".wav": called.append(1))
    engine.speak("")
    assert called == []


# --------------------------------------------------------------------------- #
#  Contract across the family
# --------------------------------------------------------------------------- #
def test_every_engine_declares_an_output_format():
    """Callers rely on this attribute to pick a file extension and a player."""
    for engine in (NullTTS(TTSConfig()), EdgeTTS(TTSConfig())):
        assert getattr(engine, "output_format", None) in ("wav", "mp3")


def test_disabled_tts_yields_the_null_engine():
    cfg = TTSConfig()
    cfg.enabled = False
    assert create_tts(cfg).name == "null"


@pytest.mark.parametrize("text", ["", "   ", "Café — naïve", "x" * 5000, "\n\t"])
def test_synthesize_never_raises_on_odd_input(text):
    assert isinstance(NullTTS(TTSConfig()).synthesize(text), bytes)
