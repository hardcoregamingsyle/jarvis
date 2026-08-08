"""Tests for jarvis.speech.tts.

No network, no audio device, no models.  Uses stdlib + pytest only.
"""

from __future__ import annotations

import importlib
import io
import subprocess
import sys
import threading
import time
import wave
from typing import List, Optional

import pytest

from jarvis.core.config import TTSConfig
from jarvis.core.contracts import TTSEngine
from jarvis.speech import tts


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _parse_wav(data: bytes):
    return wave.open(io.BytesIO(data), "rb")


class _FakePopen:
    """Enough of subprocess.Popen to test the ffmpeg branch of mp3_to_wav."""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self._out = stdout_bytes
        self.returncode = returncode
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self._killed = False

    def communicate(self, input=None, timeout=None):
        return (self._out, b"")

    def kill(self) -> None:
        self._killed = True

    def wait(self, timeout=None) -> int:
        return self.returncode


# --------------------------------------------------------------------------- #
#  NullTTS
# --------------------------------------------------------------------------- #
class TestNullTTS:
    def test_produces_parseable_wav_with_correct_header(self):
        engine = tts.NullTTS()
        data = engine.synthesize("hello")
        assert data.startswith(b"RIFF")
        with _parse_wav(data) as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            assert w.getnframes() > 0
            # All samples are silent.
            frames = w.readframes(w.getnframes())
            assert set(frames) == {0}

    def test_accepts_ttsconfig_first_arg(self):
        cfg = TTSConfig()
        engine = tts.NullTTS(cfg)
        assert engine.sample_rate == 16000
        assert engine.cfg is cfg or isinstance(engine.cfg, TTSConfig)

    def test_accepts_int_first_arg_as_sample_rate(self):
        engine = tts.NullTTS(22050)
        assert engine.sample_rate == 22050
        with _parse_wav(engine.synthesize("x")) as w:
            assert w.getframerate() == 22050

    def test_is_always_available_and_has_output_format(self):
        engine = tts.NullTTS()
        assert engine.is_available() is True
        assert engine.output_format == "wav"

    def test_speak_never_raises(self):
        engine = tts.NullTTS()
        engine.speak("some words")


# --------------------------------------------------------------------------- #
#  Factory / availability
# --------------------------------------------------------------------------- #
class TestFactory:
    def test_disabled_returns_null(self):
        cfg = TTSConfig()
        cfg.enabled = False
        assert isinstance(tts.create_tts(cfg), tts.NullTTS)

    def test_auto_reaches_null_when_nothing_installed(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)

        cfg = TTSConfig()
        cfg.engine = "auto"
        cfg.piper_model_path = ""
        engine = tts.create_tts(cfg, voices_dir=tmp_path / "nowhere")
        assert isinstance(engine, tts.NullTTS)

    def test_named_engine_falls_back_when_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)

        cfg = TTSConfig()
        cfg.engine = "edge"
        engine = tts.create_tts(cfg, voices_dir=tmp_path)
        # Should never raise; should end up as NullTTS since nothing works.
        assert isinstance(engine, tts.NullTTS)

    def test_available_tts_engines_always_includes_null(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)
        names = tts.available_tts_engines(TTSConfig())
        assert names[-1] == "null"


# --------------------------------------------------------------------------- #
#  is_available() safety net
# --------------------------------------------------------------------------- #
class TestIsAvailable:
    def test_piper_false_when_model_missing(self, tmp_path):
        cfg = TTSConfig()
        cfg.piper_voice = "definitely-not-a-real-voice-xyz"
        cfg.piper_model_path = ""
        engine = tts.PiperTTS(cfg, voices_dir=tmp_path)
        assert engine.is_available() is False

    def test_piper_false_when_deps_and_binary_missing(self, tmp_path, monkeypatch):
        # Even with a "model file" present, no deps and no binary => False.
        model = tmp_path / "en_GB-alan-medium.onnx"
        model.write_bytes(b"stub-onnx")
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)
        cfg = TTSConfig()
        cfg.piper_model_path = str(model)
        engine = tts.PiperTTS(cfg, voices_dir=tmp_path)
        assert engine.is_available() is False

    def test_piper_is_available_never_hits_network(self, tmp_path, monkeypatch):
        called = []

        def boom(*a, **kw):
            called.append((a, kw))
            raise AssertionError("network call from is_available()")

        # Patch stdlib urllib on the off-chance ensure_voice-like code runs.
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", boom)
        cfg = TTSConfig()
        cfg.piper_voice = "no-such-voice"
        engine = tts.PiperTTS(cfg, voices_dir=tmp_path)
        assert engine.is_available() is False
        assert called == []

    def test_edge_false_when_package_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        engine = tts.EdgeTTS(TTSConfig())
        assert engine.is_available() is False

    def test_pyttsx3_false_when_package_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        engine = tts.Pyttsx3TTS(TTSConfig())
        assert engine.is_available() is False

    def test_espeak_false_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)
        engine = tts.EspeakTTS(TTSConfig())
        assert engine.is_available() is False


# --------------------------------------------------------------------------- #
#  output_format attribute
# --------------------------------------------------------------------------- #
class TestOutputFormatAttribute:
    def test_every_engine_declares_output_format(self, tmp_path):
        cfg = TTSConfig()
        for engine in (
            tts.NullTTS(cfg),
            tts.PiperTTS(cfg, voices_dir=tmp_path),
            tts.EdgeTTS(cfg),
            tts.Pyttsx3TTS(cfg),
            tts.EspeakTTS(cfg),
        ):
            assert getattr(engine, "output_format", None) in ("wav", "mp3")


# --------------------------------------------------------------------------- #
#  british_polish
# --------------------------------------------------------------------------- #
class TestBritishPolish:
    def test_none_and_empty(self):
        assert tts.british_polish(None) == ""
        assert tts.british_polish("") == ""

    def test_percent_and_ampersand(self):
        out = tts.british_polish("50% of R&D")
        assert "%" not in out
        assert "&" not in out
        assert "percent" in out
        assert " and " in out

    def test_letter_wise_acronyms(self):
        out = tts.british_polish("Check the CPU and GPU load, RAM usage, SSD, USB, API, URL.")
        for spaced in ("C P U", "G P U", "R A M", "S S D", "U S B", "A P I", "U R L"):
            assert spaced in out
        assert "CPU" not in out and "GPU" not in out

    def test_abbreviations(self):
        out = tts.british_polish("Dr. Watson, approx. 3 metres. e.g. this.")
        assert "Doctor Watson" in out
        assert "approximately" in out
        assert "Dr." not in out
        assert "for example" in out
        assert "e.g." not in out

    def test_americanised_spellings(self):
        out = tts.british_polish("The color of the center is analyzed.")
        assert "colour" in out
        assert "centre" in out
        assert "analysed" in out
        assert "color" not in out and "center" not in out and "analyzed" not in out

    def test_24h_time_zero_minutes(self):
        out = tts.british_polish("Meeting at 14:00.")
        assert "fourteen hundred hours" in out
        assert "14:00" not in out

    def test_24h_time_nonzero_minutes(self):
        out = tts.british_polish("It's 09:30 sharp.")
        assert "nine thirty" in out
        assert "09:30" not in out

    def test_whitespace_collapsed(self):
        out = tts.british_polish("hello\n\n   world\ttoday")
        assert out == "hello world today"

    def test_idempotent(self):
        samples = [
            "50% of R&D at 14:30",
            "Dr. Watson consults on CPU/GPU with approx. 8 GB RAM",
            "The color center analyzed the URL",
            "It's 09:30 in the theater",
            "  leading  and   trailing   ",
        ]
        for s in samples:
            once = tts.british_polish(s)
            twice = tts.british_polish(once)
            assert once == twice, f"not idempotent: {s!r} -> {once!r} -> {twice!r}"

    def test_unicode_safe(self):
        # Non-ASCII characters must survive.
        text = "Café résumé — naïve façade → 100%"
        out = tts.british_polish(text)
        for ch in "Café résumé naïve façade →":
            assert ch in out
        assert "percent" in out


# --------------------------------------------------------------------------- #
#  EdgeTTS — format decided by bytes, speak() never silent
# --------------------------------------------------------------------------- #
class TestEdgeSynthesize:
    _FAKE_MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00some-mp3-payload"
    _FAKE_WAV = (b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"fmt " + b"\x00" * 40)

    def test_returns_mp3_when_bytes_are_mp3_and_no_decoder(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: self._FAKE_MP3)
        monkeypatch.setattr(tts, "mp3_to_wav", lambda data, **kw: None)

        out = engine.synthesize("anything")
        assert out == self._FAKE_MP3
        assert engine.output_format == "mp3"

    def test_upgrades_to_wav_when_conversion_succeeds(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: self._FAKE_MP3)
        monkeypatch.setattr(tts, "mp3_to_wav", lambda data, **kw: self._FAKE_WAV)

        out = engine.synthesize("hi")
        assert out.startswith(b"RIFF")
        assert engine.output_format == "wav"

    def test_returns_wav_directly_when_upstream_already_wav(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: self._FAKE_WAV)

        out = engine.synthesize("hello")
        assert out == self._FAKE_WAV
        assert engine.output_format == "wav"

    def test_returns_empty_when_fetch_returns_nothing(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: b"")
        assert engine.synthesize("x") == b""

    def test_speak_falls_back_to_os_when_not_wav(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: self._FAKE_MP3)
        monkeypatch.setattr(tts, "mp3_to_wav", lambda data, **kw: None)

        calls: List[dict] = []

        def fake_play(data, *, suffix=".wav"):
            calls.append({"data": data, "suffix": suffix})

        monkeypatch.setattr(tts, "_play_via_os", fake_play)
        engine.speak("hi there")
        assert len(calls) == 1
        assert calls[0]["data"] == self._FAKE_MP3
        assert calls[0]["suffix"] == ".mp3"

    def test_speak_is_silent_no_op_when_synth_empty(self, monkeypatch):
        engine = tts.EdgeTTS(TTSConfig())
        monkeypatch.setattr(engine, "_fetch_bytes", lambda _t: b"")
        called = []
        monkeypatch.setattr(tts, "_play_via_os", lambda *a, **k: called.append(1))
        engine.speak("nothing")
        assert called == []


# --------------------------------------------------------------------------- #
#  mp3_to_wav
# --------------------------------------------------------------------------- #
class TestMp3ToWav:
    def test_returns_none_when_no_decoder_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pydub", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: None)
        from jarvis.speech import windows_speech
        monkeypatch.setattr(windows_speech.SapiTTS, "is_available", lambda self: False)
        assert tts.mp3_to_wav(b"\xff\xfbfake-mp3") is None

    def test_returns_none_for_empty_input(self):
        assert tts.mp3_to_wav(b"") is None

    def test_rejects_non_riff_ffmpeg_output(self, monkeypatch):
        # pydub unavailable, ffmpeg present but returns garbage.
        monkeypatch.setitem(sys.modules, "pydub", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: "ffmpeg-stub")

        def fake_popen(argv, **kwargs):
            return _FakePopen(b"not-a-wav-at-all", returncode=0)

        monkeypatch.setattr(tts.subprocess, "Popen", fake_popen)
        assert tts.mp3_to_wav(b"\xff\xfbfake") is None

    def test_accepts_riff_ffmpeg_output(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pydub", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: "ffmpeg-stub")
        riff = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVEfmt " + b"\x00" * 32

        def fake_popen(argv, **kwargs):
            return _FakePopen(riff, returncode=0)

        monkeypatch.setattr(tts.subprocess, "Popen", fake_popen)
        assert tts.mp3_to_wav(b"\xff\xfbfake") == riff

    def test_rejects_nonzero_return_code(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pydub", None)
        monkeypatch.setattr(tts.platform_utils, "which", lambda name: "ffmpeg-stub")
        riff = b"RIFF" + b"\x00" * 40

        def fake_popen(argv, **kwargs):
            return _FakePopen(riff, returncode=1)

        monkeypatch.setattr(tts.subprocess, "Popen", fake_popen)
        assert tts.mp3_to_wav(b"\xff\xfbfake") is None


# --------------------------------------------------------------------------- #
#  _play_via_os safety
# --------------------------------------------------------------------------- #
class TestPlayViaOs:
    def test_never_raises_on_bad_input(self, monkeypatch):
        # open_path always returns; make it raise to test suppression.
        def boom(target):
            raise RuntimeError("nope")

        monkeypatch.setattr(tts.platform_utils, "open_path", boom)
        tts._play_via_os(b"junk-bytes", suffix=".wav")  # must not raise

    def test_no_op_on_empty(self, monkeypatch):
        called = []
        monkeypatch.setattr(tts.platform_utils, "open_path", lambda t: called.append(t))
        tts._play_via_os(b"", suffix=".wav")
        assert called == []


# --------------------------------------------------------------------------- #
#  SpeechQueue
# --------------------------------------------------------------------------- #
class _RecordingEngine(TTSEngine):
    """Minimal engine that records what it was asked to speak."""

    name = "recording"
    output_format = "wav"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.spoken: List[str] = []
        self.interrupted = threading.Event()
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str) -> bytes:
        return b"RIFF\x00\x00\x00\x00WAVE"

    def speak(self, text: str) -> None:
        # Simulate playback in small slices so stop() can interrupt.
        end = time.monotonic() + self.delay
        while time.monotonic() < end:
            if self.interrupted.is_set():
                return
            time.sleep(0.01)
        with self._lock:
            self.spoken.append(text)

    def stop(self) -> None:
        self.interrupted.set()


class TestSpeechQueue:
    def test_ordering(self):
        engine = _RecordingEngine(delay=0.01)
        q = tts.SpeechQueue(engine)
        try:
            q.say("one")
            q.say("two")
            q.say("three")
            assert q.wait(timeout=5.0)
            assert engine.spoken == ["one", "two", "three"]
        finally:
            q.shutdown(timeout=2.0)

    def test_stop_clears_pending(self):
        engine = _RecordingEngine(delay=0.5)
        q = tts.SpeechQueue(engine)
        try:
            for word in ("a", "b", "c", "d", "e"):
                q.say(word)
            # Let the worker pick up the first item, then barge-in.
            time.sleep(0.1)
            q.stop()
            time.sleep(0.2)
            # After stop(): the currently-playing item was interrupted, and
            # the queued b..e were dropped.  So `spoken` must be strictly
            # shorter than what was queued.
            assert len(engine.spoken) < 5
        finally:
            q.shutdown(timeout=2.0)

    def test_shutdown_kills_worker(self):
        engine = _RecordingEngine(delay=0.0)
        q = tts.SpeechQueue(engine)
        q.say("hello")
        q.wait(timeout=2.0)
        q.shutdown(timeout=2.0)
        # Worker thread must be dead after shutdown.
        assert q._worker.is_alive() is False

    def test_shutdown_is_idempotent(self):
        engine = _RecordingEngine()
        q = tts.SpeechQueue(engine)
        q.shutdown(timeout=2.0)
        q.shutdown(timeout=2.0)
        assert q._worker.is_alive() is False

    def test_say_after_shutdown_is_noop(self):
        engine = _RecordingEngine()
        q = tts.SpeechQueue(engine)
        q.shutdown(timeout=2.0)
        q.say("ignored")
        # Not spoken because shutdown was called first.
        assert engine.spoken == []

    def test_engine_speak_exception_does_not_break_queue(self):
        class Boom(TTSEngine):
            name = "boom"
            output_format = "wav"
            def __init__(self):
                self.count = 0
            def is_available(self): return True
            def synthesize(self, text): return b""
            def speak(self, text):
                self.count += 1
                if self.count == 1:
                    raise RuntimeError("kaboom")

        engine = Boom()
        q = tts.SpeechQueue(engine)
        try:
            q.say("first")
            q.say("second")
            assert q.wait(timeout=5.0)
            assert engine.count == 2   # queue kept going after the first raised
        finally:
            q.shutdown(timeout=2.0)


# --------------------------------------------------------------------------- #
#  Piper synthesis header discipline (issue described in the spec)
# --------------------------------------------------------------------------- #
class TestPiperHeaderDiscipline:
    """A voice that lands writes with an unset sample rate has been the source
    of silent-mute regressions.  Simulate it: fake PiperVoice.load() and check
    that synthesise() opens the wave with the framerate before any writes.
    """

    def test_setframerate_before_writeframes(self, tmp_path, monkeypatch):
        model = tmp_path / "voice.onnx"
        cfg_json = tmp_path / "voice.onnx.json"
        model.write_bytes(b"stub")
        cfg_json.write_text("{}", encoding="utf-8")

        rate_at_first_write: dict = {}

        class FakeVoice:
            class config:
                sample_rate = 22050

            @classmethod
            def load(cls, *a, **k):
                return cls()

            def synthesize(self, text, wav_file):
                # If the wave file has no framerate set, this raises inside
                # writeframes.  Record what we see instead of relying on that.
                rate_at_first_write["rate"] = wav_file.getframerate()
                wav_file.writeframes(b"\x00\x00" * 100)

        fake_module = type(sys)("piper.voice")
        fake_module.PiperVoice = FakeVoice
        parent = type(sys)("piper")
        parent.voice = fake_module
        monkeypatch.setitem(sys.modules, "piper", parent)
        monkeypatch.setitem(sys.modules, "piper.voice", fake_module)

        cfg = TTSConfig()
        cfg.piper_model_path = str(model)
        engine = tts.PiperTTS(cfg, voices_dir=tmp_path)
        data = engine.synthesize("hello")
        assert data.startswith(b"RIFF")
        assert rate_at_first_write.get("rate") == 22050
        # Bytes really parse.
        with _parse_wav(data) as w:
            assert w.getframerate() == 22050
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2


# --------------------------------------------------------------------------- #
#  Sanity: module imports on a bare stdlib (no heavy deps at import time)
# --------------------------------------------------------------------------- #
def test_module_imports_without_heavy_deps(monkeypatch):
    # Simulate a bare stdlib environment: block imports of the heavy deps.
    for name in ("numpy", "torch", "sounddevice", "edge_tts", "pyttsx3",
                 "piper", "piper.voice", "pydub"):
        monkeypatch.setitem(sys.modules, name, None)
    # Reload should still succeed.
    reloaded = importlib.reload(tts)
    assert reloaded.NullTTS is not None
