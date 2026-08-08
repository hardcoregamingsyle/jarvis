"""Tests for :mod:`jarvis.speech.audio_io`.

Everything here runs from stdlib + pytest.  No audio device is opened, no
network is touched, and every WAV used as fixture data is built on the fly.
"""

from __future__ import annotations

import io
import struct
import sys
import threading
import time
import types
import wave

import pytest

from jarvis.core.config import STTConfig
from jarvis.speech.audio_io import (
    MIN_NOISE_FLOOR,
    AudioPlayer,
    AudioRecorder,
    UtteranceAssembler,
    is_silent,
    read_wav,
    resample,
    rms,
    wav_bytes,
)


# --------------------------------------------------------------------------- #
#  WAV encode + decode
# --------------------------------------------------------------------------- #
def test_wav_bytes_starts_with_riff_header():
    data = wav_bytes([0.0, 0.1, -0.1], sample_rate=16000)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"


def test_wav_roundtrip_16bit_within_tolerance():
    samples = [0.0, 0.5, -0.5, 0.99, -0.99, 0.25, -0.25, 0.125]
    data = wav_bytes(samples, sample_rate=22050)
    recovered, sr = read_wav(data)
    assert sr == 22050
    assert len(recovered) == len(samples)
    for want, got in zip(samples, recovered):
        assert abs(want - got) < 1e-3


def test_wav_roundtrip_clamps_out_of_range_floats():
    data = wav_bytes([2.0, -3.0, 1.5], sample_rate=16000)
    recovered, _ = read_wav(data)
    assert all(-1.0 <= v <= 1.0 for v in recovered)
    assert recovered[0] > 0.99
    assert recovered[1] < -0.99


def test_wav_bytes_accepts_int_list():
    data = wav_bytes([0, 1000, -1000, 32000, -32000], sample_rate=16000)
    recovered, sr = read_wav(data)
    assert sr == 16000
    # 1000 / 32768 ≈ 0.0305
    assert abs(recovered[1] - 1000 / 32768) < 1e-6
    assert abs(recovered[2] - (-1000 / 32768)) < 1e-6


def test_wav_bytes_accepts_raw_bytes_passthrough():
    raw = struct.pack("<4h", 0, 8000, -8000, 16000)
    data = wav_bytes(raw, sample_rate=8000)
    recovered, sr = read_wav(data)
    assert sr == 8000
    assert len(recovered) == 4
    assert abs(recovered[1] - 8000 / 32768) < 1e-6


def test_read_wav_handles_8bit_pcm():
    data = wav_bytes([0.0, 0.5, -0.5, 1.0, -1.0], sample_rate=16000, sample_width=1)
    recovered, sr = read_wav(data)
    assert sr == 16000
    assert len(recovered) == 5
    # 8-bit quantisation ~1/127
    assert abs(recovered[0]) < 0.02
    assert abs(recovered[1] - 0.5) < 0.02
    assert abs(recovered[2] + 0.5) < 0.02


def test_read_wav_handles_32bit_pcm():
    data = wav_bytes([0.0, 0.5, -0.5, 0.9, -0.9], sample_rate=16000, sample_width=4)
    recovered, sr = read_wav(data)
    assert sr == 16000
    assert len(recovered) == 5
    for want, got in zip([0.0, 0.5, -0.5, 0.9, -0.9], recovered):
        assert abs(want - got) < 1e-6


def test_read_wav_downmixes_stereo_to_mono():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        # Four stereo frames: (L, R) = (1000, 3000), (2000, 4000), (0, 0), (-2000, -4000)
        frames = struct.pack("<8h", 1000, 3000, 2000, 4000, 0, 0, -2000, -4000)
        w.writeframes(frames)
    mono, sr = read_wav(buf.getvalue())
    assert sr == 16000
    assert len(mono) == 4  # 4 stereo frames -> 4 mono samples
    # Averaged, then divided by 32768
    assert abs(mono[0] - (2000 / 32768)) < 1e-6
    assert abs(mono[1] - (3000 / 32768)) < 1e-6
    assert abs(mono[2]) < 1e-9
    assert abs(mono[3] - (-3000 / 32768)) < 1e-6


def test_read_wav_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        read_wav(b"this is definitely not a WAV file")


def test_read_wav_rejects_empty_bytes():
    with pytest.raises(ValueError):
        read_wav(b"")


def test_read_wav_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        read_wav(str(tmp_path / "does_not_exist.wav"))


def test_read_wav_reads_from_disk_path(tmp_path):
    src = tmp_path / "hello.wav"
    src.write_bytes(wav_bytes([0.25, -0.25, 0.5], sample_rate=8000))
    recovered, sr = read_wav(str(src))
    assert sr == 8000
    assert len(recovered) == 3
    assert abs(recovered[0] - 0.25) < 1e-3


def test_read_wav_rejects_unsupported_input_type():
    with pytest.raises(ValueError):
        read_wav(12345)


# --------------------------------------------------------------------------- #
#  Resample
# --------------------------------------------------------------------------- #
def test_resample_identity_returns_new_list_with_same_values():
    src = [0.1, 0.2, 0.3, 0.4]
    out = resample(src, 16000, 16000)
    assert out == src
    assert out is not src   # must be a fresh list


def test_resample_2x_up_produces_double_length():
    out = resample([0.0, 1.0], 8000, 16000)
    assert len(out) == 4
    # First and midpoint samples respect linear interpolation.
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)


def test_resample_2x_down_halves_length():
    src = [0.0, 0.25, 0.5, 0.75]
    out = resample(src, 16000, 8000)
    assert len(out) == 2
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)


def test_resample_empty_returns_empty():
    assert resample([], 16000, 8000) == []


def test_resample_rejects_bad_rates():
    with pytest.raises(ValueError):
        resample([0.0, 1.0], 0, 16000)
    with pytest.raises(ValueError):
        resample([0.0, 1.0], 16000, -1)


# --------------------------------------------------------------------------- #
#  RMS / silence
# --------------------------------------------------------------------------- #
def test_rms_matches_reference_formula():
    assert rms([0.5, -0.5]) == pytest.approx(0.5)
    assert rms([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)


def test_rms_zero_for_empty_and_silence():
    assert rms([]) == 0.0
    assert rms([0.0, 0.0, 0.0]) == 0.0


def test_is_silent_uses_rms_against_threshold():
    quiet = [0.001, -0.001, 0.002, -0.002]
    loud = [0.5, -0.5, 0.5, -0.5]
    assert is_silent(quiet, 0.01) is True
    assert is_silent(loud, 0.01) is False


# --------------------------------------------------------------------------- #
#  AudioRecorder
# --------------------------------------------------------------------------- #
def test_recorder_is_available_false_when_sounddevice_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    rec = AudioRecorder(STTConfig())
    # Must NEVER raise, must be False when the backend cannot import.
    assert rec.is_available() is False


def test_recorder_stop_flag_short_circuits_record_until_silence(monkeypatch):
    fake_sd = types.ModuleType("sounddevice")

    class _Stream:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):  # pragma: no cover - never called once stop is set
            raise AssertionError("read should not run — stop() was called first")

    fake_sd.InputStream = _Stream
    fake_sd.query_devices = lambda: []

    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    rec = AudioRecorder(STTConfig())
    rec.stop()   # arm before recording
    out = rec.record_until_silence(max_seconds=5.0)
    assert out == []


def test_recorder_record_seconds_returns_empty_when_backend_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert AudioRecorder(STTConfig()).record_seconds(0.5) == []


def test_recorder_list_devices_returns_list_on_backend_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert AudioRecorder(STTConfig()).list_devices() == []


def test_recorder_context_manager_calls_stop():
    rec = AudioRecorder(STTConfig())
    with rec:
        assert not rec._stop_event.is_set()
    assert rec._stop_event.is_set()


# --------------------------------------------------------------------------- #
#  AudioPlayer
# --------------------------------------------------------------------------- #
def test_player_is_available_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setitem(sys.modules, "simpleaudio", None)
    monkeypatch.setattr("jarvis.speech.audio_io.which", lambda name: None)
    # winsound is a Windows stdlib module; poison it to force the "no backend"
    # path on any platform.
    monkeypatch.setitem(sys.modules, "winsound", None)
    player = AudioPlayer()
    result = player.is_available()
    assert result is False


def test_player_play_wav_returns_false_when_no_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setitem(sys.modules, "simpleaudio", None)
    monkeypatch.setitem(sys.modules, "winsound", None)
    monkeypatch.setattr("jarvis.speech.audio_io.which", lambda name: None)
    monkeypatch.setattr("jarvis.speech.audio_io.IS_WINDOWS", False)
    monkeypatch.setattr("jarvis.speech.audio_io.IS_LINUX", False)
    player = AudioPlayer()
    assert player.play_wav(wav_bytes([0.0] * 16, sample_rate=16000)) is False


def test_player_play_wav_returns_false_for_empty_data():
    assert AudioPlayer().play_wav(b"") is False


class _FakePlayObj:
    """Simulates a simpleaudio ``PlayObject`` that finishes after N polls."""

    def __init__(self, iterations: int = 3) -> None:
        self.iterations = iterations
        self.is_playing_calls = 0
        self.stop_called = False

    def is_playing(self) -> bool:
        self.is_playing_calls += 1
        return self.is_playing_calls < self.iterations

    def stop(self) -> None:
        self.stop_called = True


def test_simpleaudio_wait_uses_stop_event_and_does_not_spin(monkeypatch):
    """The simpleaudio path MUST poll via ``self._stop_event.wait(0.05)``.

    A bare ``while play_obj.is_playing()`` loop would peg a CPU core; assert
    we go through the event's timed wait between polls with a small,
    non-zero interval.
    """
    fake_play_obj = _FakePlayObj(iterations=4)
    fake_sa = types.ModuleType("simpleaudio")
    fake_sa.play_buffer = lambda audio, ch, sw, fr: fake_play_obj
    monkeypatch.setitem(sys.modules, "simpleaudio", fake_sa)

    player = AudioPlayer()
    wait_calls: list = []
    real_wait = player._stop_event.wait

    def spy_wait(timeout=None):
        wait_calls.append(timeout)
        # Never signal — let the fake play object finish naturally.
        return False

    monkeypatch.setattr(player._stop_event, "wait", spy_wait)

    data = wav_bytes([0.0, 0.1, -0.1, 0.2, -0.2] * 4, sample_rate=16000)
    ok = player._play_simpleaudio(data)

    assert ok is True
    assert wait_calls, "player never went through the stop-event wait — busy loop!"
    for t in wait_calls:
        assert t is not None, "wait() called with no timeout (busy spin)"
        assert 0.0 < t <= 0.5, f"wait interval {t} not a short poll"
    # Poll interval matches the documented constant.
    assert wait_calls[0] == pytest.approx(AudioPlayer._POLL_INTERVAL)
    # And we polled more than once (not a single-shot wait either).
    assert fake_play_obj.is_playing_calls >= 2
    # Silence the unused reference warning.
    assert real_wait is not None


def test_simpleaudio_stop_event_breaks_playback(monkeypatch):
    fake_play_obj = _FakePlayObj(iterations=100)
    fake_sa = types.ModuleType("simpleaudio")
    fake_sa.play_buffer = lambda *a, **kw: fake_play_obj
    monkeypatch.setitem(sys.modules, "simpleaudio", fake_sa)

    player = AudioPlayer()

    # Return True from wait() → signal set → break immediately.
    monkeypatch.setattr(player._stop_event, "wait", lambda timeout=None: True)

    ok = player._play_simpleaudio(wav_bytes([0.0] * 16, sample_rate=16000))
    assert ok is True
    assert fake_play_obj.stop_called is True


def test_simpleaudio_rejects_garbage_wav(monkeypatch):
    fake_sa = types.ModuleType("simpleaudio")
    fake_sa.play_buffer = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "simpleaudio", fake_sa)

    player = AudioPlayer()
    assert player._play_simpleaudio(b"garbage-not-wav") is False


def test_player_is_not_playing_when_idle():
    assert AudioPlayer().is_playing is False


class _BlockingPlayObj:
    """A simpleaudio PlayObject that plays until somebody stops it."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.stop_called = False

    def is_playing(self) -> bool:
        return not self._done.is_set()

    def stop(self) -> None:
        self.stop_called = True
        self._done.set()


def _force_simpleaudio(monkeypatch, play_obj):
    """Make simpleaudio the only available playback backend."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    fake_sa = types.ModuleType("simpleaudio")
    fake_sa.play_buffer = lambda *a, **kw: play_obj
    monkeypatch.setitem(sys.modules, "simpleaudio", fake_sa)


def test_stop_cuts_playback_promptly_and_clears_is_playing(monkeypatch):
    """Barge-in is only believable if stop() returns the speaker at once."""
    play_obj = _BlockingPlayObj()
    _force_simpleaudio(monkeypatch, play_obj)

    player = AudioPlayer()
    outcome: dict = {}
    data = wav_bytes([0.1, -0.1] * 64, sample_rate=16000)

    worker = threading.Thread(
        target=lambda: outcome.update(ok=player.play_wav(data)), daemon=True
    )
    worker.start()

    deadline = time.monotonic() + 2.0
    while player._play_obj is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert player.is_playing is True, "playback never reported itself as active"

    started = time.monotonic()
    player.stop()
    worker.join(timeout=2.0)
    elapsed = time.monotonic() - started

    assert worker.is_alive() is False, "playback did not unwind after stop()"
    assert elapsed < 0.15, f"stop() took {elapsed:.3f}s — far too slow to barge in on"
    assert play_obj.stop_called is True
    assert player.is_playing is False
    assert outcome["ok"] is True


def test_stop_purges_the_winsound_backend(monkeypatch):
    """winsound plays asynchronously, so stop() must explicitly purge it."""
    purged: list = []
    fake_ws = types.ModuleType("winsound")
    fake_ws.SND_MEMORY = 4
    fake_ws.SND_ASYNC = 1
    fake_ws.SND_PURGE = 64
    fake_ws.PlaySound = lambda sound, flags: purged.append((sound, flags))
    monkeypatch.setitem(sys.modules, "winsound", fake_ws)
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setitem(sys.modules, "simpleaudio", None)
    monkeypatch.setattr("jarvis.speech.audio_io.IS_WINDOWS", True)

    player = AudioPlayer()
    assert player.play_wav(wav_bytes([0.0] * 16, sample_rate=16000)) is True
    player.stop()

    assert (None, fake_ws.SND_PURGE) in purged, "the async clip was never silenced"


def test_player_stop_kills_current_process(monkeypatch):
    """AudioPlayer.stop() must terminate any pending CLI subprocess."""
    class _FakeProc:
        def __init__(self):
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waited = True

    player = AudioPlayer()
    proc = _FakeProc()
    player._current_process = proc
    player.stop()
    assert proc.killed is True
    assert proc.waited is True
    assert player._stop_event.is_set()


# --------------------------------------------------------------------------- #
#  Continuous listening: pre-roll, minimum speech, monitor, calibration
# --------------------------------------------------------------------------- #
#  A deliberately small sample rate keeps the fixture buffers readable: at
#  1600 Hz a 50 ms block is 80 samples.
BLOCK = 80


def _cfg(**overrides) -> STTConfig:
    settings = dict(
        sample_rate=1600,
        silence_threshold=0.02,
        silence_duration=0.1,     # -> 2 silent blocks end an utterance
        max_utterance_seconds=5.0,
    )
    settings.update(overrides)
    return STTConfig(**settings)


def _block(value: float, n: int = BLOCK) -> list:
    """A block of constant amplitude, whose RMS is exactly ``abs(value)``."""
    return [value] * n


def _assembler(**overrides) -> UtteranceAssembler:
    settings = dict(
        sample_rate=1600,
        chunk_size=BLOCK,
        silence_threshold=0.02,
        silence_duration=0.1,
        preroll_seconds=0.1,       # -> 2 blocks of history
        min_speech_seconds=0.15,   # -> 3 loud blocks required
        max_utterance_seconds=5.0,
    )
    settings.update(overrides)
    return UtteranceAssembler(**settings)


def test_assembler_prepends_the_preroll_buffer():
    """Without pre-roll the first syllable is always missing."""
    asm = _assembler()
    quiet = [_block(0.001), _block(0.002), _block(0.003)]
    speech = [_block(0.5)] * 3
    trailing = [_block(0.0)] * 2

    out = None
    for chunk in quiet + speech + trailing:
        result = asm.feed(chunk)
        if result is not None:
            out = result

    assert out is not None, "the utterance never ended"
    # Two blocks of history precede the block that tripped detection.
    assert out[0] == pytest.approx(0.002)
    assert out[BLOCK] == pytest.approx(0.003)
    assert out[2 * BLOCK] == pytest.approx(0.5), "the first loud block was lost"
    # 2 pre-roll + 3 speech + 2 trailing silent blocks.
    assert len(out) == 7 * BLOCK


def test_assembler_without_preroll_loses_the_leading_audio():
    """The control case: proves the pre-roll assertion above can fail."""
    asm = _assembler(preroll_seconds=0.0)
    out = None
    for chunk in [_block(0.001)] * 3 + [_block(0.5)] * 3 + [_block(0.0)] * 2:
        result = asm.feed(chunk)
        if result is not None:
            out = result

    assert out is not None
    assert out[0] == pytest.approx(0.5)
    assert len(out) == 5 * BLOCK


def test_assembler_rejects_a_click_but_keeps_real_speech():
    asm = _assembler()
    click = [_block(0.0), _block(0.6)] + [_block(0.0)] * 3
    assert [asm.feed(c) for c in click] == [None] * len(click)
    assert asm.rejected == 1, "a one-block transient was treated as speech"

    speech = [_block(0.5)] * 4 + [_block(0.0)] * 2
    results = [asm.feed(c) for c in speech]
    kept = [r for r in results if r]
    assert len(kept) == 1, "sustained speech was dropped along with the click"
    assert asm.rejected == 1


def test_assembler_caps_an_endless_utterance():
    asm = _assembler(max_utterance_seconds=0.3)   # 6 blocks
    emitted = [asm.feed(_block(0.5)) for _ in range(12)]
    kept = [e for e in emitted if e]
    assert kept, "a speaker who never pauses would never be transcribed"
    assert all(len(u) <= 0.3 * 1600 + BLOCK for u in kept)


def test_assembler_flush_returns_speech_in_progress():
    asm = _assembler()
    for _ in range(4):
        assert asm.feed(_block(0.5)) is None
    assert asm.in_speech is True
    tail = asm.flush()
    assert tail is not None and len(tail) == 4 * BLOCK
    assert asm.in_speech is False
    assert asm.flush() is None


def _fake_mic(recorder: AudioRecorder, chunks, *, loop: bool = False, delay: float = 0.0):
    """Replace the recorder's one device call with a scripted chunk stream."""
    consumed: list = []

    def _iter(chunk_size, *, stop_event=None):
        while True:
            for chunk in chunks:
                if stop_event is not None and stop_event.is_set():
                    return
                if delay:
                    time.sleep(delay)
                consumed.append(chunk)
                yield list(chunk)
            if not loop:
                return

    recorder._iter_chunks = _iter
    return consumed


def test_listen_stream_delivers_utterances_with_preroll():
    rec = AudioRecorder(_cfg(), preroll_seconds=0.1, min_speech_seconds=0.15)
    _fake_mic(rec, [_block(0.001), _block(0.002)] + [_block(0.5)] * 3 + [_block(0.0)] * 2)

    heard: list = []
    delivered = rec.listen_stream(heard.append)

    assert delivered == 1
    assert len(heard) == 1
    assert heard[0][0] == pytest.approx(0.001), "pre-roll was not prepended"
    assert heard[0][2 * BLOCK] == pytest.approx(0.5)


def test_listen_stream_drops_a_click():
    rec = AudioRecorder(_cfg(), preroll_seconds=0.1, min_speech_seconds=0.15)
    _fake_mic(rec, [_block(0.0), _block(0.7)] + [_block(0.0)] * 3)

    heard: list = []
    assert rec.listen_stream(heard.append) == 0
    assert heard == []


def test_listen_stream_stops_promptly_when_asked():
    rec = AudioRecorder(_cfg())
    consumed = _fake_mic(rec, [_block(0.5), _block(0.0)], loop=True)

    calls: list = []

    def should_stop():
        calls.append(1)
        return len(calls) >= 3

    assert rec.listen_stream(lambda utt: None, should_stop) == 0
    # Two polls per block: stopping must not consume the whole stream.
    assert len(consumed) <= 4, f"kept recording for {len(consumed)} blocks after stop"


def test_listen_stream_honours_a_false_return_from_the_handler():
    rec = AudioRecorder(_cfg(), preroll_seconds=0.0, min_speech_seconds=0.05)
    _fake_mic(rec, [_block(0.5), _block(0.0), _block(0.0)], loop=True)

    heard: list = []

    def handler(utterance):
        heard.append(utterance)
        return False

    assert rec.listen_stream(handler) == 1
    assert len(heard) == 1


def test_listen_stream_survives_a_throwing_handler():
    rec = AudioRecorder(_cfg(), preroll_seconds=0.0, min_speech_seconds=0.05)
    _fake_mic(rec, [_block(0.5), _block(0.0), _block(0.0)] * 2)

    def handler(_utterance):
        raise RuntimeError("the transcriber exploded")

    assert rec.listen_stream(handler) == 2


def test_listen_stream_returns_zero_without_a_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    rec = AudioRecorder(_cfg())
    heard: list = []
    assert rec.listen_stream(heard.append) == 0
    assert heard == []


def test_monitor_fires_on_sustained_speech_and_dies_after_stop():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.5)], loop=True, delay=0.005)

    fired = threading.Event()
    thread = rec.start_monitor(fired.set, threshold=0.1, sustain_seconds=0.1)

    assert thread is not None
    assert fired.wait(3.0) is True, "sustained speech did not wake the monitor"
    assert rec.monitor_running is True

    rec.stop_monitor(timeout=3.0)
    assert thread.is_alive() is False, "the monitor thread outlived stop_monitor()"
    assert rec.monitor_running is False


def test_monitor_ignores_levels_below_the_threshold():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.05)] * 30)     # steady room noise, well under 0.4

    fired = threading.Event()
    thread = rec.start_monitor(fired.set, threshold=0.4, sustain_seconds=0.1)
    assert thread is not None
    thread.join(timeout=3.0)

    assert thread.is_alive() is False
    assert fired.is_set() is False, "room noise was mistaken for the user talking"
    rec.stop_monitor(timeout=1.0)


def test_monitor_start_is_idempotent():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.0)], loop=True, delay=0.005)
    first = rec.start_monitor(lambda: None, threshold=0.9)
    second = rec.start_monitor(lambda: None, threshold=0.9)
    try:
        assert first is second
    finally:
        rec.stop_monitor(timeout=3.0)
    assert first is not None and first.is_alive() is False


def test_monitor_survives_a_throwing_callback():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.5)] * 10, delay=0.001)
    calls: list = []

    def boom():
        calls.append(1)
        raise RuntimeError("the player is on fire")

    thread = rec.start_monitor(boom, threshold=0.1, sustain_seconds=0.05)
    assert thread is not None
    thread.join(timeout=3.0)
    assert thread.is_alive() is False
    assert calls, "the callback never ran"


def test_monitor_start_returns_none_without_a_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    rec = AudioRecorder(_cfg())
    fired = threading.Event()
    thread = rec.start_monitor(fired.set)
    # A missing backend must degrade to "no barge-in", never to an exception.
    if thread is not None:
        thread.join(timeout=2.0)
        assert thread.is_alive() is False
    assert fired.is_set() is False
    rec.stop_monitor(timeout=1.0)


def test_calibrate_noise_floor_scales_the_measured_level():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.01)] * 40)
    assert rec.calibrate_noise_floor(0.5, margin=2.0) == pytest.approx(0.02)


def test_calibrate_noise_floor_ignores_a_single_bang():
    rec = AudioRecorder(_cfg())
    # A door slams once during an otherwise quiet second — the median holds.
    _fake_mic(rec, [_block(0.01)] * 9 + [_block(0.9)])
    assert rec.calibrate_noise_floor(0.5, margin=2.0) == pytest.approx(0.02)


def test_calibrate_noise_floor_never_returns_zero():
    rec = AudioRecorder(_cfg())
    _fake_mic(rec, [_block(0.0)] * 20)
    assert rec.calibrate_noise_floor(0.5) == pytest.approx(MIN_NOISE_FLOOR)


def test_calibrate_noise_floor_falls_back_without_a_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    cfg = _cfg(silence_threshold=0.033)
    assert AudioRecorder(cfg).calibrate_noise_floor(0.2) == pytest.approx(0.033)
