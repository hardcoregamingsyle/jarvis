"""Audio I/O primitives: WAV encode/decode, resample, VAD helpers, and
optional microphone/speaker devices.

The heavyweight audio backends (``sounddevice``, ``simpleaudio``, ``numpy``,
``winsound``) are all imported lazily inside the function that needs them.
Nothing here refuses to import on a bare stdlib Python — encoding, decoding,
resampling and VAD all fall back to pure-Python implementations.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union

from ..core.config import STTConfig
from ..core.platform_utils import IS_LINUX, IS_WINDOWS, which

log = logging.getLogger(__name__)

# Block size used by every streaming path.  50 ms is short enough that a stop
# request is honoured almost immediately and long enough that RMS is a stable
# estimate of loudness.
CHUNK_MS = 50.0

# A digitally-silent microphone would otherwise calibrate to a threshold of
# zero, which makes every sample "speech".
MIN_NOISE_FLOOR = 0.002


# --------------------------------------------------------------------------- #
#  Small numerical helpers
# --------------------------------------------------------------------------- #
def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _looks_floaty(seq: Iterable[Any]) -> bool:
    """Detect a float sequence without importing numpy."""
    for i, v in enumerate(seq):
        if isinstance(v, float):
            return True
        if i > 8:
            break
    return False


# --------------------------------------------------------------------------- #
#  Encoding
# --------------------------------------------------------------------------- #
def _samples_to_raw(samples: Any, sample_width: int) -> bytes:
    """Convert accepted sample forms to raw little-endian PCM bytes."""
    if isinstance(samples, (bytes, bytearray, memoryview)):
        return bytes(samples)

    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore

    if np is not None:
        try:
            if isinstance(samples, np.ndarray):
                if samples.dtype.kind == "f":
                    clipped = np.clip(samples, -1.0, 1.0)
                    if sample_width == 1:
                        return (np.round(clipped * 127.0) + 128.0).astype("<u1").tobytes()
                    if sample_width == 2:
                        return np.round(clipped * 32767.0).astype("<i2").tobytes()
                    if sample_width == 4:
                        scaled = np.clip(clipped * 2147483647.0, -2147483648.0, 2147483647.0)
                        return np.round(scaled).astype("<i4").tobytes()
                if sample_width == 1:
                    return np.asarray(samples, dtype="<u1").tobytes()
                if sample_width == 2:
                    return np.asarray(samples, dtype="<i2").tobytes()
                if sample_width == 4:
                    return np.asarray(samples, dtype="<i4").tobytes()
        except Exception as exc:
            log.debug("numpy fast path failed, falling back to stdlib: %s", exc)

    seq = list(samples)
    if not seq:
        return b""

    if _looks_floaty(seq):
        floats = [_clip(float(s), -1.0, 1.0) for s in seq]
        if sample_width == 1:
            return struct.pack(
                f"<{len(floats)}B",
                *(int(round(s * 127.0 + 128.0)) for s in floats),
            )
        if sample_width == 2:
            return struct.pack(
                f"<{len(floats)}h",
                *(int(round(s * 32767.0)) for s in floats),
            )
        if sample_width == 4:
            return struct.pack(
                f"<{len(floats)}i",
                *(
                    int(round(_clip(s * 2147483647.0, -2147483648.0, 2147483647.0)))
                    for s in floats
                ),
            )
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    ints = [int(s) for s in seq]
    if sample_width == 1:
        return struct.pack(f"<{len(ints)}B", *ints)
    if sample_width == 2:
        return struct.pack(f"<{len(ints)}h", *ints)
    if sample_width == 4:
        return struct.pack(f"<{len(ints)}i", *ints)
    raise ValueError(f"Unsupported sample_width: {sample_width}")


def wav_bytes(
    samples: Any,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> bytes:
    """Encode ``samples`` as a valid little-endian PCM WAV.

    ``samples`` may be a bytes-like of raw PCM in ``sample_width`` format, a
    numpy array (float or int), a list of ints, or a list of floats in the
    range ``[-1.0, 1.0]``.  Floats are clamped and quantised.
    """
    if sample_width not in (1, 2, 4):
        raise ValueError(f"Unsupported sample_width: {sample_width}")
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    raw = _samples_to_raw(samples, sample_width)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(int(sample_rate))
        w.writeframes(raw)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Decoding
# --------------------------------------------------------------------------- #
def read_wav(path_or_bytes: Any) -> Tuple[List[float], int]:
    """Decode a WAV to mono ``float`` samples in ``[-1, 1]`` plus the sample rate.

    Handles 8/16/32-bit signed/unsigned PCM and stereo→mono downmix.
    ``audioop`` was removed in Python 3.13, so all conversions are done here
    with :mod:`struct`.  Non-WAV or compressed inputs raise :class:`ValueError`.
    """
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        src: Union[str, io.BytesIO] = io.BytesIO(bytes(path_or_bytes))
    elif isinstance(path_or_bytes, (str, Path)):
        src = str(path_or_bytes)
    elif hasattr(path_or_bytes, "read"):
        src = path_or_bytes
    else:
        raise ValueError(
            f"read_wav: unsupported input type {type(path_or_bytes).__name__!r}"
        )

    try:
        with wave.open(src, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except wave.Error as exc:
        raise ValueError(f"Not a readable WAV file: {exc}") from exc
    except EOFError as exc:
        raise ValueError(f"Truncated WAV: {exc}") from exc
    except FileNotFoundError as exc:
        raise ValueError(f"WAV not found: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read WAV: {exc}") from exc

    if sampwidth == 1:
        # 8-bit WAV is UNSIGNED, silence at 128.
        count = len(raw)
        ints = struct.unpack(f"<{count}B", raw) if count else ()
        floats = [(v - 128) / 128.0 for v in ints]
    elif sampwidth == 2:
        count = len(raw) // 2
        ints = struct.unpack(f"<{count}h", raw[: count * 2]) if count else ()
        floats = [v / 32768.0 for v in ints]
    elif sampwidth == 4:
        count = len(raw) // 4
        ints = struct.unpack(f"<{count}i", raw[: count * 4]) if count else ()
        floats = [v / 2147483648.0 for v in ints]
    elif sampwidth == 3:
        # 24-bit PCM: three little-endian bytes, signed.
        count = len(raw) // 3
        floats = []
        for i in range(count):
            b0, b1, b2 = raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val & 0x800000:
                val -= 0x1000000
            floats.append(val / 8388608.0)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if nchannels > 1:
        mono: List[float] = []
        step = nchannels
        for i in range(0, len(floats) - step + 1, step):
            mono.append(sum(floats[i : i + step]) / step)
        floats = mono

    return floats, framerate


# --------------------------------------------------------------------------- #
#  Resample / VAD helpers
# --------------------------------------------------------------------------- #
def resample(samples: Any, src_rate: int, dst_rate: int) -> list:
    """Linear-interpolation resampler.  Returns ``list(samples)`` when rates match."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    seq = samples if isinstance(samples, list) else list(samples)
    if src_rate == dst_rate:
        return list(seq)
    n_in = len(seq)
    if n_in == 0:
        return []
    if n_in == 1:
        return [float(seq[0])]

    try:
        import numpy as np  # type: ignore
        arr = np.asarray(seq, dtype=np.float64)
        out_len = int(round(n_in * (dst_rate / src_rate)))
        if out_len <= 0:
            return []
        idx = np.arange(out_len, dtype=np.float64) * (src_rate / dst_rate)
        xp = np.arange(n_in, dtype=np.float64)
        return np.interp(idx, xp, arr).tolist()
    except ImportError:
        pass

    ratio = src_rate / dst_rate
    out_len = int(round(n_in * (dst_rate / src_rate)))
    if out_len <= 0:
        return []
    out: List[float] = []
    last_idx = n_in - 1
    for i in range(out_len):
        src_i = i * ratio
        lo = int(src_i)
        if lo >= last_idx:
            out.append(float(seq[last_idx]))
            continue
        frac = src_i - lo
        out.append(float(seq[lo]) * (1.0 - frac) + float(seq[lo + 1]) * frac)
    return out


def rms(samples: Any) -> float:
    """Root-mean-square level of a sample buffer."""
    try:
        import numpy as np  # type: ignore
        if hasattr(samples, "dtype") or isinstance(samples, (list, tuple)):
            arr = np.asarray(samples, dtype=np.float64)
            if arr.size == 0:
                return 0.0
            return float((arr * arr).mean() ** 0.5)
    except ImportError:
        pass
    seq = list(samples)
    if not seq:
        return 0.0
    total = 0.0
    for s in seq:
        f = float(s)
        total += f * f
    return (total / len(seq)) ** 0.5


def is_silent(samples: Any, threshold: float) -> bool:
    """True when the RMS energy is below ``threshold``."""
    return rms(samples) < float(threshold)


# --------------------------------------------------------------------------- #
#  Continuous listening: the VAD state machine
# --------------------------------------------------------------------------- #
class UtteranceAssembler:
    """Turns a stream of fixed-size blocks into complete utterances.

    Pure logic — no device, no threads, no dependencies — so the awkward parts
    (pre-roll, minimum speech duration, end-of-utterance) can be exercised
    without a microphone.

    Two details make an open microphone usable rather than merely working:

    * **Pre-roll.**  Energy detection necessarily trips *after* the speaker has
      begun, so the block that crosses the threshold already has the first
      syllable half-gone.  A ring buffer of the preceding blocks is prepended
      to every utterance, which is why "Jarvis" survives instead of arriving as
      "-arvis".
    * **Minimum speech duration.**  A keyboard click or a cough easily clears
      an energy threshold for one block.  An utterance is only released once
      enough *loud* blocks have accumulated.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        chunk_size: int = 800,
        silence_threshold: float = 0.015,
        silence_duration: float = 0.9,
        preroll_seconds: float = 0.5,
        min_speech_seconds: float = 0.3,
        max_utterance_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.chunk_size = max(1, int(chunk_size))
        self.silence_threshold = float(silence_threshold)
        chunk_seconds = self.chunk_size / float(self.sample_rate)
        self.chunk_seconds = chunk_seconds

        preroll_chunks = max(0, int(round(max(0.0, preroll_seconds) / chunk_seconds)))
        # +1 so the ring still holds ``preroll_seconds`` of history *before*
        # the block that triggered detection, which is itself in the ring.
        self._preroll: deque = deque(maxlen=preroll_chunks + 1)
        self.preroll_chunks = preroll_chunks

        self.silence_chunks = max(1, int(round(max(0.0, silence_duration) / chunk_seconds)))
        self.min_speech_chunks = max(
            1, int(math.ceil(max(0.0, min_speech_seconds) / chunk_seconds))
        )
        self.max_samples = max(
            self.chunk_size,
            int(max(0.0, max_utterance_seconds) * self.sample_rate),
        )

        self._collected: List[float] = []
        self._speaking = False
        self._speech_chunks = 0
        self._silent_run = 0
        self.rejected = 0

    @property
    def in_speech(self) -> bool:
        """True while an utterance is being accumulated."""
        return self._speaking

    def reset(self) -> None:
        """Drop any partial utterance and the pre-roll history."""
        self._collected = []
        self._speaking = False
        self._speech_chunks = 0
        self._silent_run = 0
        self._preroll.clear()

    def feed(self, chunk: Any) -> Optional[List[float]]:
        """Consume one block; return a finished utterance or None."""
        block = list(chunk)
        if not block:
            return None
        loud = not is_silent(block, self.silence_threshold)

        if not self._speaking:
            self._preroll.append(block)
            if not loud:
                return None
            self._collected = [s for held in self._preroll for s in held]
            self._preroll.clear()
            self._speaking = True
            self._speech_chunks = 1
            self._silent_run = 0
            return None

        self._collected.extend(block)
        if loud:
            self._speech_chunks += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
            if self._silent_run >= self.silence_chunks:
                return self._finish()
        if len(self._collected) >= self.max_samples:
            return self._finish()
        return None

    def flush(self) -> Optional[List[float]]:
        """End the utterance in progress, if there is one worth keeping."""
        if not self._speaking:
            return None
        return self._finish()

    def _finish(self) -> Optional[List[float]]:
        utterance = self._collected
        speech_chunks = self._speech_chunks
        self.reset()
        if speech_chunks < self.min_speech_chunks:
            self.rejected += 1
            log.debug(
                "discarded %.2fs of audio: only %d speech blocks, %d required",
                len(utterance) / float(self.sample_rate),
                speech_chunks,
                self.min_speech_chunks,
            )
            return None
        return utterance


# --------------------------------------------------------------------------- #
#  Microphone
# --------------------------------------------------------------------------- #
class AudioRecorder:
    """Microphone access: one-shot recordings, an open-mic stream, and a
    level monitor for barge-in.  ``sounddevice`` is imported lazily.

    Every device call funnels through :meth:`_iter_chunks`, the one place that
    touches the audio backend.
    """

    def __init__(
        self,
        cfg: STTConfig,
        *,
        preroll_seconds: Optional[float] = None,
        min_speech_seconds: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self._stop_event = threading.Event()
        self.preroll_seconds = float(
            preroll_seconds
            if preroll_seconds is not None
            else getattr(cfg, "preroll_seconds", 0.5)
        )
        self.min_speech_seconds = float(
            min_speech_seconds
            if min_speech_seconds is not None
            else getattr(cfg, "min_speech_seconds", 0.3)
        )
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_lock = threading.RLock()

    def __enter__(self) -> "AudioRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def is_available(self) -> bool:
        """True when the audio backend is importable and reports a device."""
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return False
        try:
            _ = sd.query_devices()
            return True
        except Exception:
            # Backend imported but no device / driver — treat as unavailable
            # rather than throwing at call sites.
            return False

    def stop(self) -> None:
        """Signal in-flight recordings to break out of their poll loops."""
        self._stop_event.set()

    def chunk_size(self) -> int:
        """Samples per streaming block."""
        return max(1, int(int(self.cfg.sample_rate) * CHUNK_MS / 1000.0))

    def list_devices(self) -> list:
        """List host audio devices, empty list on failure."""
        try:
            import sounddevice as sd  # type: ignore
            return list(sd.query_devices())
        except Exception:
            return []

    def record_seconds(self, n: float) -> list:
        """Record exactly ``n`` seconds; returns [] if the backend is missing."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            return []
        try:
            import numpy as np  # type: ignore
        except ImportError:
            np = None  # type: ignore
        self._stop_event.clear()
        frames = max(1, int(float(n) * self.cfg.sample_rate))
        try:
            data = sd.rec(
                frames,
                samplerate=self.cfg.sample_rate,
                channels=1,
                dtype="float32",
                device=self.cfg.input_device,
            )
            sd.wait()
        except Exception as exc:
            log.warning("record_seconds failed: %s", exc)
            return []
        if np is not None and hasattr(data, "flatten"):
            return data.flatten().tolist()
        return list(data)

    def record_until_silence(self, max_seconds: Optional[float] = None) -> list:
        """Record until ``silence_duration`` of quiet or the hard cap fires.

        The hard cap is always applied — a broken or muted microphone must
        never leave this method spinning forever.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            return []
        self._stop_event.clear()

        hard_cap = float(self.cfg.max_utterance_seconds)
        if max_seconds is not None:
            hard_cap = min(hard_cap, float(max_seconds))
        if hard_cap <= 0:
            return []

        sr = int(self.cfg.sample_rate)
        chunk_ms = 50.0
        chunk_size = max(1, int(sr * chunk_ms / 1000.0))
        silence_chunks = max(1, int(self.cfg.silence_duration * 1000.0 / chunk_ms))
        max_chunks = max(1, int(hard_cap * 1000.0 / chunk_ms))

        collected: List[float] = []
        silent_run = 0
        started = False
        start_ts = time.monotonic()

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                device=self.cfg.input_device,
            ) as stream:
                for _ in range(max_chunks):
                    if self._stop_event.is_set():
                        break
                    if time.monotonic() - start_ts > hard_cap:
                        break
                    try:
                        block, _overflow = stream.read(chunk_size)
                    except Exception as exc:
                        log.warning("mic read failed: %s", exc)
                        break
                    samples = list(block.flatten()) if hasattr(block, "flatten") else list(block)
                    collected.extend(samples)
                    if is_silent(samples, self.cfg.silence_threshold):
                        silent_run += 1
                        if started and silent_run >= silence_chunks:
                            break
                    else:
                        started = True
                        silent_run = 0
        except Exception as exc:
            log.warning("record_until_silence failed: %s", exc)
            return collected

        return collected

    # -- continuous listening ---------------------------------------------- #
    def _iter_chunks(
        self,
        chunk_size: int,
        *,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[List[float]]:
        """Yield blocks of ``chunk_size`` mono float samples from the microphone.

        The single point in this class that touches the audio backend: it
        yields nothing at all when ``sounddevice`` is missing or the device
        refuses to open, and it never raises.  Tests replace it to feed
        synthetic audio.
        """
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return
        try:
            stream = sd.InputStream(
                samplerate=int(self.cfg.sample_rate),
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                device=self.cfg.input_device,
            )
        except Exception as exc:
            log.warning("could not open the microphone: %s", exc)
            return
        try:
            with stream:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        block, _overflow = stream.read(chunk_size)
                    except Exception as exc:
                        log.warning("mic read failed: %s", exc)
                        return
                    if hasattr(block, "flatten"):
                        yield list(block.flatten())
                    else:
                        yield list(block)
        except Exception as exc:
            log.warning("microphone stream failed: %s", exc)
            return

    def _assembler(self) -> UtteranceAssembler:
        return UtteranceAssembler(
            sample_rate=int(self.cfg.sample_rate),
            chunk_size=self.chunk_size(),
            silence_threshold=float(self.cfg.silence_threshold),
            silence_duration=float(self.cfg.silence_duration),
            preroll_seconds=self.preroll_seconds,
            min_speech_seconds=self.min_speech_seconds,
            max_utterance_seconds=float(self.cfg.max_utterance_seconds),
        )

    def listen_stream(
        self,
        on_utterance: Callable[[List[float]], Any],
        should_stop: Optional[Callable[[], bool]] = None,
        *,
        max_utterances: Optional[int] = None,
    ) -> int:
        """Hold the microphone open and hand over each utterance as it ends.

        ``on_utterance`` receives the samples, pre-roll included; returning
        ``False`` from it ends the stream.  ``should_stop`` is polled once per
        block (50 ms), so stopping is prompt.  Returns the number of utterances
        delivered; returns 0 rather than raising when no device is available.
        """
        self._stop_event.clear()
        assembler = self._assembler()
        delivered = 0
        halted = False

        def _finished() -> bool:
            if self._stop_event.is_set():
                return True
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception:  # noqa: BLE001 - a broken predicate ends the stream
                log.exception("should_stop() failed; ending the listen stream")
                return True

        try:
            for chunk in self._iter_chunks(
                self.chunk_size(), stop_event=self._stop_event
            ):
                if _finished():
                    halted = True
                    break
                utterance = assembler.feed(chunk)
                if utterance:
                    delivered += 1
                    if self._deliver(on_utterance, utterance) is False:
                        halted = True
                        break
                    if max_utterances is not None and delivered >= max_utterances:
                        halted = True
                        break
                if _finished():
                    halted = True
                    break
        except Exception:  # noqa: BLE001 - an open mic must not take the loop down
            log.exception("listen_stream failed")
            return delivered

        if not halted:
            tail = assembler.flush()
            if tail:
                delivered += 1
                self._deliver(on_utterance, tail)
        return delivered

    @staticmethod
    def _deliver(callback: Callable[[List[float]], Any], utterance: List[float]) -> Any:
        try:
            return callback(utterance)
        except Exception:  # noqa: BLE001 - a bad handler must not stop the mic
            log.exception("utterance handler failed")
            return None

    # -- level monitor (barge-in) ------------------------------------------ #
    @property
    def monitor_running(self) -> bool:
        thread = self._monitor_thread
        return thread is not None and thread.is_alive()

    def start_monitor(
        self,
        on_speech_detected: Callable[[], Any],
        *,
        threshold: Optional[float] = None,
        sustain_seconds: float = 0.2,
        rearm_seconds: float = 0.6,
    ) -> Optional[threading.Thread]:
        """Watch the input level on a background thread and call back on speech.

        Deliberately does no transcription: this runs *while JARVIS is
        speaking*, and its only job is to notice that the user has started
        talking so playback can be cut.  ``threshold`` is an RMS level — the
        caller passes the ambient floor times its barge-in margin so JARVIS's
        own voice leaking through the speakers does not trigger it.

        Returns the monitor thread.  With no audio backend the thread starts,
        finds nothing to read and exits: barge-in degrades to "off" rather
        than to an exception.
        """
        with self._monitor_lock:
            if self.monitor_running:
                return self._monitor_thread
            level = (
                float(threshold)
                if threshold is not None
                else max(float(self.cfg.silence_threshold) * 2.5, MIN_NOISE_FLOOR)
            )
            chunk_seconds = self.chunk_size() / float(max(1, int(self.cfg.sample_rate)))
            sustain = max(1, int(math.ceil(max(0.0, sustain_seconds) / chunk_seconds)))
            rearm = max(1, int(math.ceil(max(0.0, rearm_seconds) / chunk_seconds)))
            self._monitor_stop.clear()
            thread = threading.Thread(
                target=self._monitor_run,
                args=(on_speech_detected, level, sustain, rearm),
                name="jarvis-mic-monitor",
                daemon=True,
            )
            self._monitor_thread = thread
            thread.start()
            return thread

    def stop_monitor(self, *, timeout: float = 2.0) -> None:
        """Stop the level monitor.  Its thread is dead when this returns."""
        with self._monitor_lock:
            thread = self._monitor_thread
            self._monitor_stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("microphone monitor did not stop within %.1fs", timeout)
        with self._monitor_lock:
            if self._monitor_thread is thread:
                self._monitor_thread = None

    def _monitor_run(
        self,
        callback: Callable[[], Any],
        threshold: float,
        sustain_chunks: int,
        rearm_chunks: int,
    ) -> None:
        loud_run = 0
        quiet_run = 0
        armed = True
        try:
            for chunk in self._iter_chunks(
                self.chunk_size(), stop_event=self._monitor_stop
            ):
                if self._monitor_stop.is_set():
                    break
                if rms(chunk) >= threshold:
                    loud_run += 1
                    quiet_run = 0
                    if armed and loud_run >= sustain_chunks:
                        armed = False
                        loud_run = 0
                        try:
                            callback()
                        except Exception:  # noqa: BLE001
                            log.exception("barge-in callback failed")
                else:
                    loud_run = 0
                    quiet_run += 1
                    if quiet_run >= rearm_chunks:
                        armed = True
                if self._monitor_stop.is_set():
                    break
        except Exception:  # noqa: BLE001 - the monitor is best-effort
            log.exception("microphone monitor failed")

    # -- calibration -------------------------------------------------------- #
    def calibrate_noise_floor(self, seconds: float = 1.0, *, margin: float = 2.0) -> float:
        """Sample the room and suggest a ``silence_threshold``.

        Returns the median block RMS times ``margin``, floored at
        :data:`MIN_NOISE_FLOOR`.  The median rather than the mean so a cough
        or a door during calibration does not deafen the assistant.  Falls back
        to the configured threshold when no audio can be captured.
        """
        fallback = float(self.cfg.silence_threshold)
        chunk_size = self.chunk_size()
        chunk_seconds = chunk_size / float(max(1, int(self.cfg.sample_rate)))
        wanted = max(1, int(math.ceil(max(0.0, seconds) / chunk_seconds)))
        levels: List[float] = []
        try:
            for chunk in self._iter_chunks(chunk_size):
                levels.append(rms(chunk))
                if len(levels) >= wanted:
                    break
        except Exception:  # noqa: BLE001
            log.exception("noise calibration failed")
            return fallback
        if not levels:
            return fallback
        levels.sort()
        median = levels[len(levels) // 2]
        return float(max(median * float(margin), MIN_NOISE_FLOOR))


# --------------------------------------------------------------------------- #
#  Speaker
# --------------------------------------------------------------------------- #
class AudioPlayer:
    """Play WAV bytes via the first available backend.

    Order: ``sounddevice`` → ``simpleaudio`` → OS default (``winsound`` /
    ``aplay`` / ``paplay``).
    """

    _POLL_INTERVAL = 0.05

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._current_process: Optional[Any] = None
        self._playing = threading.Event()
        self._backend: Optional[str] = None
        self._play_obj: Optional[Any] = None

    def is_available(self) -> bool:
        """True when at least one playback backend is importable/discoverable.

        Never raises — a missing or throwing dependency reports False so the
        caller can decide whether to speak or stay silent.
        """
        try:
            import sounddevice  # type: ignore  # noqa: F401
            return True
        except Exception:
            pass
        try:
            import simpleaudio  # type: ignore  # noqa: F401
            return True
        except Exception:
            pass
        if IS_WINDOWS:
            try:
                import winsound  # type: ignore  # noqa: F401
                return True
            except Exception:
                pass
        else:
            if which("aplay") or which("paplay"):
                return True
        return False

    @property
    def is_playing(self) -> bool:
        """True while :meth:`play_wav` is holding the speaker."""
        return self._playing.is_set()

    def stop(self) -> None:
        """Cut playback now.

        Barge-in depends on this returning the speaker within a few tens of
        milliseconds, so as well as signalling the poll loop it reaches into
        whichever backend is live and stops it directly.
        """
        self._stop_event.set()
        backend = self._backend

        proc = self._current_process
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass

        play_obj = self._play_obj
        if play_obj is not None:
            try:
                play_obj.stop()
            except Exception:
                log.debug("simpleaudio stop() failed", exc_info=True)

        if backend == "sounddevice":
            try:
                import sounddevice as sd  # type: ignore

                sd.stop()
            except Exception:
                log.debug("sounddevice stop() failed", exc_info=True)
        elif backend == "winsound" and IS_WINDOWS:
            try:
                import winsound  # type: ignore

                # SND_PURGE is the only way to silence an SND_ASYNC clip.
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                log.debug("winsound purge failed", exc_info=True)

    def play_wav(self, data: bytes) -> bool:
        """Play the given WAV bytes.  Returns True when playback succeeded."""
        if not data:
            return False
        self._stop_event.clear()
        self._playing.set()
        try:
            if self._play_sounddevice(data):
                return True
            if self._play_simpleaudio(data):
                return True
            return self._play_platform(data)
        finally:
            self._playing.clear()
            self._play_obj = None
            # ``_backend`` deliberately survives: the winsound path returns
            # while the clip is still sounding (SND_ASYNC), and stop() needs to
            # know which backend to silence.

    # -- backend implementations ------------------------------------------- #
    def _play_sounddevice(self, data: bytes) -> bool:
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return False
        try:
            import numpy as np  # type: ignore
        except ImportError:
            return False
        try:
            samples, sr = read_wav(data)
        except ValueError:
            return False
        self._backend = "sounddevice"
        try:
            arr = np.asarray(samples, dtype=np.float32)
            sd.play(arr, samplerate=sr)
            while True:
                try:
                    stream = sd.get_stream()
                except Exception:
                    break
                if stream is None or not getattr(stream, "active", False):
                    break
                if self._stop_event.wait(self._POLL_INTERVAL):
                    sd.stop()
                    break
        except Exception as exc:
            log.warning("sounddevice playback failed: %s", exc)
            return False
        return True

    def _play_simpleaudio(self, data: bytes) -> bool:
        try:
            import simpleaudio  # type: ignore
        except ImportError:
            return False
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                audio = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            log.warning("simpleaudio: bad WAV: %s", exc)
            return False
        try:
            play_obj = simpleaudio.play_buffer(audio, nchannels, sampwidth, framerate)
        except Exception as exc:
            log.warning("simpleaudio play_buffer failed: %s", exc)
            return False
        self._backend = "simpleaudio"
        self._play_obj = play_obj
        # Poll with a timed event wait so a caller stopping playback wakes
        # us immediately, and an idle wait never pins a CPU core.
        while True:
            try:
                still = play_obj.is_playing()
            except Exception:
                break
            if not still:
                break
            if self._stop_event.wait(self._POLL_INTERVAL):
                try:
                    play_obj.stop()
                except Exception:
                    pass
                break
        return True

    def _play_platform(self, data: bytes) -> bool:
        if IS_WINDOWS:
            return self._play_winsound(data)
        if IS_LINUX:
            return self._play_linux_cli(data)
        return False

    def _play_winsound(self, data: bytes) -> bool:
        if not IS_WINDOWS:
            return False
        try:
            import winsound  # type: ignore
        except ImportError:
            return False
        self._backend = "winsound"
        try:
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_ASYNC)
        except Exception as exc:
            log.warning("winsound playback failed: %s", exc)
            return False
        return True

    def _play_linux_cli(self, data: bytes) -> bool:
        import subprocess
        import tempfile
        player = which("aplay") or which("paplay")
        if not player:
            return False
        with tempfile.NamedTemporaryFile(prefix="jarvis-", suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        self._backend = "cli"
        try:
            proc = subprocess.Popen(
                [player, tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_process = proc
            while proc.poll() is None:
                if self._stop_event.wait(self._POLL_INTERVAL):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                    break
            self._current_process = None
        except Exception as exc:
            log.warning("linux CLI playback failed: %s", exc)
            return False
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        return True


__all__ = [
    "CHUNK_MS",
    "MIN_NOISE_FLOOR",
    "wav_bytes",
    "read_wav",
    "resample",
    "rms",
    "is_silent",
    "UtteranceAssembler",
    "AudioRecorder",
    "AudioPlayer",
]
