"""Speech-to-text engines with graceful degradation.

Concrete backends (``faster_whisper``, ``openai-whisper``, ``vosk``) are all
imported lazily inside ``is_available`` and ``transcribe``.  When nothing is
installed :func:`create_stt` returns :class:`NullSTT`, which is always
available and returns an empty transcript.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from ..core.config import STTConfig
from ..core.contracts import STTEngine, Transcript
from ..core.platform_utils import data_dir
from .audio_io import read_wav, resample
from .windows_speech import WindowsSTT

log = logging.getLogger(__name__)


_TARGET_SR = 16000

# Whisper emits these over silence, breathing and background noise with high
# confidence -- they are memorised from the subtitle corpora it was trained on,
# not heard. A voice assistant that acts on "Thanks for watching!" every time
# the fridge compressor starts is worse than one that mishears occasionally, so
# a *short* transcript consisting only of one of these is discarded.
#
# Matched after lowercasing and stripping punctuation. Deliberately conservative:
# every entry is a phrase no one plausibly says to an assistant on its own.
HALLUCINATIONS = frozenset({
    "thank you",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you very much",
    "you",
    "bye",
    "bye bye",
    "okay",
    "ok",
    "oh",
    "mm",
    "mmm",
    "hmm",
    "uh",
    "um",
    "ah",
    "so",
    "the",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "i'm going to go ahead and put that in the oven",
    "transcription by castingwords",
    "subtitles by the amara.org community",
    "www.mooji.org",
})

# Above this many characters a transcript is assumed to be genuine speech even
# if it happens to open with one of the phrases above. Sized to fit the longest
# entry in HALLUCINATIONS -- a listed phrase that can never match would be a
# quiet lie about what is actually filtered.
_HALLUCINATION_MAX_CHARS = 48


def _looks_hallucinated(text: str) -> bool:
    """True when ``text`` is one of Whisper's stock phrases and nothing more.

    Only ever applied to short transcripts: "thank you" as a complete utterance
    is almost always noise, while "thank you, that worked" is a real thing to
    say and must survive.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) > _HALLUCINATION_MAX_CHARS:
        return False
    normalised = re.sub(r"[^\w\s']", "", stripped).strip().lower()
    normalised = re.sub(r"\s+", " ", normalised)
    if not normalised:
        return True
    return normalised in HALLUCINATIONS


# --------------------------------------------------------------------------- #
#  Input preparation
# --------------------------------------------------------------------------- #
def _prepare_audio(audio: Any, sample_rate: int) -> Tuple[List[float], int]:
    """Coerce accepted transcribe() inputs into (mono float samples, sr).

    Accepted: :class:`str` or :class:`pathlib.Path` (WAV file), :class:`bytes`
    (WAV data), :class:`list` of floats, or a numpy array.  Everything is
    resampled to 16 kHz — the rate every supported STT expects.
    """
    if isinstance(audio, (str, Path)):
        samples, sr = read_wav(audio)
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        samples, sr = read_wav(audio)
    else:
        try:
            import numpy as np  # type: ignore
            if isinstance(audio, np.ndarray):
                samples = np.asarray(audio, dtype=np.float32).flatten().tolist()
                sr = int(sample_rate)
            else:
                samples = [float(x) for x in audio]
                sr = int(sample_rate)
        except ImportError:
            samples = [float(x) for x in audio]
            sr = int(sample_rate)

    if sr != _TARGET_SR:
        samples = resample(samples, sr, _TARGET_SR)
        sr = _TARGET_SR
    return samples, sr


def _empty_transcript(language: Optional[str] = None) -> Transcript:
    return Transcript(text="", language=language, confidence=0.0, segments=())


# --------------------------------------------------------------------------- #
#  faster-whisper
# --------------------------------------------------------------------------- #
class FasterWhisperSTT(STTEngine):
    """CTranslate2-backed Whisper — the fastest CPU option and the default."""

    name = "faster-whisper"

    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model: Any = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    def _resolve_device(self) -> str:
        device = (self.cfg.device or "auto").lower()
        if device != "auto":
            return device
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            return
        device = self._resolve_device()
        compute_type = self.cfg.compute_type
        # float16 is a GPU format; CTranslate2 refuses it on CPU. Silently
        # correcting this is better than a load failure that reads as "STT is
        # broken" when the only problem is one config line.
        if device == "cpu" and str(compute_type).lower() in ("float16", "fp16", "half"):
            log.info("compute_type %r is GPU-only; using int8 on CPU.", compute_type)
            compute_type = "int8"

        kwargs: dict = {"device": device, "compute_type": compute_type}
        threads = int(getattr(self.cfg, "cpu_threads", 0) or 0)
        if threads <= 0:
            # CTranslate2 defaults to a single thread in several builds, which
            # makes transcription several times slower than the hardware allows.
            import os

            threads = max(1, (os.cpu_count() or 4))
        if device == "cpu":
            kwargs["cpu_threads"] = threads

        try:
            self._model = WhisperModel(self.cfg.model, **kwargs)
        except Exception as exc:
            log.warning("faster-whisper load failed: %s", exc)
            # An unknown or undownloadable model name is the usual cause. Fall
            # back to one that is certain to exist rather than going mute.
            fallback = "base.en"
            if str(self.cfg.model) != fallback:
                log.info("retrying faster-whisper with %r", fallback)
                try:
                    self._model = WhisperModel(fallback, **kwargs)
                    return
                except Exception as exc2:
                    log.warning("faster-whisper fallback load failed: %s", exc2)
            self._model = None

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        try:
            samples, _sr = _prepare_audio(audio, sample_rate)
        except Exception as exc:
            log.warning("audio preparation failed: %s", exc)
            return _empty_transcript()
        if not samples:
            return _empty_transcript()
        self._load()
        if self._model is None:
            return _empty_transcript()

        try:
            import numpy as np  # type: ignore
            arr = np.asarray(samples, dtype=np.float32)
        except ImportError:
            arr = samples  # type: ignore

        # Quality knobs, all defaulted in STTConfig. Passed through **kwargs so
        # an older faster-whisper that lacks one of them still runs (see the
        # TypeError retry below) rather than refusing to transcribe at all.
        options: dict = {
            "vad_filter": self.cfg.vad_filter,
            "language": self.cfg.language or None,
            "beam_size": int(getattr(self.cfg, "beam_size", 5) or 5),
            "condition_on_previous_text": bool(
                getattr(self.cfg, "condition_on_previous_text", False)
            ),
            "no_speech_threshold": float(getattr(self.cfg, "no_speech_threshold", 0.6)),
            "log_prob_threshold": float(getattr(self.cfg, "log_prob_threshold", -1.0)),
        }
        prompt = str(getattr(self.cfg, "initial_prompt", "") or "").strip()
        if prompt:
            options["initial_prompt"] = prompt

        try:
            seg_iter, info = self._model.transcribe(arr, **options)
        except TypeError as exc:
            # An older build that does not know one of the newer keywords.
            log.debug("faster-whisper rejected an option (%s); retrying plainly", exc)
            try:
                seg_iter, info = self._model.transcribe(
                    arr,
                    vad_filter=self.cfg.vad_filter,
                    language=self.cfg.language or None,
                )
            except Exception as exc2:
                log.warning("faster-whisper transcribe failed: %s", exc2)
                return _empty_transcript()
        except Exception as exc:
            log.warning("faster-whisper transcribe failed: %s", exc)
            return _empty_transcript()

        text_parts: List[str] = []
        structured: List[dict] = []
        confidences: List[float] = []
        try:
            for seg in seg_iter:
                seg_text = getattr(seg, "text", "") or ""
                text_parts.append(seg_text)
                structured.append(
                    {
                        "start": float(getattr(seg, "start", 0.0) or 0.0),
                        "end": float(getattr(seg, "end", 0.0) or 0.0),
                        "text": seg_text,
                    }
                )
                avg = getattr(seg, "avg_logprob", None)
                if avg is not None:
                    try:
                        confidences.append(max(0.0, min(1.0, math.exp(float(avg)))))
                    except (OverflowError, ValueError):
                        pass
        except Exception as exc:
            log.warning("faster-whisper segment iteration failed: %s", exc)
            return _empty_transcript()

        conf = sum(confidences) / len(confidences) if confidences else 1.0
        language = getattr(info, "language", None) or self.cfg.language
        text = " ".join(t.strip() for t in text_parts).strip()

        if getattr(self.cfg, "filter_hallucinations", True) and _looks_hallucinated(text):
            log.debug("discarding a likely Whisper hallucination: %r", text)
            return _empty_transcript(language)

        return Transcript(
            text=text,
            language=language,
            confidence=conf,
            segments=tuple(structured),
        )


# --------------------------------------------------------------------------- #
#  openai-whisper
# --------------------------------------------------------------------------- #
class WhisperSTT(STTEngine):
    """Reference OpenAI Whisper (Python package ``whisper``)."""

    name = "whisper"

    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model: Any = None

    def is_available(self) -> bool:
        try:
            import whisper  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    def _resolve_device(self) -> str:
        device = (self.cfg.device or "auto").lower()
        if device != "auto":
            return device
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore
        except ImportError:
            return
        try:
            self._model = whisper.load_model(self.cfg.model, device=self._resolve_device())
        except Exception as exc:
            log.warning("whisper load failed: %s", exc)
            self._model = None

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        try:
            samples, _sr = _prepare_audio(audio, sample_rate)
        except Exception as exc:
            log.warning("audio preparation failed: %s", exc)
            return _empty_transcript()
        if not samples:
            return _empty_transcript()
        self._load()
        if self._model is None:
            return _empty_transcript()

        try:
            import numpy as np  # type: ignore
            arr = np.asarray(samples, dtype=np.float32)
        except ImportError:
            return _empty_transcript()

        try:
            result = self._model.transcribe(
                arr,
                language=self.cfg.language or None,
                fp16=False,
                beam_size=int(getattr(self.cfg, "beam_size", 5) or 5),
                condition_on_previous_text=bool(
                    getattr(self.cfg, "condition_on_previous_text", False)
                ),
                initial_prompt=str(getattr(self.cfg, "initial_prompt", "") or "") or None,
            )
        except Exception as exc:
            log.warning("whisper transcribe failed: %s", exc)
            return _empty_transcript()

        raw_text = str(result.get("text", "")).strip()
        language = result.get("language") or self.cfg.language
        structured: List[dict] = []
        confidences: List[float] = []
        for seg in result.get("segments", []) or []:
            structured.append(
                {
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                    "text": str(seg.get("text", "")),
                }
            )
            avg = seg.get("avg_logprob")
            if avg is not None:
                try:
                    confidences.append(max(0.0, min(1.0, math.exp(float(avg)))))
                except (OverflowError, ValueError):
                    pass

        conf = sum(confidences) / len(confidences) if confidences else 1.0
        if getattr(self.cfg, "filter_hallucinations", True) and _looks_hallucinated(raw_text):
            log.debug("discarding a likely Whisper hallucination: %r", raw_text)
            return _empty_transcript(language)
        return Transcript(
            text=raw_text,
            language=language,
            confidence=conf,
            segments=tuple(structured),
        )


# --------------------------------------------------------------------------- #
#  Vosk
# --------------------------------------------------------------------------- #
class VoskSTT(STTEngine):
    """Kaldi-based offline recognizer (no GPU required)."""

    name = "vosk"

    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._recognizer: Any = None
        self._model: Any = None

    def _model_path(self) -> Path:
        raw = (self.cfg.model or "").strip()
        if raw and raw not in ("base.en", "tiny.en", "small.en"):
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate
        return data_dir() / "models" / "vosk"

    def is_available(self) -> bool:
        try:
            import vosk  # type: ignore  # noqa: F401
        except Exception:
            return False
        return self._model_path().exists()

    def _load(self) -> None:
        if self._recognizer is not None:
            return
        try:
            import vosk  # type: ignore
        except ImportError:
            return
        path = self._model_path()
        if not path.exists():
            return
        try:
            self._model = vosk.Model(str(path))
            self._recognizer = vosk.KaldiRecognizer(self._model, _TARGET_SR)
        except Exception as exc:
            log.warning("vosk load failed: %s", exc)
            self._recognizer = None

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        try:
            samples, _sr = _prepare_audio(audio, sample_rate)
        except Exception as exc:
            log.warning("audio preparation failed: %s", exc)
            return _empty_transcript()
        if not samples:
            return _empty_transcript()
        self._load()
        if self._recognizer is None:
            return _empty_transcript()

        # Vosk needs 16-bit little-endian PCM bytes.
        pcm = _floats_to_int16_bytes(samples)
        try:
            self._recognizer.AcceptWaveform(pcm)
            raw = self._recognizer.FinalResult()
            payload = json.loads(raw) if isinstance(raw, str) else {}
        except Exception as exc:
            log.warning("vosk transcribe failed: %s", exc)
            return _empty_transcript()

        text = str(payload.get("text", "")).strip()
        return Transcript(
            text=text,
            language=self.cfg.language,
            confidence=1.0 if text else 0.0,
            segments=(),
        )


def _floats_to_int16_bytes(samples: Sequence[float]) -> bytes:
    import struct
    ints = []
    for s in samples:
        f = float(s)
        if f > 1.0:
            f = 1.0
        elif f < -1.0:
            f = -1.0
        ints.append(int(round(f * 32767.0)))
    return struct.pack(f"<{len(ints)}h", *ints)


# --------------------------------------------------------------------------- #
#  Null fallback
# --------------------------------------------------------------------------- #
class NullSTT(STTEngine):
    """No-op engine used when nothing else works.  Always available."""

    name = "null"

    def __init__(self, cfg: Optional[STTConfig] = None) -> None:
        self.cfg = cfg

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        return _empty_transcript()


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
_ENGINE_CLASSES = {
    "faster-whisper": FasterWhisperSTT,
    "whisper": WhisperSTT,
    "vosk": VoskSTT,
    "windows": WindowsSTT,
    "null": NullSTT,
    "stub": NullSTT,
    "none": NullSTT,
}

# WindowsSTT sits below the real engines because it is markedly less accurate,
# and above null because — unlike null — it actually transcribes.
_AUTO_ORDER = ("faster-whisper", "whisper", "vosk", "windows")


def available_stt_engines(cfg: STTConfig) -> List[STTEngine]:
    """Every engine that is currently available for ``cfg``."""
    out: List[STTEngine] = []
    for key in _AUTO_ORDER:
        cls = _ENGINE_CLASSES[key]
        engine = cls(cfg)
        if engine.is_available():
            out.append(engine)
    out.append(NullSTT(cfg))
    return out


def create_stt(cfg: STTConfig) -> STTEngine:
    """Pick an :class:`STTEngine` based on ``cfg.engine`` (or auto-probe)."""
    requested = (cfg.engine or "auto").strip().lower()
    if requested == "auto":
        for key in _AUTO_ORDER:
            cls = _ENGINE_CLASSES[key]
            engine = cls(cfg)
            if engine.is_available():
                return engine
        return NullSTT(cfg)

    cls = _ENGINE_CLASSES.get(requested)
    if cls is None:
        log.warning("Unknown STT engine %r; falling back to null.", requested)
        return NullSTT(cfg)
    engine = cls(cfg)
    if engine.is_available():
        return engine
    log.info("STT engine %r not available; using null.", requested)
    return NullSTT(cfg)


__all__ = [
    "FasterWhisperSTT",
    "WhisperSTT",
    "VoskSTT",
    "WindowsSTT",
    "NullSTT",
    "create_stt",
    "available_stt_engines",
]
