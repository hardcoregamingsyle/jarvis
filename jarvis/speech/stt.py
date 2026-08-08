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
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from ..core.config import STTConfig
from ..core.contracts import STTEngine, Transcript
from ..core.platform_utils import data_dir
from .audio_io import read_wav, resample
from .windows_speech import WindowsSTT

log = logging.getLogger(__name__)


_TARGET_SR = 16000


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
        try:
            self._model = WhisperModel(
                self.cfg.model,
                device=self._resolve_device(),
                compute_type=self.cfg.compute_type,
            )
        except Exception as exc:
            log.warning("faster-whisper load failed: %s", exc)
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

        try:
            seg_iter, info = self._model.transcribe(
                arr,
                vad_filter=self.cfg.vad_filter,
                language=self.cfg.language or None,
            )
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
        return Transcript(
            text=" ".join(t.strip() for t in text_parts).strip(),
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
