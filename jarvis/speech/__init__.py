"""Speech I/O: audio primitives, STT engines, and a TTS entry point.

The concrete backends behind :func:`create_stt` and :func:`create_tts` are
imported lazily so this package always imports cleanly, even on a machine
where none of the audio libraries are installed and even if the sibling
``tts`` module (authored elsewhere) is missing or malformed.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any, Optional

from ..core.contracts import STTEngine, TTSEngine
from .stt import (
    FasterWhisperSTT,
    NullSTT,
    VoskSTT,
    WhisperSTT,
    available_stt_engines,
    create_stt,
)

log = logging.getLogger(__name__)


class _NullTTS(TTSEngine):
    """Silent TTS used only when :mod:`jarvis.speech.tts` cannot be loaded.

    Subclasses :class:`TTSEngine` so isinstance checks in the orchestrator
    continue to work — a fallback that fails ``isinstance(tts, TTSEngine)``
    would be worse than useless.
    """

    name = "null"

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16)
        return buf.getvalue()

    def speak(self, text: str) -> None:
        return None


def create_tts(cfg: Any, *, voices_dir: Optional[Path] = None) -> TTSEngine:
    """Build a TTS engine, delegating to :mod:`jarvis.speech.tts` when possible.

    The sibling module is imported lazily so that a missing or broken
    ``tts.py`` (written by a concurrent author) can never break the import
    of :mod:`jarvis.speech`.  Any failure yields a :class:`_NullTTS` fallback.
    """
    # A config missing the TTS fields would yield an engine that returns empty
    # audio on every call — silence, forever, with only a log line to show for
    # it. An honest null engine is strictly better than that.
    required = ("engine", "piper_voice", "edge_voice", "enabled")
    missing = [field for field in required if not hasattr(cfg, field)]
    if missing:
        log.warning(
            "create_tts got %s which lacks %s; using null TTS",
            type(cfg).__name__, ", ".join(missing),
        )
        return _NullTTS()

    try:
        from . import tts as _tts_mod
    except Exception as exc:
        log.warning("jarvis.speech.tts unavailable, using null TTS: %s", exc)
        return _NullTTS()
    factory = getattr(_tts_mod, "create_tts", None)
    if factory is None:
        log.warning("jarvis.speech.tts has no create_tts(); using null TTS.")
        return _NullTTS()
    try:
        engine = factory(cfg, voices_dir=voices_dir)
    except Exception as exc:
        log.warning("create_tts() failed, using null TTS: %s", exc)
        return _NullTTS()
    if not isinstance(engine, TTSEngine):
        log.warning("create_tts() returned a non-TTSEngine; using null TTS.")
        return _NullTTS()
    return engine


__all__ = [
    "create_stt",
    "create_tts",
    "available_stt_engines",
    "FasterWhisperSTT",
    "WhisperSTT",
    "VoskSTT",
    "NullSTT",
]
