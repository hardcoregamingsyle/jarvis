"""Text-to-speech: a ladder of engines from best-sounding to always-works.

The public voice of JARVIS is a polished British (RP) baritone.  The ladder,
in order of preference, is:

    piper  ->  edge  ->  sapi  ->  pyttsx3  ->  espeak  ->  null

``piper`` is the default because it is fully offline and the closest to the
film Jarvis's calm British delivery.  ``edge`` sounds even better on some
Neural voices (Ryan, Thomas, Sonia) but needs internet.  ``pyttsx3`` bridges
to whatever the OS has (SAPI5 on Windows, espeak on Linux) and always ships.
``espeak`` is the last resort that still speaks; ``NullTTS`` is a silent
placeholder that never fails.

Every engine exposes an ``output_format`` attribute (``"wav"`` or ``"mp3"``)
so callers know what extension to write and which player to hand it to.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from ..core import platform_utils
from ..core.config import TTSConfig
from ..core.contracts import TTSEngine

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  MP3 -> WAV helper
# --------------------------------------------------------------------------- #
def mp3_to_wav(data: bytes, *, sample_rate: int = 16000) -> Optional[bytes]:
    """Decode MP3 bytes to 16-bit mono PCM WAV.

    Returns ``None`` when neither ``pydub`` nor an ``ffmpeg`` binary is
    available so the caller can fall back to keeping the raw MP3.  Never
    raises.  ffmpeg's stdout is accepted ONLY when it truly begins with
    ``b"RIFF"`` — a mis-invocation that produces junk must not masquerade
    as a wav file.
    """
    if not data:
        return None

    # 1) pydub first: cleaner, gives us control over channel/rate.
    try:
        from pydub import AudioSegment  # type: ignore

        seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        seg = seg.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        out = buf.getvalue()
        if out.startswith(b"RIFF"):
            return out
    except ImportError:
        pass
    except Exception as exc:  # pydub throws CouldntDecodeError, ffmpeg errors, ...
        log.debug("mp3_to_wav: pydub path failed: %s", exc)

    # 2) ffmpeg binary via stdin/stdout so we do not touch the filesystem.
    ffmpeg = platform_utils.which("ffmpeg")
    if not ffmpeg:
        return None
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "mp3", "-i", "pipe:0",
                "-ac", "1", "-ar", str(sample_rate),
                "-f", "wav", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        log.debug("mp3_to_wav: could not launch ffmpeg: %s", exc)
        return None

    try:
        out, _ = proc.communicate(input=data, timeout=30.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        log.warning("mp3_to_wav: ffmpeg timed out")
        return None
    except Exception as exc:
        log.debug("mp3_to_wav: ffmpeg communication failed: %s", exc)
        return None

    if proc.returncode == 0 and out and out.startswith(b"RIFF"):
        return out
    return None


# --------------------------------------------------------------------------- #
#  OS-level playback fallback
# --------------------------------------------------------------------------- #
def _play_via_os(data: bytes, *, suffix: str = ".wav") -> None:
    """Last-resort playback: dump to a temp file and hand it to the OS.

    Never raises.  A silent voice assistant is worse than one that opens the
    system player, so we swallow every error rather than let the utterance
    disappear.
    """
    if not data:
        return
    if not suffix.startswith("."):
        suffix = "." + suffix
    try:
        fd, path = tempfile.mkstemp(prefix="jarvis-tts-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        platform_utils.open_path(path)
    except Exception as exc:
        log.warning("_play_via_os failed: %s", exc)


# --------------------------------------------------------------------------- #
#  British polish (idempotent, unicode-safe)
# --------------------------------------------------------------------------- #
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty")


def _num_to_words(n: int) -> str:
    if n < 0 or n >= 60:
        return str(n)
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_ONES[ones]}"


_LETTER_ACRONYMS = ("CPU", "GPU", "RAM", "SSD", "USB", "API", "URL")

_ABBREVIATIONS = (
    (re.compile(r"\bDr\.(?=\s+[A-Z])"), "Doctor"),
    (re.compile(r"\bDr\.\B"), "Doctor"),
    (re.compile(r"\bMr\.(?=\s)"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s)"), "Missus"),
    (re.compile(r"\bMs\.(?=\s)"), "Miss"),
    (re.compile(r"\bSt\.(?=\s+[A-Z])"), "Saint"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
)

# American -> British respellings.  Ordering matters: the "-ize"/"-yze" family
# is handled first so we don't double-transform.  Every rule leaves output that
# will not re-match on a second pass, so british_polish is idempotent.
_SPELLINGS = (
    (re.compile(r"\bcolor(s?)\b"), r"colour\1"),
    (re.compile(r"\bColor(s?)\b"), r"Colour\1"),
    (re.compile(r"\bflavor(s?)\b"), r"flavour\1"),
    (re.compile(r"\bfavor(s?)\b"), r"favour\1"),
    (re.compile(r"\bhonor(s?)\b"), r"honour\1"),
    (re.compile(r"\bneighbor(s?)\b"), r"neighbour\1"),
    (re.compile(r"\bbehavior(s?)\b"), r"behaviour\1"),
    (re.compile(r"\bcenter(s?)\b"), r"centre\1"),
    (re.compile(r"\bmeter(s?)\b"), r"metre\1"),
    (re.compile(r"\bliter(s?)\b"), r"litre\1"),
    (re.compile(r"\btheater(s?)\b"), r"theatre\1"),
    (re.compile(r"\banaly(z)(e|ed|es|ing)\b"), r"analys\2"),
    (re.compile(r"\borgani(z)(e|ed|es|ing|ation|ations)\b"), r"organis\2"),
    (re.compile(r"\brecogni(z)(e|ed|es|ing)\b"), r"recognis\2"),
    (re.compile(r"\brealize(d|s)?\b"), r"realise\1"),
    (re.compile(r"\bdefense\b"), "defence"),
    (re.compile(r"\boffense\b"), "offence"),
    (re.compile(r"\blicense\b"), "licence"),
    (re.compile(r"\btraveled\b"), "travelled"),
    (re.compile(r"\btraveling\b"), "travelling"),
    (re.compile(r"\bcancelled\b"), "cancelled"),
    (re.compile(r"\bcanceled\b"), "cancelled"),
    (re.compile(r"\bAluminum\b"), "Aluminium"),
    (re.compile(r"\baluminum\b"), "aluminium"),
)

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_ACRONYM_RE = re.compile(r"\b(" + "|".join(_LETTER_ACRONYMS) + r")\b")


def _spell_out_time(match: "re.Match[str]") -> str:
    hours = int(match.group(1))
    minutes = int(match.group(2))
    h_words = _num_to_words(hours)
    if minutes == 0:
        return f"{h_words} hundred hours"
    return f"{h_words} {_num_to_words(minutes)}"


def british_polish(text: Any) -> str:
    """Light normalisation that sells the accent.

    Idempotent: ``british_polish(british_polish(x)) == british_polish(x)`` for
    any input.  Unicode-safe: non-ASCII characters are preserved.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""

    # Symbols first.  Guard against re-substitution by requiring the raw glyphs.
    text = text.replace("&", " and ")
    text = re.sub(r"\s*%\s*", " percent ", text)

    # 24-hour times before acronyms so we don't turn "14:30" into odd letters.
    text = _TIME_RE.sub(_spell_out_time, text)

    # Letter-wise acronyms — CPU -> "C P U".  Word boundaries plus the fact
    # that spaced-out letters no longer match \bCPU\b keeps this idempotent.
    def _spell_acr(m: "re.Match[str]") -> str:
        return " ".join(m.group(1))

    text = _ACRONYM_RE.sub(_spell_acr, text)

    # Common abbreviations.
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # American -> British respellings.
    for pattern, replacement in _SPELLINGS:
        text = pattern.sub(replacement, text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


# --------------------------------------------------------------------------- #
#  Silent WAV builder (used by NullTTS and as a safe stand-in)
# --------------------------------------------------------------------------- #
def _silent_wav(*, seconds: float = 0.15, sample_rate: int = 16000) -> bytes:
    """A short valid 16-bit mono WAV of silence."""
    frames = max(1, int(seconds * sample_rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _looks_like_wav(data: bytes) -> bool:
    return bool(data) and data[:4] == b"RIFF" and b"WAVE" in data[:16]


# --------------------------------------------------------------------------- #
#  Null engine — always available
# --------------------------------------------------------------------------- #
class NullTTS(TTSEngine):
    """Silent placeholder.  Always available; used as the ultimate fallback."""

    name = "null"

    def __init__(self, cfg: Optional[Any] = None, *, sample_rate: int = 16000) -> None:
        # Accept either a TTSConfig or a bare int sample rate so it slots into
        # any factory position — the sibling engines all take a config first.
        if isinstance(cfg, int):
            sample_rate = cfg
            cfg = None
        self.cfg = cfg if isinstance(cfg, TTSConfig) else TTSConfig()
        self.sample_rate = int(sample_rate)
        self.output_format = "wav"

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str) -> bytes:
        return _silent_wav(sample_rate=self.sample_rate)

    def speak(self, text: str) -> None:
        log.info("NullTTS.speak (silent): %s", text[:120].replace("\n", " "))


# --------------------------------------------------------------------------- #
#  Piper — primary offline voice
# --------------------------------------------------------------------------- #
class PiperTTS(TTSEngine):
    """Piper: crisp offline RP voice.  Prefers the Python API, falls back to CLI."""

    name = "piper"
    #: rhasspy/piper-voices Hugging Face layout.  Format: {voice}/{lang}/{voice}/{voice}.onnx
    _HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

    def __init__(self, cfg: TTSConfig, *, voices_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.voices_dir = Path(voices_dir) if voices_dir else (platform_utils.data_dir() / "voices")
        self.output_format = "wav"
        self._voice = None  # cached PiperVoice
        self._model_path: Optional[Path] = None

    # ---- discovery ---------------------------------------------------------
    def _resolve_model_path(self) -> Path:
        if self.cfg.piper_model_path:
            return Path(self.cfg.piper_model_path).expanduser()
        return self.voices_dir / f"{self.cfg.piper_voice}.onnx"

    def _has_python_package(self) -> bool:
        try:
            import importlib
            for name in ("piper", "piper_tts"):
                try:
                    importlib.import_module(name)
                    return True
                except ImportError:
                    continue
        except Exception:
            pass
        return False

    def _has_cli(self) -> bool:
        return platform_utils.which("piper") is not None

    def is_available(self) -> bool:
        """Pure import/filesystem check — no network."""
        model = self._resolve_model_path()
        if not model.exists():
            return False
        return self._has_python_package() or self._has_cli()

    def ensure_voice(self, download: bool = False) -> dict:
        """Report expected model paths, optionally downloading from HF.

        ``download=False`` (the default) NEVER touches the network.  When the
        caller opts in we fetch ``<voice>.onnx`` and ``<voice>.onnx.json``
        with stdlib urllib into ``voices_dir``.
        """
        model = self._resolve_model_path()
        cfg_path = model.with_suffix(".onnx.json") if model.suffix == ".onnx" else Path(str(model) + ".json")
        info = {
            "voice": self.cfg.piper_voice,
            "model_path": model,
            "config_path": cfg_path,
            "downloaded": False,
            "errors": [],
        }
        if model.exists() and cfg_path.exists():
            return info
        if not download:
            info["errors"].append("model missing; call ensure_voice(download=True)")
            return info

        # Piper voices live at rhasspy/piper-voices/<lang>/<region>/<name>/<quality>/<file>.
        # The convention is <name>_<lang>_<region>_<quality> in the flat name,
        # but the on-disk file we care about is <voice>.onnx.  We build the URL
        # from the voice slug: e.g. en_GB-alan-medium -> en/en_GB/alan/medium/en_GB-alan-medium.onnx
        try:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            from urllib.request import urlopen  # stdlib
            slug = self.cfg.piper_voice
            parts = slug.split("-")
            lang_region = parts[0] if parts else "en_GB"
            speaker = parts[1] if len(parts) > 1 else ""
            quality = parts[2] if len(parts) > 2 else "medium"
            lang_short = lang_region.split("_")[0]
            base = f"{self._HF_BASE}/{lang_short}/{lang_region}/{speaker}/{quality}/{slug}"
            for suffix, dest in ((".onnx", model), (".onnx.json", cfg_path)):
                url = base + suffix
                with urlopen(url, timeout=60.0) as resp:  # nosec - opt-in
                    dest.write_bytes(resp.read())
            info["downloaded"] = True
        except Exception as exc:
            info["errors"].append(f"download failed: {exc}")
        return info

    # ---- synthesis ---------------------------------------------------------
    def _load_python_voice(self):
        if self._voice is not None:
            return self._voice
        try:
            try:
                from piper.voice import PiperVoice  # type: ignore
            except ImportError:
                from piper import PiperVoice  # type: ignore
        except ImportError:
            return None
        model = self._resolve_model_path()
        cfg_path = model.with_suffix(".onnx.json")
        try:
            if cfg_path.exists():
                self._voice = PiperVoice.load(str(model), config_path=str(cfg_path))
            else:
                self._voice = PiperVoice.load(str(model))
        except Exception as exc:
            log.warning("Piper: failed to load voice %s: %s", model, exc)
            self._voice = None
        return self._voice

    @staticmethod
    def _piper_sample_rate(voice: Any) -> int:
        # Probe defensively — the attribute path has shifted across versions.
        for path in (
            ("config", "sample_rate"),
            ("config", "audio", "sample_rate"),
            ("sample_rate",),
        ):
            obj: Any = voice
            try:
                for a in path:
                    obj = getattr(obj, a)
                if isinstance(obj, int) and obj > 0:
                    return obj
            except AttributeError:
                continue
        return 22050

    def synthesize(self, text: str) -> bytes:
        text = british_polish(text)
        if not text:
            return b""
        # Python API preferred.
        voice = self._load_python_voice()
        if voice is not None:
            try:
                sr = self._piper_sample_rate(voice)
                buf = io.BytesIO()
                # NOTE: setnchannels/setsampwidth/setframerate MUST all be set
                # before the first writeframes(), or wave.Error("sampling rate
                # not set") fires and the whole synthesise returns b"" every
                # time — i.e. the default voice would be silently mute.
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    if hasattr(voice, "synthesize_wav"):
                        voice.synthesize_wav(text, wf)
                    else:
                        voice.synthesize(text, wf)
                data = buf.getvalue()
                if _looks_like_wav(data):
                    return data
                log.warning("Piper: synthesise produced non-WAV output; returning empty")
                return b""
            except Exception as exc:
                log.warning("Piper Python API failed: %s", exc)

        # CLI fallback.
        binary = platform_utils.which("piper")
        if not binary:
            return b""
        model = self._resolve_model_path()
        if not model.exists():
            return b""
        tmp_out = None
        try:
            fd, tmp_out = tempfile.mkstemp(prefix="jarvis-piper-", suffix=".wav")
            os.close(fd)
            result = platform_utils.run_command(
                [binary, "--model", str(model), "--output_file", tmp_out],
                timeout=60.0,
            )
            if not result.ok:
                # run_command doesn't send stdin — some piper builds want it.
                proc = subprocess.Popen(
                    [binary, "--model", str(model), "--output_file", tmp_out],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                try:
                    proc.communicate(input=text.encode("utf-8"), timeout=60.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
                    return b""
            data = Path(tmp_out).read_bytes() if Path(tmp_out).exists() else b""
            return data if _looks_like_wav(data) else b""
        except Exception as exc:
            log.warning("Piper CLI failed: %s", exc)
            return b""
        finally:
            if tmp_out and os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except OSError:
                    pass

    def speak(self, text: str) -> None:
        data = self.synthesize(text)
        if not data:
            return
        try:
            from .audio_io import AudioPlayer  # type: ignore
            player = AudioPlayer()
            player.play(data)
            if hasattr(player, "wait"):
                player.wait()
            return
        except Exception:
            _play_via_os(data, suffix=".wav")


# --------------------------------------------------------------------------- #
#  Edge TTS — highest fidelity, requires internet
# --------------------------------------------------------------------------- #
class EdgeTTS(TTSEngine):
    """Microsoft edge_tts.

    Suggested voices: en-GB-RyanNeural (default), en-GB-ThomasNeural,
    en-GB-SoniaNeural.
    """

    name = "edge"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        # We do not know until we see bytes; default to wav so a static reader
        # of the attribute never lies about the file extension of a *silent*
        # result.  synthesize() sets this to the actual format on every call.
        self.output_format = "wav"
        # Hard cap on the network round-trip.  Without it, a stalled TLS
        # handshake would wedge SpeechQueue forever and JARVIS would go silent.
        self.request_timeout: float = 60.0

    def is_available(self) -> bool:
        try:
            import importlib
            importlib.import_module("edge_tts")
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _fetch_bytes(self, text: str) -> bytes:
        """Perform the actual edge_tts network call and return raw audio bytes.

        Runs the async client on a dedicated thread with a fresh event loop,
        so we work correctly whether the caller has a running loop or not,
        and never asyncio.run() into a live loop.
        """
        try:
            import asyncio
            import edge_tts  # type: ignore
        except ImportError:
            return b""

        result: dict = {}

        def worker() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)

                async def run() -> bytes:
                    kwargs: dict = {}
                    if getattr(self.cfg, "edge_rate", ""):
                        kwargs["rate"] = self.cfg.edge_rate
                    if getattr(self.cfg, "edge_pitch", ""):
                        kwargs["pitch"] = self.cfg.edge_pitch
                    comm = edge_tts.Communicate(text, voice=self.cfg.edge_voice, **kwargs)
                    audio = bytearray()
                    async for chunk in comm.stream():
                        if isinstance(chunk, dict) and chunk.get("type") == "audio":
                            audio.extend(chunk.get("data", b""))
                    return bytes(audio)

                result["value"] = loop.run_until_complete(run())
            except Exception as exc:
                result["error"] = exc
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=worker, name="edge-tts", daemon=True)
        thread.start()
        thread.join(self.request_timeout)
        if thread.is_alive():
            log.warning("EdgeTTS: network call exceeded %.1fs; giving up", self.request_timeout)
            return b""
        if "error" in result:
            log.warning("EdgeTTS: synthesis error: %s", result["error"])
            return b""
        return result.get("value", b"")

    def synthesize(self, text: str) -> bytes:
        text = british_polish(text)
        if not text:
            return b""
        raw = self._fetch_bytes(text)
        if not raw:
            return b""
        if raw.startswith(b"RIFF"):
            self.output_format = "wav"
            return raw
        # Almost always MP3.  Try to convert; only if that actually works do
        # we advertise "wav".
        converted = mp3_to_wav(raw)
        if converted and converted.startswith(b"RIFF"):
            self.output_format = "wav"
            return converted
        self.output_format = "mp3"
        return raw

    def speak(self, text: str) -> None:
        data = self.synthesize(text)
        if not data:
            return
        if self.output_format == "wav":
            try:
                from .audio_io import AudioPlayer  # type: ignore
                player = AudioPlayer()
                player.play(data)
                if hasattr(player, "wait"):
                    player.wait()
                return
            except Exception:
                pass
        # Never leave the user in silence because the format is unexpected —
        # hand it to the OS default player rather than swallowing it.
        _play_via_os(data, suffix=("." + (self.output_format or "wav")))


# --------------------------------------------------------------------------- #
#  pyttsx3 — SAPI5 (Windows) / espeak (Linux) via a Python API
# --------------------------------------------------------------------------- #
class Pyttsx3TTS(TTSEngine):
    """pyttsx3 wrapper.  Picks a UK-English voice when available."""

    name = "pyttsx3"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.output_format = "wav"

    def is_available(self) -> bool:
        try:
            import importlib
            importlib.import_module("pyttsx3")
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _make_engine(self):
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        # Voice selection.
        try:
            voices = engine.getProperty("voices") or []
            hint = (self.cfg.sapi_voice_hint or "").lower()
            keywords = [hint] if hint else []
            keywords.extend(["en-gb", "british", "hazel", "george", "ryan", "sonia", "united kingdom"])
            chosen = None
            for voice in voices:
                fields = " ".join(
                    str(getattr(voice, attr, "") or "")
                    for attr in ("name", "id", "languages")
                ).lower()
                if any(kw and kw in fields for kw in keywords):
                    chosen = voice.id
                    break
            if chosen:
                engine.setProperty("voice", chosen)
        except Exception as exc:
            log.debug("Pyttsx3: voice pick failed: %s", exc)
        # Rate: pyttsx3's default is ~200 wpm.
        try:
            base_rate = int(engine.getProperty("rate") or 200)
            engine.setProperty("rate", int(base_rate * max(0.25, self.cfg.speed)))
        except Exception:
            pass
        try:
            engine.setProperty("volume", float(max(0.0, min(1.0, self.cfg.volume))))
        except Exception:
            pass
        return engine

    def _synth_once(self, engine, text: str, out_path: str) -> bytes:
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        if os.path.exists(out_path):
            return Path(out_path).read_bytes()
        return b""

    def synthesize(self, text: str) -> bytes:
        text = british_polish(text)
        if not text:
            return b""
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="jarvis-pyttsx3-", suffix=".wav")
            os.close(fd)
            try:
                engine = self._make_engine()
            except Exception as exc:
                log.warning("Pyttsx3: engine init failed: %s", exc)
                return b""
            try:
                data = self._synth_once(engine, text, tmp_path)
                return data if _looks_like_wav(data) else b""
            except RuntimeError as exc:
                # The classic "run loop already started" case — pyttsx3 stores
                # engine state module-globally.  Reset and try ONCE more.
                if "run loop already started" in str(exc).lower() or "already started" in str(exc).lower():
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    try:
                        engine = self._make_engine()
                        data = self._synth_once(engine, text, tmp_path)
                        return data if _looks_like_wav(data) else b""
                    except Exception as exc2:
                        log.warning("Pyttsx3: retry after 'run loop already started' failed: %s", exc2)
                        return b""
                log.warning("Pyttsx3: synthesis RuntimeError: %s", exc)
                return b""
            except Exception as exc:
                log.warning("Pyttsx3: synthesis failed: %s", exc)
                return b""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def speak(self, text: str) -> None:
        text = british_polish(text)
        if not text:
            return
        try:
            engine = self._make_engine()
        except Exception as exc:
            log.warning("Pyttsx3.speak: engine init failed: %s", exc)
            return
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as exc:
            if "already started" in str(exc).lower():
                try:
                    engine.stop()
                except Exception:
                    pass
                try:
                    engine = self._make_engine()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc2:
                    log.warning("Pyttsx3.speak: retry failed: %s", exc2)
            else:
                log.warning("Pyttsx3.speak: %s", exc)
        except Exception as exc:
            log.warning("Pyttsx3.speak: %s", exc)


# --------------------------------------------------------------------------- #
#  espeak-ng / espeak — headless-Linux workhorse
# --------------------------------------------------------------------------- #
class EspeakTTS(TTSEngine):
    """Direct wrapper around ``espeak-ng`` or ``espeak``."""

    name = "espeak"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.output_format = "wav"

    def _binary(self) -> Optional[str]:
        return platform_utils.which("espeak-ng") or platform_utils.which("espeak")

    def is_available(self) -> bool:
        return self._binary() is not None

    def _voice_flag(self) -> str:
        # RP variant when the installed espeak knows it; fall back to en-gb.
        binary = self._binary()
        if not binary:
            return "en-gb"
        # No cheap way to enumerate voices without invoking espeak; try
        # en-gb-x-rp first at synthesis time and fall back if it fails.
        return "en-gb-x-rp"

    def _speed_wpm(self) -> int:
        base = 175
        return max(80, min(450, int(base * max(0.25, self.cfg.speed))))

    def synthesize(self, text: str) -> bytes:
        text = british_polish(text)
        if not text:
            return b""
        binary = self._binary()
        if not binary:
            return b""
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="jarvis-espeak-", suffix=".wav")
            os.close(fd)
            for voice in (self._voice_flag(), "en-gb"):
                result = platform_utils.run_command(
                    [binary, "-v", voice, "-s", str(self._speed_wpm()), "-w", tmp_path, text],
                    timeout=45.0,
                )
                if result.ok and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 44:
                    data = Path(tmp_path).read_bytes()
                    if _looks_like_wav(data):
                        return data
            return b""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def speak(self, text: str) -> None:
        data = self.synthesize(text)
        if not data:
            return
        try:
            from .audio_io import AudioPlayer  # type: ignore
            player = AudioPlayer()
            player.play(data)
            if hasattr(player, "wait"):
                player.wait()
        except Exception:
            _play_via_os(data, suffix=".wav")


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
_ENGINE_ORDER = ("piper", "edge", "sapi", "pyttsx3", "espeak")


def _make(name: str, cfg: TTSConfig, voices_dir: Optional[Path]) -> Optional[TTSEngine]:
    try:
        if name == "piper":
            return PiperTTS(cfg, voices_dir=voices_dir)
        if name == "edge":
            return EdgeTTS(cfg)
        if name == "sapi":
            from .windows_speech import SapiTTS
            return SapiTTS(cfg)
        if name == "pyttsx3":
            return Pyttsx3TTS(cfg)
        if name == "espeak":
            return EspeakTTS(cfg)
        if name == "null":
            return NullTTS(cfg)
    except Exception as exc:
        log.warning("TTS: could not construct %s: %s", name, exc)
    return None


def available_tts_engines(cfg: TTSConfig) -> List[str]:
    """List the engine names whose dependencies/binaries look present."""
    names: List[str] = []
    for name in _ENGINE_ORDER:
        engine = _make(name, cfg, None)
        if engine is not None:
            try:
                if engine.is_available():
                    names.append(name)
            except Exception:
                pass
    names.append("null")
    return names


def create_tts(cfg: TTSConfig, *, voices_dir: Optional[Path] = None) -> TTSEngine:
    """Pick a TTS engine according to ``cfg`` with graceful fallback."""
    if not getattr(cfg, "enabled", True):
        return NullTTS(cfg)

    engine_name = (cfg.engine or "auto").strip().lower()

    if engine_name != "auto":
        engine = _make(engine_name, cfg, voices_dir)
        if engine is not None:
            try:
                if engine.is_available():
                    return engine
            except Exception as exc:
                log.warning("TTS: %s.is_available raised: %s", engine_name, exc)
        log.info("TTS: %s not available, probing alternatives", engine_name)

    for name in _ENGINE_ORDER:
        engine = _make(name, cfg, voices_dir)
        if engine is None:
            continue
        try:
            if engine.is_available():
                return engine
        except Exception as exc:
            log.warning("TTS: %s.is_available raised: %s", name, exc)
    return NullTTS(cfg)


# --------------------------------------------------------------------------- #
#  SpeechQueue — never let synthesis block the agent
# --------------------------------------------------------------------------- #
# Sentence splitting for streamed speech. Abbreviations are the whole
# difficulty: naively breaking on every full stop turns "Dr. Hall" into two
# utterances with a pause in the middle of the name.
_ABBREV_GUARD = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|St|Prof|Sgt|Capt|Lt|Col|Gen|Rev|Hon|Jr|Sr|vs|etc|"
    r"approx|e\.g|i\.e|a\.m|p\.m|No|Fig|Vol|Ch|Sec)\.$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
#: Does this chunk actually finish a sentence?
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]*$")

#: A trailing fragment shorter than this, and with no sentence-ending
#: punctuation, is merged rather than spoken alone -- a dangling clause on its
#: own sounds clipped. A *complete* short sentence ("Yes, Sir.") is fine to
#: speak by itself and is never merged on length alone.
_MIN_CHUNK_CHARS = 24
#: Longer sentences are still worth splitting on a clause boundary, since a
#: 400-character sentence defeats the point of streaming.
_MAX_CHUNK_CHARS = 240


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into speakable chunks, respecting abbreviations.

    Returns whole sentences: handing a TTS engine half a sentence makes it
    guess the wrong intonation, which sounds worse than the delay it saves.
    """
    if not text or not text.strip():
        return []

    pieces: List[str] = []
    buffer = ""
    for candidate in _SENTENCE_END.split(text.strip()):
        candidate = candidate.strip()
        if not candidate:
            continue
        buffer = f"{buffer} {candidate}".strip() if buffer else candidate
        # A full stop that closes an abbreviation ("Dr.") is not the end of a
        # sentence, so keep accumulating until a real terminator turns up.
        if _ABBREV_GUARD.search(buffer):
            continue
        # A complete sentence always stands on its own, however short. Only an
        # unterminated fragment is held back to join what follows.
        if not _ENDS_SENTENCE.search(buffer) and len(buffer) < _MIN_CHUNK_CHARS:
            continue
        pieces.append(buffer)
        buffer = ""
    if buffer.strip():
        tail = buffer.strip()
        if pieces and not _ENDS_SENTENCE.search(tail) and len(tail) < _MIN_CHUNK_CHARS:
            pieces[-1] = f"{pieces[-1]} {tail}"
        else:
            pieces.append(tail)

    # Break up anything still too long, on clause boundaries where possible.
    out: List[str] = []
    for piece in pieces:
        while len(piece) > _MAX_CHUNK_CHARS:
            window = piece[:_MAX_CHUNK_CHARS]
            cut = max(window.rfind("; "), window.rfind(", "), window.rfind(" -- "))
            if cut < _MIN_CHUNK_CHARS:
                cut = window.rfind(" ")
            if cut < _MIN_CHUNK_CHARS:
                break
            out.append(piece[: cut + 1].strip())
            piece = piece[cut + 1:].strip()
        if piece:
            out.append(piece)
    return out or [text.strip()]


class SpeechQueue:
    """Threaded speaker: the agent enqueues text, a worker speaks it.

    ``stop()`` performs barge-in — clears the queue and asks the engine to
    stop its current playback if it can.  ``shutdown()`` tears the worker
    thread down cleanly; after it returns the thread is dead.
    """

    _SENTINEL: Any = object()

    def __init__(self, engine: TTSEngine, *, name: str = "jarvis-speech") -> None:
        self._engine = engine
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._speaking = threading.Event()
        self._stop_current = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name=name, daemon=True)
        self._worker.start()

    # ---- public API --------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str) -> None:
        """Enqueue an utterance.  Non-blocking.

        Long text is split into sentences and queued separately so the first
        one starts playing while the rest is still being synthesised. Piper
        takes roughly as long as the audio it produces, so speaking a
        paragraph as a single unit means several seconds of silence before
        anything is heard; per-sentence it is a fraction of a second.

        It also makes barge-in responsive: :meth:`stop` can drop the sentences
        that have not been spoken yet, instead of having to wait out one large
        buffer that is already inside the engine.
        """
        if self._shutdown.is_set():
            return
        if text is None:
            return
        text = str(text)
        if not text.strip():
            return
        for chunk in split_sentences(text):
            self._queue.put(chunk)

    def stop(self) -> None:
        """Barge-in: drop queued items and stop current playback if possible."""
        # Drain everything queued so far.
        drained = 0
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
        except queue.Empty:
            pass
        self._stop_current.set()
        # Ask the engine to interrupt if it supports it.
        for attr in ("stop", "interrupt", "cancel"):
            method = getattr(self._engine, attr, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    log.debug("SpeechQueue: engine.%s() raised: %s", attr, exc)
                break

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue is empty and no utterance is in progress.

        Returns True if fully drained within ``timeout`` (or without one),
        False if the timeout elapsed with work still in flight.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        # queue.join() blocks until all task_done() calls have caught up.
        while True:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return not self._speaking.is_set() and self._queue.unfinished_tasks == 0
            # A short poll — queue.Queue.join has no timeout.
            slice_ = 0.05 if remaining is None else min(0.05, remaining)
            if self._queue.unfinished_tasks == 0 and not self._speaking.is_set():
                return True
            time.sleep(max(0.001, slice_))

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the worker cleanly.  Thread is dead when this returns."""
        if self._shutdown.is_set() and not self._worker.is_alive():
            return
        self._shutdown.set()
        self.stop()
        # Unblock the worker's queue.get().
        self._queue.put(self._SENTINEL)
        self._worker.join(timeout=timeout)

    # ---- worker ------------------------------------------------------------
    def _run(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._queue.get()
            except Exception:
                continue
            try:
                if item is self._SENTINEL or self._shutdown.is_set():
                    return
                self._stop_current.clear()
                self._speaking.set()
                try:
                    self._engine.speak(item)
                except NotImplementedError:
                    # Fallback: synthesize and play through the OS default.
                    try:
                        data = self._engine.synthesize(item)
                        if data:
                            suffix = "." + getattr(self._engine, "output_format", "wav")
                            _play_via_os(data, suffix=suffix)
                    except Exception as exc:
                        log.warning("SpeechQueue: fallback synth/play failed: %s", exc)
                except Exception as exc:
                    log.warning("SpeechQueue: engine.speak raised: %s", exc)
                finally:
                    self._speaking.clear()
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass


__all__ = [
    "mp3_to_wav",
    "british_polish",
    "PiperTTS",
    "EdgeTTS",
    "Pyttsx3TTS",
    "EspeakTTS",
    "NullTTS",
    "create_tts",
    "available_tts_engines",
    "SpeechQueue",
    "split_sentences",
]
