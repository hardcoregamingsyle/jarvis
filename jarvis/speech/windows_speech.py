"""Windows-native speech: SAPI5 output and System.Speech dictation input.

Every other engine in :mod:`jarvis.speech` needs something downloaded — a
Piper voice, a Whisper checkpoint, a Vosk model, or an internet connection.
Windows already ships a complete speech stack that needs none of that:

* **SAPI5** (``SAPI.SpVoice``) for text-to-speech, and
* **System.Speech.Recognition** for offline dictation.

This module reaches both of them, so a stock Windows box has a working voice
on the first run with nothing installed at all.  Two bridges are used, in
order of preference:

``win32com``
    Present when ``pywin32`` is installed.  Fast, in-process, no subprocess.

PowerShell
    ``Add-Type -AssemblyName System.Speech``.  Slower (a process per call)
    but requires *nothing* to be installed, which is the entire point of this
    module.  Scripts are passed with ``-EncodedCommand`` so no temporary
    ``.ps1`` file is written and the execution policy never applies, and all
    user text travels in environment variables so it is never interpolated
    into a script body.

Neither engine is the best of its kind.  :class:`~jarvis.speech.tts.EdgeTTS`
sounds markedly more British and ``faster-whisper`` transcribes markedly more
accurately.  These exist so that voice works before any of that is installed.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from ..core import platform_utils
from ..core.config import STTConfig, TTSConfig
from ..core.contracts import STTEngine, TTSEngine, Transcript

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  PowerShell bridge
# --------------------------------------------------------------------------- #
#: Prefixed to every generated script.  ``$ProgressPreference`` suppresses the
#: CLIXML progress records PowerShell otherwise writes to stderr on first use,
#: and pinning the console encoding keeps non-ASCII recognised text intact.
_PS_PREAMBLE = (
    "$ErrorActionPreference = 'Stop'\n"
    "$ProgressPreference = 'SilentlyContinue'\n"
    "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }\n"
)

#: Environment values longer than this are refused rather than risking the
#: ~32 KB Windows environment-block limit on a runaway utterance.
_MAX_BRIDGE_TEXT = 6000


def powershell_path() -> Optional[str]:
    """Absolute path to a PowerShell that can host System.Speech, else ``None``.

    Windows PowerShell (5.1) is preferred over ``pwsh``: PowerShell 7 runs on
    .NET Core, where ``Add-Type -AssemblyName System.Speech`` frequently fails.
    """
    if not platform_utils.IS_WINDOWS:
        return None
    try:
        return platform_utils.which("powershell") or platform_utils.which("pwsh")
    except Exception:  # a patched/broken which() must not take the engine down
        return None


def _run_powershell(
    script: str,
    *,
    timeout: float,
    env: Optional[dict] = None,
) -> Optional[Any]:
    """Run ``script`` in PowerShell and return a ``CommandResult``, or ``None``.

    ``None`` means PowerShell itself could not be found.  Delegates to
    :func:`jarvis.core.platform_utils.run_command`, which applies the timeout
    and kills *and* reaps the child when it expires, and never raises.
    """
    exe = powershell_path()
    if exe is None:
        return None
    payload = (_PS_PREAMBLE + script).encode("utf-16-le")
    encoded = base64.b64encode(payload).decode("ascii")
    return platform_utils.run_command(
        [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        timeout=timeout,
        env=env,
    )


def _bridge_ok(result: Optional[Any]) -> bool:
    """True when a bridge call ran to completion with a zero exit status.

    stderr is deliberately ignored: PowerShell writes CLIXML progress records
    there on a cold start even when the script succeeds.
    """
    if result is None:
        return False
    if getattr(result, "timed_out", False):
        return False
    return getattr(result, "returncode", 1) == 0


def _has_win32com() -> bool:
    """True when ``pywin32``'s COM client can be imported.  Never raises."""
    try:
        import win32com.client  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


class _ComApartment:
    """Context manager that initialises COM for the calling thread.

    ``SpeechQueue`` speaks on a worker thread, and ``Dispatch`` there fails
    with "CoInitialize has not been called" unless the thread joins an
    apartment first.  ``CoInitialize``/``CoUninitialize`` are reference
    counted per thread, so balancing them is safe even on the main thread
    where something else already initialised COM.
    """

    def __init__(self) -> None:
        self._pythoncom: Any = None

    def __enter__(self) -> "_ComApartment":
        try:
            import pythoncom  # type: ignore

            pythoncom.CoInitialize()
            self._pythoncom = pythoncom
        except Exception as exc:
            log.debug("COM apartment init skipped: %s", exc)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._pythoncom is None:
            return
        try:
            self._pythoncom.CoUninitialize()
        except Exception as exc:
            log.debug("COM apartment teardown failed: %s", exc)


def _com_voice_descriptions() -> List[Tuple[Any, str]]:
    """(index, description) for every SAPI voice.  Must run inside an apartment.

    A free function rather than a method so that its frame — and with it every
    COM pointer it touched — is gone before the caller leaves the apartment.
    """
    import win32com.client  # type: ignore

    voice = None
    tokens = None
    try:
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        tokens = voice.GetVoices()
        out: List[Tuple[Any, str]] = []
        for index in range(tokens.Count):
            try:
                description = str(tokens.Item(index).GetDescription())
            except Exception:
                description = ""
            out.append((index, description))
        return out
    finally:
        tokens = None
        voice = None


def _polish(text: Any) -> str:
    """Apply the shared British text polish, degrading to ``str()``.

    Imported lazily: :mod:`jarvis.speech.tts` imports *this* module to build
    :class:`SapiTTS`, so a module-level import would close the loop.
    """
    try:
        from .tts import british_polish

        return british_polish(text)
    except Exception:
        return "" if text is None else str(text)


def _looks_like_wav(data: bytes) -> bool:
    return bool(data) and data[:4] == b"RIFF" and b"WAVE" in data[:16]


# --------------------------------------------------------------------------- #
#  Voice selection (pure, so it is testable without a Windows box)
# --------------------------------------------------------------------------- #
#: Substrings that mark a voice as British.  SAPI spells the desktop voices'
#: region "Great Britain" while the OneCore voices say "United Kingdom", so
#: both have to be here; Hazel, George and Susan are the shipping en-GB names.
_UK_MARKERS: Tuple[str, ...] = (
    "united kingdom", "great britain", "en-gb", "en_gb", "british",
    "hazel", "george", "susan",
)

_ENGLISH_MARKERS: Tuple[str, ...] = (
    "english", "en-us", "en_us", "en-gb", "en-au", "en-ca", "en-in",
    "en-ie", "en-nz", "en-za",
)


def select_voice(descriptions: Sequence[str], hint: str = "") -> Optional[int]:
    """Index of the best voice in ``descriptions``, or ``None`` for the default.

    The ladder is: an explicit ``hint`` substring, then any British voice,
    then any English voice, then ``None`` — which leaves SAPI on whatever the
    user has configured system-wide.  ``None`` is a success, not a failure:
    on a machine with no English voice at all, the system default is a better
    answer than an arbitrary pick.
    """
    items = [str(d or "").lower() for d in (descriptions or ())]
    if not items:
        return None

    needle = str(hint or "").strip().lower()
    if needle:
        for index, description in enumerate(items):
            if needle in description:
                return index

    for markers in (_UK_MARKERS, _ENGLISH_MARKERS):
        for index, description in enumerate(items):
            if any(marker in description for marker in markers):
                return index
    return None


def sapi_rate(speed: float) -> int:
    """Map ``TTSConfig.speed`` (1.0 = normal) onto SAPI's ``Rate`` of -10..10.

    SAPI's scale is roughly geometric and spans about a third to triple speed
    across its range, so the mapping is logarithmic with base 3: 1.0 maps to
    0, 3.0 to +10, and 1/3 to -10.  Anything outside is clamped.
    """
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return 0
    if value <= 0.0:
        return -10
    rate = int(round(10.0 * math.log(value) / math.log(3.0)))
    return max(-10, min(10, rate))


def sapi_volume(volume: float) -> int:
    """Map ``TTSConfig.volume`` (0.0..1.0) onto SAPI's ``Volume`` of 0..100."""
    try:
        value = float(volume)
    except (TypeError, ValueError):
        return 100
    return max(0, min(100, int(round(value * 100.0))))


# --------------------------------------------------------------------------- #
#  SAPI5 text-to-speech
# --------------------------------------------------------------------------- #
#: SpeechAudioFormatType.SAFT22kHz16BitMono — SAPI's own native output format.
_SAFT_22KHZ_16BIT_MONO = 22
#: SpeechStreamFileMode.SSFMCreateForWrite.
_SSFM_CREATE_FOR_WRITE = 3
#: SpeechVoiceSpeakFlags.SVSFDefault — synchronous.
_SVSF_DEFAULT = 0

_PS_LIST_VOICES = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $s.GetInstalledVoices()) {
  if ($v.Enabled) {
    $i = $v.VoiceInfo
    Write-Output ('JARVIS_VOICE:' + $i.Name + "`t" + $i.Culture.Name + "`t" + $i.Culture.DisplayName)
  }
}
$s.Dispose()
"""

_PS_SPEAK = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:JARVIS_TTS_VOICE) { try { $s.SelectVoice($env:JARVIS_TTS_VOICE) } catch { } }
$s.Rate = [int]$env:JARVIS_TTS_RATE
$s.Volume = [int]$env:JARVIS_TTS_VOLUME
if ($env:JARVIS_TTS_OUT) { $s.SetOutputToWaveFile($env:JARVIS_TTS_OUT) }
try {
  $s.Speak($env:JARVIS_TTS_TEXT)
} finally {
  try { $s.SetOutputToNull() } catch { }
  $s.Dispose()
}
Write-Output 'JARVIS_TTS_DONE'
"""


class SapiTTS(TTSEngine):
    """Windows SAPI5 speech, through ``win32com`` or a PowerShell bridge.

    Needs nothing installed: ``pywin32`` is used when present purely because
    it is faster than spawning a process per utterance.  Voice choice follows
    :func:`select_voice` against ``TTSConfig.sapi_voice_hint``.

    A stock Windows 11 install ships only ``Microsoft David Desktop`` and
    ``Microsoft Zira Desktop``, both en-US, so the default "United Kingdom"
    hint usually matches nothing and the engine falls back to an American
    voice.  :meth:`install_hint` explains how to add a British one — and why
    ``edge_tts`` is still the better choice if there is any internet.
    """

    name = "sapi"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.output_format = "wav"
        #: (identifier, description) pairs cached from the PowerShell bridge.
        self._ps_voices: Optional[List[Tuple[str, str]]] = None

    # ---- availability ------------------------------------------------------
    def is_available(self) -> bool:
        """True on Windows when either bridge can be reached.  Never raises."""
        if not platform_utils.IS_WINDOWS:
            return False
        try:
            return _has_win32com() or powershell_path() is not None
        except Exception:
            return False

    # ---- voice discovery ---------------------------------------------------
    @classmethod
    def list_voices(cls) -> List[str]:
        """Descriptions of every installed SAPI voice.  ``[]`` when unavailable."""
        if not platform_utils.IS_WINDOWS:
            return []
        com = cls._list_voices_com()
        if com:
            return [description for _token, description in com]
        return [description for _name, description in cls._list_voices_ps()]

    @staticmethod
    def install_hint() -> str:
        """Honest guidance on adding a British SAPI voice."""
        return (
            "To add a British voice to Windows:\n"
            "  Settings > Time & language > Speech > Manage voices > Add voices,\n"
            "  then choose 'English (United Kingdom)'. The equivalent route is\n"
            "  Settings > Apps > Optional features > Add a feature.\n"
            "  A restart (or at least a new sign-in) is usually needed before the\n"
            "  voice appears.\n"
            "  Caveat: voices added this way are OneCore voices, and Windows does\n"
            "  not always expose them to classic SAPI5 applications such as this\n"
            "  one, so the new voice may not show up in list_voices().\n"
            "  Honestly: edge_tts with en-GB-RyanNeural remains the better British\n"
            "  voice by a wide margin. SAPI is here so that speech works with\n"
            "  nothing downloaded and no internet, not because it sounds good."
        )

    @staticmethod
    def _list_voices_com() -> List[Tuple[Any, str]]:
        """(index, description) for each SAPI voice via COM.  ``[]`` on failure."""
        if not _has_win32com():
            return []
        try:
            with _ComApartment():
                return _com_voice_descriptions()
        except Exception as exc:
            log.debug("SapiTTS: COM voice enumeration failed: %s", exc)
            return []

    @classmethod
    def _list_voices_ps(cls) -> List[Tuple[str, str]]:
        """(name, description) for each SAPI voice via PowerShell."""
        result = _run_powershell(_PS_LIST_VOICES, timeout=30.0)
        if not _bridge_ok(result):
            log.debug("SapiTTS: PowerShell voice enumeration unavailable")
            return []
        out: List[Tuple[str, str]] = []
        for line in (getattr(result, "stdout", "") or "").splitlines():
            line = line.strip()
            if not line.startswith("JARVIS_VOICE:"):
                continue
            fields = line[len("JARVIS_VOICE:"):].split("\t")
            voice_name = fields[0].strip() if fields else ""
            if not voice_name:
                continue
            culture = fields[1].strip() if len(fields) > 1 else ""
            display = fields[2].strip() if len(fields) > 2 else ""
            out.append((voice_name, f"{voice_name} - {display} [{culture}]"))
        return out

    def _ps_voice_name(self) -> str:
        """Name of the PowerShell voice to select, or ``""`` for the default."""
        if self._ps_voices is None:
            self._ps_voices = self._list_voices_ps()
        pairs = self._ps_voices
        if not pairs:
            return ""
        index = select_voice([description for _n, description in pairs],
                             getattr(self.cfg, "sapi_voice_hint", "") or "")
        if index is None:
            return ""
        return pairs[index][0]

    # ---- COM path ----------------------------------------------------------
    def _speak_com(self, text: str, out_path: Optional[str]) -> bool:
        """Speak (or render to ``out_path``) through COM.  False if unusable.

        The COM work lives in a nested call so every interface pointer is
        released when that frame unwinds — i.e. strictly before the apartment
        is torn down.  Releasing an ``IUnknown`` after ``CoUninitialize`` makes
        pywin32 print a Win32 exception straight to stderr, which no library
        should ever do.
        """
        if not _has_win32com():
            return False
        try:
            with _ComApartment():
                return self._speak_com_inner(text, out_path)
        except Exception as exc:
            log.warning("SapiTTS: COM path failed: %s", exc)
            return False

    def _speak_com_inner(self, text: str, out_path: Optional[str]) -> bool:
        import win32com.client  # type: ignore

        voice = None
        stream = None
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._apply_com_settings(voice)
            if out_path is not None:
                stream = win32com.client.Dispatch("SAPI.SpFileStream")
                try:
                    stream.Format.Type = _SAFT_22KHZ_16BIT_MONO
                except Exception as exc:
                    log.debug("SapiTTS: could not pin wave format: %s", exc)
                stream.Open(out_path, _SSFM_CREATE_FOR_WRITE, False)
                voice.AudioOutputStream = stream
            voice.Speak(text, _SVSF_DEFAULT)
            return True
        finally:
            if stream is not None:
                try:
                    stream.Close()
                except Exception as exc:
                    log.debug("SapiTTS: stream close failed: %s", exc)
            stream = None
            voice = None

    def _apply_com_settings(self, voice: Any) -> None:
        """Push voice choice, rate and volume onto a live ``SpVoice``."""
        try:
            tokens = voice.GetVoices()
            descriptions: List[str] = []
            for index in range(tokens.Count):
                try:
                    descriptions.append(str(tokens.Item(index).GetDescription()))
                except Exception:
                    descriptions.append("")
            chosen = select_voice(descriptions, getattr(self.cfg, "sapi_voice_hint", "") or "")
            if chosen is not None:
                voice.Voice = tokens.Item(chosen)
        except Exception as exc:
            log.debug("SapiTTS: voice selection failed, keeping default: %s", exc)
        try:
            voice.Rate = sapi_rate(getattr(self.cfg, "speed", 1.0))
            voice.Volume = sapi_volume(getattr(self.cfg, "volume", 1.0))
        except Exception as exc:
            log.debug("SapiTTS: rate/volume not applied: %s", exc)

    # ---- PowerShell path ---------------------------------------------------
    def _speak_ps(self, text: str, out_path: Optional[str]) -> bool:
        """Speak (or render to ``out_path``) through PowerShell.  False on failure."""
        if len(text) > _MAX_BRIDGE_TEXT:
            log.warning(
                "SapiTTS: utterance of %d characters truncated to %d for the "
                "PowerShell bridge", len(text), _MAX_BRIDGE_TEXT,
            )
            text = text[:_MAX_BRIDGE_TEXT]
        env = {
            "JARVIS_TTS_TEXT": text,
            "JARVIS_TTS_VOICE": self._ps_voice_name(),
            "JARVIS_TTS_RATE": str(sapi_rate(getattr(self.cfg, "speed", 1.0))),
            "JARVIS_TTS_VOLUME": str(sapi_volume(getattr(self.cfg, "volume", 1.0))),
            "JARVIS_TTS_OUT": out_path or "",
        }
        # Playback happens in real time, so the budget must cover the utterance
        # itself.  Speech runs at roughly 12 characters a second; dividing by
        # six leaves a factor of two in hand on top of a generous floor.
        timeout = min(600.0, 60.0 + len(text) / 6.0)
        result = _run_powershell(_PS_SPEAK, timeout=timeout, env=env)
        if not _bridge_ok(result):
            log.warning(
                "SapiTTS: PowerShell bridge failed (%s)",
                (getattr(result, "stderr", "") or "no PowerShell").strip()[:200] or "non-zero exit",
            )
            return False
        return True

    # ---- TTSEngine contract ------------------------------------------------
    def synthesize(self, text: str) -> bytes:
        """Render ``text`` to WAV bytes.  ``b""`` when nothing could be produced."""
        spoken = _polish(text)
        if not spoken:
            return b""
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="jarvis-sapi-", suffix=".wav")
            os.close(fd)
            if not self._speak_com(spoken, tmp_path):
                if not self._speak_ps(spoken, tmp_path):
                    return b""
            data = Path(tmp_path).read_bytes() if os.path.exists(tmp_path) else b""
            if _looks_like_wav(data):
                return data
            log.warning("SapiTTS: bridge produced %d bytes that are not WAV", len(data))
            return b""
        except Exception as exc:
            log.warning("SapiTTS.synthesize failed: %s", exc)
            return b""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    log.debug("SapiTTS: temp file cleanup failed: %s", exc)

    def speak(self, text: str) -> None:
        """Speak ``text`` synchronously through the default output device."""
        spoken = _polish(text)
        if not spoken:
            return
        if self._speak_com(spoken, None):
            return
        if not self._speak_ps(spoken, None):
            log.warning("SapiTTS.speak: no working SAPI bridge; utterance dropped")


# --------------------------------------------------------------------------- #
#  System.Speech offline dictation
# --------------------------------------------------------------------------- #
_PS_RECOGNIZER_PROBE = """
Add-Type -AssemblyName System.Speech
$e = New-Object System.Speech.Recognition.SpeechRecognitionEngine
Write-Output ('JARVIS_RECOGNIZER:' + $e.RecognizerInfo.Id + "`t" + $e.RecognizerInfo.Culture.Name)
$e.Dispose()
"""

_PS_DICTATE = """
Add-Type -AssemblyName System.Speech
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$engine = $null
if ($env:JARVIS_STT_CULTURE) {
  foreach ($info in [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()) {
    if ($info.Culture.Name -eq $env:JARVIS_STT_CULTURE -or
        $info.Culture.TwoLetterISOLanguageName -eq $env:JARVIS_STT_CULTURE) {
      $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($info)
      break
    }
  }
}
if ($null -eq $engine) { $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine }
$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$engine.SetInputToWaveFile($env:JARVIS_STT_WAV)
try {
  $guard = 0
  while ($guard -lt 500) {
    $guard = $guard + 1
    $result = $engine.Recognize()
    if ($null -eq $result) { break }
    Write-Output ('JARVIS_RESULT:' + $result.Confidence.ToString($ci) + "`t" + $result.Text)
  }
} finally {
  try { $engine.SetInputToNull() } catch { }
  $engine.Dispose()
}
"""


def _parse_bridge_output(stdout: str) -> Tuple[str, float]:
    """Join ``JARVIS_RESULT:`` lines into (text, mean confidence).

    A run that recognised nothing yields ``("", 0.0)`` — which is exactly what
    an empty :class:`Transcript` carries, so callers need no special case.
    """
    phrases: List[str] = []
    confidences: List[float] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("JARVIS_RESULT:"):
            continue
        payload = line[len("JARVIS_RESULT:"):]
        confidence_text, _, phrase = payload.partition("\t")
        phrase = phrase.strip()
        if phrase:
            phrases.append(phrase)
        try:
            confidences.append(max(0.0, min(1.0, float(confidence_text))))
        except (TypeError, ValueError):
            pass
    text = " ".join(phrases).strip()
    if not text:
        return "", 0.0
    confidence = sum(confidences) / len(confidences) if confidences else 1.0
    return text, confidence


class WindowsSTT(STTEngine):
    """Offline dictation via .NET ``System.Speech.Recognition``.

    No model download, no torch, no network — the recogniser is part of
    Windows.  It is driven through a generated PowerShell script because this
    interpreter has no usable binding to it: the script builds a
    ``SpeechRecognitionEngine``, loads a ``DictationGrammar``, and reads from a
    **wave file** rather than the live microphone, since JARVIS already owns
    audio capture via :class:`~jarvis.speech.audio_io.AudioRecorder`.

    Two honest caveats.  Only the recognisers Windows ships are available, and
    a stock machine has just ``MS-1033-80-DESK`` (en-US) — so a British speaker
    is transcribed by an American model.  And accuracy is well below
    ``faster-whisper`` on any realistic utterance.  This engine exists so voice
    works with zero downloads, not because it is better; it therefore sits
    below the real engines in the auto-probe order but above :class:`NullSTT`,
    which does nothing at all.
    """

    name = "windows"

    #: Process-wide probe cache.  Constructing the recogniser costs about a
    #: second, which is intolerable per utterance, and the answer cannot change
    #: while the process runs unless the user installs a speech feature —
    #: hence :meth:`reset_probe_cache`.
    _probe_lock = threading.Lock()
    _probe_result: Optional[bool] = None
    _probe_info: Optional[str] = None

    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg

    # ---- availability ------------------------------------------------------
    @classmethod
    def reset_probe_cache(cls) -> None:
        """Forget the cached recogniser probe so the next call re-runs it."""
        with cls._probe_lock:
            cls._probe_result = None
            cls._probe_info = None

    @classmethod
    def recognizer_info(cls) -> Optional[str]:
        """``"<id>\\t<culture>"`` of the probed recogniser, if one was found."""
        return cls._probe_info

    @classmethod
    def _probe(cls) -> bool:
        """Construct a recogniser once and remember whether it worked."""
        with cls._probe_lock:
            if cls._probe_result is not None:
                return cls._probe_result
            ok = False
            info: Optional[str] = None
            result = _run_powershell(_PS_RECOGNIZER_PROBE, timeout=45.0)
            if _bridge_ok(result):
                for line in (getattr(result, "stdout", "") or "").splitlines():
                    line = line.strip()
                    if line.startswith("JARVIS_RECOGNIZER:"):
                        info = line[len("JARVIS_RECOGNIZER:"):].strip()
                        ok = True
                        break
            if ok:
                log.info("WindowsSTT: recogniser available (%s)", info)
            else:
                log.debug("WindowsSTT: no usable System.Speech recogniser")
            cls._probe_result = ok
            cls._probe_info = info
            return ok

    def is_available(self) -> bool:
        """True on Windows when PowerShell hosts a working recogniser."""
        if not platform_utils.IS_WINDOWS:
            return False
        try:
            if powershell_path() is None:
                return False
            return self._probe()
        except Exception:
            return False

    # ---- transcription -----------------------------------------------------
    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        """Transcribe ``audio``; an empty :class:`Transcript` on any failure."""
        try:
            samples, rate = self._prepare(audio, sample_rate)
        except Exception as exc:
            log.warning("WindowsSTT: audio preparation failed: %s", exc)
            return self._empty()
        if not samples:
            return self._empty()

        try:
            from .audio_io import wav_bytes

            data = wav_bytes(samples, sample_rate=rate)
        except Exception as exc:
            log.warning("WindowsSTT: could not encode audio: %s", exc)
            return self._empty()

        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="jarvis-winstt-", suffix=".wav")
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            # Recognition runs faster than real time but not by a fixed factor;
            # budget twice the utterance plus a fixed start-up allowance.
            duration = len(samples) / float(rate or 1)
            timeout = min(600.0, 30.0 + 2.0 * duration)
            env = {
                "JARVIS_STT_WAV": tmp_path,
                "JARVIS_STT_CULTURE": str(getattr(self.cfg, "language", "") or ""),
            }
            result = _run_powershell(_PS_DICTATE, timeout=timeout, env=env)
        except Exception as exc:
            log.warning("WindowsSTT: bridge invocation failed: %s", exc)
            return self._empty()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    log.debug("WindowsSTT: temp file cleanup failed: %s", exc)

        if result is None:
            log.warning("WindowsSTT: PowerShell is not available")
            return self._empty()
        if getattr(result, "timed_out", False):
            log.warning("WindowsSTT: recognition timed out after %.0fs", timeout)
            return self._empty()
        if getattr(result, "returncode", 1) != 0:
            log.warning(
                "WindowsSTT: bridge exited %s: %s",
                getattr(result, "returncode", "?"),
                (getattr(result, "stderr", "") or "").strip()[:200],
            )
            return self._empty()

        text, confidence = _parse_bridge_output(getattr(result, "stdout", "") or "")
        if not text:
            return self._empty()
        return Transcript(
            text=text,
            language=getattr(self.cfg, "language", None) or None,
            confidence=confidence,
            segments=(),
        )

    def _empty(self) -> Transcript:
        return Transcript(
            text="",
            language=getattr(self.cfg, "language", None) or None,
            confidence=0.0,
            segments=(),
        )

    @staticmethod
    def _prepare(audio: Any, sample_rate: int) -> Tuple[List[float], int]:
        """Coerce any accepted input to 16 kHz mono float samples.

        Reuses the sibling module's converter rather than growing a second one
        that could drift; the import is deferred because :mod:`jarvis.speech.stt`
        imports this module to register the engine.
        """
        from .stt import _prepare_audio

        return _prepare_audio(audio, sample_rate)


__all__ = [
    "SapiTTS",
    "WindowsSTT",
    "select_voice",
    "sapi_rate",
    "sapi_volume",
    "powershell_path",
]
