"""Tests for the zero-download Windows speech stack.

Hermetic by construction: no test here starts a real recogniser, speaks
aloud, or spawns PowerShell.  Both bridges are replaced — ``_run_powershell``
for the subprocess route and ``_has_win32com`` for the COM route — so the
suite behaves identically on Linux CI and on the Windows developer box.  The
handful of tests that genuinely need Windows say so with ``skipif``.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import wave
from typing import Any, List, Optional

import pytest

from jarvis.core.config import STTConfig, TTSConfig
from jarvis.core.contracts import STTEngine, TTSEngine, Transcript
from jarvis.core.platform_utils import IS_WINDOWS, CommandResult
from jarvis.speech import stt as stt_mod
from jarvis.speech import tts as tts_mod
from jarvis.speech import windows_speech as ws
from jarvis.speech.windows_speech import (
    SapiTTS,
    WindowsSTT,
    sapi_rate,
    sapi_volume,
    select_voice,
)

requires_windows = pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only behaviour")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def ok(stdout: str = "") -> CommandResult:
    return CommandResult(0, stdout, "")


def failed(stderr: str = "boom") -> CommandResult:
    return CommandResult(1, "", stderr)


def timed_out() -> CommandResult:
    return CommandResult(124, "", "[timed out]", timed_out=True)


def tiny_wav(seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


# Markers that identify each generated script uniquely.  "SpeechSynthesizer"
# is NOT usable: both the voice listing and the speak script contain it.
LISTING = "GetInstalledVoices"
SPEAK = "JARVIS_TTS_TEXT"
DICTATE = "DictationGrammar"
PROBE = "RecognizerInfo"


class FakeBridge:
    """Stand-in for :func:`jarvis.speech.windows_speech._run_powershell`.

    Records every ``(script, timeout, env)`` and replies from a marker-keyed
    routing table, so a test can assert both what was asked and how the reply
    was interpreted.  Voice-listing calls answer from ``voices`` by default so
    that a test focused on synthesis need not care about them.
    """

    def __init__(
        self,
        default: Optional[Any] = None,
        voices: Optional[CommandResult] = None,
        routes: Optional[dict] = None,
    ) -> None:
        self.calls: List[dict] = []
        self.default: Any = default if default is not None else ok()
        self.voices = voices if voices is not None else ok("")
        #: marker substring -> CommandResult or callable(env) -> CommandResult
        self.routes = dict(routes or {})

    def __call__(self, script: str, *, timeout: float, env: Optional[dict] = None):
        self.calls.append({"script": script, "timeout": timeout, "env": dict(env or {})})
        for marker, reply in self.routes.items():
            if marker in script:
                return reply(env or {}) if callable(reply) else reply
        if LISTING in script:
            return self.voices
        return self.default(env or {}) if callable(self.default) else self.default

    @property
    def scripts(self) -> List[str]:
        return [call["script"] for call in self.calls]

    def env_of(self, marker: str) -> dict:
        for call in self.calls:
            if marker in call["script"]:
                return call["env"]
        raise AssertionError(f"no bridge call contained {marker!r}")


@pytest.fixture(autouse=True)
def clean_probe_cache():
    """The recogniser probe cache is process-wide; never leak it between tests."""
    WindowsSTT.reset_probe_cache()
    yield
    WindowsSTT.reset_probe_cache()


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be Windows with PowerShell present but pywin32 absent."""
    monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
    monkeypatch.setattr(ws, "powershell_path", lambda: r"C:\Windows\powershell.exe")
    monkeypatch.setattr(ws, "_has_win32com", lambda: False)


# --------------------------------------------------------------------------- #
#  Voice selection
# --------------------------------------------------------------------------- #
class TestSelectVoice:
    UK = "Microsoft Hazel Desktop - English (Great Britain)"
    US_DAVID = "Microsoft David Desktop - English (United States)"
    US_ZIRA = "Microsoft Zira Desktop - English (United States)"
    RU = "Microsoft Irina Desktop - Russian (Russia)"

    def test_prefers_british_over_american(self):
        assert select_voice([self.US_DAVID, self.UK, self.US_ZIRA]) == 1

    def test_united_kingdom_wording_also_matches(self):
        voices = [self.US_DAVID, "Microsoft George - English (United Kingdom)"]
        assert select_voice(voices) == 1

    def test_falls_back_to_any_english_when_no_british_voice(self):
        """The verified state of this machine: two en-US voices, no en-GB."""
        assert select_voice([self.RU, self.US_DAVID, self.US_ZIRA]) == 1

    def test_falls_back_to_system_default_when_no_english_voice(self):
        assert select_voice([self.RU]) is None

    def test_empty_voice_list_does_not_crash(self):
        assert select_voice([]) is None
        assert select_voice(()) is None

    def test_none_entries_are_tolerated(self):
        assert select_voice([None, self.UK]) == 1  # type: ignore[list-item]

    def test_explicit_hint_wins_over_the_british_preference(self):
        chosen = select_voice([self.UK, self.US_ZIRA], hint="Zira")
        assert chosen == 1, "an explicit hint must beat the en-GB default"

    def test_hint_is_case_insensitive(self):
        assert select_voice([self.US_DAVID, self.US_ZIRA], hint="ZIRA") == 1

    def test_unmatched_hint_falls_through_to_the_ladder(self):
        """The shipped default hint matches nothing on a stock en-US machine."""
        chosen = select_voice([self.US_DAVID, self.US_ZIRA], hint="United Kingdom")
        assert chosen == 0, "an unmatched hint must degrade to the English voice"

    def test_config_default_hint_is_honoured_when_a_uk_voice_exists(self):
        hint = TTSConfig().sapi_voice_hint
        assert select_voice([self.US_DAVID, "Susan - English (United Kingdom)"], hint) == 1


# --------------------------------------------------------------------------- #
#  Rate / volume mapping
# --------------------------------------------------------------------------- #
class TestSapiScales:
    def test_normal_speed_is_sapi_zero(self):
        assert sapi_rate(1.0) == 0

    def test_rate_boundaries_clamp_to_the_sapi_range(self):
        assert sapi_rate(3.0) == 10
        assert sapi_rate(1.0 / 3.0) == -10
        assert sapi_rate(100.0) == 10
        assert sapi_rate(0.001) == -10

    def test_rate_is_monotonic_and_signed_correctly(self):
        assert sapi_rate(0.5) < 0 < sapi_rate(2.0)
        assert sapi_rate(1.2) < sapi_rate(1.8)

    def test_rate_survives_nonsense_input(self):
        assert sapi_rate(0.0) == -10
        assert sapi_rate(-4.0) == -10
        assert sapi_rate("fast") == 0  # type: ignore[arg-type]
        assert sapi_rate(None) == 0  # type: ignore[arg-type]

    def test_volume_boundaries(self):
        assert sapi_volume(0.0) == 0
        assert sapi_volume(1.0) == 100
        assert sapi_volume(0.5) == 50
        assert sapi_volume(2.5) == 100
        assert sapi_volume(-1.0) == 0

    def test_volume_survives_nonsense_input(self):
        assert sapi_volume("loud") == 100  # type: ignore[arg-type]

    def test_config_defaults_map_to_neutral_sapi_settings(self):
        cfg = TTSConfig()
        assert (sapi_rate(cfg.speed), sapi_volume(cfg.volume)) == (0, 100)


# --------------------------------------------------------------------------- #
#  SapiTTS availability
# --------------------------------------------------------------------------- #
class TestSapiAvailability:
    def test_false_off_windows(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", False)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)
        assert SapiTTS(TTSConfig()).is_available() is False

    def test_false_when_neither_bridge_exists(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: False)
        monkeypatch.setattr(ws, "powershell_path", lambda: None)
        assert SapiTTS(TTSConfig()).is_available() is False

    def test_true_with_powershell_alone(self, windows):
        """The whole point: nothing pip-installed, PowerShell only."""
        assert SapiTTS(TTSConfig()).is_available() is True

    def test_true_with_win32com_alone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)
        monkeypatch.setattr(ws, "powershell_path", lambda: None)
        assert SapiTTS(TTSConfig()).is_available() is True

    def test_returns_false_rather_than_raising(self, monkeypatch: pytest.MonkeyPatch):
        def explode() -> bool:
            raise OSError("registry on fire")

        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", explode)
        assert SapiTTS(TTSConfig()).is_available() is False

    def test_powershell_path_is_none_off_windows(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", False)
        assert ws.powershell_path() is None

    def test_powershell_path_survives_a_broken_which(self, monkeypatch: pytest.MonkeyPatch):
        def explode(_name: str):
            raise RuntimeError("PATH scan failed")

        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws.platform_utils, "which", explode)
        assert ws.powershell_path() is None

    def test_declares_the_engine_contract(self):
        engine = SapiTTS(TTSConfig())
        assert isinstance(engine, TTSEngine)
        assert engine.name == "sapi"
        assert engine.output_format == "wav"


# --------------------------------------------------------------------------- #
#  SapiTTS synthesis through the PowerShell bridge
# --------------------------------------------------------------------------- #
class TestSapiSynthesis:
    def test_empty_text_never_touches_the_bridge(self, windows, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(ws, "_run_powershell", bridge)
        assert SapiTTS(TTSConfig()).synthesize("") == b""
        assert SapiTTS(TTSConfig()).synthesize("   ") == b""
        assert bridge.calls == []

    def test_returns_the_wav_the_bridge_wrote(self, windows, monkeypatch):
        payload = tiny_wav()

        def write_wav(env: dict) -> CommandResult:
            with open(env["JARVIS_TTS_OUT"], "wb") as handle:
                handle.write(payload)
            return ok("JARVIS_TTS_DONE")

        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=write_wav))
        assert SapiTTS(TTSConfig()).synthesize("All systems nominal") == payload

    def test_empty_bytes_when_the_bridge_fails(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=failed()))
        assert SapiTTS(TTSConfig()).synthesize("All systems nominal") == b""

    def test_empty_bytes_when_powershell_is_absent(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", lambda *a, **kw: None)
        assert SapiTTS(TTSConfig()).synthesize("All systems nominal") == b""

    def test_non_wav_output_is_rejected(self, windows, monkeypatch):
        def write_junk(env: dict) -> CommandResult:
            with open(env["JARVIS_TTS_OUT"], "wb") as handle:
                handle.write(b"not audio at all")
            return ok()

        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=write_junk))
        assert SapiTTS(TTSConfig()).synthesize("All systems nominal") == b""

    def test_temp_file_is_removed_afterwards(self, windows, monkeypatch):
        seen: List[str] = []

        def write_wav(env: dict) -> CommandResult:
            seen.append(env["JARVIS_TTS_OUT"])
            with open(env["JARVIS_TTS_OUT"], "wb") as handle:
                handle.write(tiny_wav())
            return ok()

        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=write_wav))
        assert SapiTTS(TTSConfig()).synthesize("Cleanup check")
        assert seen and not os.path.exists(seen[0])

    def test_rate_volume_and_voice_travel_in_the_environment(self, windows, monkeypatch):
        bridge = FakeBridge(voices=ok(
            "JARVIS_VOICE:Microsoft Hazel Desktop\ten-GB\tEnglish (United Kingdom)\n"
            "JARVIS_VOICE:Microsoft David Desktop\ten-US\tEnglish (United States)\n"
        ))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        cfg = TTSConfig()
        cfg.speed = 3.0
        cfg.volume = 0.25
        SapiTTS(cfg).speak("Evening, Sir")

        env = bridge.env_of(SPEAK)
        assert env["JARVIS_TTS_RATE"] == "10"
        assert env["JARVIS_TTS_VOLUME"] == "25"
        assert env["JARVIS_TTS_VOICE"] == "Microsoft Hazel Desktop"

    def test_voice_stays_empty_when_nothing_english_is_installed(self, windows, monkeypatch):
        bridge = FakeBridge(
            voices=ok("JARVIS_VOICE:Microsoft Irina\tru-RU\tRussian (Russia)\n")
        )
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        SapiTTS(TTSConfig()).speak("Evening, Sir")
        assert bridge.env_of(SPEAK)["JARVIS_TTS_VOICE"] == ""

    def test_voice_list_is_fetched_once_per_engine(self, windows, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        engine = SapiTTS(TTSConfig())
        engine.speak("one")
        engine.speak("two")
        listings = [s for s in bridge.scripts if LISTING in s]
        assert len(listings) == 1, "the voice list must not be re-enumerated per utterance"

    def test_speech_text_is_never_interpolated_into_the_script(self, windows, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        SapiTTS(TTSConfig()).speak("'; Remove-Item C:\\ -Recurse; '")

        env = bridge.env_of(SPEAK)
        assert "Remove-Item" in env["JARVIS_TTS_TEXT"], "the text must travel in the environment"
        assert all("Remove-Item" not in s for s in bridge.scripts)

    def test_over_long_text_is_truncated_for_the_environment_block(self, windows, monkeypatch):
        bridge = FakeBridge()
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        SapiTTS(TTSConfig()).speak("word " * 4000)
        assert len(bridge.env_of(SPEAK)["JARVIS_TTS_TEXT"]) == ws._MAX_BRIDGE_TEXT

    def test_speak_does_not_raise_when_every_bridge_fails(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=failed()))
        SapiTTS(TTSConfig()).speak("Nobody can hear this")

    def test_speak_prefers_com_and_skips_powershell(self, monkeypatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)
        bridge = FakeBridge(default=ok())
        monkeypatch.setattr(ws, "_run_powershell", bridge)
        spoken: List[tuple] = []
        monkeypatch.setattr(
            SapiTTS, "_speak_com",
            lambda self, text, out: spoken.append((text, out)) or True,
        )

        SapiTTS(TTSConfig()).speak("Evening, Sir")
        assert spoken and spoken[0][1] is None
        assert bridge.calls == [], "PowerShell must not run when COM succeeded"

    def test_speak_falls_back_to_powershell_when_com_fails(self, monkeypatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)
        monkeypatch.setattr(SapiTTS, "_speak_com", lambda self, text, out: False)
        bridge = FakeBridge(default=ok())
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        SapiTTS(TTSConfig()).speak("Evening, Sir")
        assert any("SpeechSynthesizer" in s for s in bridge.scripts)


# --------------------------------------------------------------------------- #
#  Generated PowerShell
# --------------------------------------------------------------------------- #
class TestGeneratedPowerShell:
    def _argv(self, monkeypatch, run) -> List[str]:
        captured: dict = {}

        def fake_run_command(command, *, timeout, cwd=None, env=None, shell=False):
            captured["argv"] = list(command)
            captured["timeout"] = timeout
            captured["env"] = dict(env or {})
            return ok()

        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "powershell_path", lambda: r"C:\powershell.exe")
        monkeypatch.setattr(ws.platform_utils, "run_command", fake_run_command)
        run()
        return captured

    @staticmethod
    def _decode(argv: List[str]) -> str:
        assert "-EncodedCommand" in argv, "scripts must be passed inline, not as a temp file"
        blob = argv[argv.index("-EncodedCommand") + 1]
        return base64.b64decode(blob).decode("utf-16-le")

    def test_dictation_script_builds_a_recogniser_with_a_dictation_grammar(self, monkeypatch):
        captured = self._argv(
            monkeypatch,
            lambda: WindowsSTT(STTConfig()).transcribe([0.1, -0.1] * 800, 16000),
        )
        script = self._decode(captured["argv"])
        assert "SpeechRecognitionEngine" in script
        assert "DictationGrammar" in script
        assert "SetInputToWaveFile" in script
        assert "Add-Type -AssemblyName System.Speech" in script

    def test_dictation_reads_a_wave_file_not_the_live_microphone(self, monkeypatch):
        captured = self._argv(
            monkeypatch,
            lambda: WindowsSTT(STTConfig()).transcribe([0.1, -0.1] * 800, 16000),
        )
        script = self._decode(captured["argv"])
        assert "SetInputToDefaultAudioDevice" not in script
        assert captured["env"]["JARVIS_STT_WAV"].endswith(".wav")

    def test_synthesis_script_uses_system_speech(self, monkeypatch):
        monkeypatch.setattr(ws, "_has_win32com", lambda: False)
        captured = self._argv(monkeypatch, lambda: SapiTTS(TTSConfig()).speak("Hello"))
        script = self._decode(captured["argv"])
        assert "SpeechSynthesizer" in script or "GetInstalledVoices" in script

    def test_every_bridge_call_carries_a_positive_timeout(self, monkeypatch):
        captured = self._argv(
            monkeypatch,
            lambda: WindowsSTT(STTConfig()).transcribe([0.1] * 1600, 16000),
        )
        assert isinstance(captured["timeout"], float) and captured["timeout"] > 0

    def test_scripts_run_without_a_user_profile(self, monkeypatch):
        captured = self._argv(
            monkeypatch,
            lambda: WindowsSTT(STTConfig()).transcribe([0.1] * 1600, 16000),
        )
        assert "-NoProfile" in captured["argv"]
        assert "-NonInteractive" in captured["argv"]

    def test_bridge_returns_none_without_powershell(self, monkeypatch):
        monkeypatch.setattr(ws, "powershell_path", lambda: None)
        assert ws._run_powershell("Write-Output 'hi'", timeout=5.0) is None


# --------------------------------------------------------------------------- #
#  WindowsSTT availability + probe caching
# --------------------------------------------------------------------------- #
PROBE_OK = "JARVIS_RECOGNIZER:MS-1033-80-DESK\ten-US"


class TestWindowsSTTAvailability:
    def test_false_off_windows(self, monkeypatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", False)
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(PROBE_OK)))
        assert WindowsSTT(STTConfig()).is_available() is False

    def test_false_when_powershell_is_missing(self, monkeypatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "powershell_path", lambda: None)
        assert WindowsSTT(STTConfig()).is_available() is False

    def test_false_when_the_recogniser_will_not_construct(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=failed("no recogniser")))
        assert WindowsSTT(STTConfig()).is_available() is False

    def test_false_when_the_probe_output_is_unrecognised(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok("something else")))
        assert WindowsSTT(STTConfig()).is_available() is False

    def test_true_when_the_recogniser_constructs(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(PROBE_OK)))
        engine = WindowsSTT(STTConfig())
        assert engine.is_available() is True
        assert WindowsSTT.recognizer_info() == "MS-1033-80-DESK\ten-US"

    def test_returns_false_rather_than_raising(self, monkeypatch):
        def explode():
            raise OSError("PATH is gone")

        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "powershell_path", explode)
        assert WindowsSTT(STTConfig()).is_available() is False

    def test_probe_runs_once_across_two_calls(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok(PROBE_OK))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        engine = WindowsSTT(STTConfig())
        assert engine.is_available() is True
        assert engine.is_available() is True
        assert len(bridge.calls) == 1, "the ~1s recogniser probe must be cached"

    def test_probe_is_shared_across_instances(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok(PROBE_OK))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        assert WindowsSTT(STTConfig()).is_available() is True
        assert WindowsSTT(STTConfig()).is_available() is True
        assert len(bridge.calls) == 1

    def test_a_negative_probe_is_cached_too(self, windows, monkeypatch):
        bridge = FakeBridge(default=failed())
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        engine = WindowsSTT(STTConfig())
        assert engine.is_available() is False
        assert engine.is_available() is False
        assert len(bridge.calls) == 1

    def test_reset_forces_a_fresh_probe(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok(PROBE_OK))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        engine = WindowsSTT(STTConfig())
        engine.is_available()
        WindowsSTT.reset_probe_cache()
        assert WindowsSTT.recognizer_info() is None
        engine.is_available()
        assert len(bridge.calls) == 2

    def test_declares_the_engine_contract(self):
        engine = WindowsSTT(STTConfig())
        assert isinstance(engine, STTEngine)
        assert engine.name == "windows"


# --------------------------------------------------------------------------- #
#  WindowsSTT transcription
# --------------------------------------------------------------------------- #
class TestWindowsSTTTranscribe:
    SPEECH = [0.2, -0.2] * 1600  # 0.2 s at 16 kHz

    def test_parses_a_normal_result(self, windows, monkeypatch):
        stdout = "JARVIS_RESULT:0.8125\tturn on the lights\n"
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(stdout)))

        result = WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000)
        assert isinstance(result, Transcript)
        assert result.text == "turn on the lights"
        assert result.confidence == pytest.approx(0.8125)
        assert result.language == "en"

    def test_joins_multiple_recognition_passes(self, windows, monkeypatch):
        stdout = (
            "JARVIS_RESULT:1.0\tgood evening\n"
            "some unrelated PowerShell chatter\n"
            "JARVIS_RESULT:0.5\tsir\n"
        )
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(stdout)))

        result = WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000)
        assert result.text == "good evening sir"
        assert result.confidence == pytest.approx(0.75)

    def test_empty_transcript_on_bridge_failure(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=failed("Add-Type failed")))
        result = WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_empty_transcript_on_timeout(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=timed_out()))
        assert WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000).text == ""

    def test_empty_transcript_when_powershell_is_absent(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", lambda *a, **kw: None)
        assert WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000).text == ""

    def test_empty_audio_short_circuits_the_bridge(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok("JARVIS_RESULT:1.0\tghost words\n"))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        result = WindowsSTT(STTConfig()).transcribe([], 16000)
        assert result.text == ""
        assert bridge.calls == [], "empty audio must not spawn a recogniser"

    def test_recognising_nothing_yields_an_empty_transcript(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok("")))
        result = WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_undecodable_audio_yields_an_empty_transcript(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok("JARVIS_RESULT:1.0\tnope\n"))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        assert WindowsSTT(STTConfig()).transcribe(object(), 16000).text == ""
        assert bridge.calls == []

    def test_accepts_wav_bytes(self, windows, monkeypatch):
        monkeypatch.setattr(
            ws, "_run_powershell", FakeBridge(default=ok("JARVIS_RESULT:0.9\tstatus report\n"))
        )
        assert WindowsSTT(STTConfig()).transcribe(tiny_wav()).text == "status report"

    def test_accepts_a_wav_path(self, windows, monkeypatch, tmp_path):
        path = tmp_path / "utterance.wav"
        path.write_bytes(tiny_wav())
        monkeypatch.setattr(
            ws, "_run_powershell", FakeBridge(default=ok("JARVIS_RESULT:0.9\tfrom a file\n"))
        )
        assert WindowsSTT(STTConfig()).transcribe(str(path)).text == "from a file"

    def test_temp_wav_is_removed_even_when_the_bridge_fails(self, windows, monkeypatch):
        seen: List[str] = []

        def record(env: dict) -> CommandResult:
            seen.append(env["JARVIS_STT_WAV"])
            assert os.path.exists(env["JARVIS_STT_WAV"]), "the bridge needs a real file"
            return failed()

        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=record))
        WindowsSTT(STTConfig()).transcribe(self.SPEECH, 16000)
        assert seen and not os.path.exists(seen[0])

    def test_configured_language_is_offered_to_the_recogniser(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok(""))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        cfg = STTConfig()
        cfg.language = "en-GB"
        WindowsSTT(cfg).transcribe(self.SPEECH, 16000)
        assert bridge.calls[0]["env"]["JARVIS_STT_CULTURE"] == "en-GB"

    def test_timeout_scales_with_utterance_length(self, windows, monkeypatch):
        bridge = FakeBridge(default=ok(""))
        monkeypatch.setattr(ws, "_run_powershell", bridge)

        engine = WindowsSTT(STTConfig())
        engine.transcribe([0.1] * 16000, 16000)          # 1 s
        engine.transcribe([0.1] * (16000 * 20), 16000)   # 20 s
        short, long = bridge.calls[0]["timeout"], bridge.calls[1]["timeout"]
        assert long > short > 0


class TestBridgeOutputParser:
    def test_parses_confidence_and_text(self):
        text, confidence = ws._parse_bridge_output("JARVIS_RESULT:0.5\thello there\n")
        assert (text, confidence) == ("hello there", 0.5)

    def test_ignores_unmarked_lines(self):
        text, _ = ws._parse_bridge_output("warning: something\nJARVIS_RESULT:1.0\tokay\n")
        assert text == "okay"

    def test_blank_output_is_empty(self):
        assert ws._parse_bridge_output("") == ("", 0.0)
        assert ws._parse_bridge_output("nothing useful here") == ("", 0.0)

    def test_unparseable_confidence_still_yields_the_text(self):
        text, confidence = ws._parse_bridge_output("JARVIS_RESULT:NaNsense\tlights on\n")
        assert text == "lights on"
        assert confidence == 1.0

    def test_confidence_is_clamped(self):
        _, confidence = ws._parse_bridge_output("JARVIS_RESULT:9.5\tloud\n")
        assert confidence == 1.0


# --------------------------------------------------------------------------- #
#  Factory registration and probe order
# --------------------------------------------------------------------------- #
class TestFactoryRegistration:
    def test_stt_auto_order_is_the_documented_ladder(self):
        assert stt_mod._AUTO_ORDER == ("faster-whisper", "whisper", "vosk", "windows")

    def test_windows_engine_is_addressable_by_name(self):
        assert stt_mod._ENGINE_CLASSES["windows"] is WindowsSTT

    def test_tts_auto_order_is_the_documented_ladder(self):
        assert tts_mod._ENGINE_ORDER == ("piper", "edge", "sapi", "pyttsx3", "espeak")

    def test_sapi_engine_is_addressable_by_name(self):
        engine = tts_mod._make("sapi", TTSConfig(), None)
        assert isinstance(engine, SapiTTS)

    def test_auto_picks_windows_stt_when_no_real_engine_is_installed(
        self, windows, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        monkeypatch.setitem(sys.modules, "whisper", None)
        monkeypatch.setitem(sys.modules, "vosk", None)
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(PROBE_OK)))

        engine = stt_mod.create_stt(STTConfig(engine="auto"))
        assert isinstance(engine, WindowsSTT)

    def test_windows_stt_ranks_above_null_and_below_the_real_engines(
        self, windows, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        monkeypatch.setitem(sys.modules, "whisper", None)
        monkeypatch.setitem(sys.modules, "vosk", None)
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(PROBE_OK)))

        names = [e.name for e in stt_mod.available_stt_engines(STTConfig())]
        assert names == ["windows", "null"]

    def test_a_real_stt_engine_still_beats_windows(self, windows, monkeypatch):
        import types

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(default=ok(PROBE_OK)))

        engine = stt_mod.create_stt(STTConfig(engine="auto"))
        assert engine.name == "faster-whisper"

    def test_auto_picks_sapi_when_nothing_is_installed(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        monkeypatch.setattr(tts_mod.platform_utils, "which", lambda name: None)
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)

        engine = tts_mod.create_tts(TTSConfig(), voices_dir=tmp_path / "nowhere")
        assert isinstance(engine, SapiTTS)
        assert tts_mod.available_tts_engines(TTSConfig()) == ["sapi", "null"]

    def test_edge_outranks_sapi_because_it_has_the_british_voice(self, tmp_path, monkeypatch):
        import types

        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setitem(sys.modules, "edge_tts", types.ModuleType("edge_tts"))
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)

        engine = tts_mod.create_tts(TTSConfig(), voices_dir=tmp_path / "nowhere")
        assert engine.name == "edge"

    def test_sapi_outranks_pyttsx3_which_only_wraps_it(self, tmp_path, monkeypatch):
        import types

        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", types.ModuleType("pyttsx3"))
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(ws, "_has_win32com", lambda: True)

        engine = tts_mod.create_tts(TTSConfig(), voices_dir=tmp_path / "nowhere")
        assert engine.name == "sapi"

    def test_sapi_is_skipped_off_windows(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "piper", None)
        monkeypatch.setitem(sys.modules, "piper.voice", None)
        monkeypatch.setitem(sys.modules, "piper_tts", None)
        monkeypatch.setitem(sys.modules, "edge_tts", None)
        monkeypatch.setitem(sys.modules, "pyttsx3", None)
        monkeypatch.setattr(tts_mod.platform_utils, "which", lambda name: None)
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", False)

        engine = tts_mod.create_tts(TTSConfig(), voices_dir=tmp_path / "nowhere")
        assert isinstance(engine, tts_mod.NullTTS)


# --------------------------------------------------------------------------- #
#  Guidance
# --------------------------------------------------------------------------- #
class TestInstallHint:
    def test_names_both_routes_to_a_british_voice(self):
        hint = SapiTTS.install_hint()
        assert "Speech" in hint and "Add voices" in hint
        assert "Optional features" in hint

    def test_is_honest_that_edge_sounds_better(self):
        hint = SapiTTS.install_hint()
        assert "edge_tts" in hint
        assert "en-GB-RyanNeural" in hint

    def test_list_voices_is_empty_off_windows(self, monkeypatch):
        monkeypatch.setattr(ws.platform_utils, "IS_WINDOWS", False)
        assert SapiTTS.list_voices() == []

    def test_list_voices_parses_the_powershell_listing(self, windows, monkeypatch):
        listing = ok(
            "JARVIS_VOICE:Microsoft Zira Desktop\ten-US\tEnglish (United States)\n"
            "noise\n"
        )
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(voices=listing))
        voices = SapiTTS.list_voices()
        assert voices == ["Microsoft Zira Desktop - English (United States) [en-US]"]

    def test_list_voices_is_empty_when_the_bridge_fails(self, windows, monkeypatch):
        monkeypatch.setattr(ws, "_run_powershell", FakeBridge(voices=failed("Add-Type failed")))
        assert SapiTTS.list_voices() == []


# --------------------------------------------------------------------------- #
#  Windows-only smoke checks (no speech, no recognition)
# --------------------------------------------------------------------------- #
@requires_windows
class TestOnRealWindows:
    def test_powershell_ships_with_windows(self):
        assert ws.powershell_path() is not None

    def test_sapi_is_available_out_of_the_box(self):
        assert SapiTTS(TTSConfig()).is_available() is True

    def test_the_machine_reports_real_sapi_voices(self):
        voices = SapiTTS.list_voices()
        assert voices, "a Windows box always has at least one SAPI voice"
        assert all(isinstance(v, str) and v for v in voices)

    def test_a_voice_is_chosen_from_whatever_is_installed(self):
        """Verified state here: only en-US voices, so the en-GB hint misses."""
        voices = SapiTTS.list_voices()
        chosen = select_voice(voices, TTSConfig().sapi_voice_hint)
        assert chosen is None or 0 <= chosen < len(voices)
