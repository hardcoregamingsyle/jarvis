"""Transcription quality: the settings that decide whether STT is usable.

Accuracy is not something this suite can measure without audio and weights, so
what it pins instead is the *configuration* that produces it, plus the two
pieces of real logic that sit around the model:

* hallucination filtering — Whisper inventing "Thank you." over silence is the
  single most visible way transcription looks broken, because the assistant
  reacts to something nobody said;
* the decode options actually reaching faster-whisper — greedy decoding and
  ``condition_on_previous_text`` are what make a small model sound useless, and
  a default that never gets passed through is the same as no default at all.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from jarvis.core.config import STTConfig
from jarvis.speech.stt import (
    HALLUCINATIONS,
    FasterWhisperSTT,
    _looks_hallucinated,
)


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #
def test_decoding_defaults_are_the_accurate_ones():
    """Each of these was a specific cause of bad transcripts."""
    cfg = STTConfig()
    assert cfg.beam_size >= 5, "greedy decoding is what makes small models awful"
    assert cfg.condition_on_previous_text is False, (
        "conditioning propagates one mistake through the whole session"
    )
    assert "jarvis" in cfg.initial_prompt.lower(), (
        "the wake word must be biased for, or it is never recognised"
    )
    assert cfg.filter_hallucinations is True


def test_the_silence_window_does_not_cut_people_off():
    """0.9s truncates anyone who pauses to think mid-sentence."""
    assert STTConfig().silence_duration >= 1.1


def test_preroll_is_kept_so_the_wake_word_survives():
    """Energy detection trips after speech starts; without pre-roll the wake
    word arrives as "-arvis" and never matches."""
    assert STTConfig().preroll_seconds > 0


# --------------------------------------------------------------------------- #
#  Hallucination filtering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Thank you.",
        "thank you",
        "Thanks for watching!",
        "Thank you for watching",
        "you",
        "Bye.",
        "Okay.",
        "  Hmm  ",
        "Please subscribe",
    ],
)
def test_stock_phrases_over_silence_are_discarded(text):
    assert _looks_hallucinated(text) is True


def test_an_empty_transcript_is_not_treated_as_a_hallucination():
    """Nothing to discard, and the caller already handles empty separately."""
    assert _looks_hallucinated("") is False


@pytest.mark.parametrize(
    "text",
    [
        "thank you, that worked perfectly",
        "what is the time",
        "Jarvis, open the terminal",
        "okay, now shut down the machine",
        "bye for now, but first save the file",
    ],
)
def test_real_speech_is_never_discarded(text):
    """A false positive silently eats a real instruction, so the filter only
    ever fires on a short transcript that is *nothing but* a stock phrase."""
    assert _looks_hallucinated(text) is False


def test_the_filter_is_case_and_punctuation_insensitive():
    assert _looks_hallucinated("THANK YOU!!!") is True
    assert _looks_hallucinated("...thanks for watching...") is True


def test_every_listed_phrase_is_short_enough_to_ever_match():
    """A phrase longer than the length guard could never fire, so listing it
    would be a quiet lie about what is filtered."""
    from jarvis.speech.stt import _HALLUCINATION_MAX_CHARS

    unreachable = [p for p in HALLUCINATIONS if len(p) > _HALLUCINATION_MAX_CHARS]
    assert not unreachable, f"these can never match: {unreachable}"


# --------------------------------------------------------------------------- #
#  The options actually reaching the model
# --------------------------------------------------------------------------- #
class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.start = 0.0
        self.end = 1.0
        self.avg_logprob = -0.2


class FakeInfo:
    language = "en"


class FakeWhisper:
    """Records how it was called so the wiring can be asserted."""

    def __init__(self, text: str = "what is the time") -> None:
        self.text = text
        self.options: dict = {}

    def transcribe(self, audio: Any, **options):
        self.options = options
        return iter([FakeSegment(self.text)]), FakeInfo()


def engine_with(model: FakeWhisper, **cfg_overrides) -> FasterWhisperSTT:
    engine = FasterWhisperSTT(STTConfig(**cfg_overrides))
    engine._model = model
    return engine


def test_the_quality_options_are_passed_through_to_faster_whisper():
    model = FakeWhisper()
    engine = engine_with(model)

    engine.transcribe([0.2] * 16000, sample_rate=16000)

    assert model.options["beam_size"] == STTConfig().beam_size
    assert model.options["condition_on_previous_text"] is False
    assert model.options["initial_prompt"] == STTConfig().initial_prompt
    assert model.options["vad_filter"] is True


def test_a_hallucinated_transcript_comes_back_empty():
    engine = engine_with(FakeWhisper("Thank you."))
    assert engine.transcribe([0.2] * 16000).text == ""


def test_a_real_transcript_survives():
    engine = engine_with(FakeWhisper("open the terminal"))
    assert engine.transcribe([0.2] * 16000).text == "open the terminal"


def test_the_filter_can_be_switched_off():
    engine = engine_with(FakeWhisper("Thank you."), filter_hallucinations=False)
    assert engine.transcribe([0.2] * 16000).text == "Thank you."


class OldFasterWhisper(FakeWhisper):
    """A build that predates the newer keyword arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def transcribe(self, audio: Any, **options):
        self.attempts += 1
        if "beam_size" in options or "initial_prompt" in options:
            raise TypeError("unexpected keyword argument")
        self.options = options
        return iter([FakeSegment("still works")]), FakeInfo()


def test_an_older_faster_whisper_still_transcribes():
    """Passing an option an old build rejects must not mean no transcription."""
    model = OldFasterWhisper()
    engine = engine_with(model)

    result = engine.transcribe([0.2] * 16000)

    assert result.text == "still works"
    assert model.attempts == 2, "it should retry once without the new options"


def test_float16_is_corrected_to_int8_on_cpu(monkeypatch):
    """float16 is GPU-only; CTranslate2 refuses it and STT looks 'broken'."""
    built: List[dict] = []

    class Recorder:
        def __init__(self, name, **kwargs):
            built.append({"name": name, **kwargs})

    import sys
    import types

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = Recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    engine = FasterWhisperSTT(STTConfig(device="cpu", compute_type="float16"))
    engine._load()

    assert built and built[0]["compute_type"] == "int8"
    assert built[0]["cpu_threads"] >= 1, "all cores should be used on CPU"
