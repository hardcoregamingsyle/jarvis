"""The two-model speech path, and the silent-reply bug that made it necessary.

Two separate things are covered here.

**The silent reply.** A thinking model that runs out of token budget mid-thought
returns an unclosed ``<think>`` block and nothing else. ``strip_thinking``
correctly reduces that to ``""``, and before :func:`salvage_thinking` existed
the agent loop then had literally nothing to say — which, in a *voice*
assistant, is indistinguishable from a crash. These tests pin the recovery.

**The voice model.** The small model that phrases replies aloud must never be
able to make things up, and must never be able to break a turn. Both are
tested against a backend that is deliberately hostile.
"""

from __future__ import annotations

from typing import Optional

import pytest

from jarvis.core.config import LLMConfig
from jarvis.core.contracts import GenerationConfig, LLMBackend, LLMResult
from jarvis.llm.base import salvage_thinking, strip_thinking, wants_thinking_disabled
from jarvis.llm.voice_model import (
    VoiceModel,
    create_voice_model,
    strip_markup,
)


class FakeBackend(LLMBackend):
    """A backend whose reply — or explosion — is chosen by the test."""

    name = "fake"

    def __init__(
        self,
        reply: str = "",
        *,
        available: bool = True,
        raises: bool = False,
    ) -> None:
        self.reply = reply
        self.available = available
        self.raises = raises
        self.calls: list = []

    def is_available(self) -> bool:
        if self.raises:
            raise RuntimeError("the probe was told to fail")
        return self.available

    def load(self) -> None:
        return None

    def generate(self, messages, config: Optional[GenerationConfig] = None) -> LLMResult:
        self.calls.append(list(messages))
        if self.raises:
            raise RuntimeError("the backend was told to fail")
        return LLMResult(text=self.reply)

    def stream(self, messages, config=None):
        yield self.generate(messages, config).text


def voice(backend, **overrides) -> VoiceModel:
    cfg = LLMConfig(voice_model="qwen3:1.7b", **overrides)
    return VoiceModel(backend, cfg)


# --------------------------------------------------------------------------- #
#  The silent reply
# --------------------------------------------------------------------------- #
def test_an_unclosed_thinking_block_strips_to_nothing():
    """The precondition for the whole bug — pinned so it cannot drift."""
    raw = "<think>Let me work out what the user is asking for here"
    assert strip_thinking(raw) == ""


def test_reasoning_that_never_closed_is_salvaged_into_prose():
    """A truncated thought still contains the answer; speak it."""
    raw = (
        "<think>The user asked for the time. Looking at the clock, it is half "
        "past four in the afternoon. I should tell them that plainly. Now let "
        "me also consider whether they wanted"
    )
    salvaged = salvage_thinking(raw)

    assert salvaged, "an unclosed thought must not reduce to silence"
    assert "half past four" in salvaged
    # The dangling fragment must not be read aloud.
    assert not salvaged.endswith("wanted")
    assert "<think>" not in salvaged


def test_salvage_prefers_anything_said_outside_the_thinking_block():
    raw = "<think>internal deliberation here</think>It is half past four, Sir."
    assert salvage_thinking(raw) == "It is half past four, Sir."


def test_salvage_returns_empty_when_there_is_genuinely_nothing():
    assert salvage_thinking("") == ""
    assert salvage_thinking("<think></think>") == ""


def test_the_default_model_family_gets_thinking_disabled():
    """If this stops being true, every spoken reply goes silent again."""
    assert wants_thinking_disabled("Qwen/Qwen3.8-27B") is True
    assert wants_thinking_disabled("qwen3.8:27b") is True
    assert wants_thinking_disabled("qwen3:1.7b") is True
    assert wants_thinking_disabled("meta-llama/Llama-3.1-8B-Instruct") is False
    assert wants_thinking_disabled("") is False


def test_ollama_payload_disables_thinking_for_qwen():
    """The fix has to reach the wire, not just the helper."""
    from jarvis.llm.ollama_backend import OllamaBackend

    backend = OllamaBackend(LLMConfig(ollama_model="qwen3.8:27b"))
    payload = backend._payload([], GenerationConfig(), stream=False)
    assert payload["think"] is False


def test_ollama_leaves_thinking_alone_for_models_that_do_not_reason():
    from jarvis.llm.ollama_backend import OllamaBackend

    backend = OllamaBackend(LLMConfig(ollama_model="llama3.1:8b"))
    payload = backend._payload([], GenerationConfig(), stream=False)
    assert "think" not in payload


# --------------------------------------------------------------------------- #
#  Markup that must never be spoken
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("**important**", "important"),
        ("# A heading", "A heading"),
        ("- a bullet", "a bullet"),
        ("1. a numbered item", "a numbered item"),
        ("use `ls -la` here", "use ls -la here"),
        ("[the docs](https://example.com)", "the docs"),
        ("see https://example.com/x", "see a link"),
        ("a\nb\nc", "a b c"),
    ],
)
def test_screen_formatting_is_removed_before_speaking(raw, expected):
    assert strip_markup(raw) == expected


def test_code_fences_are_dropped_entirely():
    spoken = strip_markup("Here you go:\n```python\nprint(1)\n```\nThat is all.")
    assert "print" not in spoken
    assert "Here you go:" in spoken and "That is all." in spoken


# --------------------------------------------------------------------------- #
#  The voice model
# --------------------------------------------------------------------------- #
def test_it_speaks_the_phrasing_the_small_model_returned():
    model = voice(FakeBackend("It is half past four, Sir."))
    assert model.speakable("16:30") == "It is half past four, Sir."


def test_it_refuses_a_phrasing_that_invented_content():
    """The small model's failure mode is enthusiasm, not gibberish.

    Told to phrase one line it sometimes writes a paragraph of its own. That
    paragraph is plausible and wrong, which is the worst combination, so it is
    rejected in favour of the brain's actual answer.
    """
    invention = (
        "It is half past four, Sir. I have also taken the liberty of cancelling "
        "your four forty-five with Dr Hall, rescheduling the lab visit to "
        "Thursday, and ordering a replacement part for the workshop compressor, "
        "which should arrive by the weekend at the latest."
    )
    model = voice(FakeBackend(invention))
    assert model.speakable("It is half past four.") == "It is half past four."


def test_a_short_expansion_is_allowed():
    """"yes" -> "Yes, Sir." is a legitimate rephrasing, not an invention."""
    model = voice(FakeBackend("Yes, Sir."))
    assert model.speakable("yes") == "Yes, Sir."


def test_a_broken_voice_model_never_breaks_the_reply():
    model = voice(FakeBackend(raises=True))
    assert model.speakable("It is half past four.") == "It is half past four."


def test_an_empty_phrasing_falls_back_to_the_original():
    model = voice(FakeBackend(""))
    assert model.speakable("It is half past four.") == "It is half past four."


def test_markup_is_stripped_even_with_no_voice_model_available():
    model = voice(FakeBackend("", available=False))
    assert model.speakable("**half past four**") == "half past four"


def test_it_is_unavailable_when_disabled_or_unnamed():
    assert VoiceModel(FakeBackend("x"), LLMConfig(voice_model="")).is_available() is False
    cfg = LLMConfig(voice_model="qwen3:1.7b", voice_model_enabled=False)
    assert VoiceModel(FakeBackend("x"), cfg).is_available() is False


def test_availability_survives_a_backend_that_explodes_when_probed():
    assert voice(FakeBackend(raises=True)).is_available() is False


def test_the_acknowledgement_addresses_the_user_properly():
    model = VoiceModel(FakeBackend("x"), LLMConfig(), user_title="Sir")
    line = model.acknowledge(seed=0)
    assert line and "Sir" in line


def test_the_acknowledgement_can_be_switched_off():
    cfg = LLMConfig(voice_ack_enabled=False)
    assert VoiceModel(FakeBackend("x"), cfg).acknowledge() == ""


def test_the_phrasing_prompt_carries_the_answer_and_forbids_invention():
    backend = FakeBackend("Half past four, Sir.")
    voice(backend).speakable("16:30", user_input="what time is it")

    prompt = "\n".join(m.content for m in backend.calls[-1])
    assert "16:30" in prompt
    assert "what time is it" in prompt
    assert "no facts" in prompt.lower() or "add no facts" in prompt.lower()


# --------------------------------------------------------------------------- #
#  The factory
# --------------------------------------------------------------------------- #
def test_the_factory_declines_when_the_split_is_switched_off():
    assert create_voice_model(LLMConfig(voice_model_enabled=False)) is None
    assert create_voice_model(LLMConfig(voice_model="")) is None


def test_the_factory_forces_thinking_off_for_the_voice_model(monkeypatch):
    """A mouth that stops to reason defeats the entire point of it."""
    captured: list = []

    def fake_create_llm(cfg, **_kwargs):
        captured.append(cfg)
        return FakeBackend("spoken")

    monkeypatch.setattr("jarvis.llm.create_llm", fake_create_llm)

    model = create_voice_model(LLMConfig(voice_model="qwen3:1.7b"))

    assert model is not None
    assert captured, "the factory never built a backend"
    sub_cfg = captured[0]
    assert sub_cfg.thinking == "off"
    assert sub_cfg.model == "qwen3:1.7b"
    assert sub_cfg.ollama_model == "qwen3:1.7b"
    # It must not silently fall back to the 27B if the small model is missing.
    assert sub_cfg.allow_fallback is False


def test_the_factory_returns_none_when_the_small_model_is_absent(monkeypatch):
    monkeypatch.setattr(
        "jarvis.llm.create_llm", lambda cfg, **k: FakeBackend("", available=False)
    )
    assert create_voice_model(LLMConfig(voice_model="qwen3:1.7b")) is None


# --------------------------------------------------------------------------- #
#  End to end: the bug as the user experienced it
# --------------------------------------------------------------------------- #
class TruncatedThinker(LLMBackend):
    """Behaves exactly like Qwen3.8-27B with thinking left enabled.

    It spends its whole token budget reasoning and never closes the block --
    which is what a dense 27B does on a CPU with max_new_tokens at 512.
    """

    name = "truncated-thinker"

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def generate(self, messages, config=None) -> LLMResult:
        return LLMResult(
            text=(
                "<think>The user asked what time it is. Checking the clock, it "
                "is half past four in the afternoon. I should also consider "
                "whether they wanted the date, since"
            )
        )

    def stream(self, messages, config=None):
        yield self.generate(messages, config).text


def test_a_model_that_only_thinks_still_produces_a_spoken_reply(tmp_path):
    """The original bug: the assistant simply never answered.

    Nothing about the reply may be empty, and no reasoning markup may reach
    the speaker.
    """
    from jarvis.agent.orchestrator import Orchestrator
    from jarvis.core.config import Config
    from jarvis.memory import create_context, create_memory

    cfg = Config()
    cfg.tts.enabled = False
    cfg.memory.db_path = str(tmp_path / "memory.db")
    store = create_memory(cfg.memory)
    context = create_context(cfg.memory, store=store)

    agent = Orchestrator(cfg, TruncatedThinker(), None, context)
    reply = agent.chat("what time is it", speak=False)

    assert reply.strip(), "a voice assistant must never answer with silence"
    assert "<think>" not in reply and "</think>" not in reply
    assert "half past four" in reply


def test_raw_thinking_markup_never_reaches_the_speaker(tmp_path):
    """The agent loop is the last thing before the speaker, so it must strip
    reasoning itself rather than trusting every backend to have done it."""
    from jarvis.agent.subagent import run_agent_loop
    from jarvis.core.contracts import Message

    class LeakyBackend(TruncatedThinker):
        def generate(self, messages, config=None) -> LLMResult:
            return LLMResult(text="<think>deliberating</think>It is half past four.")

    turn = run_agent_loop(
        LeakyBackend(), None, [Message.user("what time is it")], max_iterations=2
    )
    assert turn.text == "It is half past four."


def test_the_factory_refuses_a_stub_backend(monkeypatch):
    """`create_llm` falls back to a stub when nothing real is reachable.

    A stub mouth would speak canned text over the top of the real answer,
    which is worse than having no voice model at all.
    """
    class Stub(FakeBackend):
        name = "stub"

    monkeypatch.setattr("jarvis.llm.create_llm", lambda cfg, **k: Stub("canned"))
    assert create_voice_model(LLMConfig(voice_model="qwen3:1.7b")) is None


# --------------------------------------------------------------------------- #
#  CPU tuning
# --------------------------------------------------------------------------- #
class TestCpuTuning:
    """Dense CPU inference is memory-bandwidth bound, not compute bound.

    Hyperthreaded siblings share one memory port, so running 8 threads on 4
    physical cores has them contending for the exact resource that is already
    the bottleneck. Ollama's default is all logical processors; physical is
    measurably better.
    """

    def test_thread_count_prefers_physical_cores(self, monkeypatch, tmp_path):
        from jarvis.llm import ollama_backend

        # An i5-10210U: 8 logical processors across 4 physical cores.
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "".join(
                f"processor\t: {i}\ncore id\t\t: {i % 4}\n\n" for i in range(8)
            )
        )
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/cpuinfo":
                return real_open(cpuinfo, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert ollama_backend._default_thread_count() == 4

    def test_it_halves_logical_cores_when_topology_is_unreadable(self, monkeypatch):
        from jarvis.llm import ollama_backend

        def boom(path, *args, **kwargs):
            raise OSError("no /proc here")

        monkeypatch.setattr("builtins.open", boom)
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert ollama_backend._default_thread_count() == 4

    def test_tuning_reaches_the_wire(self):
        from jarvis.llm.ollama_backend import OllamaBackend

        backend = OllamaBackend(LLMConfig(num_threads=4, use_mmap=True))
        options = backend._build_options(GenerationConfig())
        assert options["num_thread"] == 4
        assert options["use_mmap"] is True
        # mlock is off unless asked for: on a tight machine it causes swapping.
        assert "use_mlock" not in options

    def test_mlock_is_opt_in(self):
        from jarvis.llm.ollama_backend import OllamaBackend

        backend = OllamaBackend(LLMConfig(use_mlock=True))
        assert backend._build_options(GenerationConfig())["use_mlock"] is True

    def test_an_explicit_thread_count_wins(self):
        from jarvis.llm.ollama_backend import OllamaBackend

        backend = OllamaBackend(LLMConfig(num_threads=2))
        assert backend._build_options(GenerationConfig())["num_thread"] == 2
