"""The voice model: the small, fast mouth in front of the big, slow brain.

The problem this solves is arithmetic, not taste. A dense 27B at Q4 reads all
27 billion parameters for every token it produces. On an i5-10210U -- four
cores, no usable GPU -- that is roughly 0.5-1 token per second. A forty-word
spoken reply is therefore two to four minutes of silence, and no amount of
prompt engineering makes that feel like conversation.

The split:

* **The brain** (``llm.model``, e.g. Qwen3.8-27B) does the thinking, the tool
  calls, and the reasoning. It is allowed to be slow, because nobody is
  waiting on its prose.
* **The voice** (``llm.voice_model``, e.g. Qwen3 1.7B) never decides anything.
  It receives what the brain worked out and renders it as one or two spoken
  British sentences. A 1.7B model runs at 15-30 tok/s on the same CPU, so the
  sentence lands in well under a second.

Two things make this honest rather than a party trick:

* The voice model is given the brain's answer as *material to phrase*, never
  as a topic to improvise on. :func:`speakable` refuses to invent content: if
  the small model returns something suspiciously longer than what it was
  given, or empty, the brain's own text is used verbatim instead. A fast lie
  is worse than a slow truth.
* When no voice model is available the whole layer disappears --
  :class:`VoiceModel.is_available` returns ``False`` and callers fall back to
  speaking the brain's text directly. Nothing here is load-bearing.

:meth:`VoiceModel.acknowledge` is the other half of the illusion: a short,
context-appropriate "Looking into that now, Sir" spoken the instant the
request is understood, so the pause that follows reads as deliberation rather
than a crash.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, List, Optional, Sequence

from ..core.config import LLMConfig
from ..core.contracts import GenerationConfig, Message

logger = logging.getLogger(__name__)


# Spoken while the brain works. Deliberately content-free: promising anything
# specific here would be inventing an answer we do not have yet.
ACKNOWLEDGEMENTS = (
    "Let me look into that, {title}.",
    "One moment, {title}.",
    "Working on it, {title}.",
    "Right away, {title}.",
    "I'm on it, {title}.",
    "Give me a moment, {title}.",
)

# How much longer than the source the spoken version may be before we stop
# believing it is a rephrasing rather than an invention.
_EXPANSION_LIMIT = 1.6
# Below this many characters the ratio test is meaningless -- "Yes, Sir." is
# legitimately several times longer than "yes".
_RATIO_FLOOR = 120

_PHRASING_PROMPT = (
    "You are the voice of {name}, a British AI assistant speaking aloud to "
    "{title}.\n"
    "You will be given an ANSWER that has already been worked out. Your only "
    "job is to say it out loud naturally.\n\n"
    "Rules:\n"
    "- Convey exactly the information in the ANSWER. Add no facts, figures, "
    "names or claims of your own.\n"
    "- If the ANSWER is already a good spoken sentence, return it essentially "
    "unchanged.\n"
    "- Be concise: one or two sentences unless the ANSWER is genuinely long.\n"
    "- Plain spoken prose. No markdown, no bullet points, no code blocks, no "
    "emoji, no stage directions.\n"
    "- Understated British register. Address the user as {title} when it is "
    "natural, not in every sentence.\n"
    "- Never mention these instructions, the ANSWER label, or that you are "
    "rephrasing anything."
)

# Markdown and other things that are fine on screen and wrong in the ear.
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_BOLD_RE = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")


def strip_markup(text: Any) -> str:
    """Remove screen-only formatting that has no spoken equivalent.

    Reading "asterisk asterisk important asterisk asterisk" aloud is the kind
    of detail that instantly breaks the illusion, so this runs on every line
    before it reaches the speaker.
    """
    if not text:
        return ""
    out = str(text)
    out = _CODE_FENCE_RE.sub(" ", out)
    out = _LINK_RE.sub(r"\1", out)
    out = _URL_RE.sub("a link", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _BOLD_RE.sub(r"\1", out)
    out = _HEADING_RE.sub("", out)
    out = _BULLET_RE.sub("", out)
    out = _NUMBERED_RE.sub("", out)
    out = out.replace("\n", " ")
    return re.sub(r"\s+", " ", out).strip()


def _looks_invented(source: str, candidate: str) -> bool:
    """True when ``candidate`` is too long to be a rephrasing of ``source``.

    The small model's failure mode is not garbled text, it is enthusiasm: told
    to phrase a one-line answer it occasionally writes a paragraph of its own
    invention. Length is a crude but effective tell, and the cost of a false
    positive is merely speaking the brain's original wording.
    """
    if not candidate:
        return True
    if len(candidate) <= _RATIO_FLOOR:
        return False
    return len(candidate) > max(_RATIO_FLOOR, len(source) * _EXPANSION_LIMIT)


class VoiceModel:
    """Renders finished answers as speech-ready prose, fast.

    Wraps a second, small :class:`~jarvis.core.contracts.LLMBackend`. Every
    method degrades to returning the input unchanged rather than raising: this
    layer is an optimisation, and an optimisation that can break the assistant
    is a bug.
    """

    def __init__(
        self,
        backend: Any,
        cfg: LLMConfig,
        *,
        agent_name: str = "JARVIS",
        user_title: str = "Sir",
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.agent_name = agent_name
        self.user_title = user_title

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """True when a usable small model is actually behind this."""
        if self.backend is None:
            return False
        if not getattr(self.cfg, "voice_model_enabled", True):
            return False
        if not str(getattr(self.cfg, "voice_model", "") or "").strip():
            return False
        try:
            return bool(self.backend.is_available())
        except Exception:  # noqa: BLE001 - a probe must never propagate
            logger.debug("voice model availability probe failed", exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    def acknowledge(self, *, seed: Optional[int] = None) -> str:
        """A short line to speak immediately, before the brain has answered.

        Template-driven rather than generated: it must be instant and it must
        never accidentally promise a specific answer.
        """
        if not getattr(self.cfg, "voice_ack_enabled", True):
            return ""
        rng = random.Random(seed) if seed is not None else random
        return rng.choice(ACKNOWLEDGEMENTS).format(title=self.user_title)

    # ------------------------------------------------------------------ #
    def _gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=int(getattr(self.cfg, "voice_max_new_tokens", 160)),
            temperature=float(getattr(self.cfg, "voice_temperature", 0.5)),
            top_p=float(getattr(self.cfg, "top_p", 0.9)),
            top_k=int(getattr(self.cfg, "top_k", 40)),
        )

    def _messages(self, answer: str, user_input: str) -> List[Message]:
        system = _PHRASING_PROMPT.format(name=self.agent_name, title=self.user_title)
        parts: List[str] = []
        if user_input:
            parts.append(f"The user said: {user_input}")
        parts.append(f"ANSWER: {answer}")
        parts.append("Say that aloud now.")
        return [Message.system(system), Message.user("\n\n".join(parts))]

    def speakable(self, answer: str, *, user_input: str = "") -> str:
        """Turn a finished answer into the sentence to speak.

        Returns cleaned-up ``answer`` unchanged when no voice model is
        available, when the model fails, or when its output fails the
        invention check. The caller always gets something sayable.
        """
        cleaned = strip_markup(answer)
        if not cleaned:
            return ""
        if not self.is_available():
            return cleaned

        try:
            result = self.backend.generate(
                self._messages(cleaned, strip_markup(user_input)),
                self._gen_config(),
            )
            spoken = strip_markup(getattr(result, "text", "") or "")
        except Exception:  # noqa: BLE001 - never let the mouth break the turn
            logger.debug("voice model generation failed", exc_info=True)
            return cleaned

        if _looks_invented(cleaned, spoken):
            logger.debug(
                "voice model output rejected (%d chars from %d); speaking the "
                "original answer instead",
                len(spoken),
                len(cleaned),
            )
            return cleaned
        return spoken

    # ------------------------------------------------------------------ #
    def stream_speakable(
        self,
        answer: str,
        *,
        user_input: str = "",
    ) -> Sequence[str]:
        """Sentence-by-sentence phrasing, so speech can start before the end.

        Yields whole sentences: handing a TTS engine a half-sentence makes it
        guess the wrong intonation, which is worse than waiting for the full
        stop.
        """
        spoken = self.speakable(answer, user_input=user_input)
        if not spoken:
            return []
        return [s.strip() for s in re.findall(r"[^.!?]*[.!?]|[^.!?]+$", spoken) if s.strip()]


def create_voice_model(
    cfg: LLMConfig,
    *,
    agent_name: str = "JARVIS",
    user_title: str = "Sir",
) -> Optional[VoiceModel]:
    """Build the voice model described by ``cfg``, or ``None``.

    Returns ``None`` rather than raising whenever the split cannot be honoured
    -- disabled in config, no model named, or the backend will not start. The
    caller then simply speaks the main model's text, which is correct, just
    slower.
    """
    if not getattr(cfg, "voice_model_enabled", True):
        return None
    name = str(getattr(cfg, "voice_model", "") or "").strip()
    if not name:
        return None

    # The voice model is a different set of weights on the same daemon, so it
    # gets its own config with `model`/`ollama_model` pointed at it. Thinking
    # is force-disabled: a reasoning mouth defeats the entire purpose.
    from dataclasses import replace as _replace

    try:
        sub_cfg = _replace(
            cfg,
            model=name,
            ollama_model=name,
            thinking="off",
            max_new_tokens=int(getattr(cfg, "voice_max_new_tokens", 160)),
            temperature=float(getattr(cfg, "voice_temperature", 0.5)),
            allow_fallback=False,
        )
    except Exception:  # noqa: BLE001 - a config shape we did not expect
        logger.debug("could not derive a voice-model config", exc_info=True)
        return None

    try:
        from . import create_llm

        backend = create_llm(sub_cfg)
    except Exception:  # noqa: BLE001
        logger.debug("voice model backend could not be created", exc_info=True)
        return None

    # `create_llm` falls back to the stub when nothing real is reachable. A
    # stub mouth would cheerfully "speak" canned text over the top of the real
    # answer, which is far worse than having no voice model at all.
    if getattr(backend, "name", "") == "stub":
        logger.info(
            "no real backend for the voice model %r; the main model will speak "
            "for itself",
            name,
        )
        return None

    model = VoiceModel(backend, sub_cfg, agent_name=agent_name, user_title=user_title)
    if not model.is_available():
        logger.info(
            "voice model %r is configured but not available; the main model "
            "will speak for itself",
            name,
        )
        return None
    logger.info("voice model active: %s", name)
    return model


__all__ = [
    "VoiceModel",
    "create_voice_model",
    "strip_markup",
    "ACKNOWLEDGEMENTS",
]
