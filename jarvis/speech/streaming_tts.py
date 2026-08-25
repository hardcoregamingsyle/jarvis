"""Live, sentence-pipelined text-to-speech.

Every engine in :mod:`speech.tts` is whole-utterance: ``synthesize()`` builds
one complete WAV, ``speak()`` plays it, and nothing starts sounding until the
full reply text exists. For a router model whose entire point is to feel
instant, that is the wrong shape — waiting for the full reply throws away the
model's own token stream.

:class:`StreamingSpeaker` sits between an LLM's token stream and a
:class:`~jarvis.core.contracts.TTSEngine`. It buffers incoming deltas, cuts
off each complete sentence as soon as it can, and runs a two-stage pipeline —
one thread synthesizing, one thread playing — so sentence *N+1* is being
turned into audio while sentence *N* is still sounding. The engine's own
synthesis speed no longer gates anything but the very first sentence.

Two kinds of markup a model streams inline with ordinary prose must never
reach the speaker: a JARVIS tool call,
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (see
:mod:`jarvis.agent.protocol`), and Qwen3-style reasoning,
``<think>...</think>`` (see :func:`jarvis.llm.base.strip_thinking` — a
thinking model that runs out of budget mid-block is exactly why that module
also has ``salvage_thinking``: the failure mode is real, not hypothetical, so
a live listener must be just as deaf to an unclosed ``<think>`` as a
non-streaming caller is). :func:`segment_sentences` and the tag-aware loop in
:meth:`StreamingSpeaker.feed` keep prose apart from both, using the exact same
tag strings ``protocol.py``/``base.py`` parse against, so the three can never
drift apart.

Public surface intentionally matches :class:`~jarvis.speech.tts.SpeechQueue`
(``say``, ``wait``, ``stop``, ``is_speaking``) so a :class:`StreamingSpeaker`
is a drop-in replacement anywhere a ``SpeechQueue`` is used today — including
``Orchestrator.tts`` and the barge-in wiring in ``voice.py``, none of which
needs to change.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from typing import Any, Optional

from ..agent.protocol import TOOL_CALL_CLOSE, TOOL_CALL_OPEN
from .audio_io import AudioPlayer

log = logging.getLogger(__name__)

# Split after sentence-ending punctuation or a newline, keeping the
# terminator with the sentence that precedes it.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ENDS_CLEAN_RE = re.compile(r"[.!?]\s*$")

# No terminator for this long -> speak on a whitespace boundary anyway, so a
# model that rambles without punctuation still gets heard incrementally
# rather than staying silent until the whole reply is done.
DEFAULT_MAX_BUFFER = 220
# Below this, a "complete" sentence is still held back one more feed() in case
# more text is seconds away — avoids firing a synth+play round trip for "Ah."
# when "Ah, quite so." was one token away. Ignored once force=True (finish()).
DEFAULT_MIN_CHARS = 12

# Tag pairs whose content must never reach the speaker. Matched case-
# sensitively: both are documented, single-form tokens (``base.py``'s own
# regexes use IGNORECASE for defensive robustness, but real backends emit the
# lowercase form consistently) and a hot per-chunk loop is not the place to
# add regex overhead for a case a real model does not produce. THINK_OPEN
# checked before TOOL_CALL_OPEN: reasoning always precedes any tool call.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_OPAQUE_TAGS = ((THINK_OPEN, THINK_CLOSE), (TOOL_CALL_OPEN, TOOL_CALL_CLOSE))

_SENTINEL = object()


def _tag_overlap(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``tag``.

    Used to hold back text that might be the leading edge of a tag still
    arriving token-by-token (e.g. buffer ends in ``"<tool_c"``) so it is never
    mistaken for speakable prose.
    """
    n = min(len(buf), len(tag) - 1)
    for k in range(n, 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


def _max_tag_overlap(buf: str, tags) -> int:
    """The largest hold-back needed across every opaque tag's OPEN string."""
    return max((_tag_overlap(buf, tag) for tag in tags), default=0)


def segment_sentences(
    buf: str, *, force: bool, max_buffer: int = DEFAULT_MAX_BUFFER,
) -> tuple:
    """Split complete sentences off the front of ``buf``.

    Returns ``(units, remainder)``. A returned unit is never one that could
    still grow into a longer sentence with more text — unless ``force`` is
    set, which flushes everything as-is (used at the end of a reply, or right
    before an opaque tag where the boundary is already certain).
    """
    if force:
        text = buf.strip()
        return ([text] if text else []), ""

    # Hold back a suffix that might be the start of a still-arriving tag —
    # whichever of <think> / <tool_call> would need the most characters held.
    safe_len = len(buf) - _max_tag_overlap(buf, (o for o, _ in _OPAQUE_TAGS))
    working, held = buf[:safe_len], buf[safe_len:]

    parts = _SENT_SPLIT_RE.split(working)
    if not parts or not working:
        return [], buf

    ends_clean = bool(_ENDS_CLEAN_RE.search(working)) or working.endswith("\n")
    complete_parts = parts if ends_clean else parts[:-1]
    remainder_parts = [] if ends_clean else parts[-1:]

    units = [p.strip() for p in complete_parts if p.strip()]
    remainder = (" ".join(remainder_parts) + held).strip()

    if not units and len(remainder) > max_buffer:
        cut = remainder.rfind(" ", 0, max_buffer)
        if cut <= 0:
            cut = max_buffer
        units.append(remainder[:cut].strip())
        remainder = remainder[cut:].strip()

    return units, remainder


class StreamingSpeaker:
    """Pipelined, tool-call-aware, sentence-level streaming TTS.

    Two dedicated daemon threads: one synthesizes queued text units to WAV
    bytes as fast as the engine allows, the other plays finished WAV buffers
    in order. Bounding the audio queue (``max_ahead``) stops the synth thread
    racing arbitrarily far ahead of playback — pointless work if a barge-in
    is going to discard it a moment later anyway.
    """

    def __init__(
        self,
        engine: Any,
        *,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        max_ahead: int = 3,
        player: Optional[AudioPlayer] = None,
    ) -> None:
        self.engine = engine
        self.name = f"streaming({getattr(engine, 'name', 'tts')})"
        self._min_chars = max(0, int(min_chars))
        self._max_buffer = max(40, int(max_buffer))
        self._player = player or AudioPlayer()

        self._text_q: "queue.Queue" = queue.Queue()
        self._audio_q: "queue.Queue" = queue.Queue(maxsize=max(1, int(max_ahead)))

        self._buf = ""
        self._held_short = ""      # a short complete sentence held for merging
        self._in_tag: Optional[str] = None   # the CLOSE string currently awaited, if any
        self._buf_lock = threading.Lock()

        # Guards both `_generation` (bumped on stop() so stale queued items
        # from an interrupted utterance are dropped rather than spoken late)
        # and `_pending` (how many units are still somewhere in the pipeline).
        #
        # `_pending` exists because "both queues are empty" is NOT the same
        # as "nothing is in flight": a unit spends time popped off text_q but
        # not yet pushed to audio_q (mid-synthesis), during which BOTH queues
        # can read empty even though a third sentence is still on its way.
        # Inferring "drained" from that snapshot is a real race — verified by
        # a timing-based test catching it — so completion is tracked
        # explicitly instead: +1 on every enqueue, -1 only once a unit is
        # truly done (played, or failed/discarded before it could be).
        self._state_lock = threading.Lock()
        self._generation = 0
        self._pending = 0

        self._drained = threading.Event()
        self._drained.set()

        self._closed = False
        self._synth_thread = threading.Thread(
            target=self._synth_loop, name="jarvis-tts-synth", daemon=True,
        )
        self._play_thread = threading.Thread(
            target=self._play_loop, name="jarvis-tts-play", daemon=True,
        )
        self._synth_thread.start()
        self._play_thread.start()

    # ------------------------------------------------------------------ #
    #  Feeding text in
    # ------------------------------------------------------------------ #
    def _gen(self) -> int:
        with self._state_lock:
            return self._generation

    def _enqueue(self, unit: str) -> None:
        if not unit:
            return
        with self._state_lock:
            self._pending += 1
            gen = self._generation
        self._drained.clear()
        self._text_q.put((gen, unit))

    def _mark_done(self, gen: int) -> None:
        """One unit has left the pipeline for good: played, or discarded
        before it could be (synthesis failure, or gone stale mid-flight)."""
        with self._state_lock:
            if gen != self._generation:
                return   # a stop() already zeroed the count for this generation
            self._pending -= 1
            settled = self._pending <= 0
            if settled:
                self._pending = 0
        if settled:
            self._drained.set()

    def feed(self, chunk: str) -> None:
        """Feed one raw text delta from an LLM's token stream.

        Safe to call from the same thread driving ``llm.stream()`` — this
        never blocks on synthesis or playback, only on the (tiny, in-memory)
        segmentation work, so it cannot stall generation.
        """
        if not chunk:
            return
        with self._buf_lock:
            self._buf += chunk
            while True:
                if self._in_tag is None:
                    # Whichever opaque tag (<think>, <tool_call>) opens
                    # soonest in the buffer, if either has arrived complete.
                    opens = [
                        (i, close) for open_, close in _OPAQUE_TAGS
                        if (i := self._buf.find(open_)) != -1
                    ]
                    if opens:
                        idx, close = min(opens, key=lambda p: p[0])
                        prefix = self._buf[:idx]
                        units, _ = segment_sentences(prefix, force=True)
                        for u in units:
                            self._flush_unit(u)
                        self._buf = self._buf[idx:]
                        self._in_tag = close
                        continue
                    units, self._buf = segment_sentences(
                        self._buf, force=False, max_buffer=self._max_buffer,
                    )
                    if not units:
                        break
                    for u in units:
                        self._flush_unit(u)
                else:
                    idx = self._buf.find(self._in_tag)
                    if idx == -1:
                        break
                    self._buf = self._buf[idx + len(self._in_tag):]
                    self._in_tag = None
                    continue

    def _flush_unit(self, unit: str) -> None:
        """Speak ``unit``, merging a too-short one into the next rather than
        firing a synth+play round trip for a one-word sentence."""
        if self._held_short:
            unit = f"{self._held_short} {unit}"
            self._held_short = ""
        if len(unit) < self._min_chars:
            self._held_short = unit
            return
        self._enqueue(unit)

    def finish(self) -> None:
        """No more text is coming for this utterance: flush whatever remains."""
        with self._buf_lock:
            if self._in_tag is not None:
                # An unterminated <think> or <tool_call> at end of stream:
                # never spoken, same as a truncated one strip_thinking would
                # also discard.
                self._buf = ""
                self._in_tag = None
            else:
                units, self._buf = segment_sentences(self._buf, force=True)
                for u in units:
                    if self._held_short:
                        u = f"{self._held_short} {u}"
                        self._held_short = ""
                    self._enqueue(u)
            if self._held_short:
                self._enqueue(self._held_short)
                self._held_short = ""

    def say(self, text: str) -> None:
        """Speak a complete string non-streamed — drop-in for ``SpeechQueue.say``."""
        if not text:
            return
        self.feed(text)
        self.finish()

    # ------------------------------------------------------------------ #
    #  Pipeline
    # ------------------------------------------------------------------ #
    def _synth_loop(self) -> None:
        while not self._closed:
            try:
                gen, unit = self._text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen():
                continue        # stale: stop() already released its pending slot
            try:
                audio = self.engine.synthesize(unit)
            except Exception:  # noqa: BLE001 - one bad sentence must not end the voice
                log.exception("synthesis failed for %r", unit[:80])
                audio = None
            if audio and gen == self._gen():
                self._audio_q.put((gen, audio))
            else:
                # Failed, or went stale while synthesizing: this unit will
                # never reach playback, so release its pending slot here —
                # otherwise wait()/is_speaking would hang waiting for a unit
                # that is never coming.
                self._mark_done(gen)

    def _play_loop(self) -> None:
        while not self._closed:
            try:
                gen, audio = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen():
                continue        # stale: stop() already released its pending slot
            try:
                self._player.play_wav(audio)
            except Exception:  # noqa: BLE001
                log.exception("playback failed")
            self._mark_done(gen)

    # ------------------------------------------------------------------ #
    #  SpeechQueue-compatible surface
    # ------------------------------------------------------------------ #
    @property
    def is_speaking(self) -> bool:
        with self._state_lock:
            return self._pending > 0

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until fully drained. Returns True if it drained in time."""
        return self._drained.wait(timeout)

    def stop(self) -> None:
        """Barge-in: discard everything queued and cut current playback.

        Bumps the generation counter first so anything the synth/play threads
        are mid-processing (or about to pull) is recognised as stale and
        dropped rather than spoken after the interruption, and zeroes the
        pending count in the same step so ``wait()`` does not hang waiting
        for units that will now never complete.
        """
        with self._state_lock:
            self._generation += 1
            self._pending = 0
        with self._buf_lock:
            self._buf = ""
            self._held_short = ""
            self._in_tag = None
        for q in (self._text_q, self._audio_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        try:
            self._player.stop()
        except Exception:  # noqa: BLE001
            log.debug("player.stop() failed", exc_info=True)
        self._drained.set()

    def shutdown(self) -> None:
        self._closed = True
        self.stop()


__all__ = ["StreamingSpeaker", "segment_sentences", "DEFAULT_MIN_CHARS", "DEFAULT_MAX_BUFFER"]
