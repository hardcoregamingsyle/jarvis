"""The hands-free conversation loop.

Flow per cycle:
    record until silence -> transcribe -> apply the addressing policy ->
    hand the utterance to the orchestrator -> speak the reply ->
    announce any background task that finished while we were talking.

Three modes, chosen by ``voice.mode``:

``wake``
    The default.  An utterance is only acted on when it opens with a wake
    word, except inside the short follow-up window that each reply opens.
``continuous``
    An open conversation: no wake word for as long as somebody keeps talking.
    After ``continuous_timeout`` of silence it reverts to requiring the wake
    word, so a room left alone does not answer the television.
``push``
    Push-to-talk.  Nothing is recorded until :meth:`begin_utterance` is
    called and the recording ends at :meth:`end_utterance`.  The hotkey
    itself is bound elsewhere; this class only exposes the two calls.

In every mode, **barge-in**: while JARVIS is speaking, the recorder's level
monitor keeps watching the microphone, and the moment the user starts talking
playback is cut and the rest of the reply is dropped.  It is gated on
``voice.allow_interrupt`` and on a loudness margin over the measured ambient
floor, because the one thing barge-in must never do is interrupt JARVIS
because it heard JARVIS.

Wake-word detection is done on the transcript rather than with a dedicated
keyword-spotter.  On a CPU-only laptop a small Whisper model is cheap enough to
run on every utterance, and this avoids a second model, a second set of weights,
and an extra dependency.

Configuration read from ``config.voice`` (all optional — every one is read with
a default so an older config still works):

===========================  =========  ==========================================
name                         default    meaning
===========================  =========  ==========================================
``mode``                     ``wake``   ``wake`` | ``continuous`` | ``push``
``follow_up_seconds``        ``15.0``   no wake word needed for this long after a reply
``continuous_timeout``       ``120.0``  inactivity after which ``continuous`` re-arms the wake word
``interrupt_margin``         ``2.5``    multiple of the ambient floor that counts as barge-in
``preroll_seconds``          ``0.5``    audio kept from *before* speech was detected
``min_speech_seconds``       ``0.3``    shorter bursts are coughs and clicks, not speech
``acknowledge``              ``""``     reply to the bare wake word; empty means "Yes, <title>?"
===========================  =========  ==========================================
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Optional

from .core.config import Config
from .core.events import EventBus, Events, get_bus

log = logging.getLogger(__name__)


# The bus channel on which every state transition is published.  Deliberately
# a module constant rather than an entry in Events: the state machine belongs
# to the voice loop, and core.events stays a shared vocabulary.
VOICE_STATE = "voice.state"

STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"

MODE_WAKE = "wake"
MODE_CONTINUOUS = "continuous"
MODE_PUSH = "push"
_MODES = frozenset({MODE_WAKE, MODE_CONTINUOUS, MODE_PUSH})

_DEFAULT_FOLLOW_UP = 15.0
_DEFAULT_CONTINUOUS_TIMEOUT = 120.0
_DEFAULT_INTERRUPT_MARGIN = 2.5
_DEFAULT_PREROLL = 0.5
_DEFAULT_MIN_SPEECH = 0.3

# How often the push-to-talk gate wakes to look at the world.
_PTT_POLL = 0.05

# Longest we will ever sit waiting for the speech layer to drain, so a wedged
# TTS engine costs one reply rather than the whole conversation.
_SPEECH_WAIT_CAP = 120.0


# Words a speaker may put in front of the wake word without meaning anything by
# them.  Anything else before the name means JARVIS was mentioned, not addressed.
_FILLERS = frozenset({
    "um", "uh", "erm", "er", "ah", "oh", "hey", "hi", "hello", "ok", "okay",
    "so", "well", "right", "now", "please", "excuse", "me", "sorry",
})


def strip_wake_word(text: str, wake_words: list) -> tuple:
    """Detect a wake word and return the remainder of the utterance.

    Returns ``(matched, remainder)``.  When the utterance is *only* the wake
    word, the remainder is empty — the caller should then acknowledge and listen
    again, which is how "Jarvis?" ... "Sir?" works.

    The wake word must open the utterance, optionally preceded by filler
    ("um, jarvis, ..."). Merely *mentioning* the name — "tell me about the
    jarvis project" — must not trigger, so an arbitrary mid-sentence match is
    deliberately rejected.
    """
    if not text:
        return False, ""

    stripped = text.strip()
    lowered = stripped.lower()

    # Longest first, so "hey jarvis" wins over the bare "jarvis".
    for word in sorted(wake_words, key=len, reverse=True):
        word = word.strip().lower()
        if not word:
            continue

        pattern = re.compile(r"\b" + re.escape(word) + r"\b[\s,.:;!?-]*")
        for match in pattern.finditer(lowered):
            prefix = lowered[: match.start()]
            tokens = re.findall(r"[a-z']+", prefix)
            if all(token in _FILLERS for token in tokens):
                return True, stripped[match.end():].strip()
            # A non-filler word came first: this is a mention, not an address.
            break

    return False, stripped


class VoiceLoop:
    """Drives microphone -> agent -> speaker until stopped."""

    def __init__(
        self,
        config: Config,
        orchestrator: Any,
        stt: Any,
        recorder: Any = None,
        *,
        bus: Optional[EventBus] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_reply: Optional[Callable[[str], None]] = None,
        player: Any = None,
        calibrate: bool = True,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.stt = stt
        self.bus = bus or get_bus()
        self.on_transcript = on_transcript
        self.on_reply = on_reply

        self._recorder = recorder
        self._player = player
        self._calibrate_on_start = bool(calibrate)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._awake_until = 0.0
        self._continuous_until = 0.0
        self._last_announce = 0.0

        self._state = STATE_IDLE
        self._state_lock = threading.RLock()

        # Push-to-talk: set between begin_utterance() and end_utterance().
        self._ptt = threading.Event()

        self._interrupted = False
        self._noise_floor = float(getattr(config.stt, "silence_threshold", 0.015))
        self._monitor_armed = False

    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        """The addressing mode; anything unrecognised falls back to ``wake``."""
        raw = getattr(self.config.voice, "mode", MODE_WAKE)
        try:
            name = str(raw or MODE_WAKE).strip().lower()
        except Exception:  # noqa: BLE001
            return MODE_WAKE
        return name if name in _MODES else MODE_WAKE

    def _num(self, name: str, default: float) -> float:
        """A numeric voice setting, tolerant of a config that predates it."""
        try:
            return float(getattr(self.config.voice, name, default))
        except (TypeError, ValueError):
            return default

    @property
    def noise_floor(self) -> float:
        """Ambient level used as the reference for barge-in detection."""
        return self._noise_floor

    @property
    def state(self) -> str:
        """One of ``idle``/``listening``/``thinking``/``speaking``."""
        return self._state

    @property
    def interrupted(self) -> bool:
        """True when the last spoken reply was cut short by the user."""
        return self._interrupted

    # ------------------------------------------------------------------ #
    def _ensure_recorder(self) -> Any:
        if self._recorder is None:
            from .speech.audio_io import AudioRecorder

            self._recorder = AudioRecorder(
                self.config.stt,
                preroll_seconds=self._num("preroll_seconds", _DEFAULT_PREROLL),
                min_speech_seconds=self._num("min_speech_seconds", _DEFAULT_MIN_SPEECH),
            )
        return self._recorder

    def available(self) -> bool:
        """True when we can actually capture and transcribe audio."""
        try:
            recorder = self._ensure_recorder()
        except Exception:  # noqa: BLE001
            log.exception("no audio recorder")
            return False
        if not getattr(recorder, "is_available", lambda: False)():
            return False
        return bool(self.stt is not None and getattr(self.stt, "is_available", lambda: False)())

    # ------------------------------------------------------------------ #
    #  State
    # ------------------------------------------------------------------ #
    def _set_state(self, new_state: str) -> None:
        """Move to ``new_state`` and publish it.  A no-op if unchanged."""
        with self._state_lock:
            previous = self._state
            if previous == new_state:
                return
            self._state = new_state
        # The microphone closes before the new state opens, so a listener sees
        # "stopped listening" ahead of "now thinking" rather than after it.
        if previous == STATE_LISTENING:
            self.bus.emit(Events.LISTEN_STOP, None)
        self.bus.emit(VOICE_STATE, new_state)
        if new_state == STATE_LISTENING:
            self.bus.emit(Events.LISTEN_START, None)

    # ------------------------------------------------------------------ #
    #  Push-to-talk
    # ------------------------------------------------------------------ #
    def begin_utterance(self) -> None:
        """Open the microphone for push-to-talk.  Safe to call in any mode."""
        self._ptt.set()

    def end_utterance(self) -> None:
        """Close the push-to-talk microphone and cut the recording short."""
        self._ptt.clear()
        recorder = self._recorder
        stop = getattr(recorder, "stop", None) if recorder is not None else None
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                log.debug("recorder.stop() failed", exc_info=True)

    def _await_turn(self) -> bool:
        """Block until it is our turn to record.  False means "loop again"."""
        if self.mode != MODE_PUSH:
            return True
        self._set_state(STATE_IDLE)
        while not self._stop.is_set():
            if self._ptt.wait(_PTT_POLL):
                return True
            self._announce_periodically()
        return False

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Blocking loop.  Returns when :meth:`stop` is called."""
        cfg = self.config
        recorder = self._ensure_recorder()
        self._stop.clear()
        self._calibrate(recorder)

        log.info(
            "voice loop active (mode=%s, wake word: %s, barge-in: %s)",
            self.mode,
            "required" if cfg.voice.require_wake_word else "not required",
            "on" if getattr(cfg.voice, "allow_interrupt", True) else "off",
        )
        if self.mode == MODE_CONTINUOUS:
            self._continuous_until = time.monotonic() + self._num(
                "continuous_timeout", _DEFAULT_CONTINUOUS_TIMEOUT
            )

        while not self._stop.is_set():
            if not self._await_turn():
                break

            self._set_state(STATE_LISTENING)
            try:
                samples = recorder.record_until_silence(
                    max_seconds=cfg.stt.max_utterance_seconds
                )
            except Exception:  # noqa: BLE001 - a device hiccup must not end the loop
                log.exception("recording failed; retrying shortly")
                self._stop.wait(1.0)
                continue

            if self._stop.is_set():
                break
            if not samples:
                self._idle_tick()
                if self.mode == MODE_PUSH:
                    # Nothing captured while the key is held: do not spin.
                    self._stop.wait(_PTT_POLL)
                continue

            try:
                transcript = self.stt.transcribe(samples, sample_rate=cfg.stt.sample_rate)
                text = (transcript.text or "").strip()
            except Exception:  # noqa: BLE001
                log.exception("transcription failed")
                continue

            if not text:
                self._idle_tick()
                continue

            log.info("heard: %s", text)
            self.bus.emit(Events.USER_UTTERANCE, text)
            if self.on_transcript is not None:
                try:
                    self.on_transcript(text)
                except Exception:  # noqa: BLE001 - a UI callback must not end the loop
                    log.exception("on_transcript callback failed")

            spoken = self._gate_wake_word(text)
            if spoken is None:
                continue

            self._handle(spoken)

        self._set_state(STATE_IDLE)
        self._disarm_barge_in()
        log.info("voice loop stopped")

    # ------------------------------------------------------------------ #
    #  Addressing policy
    # ------------------------------------------------------------------ #
    def _gate_wake_word(self, text: str) -> Optional[str]:
        """Apply the addressing policy.  Returns the payload, or None to skip."""
        cfg = self.config
        mode = self.mode
        now = time.monotonic()

        # Holding the key *is* the address.
        if mode == MODE_PUSH:
            return text

        if not cfg.voice.require_wake_word:
            return text

        # A short follow-up window after a reply, so you needn't repeat "Jarvis".
        if now < self._awake_until:
            return text

        # An open conversation stays open until it goes quiet for long enough.
        if mode == MODE_CONTINUOUS and now < self._continuous_until:
            self._continuous_until = now + self._num(
                "continuous_timeout", _DEFAULT_CONTINUOUS_TIMEOUT
            )
            return text

        matched, remainder = strip_wake_word(text, list(cfg.voice.wake_words))
        if not matched:
            log.debug("no wake word; ignoring")
            return None

        self.bus.emit(Events.WAKE, text)
        if mode == MODE_CONTINUOUS:
            self._continuous_until = now + self._num(
                "continuous_timeout", _DEFAULT_CONTINUOUS_TIMEOUT
            )
        if not remainder:
            # Addressed by name with no request — acknowledge and stay awake.
            self._speak(self._acknowledgement())
            self._awake_until = time.monotonic() + self._num(
                "follow_up_seconds", _DEFAULT_FOLLOW_UP
            )
            self._set_state(STATE_IDLE)
            return None
        return remainder

    def _acknowledgement(self) -> str:
        """What JARVIS says when addressed by name and nothing else."""
        configured = getattr(self.config.voice, "acknowledge", "")
        if configured:
            return str(configured)
        return "Yes, {0}?".format(self.config.agent.user_title)

    # ------------------------------------------------------------------ #
    #  Turn handling
    # ------------------------------------------------------------------ #
    def _handle(self, text: str) -> None:
        self._set_state(STATE_THINKING)

        # Speak a holding line immediately. On a CPU-only box the main model
        # can take a long time; without this the user hears nothing at all and
        # reasonably concludes it is broken.
        acknowledge = getattr(self.orchestrator, "acknowledge", None)
        if callable(acknowledge):
            try:
                acknowledge()
            except Exception:  # noqa: BLE001 - never let the filler break a turn
                log.debug("acknowledgement failed", exc_info=True)

        try:
            reply = self.orchestrator.chat(text, speak=False)
        except Exception:  # noqa: BLE001
            log.exception("agent turn failed")
            reply = "I'm afraid something went wrong handling that."

        if self.on_reply is not None:
            try:
                self.on_reply(reply)
            except Exception:  # noqa: BLE001 - a UI callback must not end the loop
                log.exception("on_reply callback failed")

        self._speak(reply, phrase=True, user_input=text)

        now = time.monotonic()
        # Allow a follow-up without the wake word.
        self._awake_until = now + self._num("follow_up_seconds", _DEFAULT_FOLLOW_UP)
        if self.mode == MODE_CONTINUOUS:
            self._continuous_until = now + self._num(
                "continuous_timeout", _DEFAULT_CONTINUOUS_TIMEOUT
            )
        if not self._interrupted:
            # After a barge-in the user is mid-sentence; queueing task reports
            # on top of them would be exactly the wrong instinct.
            self._idle_tick()
        self._set_state(STATE_IDLE)

    def _idle_tick(self) -> None:
        """Announce finished background work while nobody is talking."""
        if not self.config.agent.announce_updates:
            return
        try:
            updates = self.orchestrator.pending_updates()
        except Exception:  # noqa: BLE001
            log.debug("could not collect task updates", exc_info=True)
            return
        for update in updates:
            self._speak(update)

    def _announce_periodically(self) -> None:
        """``_idle_tick`` for poll loops — rate-limited to once a second."""
        now = time.monotonic()
        if now - self._last_announce < 1.0:
            return
        self._last_announce = now
        self._idle_tick()

    # ------------------------------------------------------------------ #
    #  Speaking, and being interrupted
    # ------------------------------------------------------------------ #
    def _speak(self, text: str, *, phrase: bool = False, user_input: str = "") -> None:
        if not text:
            return
        self._set_state(STATE_SPEAKING)
        self._interrupted = False
        armed = self._arm_barge_in()
        try:
            try:
                self.orchestrator.say(text, phrase=phrase, user_input=user_input)
            except TypeError:
                # An orchestrator (or test double) with the older say(text)
                # signature.
                self.orchestrator.say(text)
            self._await_speech_end()
        except Exception:  # noqa: BLE001
            log.exception("speech failed")
        finally:
            if armed:
                self._disarm_barge_in()
        if self._interrupted:
            # Whatever the engine still had queued belongs to a reply the user
            # has already talked over.
            self._stop_playback()
            log.info("interrupted; the rest of the reply was discarded")

    def _await_speech_end(self, *, cap: float = _SPEECH_WAIT_CAP) -> None:
        """Stay in the speaking state until the speech layer has drained.

        The speech queue is asynchronous — ``say()`` returns the moment the
        text is enqueued.  Without this wait the barge-in monitor would be
        disarmed before a single word had been spoken, which is precisely when
        it is needed.  A loop that speaks into a void must not hang, hence the
        cap and the stop-event polling.
        """
        speaker = getattr(self.orchestrator, "tts", None)
        if speaker is None:
            return
        wait = getattr(speaker, "wait", None)
        has_flag = hasattr(speaker, "is_speaking")
        if not callable(wait) and not has_flag:
            return

        deadline = time.monotonic() + max(0.0, cap)
        while not self._stop.is_set() and time.monotonic() < deadline:
            if self._interrupted:
                return
            try:
                if callable(wait):
                    if wait(timeout=0.1):
                        return
                    continue
                if not bool(getattr(speaker, "is_speaking", False)):
                    return
            except Exception:  # noqa: BLE001 - a broken speaker must not hang us
                log.debug("waiting on the speech queue failed", exc_info=True)
                return
            self._stop.wait(0.05)

    def _barge_in_threshold(self) -> float:
        margin = self._num("interrupt_margin", _DEFAULT_INTERRUPT_MARGIN)
        if margin <= 0:
            margin = _DEFAULT_INTERRUPT_MARGIN
        return max(self._noise_floor, 0.001) * margin

    def _arm_barge_in(self) -> bool:
        """Start watching the microphone for the user talking over us."""
        if not getattr(self.config.voice, "allow_interrupt", True):
            return False
        recorder = self._recorder
        start = getattr(recorder, "start_monitor", None) if recorder is not None else None
        if not callable(start):
            return False
        threshold = self._barge_in_threshold()
        try:
            start(self._on_barge_in, threshold=threshold)
        except TypeError:
            # A recorder with a simpler monitor signature.
            try:
                start(self._on_barge_in)
            except Exception:  # noqa: BLE001
                log.debug("could not arm barge-in", exc_info=True)
                return False
        except Exception:  # noqa: BLE001
            log.debug("could not arm barge-in", exc_info=True)
            return False
        self._monitor_armed = True
        return True

    def _disarm_barge_in(self) -> None:
        if not self._monitor_armed:
            return
        self._monitor_armed = False
        recorder = self._recorder
        stop = getattr(recorder, "stop_monitor", None) if recorder is not None else None
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                log.debug("stop_monitor() failed", exc_info=True)

    def _on_barge_in(self) -> None:
        """The level monitor heard sustained speech while we were talking."""
        if self._state != STATE_SPEAKING:
            return
        self._interrupted = True
        log.info("barge-in: the user started speaking")
        self._stop_playback()
        # The user is talking to us, so the next utterance needs no wake word.
        self._awake_until = time.monotonic() + self._num(
            "follow_up_seconds", _DEFAULT_FOLLOW_UP
        )

    def _stop_playback(self) -> None:
        """Silence every speech layer we can reach.  Never raises."""
        targets = [
            (self._player, "stop"),
            (self.orchestrator, "stop_speaking"),
            (getattr(self.orchestrator, "tts", None), "stop"),
        ]
        for target, attribute in targets:
            if target is None:
                continue
            method = getattr(target, attribute, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:  # noqa: BLE001
                log.debug("%s.%s() failed", type(target).__name__, attribute, exc_info=True)

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #
    def _calibrate(self, recorder: Any) -> None:
        """Measure the room so barge-in is not tuned by hand."""
        if not self._calibrate_on_start:
            return
        calibrate = getattr(recorder, "calibrate_noise_floor", None)
        if not callable(calibrate):
            return
        try:
            measured = float(calibrate())
        except Exception:  # noqa: BLE001
            log.debug("noise calibration failed", exc_info=True)
            return
        if measured > 0:
            self._noise_floor = measured
            log.info("ambient noise floor measured at %.4f", measured)

    # ------------------------------------------------------------------ #
    def start(self) -> threading.Thread:
        """Run the loop on a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="jarvis-voice", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, *, timeout: float = 5.0) -> None:
        """Ask the loop to finish and wait briefly for it."""
        self._stop.set()
        self._ptt.clear()
        recorder = self._recorder
        if recorder is not None:
            stop = getattr(recorder, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001
                    log.debug("recorder.stop() failed", exc_info=True)
        self._disarm_barge_in()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = [
    "VoiceLoop",
    "strip_wake_word",
    "VOICE_STATE",
    "STATE_IDLE",
    "STATE_LISTENING",
    "STATE_THINKING",
    "STATE_SPEAKING",
    "MODE_WAKE",
    "MODE_CONTINUOUS",
    "MODE_PUSH",
]
