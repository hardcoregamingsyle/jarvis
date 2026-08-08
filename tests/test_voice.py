"""The hands-free loop: addressing modes, barge-in, and resilience to failure."""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.core.contracts import Transcript
from jarvis.core.events import EventBus, Events
from jarvis.voice import (
    MODE_CONTINUOUS,
    MODE_PUSH,
    MODE_WAKE,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    VOICE_STATE,
    VoiceLoop,
    strip_wake_word,
)


WAKE = ["jarvis", "hey jarvis"]


# --------------------------------------------------------------------------- #
#  Wake-word matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("said,expected", [
    ("Jarvis, what is the time?", "what is the time?"),
    ("jarvis what is the time?", "what is the time?"),
    ("Hey Jarvis, open the browser", "open the browser"),
    ("JARVIS: status report", "status report"),
    ("Jarvis - close that window", "close that window"),
    ("  jarvis   dim the lights ", "dim the lights"),
])
def test_wake_word_is_stripped(said, expected):
    matched, remainder = strip_wake_word(said, WAKE)
    assert matched is True
    assert remainder == expected


def test_longer_wake_phrase_wins():
    """'hey jarvis' must not leave a stray 'jarvis' in the payload."""
    matched, remainder = strip_wake_word("hey jarvis do the thing", WAKE)
    assert matched is True
    assert remainder == "do the thing"


def test_wake_word_alone_yields_an_empty_payload():
    matched, remainder = strip_wake_word("Jarvis?", WAKE)
    assert matched is True
    assert remainder == ""


@pytest.mark.parametrize("filler", ["um", "uh", "okay", "so", "well", "hello", "right"])
def test_filler_before_the_wake_word_is_allowed(filler):
    matched, remainder = strip_wake_word(f"{filler}, jarvis, what is the time?", WAKE)
    assert matched is True
    assert remainder == "what is the time?"


@pytest.mark.parametrize("said", [
    "",
    "what is the time?",
    # Mentioning the name is not addressing it.
    "tell me about the jarvis project later on",
    "I was reading about Jarvis yesterday",
    "the jarvis file is in my documents",
])
def test_non_addressing_utterances_are_not_matched(said):
    matched, remainder = strip_wake_word(said, WAKE)
    assert matched is False
    assert remainder == said.strip()


def test_empty_wake_word_list_matches_nothing():
    assert strip_wake_word("jarvis hello", [])[0] is False
    assert strip_wake_word("jarvis hello", ["", "  "])[0] is False


def test_unicode_utterance_does_not_crash():
    matched, remainder = strip_wake_word("Jarvis, jouez la Marseillaise", WAKE)
    assert matched is True
    assert "Marseillaise" in remainder


# --------------------------------------------------------------------------- #
#  Loop behaviour
# --------------------------------------------------------------------------- #
class FakeRecorder:
    """Yields scripted utterances, then blocks until the loop is stopped."""

    def __init__(self, script):
        self.script = list(script)
        self.stopped = threading.Event()
        self.calls = 0

    def is_available(self):
        return True

    def record_until_silence(self, max_seconds=None):
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        self.stopped.wait(timeout=5)
        return []

    def stop(self):
        self.stopped.set()


class FakeSTT:
    """Maps each recorded buffer straight to its text."""

    name = "fake"

    def __init__(self, texts):
        self.texts = list(texts)

    def is_available(self):
        return True

    def transcribe(self, audio, sample_rate=16000):
        return Transcript(self.texts.pop(0) if self.texts else "")


class FakeAgent:
    def __init__(self, reply="Very good, Sir."):
        self.reply = reply
        self.heard: list = []
        self.spoken: list = []
        self.updates: list = []

    def chat(self, text, speak=False):
        self.heard.append(text)
        return self.reply

    def say(self, text):
        self.spoken.append(text)

    def pending_updates(self):
        out, self.updates = self.updates, []
        return out


def run_loop(config, agent, utterances, *, require_wake_word=True):
    """Drive the loop over a fixed script and return once it is exhausted."""
    config.voice.require_wake_word = require_wake_word
    recorder = FakeRecorder([[0.1]] * len(utterances))
    stt = FakeSTT(utterances)
    loop = VoiceLoop(config, agent, stt, recorder)

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5
    while recorder.calls <= len(utterances) and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive(), "voice loop did not stop"
    return loop


def test_utterance_with_wake_word_reaches_the_agent(config):
    agent = FakeAgent()
    run_loop(config, agent, ["Jarvis, what is the time?"])
    assert agent.heard == ["what is the time?"]
    assert "Very good, Sir." in agent.spoken


def test_utterance_without_wake_word_is_ignored(config):
    agent = FakeAgent()
    run_loop(config, agent, ["what is the time?"])
    assert agent.heard == []


def test_wake_word_not_required_passes_everything(config):
    agent = FakeAgent()
    run_loop(config, agent, ["what is the time?"], require_wake_word=False)
    assert agent.heard == ["what is the time?"]


def test_bare_wake_word_is_acknowledged_and_opens_a_follow_up_window(config):
    agent = FakeAgent()
    run_loop(config, agent, ["Jarvis?", "what is the time?"])

    # The name alone gets an acknowledgement, not a chat turn...
    assert any("Sir" in line for line in agent.spoken)
    # ...and the next sentence needs no wake word.
    assert agent.heard == ["what is the time?"]


def test_empty_transcript_is_skipped(config):
    agent = FakeAgent()
    run_loop(config, agent, ["", "   "])
    assert agent.heard == []


def test_background_updates_are_announced_when_idle(config):
    agent = FakeAgent()
    agent.updates = ["Task 'scan' completed."]
    run_loop(config, agent, [""])
    assert any("scan" in line for line in agent.spoken)


def test_a_recording_failure_does_not_end_the_loop(config):
    """A device hiccup must be survivable — this loop runs for hours."""
    class FlakyRecorder(FakeRecorder):
        def record_until_silence(self, max_seconds=None):
            self.calls += 1
            if self.calls == 1:
                raise OSError("device disappeared")
            return super().record_until_silence(max_seconds)

    agent = FakeAgent()
    recorder = FlakyRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder)

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 6
    while not agent.heard and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop(timeout=5)
    thread.join(timeout=5)

    assert agent.heard == ["hello"], "loop did not recover from a recording error"


def test_a_transcription_failure_does_not_end_the_loop(config):
    class FlakySTT(FakeSTT):
        def transcribe(self, audio, sample_rate=16000):
            if self.texts and self.texts[0] == "BOOM":
                self.texts.pop(0)
                raise RuntimeError("model exploded")
            return super().transcribe(audio, sample_rate)

    agent = FakeAgent()
    recorder = FakeRecorder([[0.1], [0.1]])
    loop = VoiceLoop(config, agent, FlakySTT(["BOOM", "Jarvis, hello"]), recorder)

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 6
    while not agent.heard and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop(timeout=5)
    thread.join(timeout=5)

    assert agent.heard == ["hello"]


def test_an_agent_failure_is_reported_but_survivable(config):
    class BrokenAgent(FakeAgent):
        def chat(self, text, speak=False):
            raise RuntimeError("model gone")

    agent = BrokenAgent()
    run_loop(config, agent, ["Jarvis, hello"])
    assert agent.spoken, "the user was left with silence after an agent error"
    assert "wrong" in agent.spoken[0].lower()


def test_available_is_false_without_a_working_recorder(config):
    class DeadRecorder(FakeRecorder):
        def is_available(self):
            return False

    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), DeadRecorder([]))
    assert loop.available() is False


def test_available_is_false_without_stt(config):
    loop = VoiceLoop(config, FakeAgent(), None, FakeRecorder([]))
    assert loop.available() is False


def test_start_and_stop_are_idempotent(config):
    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), FakeRecorder([]))
    first = loop.start()
    second = loop.start()
    assert first is second
    loop.stop(timeout=5)
    loop.stop(timeout=5)
    assert loop.running is False


# --------------------------------------------------------------------------- #
#  Shared harness for the mode / barge-in tests
# --------------------------------------------------------------------------- #
class DelayedRecorder(FakeRecorder):
    """A recorder whose utterances arrive after a scripted pause."""

    def __init__(self, script, delays):
        super().__init__(script)
        self.delays = list(delays)

    def record_until_silence(self, max_seconds=None):
        if self.delays:
            pause = self.delays.pop(0)
            if pause and self.stopped.wait(timeout=pause):
                return []
        return super().record_until_silence(max_seconds)


class MonitorRecorder(FakeRecorder):
    """A recorder that records how the barge-in monitor was driven."""

    def __init__(self, script):
        super().__init__(script)
        self.monitor_cb = None
        self.monitor_kwargs: dict = {}
        self.monitor_calls = 0
        self.stop_monitor_calls = 0

    def start_monitor(self, on_speech_detected, **kwargs):
        self.monitor_calls += 1
        self.monitor_cb = on_speech_detected
        self.monitor_kwargs = dict(kwargs)
        return None

    def stop_monitor(self, **_kwargs):
        self.stop_monitor_calls += 1
        self.monitor_cb = None


class FakePlayer:
    """Stands in for the speaker; records exactly when it was silenced."""

    def __init__(self, timeline, cut):
        self.timeline = timeline
        self.cut = cut
        self.calls = 0

    def stop(self):
        self.calls += 1
        self.timeline.append("player.stop")
        self.cut.set()


class InterruptingAgent(FakeAgent):
    """Speaks, and is talked over by the user halfway through the sentence."""

    def __init__(self, timeline, recorder, cut, reply="Very good, Sir."):
        super().__init__(reply)
        self.timeline = timeline
        self.recorder = recorder
        self.cut = cut
        self.fired = False

    def chat(self, text, speak=False):
        self.timeline.append("chat:" + text)
        return super().chat(text, speak=speak)

    def say(self, text):
        self.timeline.append("say:" + text)
        self.spoken.append(text)
        callback = self.recorder.monitor_cb
        if callback is None or self.fired:
            return
        self.fired = True
        # The real monitor calls back from its own thread while playback runs.
        threading.Thread(target=callback, daemon=True).start()
        if not self.cut.wait(timeout=3.0):
            self.timeline.append("playback-ran-to-completion")


def drive(loop, recorder, expected_calls, *, timeout=8.0):
    """Run the loop until the recorder has been asked for one more buffer."""
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while recorder.calls <= expected_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    loop.stop(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive(), "voice loop did not stop"
    return loop


# --------------------------------------------------------------------------- #
#  Barge-in
# --------------------------------------------------------------------------- #
def test_barge_in_stops_playback_before_the_next_utterance_is_handled(config):
    """The headline: talking over JARVIS silences it, and it hears you."""
    timeline: list = []
    cut = threading.Event()
    recorder = MonitorRecorder([[0.1], [0.1]])
    player = FakePlayer(timeline, cut)
    agent = InterruptingAgent(timeline, recorder, cut)
    stt = FakeSTT(["Jarvis, hello", "what is the time?"])

    loop = VoiceLoop(config, agent, stt, recorder, player=player)
    drive(loop, recorder, 2)

    assert player.calls >= 1, "the user talked over JARVIS and it kept speaking"
    assert "playback-ran-to-completion" not in timeline
    assert "chat:what is the time?" in timeline, "the interrupting words were lost"
    assert timeline.index("player.stop") < timeline.index("chat:what is the time?")
    # The interrupting sentence carried no wake word and still got through.
    assert agent.heard == ["hello", "what is the time?"]


def test_barge_in_arms_and_disarms_the_monitor_around_each_reply(config):
    timeline: list = []
    cut = threading.Event()
    recorder = MonitorRecorder([[0.1]])
    agent = InterruptingAgent(timeline, recorder, cut)
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder,
                     player=FakePlayer(timeline, cut))
    drive(loop, recorder, 1)

    assert recorder.monitor_calls >= 1, "the mic was deaf while JARVIS spoke"
    assert recorder.stop_monitor_calls >= 1, "the monitor was left running"
    assert recorder.monitor_cb is None


def test_barge_in_is_not_armed_when_interruption_is_disabled(config):
    config.voice.allow_interrupt = False
    timeline: list = []
    cut = threading.Event()
    recorder = MonitorRecorder([[0.1], [0.1]])
    player = FakePlayer(timeline, cut)
    agent = InterruptingAgent(timeline, recorder, cut)
    stt = FakeSTT(["Jarvis, hello", "what is the time?"])

    loop = VoiceLoop(config, agent, stt, recorder, player=player)
    drive(loop, recorder, 2)

    assert recorder.monitor_calls == 0
    assert player.calls == 0, "playback was cut despite allow_interrupt=False"
    assert agent.heard == ["hello", "what is the time?"]
    assert loop.interrupted is False


def test_barge_in_threshold_scales_with_the_measured_noise_floor(config):
    """The margin over the ambient floor is what stops JARVIS interrupting itself."""
    config.voice.interrupt_margin = 3.0

    class CalibratingRecorder(MonitorRecorder):
        def calibrate_noise_floor(self, seconds=1.0):
            return 0.04

    timeline: list = []
    cut = threading.Event()
    recorder = CalibratingRecorder([[0.1]])
    agent = InterruptingAgent(timeline, recorder, cut)
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder,
                     player=FakePlayer(timeline, cut))
    drive(loop, recorder, 1)

    assert loop.noise_floor == pytest.approx(0.04)
    assert recorder.monitor_kwargs.get("threshold") == pytest.approx(0.12)


def test_a_failing_calibration_does_not_stop_the_loop(config):
    class BadCalibrationRecorder(FakeRecorder):
        def calibrate_noise_floor(self, seconds=1.0):
            raise OSError("the microphone vanished mid-calibration")

    agent = FakeAgent()
    recorder = BadCalibrationRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder)
    drive(loop, recorder, 1)

    assert agent.heard == ["hello"]
    assert loop.noise_floor == pytest.approx(config.stt.silence_threshold)


def test_a_barge_in_signal_while_silent_is_ignored(config):
    """Nothing is playing, so there is nothing to cut."""
    timeline: list = []
    player = FakePlayer(timeline, threading.Event())
    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), FakeRecorder([]), player=player)

    loop._on_barge_in()

    assert player.calls == 0
    assert loop.interrupted is False


# --------------------------------------------------------------------------- #
#  Modes
# --------------------------------------------------------------------------- #
def test_default_mode_is_wake(config):
    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), FakeRecorder([]))
    assert loop.mode == MODE_WAKE


@pytest.mark.parametrize("raw", ["telepathy", "", None, 17])
def test_an_unknown_mode_falls_back_to_wake(config, raw):
    config.voice.mode = raw
    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), FakeRecorder([]))
    assert loop.mode == MODE_WAKE


def test_continuous_mode_needs_no_wake_word(config):
    config.voice.mode = MODE_CONTINUOUS
    config.voice.continuous_timeout = 60.0
    assert config.voice.require_wake_word is True   # the mode must override it

    agent = FakeAgent()
    recorder = FakeRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["what is the time?"]), recorder)
    drive(loop, recorder, 1)

    assert agent.heard == ["what is the time?"]


def test_continuous_mode_reverts_to_the_wake_word_after_the_timeout(config):
    """An open conversation must not stay open all afternoon."""
    config.voice.mode = MODE_CONTINUOUS
    config.voice.continuous_timeout = 0.5
    config.voice.follow_up_seconds = 0.05   # isolate the continuous window

    agent = FakeAgent()
    recorder = DelayedRecorder([[0.1]] * 4, [0.0, 0.2, 0.8, 0.0])
    stt = FakeSTT([
        "open the door",            # inside the window
        "close the door",           # follow-up expired, conversation still open
        "lock the door",            # conversation timed out -> ignored
        "Jarvis, lock the door",    # addressed by name -> heard again
    ])
    loop = VoiceLoop(config, agent, stt, recorder)
    drive(loop, recorder, 4)

    assert agent.heard == ["open the door", "close the door", "lock the door"]


def test_push_mode_records_nothing_until_the_key_is_pressed(config):
    config.voice.mode = MODE_PUSH
    agent = FakeAgent()
    recorder = FakeRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["what is the time?"]), recorder)

    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        assert recorder.calls == 0, "push-to-talk opened the microphone unasked"
        assert loop.state == STATE_IDLE

        loop.begin_utterance()
        deadline = time.monotonic() + 5
        while not agent.heard and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        loop.stop(timeout=5)
        thread.join(timeout=5)

    assert not thread.is_alive()
    # The key press is the address: no wake word required.
    assert agent.heard == ["what is the time?"]


def test_end_utterance_cuts_the_recording_short(config):
    config.voice.mode = MODE_PUSH
    recorder = FakeRecorder([])
    loop = VoiceLoop(config, FakeAgent(), FakeSTT([]), recorder)

    loop.begin_utterance()
    loop.end_utterance()

    assert recorder.stopped.is_set(), "the held recording was never released"


# --------------------------------------------------------------------------- #
#  State machine
# --------------------------------------------------------------------------- #
def test_state_transitions_are_emitted_in_order(config):
    seen: list = []
    event_bus = EventBus()
    event_bus.subscribe(VOICE_STATE, seen.append)

    agent = FakeAgent()
    recorder = FakeRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder, bus=event_bus)
    assert loop.state == STATE_IDLE

    drive(loop, recorder, 1)

    assert seen[:4] == [STATE_LISTENING, STATE_THINKING, STATE_SPEAKING, STATE_IDLE]
    assert loop.state == STATE_IDLE


def test_listening_events_bracket_the_microphone(config):
    timeline: list = []
    event_bus = EventBus()
    event_bus.subscribe(Events.LISTEN_START, lambda _p: timeline.append("start"))
    event_bus.subscribe(Events.LISTEN_STOP, lambda _p: timeline.append("stop"))
    event_bus.subscribe(VOICE_STATE, lambda s: timeline.append("state:" + s))

    recorder = FakeRecorder([[0.1]])
    loop = VoiceLoop(config, FakeAgent(), FakeSTT(["Jarvis, hello"]), recorder,
                     bus=event_bus)
    drive(loop, recorder, 1)

    assert timeline[:4] == ["state:listening", "start", "stop", "state:thinking"]
    # The mic must close before the reply is spoken, not after.
    assert timeline.index("stop") < timeline.index("state:speaking")


def test_no_state_is_emitted_twice_in_a_row(config):
    seen: list = []
    event_bus = EventBus()
    event_bus.subscribe(VOICE_STATE, seen.append)

    recorder = FakeRecorder([[0.1], [0.1]])
    agent = FakeAgent()
    agent.updates = ["Task 'scan' completed."]
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello", ""]), recorder,
                     bus=event_bus)
    drive(loop, recorder, 2)

    assert seen, "no state was published at all"
    assert all(a != b for a, b in zip(seen, seen[1:]))


# --------------------------------------------------------------------------- #
#  Configurable timings and wording
# --------------------------------------------------------------------------- #
def test_the_follow_up_window_is_configurable(config):
    config.voice.follow_up_seconds = 0.05

    agent = FakeAgent()
    recorder = DelayedRecorder([[0.1], [0.1]], [0.0, 0.4])
    stt = FakeSTT(["Jarvis, hello", "what is the time?"])
    loop = VoiceLoop(config, agent, stt, recorder)
    drive(loop, recorder, 2)

    assert agent.heard == ["hello"], "a stale follow-up window let a stray sentence in"


def test_the_acknowledgement_is_configurable(config):
    config.voice.acknowledge = "At your service."
    agent = FakeAgent()
    run_loop(config, agent, ["Jarvis?"])
    assert "At your service." in agent.spoken


def test_the_default_acknowledgement_uses_the_user_title(config):
    agent = FakeAgent()
    run_loop(config, agent, ["Jarvis?"])
    assert agent.spoken[0] == "Yes, {0}?".format(config.agent.user_title)


# --------------------------------------------------------------------------- #
#  The real speech-queue wiring: say() returns long before the words do
# --------------------------------------------------------------------------- #
class FakeSpeechQueue:
    """Mimics SpeechQueue: ``say`` enqueues and returns immediately."""

    def __init__(self, drain_after: float = 0.0):
        self.spoken: list = []
        self.stop_calls = 0
        self._empty_at = 0.0
        self._drain_after = drain_after

    @property
    def is_speaking(self) -> bool:
        return time.monotonic() < self._empty_at

    def say(self, text):
        self.spoken.append(text)
        self._empty_at = time.monotonic() + self._drain_after

    def wait(self, timeout=None):
        time.sleep(min(0.02, timeout) if timeout else 0.02)
        return not self.is_speaking

    def stop(self):
        self.stop_calls += 1
        self._empty_at = 0.0


class QueueAgent(FakeAgent):
    """An orchestrator whose speech goes through an asynchronous queue."""

    def __init__(self, tts):
        super().__init__()
        self.tts = tts

    def say(self, text):
        self.spoken.append(text)
        self.tts.say(text)


def _trip_the_monitor(recorder, fired, *, timeout=5.0):
    """Call the barge-in callback the instant the loop arms it."""
    def _run():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            callback = recorder.monitor_cb
            if callback is not None:
                callback()
                fired.set()
                return
            time.sleep(0.005)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def test_the_speech_queue_is_drained_when_the_user_interrupts(config):
    """say() returns at once, so the loop must hold the mic open until it ends."""
    speech = FakeSpeechQueue(drain_after=30.0)   # a very long reply
    agent = QueueAgent(speech)
    recorder = MonitorRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder)

    fired = threading.Event()
    _trip_the_monitor(recorder, fired)
    started = time.monotonic()
    drive(loop, recorder, 1)

    assert fired.is_set(), "the barge-in monitor was never armed"
    assert speech.stop_calls >= 1, "the queued reply kept playing over the user"
    assert loop.interrupted is True
    assert time.monotonic() - started < 10, "the loop waited out the whole reply"


def test_a_reply_that_finishes_naturally_is_not_treated_as_interrupted(config):
    speech = FakeSpeechQueue(drain_after=0.1)
    agent = QueueAgent(speech)
    recorder = MonitorRecorder([[0.1]])
    loop = VoiceLoop(config, agent, FakeSTT(["Jarvis, hello"]), recorder)
    drive(loop, recorder, 1)

    assert speech.spoken == ["Very good, Sir."]
    assert speech.stop_calls == 0
    assert loop.interrupted is False
    assert recorder.stop_monitor_calls >= 1


def test_ui_callbacks_that_raise_do_not_end_the_loop(config):
    agent = FakeAgent()
    recorder = FakeRecorder([[0.1], [0.1]])
    # The second sentence rides the follow-up window, so it arrives verbatim.
    stt = FakeSTT(["Jarvis, hello", "and again"])

    def boom(_text):
        raise RuntimeError("the console is on fire")

    loop = VoiceLoop(config, agent, stt, recorder, on_transcript=boom, on_reply=boom)
    drive(loop, recorder, 2)

    assert agent.heard == ["hello", "and again"], "a broken UI callback stopped the loop"
