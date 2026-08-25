"""Triage: decide what a turn needs before paying for it.

The small model reads every incoming utterance first. Most turns never need
the 27B at all, and on a CPU-only box that distinction is the whole difference
between an assistant and a batch job:

======================  ==========================================  ==========
route                   what it is                                  cost
======================  ==========================================  ==========
``CHITCHAT``            greetings, thanks, acknowledgements          ~0.3 s
``STATUS``              "what are you working on", "is it done yet"  ~0.3 s
``CONTROL``             pause / resume / cancel / switch project     ~0.3 s
``TASK_NEW``            a new piece of real work                     minutes
``TASK_EDIT``           a change to work already running             minutes
======================  ==========================================  ==========

The first three are answered by the small model from local state — the task
tree, the project list — and never wake the big one. Only the last two do.

Two design rules, both learned the hard way:

* **Routing is a hint, never a gate.** Every classification is advisory and
  every failure falls through to ``TASK_NEW``, which wakes the big model. A
  misrouted request that is merely slow is a nuisance; a misrouted request
  that is silently answered by a 1.7B model pretending to know the answer is a
  liar. When in doubt, escalate.
* **The router never invents facts.** For ``STATUS`` it is handed the actual
  task tree and asked to read it aloud. It is not asked what it *thinks* is
  happening.

``TASK_EDIT`` additionally resolves *which* running task is meant, so "make it
also check the logs" amends the existing subagent instead of spawning a rival
one that fights it for the same files.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
CHITCHAT = "chitchat"
STATUS = "status"
CONTROL = "control"
TASK_NEW = "task_new"
TASK_EDIT = "task_edit"

ROUTES = (CHITCHAT, STATUS, CONTROL, TASK_NEW, TASK_EDIT)

#: Routes the small model handles alone. Everything else wakes the big model.
LIGHT_ROUTES = frozenset({CHITCHAT, STATUS, CONTROL})

# Control verbs, matched before the model is consulted at all. "Stop" must be
# instant and must never depend on a language model being reachable -- if the
# agent is doing something you want halted, waiting on inference to understand
# the word "stop" is unacceptable.
CONTROL_ACTIONS = {
    "pause": ("pause", "hold", "wait", "freeze", "suspend", "hold on"),
    "resume": ("resume", "continue", "carry on", "unpause", "go on", "keep going"),
    "cancel": ("cancel", "stop", "abort", "kill", "forget it", "never mind", "drop it"),
    # Only phrases that unambiguously ask about *the agent's own work*.
    # Bare "status" and "report" are deliberately absent: "status report" and
    # "disk status" are questions about the machine, answered with tools by
    # the big model, and stealing them here would replace a real answer with
    # a task-list recital.
    "status": ("how's it going", "how is it going", "how are we doing",
               "what are you doing", "what are you working on",
               "what's running", "what is running", "are you done",
               "is it done", "is that done", "are you finished",
               "any updates", "task status", "progress update",
               "how far along", "still working"),
}

# Openers that are unambiguously social. Matched only when the whole utterance
# is short -- "thanks, now also check the disk" is not chitchat.
_CHITCHAT_PATTERNS = (
    r"^(hi|hey|hello|yo|good (morning|afternoon|evening|day))\b",
    r"^(thanks|thank you|cheers|ta|nice one|good (job|work)|well done)\b",
    r"^(bye|goodbye|good ?night|see you|that'?s all|that is all)\b",
    r"^(ok|okay|right|cool|great|excellent|perfect|understood|got it|fine)[.!]?$",
    r"^(who are you|what are you|what'?s your name)\b",
    r"^(are you (there|awake|listening|online))\b",
)
_CHITCHAT_RE = tuple(re.compile(p, re.IGNORECASE) for p in _CHITCHAT_PATTERNS)

#: Above this many characters an utterance is treated as real work even if it
#: opens with a greeting.
_CHITCHAT_MAX_CHARS = 60

# Phrases that mean "change what you are already doing" rather than "start
# something new". The distinction matters: spawning a second subagent for an
# amendment gives you two agents editing the same files.
_EDIT_MARKERS = (
    "instead", "actually", "also ", "as well", "additionally", "on top of that",
    "change it", "change that", "make it", "amend", "adjust", "tweak", "revise",
    "add to", "rather than", "not that", "scratch that", "correction",
    "update the", "modify", "and also", "one more thing",
)


@dataclass
class RouteDecision:
    """What the router concluded, and why.

    ``reason`` exists so a wrong decision can be read out of the logs rather
    than guessed at. ``confident`` is False whenever the router fell back
    rather than genuinely deciding.
    """

    route: str
    reason: str = ""
    action: str = ""                      # for CONTROL: pause/resume/cancel/status
    target_task_id: str = ""              # for TASK_EDIT
    project: str = ""
    confident: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_big_model(self) -> bool:
        """True when this turn has to wake the expensive model."""
        return self.route not in LIGHT_ROUTES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "action": self.action,
            "target_task_id": self.target_task_id,
            "project": self.project,
            "confident": self.confident,
            "needs_big_model": self.needs_big_model,
        }


# --------------------------------------------------------------------------- #
#  Deterministic pre-classification
# --------------------------------------------------------------------------- #
def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def match_control(text: str) -> Optional[str]:
    """The control verb in ``text``, or ``None``.

    Deliberately not model-driven. "Stop" has to work when the model is
    unreachable, mid-download, or busy — that is precisely when you want it.
    """
    lowered = _normalise(text).lower().rstrip(".!?")
    if not lowered:
        return None

    for action, phrases in CONTROL_ACTIONS.items():
        for phrase in phrases:
            # The whole utterance, or the utterance plus an object:
            # "pause", "pause it", "pause the backup" -- but not
            # "the pause button is broken".
            if lowered == phrase or lowered.startswith(phrase + " "):
                return action
    return None


#: A social opener followed by one of these is a real request wearing a polite
#: hat -- "good morning, now audit the disk" must not be answered with "Good
#: morning, Sir" and nothing else.
_REQUEST_AFTER_GREETING = re.compile(
    r"\b(now|please|could you|can you|would you|i need|i want|let'?s|"
    r"go ahead|next|then)\b",
    re.IGNORECASE,
)


def looks_like_chitchat(text: str) -> bool:
    """True for short, purely social utterances.

    "Purely" is the load-bearing word. A greeting with a request attached is a
    request; treating it as small talk drops the actual instruction on the
    floor, which is the worst failure mode this function has.
    """
    cleaned = _normalise(text)
    if not cleaned or len(cleaned) > _CHITCHAT_MAX_CHARS:
        return False
    if not any(pattern.search(cleaned) for pattern in _CHITCHAT_RE):
        return False
    # A greeting is only chitchat if nothing was asked for alongside it.
    if _REQUEST_AFTER_GREETING.search(cleaned):
        return False
    # More than one clause usually means the second one is the point.
    body = re.split(r"[,;:]", cleaned, maxsplit=1)
    if len(body) > 1 and len(body[1].strip()) > 12:
        return False
    return True


def looks_like_edit(text: str) -> bool:
    """True when the phrasing amends existing work rather than starting new."""
    lowered = _normalise(text).lower()
    return any(marker in lowered for marker in _EDIT_MARKERS)


# --------------------------------------------------------------------------- #
#  The router
# --------------------------------------------------------------------------- #
_CLASSIFY_PROMPT = """You are the dispatcher for an AI assistant. Classify the \
user's message into exactly one route. Reply with the route word alone, nothing \
else.

Routes:
- chitchat  : greetings, thanks, small talk, questions about you
- status    : asking about progress, what is running, whether something finished
- control   : pause, resume, cancel, stop, or switch project
- task_edit : changing, extending or correcting work that is ALREADY running
- task_new  : anything else -- a new request, question, or piece of work

{running}

Message: {message}

Route:"""


class Router:
    """Classifies turns so the expensive model is woken only when needed.

    ``small_model`` is a :class:`~jarvis.llm.voice_model.VoiceModel` or any
    object exposing a compatible ``backend.generate``. It may be ``None``, in
    which case only the deterministic rules apply and everything ambiguous
    escalates.
    """

    def __init__(self, small_model: Any = None, *, task_manager: Any = None) -> None:
        self.small_model = small_model
        self.tasks = task_manager

    # ------------------------------------------------------------------ #
    def _running_tasks(self) -> List[Any]:
        if self.tasks is None:
            return []
        try:
            from ..core.contracts import TaskState

            return [
                t for t in self.tasks.list()
                if t.state in (TaskState.RUNNING, TaskState.PENDING)
            ]
        except Exception:  # noqa: BLE001 - the task pool must not break routing
            logger.debug("could not read the task list", exc_info=True)
            return []

    def _running_block(self) -> str:
        running = self._running_tasks()
        if not running:
            return "Nothing is currently running."
        lines = [f"- [{t.id}] {t.goal}" for t in running[:8]]
        return "Currently running:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _ask_model(self, text: str) -> Optional[str]:
        """Ask the small model for a route. ``None`` when it cannot answer."""
        backend = getattr(self.small_model, "backend", None)
        if backend is None:
            return None
        try:
            from ..core.contracts import GenerationConfig, Message

            prompt = _CLASSIFY_PROMPT.format(
                running=self._running_block(), message=text
            )
            result = backend.generate(
                [Message.user(prompt)],
                # One word out. Greedy, because a creative dispatcher is a bug.
                GenerationConfig(max_new_tokens=8, temperature=0.0),
            )
            answer = str(getattr(result, "text", "") or "").strip().lower()
        except Exception:  # noqa: BLE001
            logger.debug("router classification failed", exc_info=True)
            return None

        # The model occasionally editorialises; take the first route word it
        # produced rather than demanding exact obedience.
        for route in (TASK_EDIT, TASK_NEW, CHITCHAT, STATUS, CONTROL):
            if route in answer:
                return route
        first = re.split(r"[^a-z_]+", answer)[0] if answer else ""
        return first if first in ROUTES else None

    # ------------------------------------------------------------------ #
    def route(self, text: str) -> RouteDecision:
        """Classify one utterance. Never raises."""
        cleaned = _normalise(text)
        if not cleaned:
            return RouteDecision(CHITCHAT, reason="empty utterance")

        # 1. Control verbs win outright, with no model in the loop.
        action = match_control(cleaned)
        if action == "status":
            # With nothing in flight, a progress question is almost certainly
            # about the machine rather than an empty task list. Escalate so it
            # gets a real answer instead of "nothing is running".
            if not self._running_tasks():
                return RouteDecision(
                    TASK_NEW,
                    reason="progress question with no tasks running; "
                           "treating it as a question about the system",
                )
            return RouteDecision(
                STATUS, reason="matched a status phrase", action="status"
            )
        if action:
            return RouteDecision(
                CONTROL, reason=f"matched the control verb {action!r}", action=action
            )

        # 2. Short social utterances never need a 27B model.
        if looks_like_chitchat(cleaned):
            return RouteDecision(CHITCHAT, reason="short social utterance")

        # 3. Ask the small model.
        guess = self._ask_model(cleaned)
        running = self._running_tasks()

        if guess == TASK_EDIT and not running:
            # Nothing to amend, so it is new work whatever the phrasing implies.
            return RouteDecision(
                TASK_NEW,
                reason="phrased as an edit, but nothing is running",
            )
        if guess == TASK_EDIT:
            target = self._resolve_target(cleaned, running)
            return RouteDecision(
                TASK_EDIT,
                reason="amends work already in flight",
                target_task_id=target,
            )
        if guess in (CHITCHAT, STATUS, CONTROL, TASK_NEW):
            return RouteDecision(guess, reason="classified by the small model")

        # 4. The model gave nothing usable. Fall back on phrasing, and when
        #    that is inconclusive escalate -- slow beats wrong.
        if running and looks_like_edit(cleaned):
            return RouteDecision(
                TASK_EDIT,
                reason="edit phrasing while work is running (heuristic)",
                target_task_id=self._resolve_target(cleaned, running),
                confident=False,
            )
        return RouteDecision(
            TASK_NEW,
            reason="no confident classification; escalating to the main model",
            confident=False,
        )

    # ------------------------------------------------------------------ #
    def _resolve_target(self, text: str, running: List[Any]) -> str:
        """Which running task an edit refers to.

        Word overlap against each goal, with the most recent task as the
        tie-break — "also check the logs" almost always means the thing just
        asked for, not something started an hour ago.
        """
        if not running:
            return ""
        if len(running) == 1:
            return str(running[0].id)

        words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}
        best, best_score = "", 0
        for task in running:
            goal_words = {
                w for w in re.findall(r"[a-z0-9]+", str(task.goal).lower())
                if len(w) > 3
            }
            score = len(words & goal_words)
            if score > best_score:
                best, best_score = str(task.id), score

        if best_score > 0:
            return best
        # No overlap: the most recently created task is the best guess.
        try:
            return str(max(running, key=lambda t: getattr(t, "created", 0)).id)
        except Exception:  # noqa: BLE001
            return str(running[-1].id)


__all__ = [
    "Router",
    "RouteDecision",
    "match_control",
    "looks_like_chitchat",
    "looks_like_edit",
    "CHITCHAT",
    "STATUS",
    "CONTROL",
    "TASK_NEW",
    "TASK_EDIT",
    "ROUTES",
    "LIGHT_ROUTES",
    "CONTROL_ACTIONS",
]
