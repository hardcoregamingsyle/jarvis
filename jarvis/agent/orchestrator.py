"""The main agent: the thing you actually talk to.

Responsibilities:
  * hold the long-lived pieces (LLM, tools, memory, task pool, voice),
  * answer conversationally and quickly,
  * delegate anything slow to a subagent so the conversation never blocks,
  * surface subagent reports at the next opportunity,
  * register the "meta" tools through which it manages its own delegation,
    memory and self-extension.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from ..core.config import Config
from ..core.contracts import (
    GenerationConfig,
    LLMBackend,
    Message,
    Task,
    TaskState,
)
from ..core.events import EventBus, Events, get_bus
from ..core.platform_utils import system_summary
from .prompts import build_system_prompt, environment_block
from .subagent import SubAgent, current_agent_context, run_agent_loop
from .task_manager import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_TOTAL_TASKS,
    TaskManager,
)

log = logging.getLogger(__name__)


class Orchestrator:
    """The primary JARVIS agent."""

    def __init__(
        self,
        config: Config,
        llm: LLMBackend,
        registry: Any,
        context: Any,
        *,
        tts: Any = None,
        bus: Optional[EventBus] = None,
        task_manager: Optional[TaskManager] = None,
        task_llm: Optional[LLMBackend] = None,
        voice_model: Any = None,
    ) -> None:
        self.config = config
        self.llm = llm
        # The model spawn_task delegates to. Falls back to `llm` itself when
        # none is configured, which is exactly today's single-backend
        # behaviour -- nothing changes for a setup that doesn't opt in.
        self.task_llm = task_llm or llm
        self.registry = registry
        self.context = context
        self.tts = tts
        self.bus = bus or get_bus()
        # Duck-typed rather than isinstance-checked, so any object exposing
        # feed()/finish() (StreamingSpeaker's contract) qualifies -- tests can
        # swap in a fake without importing the real class.
        self._live_speech = bool(
            tts is not None and hasattr(tts, "feed") and hasattr(tts, "finish")
        )
        # The small, fast model that phrases replies for speech. Optional: when
        # absent the main model's own prose is spoken instead. Built lazily so
        # constructing an Orchestrator never blocks on a second backend probe.
        self._voice_model = voice_model
        self._voice_model_ready = voice_model is not None

        # Projects and durable task state. Built lazily and never fatal: a
        # broken store must degrade to "no persistence", not "no assistant".
        self._projects: Any = None
        self._projects_ready = False
        self._router: Any = None
        self._router_ready = False

        # Tree limits are resource management, not permission: they stop one
        # mis-prompted agent turning into a fork bomb.  Read defensively because
        # an older AgentConfig may not carry them yet.
        self.tasks = task_manager or TaskManager(
            max_workers=getattr(config.agent, "max_concurrent_tasks", 4),
            bus=self.bus,
            default_timeout=config.agent.subagent_timeout,
            max_depth=getattr(config.agent, "max_agent_depth", DEFAULT_MAX_DEPTH),
            max_total_tasks=getattr(
                config.agent, "max_total_tasks", DEFAULT_MAX_TOTAL_TASKS
            ),
        )

        env_lines = [
            f"llm backend: {getattr(llm, 'name', 'unknown')} ({config.llm.model})",
        ]
        # When a separate, more capable model is configured for spawn_task,
        # this becomes the persona's ONE job: hold the conversation, answer
        # what is genuinely quick, and delegate the rest immediately rather
        # than working through a long tool chain on a small, fast model that
        # was never meant to carry it.
        self._router_extra = ""
        if self.task_llm is not llm:
            task_model = getattr(config.llm, "task_model", "") or getattr(
                config.llm, "task_ollama_model", ""
            ) or config.llm.model
            env_lines.append(
                f"spawn_task delegates to a SEPARATE, more capable model: "
                f"{getattr(self.task_llm, 'name', 'unknown')} ({task_model})."
            )
            self._router_extra = (
                "You are the fast, conversational front end of a two-model "
                f"setup; {task_model} is the deep-work model behind spawn_task. "
                "Answer directly only what you can genuinely do in one or two "
                "quick steps. For anything that touches multiple files, runs "
                "more than a command or two, needs real research, or will take "
                "more than a few seconds — call spawn_task at once, say a short "
                "acknowledgement, and move on. Do not chain many tool calls "
                "yourself trying to finish it inline; that is spawn_task's job, "
                "not yours."
            )
        self._env = environment_block(system_summary(), extra_lines=env_lines)
        self._lock = threading.RLock()
        self._speaking_enabled = bool(config.tts.enabled)

        if self.registry is not None:
            self._register_meta_tools()
            # Give tool_maker access to the model so `create_tool` can write code.
            ctx = getattr(self.registry, "ctx", None)
            if ctx is not None and hasattr(ctx, "extra"):
                ctx.extra.setdefault("llm", llm)
                ctx.extra.setdefault("orchestrator", self)

    # ------------------------------------------------------------------ #
    #  Prompt assembly
    # ------------------------------------------------------------------ #
    def system_prompt(self) -> str:
        catalogue = self.registry.describe() if self.registry is not None else ""
        return build_system_prompt(
            name=self.config.agent.name,
            user_title=self.config.agent.user_title,
            environment=self._env,
            tools_catalogue=catalogue,
            extra=self._router_extra,
        )

    def _gen_config(self) -> GenerationConfig:
        llm_cfg = self.config.llm
        return GenerationConfig(
            max_new_tokens=llm_cfg.max_new_tokens,
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            top_k=llm_cfg.top_k,
        )

    def _task_gen_config(self) -> GenerationConfig:
        """Generation settings for work delegated via spawn_task.

        Deliberately separate from :meth:`_gen_config`: the task model is
        usually doing more substantial work and reasonably wants a longer
        leash (``task_max_new_tokens`` defaults to 1024 against the router's
        512), while top_p/top_k are shared since neither field has a
        task-specific override yet.
        """
        llm_cfg = self.config.llm
        return GenerationConfig(
            max_new_tokens=getattr(llm_cfg, "task_max_new_tokens", llm_cfg.max_new_tokens),
            temperature=getattr(llm_cfg, "task_temperature", llm_cfg.temperature),
            top_p=llm_cfg.top_p,
            top_k=llm_cfg.top_k,
        )

    # ------------------------------------------------------------------ #
    #  Conversation
    # ------------------------------------------------------------------ #
    def chat(
        self,
        user_input: str,
        *,
        speak: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        stream: bool = False,
    ) -> str:
        """Handle one user turn and return the spoken reply.

        When a live-speech TTS is configured (``StreamingSpeaker`` or
        anything duck-typed the same way) and speaking is enabled, the model
        is streamed and spoken sentence-by-sentence AS it generates, not
        after the full reply exists — this is what makes the router feel
        instant rather than waiting for a whole reply before making a sound.
        A caller-supplied ``on_chunk``/``stream`` still work as before (e.g.
        a UI transcript) and run alongside the live speech, not instead of it.
        """
        user_input = (user_input or "").strip()
        if not user_input:
            return ""

        self.bus.emit(Events.USER_UTTERANCE, user_input)
        want_speech = speak if speak is not None else self._speaking_enabled
        live = want_speech and self._live_speech

        def _on_chunk(piece: str) -> None:
            if live:
                self.tts.feed(piece)
            if on_chunk is not None:
                on_chunk(piece)

        # Triage first. Most turns -- greetings, "how's it going", "pause" --
        # are answered from local state in a fraction of a second, and never
        # wake the big model. Routing is advisory: anything it cannot classify
        # confidently escalates, because slow is better than confidently wrong.
        decision = self._route(user_input)
        if decision is not None and not decision.needs_big_model:
            reply = self._handle_light(decision, user_input)
            if reply:
                self.context.add_user(user_input)
                self.context.add_assistant(reply)
                self.bus.emit(Events.ASSISTANT_REPLY, reply)
                if speak if speak is not None else self._speaking_enabled:
                    self.say(reply, phrase=False)
                return reply

        with self._lock:
            # Any finished background work is folded into this turn's context.
            reports = self._collect_reports()

            self.context.add_user(user_input)
            messages = self.context.build(user_input, extra=reports)
            if messages and messages[0].role.value == "system":
                messages[0] = Message.system(self.system_prompt())
            else:
                messages.insert(0, Message.system(self.system_prompt()))

            turn = run_agent_loop(
                self.llm,
                self.registry,
                messages,
                max_iterations=self.config.agent.max_tool_iterations,
                gen_config=self._gen_config(),
                on_chunk=_on_chunk if (live or on_chunk is not None) else None,
                bus=self.bus,
                stream=stream or live,
                tool_result_limit=getattr(
                    self.config.llm, "max_tool_result_tokens", 8000
                ),
            )

            reply = turn.text or "I'm afraid I have nothing useful to say to that."
            self.context.add_assistant(reply)
            try:
                self.context.maybe_summarize()
            except Exception:  # noqa: BLE001 - summarisation must never break a turn
                log.exception("summarisation failed")

        self.bus.emit(Events.ASSISTANT_REPLY, reply)
        if live:
            # turn.streamed tells us whether `reply` already went out through
            # _on_chunk (the ordinary path) or was produced outside it (the
            # synthetic fallback answer, or the truncated-loop generate()) --
            # only the latter still needs to be handed to the speaker.
            #
            # Live speech deliberately skips the voice-model rephrasing pass
            # below (`phrase=True`): that pass needs the complete answer
            # before it can run, which is exactly the wait live speech exists
            # to avoid. The big model's own prose is spoken as generated.
            if not turn.streamed:
                self.tts.feed(reply)
            self.tts.finish()
        elif want_speech:
            self.say(reply, phrase=True, user_input=user_input)
        return reply

    # ------------------------------------------------------------------ #
    #  Projects and routing
    # ------------------------------------------------------------------ #
    @property
    def projects(self) -> Any:
        """The durable project/task store, or ``None`` if it cannot be opened."""
        if not self._projects_ready:
            self._projects_ready = True
            try:
                from .projects import ProjectStore

                self._projects = ProjectStore(self.config.path("projects.db"))
                # Anything still marked RUNNING belongs to a thread that did
                # not survive the last shutdown. Say so honestly.
                self._projects.mark_interrupted()
            except Exception:  # noqa: BLE001 - persistence is not load-bearing
                log.exception("project store unavailable; work will not persist")
                self._projects = None
        return self._projects

    @property
    def router(self) -> Any:
        """Triage for incoming turns, or ``None`` when routing is disabled."""
        if not self._router_ready:
            self._router_ready = True
            if not getattr(self.config.llm, "routing_enabled", True):
                self._router = None
            else:
                try:
                    from .router import Router

                    self._router = Router(self.voice_model, task_manager=self.tasks)
                except Exception:  # noqa: BLE001
                    log.debug("router unavailable", exc_info=True)
                    self._router = None
        return self._router

    def active_project(self) -> str:
        store = self.projects
        if store is None:
            from .projects import DEFAULT_PROJECT

            return DEFAULT_PROJECT
        try:
            return store.active_project()
        except Exception:  # noqa: BLE001
            log.debug("could not read the active project", exc_info=True)
            return "general"

    def handle_control(self, action: str, *, target: str = "") -> str:
        """pause / resume / cancel, applied to the running tree and to disk.

        Returns a spoken sentence. Never raises: "stop" has to work even when
        parts of the system are already unwell.
        """
        from . import projects as P

        store = self.projects
        acted = 0

        try:
            live = [
                t for t in self.tasks.list()
                if t.state.value in ("running", "pending")
            ]
        except Exception:  # noqa: BLE001
            live = []

        targets = [t for t in live if not target or t.id == target]

        if action in ("pause", "cancel"):
            for task in targets:
                try:
                    self.tasks.cancel(task.id)
                    acted += 1
                except Exception:  # noqa: BLE001
                    log.debug("could not stop %s", task.id, exc_info=True)
                if store is not None:
                    store.update_state(
                        task.id, P.PAUSED if action == "pause" else P.CANCELLED
                    )
            if action == "pause":
                return (
                    f"Paused {acted} task{'s' if acted != 1 else ''}, "
                    f"{self.config.agent.user_title}. "
                    "Their progress is saved; say resume to carry on."
                ) if acted else "Nothing was running to pause."
            return (
                f"Cancelled {acted} task{'s' if acted != 1 else ''}."
                if acted else "Nothing was running to cancel."
            )

        if action == "resume":
            if store is None:
                return "I cannot resume: the project store is unavailable."
            paused = [
                t for t in store.resumable_tasks(project=store.active_project())
                if t.state == P.PAUSED
            ]
            if not paused:
                return "Nothing is on hold."
            for task in paused:
                try:
                    # Re-dispatch with everything already learned, so it
                    # continues rather than starting over.
                    self.spawn_task(task.resume_briefing())
                    store.update_state(task.id, P.RUNNING)
                    acted += 1
                except Exception:  # noqa: BLE001
                    log.debug("could not resume %s", task.id, exc_info=True)
            return (
                f"Resumed {acted} task{'s' if acted != 1 else ''} "
                "from where they left off."
            )

        if action == "status":
            return self.status_line()

        return f"I do not know how to {action}."

    def status_line(self) -> str:
        """What is running, in one spoken sentence."""
        store = self.projects
        if store is not None:
            try:
                return store.summary()
            except Exception:  # noqa: BLE001
                log.debug("project summary failed", exc_info=True)
        try:
            stats = self.tasks.stats()
            running = stats.get("running", 0)
            return (
                f"{running} task{'s' if running != 1 else ''} running, "
                f"{self.config.agent.user_title}."
            )
        except Exception:  # noqa: BLE001
            return "I could not read the task list."

    # ------------------------------------------------------------------ #
    #  The voice model
    # ------------------------------------------------------------------ #
    @property
    def voice_model(self) -> Any:
        """The small model that phrases replies aloud, or ``None``.

        Built on first use rather than in ``__init__`` so that constructing an
        Orchestrator never pays for a second backend probe, and so a machine
        with no small model available simply never builds one.
        """
        if not self._voice_model_ready:
            self._voice_model_ready = True
            try:
                from ..llm.voice_model import create_voice_model

                self._voice_model = create_voice_model(
                    self.config.llm,
                    agent_name=self.config.agent.name,
                    user_title=self.config.agent.user_title,
                )
            except Exception:  # noqa: BLE001 - an optional layer, never fatal
                log.debug("voice model unavailable", exc_info=True)
                self._voice_model = None
        return self._voice_model

    def acknowledge(self) -> str:
        """Speak a holding line while the main model thinks.

        This is what stops a slow local model reading as a crash: the user
        hears a reply within a fraction of a second, and the real answer
        follows when it is ready. Returns the line spoken, or ``""``.
        """
        model = self.voice_model
        if model is None:
            return ""
        try:
            line = model.acknowledge()
        except Exception:  # noqa: BLE001
            log.debug("acknowledgement failed", exc_info=True)
            return ""
        if line:
            self.say(line, phrase=False)
        return line

    def _route(self, user_input: str) -> Any:
        """Classify a turn. ``None`` when routing is off or unavailable."""
        router = self.router
        if router is None:
            return None
        try:
            decision = router.route(user_input)
        except Exception:  # noqa: BLE001 - triage must never break a turn
            log.debug("routing failed; escalating", exc_info=True)
            return None
        log.info(
            "route=%s big=%s (%s)",
            decision.route, decision.needs_big_model, decision.reason,
        )
        return decision

    def _handle_light(self, decision: Any, user_input: str) -> str:
        """Answer a turn that does not need the big model. "" to escalate."""
        from .router import CHITCHAT, CONTROL, STATUS

        if decision.route == CONTROL:
            return self.handle_control(
                decision.action, target=decision.target_task_id
            )

        if decision.route == STATUS:
            # Read the real task tree aloud rather than asking a 1.7B model
            # what it imagines is happening.
            line = self.status_line()
            model = self.voice_model
            if model is not None:
                try:
                    return model.speakable(line, user_input=user_input) or line
                except Exception:  # noqa: BLE001
                    log.debug("status phrasing failed", exc_info=True)
            return line

        if decision.route == CHITCHAT:
            model = self.voice_model
            if model is None:
                # Without a small model there is nothing cheap to answer with,
                # so let the big one handle it rather than inventing a reply.
                return ""
            try:
                from ..core.contracts import GenerationConfig, Message

                result = model.backend.generate(
                    [
                        Message.system(
                            f"You are {self.config.agent.name}, a British AI "
                            f"assistant. Reply to this in ONE short spoken "
                            f"sentence. Address the user as "
                            f"{self.config.agent.user_title}."
                        ),
                        Message.user(user_input),
                    ],
                    GenerationConfig(max_new_tokens=60, temperature=0.6),
                )
                from ..llm.voice_model import strip_markup

                return strip_markup(getattr(result, "text", "") or "")
            except Exception:  # noqa: BLE001
                log.debug("chitchat reply failed; escalating", exc_info=True)
                return ""

        return ""

    def say(self, text: str, *, phrase: bool = False, user_input: str = "") -> None:
        """Speak a line, if a voice is configured.  Never raises.

        With ``phrase=True`` the text is first handed to the voice model to be
        rendered as spoken prose; that path is used for real answers, not for
        acknowledgements and greetings which are already speech-shaped.
        """
        if not text:
            return

        if phrase:
            model = self.voice_model
            if model is not None:
                try:
                    text = model.speakable(text, user_input=user_input) or text
                except Exception:  # noqa: BLE001 - fall back to the raw answer
                    log.debug("voice-model phrasing failed", exc_info=True)
            else:
                # No voice model: still strip markdown, which is never spoken.
                try:
                    from ..llm.voice_model import strip_markup

                    text = strip_markup(text) or text
                except Exception:  # noqa: BLE001
                    log.debug("markup stripping failed", exc_info=True)

        if not text:
            return
        self.bus.emit(Events.SPEAK, text)
        if self.tts is None:
            return
        try:
            speaker = getattr(self.tts, "say", None) or getattr(self.tts, "speak", None)
            if speaker is not None:
                speaker(text)
        except Exception:  # noqa: BLE001
            log.exception("speech failed")

    # ------------------------------------------------------------------ #
    #  Delegation
    # ------------------------------------------------------------------ #
    @property
    def max_agent_depth(self) -> int:
        """Deepest task depth the pool will accept."""
        return int(getattr(self.tasks, "max_depth", DEFAULT_MAX_DEPTH))

    @property
    def max_total_tasks(self) -> int:
        """How many tasks the pool will track at once."""
        return int(getattr(self.tasks, "max_total_tasks", DEFAULT_MAX_TOTAL_TASKS))

    def spawn_task(
        self,
        goal: str,
        *,
        context: str = "",
        parent_id: Optional[str] = None,
    ) -> Task:
        """Dispatch a subagent under ``parent_id``.  Returns immediately.

        With no explicit parent the task is attached to whichever agent is
        calling — a subagent's tools run on its own worker thread, so its
        children hang off it rather than becoming stray roots.  A refused spawn
        comes back as an already-failed :class:`Task`; nothing raises.
        """
        if parent_id is None:
            parent_id = current_agent_context().get("task_id")
        if parent_id and self.tasks.get(parent_id) is None:
            log.debug("parent task %s is no longer tracked; spawning at root", parent_id)
            parent_id = None

        child_depth = getattr(self.tasks, "child_depth", None)
        depth = child_depth(parent_id) if child_depth is not None else 0
        sub = SubAgent(
            self.task_llm,
            self.registry,
            agent_name=self.config.agent.name,
            environment=self._env,
            max_iterations=max(4, self.config.agent.max_tool_iterations * 2),
            gen_config=self._task_gen_config(),
            bus=self.bus,
            depth=depth,
            parent_id=parent_id,
            max_depth=self.max_agent_depth,
        )
        task = self.tasks.spawn(
            goal,
            sub.run,
            timeout=self.config.agent.subagent_timeout,
            metadata={"context": context},
            parent_id=parent_id,
        )

        # Mirror it to disk. The in-memory tree schedules; this is what
        # survives a reboot and makes pause/resume mean anything.
        store = self.projects
        if store is not None:
            try:
                from . import projects as P

                store.record_task(
                    task.id,
                    goal,
                    state=P.RUNNING,
                    parent_id=parent_id or "",
                    depth=depth,
                    metadata={"context": context},
                )
            except Exception:  # noqa: BLE001 - persistence is never fatal
                log.debug("could not persist task %s", task.id, exc_info=True)
        return task

    def task_tree(self, root_id: Optional[str] = None) -> str:
        """A compact rendering of the whole task tree, suitable to read aloud."""
        render = getattr(self.tasks, "render_tree", None)
        if render is None:  # pragma: no cover - only a stub manager lacks it
            return "(the task tree is unavailable)"
        stats = self.tasks.stats()
        header = (
            "%d task(s) tracked, %d running, deepest depth %d of %d"
            % (
                stats.get("tracked", stats.get("total", 0)),
                stats.get("running", 0),
                stats.get("deepest_depth", 0),
                stats.get("max_depth", self.max_agent_depth),
            )
        )
        return "%s\n%s" % (header, render(root_id))

    def _task_payload(self, task: Task) -> dict:
        """What a spawning agent is told about the task it just created."""
        try:
            depth = int(task.metadata.get("depth", 0) or 0)
        except (TypeError, ValueError):
            depth = 0
        tracked = len(self.tasks.list())
        payload: dict = {
            "task_id": task.id,
            "goal": task.goal,
            "state": task.state.value,
            "depth": depth,
            "parent_id": task.metadata.get("parent_id"),
            "max_depth": self.max_agent_depth,
            "levels_remaining": max(0, self.max_agent_depth - depth),
            "tasks_tracked": tracked,
            "tasks_remaining": max(0, self.max_total_tasks - tracked),
        }
        if task.metadata.get("refused"):
            payload["refused"] = True
        if task.error:
            payload["error"] = task.error
        return payload

    def _collect_reports(self) -> list:
        """Turn finished background tasks into context for the next reply."""
        if not self.config.agent.announce_updates:
            return []
        messages: list = []
        for task in self.tasks.take_reports():
            summary = self._format_report(task)
            messages.append(
                Message.system(f"[Background task update]\n{summary}")
            )
            try:
                self.context.store.add_text(
                    "task",
                    summary,
                    task_id=task.id,
                    goal=task.goal,
                    state=task.state.value,
                    depth=task.metadata.get("depth", 0),
                    parent_id=task.metadata.get("parent_id"),
                )
            except Exception:  # noqa: BLE001 - persistence is best-effort here
                log.debug("could not persist task report", exc_info=True)
        return messages

    def _lineage(self, task: Task) -> str:
        """Where in the tree a report came from, phrased for the prompt."""
        try:
            depth = int(task.metadata.get("depth", 0) or 0)
        except (TypeError, ValueError):
            depth = 0
        if not depth:
            return ""
        parent = self.tasks.get(task.metadata.get("parent_id") or "")
        if parent is not None:
            return f" (depth {depth}, delegated under '{parent.goal}')"
        return f" (depth {depth})"

    @staticmethod
    def _child_lines(result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        children = result.get("children") or []
        lines = []
        for child in children:
            if not isinstance(child, dict):
                continue
            detail = child.get("report") or child.get("error") or child.get("state", "")
            lines.append(f"  - {child.get('goal', 'subtask')}: {detail}")
        if not lines:
            return ""
        return "\nIts own subtasks reported:\n" + "\n".join(lines)

    def _format_report(self, task: Task) -> str:
        """Render one finished task — at any depth — for the next turn."""
        where = self._lineage(task)
        if task.state is TaskState.DONE:
            result = task.result
            body = result.get("report") if isinstance(result, dict) else str(result)
            return f"Task '{task.goal}'{where} completed.\n{body}{self._child_lines(result)}"
        if task.state is TaskState.CANCELLED:
            return f"Task '{task.goal}'{where} was cancelled."
        return f"Task '{task.goal}'{where} failed: {task.error}"

    def pending_updates(self) -> list:
        """Human-readable summaries of finished tasks, marking them announced."""
        return [self._format_report(t) for t in self.tasks.take_reports()]

    def announce_pending(self) -> Optional[str]:
        """Speak any finished-task reports.  Called when the agent goes idle."""
        updates = self.pending_updates()
        if not updates:
            return None
        line = " ".join(updates)
        self.say(line)
        return line

    # ------------------------------------------------------------------ #
    #  Meta-tools
    # ------------------------------------------------------------------ #
    def _register_meta_tools(self) -> None:
        """Expose delegation and memory to the model as ordinary tools."""
        register = self.registry.register_function

        def spawn_task(goal: str, context: str = "", parent_task_id: str = "") -> dict:
            """Dispatch a background subagent to pursue a goal, and return at once.

            Use this for anything slow, open-ended, or multi-step: research,
            large file operations, installations, long scans. You will be given
            the result when it is ready. The new task is attached to you unless
            parent_task_id names a different task to hang it under. The reply
            reports the new task's depth and how much delegation budget is left;
            if it comes back with an error, read it and do the work yourself.
            """
            task = self.spawn_task(
                goal, context=context, parent_id=parent_task_id or None
            )
            return self._task_payload(task)

        def task_tree(task_id: str = "") -> str:
            """Show the whole tree of background tasks: who is working on what.

            Answers "what are you working on?" — one indented line per task with
            its id, state and goal. Give task_id to show a single branch.
            """
            return self.task_tree(task_id or None)

        def list_tasks(state: str = "") -> list:
            """List background tasks, optionally filtered by state."""
            wanted = None
            if state:
                try:
                    wanted = TaskState(state.lower())
                except ValueError:
                    return [{"error": f"unknown state '{state}'"}]
            return [
                {
                    "task_id": t.id,
                    "goal": t.goal,
                    "state": t.state.value,
                    "depth": t.metadata.get("depth", 0),
                    "parent_id": t.metadata.get("parent_id"),
                    "updates": len(t.updates),
                }
                for t in self.tasks.list(state=wanted)
            ]

        def task_status(task_id: str) -> dict:
            """Get the current state, position in the tree and latest progress
            of one background task."""
            task = self.tasks.get(task_id)
            if task is None:
                return {"error": f"no such task: {task_id}"}
            last = task.updates[-1].message if task.updates else ""
            payload: dict = {
                "task_id": task.id,
                "goal": task.goal,
                "state": task.state.value,
                "latest": last,
                "depth": task.metadata.get("depth", 0),
                "parent_id": task.metadata.get("parent_id"),
                "ancestry": self.tasks.ancestry(task.id),
                "children": [c.id for c in self.tasks.children(task.id)],
            }
            if task.state is TaskState.DONE and isinstance(task.result, dict):
                payload["report"] = task.result.get("report", "")
                if task.result.get("children"):
                    payload["child_reports"] = task.result["children"]
            if task.error:
                payload["error"] = task.error
            return payload

        def cancel_task(task_id: str) -> dict:
            """Cancel a running background task and everything it delegated."""
            subtree = [t.id for t in self.tasks.descendants(task_id)]
            return {
                "cancelled": self.tasks.cancel(task_id),
                "task_id": task_id,
                "also_cancelled": subtree,
            }

        def remember(text: str, category: str = "fact") -> dict:
            """Commit a durable fact to long-term memory.

            Use for preferences, names, machine details, standing instructions —
            anything worth recalling in a later conversation.
            """
            record = self.context.remember_fact(text, category=category)
            return {"stored": True, "id": getattr(record, "id", ""), "text": text}

        def recall(query: str, limit: int = 5) -> list:
            """Search long-term memory for anything relevant to a query."""
            hits = self.context.store.search(query, k=max(1, int(limit)))
            return [
                {"text": h.text, "kind": h.kind, "score": round(h.score, 3)}
                for h in hits
            ]

        # -- projects and durable state ---------------------------------- #
        def create_project(name: str, description: str = "") -> dict:
            """Create a project: a named container for a set of related tasks.

            Use one per distinct body of work. Tasks are filed under whichever
            project is active, so status and pause/resume have an unambiguous
            subject instead of one global pile.
            """
            store = self.projects
            if store is None:
                return {"error": "the project store is unavailable"}
            project = store.create_project(name, description=description)
            return {"created": True, "project": project.to_dict()}

        def switch_project(name: str) -> dict:
            """Make a project active. New tasks are filed under it."""
            store = self.projects
            if store is None:
                return {"error": "the project store is unavailable"}
            return {"active": store.set_active_project(name)}

        def list_projects() -> list:
            """Every project, with the active one first."""
            store = self.projects
            if store is None:
                return [{"error": "the project store is unavailable"}]
            return [p.to_dict() for p in store.list_projects()]

        def pause_work(task_id: str = "") -> dict:
            """Put work on hold so it survives a reboot.

            Progress notes are written to disk; `resume_work` re-dispatches a
            subagent primed with them, so it continues rather than restarting.
            Omit task_id to pause everything running.
            """
            return {"message": self.handle_control("pause", target=task_id)}

        def resume_work() -> dict:
            """Restart paused work from where it left off."""
            return {"message": self.handle_control("resume")}

        def record_progress(task_id: str, note: str) -> dict:
            """Record what a task has achieved so far.

            This is what makes resuming meaningful: without notes a resumed
            task starts from nothing. Call it as milestones are reached, not
            at the end.
            """
            store = self.projects
            if store is None:
                return {"error": "the project store is unavailable"}
            return {"recorded": bool(store.add_progress(task_id, note))}

        for fn in (
            spawn_task, task_tree, list_tasks, task_status, cancel_task,
            remember, recall,
            create_project, switch_project, list_projects,
            pause_work, resume_work, record_progress,
        ):
            try:
                register(fn, dangerous=False)
            except Exception:  # noqa: BLE001 - duplicate registration on restart
                log.debug("meta tool %s already registered", fn.__name__)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def greet(self) -> str:
        line = self.config.voice.greeting
        self.say(line)
        return line

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop background work and release resources.  Idempotent."""
        self.bus.emit(Events.SHUTDOWN, None)
        try:
            self.tasks.shutdown(wait=wait)
        except Exception:  # noqa: BLE001
            log.debug("task manager shutdown raised", exc_info=True)
        resources = [self.tts, self.llm, getattr(self.context, "store", None)]
        if self.task_llm is not self.llm:
            resources.append(self.task_llm)
        for resource in resources:
            for method in ("shutdown", "close", "unload"):
                fn = getattr(resource, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        log.debug("%s.%s failed", resource, method, exc_info=True)
                    break

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()


__all__ = ["Orchestrator"]
