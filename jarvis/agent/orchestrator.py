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
            if not turn.streamed:
                self.tts.feed(reply)
            self.tts.finish()
        elif want_speech:
            self.say(reply)
        return reply

    def say(self, text: str) -> None:
        """Speak a line, if a voice is configured.  Never raises."""
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
        return self.tasks.spawn(
            goal,
            sub.run,
            timeout=self.config.agent.subagent_timeout,
            metadata={"context": context},
            parent_id=parent_id,
        )

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

        for fn in (
            spawn_task, task_tree, list_tasks, task_status, cancel_task,
            remember, recall,
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
