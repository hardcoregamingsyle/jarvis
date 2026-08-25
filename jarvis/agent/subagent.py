"""The reason-act loop, and the subagent that runs it autonomously.

:func:`run_agent_loop` is the single implementation of "talk to the model, run
any tools it asks for, repeat until it answers".  Both the interactive
orchestrator and the background subagents use it, so their behaviour cannot
drift apart.

Subagents form a tree: because ``spawn_task`` lives on the shared registry, a
subagent can delegate exactly as the main agent does.  A :class:`SubAgent`
therefore knows its own ``depth`` and ``parent_id``, publishes them on a
thread-local so that any tool it calls can attribute the work to it without the
model having to remember an id, and states its remaining delegation budget in
its own system prompt so the model can decide whether to delegate at all.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

from ..core.contracts import (
    GenerationConfig,
    LLMBackend,
    Message,
    Task,
    ToolResult,
)
from ..core.events import EventBus, Events
from .prompts import build_subagent_prompt
from .protocol import ToolCall, parse_tool_calls, render_tool_result, strip_tool_calls
from .task_manager import DEFAULT_MAX_DEPTH

log = logging.getLogger(__name__)

ChunkFn = Callable[[str], None]
ProgressFn = Callable[..., None]


# --------------------------------------------------------------------------- #
#  Which agent is running on this thread
# --------------------------------------------------------------------------- #
_CURRENT = threading.local()


def current_agent_context() -> dict:
    """The task id / depth / parent of the agent running on *this* thread.

    Empty when the caller is the main agent rather than a subagent.  Tools run
    on the same worker thread as the subagent that invoked them, so this is how
    ``spawn_task`` attributes a new task to its real parent without trusting the
    model to quote an id back correctly.
    """
    return dict(getattr(_CURRENT, "info", None) or {})


@contextlib.contextmanager
def agent_context(**info: Any) -> Iterator[dict]:
    """Publish ``info`` as the current agent context for the duration of a block."""
    previous = getattr(_CURRENT, "info", None)
    _CURRENT.info = dict(info)
    try:
        yield _CURRENT.info
    finally:
        _CURRENT.info = previous


def delegation_note(depth: int, max_depth: int) -> str:
    """The paragraph that tells a subagent how much further it may delegate."""
    remaining = max(0, int(max_depth) - int(depth))
    lines = [
        "Delegation budget: you are a depth-%d agent and the tree may reach "
        "depth %d." % (depth, max_depth)
    ]
    if remaining > 0:
        lines.append(
            "You may call spawn_task to delegate %d further level(s). Do so only "
            "for work that genuinely runs in parallel; your own report is not "
            "finished until the tasks you spawn have finished, and their reports "
            "are folded into yours automatically." % remaining
        )
    else:
        lines.append(
            "You are at the deepest permitted level, so spawn_task will be "
            "refused here. Carry out the work yourself and report what you found."
        )
    return "\n".join(lines)


@dataclass
class AgentTurn:
    """Everything that happened while answering one input."""

    text: str = ""
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    iterations: int = 0
    truncated: bool = False       # hit max_iterations before the model finished
    # Whether `text` was already delivered to `on_chunk` as it was generated.
    # True for the ordinary streamed path; False for the two places `text` is
    # produced OUTSIDE that path -- a non-streamed generate() call, and the
    # truncated-fallback generate() after the iteration limit. A caller
    # driving live speech from `on_chunk` needs this to know whether `text`
    # still needs to be spoken, or was already heard as it streamed.
    streamed: bool = False

    @property
    def used_tools(self) -> list:
        return [c.name for c in self.tool_calls]


def _unknown_tool_message(call: ToolCall, available: Sequence[str]) -> str:
    close = [n for n in available if call.name.lower() in n.lower() or n.lower() in call.name.lower()]
    hint = f" Did you mean: {', '.join(close[:3])}?" if close else ""
    return (
        f"ERROR: there is no tool named '{call.name}'.{hint} "
        f"Use only the tools listed in your instructions."
    )


def _fallback_answer(turn: "AgentTurn") -> str:
    """A spoken reply for when the model produces no usable prose.

    This happens in practice: a small quantised model told to stop calling tools
    will sometimes emit one more tool call anyway, and stripping it leaves an
    empty string.  Returning silence to a *voice* assistant is the worst
    possible failure, so we always say something truthful instead.
    """
    if not turn.tool_calls:
        return "I'm afraid I wasn't able to formulate an answer to that."

    names = ", ".join(dict.fromkeys(turn.used_tools))
    succeeded = sum(1 for r in turn.tool_results if r.ok)
    if succeeded:
        last = next(
            (r for r in reversed(turn.tool_results) if r.ok and r.output not in (None, "")),
            None,
        )
        detail = ""
        if last is not None:
            text = str(last.output).strip().replace("\n", " ")
            if text:
                detail = f" The last thing I found was: {text[:400]}"
        return (
            f"I worked through {names} but reached my step limit before summarising."
            f"{detail}"
        )
    return (
        f"I attempted {names}, but every step failed and I reached my limit. "
        "You may want to check the logs."
    )


def run_agent_loop(
    llm: LLMBackend,
    registry: Any,
    messages: list,
    *,
    max_iterations: int = 8,
    gen_config: Optional[GenerationConfig] = None,
    on_chunk: Optional[ChunkFn] = None,
    progress: Optional[ProgressFn] = None,
    bus: Optional[EventBus] = None,
    stream: bool = False,
) -> AgentTurn:
    """Drive the model until it produces an answer with no tool calls.

    ``messages`` is mutated in place so the caller keeps the full transcript,
    including tool turns.  ``registry`` may be ``None`` for a tool-less agent.
    """
    turn = AgentTurn(messages=messages)
    available = list(registry.names()) if registry is not None else []

    for iteration in range(1, max_iterations + 1):
        turn.iterations = iteration

        if progress is not None:
            progress(f"thinking (step {iteration})", None)

        # -- generate ------------------------------------------------------- #
        try:
            if stream and on_chunk is not None:
                pieces: list = []
                for chunk in llm.stream(messages, gen_config):
                    if chunk:
                        pieces.append(chunk)
                        on_chunk(chunk)
                        if bus is not None:
                            bus.emit(Events.ASSISTANT_CHUNK, chunk)
                raw = "".join(pieces)
            else:
                raw = llm.generate(messages, gen_config).text
        except Exception as exc:  # noqa: BLE001 - surface as a spoken failure
            log.exception("generation failed")
            turn.text = f"I'm afraid the language model failed: {exc}"
            return turn

        calls = parse_tool_calls(raw)

        # -- no tool calls: this is the answer ------------------------------ #
        if not calls:
            prose = strip_tool_calls(raw).strip()
            answer = prose or _fallback_answer(turn)
            messages.append(Message.assistant(answer))
            turn.text = answer
            # `_fallback_answer` synthesises text no chunk ever carried, and
            # the non-streaming branch above never called on_chunk at all --
            # only real prose from the streamed branch was actually heard.
            turn.streamed = bool(prose) and stream and on_chunk is not None
            return turn

        # The model may have prefaced its calls with prose; keep it verbatim so
        # the transcript stays faithful, but never show it as the final answer.
        preamble = strip_tool_calls(raw).strip()
        messages.append(
            Message.assistant(raw, metadata={"tool_calls": [c.name for c in calls]})
        )
        if preamble and progress is not None:
            progress(preamble[:200], None)

        # -- run the tools -------------------------------------------------- #
        for call in calls:
            turn.tool_calls.append(call)

            if registry is None or not registry.has(call.name):
                result = ToolResult.failure(_unknown_tool_message(call, available))
            else:
                if progress is not None:
                    progress(f"running {call.name}", None)
                if bus is not None:
                    bus.emit(Events.TOOL_CALL, {"name": call.name, "arguments": call.arguments})
                try:
                    result = registry.run(call.name, **call.arguments)
                except Exception as exc:  # noqa: BLE001 - registry.run should not raise
                    log.exception("tool %s raised", call.name)
                    result = ToolResult.failure(f"{type(exc).__name__}: {exc}")

            turn.tool_results.append(result)
            if bus is not None:
                bus.emit(Events.TOOL_RESULT, {"name": call.name, "ok": result.ok})

            messages.append(
                Message.tool(
                    render_tool_result(call, result),
                    name=call.name,
                    tool_call_id=call.id,
                )
            )

    # -- ran out of iterations --------------------------------------------- #
    turn.truncated = True
    messages.append(
        Message.user(
            "You have reached the tool-use limit. Give your final answer now, "
            "in plain spoken prose, using only what you have already learned. "
            "Do not call any more tools."
        )
    )
    try:
        final = llm.generate(messages, gen_config).text
        answer = strip_tool_calls(final).strip() or _fallback_answer(turn)
    except Exception as exc:  # noqa: BLE001
        log.exception("final generation failed")
        answer = f"I ran out of steps before finishing, and the model then failed: {exc}"

    messages.append(Message.assistant(answer))
    turn.text = answer
    return turn


class SubAgent:
    """An autonomous worker that pursues one goal and writes a report.

    ``depth`` and ``parent_id`` locate this agent in the task tree.  They are
    fallbacks: when the runner is driven by :class:`~jarvis.agent.task_manager.TaskManager`
    the authoritative values arrive on ``task.metadata``, because the manager is
    what actually assigns depth.
    """

    def __init__(
        self,
        llm: LLMBackend,
        registry: Any,
        *,
        agent_name: str = "JARVIS",
        environment: str = "",
        max_iterations: int = 12,
        gen_config: Optional[GenerationConfig] = None,
        bus: Optional[EventBus] = None,
        depth: int = 0,
        parent_id: Optional[str] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.agent_name = agent_name
        self.environment = environment
        self.max_iterations = max_iterations
        self.gen_config = gen_config
        self.bus = bus
        self.depth = int(depth)
        self.parent_id = parent_id
        self.max_depth = int(max_depth)

    def build_messages(
        self,
        goal: str,
        context: str = "",
        *,
        depth: Optional[int] = None,
    ) -> list:
        """Assemble the subagent's transcript.

        The delegation budget rides in through ``context`` rather than through a
        new prompt parameter, so ``prompts.build_subagent_prompt`` stays as it is.
        """
        catalogue = self.registry.describe() if self.registry is not None else ""
        note = delegation_note(self.depth if depth is None else depth, self.max_depth)
        merged = "\n\n".join(p for p in ((context or "").strip(), note) if p)
        system = build_subagent_prompt(
            goal,
            name=self.agent_name,
            environment=self.environment,
            tools_catalogue=catalogue,
            context=merged,
        )
        return [
            Message.system(system),
            Message.user("Begin. Report back when the goal is met."),
        ]

    def run(self, task: Task, progress: ProgressFn) -> dict:
        """Runner compatible with :meth:`TaskManager.spawn`."""
        context = str(task.metadata.get("context", ""))
        try:
            depth = int(task.metadata.get("depth", self.depth) or 0)
        except (TypeError, ValueError):
            depth = self.depth
        parent_id = task.metadata.get("parent_id", self.parent_id)
        messages = self.build_messages(task.goal, context, depth=depth)

        progress("dispatched at depth %d" % depth, 0.0)
        # Anything this agent calls — spawn_task above all — runs on this thread
        # and can therefore discover which task it belongs to.
        with agent_context(task_id=task.id, depth=depth, parent_id=parent_id):
            turn = run_agent_loop(
                self.llm,
                self.registry,
                messages,
                max_iterations=self.max_iterations,
                gen_config=self.gen_config,
                progress=progress,
                bus=self.bus,
            )
        progress("complete", 1.0)

        # ``children`` is filled in by the TaskManager as the subtree settles;
        # an empty list here keeps the shape stable for a childless agent.
        return {
            "report": turn.text,
            "tools_used": turn.used_tools,
            "iterations": turn.iterations,
            "truncated": turn.truncated,
            "task_id": task.id,
            "depth": depth,
            "parent_id": parent_id,
            "children": [],
        }

    # Convenience for callers that want a blocking run.
    def __call__(self, task: Task, progress: ProgressFn) -> dict:
        return self.run(task, progress)


__all__ = [
    "run_agent_loop",
    "AgentTurn",
    "SubAgent",
    "ChunkFn",
    "ProgressFn",
    "agent_context",
    "current_agent_context",
    "delegation_note",
]
