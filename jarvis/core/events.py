"""A tiny thread-safe publish/subscribe bus.

The orchestrator, the task manager, the voice loop and the UI all communicate
through this bus rather than holding references to each other.  Handlers may be
plain callables or coroutine functions; both are supported from sync and async
call sites.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# Well-known channel names (string constants keep the bus loosely coupled).
class Events:
    USER_UTTERANCE = "user.utterance"
    ASSISTANT_REPLY = "assistant.reply"
    ASSISTANT_CHUNK = "assistant.chunk"
    TASK_CREATED = "task.created"
    TASK_UPDATE = "task.update"
    TASK_DONE = "task.done"
    TASK_FAILED = "task.failed"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_CREATED = "tool.created"
    SPEAK = "speak"
    LISTEN_START = "listen.start"
    LISTEN_STOP = "listen.stop"
    WAKE = "wake"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class EventBus:
    """Synchronous-first pub/sub with optional async handlers."""

    def __init__(self) -> None:
        self._handlers: dict = defaultdict(list)
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- wiring ------------------------------------------------------------- #
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the main event loop so sync ``emit`` can schedule coroutines."""
        self._loop = loop

    def subscribe(self, channel: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """Register ``handler`` for ``channel``; returns an unsubscribe callable."""
        with self._lock:
            self._handlers[channel].append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(channel, handler)

        return _unsubscribe

    # Alias that reads well as a decorator: ``@bus.on("task.done")``
    def on(self, channel: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.subscribe(channel, fn)
            return fn

        return decorator

    def unsubscribe(self, channel: str, handler: Callable[..., Any]) -> None:
        with self._lock:
            if handler in self._handlers.get(channel, []):
                self._handlers[channel].remove(handler)

    def clear(self, channel: Optional[str] = None) -> None:
        with self._lock:
            if channel is None:
                self._handlers.clear()
            else:
                self._handlers.pop(channel, None)

    def handlers(self, channel: str) -> list:
        with self._lock:
            return list(self._handlers.get(channel, []))

    # -- dispatch ----------------------------------------------------------- #
    def emit(self, channel: str, payload: Any = None) -> None:
        """Fire-and-forget publish.  Never raises into the caller."""
        for handler in self.handlers(channel):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    self._schedule(result)
            except Exception:  # noqa: BLE001 - a bad handler must not kill the emitter
                log.exception("event handler for %r failed", channel)

    async def emit_async(self, channel: str, payload: Any = None) -> None:
        """Publish and await any coroutine handlers."""
        for handler in self.handlers(channel):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                log.exception("async event handler for %r failed", channel)

    def _schedule(self, coro: Any) -> None:
        """Run a coroutine returned by a handler on the bound/running loop."""
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(_wrap(coro), loop)
        else:
            # No loop available — run it to completion synchronously.
            try:
                asyncio.run(_wrap(coro))
            except Exception:  # noqa: BLE001
                log.exception("could not run async event handler")


async def _wrap(coro: Any) -> None:
    try:
        await coro
    except Exception:  # noqa: BLE001
        log.exception("async event handler raised")


# A process-wide default bus, for convenience.
_default_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_bus() -> None:
    """Test helper: drop the process-wide bus."""
    global _default_bus
    _default_bus = None


__all__ = ["EventBus", "Events", "get_bus", "reset_bus"]
