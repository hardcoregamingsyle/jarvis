"""Bootstrap: assemble a working JARVIS from a :class:`Config`.

Every subsystem is optional at runtime.  If the LLM weights are absent, or no
microphone exists, or the embedding model was never downloaded, JARVIS still
starts — degraded, loudly logged, but functional.  That property is deliberate:
a voice assistant that refuses to boot because one optional package is missing
is useless on the day you need it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .agent.orchestrator import Orchestrator
from .core.config import Config, load_config
from .core.events import EventBus, get_bus
from .core.logging_setup import setup_logging
from .core.platform_utils import system_summary
from .core.security import SecurityGate

log = logging.getLogger(__name__)


@dataclass
class Subsystems:
    """Handles to everything JARVIS is made of, for inspection and teardown."""

    config: Config
    bus: EventBus
    security: SecurityGate
    llm: Any = None
    memory: Any = None
    context: Any = None
    registry: Any = None
    stt: Any = None
    tts: Any = None
    speech_queue: Any = None
    orchestrator: Optional[Orchestrator] = None

    def status(self) -> dict:
        """A one-glance report of what actually came up."""

        def describe(obj: Any) -> str:
            if obj is None:
                return "unavailable"
            return getattr(obj, "name", type(obj).__name__)

        status = {
            "llm": describe(self.llm),
            "model": self.config.llm.model,
            "stt": describe(self.stt),
            "tts": describe(self.tts),
            "memory": "sqlite" if self.memory is not None else "unavailable",
            "tools": len(self.registry.names()) if self.registry is not None else 0,
            "security_mode": self.config.security.mode,
            "data_dir": str(self.config.home()),
        }
        # Surface a silent memory-write failure; the "forgets nothing" promise
        # is worthless if a broken store looks identical to a healthy one.
        healthy = getattr(self.context, "persistence_healthy", True)
        if healthy is False:
            status["memory"] = "sqlite (WRITE FAILURES - see log)"
        return status


def _build_llm(cfg: Config) -> Any:
    try:
        from .llm import create_llm

        backend = create_llm(cfg.llm)
        log.info("LLM backend: %s (%s)", getattr(backend, "name", "?"), cfg.llm.model)
        return backend
    except Exception:  # noqa: BLE001
        log.exception("could not create an LLM backend; conversation will not work")
        return None


def _build_memory(cfg: Config):
    try:
        from .memory import create_context, create_memory

        # Config.db_file() is the single authority on where memory lives; pin it
        # so an explicit `data_dir` in the config always wins over the platform
        # default that create_memory() would otherwise pick.
        cfg.memory.db_path = str(cfg.db_file())
        store = create_memory(cfg.memory)
        context = create_context(cfg.memory, store=store)
        log.info("memory store ready at %s", cfg.db_file())
        return store, context
    except Exception:  # noqa: BLE001
        log.exception("memory subsystem failed to start")
        return None, None


def _build_tools(cfg: Config, security: SecurityGate, bus: EventBus, memory: Any) -> Any:
    try:
        from .tools import create_registry

        registry = create_registry(cfg, security, bus=bus, memory=memory)
        count = registry.load_generated()
        if count:
            log.info("loaded %d previously generated tool module(s)", count)
        log.info("%d tools registered", len(registry.names()))
        return registry
    except Exception:  # noqa: BLE001
        log.exception("tool registry failed to start; JARVIS will be conversation-only")
        return None


def _build_speech(cfg: Config):
    stt = tts = queue = None
    try:
        from .speech import create_stt

        stt = create_stt(cfg.stt)
        log.info("speech-to-text: %s", getattr(stt, "name", "?"))
    except Exception:  # noqa: BLE001
        log.exception("speech-to-text unavailable")

    try:
        from .speech.tts import SpeechQueue, create_tts

        tts = create_tts(cfg.tts, voices_dir=cfg.voices_dir())
        log.info("text-to-speech: %s (%s)", getattr(tts, "name", "?"), cfg.tts.piper_voice)
        queue = SpeechQueue(tts)
    except Exception:  # noqa: BLE001
        log.exception("text-to-speech unavailable")

    return stt, tts, queue


def build(
    config: Optional[Config] = None,
    *,
    bus: Optional[EventBus] = None,
    configure_logging: bool = True,
) -> Subsystems:
    """Construct every subsystem and wire up the orchestrator."""
    cfg = config or load_config()
    if configure_logging:
        setup_logging(cfg.log_level, log_file=cfg.logs_dir() / "jarvis.log")

    log.info("starting %s on %s", cfg.agent.name, system_summary().get("platform", "?"))
    bus = bus or get_bus()
    security = SecurityGate(cfg.security, audit_path=cfg.logs_dir() / "audit.jsonl")

    llm = _build_llm(cfg)
    store, context = _build_memory(cfg)
    registry = _build_tools(cfg, security, bus, store)
    stt, tts, queue = _build_speech(cfg)

    orchestrator = None
    if llm is not None and context is not None:
        orchestrator = Orchestrator(
            cfg, llm, registry, context, tts=queue or tts, bus=bus
        )
    else:
        log.error("orchestrator not created: LLM=%s memory=%s", llm, context)

    subsystems = Subsystems(
        config=cfg,
        bus=bus,
        security=security,
        llm=llm,
        memory=store,
        context=context,
        registry=registry,
        stt=stt,
        tts=tts,
        speech_queue=queue,
        orchestrator=orchestrator,
    )
    log.info("subsystem status: %s", subsystems.status())
    return subsystems


def shutdown(subsystems: Subsystems) -> None:
    """Tear everything down in dependency order.  Never raises."""
    for name, obj, methods in (
        ("orchestrator", subsystems.orchestrator, ("shutdown",)),
        ("speech queue", subsystems.speech_queue, ("shutdown", "stop")),
        ("llm", subsystems.llm, ("unload",)),
        ("memory", subsystems.memory, ("close",)),
    ):
        if obj is None:
            continue
        for method in methods:
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.debug("%s.%s() failed during shutdown", name, method, exc_info=True)
                break


__all__ = ["Subsystems", "build", "shutdown"]
