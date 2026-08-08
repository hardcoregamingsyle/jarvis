"""Shared data contracts and abstract interfaces for the whole system.

Everything in JARVIS talks through the types defined here.  Leaf modules
(LLM backends, speech engines, memory stores, tools) implement the abstract
base classes; the orchestrator depends only on these interfaces, never on a
concrete implementation.  Keeping the contracts in one dependency-free module
is what lets the pieces be developed and tested independently.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


# --------------------------------------------------------------------------- #
#  Identifiers & small helpers
# --------------------------------------------------------------------------- #
def new_id(prefix: str = "id") -> str:
    """A short, sortable-ish unique id.  (uuid4 hex, prefixed.)"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ts() -> float:
    """Wall-clock seconds.  Wrapped so tests can monkeypatch a single symbol."""
    return time.time()


# --------------------------------------------------------------------------- #
#  Conversation messages
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single conversational turn."""

    role: Role
    content: str
    name: Optional[str] = None          # tool name, when role == TOOL
    tool_call_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    ts: float = field(default_factory=now_ts)

    def to_dict(self) -> dict:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @staticmethod
    def system(content: str, **kw: Any) -> "Message":
        return Message(Role.SYSTEM, content, **kw)

    @staticmethod
    def user(content: str, **kw: Any) -> "Message":
        return Message(Role.USER, content, **kw)

    @staticmethod
    def assistant(content: str, **kw: Any) -> "Message":
        return Message(Role.ASSISTANT, content, **kw)

    @staticmethod
    def tool(content: str, name: str, tool_call_id: Optional[str] = None, **kw: Any) -> "Message":
        return Message(Role.TOOL, content, name=name, tool_call_id=tool_call_id, **kw)


# --------------------------------------------------------------------------- #
#  LLM backend
# --------------------------------------------------------------------------- #
@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop: Sequence[str] = field(default_factory=tuple)
    seed: Optional[int] = None


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: Any = None


class LLMBackend(abc.ABC):
    """A text-generation backend (AirLLM, Ollama, a stub for tests, ...)."""

    name: str = "llm"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Cheap check: are the deps installed and the model reachable?"""

    @abc.abstractmethod
    def load(self) -> None:
        """Eagerly load weights / open a connection.  Idempotent."""

    @abc.abstractmethod
    def generate(self, messages: Sequence[Message], config: Optional[GenerationConfig] = None) -> LLMResult:
        """Generate a completion for a chat transcript."""

    def stream(
        self, messages: Sequence[Message], config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Yield text chunks.  Default: fall back to a single non-streamed call."""
        yield self.generate(messages, config).text

    def unload(self) -> None:
        """Release resources.  Optional."""


# --------------------------------------------------------------------------- #
#  Speech
# --------------------------------------------------------------------------- #
@dataclass
class Transcript:
    text: str
    language: Optional[str] = None
    confidence: float = 1.0
    segments: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


class STTEngine(abc.ABC):
    """Speech-to-text."""

    name: str = "stt"

    @abc.abstractmethod
    def is_available(self) -> bool: ...

    @abc.abstractmethod
    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        """Transcribe a numpy float32 mono buffer (or a path to a wav file)."""

    def transcribe_file(self, path: str) -> Transcript:
        return self.transcribe(path)


class TTSEngine(abc.ABC):
    """Text-to-speech.  The default voice is a polished British accent."""

    name: str = "tts"

    @abc.abstractmethod
    def is_available(self) -> bool: ...

    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return WAV bytes (16-bit PCM) for the given text."""

    def speak(self, text: str) -> None:
        """Synthesize and play through the default output device."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Memory
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    id: str
    kind: str                      # "conversation" | "fact" | "task" | "file" | ...
    text: str
    metadata: dict = field(default_factory=dict)
    ts: float = field(default_factory=now_ts)
    score: float = 0.0             # populated by search()

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id("mem")


class Embedder(abc.ABC):
    """Turns text into vectors for semantic recall."""

    dim: int = 0

    @abc.abstractmethod
    def is_available(self) -> bool: ...

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> list: ...

    def embed_one(self, text: str) -> list:
        return self.embed([text])[0]


class MemoryStore(abc.ABC):
    """Durable, searchable long-term memory — the "never forgets" store."""

    @abc.abstractmethod
    def add(self, record: MemoryRecord) -> str: ...

    @abc.abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]: ...

    @abc.abstractmethod
    def search(
        self, query: str, *, k: int = 8, kind: Optional[str] = None
    ) -> list:
        """Semantic + recency search."""

    @abc.abstractmethod
    def recent(self, *, k: int = 20, kind: Optional[str] = None) -> list: ...

    @abc.abstractmethod
    def all(self, *, kind: Optional[str] = None) -> Iterable: ...

    def close(self) -> None:  # optional
        ...


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #
@dataclass
class ToolParam:
    name: str
    type: str = "string"           # json-schema-ish: string|number|integer|boolean|object|array
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[Sequence[Any]] = None


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Sequence[ToolParam] = field(default_factory=tuple)
    # Free-form risk hints used by the permission layer.
    dangerous: bool = False        # may modify system / run code / spend money

    def json_schema(self) -> dict:
        props: dict = {}
        required: list = []
        for p in self.params:
            entry: dict = {"type": p.type, "description": p.description}
            if p.enum:
                entry["enum"] = list(p.enum)
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props, "required": required},
        }


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: Optional[str] = None
    # When True, the output is large/binary and should be summarised, not spoken.
    is_artifact: bool = False

    @staticmethod
    def success(output: Any = None, **kw: Any) -> "ToolResult":
        return ToolResult(True, output=output, **kw)

    @staticmethod
    def failure(error: str, **kw: Any) -> "ToolResult":
        return ToolResult(False, error=error, **kw)


class Tool(abc.ABC):
    """A callable capability the agent can invoke."""

    @property
    @abc.abstractmethod
    def spec(self) -> ToolSpec: ...

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...

    # Convenience -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name


# --------------------------------------------------------------------------- #
#  Tasks / subagents
# --------------------------------------------------------------------------- #
class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskUpdate:
    task_id: str
    state: TaskState
    message: str = ""
    progress: Optional[float] = None      # 0..1
    result: Any = None
    error: Optional[str] = None
    ts: float = field(default_factory=now_ts)


@dataclass
class Task:
    id: str
    goal: str
    state: TaskState = TaskState.PENDING
    created: float = field(default_factory=now_ts)
    updated: float = field(default_factory=now_ts)
    result: Any = None
    error: Optional[str] = None
    updates: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id("task")


# Callback signatures used across the system.
ProgressFn = Callable[[TaskUpdate], None]
EventHandler = Callable[[str, Any], Any]


__all__ = [
    "new_id", "now_ts",
    "Role", "Message",
    "GenerationConfig", "LLMResult", "LLMBackend",
    "Transcript", "STTEngine", "TTSEngine",
    "MemoryRecord", "Embedder", "MemoryStore",
    "ToolParam", "ToolSpec", "ToolResult", "Tool",
    "TaskState", "TaskUpdate", "Task",
    "ProgressFn", "EventHandler",
]
