"""Shared pytest fixtures.

Every test runs against a throwaway ``JARVIS_HOME`` so the suite can never read
or write the developer's real data directory, memory database, or generated
tools.  This is enforced session-wide rather than per-test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import pytest

# Make the package importable when the suite is run from a checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.core.contracts import (  # noqa: E402
    GenerationConfig,
    LLMBackend,
    LLMResult,
    Message,
    ToolResult,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every JARVIS path helper at a per-test temporary directory.

    Deliberately *not* named ``jarvis_home``: individual test modules define
    their own home fixture at that path, and two fixtures racing to create the
    same directory is a pointless source of failures.
    """
    home = tmp_path / "jarvis_data"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JARVIS_HOME", str(home))
    monkeypatch.setenv("JARVIS_CONFIG_DIR", str(home / "config"))
    # A stray real config file must not leak into the tests.
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    yield home


@pytest.fixture
def config(isolated_home: Path):
    """A default Config rooted in the isolated home."""
    from jarvis.core.config import load_config

    cfg = load_config(use_env=False)
    cfg.data_dir = str(isolated_home)
    cfg.tts.enabled = False
    return cfg


@pytest.fixture
def security(config):
    from jarvis.core.security import SecurityGate

    return SecurityGate(config.security, audit_path=config.logs_dir() / "audit.jsonl")


@pytest.fixture
def bus():
    from jarvis.core.events import EventBus

    return EventBus()


class ScriptedLLM(LLMBackend):
    """A deterministic LLM whose replies are supplied by the test.

    Once the script is exhausted it returns ``default`` forever, so a test that
    under-specifies the script fails on an assertion rather than an IndexError.
    """

    name = "scripted"

    def __init__(
        self,
        script: Optional[Sequence[str]] = None,
        default: str = "Understood, Sir.",
    ) -> None:
        self.script = list(script or [])
        self.default = default
        self.calls: list = []
        self.configs: list = []
        self.loaded = False

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def generate(self, messages, config: Optional[GenerationConfig] = None) -> LLMResult:
        self.calls.append(list(messages))
        self.configs.append(config)
        text = self.script.pop(0) if self.script else self.default
        return LLMResult(text=text, completion_tokens=len(text) // 4)

    def stream(self, messages, config=None):
        yield self.generate(messages, config).text

    @property
    def last_prompt(self) -> str:
        """The full text of the most recent prompt, for content assertions."""
        if not self.calls:
            return ""
        return "\n".join(m.content for m in self.calls[-1])


class FakeRegistry:
    """A minimal ToolRegistry stand-in for agent-loop tests."""

    def __init__(self, results: Optional[dict] = None) -> None:
        self.results = results or {}
        self.calls: list = []

    def names(self) -> list:
        return list(self.results.keys())

    def has(self, name: str) -> bool:
        return name in self.results

    def describe(self) -> str:
        return "\n".join(f"- {n}" for n in self.results)

    def register_function(self, fn, **kwargs):  # pragma: no cover - used by Orchestrator
        self.results[getattr(fn, "__name__", "fn")] = ToolResult.success("ok")
        return fn

    def run(self, name: str, **kwargs) -> ToolResult:
        self.calls.append((name, kwargs))
        value = self.results.get(name)
        if callable(value):
            return value(**kwargs)
        if isinstance(value, ToolResult):
            return value
        return ToolResult.success(value)


@pytest.fixture
def scripted_llm():
    return ScriptedLLM


@pytest.fixture
def fake_registry():
    return FakeRegistry


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: takes more than a second")
    config.addinivalue_line("markers", "requires_audio: needs a real audio device")
