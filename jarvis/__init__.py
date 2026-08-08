"""JARVIS — a local, voice-driven personal assistant.

A cross-platform (Windows/Linux) assistant built around:
  * a pluggable LLM backend (AirLLM / Qwen3 by default, Ollama fallback),
  * offline speech-to-text and text-to-speech (polished British voice),
  * a persistent "dynamic context" memory that never forgets,
  * a subagent/task system so the main agent stays responsive, and
  * a self-extending tool system that can author new tools on demand.

Every heavy dependency (torch, airllm, faster-whisper, piper, ...) is imported
lazily inside the module that needs it, so the core package imports cleanly on
any machine and the test-suite runs without GPUs or multi-gigabyte models.
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
