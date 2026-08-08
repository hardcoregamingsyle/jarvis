"""The thinking layer: persona, tool-calling loop, subagents and task management."""

from .prompts import (
    JARVIS_PERSONA,
    build_system_prompt,
    build_subagent_prompt,
    SUMMARIZE_PROMPT,
)
from .protocol import ToolCall, parse_tool_calls, strip_tool_calls, render_tool_result
from .task_manager import TaskManager
from .subagent import SubAgent
from .orchestrator import Orchestrator

__all__ = [
    "JARVIS_PERSONA",
    "build_system_prompt",
    "build_subagent_prompt",
    "SUMMARIZE_PROMPT",
    "ToolCall",
    "parse_tool_calls",
    "strip_tool_calls",
    "render_tool_result",
    "TaskManager",
    "SubAgent",
    "Orchestrator",
]
