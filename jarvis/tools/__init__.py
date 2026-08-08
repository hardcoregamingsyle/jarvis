"""Tool subsystem: registry, built-in tools, generated-tools loader."""

from __future__ import annotations

from typing import Any, Optional

from .registry import (
    FunctionTool,
    GENERATED_MODULE_PREFIX,
    ToolContext,
    ToolRegistry,
    generated_module_name,
    safe_truncate,
    tool,
)


def create_registry(
    config: Any,
    security: Any,
    *,
    bus: Any = None,
    memory: Any = None,
    load_builtin: bool = True,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` wired to the given services."""
    ctx = ToolContext(config=config, security=security, bus=bus, memory=memory)
    registry = ToolRegistry(ctx)
    if load_builtin:
        registry.load_builtin()
    return registry


__all__ = [
    "ToolRegistry",
    "ToolContext",
    "FunctionTool",
    "tool",
    "create_registry",
    "safe_truncate",
    "GENERATED_MODULE_PREFIX",
    "generated_module_name",
]
