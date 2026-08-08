"""Parsing the model's tool-call syntax.

Local models are not reliably well-behaved: they wrap tool calls in code fences,
emit single quotes, forget the closing tag, stream a call that is cut off by a
token limit, or produce two calls in one block.  This module is deliberately
forgiving about all of that while refusing to guess when a call is genuinely
unparseable.

The canonical form (which Qwen3 emits natively) is::

    <tool_call>
    {"name": "read_file", "arguments": {"path": "C:/notes.txt"}}
    </tool_call>
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.contracts import ToolResult, new_id

log = logging.getLogger(__name__)


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# The closing tag is optional so that a response truncated by max_new_tokens
# still yields its (complete) leading calls.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*(?:</tool_call>|$)",
    re.DOTALL | re.IGNORECASE,
)

# Some models wrap the JSON in a fence inside the tags, or use a bare fence
# labelled tool_call / json when they forget the tags entirely.
_FENCE_RE = re.compile(
    r"```(?:json|tool_call|tool|python)?\s*(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ToolCall:
    """A single parsed request to run a tool."""

    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id("call")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolCall({self.name}, {self.arguments!r})"


# --------------------------------------------------------------------------- #
#  JSON recovery
# --------------------------------------------------------------------------- #
def _balanced_objects(text: str) -> list:
    """Extract every top-level ``{...}`` region, respecting strings and escapes.

    A regex cannot do this correctly once arguments contain nested objects or
    braces inside string values (Windows paths and shell commands routinely do),
    so we scan.
    """
    objects: list = []
    depth = 0
    start = -1
    in_string = False
    quote = ""
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue

        if char in ('"', "'"):
            in_string = True
            quote = char
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start : index + 1])
                    start = -1
    return objects


def _tidy(text: str) -> str:
    """Fixes that are safe for BOTH JSON and Python literal syntax.

    Smart quotes (models love them) and trailing commas only; nothing here may
    change the meaning of an already-valid payload in either language.
    """
    tidied = text.strip()
    tidied = (
        tidied.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )
    return re.sub(r",\s*([}\]])", r"\1", tidied)


def _jsonize(text: str) -> str:
    """Python literals -> JSON literals.  Only valid as a *JSON* repair.

    Applied after :func:`_tidy` and never before the ``literal_eval`` fallback,
    since ``true``/``null`` are not Python literals — doing this first would
    break single-quoted payloads that ``literal_eval`` could otherwise read.
    """
    jsonized = re.sub(r"\bTrue\b", "true", text)
    jsonized = re.sub(r"\bFalse\b", "false", jsonized)
    return re.sub(r"\bNone\b", "null", jsonized)


def _loads(text: str) -> Optional[dict]:
    """Parse a JSON object, escalating through progressively looser readers."""
    tidied = _tidy(text)
    for candidate in (text, tidied, _jsonize(tidied)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, RecursionError):
            continue
        if isinstance(value, dict):
            return value

    # Last resort: a Python-dict-shaped payload (single quotes, True/False/None).
    try:
        import ast

        value = ast.literal_eval(tidied)
        if isinstance(value, dict):
            return value
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        pass
    return None


def _normalise_call(payload: dict, raw: str) -> Optional[ToolCall]:
    """Accept the several shapes models use for the same idea."""
    if not isinstance(payload, dict):
        return None

    # OpenAI-style nesting: {"type": "function", "function": {...}}
    if "function" in payload and isinstance(payload["function"], dict):
        inner = dict(payload["function"])
        inner.setdefault("id", payload.get("id", ""))
        payload = inner

    name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
    if not name or not isinstance(name, str):
        return None

    args: Any = None
    for key in ("arguments", "args", "parameters", "params", "input", "kwargs"):
        if key in payload:
            args = payload[key]
            break

    if args is None:
        # Some models inline the parameters alongside the name.
        args = {
            k: v
            for k, v in payload.items()
            if k not in ("name", "tool", "tool_name", "id", "type")
        }

    # Some APIs deliver `arguments` as a JSON *string* rather than an object.
    if isinstance(args, str):
        parsed = _loads(args)
        args = parsed if parsed is not None else {"value": args}
    if not isinstance(args, dict):
        args = {"value": args}

    call_id = payload.get("id")
    return ToolCall(
        name=name.strip(),
        arguments=args,
        id=str(call_id) if call_id else "",
        raw=raw,
    )


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def parse_tool_calls(text: str) -> list:
    """Extract every tool call from a model response.

    Returns an empty list when the response contains no tool call, which is how
    the agent loop detects that the model has produced its final answer.
    """
    if not text or "{" not in text:
        return []

    calls: list = []

    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group("body") or ""
        # Strip a fence the model may have put inside the tags.
        fence = _FENCE_RE.search(body)
        if fence:
            body = fence.group("body")
        for blob in _balanced_objects(body) or [body]:
            payload = _loads(blob)
            if payload is None:
                log.debug("unparseable tool_call body: %r", blob[:200])
                continue
            call = _normalise_call(payload, blob)
            if call:
                calls.append(call)

    if calls:
        return calls

    # No tagged calls parsed — try fenced blocks that look like a tool call.
    for fence in _FENCE_RE.finditer(text):
        body = fence.group("body")
        if '"name"' not in body and "'name'" not in body:
            continue
        for blob in _balanced_objects(body):
            payload = _loads(blob)
            if payload is None:
                continue
            call = _normalise_call(payload, blob)
            # Only accept a fenced object that really looks like a call.
            if call and ("arguments" in payload or "args" in payload
                         or "parameters" in payload or "tool" in payload):
                calls.append(call)

    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call markup, leaving whatever prose surrounded it."""
    if not text:
        return ""
    cleaned = _TOOL_CALL_RE.sub("", text)
    cleaned = cleaned.replace(TOOL_CALL_OPEN, "").replace(TOOL_CALL_CLOSE, "")
    return cleaned.strip()


def render_tool_result(call: ToolCall, result: ToolResult, *, limit: int = 4000) -> str:
    """Format a tool result for feeding back into the transcript."""
    if result.ok:
        output = result.output
        if isinstance(output, (dict, list)):
            try:
                body = json.dumps(output, indent=2, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                body = str(output)
        elif output is None:
            body = "(no output)"
        else:
            body = str(output)
        if result.is_artifact:
            body = f"[artifact] {body}"
    else:
        body = f"ERROR: {result.error}"

    if len(body) > limit:
        head = body[: int(limit * 0.7)]
        tail = body[-int(limit * 0.2):]
        body = f"{head}\n... [{len(body) - limit} characters truncated] ...\n{tail}"
    return body


def format_tool_call(name: str, arguments: dict) -> str:
    """Produce the canonical wire form — used in prompts and tests."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


__all__ = [
    "ToolCall",
    "parse_tool_calls",
    "strip_tool_calls",
    "render_tool_result",
    "format_tool_call",
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
]
