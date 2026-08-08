"""Tool-call parsing — the part most exposed to a sloppy local model."""

from __future__ import annotations

import pytest

from jarvis.agent.protocol import (
    ToolCall,
    format_tool_call,
    parse_tool_calls,
    render_tool_result,
    strip_tool_calls,
)
from jarvis.core.contracts import ToolResult


def call_of(text: str) -> list:
    return parse_tool_calls(text)


# --------------------------------------------------------------------------- #
#  Well-formed input
# --------------------------------------------------------------------------- #
def test_canonical_form():
    calls = call_of('<tool_call>\n{"name":"read_file","arguments":{"path":"a.txt"}}\n</tool_call>')
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.txt"}
    assert calls[0].id.startswith("call_")


def test_round_trip_through_format_tool_call():
    text = format_tool_call("run_command", {"command": "dir", "timeout": 5})
    (call,) = call_of(text)
    assert call.name == "run_command"
    assert call.arguments == {"command": "dir", "timeout": 5}


def test_nested_object_arguments():
    (call,) = call_of('<tool_call>{"name":"t","arguments":{"a":{"b":[1,2,{"c":3}]}}}</tool_call>')
    assert call.arguments["a"]["b"][2]["c"] == 3


def test_multiple_blocks_and_multiple_objects_in_one_block():
    two_blocks = call_of(
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        ' and then '
        '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    )
    assert [c.name for c in two_blocks] == ["a", "b"]

    one_block = call_of(
        '<tool_call>{"name":"a","arguments":{}}\n{"name":"b","arguments":{}}</tool_call>'
    )
    assert [c.name for c in one_block] == ["a", "b"]


# --------------------------------------------------------------------------- #
#  Malformed input the models really do produce
# --------------------------------------------------------------------------- #
def test_braces_inside_a_string_value_do_not_break_scanning():
    raw = r'<tool_call>{"name":"run_command","arguments":{"command":"echo {x} && dir C:\\Users"}}</tool_call>'
    (call,) = call_of(raw)
    assert call.arguments["command"] == r"echo {x} && dir C:\Users"


def test_unclosed_tag_from_a_truncated_response():
    (call,) = call_of('<tool_call>{"name":"a","arguments":{"p":1}}')
    assert call.name == "a"


def test_code_fence_inside_the_tags():
    (call,) = call_of('<tool_call>\n```json\n{"name":"a","arguments":{"p":1}}\n```\n</tool_call>')
    assert call.arguments == {"p": 1}


def test_bare_fence_without_tags():
    (call,) = call_of('Certainly.\n```json\n{"name":"list_dir","arguments":{"path":"."}}\n```')
    assert call.name == "list_dir"


def test_python_literals_single_quotes_and_trailing_commas():
    (call,) = call_of("<tool_call>{'name': 'a', 'arguments': {'flag': True, 'z': None,}}</tool_call>")
    assert call.arguments == {"flag": True, "z": None}


def test_smart_quotes():
    (call,) = call_of('<tool_call>{\u201cname\u201d: \u201ca\u201d, \u201carguments\u201d: {}}</tool_call>')
    assert call.name == "a"


def test_openai_style_nesting_with_stringified_arguments():
    (call,) = call_of(
        '<tool_call>{"type":"function","function":{"name":"a","arguments":"{\\"p\\":2}"}}</tool_call>'
    )
    assert call.name == "a" and call.arguments == {"p": 2}


def test_parameters_inlined_beside_the_name():
    (call,) = call_of('<tool_call>{"name":"a","path":"x","n":3}</tool_call>')
    assert call.arguments == {"path": "x", "n": 3}


@pytest.mark.parametrize("key", ["arguments", "args", "parameters", "params", "input", "kwargs"])
def test_alternative_argument_keys(key):
    (call,) = call_of('<tool_call>{"name":"a","%s":{"q":1}}</tool_call>' % key)
    assert call.arguments == {"q": 1}


@pytest.mark.parametrize("key", ["name", "tool", "tool_name"])
def test_alternative_name_keys(key):
    (call,) = call_of('<tool_call>{"%s":"a","arguments":{}}</tool_call>' % key)
    assert call.name == "a"


def test_unicode_arguments_survive():
    (call,) = call_of('<tool_call>{"name":"say","arguments":{"text":"Caf\u00e9 \u2014 na\u00efve \u65e5\u672c\u8a9e \U0001f3a9"}}</tool_call>')
    assert call.arguments["text"] == "Caf\u00e9 \u2014 na\u00efve \u65e5\u672c\u8a9e \U0001f3a9"


def test_uppercase_tags():
    (call,) = call_of('<TOOL_CALL>{"name":"a","arguments":{}}</TOOL_CALL>')
    assert call.name == "a"


# --------------------------------------------------------------------------- #
#  Things that must NOT be read as tool calls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "",
    "The disk is forty-three percent full, Sir.",
    "Nothing here but prose and a stray brace }",
    'I considered {"a": 1} but decided against it.',
    '<tool_call>not json at all</tool_call>',
])
def test_no_false_positives(text):
    assert parse_tool_calls(text) == []


def test_a_fenced_object_without_call_shape_is_ignored():
    assert parse_tool_calls('```json\n{"colour": "red", "size": 3}\n```') == []


# --------------------------------------------------------------------------- #
#  Stripping and rendering
# --------------------------------------------------------------------------- #
def test_strip_tool_calls_keeps_surrounding_prose():
    cleaned = strip_tool_calls('Right away.\n<tool_call>{"name":"a","arguments":{}}</tool_call>\nDone.')
    assert "Right away." in cleaned and "Done." in cleaned
    assert "tool_call" not in cleaned and "{" not in cleaned


def test_strip_tool_calls_on_empty_and_pure_call():
    assert strip_tool_calls("") == ""
    assert strip_tool_calls('<tool_call>{"name":"a","arguments":{}}</tool_call>') == ""


def test_render_success_dict_is_pretty_json():
    call = ToolCall("a", {})
    rendered = render_tool_result(call, ToolResult.success({"k": "v"}))
    assert '"k": "v"' in rendered


def test_render_failure_and_none_output():
    call = ToolCall("a", {})
    assert render_tool_result(call, ToolResult.failure("nope")) == "ERROR: nope"
    assert render_tool_result(call, ToolResult.success(None)) == "(no output)"


def test_render_truncates_large_output_and_says_so():
    call = ToolCall("a", {})
    rendered = render_tool_result(call, ToolResult.success("x" * 50_000), limit=1000)
    assert len(rendered) < 2000
    assert "truncated" in rendered


def test_render_artifact_is_labelled():
    call = ToolCall("a", {})
    rendered = render_tool_result(call, ToolResult.success("/tmp/shot.png", is_artifact=True))
    assert rendered.startswith("[artifact]")


def test_render_survives_unserialisable_output():
    call = ToolCall("a", {})
    rendered = render_tool_result(call, ToolResult.success({"obj": object()}))
    assert "obj" in rendered
