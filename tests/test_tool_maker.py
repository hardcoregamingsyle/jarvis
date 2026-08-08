"""Tests for :mod:`jarvis.tools.tool_maker`.

Nothing here talks to a real LLM: a fake LLM object is supplied when the
generate-validate-retry loop is under test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pytest

from jarvis.core.contracts import Tool, ToolResult
from jarvis.tools.registry import ToolContext, generated_module_name
from jarvis.tools.tool_maker import (
    TOOL_TEMPLATE,
    delete_generated,
    list_generated,
    make_tool,
    sanitize_name,
    validate_tool_source,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
class _Config:
    def __init__(self, tools_dir: Path) -> None:
        self._dir = tools_dir

    def tools_dir(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir


class _Security:
    """Real-shape security object (has `.cfg.allow_network`)."""

    class _Cfg:
        allow_network = True

    cfg = _Cfg()


class _DuckSecurity:
    """Duck-typed security: no `.cfg` attribute at all."""

    # Deliberately no cfg.
    def __init__(self):
        pass


@dataclass
class _FakeLLMResult:
    text: str


@dataclass
class _FakeLLM:
    """A canned-reply LLM."""

    replies: List[str] = field(default_factory=list)
    calls: List[Any] = field(default_factory=list)

    def generate(self, messages, config=None) -> _FakeLLMResult:
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else ""
        return _FakeLLMResult(text=text)


def _ctx(tmp_path: Path, security=None, extra=None) -> ToolContext:
    tools_dir = tmp_path / "gen"
    return ToolContext(
        config=_Config(tools_dir),
        security=security or _Security(),
        extra=extra or {},
    )


VALID_SOURCE = '''\
"""A hand-written valid tool module."""

from __future__ import annotations

from typing import Any

from jarvis.core.contracts import ToolResult
from jarvis.tools.registry import FunctionTool


def _do_something(**kwargs: Any) -> ToolResult:
    """Do the thing."""
    return ToolResult.success(output={"ran": True})


def build_tools(ctx: Any) -> list:
    return [FunctionTool(fn=_do_something, name="do_something",
                         description="Runs the thing.")]
'''


# --------------------------------------------------------------------------- #
#  sanitize_name
# --------------------------------------------------------------------------- #
class TestSanitizeName:

    @pytest.mark.parametrize("name", [
        "good_name", "helloWorld", "tool1", "myCustomTool",
    ])
    def test_accepts_plain_identifiers(self, name):
        assert sanitize_name(name) == name

    @pytest.mark.parametrize("name", [
        "", "  ", "..", "../etc/passwd", "foo/bar", "foo\\bar",
        "foo:bar", "foo\x00", "1starts_with_digit",
    ])
    def test_rejects_bad_shapes(self, name):
        assert sanitize_name(name) is None

    @pytest.mark.parametrize("name", ["for", "def", "class", "return", "if", "import"])
    def test_rejects_keywords(self, name):
        assert sanitize_name(name) is None

    def test_rejects_leading_underscore(self):
        assert sanitize_name("_hidden") is None

    def test_rejects_dunder(self):
        assert sanitize_name("__init__") is None

    @pytest.mark.parametrize("name", ["sys", "os", "re", "json", "subprocess",
                                       "importlib", "registry", "init", "main"])
    def test_rejects_reserved(self, name):
        assert sanitize_name(name) is None


# --------------------------------------------------------------------------- #
#  validate_tool_source
# --------------------------------------------------------------------------- #
class TestValidateSource:

    def test_template_passes(self):
        source = TOOL_TEMPLATE.format(
            name="hello", description="says hi", requirement="greet the world"
        )
        ok, problems = validate_tool_source(source)
        assert ok is True, problems
        assert problems == []

    def test_valid_source_passes(self):
        ok, problems = validate_tool_source(VALID_SOURCE)
        assert ok is True, problems

    def test_syntax_error_rejected(self):
        ok, problems = validate_tool_source("def build_tools(ctx: this is not valid")
        assert ok is False
        assert any("syntax" in p.lower() for p in problems)

    def test_missing_build_tools_rejected(self):
        src = "from jarvis.core.contracts import ToolResult\n\n" \
              "def other(): return ToolResult.success()\n"
        ok, problems = validate_tool_source(src)
        assert ok is False
        assert any("build_tools" in p for p in problems)

    @pytest.mark.parametrize("snippet, needle", [
        ("import os\ndef build_tools(ctx):\n    os.system('rm -rf /')\n    return []\n",
         "os.system"),
        ("import os\ndef build_tools(ctx):\n    os.popen('ls')\n    return []\n",
         "os.popen"),
        ("import os\ndef build_tools(ctx):\n    os.execv('/bin/sh', ['sh'])\n    return []\n",
         "os.exec"),
        ("import subprocess\ndef build_tools(ctx):\n"
         "    subprocess.run('ls', shell=True)\n    return []\n",
         "shell=True"),
        ("def build_tools(ctx):\n    eval('1+1')\n    return []\n", "eval"),
        ("def build_tools(ctx):\n    exec('x=1')\n    return []\n", "exec"),
        ("def build_tools(ctx):\n    compile('x=1', '', 'exec')\n    return []\n", "compile"),
        ("def build_tools(ctx):\n    __import__('os')\n    return []\n", "__import__"),
        ("import ctypes\ndef build_tools(ctx):\n    return []\n", "ctypes"),
        ("import shutil\ndef build_tools(ctx):\n    shutil.rmtree('/tmp/x')\n    return []\n",
         "rmtree"),
        ("def build_tools(ctx):\n    open('/etc/passwd', 'w').write('x')\n    return []\n",
         "system path"),
    ])
    def test_rejects_dangerous(self, snippet, needle):
        ok, problems = validate_tool_source(snippet)
        assert ok is False, snippet
        assert any(needle.lower() in p.lower() for p in problems), (
            f"Expected a problem mentioning {needle!r}, got {problems}"
        )


# --------------------------------------------------------------------------- #
#  make_tool with an explicit source
# --------------------------------------------------------------------------- #
class TestMakeToolExplicitSource:

    def teardown_method(self, method):
        # Evict any generated module the test may have left behind.
        for key in list(sys.modules):
            if key.startswith("jarvis_generated."):
                sys.modules.pop(key, None)

    def test_writes_registers_and_imports(self, tmp_path):
        ctx = _ctx(tmp_path)
        result = make_tool(
            ctx,
            name="do_something",
            description="A test tool",
            requirement="run",
            source=VALID_SOURCE,
        )
        assert result.ok is True, result.error
        target = Path(ctx.config.tools_dir()) / "do_something.py"
        assert target.exists()
        assert "do_something" in result.output["tools"]

    def test_module_registered_under_registry_name(self, tmp_path):
        ctx = _ctx(tmp_path)
        make_tool(ctx, name="reg_check", description="d", requirement="r",
                  source=VALID_SOURCE.replace("do_something", "reg_check"))
        expected = generated_module_name("reg_check")
        assert expected in sys.modules, (
            f"module should be registered as {expected!r}; "
            f"got jarvis_generated keys: "
            f"{[k for k in sys.modules if k.startswith('jarvis_generated')]}"
        )
        assert result_output_module_matches(expected)

    def test_bad_source_cleans_up(self, tmp_path):
        ctx = _ctx(tmp_path)
        result = make_tool(
            ctx,
            name="bad_tool",
            description="bad",
            requirement="r",
            source="import os\nos.system('rm -rf /')\n"
                   "def build_tools(ctx): return []\n",
        )
        assert result.ok is False
        target = Path(ctx.config.tools_dir()) / "bad_tool.py"
        assert not target.exists(), "invalid source must not remain on disk"
        assert generated_module_name("bad_tool") not in sys.modules

    def test_overwrite_protection(self, tmp_path):
        ctx = _ctx(tmp_path)
        first = make_tool(ctx, name="overtool", description="d", requirement="r",
                          source=VALID_SOURCE.replace("do_something", "overtool"))
        assert first.ok is True
        # Second call without overwrite must refuse.
        second = make_tool(ctx, name="overtool", description="d2", requirement="r2",
                           source=VALID_SOURCE.replace("do_something", "overtool"))
        assert second.ok is False
        assert "exists" in (second.error or "").lower()
        # With overwrite, it must succeed.
        third = make_tool(ctx, name="overtool", description="d3", requirement="r3",
                          source=VALID_SOURCE.replace("do_something", "overtool"),
                          overwrite=True)
        assert third.ok is True

    def test_duck_typed_security_does_not_raise(self, tmp_path):
        ctx = _ctx(tmp_path, security=_DuckSecurity())
        result = make_tool(ctx, name="ducktool", description="d", requirement="r",
                           source=VALID_SOURCE.replace("do_something", "ducktool"))
        assert result.ok is True, result.error

    def test_invalid_name_rejected(self, tmp_path):
        ctx = _ctx(tmp_path)
        assert make_tool(ctx, name="sys", description="d", requirement="r",
                         source=VALID_SOURCE).ok is False
        assert make_tool(ctx, name="", description="d", requirement="r",
                         source=VALID_SOURCE).ok is False


def result_output_module_matches(expected: str) -> bool:
    """Small helper so the assertion above reads cleanly."""
    return expected in sys.modules


# --------------------------------------------------------------------------- #
#  make_tool with a fake LLM (retry loop)
# --------------------------------------------------------------------------- #
class TestMakeToolWithLLM:

    def teardown_method(self, method):
        for key in list(sys.modules):
            if key.startswith("jarvis_generated."):
                sys.modules.pop(key, None)

    def test_first_invalid_then_valid_succeeds(self, tmp_path):
        # Attempt 1 returns dangerous code, attempt 2 returns a valid module.
        invalid = "```python\nimport os\ndef build_tools(ctx):\n" \
                  "    os.system('rm -rf /')\n    return []\n```"
        valid_body = VALID_SOURCE.replace("do_something", "retry_tool")
        valid = f"Here you go:\n```python\n{valid_body}```"
        llm = _FakeLLM(replies=[invalid, valid])
        ctx = _ctx(tmp_path)

        result = make_tool(
            ctx,
            name="retry_tool",
            description="Retry works",
            requirement="the LLM will get it right on the second try",
            llm=llm,
        )
        assert result.ok is True, result.error
        assert len(llm.calls) == 2, (
            f"LLM should have been called twice; got {len(llm.calls)}"
        )
        assert (Path(ctx.config.tools_dir()) / "retry_tool.py").exists()

    def test_gives_up_after_three_bad_attempts(self, tmp_path):
        bad = "```python\ndef nope(): pass\n```"  # missing build_tools
        llm = _FakeLLM(replies=[bad, bad, bad, bad])
        ctx = _ctx(tmp_path)
        result = make_tool(
            ctx, name="nogo", description="d", requirement="fail",
            llm=llm,
        )
        assert result.ok is False
        assert len(llm.calls) == 3
        assert not (Path(ctx.config.tools_dir()) / "nogo.py").exists()


# --------------------------------------------------------------------------- #
#  list / delete generated
# --------------------------------------------------------------------------- #
class TestListDelete:

    def teardown_method(self, method):
        for key in list(sys.modules):
            if key.startswith("jarvis_generated."):
                sys.modules.pop(key, None)

    def test_list_reports_generated(self, tmp_path):
        ctx = _ctx(tmp_path)
        make_tool(ctx, name="listable", description="d", requirement="r",
                  source=VALID_SOURCE.replace("do_something", "listable"))
        listing = list_generated(ctx)
        assert listing.ok is True
        names = {entry["name"] for entry in listing.output["tools"]}
        assert "listable" in names

    def test_delete_evicts_sys_modules(self, tmp_path):
        ctx = _ctx(tmp_path)
        make_tool(ctx, name="deletable", description="d", requirement="r",
                  source=VALID_SOURCE.replace("do_something", "deletable"))
        modname = generated_module_name("deletable")
        assert modname in sys.modules
        target = Path(ctx.config.tools_dir()) / "deletable.py"
        assert target.exists()

        result = delete_generated(ctx, name="deletable")
        assert result.ok is True
        assert not target.exists()
        assert modname not in sys.modules, (
            "sys.modules entry must be evicted on delete"
        )

    def test_delete_missing(self, tmp_path):
        ctx = _ctx(tmp_path)
        assert delete_generated(ctx, name="nope").ok is False
