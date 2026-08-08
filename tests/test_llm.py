"""Tests for the LLM backend layer.

Stdlib + pytest only — no network, no models. Backends we cannot exercise
directly (ollama daemon, transformers weights) are monkeypatched at the
seams they expose.
"""

from __future__ import annotations

import io
import json
from typing import Any, List, Sequence
from urllib import error as _urlerror

import pytest

from jarvis.core.config import LLMConfig
from jarvis.core.contracts import GenerationConfig, Message, Role
from jarvis.llm import (
    AUTO_PROBE_ORDER,
    BACKENDS,
    AirLLMBackend,
    OllamaBackend,
    StubBackend,
    TransformersBackend,
    apply_stop_strings,
    available_backends,
    create_llm,
    estimate_tokens,
    extract_thinking,
    format_chat,
    strip_thinking,
    trim_to_context,
)
from jarvis.llm.base import BaseLLM


# --------------------------------------------------------------------------- #
#  format_chat / thinking helpers
# --------------------------------------------------------------------------- #
def test_format_chat_chatml_exact():
    msgs = [
        Message.system("you are helpful"),
        Message.user("hi"),
        Message.assistant("hello"),
        Message.user("what's 2+2?"),
    ]
    out = format_chat(msgs, style="chatml")
    expected = (
        "<|im_start|>system\nyou are helpful<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhello<|im_end|>\n"
        "<|im_start|>user\nwhat's 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert out == expected


def test_format_chat_plain_style():
    out = format_chat([Message.user("ping")], style="plain")
    assert "USER: ping" in out
    assert out.rstrip().endswith("ASSISTANT:")


def test_format_chat_unknown_style():
    with pytest.raises(ValueError):
        format_chat([Message.user("x")], style="banana")


def test_strip_thinking_removes_closed_block():
    t = "<think>because two plus two</think>The answer is 4."
    assert strip_thinking(t) == "The answer is 4."


def test_strip_thinking_removes_unclosed_trailer_with_angle_bracket():
    # The critical case: a `[^<]*` regex would leave "if x <" in the output.
    t = "Actually,\n<think>let me check if x < y and it looks like"
    assert strip_thinking(t) == "Actually,"


def test_strip_thinking_multiple_blocks():
    t = "<think>a</think>hello<think>b</think> world"
    assert strip_thinking(t) == "hello world"


def test_extract_thinking_returns_joined_content():
    t = "<think>step one</think>ok<think>step two</think>"
    assert extract_thinking(t) == "step one\nstep two"


def test_extract_thinking_no_blocks():
    assert extract_thinking("plain text") is None
    assert extract_thinking("") is None


# --------------------------------------------------------------------------- #
#  estimate_tokens / trim_to_context
# --------------------------------------------------------------------------- #
def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    # 8 chars -> ~2 tokens.
    assert estimate_tokens("abcdefgh") == 2
    # A single character still counts as one token.
    assert estimate_tokens("x") == 1


def test_trim_to_context_never_empty_even_with_tiny_budget():
    msgs = [
        Message.system("sys" * 200),
        Message.user("hi there this is a long user turn " * 20),
    ]
    out = trim_to_context(msgs, max_tokens=1)
    assert out, "trim must never return empty"
    # System should still be there, plus the last user turn.
    assert out[0].role == Role.SYSTEM
    assert out[-1].role == Role.USER
    assert out[-1].content == msgs[1].content


def test_trim_to_context_drops_oldest_non_system_turns():
    msgs = [
        Message.system("sys"),
        Message.user("u1: " + "a" * 400),
        Message.assistant("a1: " + "b" * 400),
        Message.user("u2: " + "c" * 400),
        Message.assistant("a2: " + "d" * 400),
        Message.user("u3: " + "e" * 40),
    ]
    out = trim_to_context(msgs, max_tokens=200)
    # System + at least the last user turn survive.
    assert out[0].role == Role.SYSTEM
    assert out[-1].content == msgs[-1].content
    # Something older got dropped.
    assert len(out) < len(msgs)
    # Nothing dropped from the middle without also dropping older turns.
    kept_positions = [msgs.index(m) for m in out[1:]]
    assert kept_positions == sorted(kept_positions)


def test_trim_to_context_no_system():
    msgs = [Message.user("u1"), Message.assistant("a1"), Message.user("u2")]
    out = trim_to_context(msgs, max_tokens=100000)
    assert out == msgs


def test_trim_to_context_preserves_last_user_when_only_assistant_would_fit():
    msgs = [
        Message.system("s"),
        Message.assistant("a" * 4000),
        Message.user("u last"),
    ]
    out = trim_to_context(msgs, max_tokens=10)
    assert out[0].role == Role.SYSTEM
    assert out[-1].role == Role.USER


# --------------------------------------------------------------------------- #
#  Stop strings
# --------------------------------------------------------------------------- #
def test_apply_stop_strings_first_occurrence_wins():
    text = "one STOP two END three"
    # END appears later, STOP appears first: STOP should win.
    assert apply_stop_strings(text, ("END", "STOP")) == "one "


def test_apply_stop_strings_missing_stop_returns_text_unchanged():
    text = "no stop here"
    assert apply_stop_strings(text, ("zzz",)) == text


def test_apply_stop_strings_empty_inputs():
    assert apply_stop_strings("", ("x",)) == ""
    assert apply_stop_strings("text", ()) == "text"
    assert apply_stop_strings("text", ("",)) == "text"


# --------------------------------------------------------------------------- #
#  StubBackend
# --------------------------------------------------------------------------- #
def test_stub_backend_default_response_references_last_user():
    stub = StubBackend()
    assert stub.is_available() is True
    result = stub.generate([Message.system("s"), Message.user("Please make me tea.")])
    assert "Please make me tea." in result.text
    assert result.text.startswith("Very good, Sir")


def test_stub_backend_no_user_message():
    stub = StubBackend()
    result = stub.generate([Message.system("s")])
    assert result.text == "At your service, Sir."


def test_stub_backend_records_calls():
    stub = StubBackend()
    msgs = [Message.user("first")]
    stub.generate(msgs)
    stub.generate([Message.user("second")])
    assert len(stub.calls) == 2
    assert stub.calls[0][0].content == "first"


def test_stub_backend_cycles_injected_responses():
    stub = StubBackend(responses=["one", "two", "three"])
    outs = [stub.generate([Message.user("q")]).text for _ in range(5)]
    assert outs == ["one", "two", "three", "one", "two"]


def test_stub_backend_callable_response_gets_messages():
    seen: List[Sequence[Message]] = []

    def responder(messages):
        seen.append(list(messages))
        return f"heard-{messages[-1].content}"

    stub = StubBackend(responses=responder)
    r = stub.generate([Message.user("ping")])
    assert r.text == "heard-ping"
    assert seen[0][0].content == "ping"


def test_stub_backend_honours_stop_strings():
    stub = StubBackend(responses=["prefix STOP suffix"])
    r = stub.generate(
        [Message.user("q")],
        config=GenerationConfig(stop=("STOP",)),
    )
    assert r.text == "prefix "


def test_stub_backend_honours_max_new_tokens():
    long = "x" * 400
    stub = StubBackend(responses=[long])
    r = stub.generate(
        [Message.user("q")],
        config=GenerationConfig(max_new_tokens=5),
    )
    assert len(r.text) <= 5 * 4
    assert len(r.text) < len(long)


def test_stub_backend_stream_yields_words():
    stub = StubBackend(responses=["alpha beta gamma"])
    chunks = list(stub.stream([Message.user("q")]))
    assert "".join(chunks) == "alpha beta gamma"
    # Whitespace boundary chunking -> more than one chunk for three words.
    assert len(chunks) >= 3


# --------------------------------------------------------------------------- #
#  BaseLLM: gen config merging
# --------------------------------------------------------------------------- #
def test_gen_config_merges_defaults_from_cfg():
    cfg = LLMConfig(temperature=0.2, top_p=0.5, top_k=17, max_new_tokens=64)
    base = BaseLLM(cfg)

    # None -> everything comes from cfg.
    merged = base._gen_config(None)
    assert merged.temperature == 0.2
    assert merged.top_p == 0.5
    assert merged.top_k == 17
    assert merged.max_new_tokens == 64
    assert merged.stop == ()

    # An explicit GenerationConfig overrides cfg values.
    override = base._gen_config(
        GenerationConfig(temperature=0.9, top_p=0.11, top_k=3, max_new_tokens=8, stop=("###",))
    )
    assert override.temperature == 0.9
    assert override.top_p == pytest.approx(0.11)
    assert override.top_k == 3
    assert override.max_new_tokens == 8
    assert override.stop == ("###",)


def test_base_llm_load_is_idempotent():
    class Counting(BaseLLM):
        name = "counting"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.n = 0

        def _do_load(self) -> None:
            self.n += 1

        def is_available(self) -> bool:
            return True

    b = Counting(LLMConfig())
    b.load()
    b.load()
    b.load()
    assert b.n == 1


# --------------------------------------------------------------------------- #
#  create_llm / registry
# --------------------------------------------------------------------------- #
def _patch_all_probes_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every non-stub backend to report unavailable.

    Derived from AUTO_PROBE_ORDER rather than a hand-written list. A literal
    tuple here silently rots the moment a backend is added: the new one is never
    patched, so its real ``is_available()`` runs — which for the HTTP backends
    means these "hermetic" tests quietly probe the network. That passed on a dev
    box with nothing listening on :8000 and would have failed on the production
    machine, where a vLLM server is exactly what you would expect to be running.
    """
    for name in AUTO_PROBE_ORDER:
        cls = BACKENDS.get(name)
        if cls is not None and cls is not StubBackend:
            monkeypatch.setattr(cls, "is_available", lambda self: False, raising=True)


def test_create_llm_auto_falls_back_to_stub(monkeypatch: pytest.MonkeyPatch):
    _patch_all_probes_false(monkeypatch)
    cfg = LLMConfig(backend="auto", allow_fallback=True)
    llm = create_llm(cfg)
    assert isinstance(llm, StubBackend)
    assert llm.name == "stub"


def test_create_llm_auto_probe_order_prefers_ollama_over_airllm(
    monkeypatch: pytest.MonkeyPatch,
):
    # Both ollama and airllm claim availability. Order in AUTO_PROBE_ORDER
    # requires ollama to win — the whole point of that order.
    assert AUTO_PROBE_ORDER.index("ollama") < AUTO_PROBE_ORDER.index("airllm")
    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: True, raising=True)
    monkeypatch.setattr(AirLLMBackend, "is_available", lambda self: True, raising=True)
    monkeypatch.setattr(TransformersBackend, "is_available", lambda self: False, raising=True)
    cfg = LLMConfig(backend="auto", allow_fallback=True)
    llm = create_llm(cfg)
    assert isinstance(llm, OllamaBackend)


def test_create_llm_named_unavailable_with_fallback(monkeypatch: pytest.MonkeyPatch):
    _patch_all_probes_false(monkeypatch)
    cfg = LLMConfig(backend="airllm", allow_fallback=True)
    llm = create_llm(cfg)
    assert isinstance(llm, StubBackend)


def test_create_llm_named_unavailable_without_fallback(monkeypatch: pytest.MonkeyPatch):
    _patch_all_probes_false(monkeypatch)
    cfg = LLMConfig(backend="airllm", allow_fallback=False)
    with pytest.raises(RuntimeError) as exc:
        create_llm(cfg)
    assert "airllm" in str(exc.value).lower()


def test_create_llm_unknown_backend_no_fallback():
    cfg = LLMConfig(backend="does-not-exist", allow_fallback=False)
    with pytest.raises(RuntimeError):
        create_llm(cfg)


def test_create_llm_unknown_backend_with_fallback(monkeypatch: pytest.MonkeyPatch):
    _patch_all_probes_false(monkeypatch)
    cfg = LLMConfig(backend="does-not-exist", allow_fallback=True)
    llm = create_llm(cfg)
    assert isinstance(llm, StubBackend)


def test_create_llm_explicit_arg_overrides_cfg(monkeypatch: pytest.MonkeyPatch):
    _patch_all_probes_false(monkeypatch)
    cfg = LLMConfig(backend="ollama", allow_fallback=True)
    llm = create_llm(cfg, backend="stub")
    assert isinstance(llm, StubBackend)


def test_available_backends_lists_stub_at_minimum():
    cfg = LLMConfig()
    names = available_backends(cfg)
    # Stub has zero deps, so it should always be present.
    assert "stub" in names


# --------------------------------------------------------------------------- #
#  OllamaBackend
# --------------------------------------------------------------------------- #
def test_ollama_is_available_false_fast_when_urlopen_raises(monkeypatch: pytest.MonkeyPatch):
    calls: List[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", str(req)))
        raise _urlerror.URLError("connection refused")

    import jarvis.llm.ollama_backend as ob
    monkeypatch.setattr(ob._urlrequest, "urlopen", fake_urlopen)

    backend = OllamaBackend(LLMConfig())
    assert backend.is_available() is False
    # We must have actually tried; the fake was called.
    assert calls, "is_available() should hit urlopen"


def test_ollama_is_available_true_when_server_responds(monkeypatch: pytest.MonkeyPatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b'{"models": []}'

    import jarvis.llm.ollama_backend as ob

    def fake_urlopen(req, timeout=None):
        return FakeResp()

    monkeypatch.setattr(ob._urlrequest, "urlopen", fake_urlopen)
    backend = OllamaBackend(LLMConfig())
    assert backend.is_available() is True


def test_ollama_options_include_num_ctx(monkeypatch: pytest.MonkeyPatch):
    cfg = LLMConfig(context_tokens=6543, temperature=0.4, top_p=0.7, top_k=13)
    captured: dict = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return json.dumps({"message": {"content": "ok"}, "done": True}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    import jarvis.llm.ollama_backend as ob
    monkeypatch.setattr(ob._urlrequest, "urlopen", fake_urlopen)

    backend = OllamaBackend(cfg)
    # Passing config=None -> options are drawn from LLMConfig defaults.
    result = backend.generate([Message.user("hi")])
    assert result.text == "ok"
    payload = captured["payload"]
    opts = payload["options"]
    # num_ctx must always be set from cfg.context_tokens (the whole point:
    # Ollama silently defaults to 4096 otherwise).
    assert opts["num_ctx"] == 6543
    assert opts["temperature"] == pytest.approx(0.4)
    assert opts["top_p"] == pytest.approx(0.7)
    assert opts["top_k"] == 13
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]

    # An explicit GenerationConfig with stop + seed propagates too.
    backend.generate(
        [Message.user("hi again")],
        config=GenerationConfig(max_new_tokens=77, stop=("###",), seed=42),
    )
    opts2 = captured["payload"]["options"]
    assert opts2["num_ctx"] == 6543
    assert opts2["num_predict"] == 77
    assert opts2["stop"] == ["###"]
    assert opts2["seed"] == 42


def test_ollama_generate_strips_thinking(monkeypatch: pytest.MonkeyPatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            body = {"message": {"content": "<think>ignore</think>final answer"}}
            return json.dumps(body).encode("utf-8")

    import jarvis.llm.ollama_backend as ob
    monkeypatch.setattr(ob._urlrequest, "urlopen", lambda req, timeout=None: FakeResp())

    backend = OllamaBackend(LLMConfig())
    r = backend.generate([Message.user("hi")])
    assert r.text == "final answer"


def test_ollama_generate_raises_on_connection_error(monkeypatch: pytest.MonkeyPatch):
    import jarvis.llm.ollama_backend as ob

    def blow_up(req, timeout=None):
        raise _urlerror.URLError("no route to host")

    monkeypatch.setattr(ob._urlrequest, "urlopen", blow_up)
    backend = OllamaBackend(LLMConfig())
    with pytest.raises(RuntimeError):
        backend.generate([Message.user("hi")])


def test_ollama_list_models_raises_helpful_error(monkeypatch: pytest.MonkeyPatch):
    import jarvis.llm.ollama_backend as ob

    def blow_up(req, timeout=None):
        raise _urlerror.URLError("refused")

    monkeypatch.setattr(ob._urlrequest, "urlopen", blow_up)
    backend = OllamaBackend(LLMConfig())
    with pytest.raises(RuntimeError) as exc:
        backend.list_models()
    assert "ollama" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
#  Transformers / AirLLM availability
# --------------------------------------------------------------------------- #
def test_transformers_is_available_false_when_dep_missing(monkeypatch: pytest.MonkeyPatch):
    import jarvis.llm.transformers_backend as tb

    def fake_find_spec(name):
        if name in ("transformers", "torch"):
            return None
        return object()

    monkeypatch.setattr(tb.importlib.util, "find_spec", fake_find_spec)
    assert TransformersBackend(LLMConfig()).is_available() is False


def test_airllm_is_available_false_when_dep_missing(monkeypatch: pytest.MonkeyPatch):
    import jarvis.llm.airllm_backend as ab

    def fake_find_spec(name):
        if name == "airllm":
            return None
        return object()

    monkeypatch.setattr(ab.importlib.util, "find_spec", fake_find_spec)
    assert AirLLMBackend(LLMConfig()).is_available() is False


def test_airllm_is_available_true_when_dep_present(monkeypatch: pytest.MonkeyPatch):
    import jarvis.llm.airllm_backend as ab

    def fake_find_spec(name):
        return object() if name == "airllm" else None

    monkeypatch.setattr(ab.importlib.util, "find_spec", fake_find_spec)
    assert AirLLMBackend(LLMConfig()).is_available() is True


# --------------------------------------------------------------------------- #
#  Bare-stdlib import smoke test
# --------------------------------------------------------------------------- #
def test_llm_modules_import_without_heavy_deps():
    # These modules must be importable on a stdlib-only environment. If any of
    # them eagerly imports airllm/torch/transformers, this test will have
    # failed already at collection time — asserting the modules exist keeps
    # the intent explicit.
    import jarvis.llm as pkg
    import jarvis.llm.airllm_backend  # noqa: F401
    import jarvis.llm.ollama_backend  # noqa: F401
    import jarvis.llm.transformers_backend  # noqa: F401
    import jarvis.llm.stub_backend  # noqa: F401
    assert hasattr(pkg, "create_llm")
