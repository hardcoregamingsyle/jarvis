"""Tests for the OpenAI-compatible client and the vLLM backend.

Nothing here touches the network: ``urllib.request.urlopen`` is monkeypatched
for every test that would otherwise open a socket.  The concurrency tests use
real threads against the fake transport, which is the only way to observe that
the in-flight semaphore actually caps anything.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as _urlerror

import pytest

from jarvis.core.config import LLMConfig
from jarvis.core.contracts import GenerationConfig, Message
from jarvis.llm import (
    AUTO_PROBE_ORDER,
    BACKENDS,
    OllamaBackend,
    OpenAICompatBackend,
    StubBackend,
    VLLMBackend,
    available_backends,
    create_llm,
)
import jarvis.llm.openai_compat as oc
from jarvis.llm.openai_compat import iter_sse_payloads, normalise_base_url
from jarvis.llm.vllm_backend import DEFAULT_VLLM_HOST, max_num_seqs, server_command


PROMPT = [Message.user("hello")]


# --------------------------------------------------------------------------- #
#  Fake transport
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Stands in for an ``http.client.HTTPResponse``.

    ``chunks`` are handed out by ``__iter__`` exactly as given, so a test can
    split an SSE frame across arbitrary byte boundaries.
    """

    def __init__(self, body: bytes = b"", status: int = 200, chunks: Optional[Sequence[bytes]] = None):
        self.status = status
        self._body = body
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self._body
        out, self._body = self._body[:n], self._body[n:]
        return out

    def __iter__(self):
        if self._chunks is None:
            return iter([self._body])
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def json_response(payload: Dict[str, Any], status: int = 200) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"), status=status)


def http_error(url: str, code: int, body: str) -> _urlerror.HTTPError:
    import io

    return _urlerror.HTTPError(url, code, "err", {}, io.BytesIO(body.encode("utf-8")))


MODELS_BODY = {"object": "list", "data": [{"id": "served/model-a"}, {"id": "served/model-b"}]}


def completion_body(text: str = "Very good, sir.") -> Dict[str, Any]:
    return {
        "id": "cmpl-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }


class Recorder:
    """Records every request and replies from a routing table."""

    def __init__(self) -> None:
        self.requests: List[Any] = []
        self.lock = threading.Lock()

    def urls(self) -> List[str]:
        return [r.full_url for r in self.requests]

    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(r.data.decode("utf-8")) for r in self.requests if r.data]


def install(monkeypatch: pytest.MonkeyPatch, handler) -> Recorder:
    """Patch ``urlopen`` with ``handler(req, timeout) -> FakeResponse``."""
    rec = Recorder()

    def fake_urlopen(req, timeout=None):
        with rec.lock:
            rec.requests.append(req)
        return handler(req, timeout)

    monkeypatch.setattr(oc._urlrequest, "urlopen", fake_urlopen)
    return rec


def simple_handler(req, timeout=None):
    """Answers /models and /chat/completions with canned success bodies."""
    if req.full_url.endswith("/models"):
        return json_response(MODELS_BODY)
    return json_response(completion_body())


def backend(monkeypatch: pytest.MonkeyPatch, **cfg_kw: Any) -> OpenAICompatBackend:
    monkeypatch.delenv(oc.API_KEY_ENV, raising=False)
    cfg = LLMConfig(model="served/model-a", request_timeout=5.0, **cfg_kw)
    return OpenAICompatBackend(cfg, base_url="http://test-host:8000/v1")


# --------------------------------------------------------------------------- #
#  SSE parsing
# --------------------------------------------------------------------------- #
def test_sse_frames_split_across_chunk_boundaries():
    # "data: {" ... the JSON is torn in half mid-token, twice.
    chunks = [
        b'data: {"choices":[{"delta":{"cont',
        b'ent":"Hel"}}]}\n\ndata: {"choi',
        b'ces":[{"delta":{"content":"lo"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert [json.loads(p)["choices"][0]["delta"]["content"] for p in iter_sse_payloads(chunks)] == [
        "Hel",
        "lo",
    ]


def test_sse_ignores_keepalive_blanks_and_comments():
    chunks = [
        b"\n",
        b": ping\n\n",
        b'data: {"n":1}\n\n',
        b"\n\n",
        b": keep-alive\n",
        b'data: {"n":2}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert [json.loads(p)["n"] for p in iter_sse_payloads(chunks)] == [1, 2]


def test_sse_multi_line_data_frame_is_joined_with_newlines():
    chunks = [b"data: line one\ndata: line two\n\ndata: [DONE]\n\n"]
    assert list(iter_sse_payloads(chunks)) == ["line one\nline two"]


def test_sse_stops_at_done_and_ignores_trailing_frames():
    chunks = [b'data: {"n":1}\n\n', b"data: [DONE]\n\n", b'data: {"n":99}\n\n']
    assert [json.loads(p)["n"] for p in iter_sse_payloads(chunks)] == [1]


def test_sse_flushes_a_final_frame_with_no_trailing_blank_line():
    # Servers that close the socket right after the last frame are common.
    assert list(iter_sse_payloads([b'data: {"n":1}'])) == ['{"n":1}']


def test_stream_yields_deltas_and_closes_the_response(monkeypatch: pytest.MonkeyPatch):
    sse = FakeResponse(
        chunks=[
            b'data: {"choices":[{"delta":{"content":"Good "}}]}\n\n',
            b': keep-alive\n\n',
            b'data: {"choices":[{"delta":{"content":"evening"}}]}\n\n',
            b'data: {"choices":[{"delta":{}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        return sse

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    assert list(llm.stream(PROMPT)) == ["Good ", "evening"]
    assert sse.closed, "the streamed response must be closed"


def test_stream_sets_stream_true_in_the_payload(monkeypatch: pytest.MonkeyPatch):
    rec = install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else FakeResponse(chunks=[b"data: [DONE]\n\n"]),
    )
    llm = backend(monkeypatch)
    list(llm.stream(PROMPT))
    chat = [b for b in rec.bodies() if "messages" in b]
    assert chat and chat[0]["stream"] is True


def test_stream_truncates_at_a_stop_string(monkeypatch: pytest.MonkeyPatch):
    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        return FakeResponse(
            chunks=[
                b'data: {"choices":[{"delta":{"content":"keep "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"me<|stop|>drop me"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    out = list(llm.stream(PROMPT, GenerationConfig(stop=("<|stop|>",))))
    assert "".join(out) == "keep me"


# --------------------------------------------------------------------------- #
#  generate()
# --------------------------------------------------------------------------- #
def test_generate_returns_content_and_usage(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, simple_handler)
    llm = backend(monkeypatch)
    result = llm.generate(PROMPT, GenerationConfig(max_new_tokens=64, temperature=0.1))

    assert result.text == "Very good, sir."
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4
    assert result.finish_reason == "stop"

    assert "http://test-host:8000/v1/chat/completions" in rec.urls()
    body = [b for b in rec.bodies() if "messages" in b][0]
    assert body["model"] == "served/model-a"
    assert body["stream"] is False
    assert body["max_tokens"] == 64
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_strips_thinking_blocks(monkeypatch: pytest.MonkeyPatch):
    install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else json_response(completion_body("<think>hmm 2+2</think>Four, sir.")),
    )
    llm = backend(monkeypatch)
    assert llm.generate(PROMPT).text == "Four, sir."


def test_generate_applies_stop_strings(monkeypatch: pytest.MonkeyPatch):
    install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else json_response(completion_body("visible<|end|>hidden")),
    )
    llm = backend(monkeypatch)
    assert llm.generate(PROMPT, GenerationConfig(stop=("<|end|>",))).text == "visible"


# --------------------------------------------------------------------------- #
#  Error reporting
# --------------------------------------------------------------------------- #
def test_http_error_message_names_status_and_body(monkeypatch: pytest.MonkeyPatch):
    body = '{"error": {"message": "model `nope` does not exist", "type": "NotFoundError"}}'

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        raise http_error(req.full_url, 400, body)

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    msg = str(exc.value)
    assert "400" in msg
    assert "model `nope` does not exist" in msg
    assert "chat/completions" in msg


def test_error_shaped_200_body_does_not_raise_keyerror(monkeypatch: pytest.MonkeyPatch):
    # The classic failure: a 200 whose body is an error object, and the client
    # blows up with `KeyError: 'choices'` instead of saying what happened.
    body = {"error": {"message": "context length exceeded", "code": "ctx"}}
    install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else json_response(body),
    )
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    msg = str(exc.value)
    assert "context length exceeded" in msg
    assert "HTTP 200" in msg


def test_non_json_body_reports_status_and_snippet(monkeypatch: pytest.MonkeyPatch):
    html = "<html><body>502 Bad Gateway from the reverse proxy</body></html>"
    install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else FakeResponse(html.encode("utf-8"), status=200),
    )
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    msg = str(exc.value)
    assert "invalid JSON" in msg
    assert "502 Bad Gateway from the reverse proxy" in msg


def test_body_snippet_is_truncated(monkeypatch: pytest.MonkeyPatch):
    huge = "x" * 5000
    install(
        monkeypatch,
        lambda req, timeout=None: json_response(MODELS_BODY)
        if req.full_url.endswith("/models")
        else FakeResponse(huge.encode("utf-8")),
    )
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    assert str(exc.value).count("x") <= 250


def test_timeout_raises_runtime_error_naming_the_url(monkeypatch: pytest.MonkeyPatch):
    import socket

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        raise _urlerror.URLError(socket.timeout("timed out"))

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    msg = str(exc.value)
    assert "timed out" in msg
    assert "http://test-host:8000/v1/chat/completions" in msg


def test_unreachable_server_message_names_the_url(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(
        _urlerror.URLError("connection refused")
    ))
    llm = backend(monkeypatch)
    llm.max_attempts = 1
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    assert "http://test-host:8000/v1/chat/completions" in str(exc.value)


# --------------------------------------------------------------------------- #
#  Authorization header
# --------------------------------------------------------------------------- #
def test_authorization_header_present_only_when_key_is_set(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, simple_handler)
    monkeypatch.delenv(oc.API_KEY_ENV, raising=False)
    cfg = LLMConfig(model="served/model-a")

    anonymous = OpenAICompatBackend(cfg, base_url="http://h:8000/v1")
    anonymous.generate(PROMPT)
    assert anonymous.api_key == ""
    assert all(r.get_header("Authorization") is None for r in rec.requests)

    rec.requests.clear()
    keyed = OpenAICompatBackend(cfg, base_url="http://h:8000/v1", api_key="sk-secret")
    keyed.generate(PROMPT)
    assert rec.requests, "expected requests to be recorded"
    for req in rec.requests:
        assert req.get_header("Authorization") == "Bearer sk-secret"


def test_api_key_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(oc.API_KEY_ENV, "sk-from-env")
    llm = OpenAICompatBackend(LLMConfig(), base_url="http://h:8000/v1")
    assert llm.api_key == "sk-from-env"


# --------------------------------------------------------------------------- #
#  Retry / backoff
# --------------------------------------------------------------------------- #
def _count_sleeps(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    """Neuter the backoff sleep, recording the attempt numbers it was asked for."""
    slept: List[int] = []
    monkeypatch.setattr(
        OpenAICompatBackend,
        "_sleep_backoff",
        lambda self, attempt: slept.append(attempt),
        raising=True,
    )
    return slept


def test_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    # vLLM answers 503 while it is still loading weights.
    slept = _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        calls.append("chat")
        if len(calls) < 3:
            raise http_error(req.full_url, 503, "server not ready")
        return json_response(completion_body("ready now"))

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    assert llm.generate(PROMPT).text == "ready now"
    assert len(calls) == 3, "should have retried twice before succeeding"
    assert slept == [1, 2]


def test_retries_429(monkeypatch: pytest.MonkeyPatch):
    _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        calls.append("chat")
        if len(calls) < 2:
            raise http_error(req.full_url, 429, "slow down")
        return json_response(completion_body("ok"))

    install(monkeypatch, handler)
    assert backend(monkeypatch).generate(PROMPT).text == "ok"
    assert len(calls) == 2


def test_no_retry_on_400(monkeypatch: pytest.MonkeyPatch):
    slept = _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        calls.append("chat")
        raise http_error(req.full_url, 400, "bad request")

    install(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        backend(monkeypatch).generate(PROMPT)
    assert len(calls) == 1, "4xx other than 429 must not be retried"
    assert slept == []


def test_retry_attempts_are_bounded(monkeypatch: pytest.MonkeyPatch):
    slept = _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        calls.append("chat")
        raise http_error(req.full_url, 503, "still loading")

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    assert len(calls) == llm.max_attempts == 4
    assert len(slept) == 3, "one sleep between each pair of attempts, none after the last"
    assert "503" in str(exc.value)


def test_connection_errors_are_retried_then_give_up(monkeypatch: pytest.MonkeyPatch):
    _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        calls.append(req.full_url)
        raise _urlerror.URLError("connection refused")

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    with pytest.raises(RuntimeError):
        llm.generate(PROMPT)
    # load() swallows its own failure, so only the chat attempts are bounded.
    chat_calls = [u for u in calls if u.endswith("/chat/completions")]
    assert len(chat_calls) == llm.max_attempts


def test_backoff_delay_is_capped_and_jittered(monkeypatch: pytest.MonkeyPatch):
    llm = backend(monkeypatch)
    assert llm._backoff_delay(1) >= llm.backoff_base
    # Grows with the attempt number...
    assert llm._backoff_delay(3) > llm.backoff_base
    # ...but never past the cap (plus the 25% jitter band).
    for attempt in range(1, 30):
        assert llm._backoff_delay(attempt) <= llm.backoff_cap * 1.25


# --------------------------------------------------------------------------- #
#  is_available
# --------------------------------------------------------------------------- #
def test_is_available_false_on_exception(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(
        _urlerror.URLError("connection refused")
    ))
    llm = backend(monkeypatch)
    assert llm.is_available() is False
    assert rec.urls() == ["http://test-host:8000/v1/models"]


def test_is_available_true_when_models_answers(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    assert backend(monkeypatch).is_available() is True


def test_is_available_probe_is_cached_across_rapid_calls(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, simple_handler)
    llm = backend(monkeypatch)
    results = [llm.is_available() for _ in range(50)]
    assert all(results)
    assert len(rec.requests) == 1, "a fan-out of agents must not re-probe per call"


def test_is_available_cache_expires(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, simple_handler)
    llm = backend(monkeypatch)
    assert llm.is_available() is True
    llm.invalidate_probe()
    assert llm.is_available() is True
    assert len(rec.requests) == 2


def test_is_available_probe_uses_a_short_timeout(monkeypatch: pytest.MonkeyPatch):
    seen: List[Optional[float]] = []

    def handler(req, timeout=None):
        seen.append(timeout)
        return json_response(MODELS_BODY)

    install(monkeypatch, handler)
    # request_timeout is 600 by default; a probe must not inherit that.
    OpenAICompatBackend(LLMConfig(), base_url="http://h:8000/v1").is_available()
    assert seen == [oc.PROBE_TIMEOUT]


# --------------------------------------------------------------------------- #
#  load() / model resolution
# --------------------------------------------------------------------------- #
def test_load_keeps_a_model_that_is_actually_served(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    llm = backend(monkeypatch)
    llm.load()
    assert llm.model == "served/model-a"


def test_load_picks_the_served_model_when_config_does_not_match(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    cfg = LLMConfig(model="Qwen/nothing-like-this")
    llm = OpenAICompatBackend(cfg, base_url="http://h:8000/v1")
    llm.load()
    assert llm.model == "served/model-a"


def test_load_resolves_a_model_by_basename(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    # The server renamed "org/model-b" to its own prefix; the basename matches.
    llm = OpenAICompatBackend(LLMConfig(model="other-org/model-b"), base_url="http://h:8000/v1")
    llm.load()
    assert llm.model == "served/model-b"


def test_load_runs_once_under_concurrency(monkeypatch: pytest.MonkeyPatch):
    rec = install(monkeypatch, simple_handler)
    llm = backend(monkeypatch)
    threads = [threading.Thread(target=llm.load) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len([u for u in rec.urls() if u.endswith("/models")]) == 1


def test_load_gives_up_quickly_and_leaves_the_patience_to_the_real_call(
    monkeypatch: pytest.MonkeyPatch,
):
    _count_sleeps(monkeypatch)
    calls: List[str] = []

    def handler(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/models"):
            raise _urlerror.URLError("connection refused")
        return json_response(completion_body("ok"))

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    assert llm.generate(PROMPT).text == "ok"
    # Two probes at load, then the configured model id is used regardless.
    assert len([u for u in calls if u.endswith("/models")]) == 2
    assert llm.model == "served/model-a"


def test_load_failure_with_no_configured_model_surfaces_the_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    _count_sleeps(monkeypatch)
    install(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(
        _urlerror.URLError("connection refused")
    ))
    llm = OpenAICompatBackend(LLMConfig(model=""), base_url="http://h:8000/v1")
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    assert "http://h:8000/v1/models" in str(exc.value)
    assert llm._loaded is False, "a failed load must be retryable"


def test_generate_without_any_model_id_explains_itself(monkeypatch: pytest.MonkeyPatch):
    install(
        monkeypatch,
        lambda req, timeout=None: json_response({"object": "list", "data": []}),
    )
    llm = OpenAICompatBackend(LLMConfig(model=""), base_url="http://h:8000/v1")
    with pytest.raises(RuntimeError) as exc:
        llm.generate(PROMPT)
    assert "model" in str(exc.value).lower()


def test_list_models_returns_served_ids(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    assert backend(monkeypatch).list_models() == ["served/model-a", "served/model-b"]


# --------------------------------------------------------------------------- #
#  Concurrency
# --------------------------------------------------------------------------- #
def test_24_threads_generate_concurrently_and_the_semaphore_caps_them(
    monkeypatch: pytest.MonkeyPatch,
):
    limit = 4
    threads_count = 24
    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()
    # Only releases when `limit` requests are genuinely in flight together, so
    # a semaphore that capped *below* the limit would hang and fail the test.
    gate = threading.Barrier(limit, timeout=30)

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        try:
            gate.wait()
        finally:
            with lock:
                state["in_flight"] -= 1
        return json_response(completion_body("ok"))

    install(monkeypatch, handler)
    cfg = LLMConfig(model="served/model-a", max_concurrent_requests=limit)
    llm = OpenAICompatBackend(cfg, base_url="http://h:8000/v1")
    assert llm.max_concurrent_requests == limit

    results: List[str] = []
    errors: List[BaseException] = []
    out_lock = threading.Lock()

    def worker() -> None:
        try:
            text = llm.generate(PROMPT).text
        except BaseException as exc:  # noqa: BLE001 - reported below
            with out_lock:
                errors.append(exc)
        else:
            with out_lock:
                results.append(text)

    workers = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=60)

    assert not errors, f"threads failed: {errors[:3]}"
    assert results == ["ok"] * threads_count
    assert state["peak"] == limit, f"observed peak concurrency {state['peak']}, expected {limit}"


def test_unlimited_config_installs_no_semaphore(monkeypatch: pytest.MonkeyPatch):
    llm = backend(monkeypatch, max_concurrent_requests=0)
    assert llm.max_concurrent_requests == 0
    assert llm._semaphore is None


def test_the_config_default_installs_a_cap(monkeypatch: pytest.MonkeyPatch):
    # An agent tree must not be able to open unbounded sockets by default.
    llm = OpenAICompatBackend(LLMConfig(), base_url="http://h:8000/v1")
    assert llm.max_concurrent_requests == LLMConfig().max_concurrent_requests > 0
    assert llm._semaphore is not None


def test_concurrent_generate_calls_do_not_share_request_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every call must build its own Request object -- no per-request state on self."""
    seen: List[int] = []
    lock = threading.Lock()

    def handler(req, timeout=None):
        if req.full_url.endswith("/models"):
            return json_response(MODELS_BODY)
        payload = json.loads(req.data.decode("utf-8"))
        with lock:
            seen.append(payload["max_tokens"])
        time.sleep(0.005)
        return json_response(completion_body(payload["messages"][0]["content"]))

    install(monkeypatch, handler)
    llm = backend(monkeypatch)
    answers: Dict[int, str] = {}
    lock2 = threading.Lock()

    def worker(i: int) -> None:
        result = llm.generate(
            [Message.user(f"q{i}")], GenerationConfig(max_new_tokens=100 + i)
        )
        with lock2:
            answers[i] = result.text

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert answers == {i: f"q{i}" for i in range(12)}
    assert sorted(seen) == list(range(100, 112))


# --------------------------------------------------------------------------- #
#  base_url handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("https://api.example.com/openai/v1", "https://api.example.com/openai/v1"),
        ("box:8000", "http://box:8000/v1"),
        ("", oc.DEFAULT_BASE_URL),
    ],
)
def test_normalise_base_url(raw: str, expected: str):
    assert normalise_base_url(raw) == expected


# --------------------------------------------------------------------------- #
#  VLLMBackend
# --------------------------------------------------------------------------- #
def test_vllm_defaults_to_the_vllm_host_and_cfg_model():
    llm = VLLMBackend(LLMConfig(model="Qwen/Qwen3-30B-A3B-Instruct-2507"))
    assert llm.name == "vllm"
    assert llm.base_url == DEFAULT_VLLM_HOST
    assert llm.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_vllm_honours_a_configured_host():
    cfg = LLMConfig(vllm_host="http://gpu-box:9001")
    assert VLLMBackend(cfg).base_url == "http://gpu-box:9001/v1"


def test_server_command_contains_model_and_max_model_len():
    cfg = LLMConfig(model="Qwen/Qwen3-30B-A3B-Instruct-2507", context_tokens=16384)
    cmd = server_command(cfg)

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "16384"
    assert "--max-num-seqs" in cmd
    assert int(cmd[cmd.index("--max-num-seqs") + 1]) >= 4
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8000"
    assert all(isinstance(part, str) for part in cmd)


def test_server_command_is_available_as_a_method_too():
    cfg = LLMConfig(model="m")
    assert VLLMBackend.server_command(cfg) == server_command(cfg)
    assert VLLMBackend(cfg).server_command(cfg) == server_command(cfg)


def test_server_command_includes_api_key_and_download_dir_only_when_set(tmp_path):
    bare = server_command(LLMConfig(model="m"))
    assert "--api-key" not in bare
    assert "--download-dir" not in bare

    cfg = LLMConfig(model="m", api_key="sk-vllm", layer_shards_dir=str(tmp_path / "weights"))
    cmd = server_command(cfg)
    assert cmd[cmd.index("--api-key") + 1] == "sk-vllm"
    assert cmd[cmd.index("--download-dir") + 1] == str(tmp_path / "weights")


def test_server_command_prefers_an_explicit_models_dir(tmp_path):
    # Config.models_dir() is a method on the top-level Config object.
    cfg = LLMConfig(model="m", layer_shards_dir=str(tmp_path / "shards"))
    setattr(cfg, "models_dir", lambda: str(tmp_path / "models"))
    cmd = server_command(cfg)
    assert cmd[cmd.index("--download-dir") + 1] == str(tmp_path / "models")


def test_server_command_max_num_seqs_follows_the_client_cap():
    cfg = LLMConfig(model="m", max_concurrent_requests=12)
    cmd = server_command(cfg)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "12"
    assert max_num_seqs(cfg) == 12
    assert max_num_seqs(LLMConfig(model="m", max_concurrent_requests=0)) >= 4


def test_vllm_health_reports_reachable_and_concurrency(monkeypatch: pytest.MonkeyPatch):
    install(monkeypatch, simple_handler)
    cfg = LLMConfig(model="served/model-a", max_concurrent_requests=6)
    info = VLLMBackend(cfg, base_url="http://h:8000/v1").health()

    assert info["reachable"] is True
    assert info["models"] == ["served/model-a", "served/model-b"]
    assert info["max_concurrent_requests"] == 6
    assert info["server_max_num_seqs"] == 6
    assert info["error"] is None


def test_vllm_health_reports_an_unreachable_server_without_raising(
    monkeypatch: pytest.MonkeyPatch,
):
    slept = _count_sleeps(monkeypatch)
    rec = install(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(
        _urlerror.URLError("connection refused")
    ))
    info = VLLMBackend(LLMConfig(), base_url="http://h:8000/v1").health()
    assert info["reachable"] is False
    assert info["models"] == []
    assert "connection refused" in info["error"]
    # A health check answers now; it does not sit through the retry ladder.
    assert len(rec.requests) == 1
    assert slept == []


# --------------------------------------------------------------------------- #
#  Registry wiring
# --------------------------------------------------------------------------- #
def test_backends_registered():
    assert BACKENDS["vllm"] is VLLMBackend
    assert BACKENDS["openai-compat"] is OpenAICompatBackend


def test_auto_probe_order_puts_vllm_first():
    assert AUTO_PROBE_ORDER[0] == "vllm"
    assert AUTO_PROBE_ORDER.index("vllm") < AUTO_PROBE_ORDER.index("ollama")
    assert AUTO_PROBE_ORDER.index("ollama") < AUTO_PROBE_ORDER.index("openai-compat")
    # AirLLM's probe is only an import test, so it must stay last.
    assert AUTO_PROBE_ORDER[-1] == "airllm"


def _all_probes(monkeypatch: pytest.MonkeyPatch, **claims: bool) -> None:
    for name in AUTO_PROBE_ORDER:
        cls = BACKENDS[name]
        value = claims.get(name.replace("-", "_"), False)
        monkeypatch.setattr(cls, "is_available", lambda self, v=value: v, raising=True)


def test_auto_selects_vllm_over_ollama_when_both_are_available(
    monkeypatch: pytest.MonkeyPatch,
):
    _all_probes(monkeypatch, vllm=True, ollama=True)
    llm = create_llm(LLMConfig(backend="auto", allow_fallback=True))
    assert isinstance(llm, VLLMBackend)
    assert llm.name == "vllm"


def test_auto_selects_ollama_when_vllm_is_absent(monkeypatch: pytest.MonkeyPatch):
    _all_probes(monkeypatch, ollama=True, openai_compat=True)
    llm = create_llm(LLMConfig(backend="auto", allow_fallback=True))
    assert isinstance(llm, OllamaBackend)


def test_auto_falls_through_to_openai_compat(monkeypatch: pytest.MonkeyPatch):
    _all_probes(monkeypatch, openai_compat=True, airllm=True)
    llm = create_llm(LLMConfig(backend="auto", allow_fallback=True))
    assert isinstance(llm, OpenAICompatBackend)
    assert not isinstance(llm, VLLMBackend)


def test_auto_falls_back_to_stub_when_nothing_answers(monkeypatch: pytest.MonkeyPatch):
    _all_probes(monkeypatch)
    assert isinstance(create_llm(LLMConfig(backend="auto")), StubBackend)


def test_named_vllm_without_fallback_raises_with_a_linux_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    _all_probes(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        create_llm(LLMConfig(backend="vllm", allow_fallback=False))
    msg = str(exc.value)
    assert "vllm" in msg.lower()
    assert "linux" in msg.lower()


def test_named_backend_is_constructible_from_the_registry(monkeypatch: pytest.MonkeyPatch):
    _all_probes(monkeypatch, vllm=True, openai_compat=True)
    cfg = LLMConfig(backend="openai-compat", allow_fallback=False)
    assert isinstance(create_llm(cfg), OpenAICompatBackend)


def test_available_backends_includes_the_new_names(monkeypatch: pytest.MonkeyPatch):
    _all_probes(monkeypatch, vllm=True, openai_compat=True)
    names = available_backends(LLMConfig())
    assert "vllm" in names
    assert "openai-compat" in names
    assert "ollama" not in names
