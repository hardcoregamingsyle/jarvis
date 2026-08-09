"""The remote interface: real sockets, real requests, real refusals.

Every server here binds ``127.0.0.1`` on port 0 — a kernel-assigned ephemeral
port on the loopback interface only.  Nothing in this file listens where another
machine could reach it, which is exactly the property the code under test is
supposed to enforce.

The two things worth being pedantic about, and the reason most of these tests
exist:

* **The audio must belong to the client.**  A remote session that transcribed
  from the server's microphone or spoke through the server's speakers would be
  listening to, and talking at, an empty room.  ``/api/tts`` is therefore
  checked against a fake speech queue that records anything it is asked to say.
* **The token must actually keep people out.**  JARVIS drives its own machine
  without restriction; that is the owner's decision and no test here second-
  guesses it.  It does mean an unauthenticated port is a remote root shell, so
  the refusals are tested as hard as the successes.
"""

from __future__ import annotations

import hmac
import http.client
import json
import logging
import os
import socket
import stat
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pytest

from jarvis.core.contracts import Transcript
from jarvis.core.events import Events
from jarvis.server import app as server_app
from jarvis.server import (
    SESSION_COOKIE,
    TOKEN_ENV_VAR,
    InsecureBindError,
    JarvisServer,
    Sessions,
    check_token,
    client_page,
    generate_token,
    is_loopback,
    load_or_create_token,
    serve,
    token_path,
)
from jarvis.speech.audio_io import wav_bytes

# Obviously not a secret: it never leaves this file and no server here is
# reachable from off-box.
TOKEN = "test-token-not-a-real-secret-9f2c"

REPLY = "Very good, Sir."
TRANSCRIPT_TEXT = "Good evening, Sir."


# --------------------------------------------------------------------------- #
#  Doubles
# --------------------------------------------------------------------------- #
class FakeTasks:
    """Just enough TaskManager for /api/tasks."""

    def __init__(self) -> None:
        self.tasks: List[Any] = []

    def list(self, **_kwargs: Any) -> List[Any]:
        return list(self.tasks)

    def stats(self) -> Dict[str, Any]:
        return {"tracked": len(self.tasks), "running": 0, "deepest_depth": 0}

    def render_tree(self, root_id: Optional[str] = None, **_kwargs: Any) -> str:
        return "(no background tasks)"


class FakeOrchestrator:
    """Records how it was called; can be told to raise."""

    def __init__(self, reply: str = REPLY) -> None:
        self.reply = reply
        self.calls: List[Tuple[str, Any]] = []
        self.spoken: List[str] = []
        self.error: Optional[Exception] = None
        self.updates: List[str] = []
        self.tasks = FakeTasks()

    def chat(self, user_input: str, *, speak: Any = None, **_kwargs: Any) -> str:
        self.calls.append((user_input, speak))
        if self.error is not None:
            raise self.error
        return self.reply

    def say(self, text: str) -> None:  # pragma: no cover - must never be reached
        self.spoken.append(text)

    def pending_updates(self) -> List[str]:
        taken, self.updates = list(self.updates), []
        return taken

    def task_tree(self, root_id: Optional[str] = None) -> str:
        return self.tasks.render_tree(root_id)


class FakeSTT:
    name = "fake-stt"

    def __init__(self, text: str = TRANSCRIPT_TEXT) -> None:
        self.text = text
        self.audio: List[bytes] = []
        self.rates: List[int] = []

    @property
    def calls(self) -> int:
        return len(self.audio)

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> Transcript:
        self.audio.append(bytes(audio) if isinstance(audio, (bytes, bytearray)) else b"")
        self.rates.append(sample_rate)
        return Transcript(text=self.text, language="en", confidence=0.93, segments=())


class FakeTTS:
    """Separates synthesis from playback, which is the whole point here."""

    name = "fake-tts"
    output_format = "wav"

    def __init__(self) -> None:
        self.synthesized: List[str] = []
        self.spoken: List[str] = []

    def synthesize(self, text: str) -> bytes:
        self.synthesized.append(text)
        return wav_bytes([0.0] * 320, 16000)

    def speak(self, text: str) -> None:  # pragma: no cover - a failure if called
        self.spoken.append(text)


class FakeSpeechQueue:
    """The server's own speakers.  A remote session must never reach these."""

    def __init__(self) -> None:
        self.said: List[str] = []

    def say(self, text: str) -> None:  # pragma: no cover - a failure if called
        self.said.append(text)

    def speak(self, text: str) -> None:  # pragma: no cover - a failure if called
        self.said.append(text)


class FakeSubsystems:
    """Shaped like :class:`jarvis.app.Subsystems`, with none of the weight."""

    def __init__(self, config: Any, bus: Any) -> None:
        self.config = config
        self.bus = bus
        self.orchestrator = FakeOrchestrator()
        self.stt = FakeSTT()
        self.tts = FakeTTS()
        self.speech_queue = FakeSpeechQueue()
        self.llm = None
        self.memory = None
        self.registry = None

    def status(self) -> Dict[str, Any]:
        return {
            "llm": "scripted",
            "model": self.config.llm.model,
            "stt": self.stt.name,
            "tts": self.tts.name,
            "memory": "sqlite",
            "tools": 7,
            "security_mode": self.config.security.mode,
            "data_dir": str(self.config.home()),
        }


# --------------------------------------------------------------------------- #
#  HTTP helpers
# --------------------------------------------------------------------------- #
def _header(headers: Dict[str, str], name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def request(
    url: str,
    *,
    method: str = "GET",
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Dict[str, str], bytes]:
    """One real HTTP request.  HTTP errors come back as values, not exceptions."""
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        finally:
            exc.close()
        return exc.code, dict(exc.headers.items()), payload


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Dict[str, str], bytes]:
    merged = {"Content-Type": "application/json"}
    merged.update(headers or {})
    return request(
        url,
        method="POST",
        body=json.dumps(payload).encode("utf-8"),
        headers=merged,
        timeout=timeout,
    )


def login(base_url: str, token: str = TOKEN) -> str:
    """Exchange the shared token for a session cookie header value."""
    code, headers, _ = post_json(base_url + "/api/login", {"token": token})
    assert code == 200, "login with the correct token should succeed"
    raw = _header(headers, "Set-Cookie") or ""
    assert raw, "a successful login must set a session cookie"
    return raw.split(";", 1)[0]


def wait_until(predicate: Any, description: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for %s" % description)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def subsystems(config, bus) -> FakeSubsystems:
    return FakeSubsystems(config, bus)


@pytest.fixture
def server(subsystems):
    instance = JarvisServer(
        subsystems,
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        max_json_bytes=4096,
        max_audio_bytes=4096,
        sse_keepalive=1.0,
    )
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def base_url(server) -> str:
    return "http://127.0.0.1:%d" % server.port


# --------------------------------------------------------------------------- #
#  Tokens, addresses and sessions
# --------------------------------------------------------------------------- #
def test_generate_token_is_long_and_never_repeats():
    tokens = {generate_token() for _ in range(64)}
    assert len(tokens) == 64
    assert all(len(token) >= 40 for token in tokens)


def test_check_token_refuses_empty_none_and_near_misses():
    token = generate_token()
    assert check_token(token, token)
    assert not check_token("", token)
    assert not check_token(token, "")
    assert not check_token(None, token)
    assert not check_token(token, None)
    assert not check_token(None, None)
    assert not check_token(token + "x", token)
    assert not check_token(token[:-1], token)


def test_is_loopback_treats_every_interface_as_public():
    for local in ("127.0.0.1", "127.9.9.9", "::1", "[::1]", "localhost", "LocalHost"):
        assert is_loopback(local), "%s should be loopback" % local
    for public in ("0.0.0.0", "::", "", None, "192.168.1.42", "10.0.0.5", "example.com"):
        assert not is_loopback(public), "%r must not count as loopback" % (public,)


def test_load_or_create_token_persists_it_and_then_reuses_it(config):
    first = load_or_create_token(config, environ={})
    assert first
    path = token_path(config)
    assert path is not None and path.exists()
    assert path.read_text(encoding="utf-8").strip() == first
    assert load_or_create_token(config, environ={}) == first
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_environment_beats_the_stored_token(config):
    stored = load_or_create_token(config, environ={})
    from_env = load_or_create_token(config, environ={TOKEN_ENV_VAR: "from-the-environment"})
    assert from_env == "from-the-environment"
    assert from_env != stored


def test_load_or_create_token_can_be_told_not_to_create_one(config):
    assert load_or_create_token(config, environ={}, create=False) == ""
    path = token_path(config)
    assert path is not None and not path.exists()


def test_sessions_issue_validate_and_drop():
    sessions = Sessions(expire_after=60.0)
    session_id = sessions.create()
    assert sessions.validate(session_id)
    assert not sessions.validate("not-a-session")
    assert not sessions.validate("")
    assert not sessions.validate(None)
    assert sessions.drop(session_id)
    assert not sessions.validate(session_id)


def test_sessions_expire_but_use_renews_them(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(Sessions, "_now", staticmethod(lambda: clock["now"]))
    sessions = Sessions(expire_after=10.0)
    session_id = sessions.create()

    clock["now"] += 9.0
    assert sessions.validate(session_id), "a session in use must not expire"
    clock["now"] += 9.0
    assert sessions.validate(session_id), "using it should have renewed the clock"
    clock["now"] += 11.0
    assert not sessions.validate(session_id), "an abandoned session must expire"


def test_sessions_are_capped_and_thread_safe():
    sessions = Sessions(expire_after=60.0, max_sessions=16)
    created: List[str] = []
    lock = threading.Lock()

    def spam() -> None:
        mine = [sessions.create() for _ in range(20)]
        with lock:
            created.extend(mine)

    threads = [threading.Thread(target=spam) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(created) == 160
    assert len(set(created)) == 160, "session ids must never collide"
    assert len(sessions.active()) <= 16


# --------------------------------------------------------------------------- #
#  Refusing an unsafe bind
# --------------------------------------------------------------------------- #
def test_a_public_bind_without_a_token_is_refused_and_says_how_to_fix_it(subsystems):
    with pytest.raises(InsecureBindError) as excinfo:
        JarvisServer(subsystems, host="0.0.0.0", port=8765, token="")
    message = str(excinfo.value)
    assert TOKEN_ENV_VAR in message, "the message must name the environment variable"
    assert "--token" in message, "the message must name the command-line fix"
    assert "127.0.0.1" in message, "the message must offer staying private"
    assert "ssh -L" in message, "the message must offer the tunnel"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", ""])
def test_every_non_loopback_bind_demands_a_token(subsystems, host):
    with pytest.raises(InsecureBindError):
        JarvisServer(subsystems, host=host, port=0, token="")


def test_loopback_without_a_token_is_allowed(subsystems):
    instance = JarvisServer(subsystems, host="127.0.0.1", port=0, token="")
    assert instance.auth_required is False


def test_a_public_bind_with_a_token_is_accepted(subsystems):
    # Constructed but never started: nothing in this suite listens off-box.
    instance = JarvisServer(subsystems, host="0.0.0.0", port=0, token=TOKEN)
    assert instance.auth_required is True
    assert instance.running is False


def test_serve_refuses_a_public_bind_when_the_operator_disables_the_token(
    config, subsystems
):
    with pytest.raises(InsecureBindError):
        serve(config, subsystems=subsystems, host="0.0.0.0", port=0, token="", block=False)


# --------------------------------------------------------------------------- #
#  Authentication over the wire
# --------------------------------------------------------------------------- #
def test_status_is_public_and_leaks_no_token(server, base_url):
    code, _headers, body = request(base_url + "/api/status")
    assert code == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["auth_required"] is True
    assert payload["authenticated"] is False
    assert TOKEN not in body.decode("utf-8")


def test_an_unauthenticated_status_describes_the_browser_but_not_the_machine(
    server, base_url, config
):
    """A stranger who can reach the port must not learn the setup from it."""
    code, _headers, body = request(base_url + "/api/status")
    assert code == 200
    payload = json.loads(body)

    # Enough to render a login screen and explain the microphone rule...
    assert payload["audio"]["capture"] == "browser"
    assert payload["audio"]["microphone_note"]

    # ...and nothing about the machine itself.
    for leaked in ("model", "llm", "stt", "tts", "memory", "tools",
                   "security_mode", "data_dir"):
        assert leaked not in payload, "%r must not be public" % leaked
    text = body.decode("utf-8")
    assert config.llm.model not in text
    assert str(config.home()) not in text


def test_an_authenticated_status_reports_the_engines_and_the_model(
    server, base_url, config
):
    cookie = login(base_url)
    code, _headers, body = request(base_url + "/api/status", headers={"Cookie": cookie})
    assert code == 200
    payload = json.loads(body)
    assert payload["authenticated"] is True
    assert payload["model"] == config.llm.model
    assert payload["stt"] == "fake-stt"
    assert payload["tts"] == "fake-tts"
    assert payload["tools"] == 7
    assert payload["orchestrator"] is True
    assert TOKEN not in body.decode("utf-8")


def test_status_is_fully_detailed_when_no_token_is_configured(subsystems, config):
    """On loopback with no token the caller is already the owner."""
    instance = JarvisServer(subsystems, host="127.0.0.1", port=0, token="")
    instance.start()
    try:
        code, _headers, body = request("http://127.0.0.1:%d/api/status" % instance.port)
        assert code == 200
        payload = json.loads(body)
        assert payload["auth_required"] is False
        assert payload["model"] == config.llm.model
    finally:
        instance.stop()


def test_the_index_is_public(server, base_url):
    code, headers, body = request(base_url + "/")
    assert code == 200
    assert (_header(headers, "Content-Type") or "").startswith("text/html")
    assert _header(headers, "Content-Security-Policy")
    assert b"JARVIS" in body


def test_the_fallback_page_still_explains_browser_audio(monkeypatch):
    """With no client module installed the page must still be useful."""
    monkeypatch.setattr(server_app, "_CLIENT_MODULES", ())
    page = client_page({})
    assert "microphone" in page.lower()
    assert "/api/stt" in page and "/api/tts" in page
    assert "localhost" in page, "the HTTPS/localhost rule must survive the fallback"


def test_client_page_prefers_a_client_module_and_hands_it_the_status(monkeypatch):
    seen: List[Any] = []

    module = types.ModuleType("jarvis_test_client")

    def index_html(status: Any) -> str:
        seen.append(status)
        return "<!doctype html><title>from the client module</title>"

    module.index_html = index_html  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis_test_client", module)
    monkeypatch.setattr(server_app, "_CLIENT_MODULES", ("jarvis_test_client",))

    page = client_page({"name": "JARVIS"})
    assert "from the client module" in page
    assert seen == [{"name": "JARVIS"}]


def test_a_broken_client_module_falls_back_instead_of_500ing(monkeypatch):
    module = types.ModuleType("jarvis_broken_client")

    def index_html(status: Any) -> str:
        raise RuntimeError("the page template is malformed")

    module.index_html = index_html  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis_broken_client", module)
    monkeypatch.setattr(server_app, "_CLIENT_MODULES", ("jarvis_broken_client",))

    page = client_page({})
    assert "/api/stt" in page, "a broken client must not take the page down"


def test_the_index_serves_whatever_client_module_is_installed(server, base_url):
    code, _headers, body = request(base_url + "/")
    assert code == 200
    assert body.lstrip()[:9].lower() == b"<!doctype"


@pytest.mark.parametrize(
    "method,route",
    [
        ("POST", "/api/chat"),
        ("POST", "/api/tts"),
        ("POST", "/api/stt"),
        ("GET", "/api/tasks"),
        ("GET", "/api/stream"),
    ],
)
def test_protected_routes_refuse_an_unauthenticated_client(server, base_url, method, route):
    if method == "GET":
        code, _headers, body = request(base_url + route)
    else:
        code, _headers, body = post_json(base_url + route, {"text": "hello"})
    assert code == 401, "%s %s must require a session" % (method, route)
    assert b"/api/login" in body, "the refusal should say how to authenticate"


def test_a_wrong_token_is_rejected_and_issues_no_cookie(server, base_url):
    code, headers, body = post_json(base_url + "/api/login", {"token": "not-the-token"})
    assert code == 401
    assert _header(headers, "Set-Cookie") is None
    assert json.loads(body)["ok"] is False


def test_the_right_token_issues_a_session_that_then_works(server, base_url):
    code, headers, _body = post_json(base_url + "/api/login", {"token": TOKEN})
    assert code == 200
    raw = _header(headers, "Set-Cookie") or ""
    assert raw.startswith(SESSION_COOKIE + "=")
    assert "HttpOnly" in raw
    assert "SameSite=Strict" in raw
    assert "Secure" not in raw, "plain HTTP must not claim a Secure cookie"

    cookie = raw.split(";", 1)[0]
    code, _headers, body = request(base_url + "/api/tasks", headers={"Cookie": cookie})
    assert code == 200
    assert "tree" in json.loads(body)


def test_a_forged_session_cookie_is_refused(server, base_url):
    forged = "%s=%s" % (SESSION_COOKIE, "a" * 32)
    code, _headers, _body = request(base_url + "/api/tasks", headers={"Cookie": forged})
    assert code == 401


def test_login_compares_the_token_in_constant_time(server, base_url, monkeypatch):
    """`==` on strings leaks the token one byte at a time to anyone timing it."""
    real = hmac.compare_digest
    calls: List[Tuple[Any, Any]] = []

    def spy(left: Any, right: Any) -> bool:
        calls.append((left, right))
        return real(left, right)

    monkeypatch.setattr(hmac, "compare_digest", spy)

    code, _headers, _body = post_json(base_url + "/api/login", {"token": TOKEN})
    assert code == 200
    assert calls, "the token check must go through hmac.compare_digest"
    assert any(TOKEN.encode("utf-8") in pair for pair in calls)


def test_the_token_never_reaches_the_log(server, base_url, caplog):
    with caplog.at_level(logging.DEBUG):
        post_json(base_url + "/api/login", {"token": TOKEN})
        post_json(base_url + "/api/login", {"token": "wrong-token"})
        request(base_url + "/api/status?token=" + TOKEN)
        request(base_url + "/api/tasks", headers={"X-Jarvis-Token": TOKEN})
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert TOKEN not in logged
    assert "?token=" not in logged


def test_a_header_token_authenticates_a_non_browser_client(server, base_url):
    code, _headers, _body = request(
        base_url + "/api/tasks", headers={"Authorization": "Bearer " + TOKEN}
    )
    assert code == 200


def test_logout_invalidates_the_session_it_was_called_with(server, base_url):
    cookie = login(base_url)
    assert request(base_url + "/api/tasks", headers={"Cookie": cookie})[0] == 200

    code, headers, body = post_json(base_url + "/api/logout", {}, headers={"Cookie": cookie})
    assert code == 200
    assert json.loads(body)["ok"] is True
    assert "Max-Age=0" in (_header(headers, "Set-Cookie") or ""), "the cookie must be cleared"

    code, _headers, _body = request(base_url + "/api/tasks", headers={"Cookie": cookie})
    assert code == 401, "the session must be dead on the server, not just in the browser"


# --------------------------------------------------------------------------- #
#  DNS rebinding
# --------------------------------------------------------------------------- #
def test_a_tokenless_server_refuses_a_request_addressed_to_someone_elses_name(subsystems):
    """A web page can point its own hostname at 127.0.0.1 and reach a local port.

    Every other check passes when it does: the request really is same-origin, so
    the Origin test agrees, and SameSite is irrelevant because a server with no
    token needs no cookie.  Only the Host header still names the attacker.
    """
    instance = JarvisServer(subsystems, host="127.0.0.1", port=0, token="")
    instance.start()
    try:
        base = "http://127.0.0.1:%d" % instance.port
        code, _headers, body = post_json(
            base + "/api/chat",
            {"text": "delete every file"},
            headers={
                "Host": "rebound.example:%d" % instance.port,
                "Origin": "http://rebound.example:%d" % instance.port,
            },
        )
        assert code == 403, "a tokenless server must answer only to localhost"
        assert b"localhost" in body
        assert subsystems.orchestrator.calls == [], "the agent must never have been reached"

        # ...and the local browser it exists for is unaffected.
        code, _headers, _body = post_json(
            base + "/api/chat", {"text": "status"}, headers={"Host": "localhost:%d" % instance.port}
        )
        assert code == 200
        assert subsystems.orchestrator.calls == [("status", False)]
    finally:
        instance.stop()


def test_a_token_server_may_be_reached_under_any_name(server, base_url, subsystems):
    """The Host rule exists only to stand in for a missing token.

    With a token there is a real secret doing the work, and an operator behind a
    reverse proxy or a Tailscale name must not be locked out by a header check.
    """
    cookie = login(base_url)
    code, _headers, _body = post_json(
        base_url + "/api/chat",
        {"text": "status"},
        headers={"Cookie": cookie, "Host": "jarvis.tailnet.ts.net:%d" % server.port},
    )
    assert code == 200
    assert subsystems.orchestrator.calls == [("status", False)]


# --------------------------------------------------------------------------- #
#  Same-origin only
# --------------------------------------------------------------------------- #
def test_no_cors_header_is_ever_emitted(server, base_url):
    code, headers, _body = request(
        base_url + "/api/status", headers={"Origin": "http://evil.example"}
    )
    assert code == 200
    offenders = [key for key in headers if key.lower().startswith("access-control-")]
    assert not offenders, "the API must not advertise CORS: %s" % offenders


def test_a_cross_origin_post_is_refused(server, base_url, subsystems):
    cookie = login(base_url)
    code, _headers, _body = post_json(
        base_url + "/api/chat",
        {"text": "delete everything"},
        headers={"Cookie": cookie, "Origin": "http://evil.example"},
    )
    assert code == 403
    assert subsystems.orchestrator.calls == [], "the agent must never have been reached"


# --------------------------------------------------------------------------- #
#  Conversation
# --------------------------------------------------------------------------- #
def test_chat_reaches_the_orchestrator_without_speaking_on_the_server(
    server, base_url, subsystems
):
    cookie = login(base_url)
    subsystems.orchestrator.updates = ["Task 'read the logs' completed."]

    code, _headers, body = post_json(
        base_url + "/api/chat", {"text": "Status report."}, headers={"Cookie": cookie}
    )
    assert code == 200
    payload = json.loads(body)
    assert payload["reply"] == REPLY
    assert payload["task_updates"] == ["Task 'read the logs' completed."]

    assert subsystems.orchestrator.calls == [("Status report.", False)]
    assert subsystems.orchestrator.spoken == []
    assert subsystems.speech_queue.said == []


def test_chat_rejects_an_empty_request(server, base_url):
    cookie = login(base_url)
    code, _headers, _body = post_json(
        base_url + "/api/chat", {"text": "   "}, headers={"Cookie": cookie}
    )
    assert code == 400


def test_tasks_reports_the_agent_tree(server, base_url):
    cookie = login(base_url)
    code, _headers, body = request(base_url + "/api/tasks", headers={"Cookie": cookie})
    assert code == 200
    payload = json.loads(body)
    assert payload["tree"] == "(no background tasks)"
    assert payload["tasks"] == []
    assert payload["stats"]["tracked"] == 0


# --------------------------------------------------------------------------- #
#  Audio belongs to the client
# --------------------------------------------------------------------------- #
def test_status_advertises_browser_capture_and_never_server_devices(server, base_url):
    code, _headers, body = request(base_url + "/api/status")
    assert code == 200
    audio = json.loads(body)["audio"]
    assert audio["capture"] == "browser"
    assert audio["playback"] == "browser"
    assert audio["server_devices_used"] is False


def test_status_warns_that_a_lan_address_will_not_get_the_microphone(server, base_url):
    """The single biggest gotcha: getUserMedia needs HTTPS or a local host."""
    code, _headers, body = request(
        base_url + "/api/status", headers={"Host": "192.0.2.10:8765"}
    )
    assert code == 200
    audio = json.loads(body)["audio"]
    assert audio["secure_context"] is False
    note = audio["microphone_note"].lower()
    assert "https" in note
    assert "localhost" in note
    assert "ssh -l" in note, "the note should offer the tunnel workaround"


def test_status_reports_a_secure_context_when_the_client_is_local(server, base_url):
    code, _headers, body = request(base_url + "/api/status")
    audio = json.loads(body)["audio"]
    assert audio["secure_context"] is True
    assert audio["client_is_local"] is True


def test_stt_transcribes_a_wav_recorded_in_the_browser(server, base_url, subsystems):
    cookie = login(base_url)
    audio = wav_bytes([0.0] * 400, 16000)
    code, _headers, body = request(
        base_url + "/api/stt",
        method="POST",
        body=audio,
        headers={"Content-Type": "audio/wav", "Cookie": cookie},
    )
    assert code == 200
    payload = json.loads(body)
    assert payload["text"] == TRANSCRIPT_TEXT
    assert subsystems.stt.calls == 1
    assert subsystems.stt.audio[0][:4] == b"RIFF"
    assert subsystems.stt.rates == [16000]


def test_stt_names_ffmpeg_rather_than_raising_a_stack_trace(
    server, base_url, subsystems, monkeypatch
):
    monkeypatch.setattr(server_app, "which", lambda name: None)
    cookie = login(base_url)
    # A WebM/Opus header, the container Chrome's MediaRecorder produces.
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 64

    code, _headers, body = request(
        base_url + "/api/stt",
        method="POST",
        body=webm,
        headers={"Content-Type": "audio/webm;codecs=opus", "Cookie": cookie},
    )
    assert code == 503
    payload = json.loads(body)
    assert payload["ffmpeg_required"] is True
    assert "ffmpeg" in payload["error"].lower()
    assert "apt install ffmpeg" in payload["error"], "say how to install it"
    assert "Traceback" not in payload["error"]
    assert subsystems.stt.calls == 0, "undecodable audio must never reach the engine"


def test_stt_rejects_an_empty_body(server, base_url, subsystems):
    cookie = login(base_url)
    code, _headers, _body = request(
        base_url + "/api/stt",
        method="POST",
        body=b"",
        headers={"Content-Type": "audio/wav", "Cookie": cookie},
    )
    assert code == 400
    assert subsystems.stt.calls == 0


def test_tts_returns_audio_and_never_touches_the_server_speakers(
    server, base_url, subsystems
):
    cookie = login(base_url)
    code, headers, body = post_json(
        base_url + "/api/tts", {"text": "Good evening."}, headers={"Cookie": cookie}
    )
    assert code == 200
    assert _header(headers, "Content-Type") == "audio/wav"
    assert body[:4] == b"RIFF"

    assert subsystems.tts.synthesized == ["Good evening."]
    assert subsystems.tts.spoken == [], "the server engine must not have spoken"
    assert subsystems.speech_queue.said == [], "the server speech queue must be untouched"


def test_tts_rejects_an_empty_request(server, base_url, subsystems):
    cookie = login(base_url)
    code, _headers, _body = post_json(
        base_url + "/api/tts", {"text": ""}, headers={"Cookie": cookie}
    )
    assert code == 400
    assert subsystems.tts.synthesized == []


# --------------------------------------------------------------------------- #
#  Limits, failures and streaming
# --------------------------------------------------------------------------- #
def test_an_oversized_upload_is_refused_with_413_before_it_is_read(
    server, base_url, subsystems
):
    cookie = login(base_url)
    oversized = b"\x00" * (server.max_audio_bytes * 4)
    code, _headers, body = request(
        base_url + "/api/stt",
        method="POST",
        body=oversized,
        headers={"Content-Type": "audio/webm", "Cookie": cookie},
    )
    assert code == 413
    payload = json.loads(body)
    assert payload["limit"] == server.max_audio_bytes
    assert payload["bytes"] == len(oversized)
    assert subsystems.stt.calls == 0


def test_a_handler_that_raises_becomes_500_and_the_server_keeps_serving(
    server, base_url, subsystems
):
    subsystems.orchestrator.error = RuntimeError("the model exploded")
    cookie = login(base_url)

    code, _headers, body = post_json(
        base_url + "/api/chat", {"text": "hello"}, headers={"Cookie": cookie}
    )
    assert code == 500
    assert b"internal server error" in body
    assert b"the model exploded" not in body, "internals must not reach the client"

    subsystems.orchestrator.error = None
    code, _headers, _body = request(base_url + "/api/status")
    assert code == 200
    code, _headers, body = post_json(
        base_url + "/api/chat", {"text": "still there?"}, headers={"Cookie": cookie}
    )
    assert code == 200
    assert json.loads(body)["reply"] == REPLY


def test_unknown_routes_and_methods_answer_cleanly(server, base_url):
    cookie = login(base_url)
    code, _headers, _body = request(base_url + "/api/nonsense", headers={"Cookie": cookie})
    assert code == 404
    code, _headers, _body = request(base_url + "/api/chat", headers={"Cookie": cookie})
    assert code == 405


def test_the_stream_delivers_events_and_survives_a_client_hanging_up(
    server, base_url, subsystems
):
    cookie = login(base_url)
    bus = subsystems.bus

    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5.0)
    connection.request("GET", "/api/stream", headers={"Cookie": cookie})
    response = connection.getresponse()
    assert response.status == 200
    assert "text/event-stream" in (response.getheader("Content-Type") or "")

    wait_until(
        lambda: bus.handlers(Events.ASSISTANT_CHUNK), "the stream to subscribe to the bus"
    )
    bus.emit(Events.ASSISTANT_CHUNK, "Good evening, Sir.")

    seen = ""
    deadline = time.monotonic() + 5.0
    while "Good evening" not in seen and time.monotonic() < deadline:
        line = response.readline()
        if not line:
            break
        seen += line.decode("utf-8", "replace")
    assert "event: chunk" in seen
    assert "Good evening, Sir." in seen

    # Hang up mid-stream, exactly as a phone leaving Wi-Fi does.  The response
    # owns the socket here — an SSE reply carries "Connection: close", so
    # http.client handed the connection over to it and HTTPConnection.close()
    # alone would leave the descriptor open.
    response.close()
    connection.close()

    # The handler must notice and release its bus subscriptions.
    deadline = time.monotonic() + 2.0
    while bus.handlers(Events.ASSISTANT_CHUNK) and time.monotonic() < deadline:
        bus.emit(Events.ASSISTANT_CHUNK, "x" * 512)
        time.sleep(0.02)
    assert not bus.handlers(Events.ASSISTANT_CHUNK), (
        "a disconnected client left its subscription on the bus"
    )
    for channel, _name in server_app.STREAM_CHANNELS:
        assert not bus.handlers(channel), "channel %s leaked a handler" % channel

    # And the server is still perfectly healthy.
    code, _headers, _body = request(base_url + "/api/status")
    assert code == 200


def test_the_stream_lets_go_of_a_silent_client_without_waiting_for_an_event(
    server, base_url, subsystems
):
    """A tab closed at 3am must not hold a thread until the next event.

    The other disconnect test keeps emitting while it waits, so a failed write
    would notice on its own.  This one emits *nothing*: the only thing that can
    free the subscription is the handler actively looking for a closed peer.
    """
    cookie = login(base_url)
    bus = subsystems.bus

    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5.0)
    try:
        sock.sendall(
            (
                "GET /api/stream HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                "Cookie: %s\r\nConnection: close\r\n\r\n" % (server.port, cookie)
            ).encode("utf-8")
        )
        assert b"200" in sock.recv(256)
        wait_until(lambda: bus.handlers(Events.ASSISTANT_CHUNK), "the stream to subscribe")
    finally:
        sock.close()

    deadline = time.monotonic() + 2.0
    while bus.handlers(Events.ASSISTANT_CHUNK) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not bus.handlers(Events.ASSISTANT_CHUNK), (
        "a silent stream held its bus subscription; nothing was ever emitted, so "
        "only an active check for a closed peer could have released it"
    )


def test_head_on_the_stream_answers_instead_of_streaming(server, base_url, subsystems):
    """HEAD asks what a GET would answer, not for the answer itself."""
    cookie = login(base_url)
    bus = subsystems.bus

    started = time.monotonic()
    code, headers, body = request(base_url + "/api/stream", method="HEAD",
                                  headers={"Cookie": cookie}, timeout=5.0)
    assert code == 200
    assert time.monotonic() - started < 3.0, "HEAD must not hold the connection open"
    assert body == b"", "a HEAD response carries no body"
    assert "text/event-stream" in (_header(headers, "Content-Type") or "")
    assert not bus.handlers(Events.ASSISTANT_CHUNK), (
        "HEAD must not subscribe to the bus"
    )


def test_a_half_open_connection_is_dropped_instead_of_pinning_a_thread(
    server, monkeypatch
):
    """An unfinished request line arrives before any token is ever checked.

    Without a socket timeout it holds a handler thread for ever, which makes
    exhausting the machine something anyone who can reach the port can do
    without authenticating at all.
    """
    assert server_app.JarvisHTTPRequestHandler.timeout, "the handler must set a timeout"
    monkeypatch.setattr(server_app.JarvisHTTPRequestHandler, "timeout", 0.3)

    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5.0)
    try:
        sock.sendall(b"GET /api/sta")           # no newline: the request never ends
        sock.settimeout(2.0)
        assert sock.recv(256) == b"", "the server should have hung up on a stalled client"
    finally:
        sock.close()


def test_the_stream_reports_task_events(server, base_url, subsystems):
    cookie = login(base_url)
    bus = subsystems.bus

    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5.0)
    response = None
    try:
        connection.request("GET", "/api/stream", headers={"Cookie": cookie})
        response = connection.getresponse()
        assert response.status == 200

        wait_until(lambda: bus.handlers(Events.TASK_DONE), "the stream to subscribe")
        bus.emit(Events.TASK_DONE, {"task_id": "task-1", "goal": "read the logs"})

        seen = ""
        deadline = time.monotonic() + 5.0
        while "read the logs" not in seen and time.monotonic() < deadline:
            line = response.readline()
            if not line:
                break
            seen += line.decode("utf-8", "replace")
        assert "event: task_done" in seen
        assert "read the logs" in seen
    finally:
        if response is not None:
            response.close()
        connection.close()


def test_eight_concurrent_clients_all_get_an_answer(server, base_url, subsystems):
    cookie = login(base_url)
    results: List[Tuple[int, str]] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        code, _headers, body = post_json(
            base_url + "/api/chat",
            {"text": "ping %d" % index},
            headers={"Cookie": cookie},
            timeout=20.0,
        )
        with lock:
            results.append((code, body.decode("utf-8")))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=25.0)
    for thread in threads:
        assert not thread.is_alive(), "a concurrent request never finished"

    assert len(results) == 8
    assert all(code == 200 for code, _ in results)
    assert all(REPLY in body for _code, body in results)
    assert len(subsystems.orchestrator.calls) == 8


# --------------------------------------------------------------------------- #
#  Lifecycle
# --------------------------------------------------------------------------- #
def test_serve_starts_a_loopback_server_and_stops_it_cleanly(config, subsystems):
    instance = serve(config, subsystems=subsystems, host="127.0.0.1", port=0, block=False)
    try:
        assert instance.running
        assert instance.token, "serve() must resolve or generate a token"
        code, _headers, _body = request("http://127.0.0.1:%d/api/status" % instance.port)
        assert code == 200
    finally:
        instance.stop()
    assert not instance.running


def test_stopping_twice_is_harmless(subsystems):
    instance = JarvisServer(subsystems, host="127.0.0.1", port=0, token=TOKEN)
    instance.start()
    instance.stop()
    instance.stop()
    assert not instance.running


def test_the_server_works_as_a_context_manager(subsystems):
    with JarvisServer(subsystems, host="127.0.0.1", port=0, token=TOKEN) as instance:
        assert instance.running
        code, _headers, _body = request("http://127.0.0.1:%d/api/status" % instance.port)
        assert code == 200
    assert not instance.running


def test_shutdown_invalidates_every_session(subsystems):
    instance = JarvisServer(subsystems, host="127.0.0.1", port=0, token=TOKEN)
    instance.start()
    base = "http://127.0.0.1:%d" % instance.port
    cookie = login(base)
    instance.stop()
    assert instance.sessions.active() == []
    assert not instance.sessions.validate(cookie.split("=", 1)[1])
