"""A stdlib HTTP server that puts JARVIS in any browser on the network.

The whole assistant — model, tools, memory, machine control — stays on the box
this module runs on.  Other devices are thin clients: a phone, a laptop, a
tablet, anything with a browser and no installation at all.

**Audio is captured in the client's browser, never on the server.**  That is the
point of the feature rather than an implementation detail.  JARVIS's microphone
and speakers are wired to the machine it runs on; if a laptop two rooms away
asked the server to listen, it would listen in the wrong room.  So the browser
records with ``getUserMedia``/``MediaRecorder``, POSTs the clip to
``/api/stt``, and plays the reply that ``/api/tts`` hands back.  No route in
this module touches a server-side audio device, and ``/api/tts`` deliberately
*synthesises* rather than *speaks*.

**The single biggest gotcha** for that design is a browser rule, not a bug:
``getUserMedia`` is only offered in a secure context — HTTPS, or an address the
browser already treats as local (``localhost`` / ``127.0.0.1``).  Over plain
HTTP to a LAN address the microphone button will simply never work.
``/api/status`` reports this per-request as ``audio.secure_context`` with a note
naming the two fixes (TLS certificate, or an SSH tunnel), so the client can
explain the situation instead of appearing broken.

**Access control.**  Nothing here limits what the owner may do — the security
policy stays ``mode="open"``, with no protected paths and no confirmation
prompts.  What it limits is who may connect, because this port fronts an
unrestricted shell-and-filesystem agent.  The server binds loopback by default,
demands a shared token the moment it does not, compares tokens in constant
time, serves no CORS headers at all, and refuses point-blank to start on a
public interface without a secret.

Stdlib only: :mod:`http.server`, :mod:`ssl`, :mod:`json`, :mod:`hmac`,
:mod:`secrets`.  Streaming is Server-Sent Events over ordinary HTTP, which needs
no third-party package on either end.
"""

from __future__ import annotations

import http.cookies
import importlib
import json
import logging
import os
import queue
import select
import socket
import ssl
import subprocess
import tempfile
import threading
import urllib.parse
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.events import EventBus, Events, get_bus
from ..core.platform_utils import which
from .auth import (
    DEFAULT_SESSION_TTL,
    Sessions,
    check_token,
    is_loopback,
    load_or_create_token,
    token_path,
)

log = logging.getLogger(__name__)


#: Loopback by default.  Reaching JARVIS from another device is an explicit act
#: (``--host 0.0.0.0``), and that act requires a token.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Session cookie name.  HttpOnly + SameSite=Strict, and Secure under TLS.
SESSION_COOKIE = "jarvis_session"

#: Body caps.  Anything larger is refused with 413 *before* it is read, so an
#: upload cannot be used to exhaust memory on a 32 GB box running a 27B model.
MAX_JSON_BYTES = 256 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: How long an idle SSE connection waits before emitting a keepalive comment.
SSE_KEEPALIVE_SECONDS = 15.0
#: Slice used while waiting, so shutdown is prompt without a short keepalive.
_SSE_POLL_SECONDS = 0.25
#: Events buffered per SSE client before the slowest ones start being dropped.
_SSE_QUEUE_SIZE = 512

#: Sample rate every bundled STT engine expects.
_STT_SAMPLE_RATE = 16000

#: How long ffmpeg gets to transcode one uploaded clip.
_FFMPEG_TIMEOUT = 60.0

#: Bus channels forwarded to browsers, mapped to their SSE event names.
STREAM_CHANNELS: Tuple[Tuple[str, str], ...] = (
    (Events.ASSISTANT_CHUNK, "chunk"),
    (Events.ASSISTANT_REPLY, "reply"),
    (Events.USER_UTTERANCE, "utterance"),
    (Events.TASK_CREATED, "task_created"),
    (Events.TASK_UPDATE, "task_update"),
    (Events.TASK_DONE, "task_done"),
    (Events.TASK_FAILED, "task_failed"),
    (Events.TOOL_CALL, "tool_call"),
    (Events.TOOL_RESULT, "tool_result"),
    (Events.ERROR, "error"),
    (Events.SHUTDOWN, "shutdown"),
)

#: The browser client is authored as its own module.  Import it lazily and by
#: name so this server works with or without it.  A module may expose either a
#: callable (optionally taking the status dict) or a plain HTML string.
_CLIENT_MODULES = (
    "jarvis.server.client",
    "jarvis.server.web",
    "jarvis.server.ui",
    "jarvis.web.client",
    "jarvis.web",
)
_CLIENT_CALLABLES = (
    "index_html", "render_index", "render_page", "client_html", "page_html", "html",
)
_CLIENT_CONSTANTS = ("INDEX_HTML", "CLIENT_HTML", "PAGE_HTML")

#: The page is a single self-contained document served from this origin.
_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "media-src 'self' data: blob:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)

_AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
}


class InsecureBindError(RuntimeError):
    """Raised when a public bind is requested with no shared token.

    Deliberately fatal rather than a warning: the alternative is an
    unauthenticated remote shell on the local network, and a message in a log
    nobody is reading is not a mitigation.
    """


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _redact(text: Any) -> str:
    """Mask token-shaped substrings before anything reaches a log or a reply."""
    try:
        from ..llm.models import redact

        return redact(text)
    except Exception:  # noqa: BLE001 - redaction must never be the thing that fails
        return "" if text is None else str(text)


def _version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary payloads (tasks, transcripts, enums) into JSON types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return {str(k): _jsonable(v) for k, v in asdict(value).items()}
        except Exception:  # noqa: BLE001 - a non-serialisable field is not fatal
            log.debug("asdict failed for %r", type(value).__name__, exc_info=True)
    return str(value)


def _looks_like_wav(data: bytes) -> bool:
    return bool(data) and data[:4] == b"RIFF" and b"WAVE" in data[:16]


def _engine_name(engine: Any) -> str:
    if engine is None:
        return "unavailable"
    return str(getattr(engine, "name", type(engine).__name__))


def _hostname_of(host_header: str) -> str:
    """The bare host out of a ``Host:`` header, IPv6 literals included."""
    text = (host_header or "").strip()
    if not text:
        return ""
    if text.startswith("["):
        end = text.find("]")
        if end > 0:
            return text[1:end]
        return text
    return text.split(":", 1)[0]


def client_page(status: Optional[Dict[str, Any]] = None) -> str:
    """The single-page browser client, or an honest placeholder.

    The page itself is authored in its own module; this only locates it.  A
    module may expose a callable named ``index_html`` / ``render_index`` /
    ``render_page`` / ``client_html`` / ``page_html`` / ``html`` — taking the
    status dict or nothing — or a plain ``INDEX_HTML`` string.
    """
    payload = status or {}
    for module_name in _CLIENT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        except Exception:  # noqa: BLE001 - a broken client must not break the API
            log.exception("client module %s failed to import", module_name)
            continue
        for attr in _CLIENT_CALLABLES:
            candidate = getattr(module, attr, None)
            if not callable(candidate):
                continue
            for args in ((payload,), ()):
                try:
                    rendered = candidate(*args)
                except TypeError:
                    continue
                except Exception:  # noqa: BLE001
                    log.exception("%s.%s() failed", module_name, attr)
                    break
                if isinstance(rendered, (bytes, bytearray)):
                    rendered = bytes(rendered).decode("utf-8", "replace")
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
        for attr in _CLIENT_CONSTANTS:
            candidate = getattr(module, attr, None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return _FALLBACK_PAGE


_FALLBACK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JARVIS</title>
<style>
 body{font:16px/1.6 system-ui,sans-serif;max-width:44rem;margin:3rem auto;padding:0 1.2rem;
      background:#0e1116;color:#dfe4ea}
 code,pre{background:#1a1f27;padding:.15rem .4rem;border-radius:.25rem}
 pre{padding:.8rem;overflow-x:auto} h1{font-weight:600;letter-spacing:.04em}
 .note{border-left:3px solid #4c7dff;padding-left:1rem;color:#aeb6c2}
</style></head><body>
<h1>JARVIS</h1>
<p>The server is running, but the browser client module is not installed, so
there is no interface to show you. The HTTP API is live:</p>
<pre>POST /api/login   {"token": "..."}   -&gt; session cookie
GET  /api/status                       health, engines, audio advice
POST /api/chat    {"text": "..."}      -&gt; {"reply": ..., "task_updates": [...]}
GET  /api/stream                       server-sent events
POST /api/stt     raw audio body       -&gt; {"text": "..."}
POST /api/tts     {"text": "..."}      -&gt; audio bytes for THIS device to play
GET  /api/tasks                        the background agent tree</pre>
<p class="note">Recording happens in your browser and playback happens on your
speakers &mdash; the microphone attached to the JARVIS machine is never opened
for a remote session. Browsers only offer the microphone over HTTPS or on
<code>localhost</code>; see <code>audio.secure_context</code> in
<code>/api/status</code> if the microphone never appears.</p>
</body></html>
"""


# --------------------------------------------------------------------------- #
#  ffmpeg transcoding for browser audio containers
# --------------------------------------------------------------------------- #
def _ffmpeg_missing_message(kind: str) -> str:
    return (
        "This client uploaded %s audio. JARVIS decodes WAV on its own, but every "
        "other container (the WebM/Opus that Chrome and Edge record, the Ogg that "
        "Firefox records, the MP4 that Safari records) needs ffmpeg, and ffmpeg is "
        "not installed on the machine running JARVIS. Install it and restart "
        "(Debian/Ubuntu: sudo apt install ffmpeg; Fedora: sudo dnf install ffmpeg; "
        "Arch: sudo pacman -S ffmpeg; Windows: winget install Gyan.FFmpeg), or have "
        "the client record WAV." % (kind or "non-WAV")
    )


def transcode_to_wav(
    data: bytes,
    content_type: str = "",
    *,
    sample_rate: int = _STT_SAMPLE_RATE,
    timeout: float = _FFMPEG_TIMEOUT,
) -> Tuple[Optional[bytes], str]:
    """Turn an uploaded clip into WAV bytes.  Returns ``(wav, error)``.

    WAV passes straight through.  Anything else goes through ffmpeg via two
    temporary files — ffmpeg cannot write correct RIFF sizes to a pipe, and a
    WAV whose header claims the wrong length decodes to silence.  On failure the
    error string names ffmpeg and how to install it; it never carries a
    traceback back to the browser.
    """
    if not data:
        return None, "the upload was empty"
    if _looks_like_wav(data):
        return data, ""

    kind = (content_type or "").split(";", 1)[0].strip().lower()
    binary = which("ffmpeg")
    if not binary:
        return None, _ffmpeg_missing_message(kind)

    in_path: Optional[str] = None
    out_path: Optional[str] = None
    try:
        handle, in_path = tempfile.mkstemp(prefix="jarvis-upload-", suffix=".bin")
        os.close(handle)
        handle, out_path = tempfile.mkstemp(prefix="jarvis-upload-", suffix=".wav")
        os.close(handle)
        Path(in_path).write_bytes(data)

        argv = [
            binary, "-hide_banner", "-loglevel", "error", "-y",
            "-i", in_path,
            "-ac", "1", "-ar", str(int(sample_rate)),
            "-f", "wav", out_path,
        ]
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            return None, "could not run ffmpeg: %s" % _redact(exc)

        try:
            _stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg ignored SIGKILL while decoding an upload")
            return None, "ffmpeg took longer than %.0fs to decode the clip" % timeout

        if proc.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
            tail = detail[-1] if detail else "no diagnostic"
            return None, "ffmpeg could not decode %s: %s" % (kind or "the upload", _redact(tail))

        wav = Path(out_path).read_bytes()
        if not _looks_like_wav(wav):
            return None, "ffmpeg produced something that is not a WAV file"
        return wav, ""
    except OSError as exc:
        return None, "could not transcode the upload: %s" % _redact(exc)
    finally:
        for path in (in_path, out_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    log.debug("could not remove the temporary file %s", path, exc_info=True)


# --------------------------------------------------------------------------- #
#  Request handler
# --------------------------------------------------------------------------- #
_ROUTES: Dict[Tuple[str, str], str] = {
    ("GET", "/"): "_get_index",
    ("GET", "/api/status"): "_get_status",
    ("POST", "/api/login"): "_post_login",
    ("POST", "/api/logout"): "_post_logout",
    ("POST", "/api/chat"): "_post_chat",
    ("GET", "/api/stream"): "_get_stream",
    ("POST", "/api/stt"): "_post_stt",
    ("POST", "/api/tts"): "_post_tts",
    ("GET", "/api/tasks"): "_get_tasks",
}

#: Reachable without a session.  Everything else needs one.
PUBLIC_ROUTES = frozenset({"/", "/api/status", "/api/login"})


class JarvisHTTPRequestHandler(BaseHTTPRequestHandler):
    """One request.  Never raises into the server loop."""

    server_version = "JARVIS"
    sys_version = ""
    # Keep-alive matters for SSE and for a phone making many small calls.
    protocol_version = "HTTP/1.1"
    #: Seconds any single socket operation may block.  Without it a client that
    #: opens a connection and never finishes its request line pins a handler
    #: thread for ever, and it does so *before* the request reaches the point
    #: where the token is checked — so it is an unauthenticated way to exhaust
    #: the machine.  Comfortably longer than the SSE keepalive, so a live stream
    #: never trips it, and a slow model reply is unaffected because this bounds
    #: socket operations, not the time a handler spends thinking.
    timeout = 60.0

    # -- logging ------------------------------------------------------------ #
    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        """Log method, path and status at INFO — never the token or the body.

        The query string is dropped rather than redacted: a secret that reaches
        a log at all has already been mishandled.
        """
        route = urllib.parse.urlsplit(self.path).path
        log.info(
            "%s %s %s -> %s",
            self.address_string(),
            self.command,
            _redact(route),
            getattr(code, "value", code),
        )

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - trivial
        try:
            log.info("%s %s", self.address_string(), _redact(fmt % args))
        except (TypeError, ValueError):
            log.info("%s %s", self.address_string(), _redact(fmt))

    def log_error(self, fmt: str, *args: Any) -> None:
        # An idle keep-alive connection hitting `timeout` is routine housekeeping,
        # not something to warn an operator about once per closed browser tab.
        emit = log.debug if str(fmt).startswith("Request timed out") else log.warning
        try:
            emit("%s %s", self.address_string(), _redact(fmt % args))
        except (TypeError, ValueError):
            emit("%s %s", self.address_string(), _redact(fmt))

    # -- plumbing ----------------------------------------------------------- #
    @property
    def jarvis(self) -> Optional["JarvisServer"]:
        return getattr(self.server, "jarvis", None)

    def _route(self) -> str:
        path = urllib.parse.urlsplit(self.path).path
        return path.rstrip("/") or "/"

    def _security_headers(self) -> Dict[str, str]:
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        }
        # Deliberately no Access-Control-Allow-Origin, wildcard or otherwise:
        # the client is served from this same origin and nothing else may call
        # these endpoints from a page the owner did not open.
        return headers

    def _begin(
        self,
        status: Any,
        content_type: str,
        length: Optional[int] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        self._responded = True
        self.send_response(int(getattr(status, "value", status)))
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(int(length)))
        for key, value in self._security_headers().items():
            self.send_header(key, value)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _send_bytes(
        self,
        status: Any,
        body: bytes,
        content_type: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        payload = body or b""
        self._begin(status, content_type, length=len(payload), extra=extra)
        if self.command != "HEAD" and payload:
            self.wfile.write(payload)

    def _send_json(
        self,
        status: Any,
        payload: Dict[str, Any],
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", extra=extra)

    # -- authentication ----------------------------------------------------- #
    def _session_id(self) -> str:
        raw = self.headers.get("Cookie")
        if not raw:
            return ""
        jar: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
        try:
            jar.load(raw)
        except http.cookies.CookieError:
            return ""
        morsel = jar.get(SESSION_COOKIE)
        return morsel.value if morsel is not None else ""

    def _supplied_token(self) -> str:
        """A token from a header, for scripts and ``curl`` — never from the URL."""
        header = self.headers.get("Authorization") or ""
        if header[:7].lower() == "bearer ":
            return header[7:].strip()
        return (self.headers.get("X-Jarvis-Token") or "").strip()

    def _authenticated(self) -> bool:
        server = self.jarvis
        if server is None:
            return False
        if not server.auth_required:
            return True
        if server.sessions.validate(self._session_id()):
            return True
        return check_token(self._supplied_token(), server.token)

    def _same_origin(self) -> bool:
        """Refuse a state-changing call made from someone else's page.

        A browser attaches ``Origin`` to every cross-site POST.  Requiring it to
        match ``Host`` is the same-origin rule, enforced without ever emitting a
        CORS header.  Non-browser clients send no ``Origin`` and are unaffected.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        if origin == "null":
            # A sandboxed iframe or a file:// page. Never this client.
            return False
        try:
            parsed = urllib.parse.urlsplit(origin)
        except ValueError:
            return False
        host = (self.headers.get("Host") or "").strip().lower()
        return bool(parsed.netloc) and parsed.netloc.lower() == host

    # -- request bodies ----------------------------------------------------- #
    def _read_body(self, limit: int) -> Optional[bytes]:
        """Read the body, or answer 413/400 and return ``None``.

        An oversized body is refused from its ``Content-Length`` alone, so the
        bytes are never pulled into memory.
        """
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED,
                {"error": "chunked uploads are not supported; send Content-Length"},
                extra={"Connection": "close"},
            )
            return None
        raw = self.headers.get("Content-Length")
        if raw is None:
            return b""
        try:
            length = int(raw)
        except (TypeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return None
        if length < 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "negative Content-Length"})
            return None
        if length > limit:
            # Answer and hang up rather than draining the upload.
            self.close_connection = True
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {
                    "error": "the request body is too large",
                    "bytes": length,
                    "limit": limit,
                },
                extra={"Connection": "close"},
            )
            return None
        if length == 0:
            return b""
        data = self.rfile.read(length)
        if len(data) != length:
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "truncated request body"})
            return None
        return data

    def _read_json(self, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if limit is None:
            server = self.jarvis
            limit = server.max_json_bytes if server is not None else MAX_JSON_BYTES
        body = self._read_body(limit)
        if body is None:
            return None
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "invalid JSON: %s" % _redact(exc)}
            )
            return None
        if not isinstance(payload, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "the request body must be a JSON object"}
            )
            return None
        return payload

    # -- dispatch ----------------------------------------------------------- #
    def do_GET(self) -> None:
        self._safely("GET")

    def do_HEAD(self) -> None:
        self._safely("HEAD")

    def do_POST(self) -> None:
        self._safely("POST")

    def do_OPTIONS(self) -> None:
        # No preflight support, by design: same-origin only means there is
        # nothing to negotiate.
        self._responded = False
        self._send_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": "this API is same-origin only and serves no CORS preflight"},
            extra={"Allow": "GET, HEAD, POST"},
        )

    def _safely(self, method: str) -> None:
        """Run a route, converting any escape into a 500 the server survives."""
        self._responded = False
        try:
            self._dispatch(method)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            log.info("client hung up during %s %s", method, _redact(self._route()))
            self.close_connection = True
        except Exception:  # noqa: BLE001 - one bad request must not end the server
            log.exception("unhandled error serving %s %s", method, _redact(self._route()))
            if not getattr(self, "_responded", False):
                try:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "internal server error; see the JARVIS log"},
                    )
                except Exception:  # noqa: BLE001
                    self.close_connection = True
            else:
                # Headers are already on the wire; the only honest signal left
                # is to close the connection.
                self.close_connection = True

    def _dispatch(self, method: str) -> None:
        server = self.jarvis
        if server is None:  # pragma: no cover - only if constructed by hand
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "server not ready"})
            return

        route = self._route()
        lookup = "GET" if method == "HEAD" else method

        # With no token the only legitimate client is on this machine, so insist
        # the browser agrees about which machine it is talking to.  A page on the
        # public internet can point its own hostname at 127.0.0.1 (DNS
        # rebinding); the request is then genuinely same-origin, so the Origin
        # check below passes and SameSite=Strict is irrelevant because no cookie
        # is needed.  The Host header is the one thing that still names the
        # attacker's site, and refusing it costs a local client nothing.
        if not server.auth_required:
            addressed_to = _hostname_of(self.headers.get("Host") or "")
            if addressed_to and not is_loopback(addressed_to):
                log.warning("refused a request addressed to a non-loopback host name")
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {
                        "error": "no access token is configured, so this server "
                                 "answers only to localhost",
                        "hint": "set JARVIS_SERVER_TOKEN to reach it under any "
                                "other name",
                    },
                )
                return

        if lookup == "POST" and not self._same_origin():
            log.warning("refused a cross-origin POST to %s", _redact(route))
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "cross-origin requests are refused; this API is same-origin only"},
            )
            return

        if route not in PUBLIC_ROUTES and not self._authenticated():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {
                    "error": "authentication required",
                    "hint": "POST {\"token\": \"...\"} to /api/login to get a session",
                },
            )
            return

        name = _ROUTES.get((lookup, route))
        if name is None:
            known = {r for (_m, r) in _ROUTES}
            if route in known:
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "%s is not allowed on %s" % (method, route)},
                )
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"error": "no such endpoint: %s" % route}
                )
            return
        getattr(self, name)()

    # -- routes ------------------------------------------------------------- #
    def _get_index(self) -> None:
        server = self.jarvis
        status = server.status_payload(
            authenticated=self._authenticated(),
            host_header=self.headers.get("Host") or "",
        )
        body = client_page(status).encode("utf-8")
        self._send_bytes(
            HTTPStatus.OK,
            body,
            "text/html; charset=utf-8",
            extra={"Content-Security-Policy": _CSP, "Cache-Control": "no-store"},
        )

    def _get_status(self) -> None:
        server = self.jarvis
        payload = server.status_payload(
            authenticated=self._authenticated(),
            host_header=self.headers.get("Host") or "",
        )
        self._send_json(HTTPStatus.OK, payload, extra={"Cache-Control": "no-store"})

    def _post_login(self) -> None:
        server = self.jarvis
        payload = self._read_json()
        if payload is None:
            return
        supplied = payload.get("token")
        if not check_token(supplied, server.token):
            log.warning("rejected a login from %s", self.address_string())
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"ok": False, "error": "invalid token"},
            )
            return
        session_id = server.sessions.create()
        cookie = "%s=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=%d" % (
            SESSION_COOKIE,
            session_id,
            int(server.sessions.expire_after),
        )
        if server.tls:
            cookie += "; Secure"
        log.info("issued a session to %s", self.address_string())
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "expires_in": int(server.sessions.expire_after)},
            extra={"Set-Cookie": cookie},
        )

    def _post_logout(self) -> None:
        server = self.jarvis
        server.sessions.drop(self._session_id())
        expired = "%s=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0" % SESSION_COOKIE
        self._send_json(HTTPStatus.OK, {"ok": True}, extra={"Set-Cookie": expired})

    def _post_chat(self) -> None:
        server = self.jarvis
        payload = self._read_json()
        if payload is None:
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "'text' is required"})
            return
        agent = server.orchestrator
        if agent is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "JARVIS has no orchestrator; run 'jarvis doctor' on the server"},
            )
            return
        # speak=False without exception: the reply belongs on the CLIENT's
        # speakers, and the server's own voice must stay silent for a remote
        # session. The browser fetches the audio from /api/tts.
        reply = agent.chat(text, speak=False)
        updates: List[Any] = []
        pending = getattr(agent, "pending_updates", None)
        if callable(pending):
            try:
                updates = list(pending())
            except Exception:  # noqa: BLE001 - a report must not lose the reply
                log.exception("pending_updates() failed")
        self._send_json(HTTPStatus.OK, {"reply": reply, "task_updates": updates})

    def _get_tasks(self) -> None:
        server = self.jarvis
        agent = server.orchestrator
        if agent is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE, {"error": "JARVIS has no orchestrator"}
            )
            return
        payload: Dict[str, Any] = {"tree": "", "tasks": [], "stats": {}}
        render = getattr(agent, "task_tree", None)
        if callable(render):
            try:
                payload["tree"] = render()
            except Exception:  # noqa: BLE001
                log.exception("task_tree() failed")
        tasks = getattr(agent, "tasks", None)
        if tasks is not None:
            try:
                payload["tasks"] = [
                    {
                        "task_id": getattr(task, "id", ""),
                        "goal": getattr(task, "goal", ""),
                        "state": getattr(getattr(task, "state", None), "value", ""),
                        "depth": (getattr(task, "metadata", {}) or {}).get("depth", 0),
                        "parent_id": (getattr(task, "metadata", {}) or {}).get("parent_id"),
                        "error": getattr(task, "error", None),
                    }
                    for task in tasks.list()
                ]
            except Exception:  # noqa: BLE001
                log.exception("listing tasks failed")
            stats = getattr(tasks, "stats", None)
            if callable(stats):
                try:
                    payload["stats"] = stats()
                except Exception:  # noqa: BLE001
                    log.exception("task stats failed")
        self._send_json(HTTPStatus.OK, payload)

    def _post_stt(self) -> None:
        """Transcribe audio recorded by the CLIENT's browser.

        The server's own microphone is never opened here — it is in the wrong
        room by definition.
        """
        server = self.jarvis
        body = self._read_body(server.max_audio_bytes)
        if body is None:
            return
        if not body:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "the audio body was empty"})
            return

        content_type = self.headers.get("Content-Type") or ""
        wav, error = transcode_to_wav(body, content_type)
        if wav is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": error, "ffmpeg_required": which("ffmpeg") is None},
            )
            return

        engine = server.stt
        if engine is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "no speech-to-text engine is available on the JARVIS machine"},
            )
            return
        try:
            transcript = engine.transcribe(wav, sample_rate=_STT_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - the contract says it should not
            log.exception("transcription failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "transcription failed: %s" % _redact(exc)},
            )
            return

        text = str(getattr(transcript, "text", "") or "")
        payload: Dict[str, Any] = {
            "text": text,
            "engine": _engine_name(engine),
            "language": getattr(transcript, "language", None),
            "confidence": getattr(transcript, "confidence", None),
            "bytes": len(body),
        }
        if not text and _engine_name(engine) in ("null", "unavailable"):
            payload["note"] = (
                "No speech-to-text engine is installed on the JARVIS machine, so "
                "nothing can be transcribed. Install faster-whisper there "
                "(pip install faster-whisper) and restart."
            )
        self._send_json(HTTPStatus.OK, payload)

    def _post_tts(self) -> None:
        """Synthesise a reply and hand the bytes back for the CLIENT to play.

        This must never reach the speech queue: that plays on the speakers wired
        to the server, which for a remote session means talking to an empty
        room.
        """
        server = self.jarvis
        payload = self._read_json()
        if payload is None:
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "'text' is required"})
            return
        engine = server.tts
        synthesize = getattr(engine, "synthesize", None)
        if engine is None or not callable(synthesize):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "no text-to-speech engine is available on the JARVIS machine"},
            )
            return
        try:
            data = synthesize(text)
        except Exception as exc:  # noqa: BLE001
            log.exception("synthesis failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "synthesis failed: %s" % _redact(exc)},
            )
            return
        if not data:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "the text-to-speech engine produced no audio"},
            )
            return
        fmt = "wav" if _looks_like_wav(data) else str(
            getattr(engine, "output_format", "wav") or "wav"
        ).lower()
        content_type = _AUDIO_CONTENT_TYPES.get(fmt, "application/octet-stream")
        self._send_bytes(
            HTTPStatus.OK,
            bytes(data),
            content_type,
            extra={"Cache-Control": "no-store", "X-Jarvis-Audio-Format": fmt},
        )

    # -- server-sent events ------------------------------------------------- #
    def _sse_write(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        try:
            self.wfile.flush()
        except AttributeError:  # pragma: no cover - wfile is unbuffered in 3.6+
            pass

    def _sse_event(self, name: str, payload: Any) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False)
        self._sse_write("event: %s\ndata: %s\n\n" % (name, body))

    def _client_gone(self) -> bool:
        """True once the peer has closed its end of an idle stream.

        Waiting for a write to fail is not enough on its own: a browser tab
        closed at 3am on a silent bus would hold a thread and a bus
        subscription until the next event, which might be hours away.  A
        zero-timeout select plus a peek costs nothing and notices the FIN at
        once.  TLS sockets are skipped because MSG_PEEK is not meaningful
        through the record layer; those fall back to the failed write.
        """
        sock = self.connection
        if sock is None or isinstance(sock, ssl.SSLSocket):
            return False
        try:
            readable, _writable, _errored = select.select([sock], [], [], 0)
            if not readable:
                return False
            return sock.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _get_stream(self) -> None:
        """Forward bus events to the browser until it goes away.

        A client that closes the tab, walks out of Wi-Fi range or is killed
        mid-stream must cost exactly one failed write and one unsubscribe: the
        server keeps serving and the bus keeps no handler pointing at a dead
        socket.  That teardown lives in the ``finally`` and runs on every exit
        path, including shutdown.
        """
        server = self.jarvis
        if self.command == "HEAD":
            # A HEAD asks what a GET would answer, not for the answer itself.
            # Streaming to one would write a body nobody may read and hold a
            # thread until the peer gave up.
            self._begin(
                HTTPStatus.OK,
                "text/event-stream; charset=utf-8",
                extra={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
            )
            return
        bus = server.bus
        events: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=_SSE_QUEUE_SIZE)
        unsubscribes: List[Callable[[], None]] = []

        def make_handler(event_name: str) -> Callable[[Any], None]:
            def handler(payload: Any = None) -> None:
                try:
                    events.put_nowait((event_name, payload))
                except queue.Full:
                    # A client too slow to keep up loses events, not the agent.
                    log.debug("SSE queue full; dropping a %s event", event_name)

            return handler

        for channel, event_name in STREAM_CHANNELS:
            unsubscribes.append(bus.subscribe(channel, make_handler(event_name)))

        try:
            self._begin(
                HTTPStatus.OK,
                "text/event-stream; charset=utf-8",
                extra={
                    "Cache-Control": "no-cache, no-store",
                    "Connection": "close",
                    # nginx and friends buffer text/* by default, which turns a
                    # live stream into one long silence.
                    "X-Accel-Buffering": "no",
                },
            )
            self.close_connection = True
            self._sse_write("retry: 2000\n\n")
            self._sse_event("ready", {"ok": True, "name": server.agent_name})

            waited = 0.0
            while not server.stopping.is_set():
                # Checked before every event as well as every idle poll: a
                # client can vanish mid-burst just as easily as mid-silence.
                if self._client_gone():
                    log.info("SSE client closed the connection")
                    break
                try:
                    name, payload = events.get(timeout=_SSE_POLL_SECONDS)
                except queue.Empty:
                    waited += _SSE_POLL_SECONDS
                    if waited >= server.sse_keepalive:
                        waited = 0.0
                        self._sse_write(": keepalive\n\n")
                    continue
                waited = 0.0
                self._sse_event(name, payload)
        except (OSError, ValueError) as exc:
            # BrokenPipe, ConnectionReset, ConnectionAborted and timeouts are
            # all OSError; on a stream every one of them means the same thing —
            # the browser is gone — so they are handled together rather than
            # enumerated. ValueError covers a write to an already-closed file.
            log.info("SSE client disconnected: %s", type(exc).__name__)
            self.close_connection = True
        finally:
            for unsubscribe in unsubscribes:
                try:
                    unsubscribe()
                except Exception:  # noqa: BLE001 - teardown is best effort
                    log.debug("SSE unsubscribe failed", exc_info=True)


# --------------------------------------------------------------------------- #
#  The server
# --------------------------------------------------------------------------- #
class _ThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded, daemonised, and never joins a handler thread on close."""

    daemon_threads = True
    # A wedged SSE connection must not make shutdown hang.
    block_on_close = False
    # SO_REUSEADDR means different things on the two platforms this has to run
    # on.  On Linux it only lets a restart reclaim a port still in TIME_WAIT,
    # which is what we want.  On Windows it lets a *second* process bind a port
    # that already has a live listener, and the kernel then splits incoming
    # connections between them — a mistyped second `jarvis serve` would silently
    # answer half the requests from a different agent.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request: Any, client_address: Any) -> None:
        # BaseServer prints a traceback to stderr; a dropped phone connection
        # is not an event worth a traceback in a voice assistant's console.
        log.debug("connection error from %s", client_address, exc_info=True)


class JarvisServer:
    """The HTTP face of a running JARVIS, for browsers on other devices.

    Construct it with the :class:`~jarvis.app.Subsystems` returned by
    :func:`jarvis.app.build`, or with any object exposing ``config``, ``bus``,
    ``orchestrator``, ``stt``, ``tts`` and ``status()``.

    ``token`` is the shared secret every non-loopback client must present.  It
    is intentionally not defaulted: :func:`serve` resolves one through
    :func:`~jarvis.server.auth.load_or_create_token`, while a bare construction
    on loopback with no token runs unauthenticated, which is only reasonable
    because loopback means "a process already running as the owner".  On any
    other address, no token is a refusal to start.
    """

    def __init__(
        self,
        subsystems: Any = None,
        *,
        config: Optional[Config] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        token: Optional[str] = None,
        bus: Optional[EventBus] = None,
        sessions: Optional[Sessions] = None,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
        max_json_bytes: int = MAX_JSON_BYTES,
        max_audio_bytes: int = MAX_AUDIO_BYTES,
        session_ttl: float = DEFAULT_SESSION_TTL,
        sse_keepalive: float = SSE_KEEPALIVE_SECONDS,
    ) -> None:
        self.subsystems = subsystems
        # ``None`` means "unspecified", so it takes the loopback default.  An
        # explicit "" is not the same thing — http.server reads it as "every
        # interface" — and is left alone so the token check below sees it for
        # what it is rather than quietly narrowing the operator's request.
        self.host = DEFAULT_HOST if host is None else str(host)
        self._requested_port = int(port)
        self.token = str(token or "")
        self.certfile = str(certfile) if certfile else ""
        self.keyfile = str(keyfile) if keyfile else ""
        self.max_json_bytes = max(1024, int(max_json_bytes))
        self.max_audio_bytes = max(1024, int(max_audio_bytes))
        self.sse_keepalive = max(1.0, float(sse_keepalive))
        self.sessions = sessions or Sessions(expire_after=session_ttl)
        self.stopping = threading.Event()

        resolved = config if config is not None else getattr(subsystems, "config", None)
        self.config: Config = resolved if resolved is not None else Config()
        resolved_bus = bus if bus is not None else getattr(subsystems, "bus", None)
        self.bus: EventBus = resolved_bus if resolved_bus is not None else get_bus()

        self._httpd: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._serving = threading.Event()
        self._lock = threading.RLock()

        self._check_bind_is_safe()

    # -- security ----------------------------------------------------------- #
    def _check_bind_is_safe(self) -> None:
        """Refuse an unauthenticated listener on anything but loopback."""
        if self.token or is_loopback(self.host):
            return
        where = "every interface" if self.host in ("", "0.0.0.0", "::") else self.host
        raise InsecureBindError(
            "Refusing to listen on %s with no access token.\n"
            "JARVIS controls this machine without restriction, so an open port "
            "hands that control to anyone who can reach it.\n"
            "Fix it in one of these ways:\n"
            "  * set a token:  export JARVIS_SERVER_TOKEN='<a long random string>'\n"
            "  * or pass one:  jarvis serve --host %s --token '<a long random string>'\n"
            "  * or let JARVIS generate and store one for you by starting it "
            "through jarvis.server.serve(), which writes the token to %s\n"
            "  * or stay private:  bind %s (the default) and reach it over an "
            "SSH tunnel:  ssh -L %d:127.0.0.1:%d <this-machine>"
            % (
                where,
                self.host or "0.0.0.0",
                token_path(self.config) or "the JARVIS data directory",
                DEFAULT_HOST,
                self._requested_port or DEFAULT_PORT,
                self._requested_port or DEFAULT_PORT,
            )
        )

    @property
    def auth_required(self) -> bool:
        """True when clients must present the token or a session cookie."""
        return bool(self.token)

    @property
    def tls(self) -> bool:
        return bool(self.certfile)

    # -- subsystem accessors ------------------------------------------------ #
    @property
    def orchestrator(self) -> Any:
        return getattr(self.subsystems, "orchestrator", None)

    @property
    def stt(self) -> Any:
        return getattr(self.subsystems, "stt", None)

    @property
    def tts(self) -> Any:
        """The synthesis engine — never the speech queue.

        The queue plays on the server's own speakers.  A remote session must be
        given bytes to play on its own device instead, so every route here uses
        the engine's ``synthesize`` and nothing else.
        """
        return getattr(self.subsystems, "tts", None)

    @property
    def agent_name(self) -> str:
        return str(getattr(getattr(self.config, "agent", None), "name", "JARVIS"))

    # -- addresses ---------------------------------------------------------- #
    @property
    def port(self) -> int:
        """The bound port — the real one, after ``port=0`` was resolved."""
        httpd = self._httpd
        if httpd is None:
            return self._requested_port
        try:
            return int(httpd.server_address[1])
        except (IndexError, TypeError, ValueError):  # pragma: no cover
            return self._requested_port

    @property
    def url(self) -> str:
        scheme = "https" if self.tls else "http"
        host = self.host
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = "[%s]" % host
        return "%s://%s:%d/" % (scheme, host, self.port)

    @property
    def running(self) -> bool:
        thread = self._thread
        return self._httpd is not None and thread is not None and thread.is_alive()

    # -- status ------------------------------------------------------------- #
    def status_payload(
        self,
        *,
        authenticated: bool = False,
        host_header: str = "",
    ) -> Dict[str, Any]:
        """What ``/api/status`` reports.  Never contains the token.

        The route is public because the client has to be able to explain the
        browser's microphone rule *before* anyone logs in.  What it does not do
        is describe the machine to a stranger: the model, the engines, the tool
        count and the data directory (which carries the owner's username) are
        added only once the caller is authenticated, or when no token is
        configured at all — which only happens on loopback, where the caller is
        already a process running as the owner.
        """
        # A browser grants the microphone only in a secure context: HTTPS, or a
        # host it already treats as local. Judge the address THIS client used,
        # not the one the server bound, because they differ over a tunnel.
        client_is_local = is_loopback(_hostname_of(host_header))
        secure_context = bool(self.tls or client_is_local)
        ffmpeg = which("ffmpeg") is not None

        payload: Dict[str, Any] = {
            "ok": True,
            "name": self.agent_name,
            "version": _version(),
            "authenticated": bool(authenticated),
            "auth_required": self.auth_required,
            "tls": self.tls,
            "audio": {
                # The single most important line in this payload.
                "capture": "browser",
                "playback": "browser",
                "server_devices_used": False,
                "secure_context": secure_context,
                "client_is_local": client_is_local,
                "microphone_note": (
                    "This page is a secure context, so the browser will offer the "
                    "microphone as soon as you ask for it."
                    if secure_context
                    else
                    "Your browser will refuse the microphone on this address. "
                    "getUserMedia is only granted in a secure context: HTTPS, or a "
                    "host the browser already treats as local (localhost, "
                    "127.0.0.1). Either serve HTTPS by starting JARVIS with "
                    "--certfile and --keyfile, or reach it through an SSH tunnel — "
                    "ssh -L %d:127.0.0.1:%d <jarvis-machine> — and open "
                    "http://127.0.0.1:%d on this device."
                    % (self.port, self.port, self.port)
                ),
                "ffmpeg": ffmpeg,
                "wav_always_supported": True,
                "transcoded_formats_supported": ffmpeg,
                "max_upload_bytes": self.max_audio_bytes,
            },
            "endpoints": sorted("%s %s" % (m, r) for (m, r) in _ROUTES),
            "session_ttl": int(self.sessions.expire_after),
        }
        if not (authenticated or not self.auth_required):
            return payload

        subsystem_status: Dict[str, Any] = {}
        reporter = getattr(self.subsystems, "status", None)
        if callable(reporter):
            try:
                subsystem_status = dict(reporter())
            except Exception:  # noqa: BLE001 - status must never 500
                log.exception("subsystem status() failed")

        stt_engine = self.stt
        payload.update(
            {
                "model": subsystem_status.get(
                    "model", getattr(getattr(self.config, "llm", None), "model", "")
                ),
                "llm": subsystem_status.get("llm", "unavailable"),
                "stt": subsystem_status.get("stt", _engine_name(stt_engine)),
                "tts": subsystem_status.get("tts", _engine_name(self.tts)),
                "memory": subsystem_status.get("memory", "unknown"),
                "tools": subsystem_status.get("tools", 0),
                "security_mode": subsystem_status.get(
                    "security_mode",
                    getattr(getattr(self.config, "security", None), "mode", "open"),
                ),
                "orchestrator": self.orchestrator is not None,
                "stt_available": stt_engine is not None
                and _engine_name(stt_engine) not in ("null", "unavailable"),
                "data_dir": subsystem_status.get("data_dir", ""),
            }
        )
        return payload

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> "JarvisServer":
        """Bind, optionally wrap in TLS, and serve on a background thread.

        Raises :class:`OSError` if the address cannot be bound — a server that
        silently failed to listen is worse than one that says so.
        """
        with self._lock:
            if self.running:
                return self
            self.stopping.clear()
            self._serving.clear()

            httpd = _ThreadingHTTPServer((self.host, self._requested_port),
                                         JarvisHTTPRequestHandler)
            httpd.jarvis = self  # type: ignore[attr-defined]

            if self.certfile:
                try:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(self.certfile, self.keyfile or None)
                    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
                except (OSError, ssl.SSLError):
                    httpd.server_close()
                    log.error("could not enable TLS with %s", self.certfile)
                    raise

            self._httpd = httpd
            thread = threading.Thread(
                target=self._serve, name="jarvis-http", daemon=True
            )
            self._thread = thread
            thread.start()

        # Wait until the accept loop is actually running: shutting a server down
        # before serve_forever() entered its loop would block forever.
        self._serving.wait(timeout=5.0)
        log.info(
            "JARVIS is listening on %s (%s, token %s)",
            self.url,
            "TLS" if self.tls else "plain HTTP",
            "required" if self.auth_required else "NOT configured - loopback only",
        )
        if not self.auth_required:
            log.warning(
                "no access token is configured: every process on this machine "
                "can drive JARVIS through %s. Set JARVIS_SERVER_TOKEN to change that.",
                self.url,
            )
        return self

    def _serve(self) -> None:
        httpd = self._httpd
        if httpd is None:  # pragma: no cover - start() sets it first
            return
        self._serving.set()
        try:
            httpd.serve_forever(poll_interval=0.2)
        except Exception:  # noqa: BLE001 - the accept loop must not die silently
            log.exception("the HTTP accept loop stopped unexpectedly")
        finally:
            self._serving.clear()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Shut down cleanly.  Idempotent, and never raises."""
        with self._lock:
            self.stopping.set()
            httpd, thread = self._httpd, self._thread
            self._httpd = None
            self._thread = None

        if httpd is not None:
            if thread is not None and thread.is_alive():
                try:
                    httpd.shutdown()
                except Exception:  # noqa: BLE001
                    log.debug("httpd.shutdown() raised", exc_info=True)
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                log.debug("httpd.server_close() raised", exc_info=True)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():  # pragma: no cover - daemon thread, so harmless
                log.warning("the HTTP thread did not stop within %.1fs", timeout)
        self.sessions.clear()
        log.info("the JARVIS HTTP server has stopped")

    # Alias: matches jarvis.app.shutdown()'s vocabulary.
    shutdown = stop

    def serve_forever(self) -> None:
        """Start and block until interrupted.  Ctrl+C leaves no zombie thread."""
        self.start()
        try:
            while not self.stopping.wait(0.5):
                pass
        except KeyboardInterrupt:
            log.info("interrupted; shutting the HTTP server down")
        finally:
            self.stop()

    def __enter__(self) -> "JarvisServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def serve(
    config: Optional[Config] = None,
    *,
    subsystems: Any = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: Optional[str] = None,
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
    bus: Optional[EventBus] = None,
    block: bool = True,
    configure_logging: bool = False,
    **kwargs: Any,
) -> JarvisServer:
    """Build (if needed) and serve JARVIS over HTTP.

    ``token=None`` resolves the shared secret the usual way — environment, then
    config, then the stored file, then a freshly generated one persisted with
    owner-only permissions.  Pass ``token=""`` to say "no authentication" out
    loud, which is accepted on loopback and refused everywhere else.

    The token is never logged.  Its storage path is, so the operator can read
    it; a caller that legitimately needs to display it (the CLI, at a terminal)
    can read :attr:`JarvisServer.token`.

    With ``block=True`` this runs until interrupted and tears down whatever it
    built.  With ``block=False`` it returns the started server for the caller
    to stop.
    """
    owns_subsystems = False
    if subsystems is None:
        from ..app import build

        subsystems = build(config, bus=bus, configure_logging=configure_logging)
        owns_subsystems = True

    effective_config = config if config is not None else getattr(subsystems, "config", None)
    if token is None:
        token = load_or_create_token(effective_config)

    server = JarvisServer(
        subsystems,
        config=effective_config,
        host=host,
        port=port,
        token=token,
        bus=bus,
        certfile=certfile,
        keyfile=keyfile,
        **kwargs,
    )

    try:
        server.start()
    except Exception:
        if owns_subsystems:
            _shutdown_subsystems(subsystems)
        raise

    if not block:
        return server

    try:
        while not server.stopping.wait(0.5):
            pass
    except KeyboardInterrupt:
        log.info("interrupted; shutting JARVIS down")
    finally:
        server.stop()
        if owns_subsystems:
            _shutdown_subsystems(subsystems)
    return server


def _shutdown_subsystems(subsystems: Any) -> None:
    try:
        from ..app import shutdown as shutdown_subsystems

        shutdown_subsystems(subsystems)
    except Exception:  # noqa: BLE001 - teardown is best effort
        log.exception("subsystem shutdown failed")


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_AUDIO_BYTES",
    "MAX_JSON_BYTES",
    "PUBLIC_ROUTES",
    "SESSION_COOKIE",
    "STREAM_CHANNELS",
    "InsecureBindError",
    "JarvisHTTPRequestHandler",
    "JarvisServer",
    "client_page",
    "serve",
    "transcode_to_wav",
]
