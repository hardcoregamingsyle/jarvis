"""Generic OpenAI-compatible chat client, stdlib only.

One class covers every server that speaks the OpenAI ``/v1/chat/completions``
shape: vLLM, llama.cpp's ``llama-server``, LM Studio, text-generation-inference,
Ollama's ``/v1`` compatibility shim, and hosted endpoints.  Nothing here imports
a third-party package -- ``urllib.request`` plus ``json`` is the whole transport.

Why this module exists
----------------------
JARVIS runs a *tree* of agents: a main agent spawns subagents which spawn
sub-subagents, and all of them issue LLM calls at the same time.  A backend that
serialises requests makes that tree no faster than a single agent.  The servers
behind this client do continuous batching, so N concurrent requests share one
resident copy of the weights and cost far less than N sequential requests.

Everything below is therefore written for concurrent callers:

* :meth:`generate` and :meth:`stream` keep **no mutable per-request state on
  ``self``** -- each call builds its own :class:`urllib.request.Request` and
  reads its own response.  A single backend instance is meant to be shared by
  the whole agent tree.
* ``max_concurrent_requests`` (read from the config, ``0``/``None`` = unlimited)
  bounds how many requests are in flight at once.  This is resource management,
  not permission: it stops a runaway tree from opening a thousand sockets and
  exhausting file descriptors.  It is enforced with a semaphore held for the
  full lifetime of a request, including the whole of a streamed response.
* Transient failures are retried with capped exponential backoff and jitter.
  vLLM answers ``503`` while it is still loading weights -- which is exactly
  when a fleet of agents starts up and hammers it.
* :meth:`is_available` caches its verdict for a few seconds, because otherwise
  every agent in the tree probes ``/models`` before every call.

Configuration
-------------
Every field is read from :class:`~jarvis.core.config.LLMConfig` through
``getattr`` with a default, so this module keeps working whether or not a given
field exists on the dataclass:

===========================  ===========================================
``vllm_host``                base URL, e.g. ``http://127.0.0.1:8000/v1``
``model``                    served model id
``api_key``                  bearer token (or env ``JARVIS_LLM_API_KEY``)
``max_concurrent_requests``  in-flight cap, ``0`` = unlimited
``request_timeout``          per-request socket timeout
===========================  ===========================================

``openai_base_url`` / ``openai_model`` / ``openai_api_key`` are consulted first
when present, which lets a second, differently-configured endpoint coexist with
the vLLM one; otherwise the shared fields above are used.  Anything not in the
config can be passed to the constructor instead.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import socket
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib import error as _urlerror
from urllib import parse as _urlparse
from urllib import request as _urlrequest

from ..core.config import LLMConfig
from ..core.contracts import GenerationConfig, LLMResult, Message
from .base import BaseLLM, apply_stop_strings, strip_thinking


logger = logging.getLogger(__name__)


#: Where a local OpenAI-compatible server most often lives (vLLM's default).
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

#: Availability probes must never stall an agent: short timeout, short cache.
PROBE_TIMEOUT = 1.5
PROBE_CACHE_SECONDS = 5.0

#: ``/models`` during load() is a startup call, not a generation call.
MODELS_TIMEOUT = 15.0

#: Environment fallback for the bearer token while LLMConfig lacks a field.
API_KEY_ENV = "JARVIS_LLM_API_KEY"

_SNIPPET_CHARS = 200


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def config_value(cfg: Any, *names: str) -> str:
    """First non-empty attribute among ``names``, calling it if it is callable.

    ``Config.models_dir`` is a method while ``LLMConfig.model`` is a field, and
    callers pass either object; tolerating both keeps the call sites simple.
    """
    for name in names:
        raw = getattr(cfg, name, None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:  # pragma: no cover - defensive
                raw = None
        if raw:
            return str(raw).strip()
    return ""


def normalise_base_url(raw: str) -> str:
    """Return ``raw`` as a usable API base, appending ``/v1`` when it has no path.

    ``http://box:8000`` -> ``http://box:8000/v1`` (what people actually type),
    while ``https://api.example.com/openai/v1`` is left alone.
    """
    text = (raw or "").strip().rstrip("/")
    if not text:
        text = DEFAULT_BASE_URL
    if "://" not in text:
        text = "http://" + text
    parts = _urlparse.urlsplit(text)
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    return _urlparse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _snippet(body: str, limit: int = _SNIPPET_CHARS) -> str:
    """Collapse a response body to one short line for an error message."""
    return " ".join((body or "").split())[:limit]


def _response_status(resp: Any) -> int:
    for attr in ("status", "code"):
        value = getattr(resp, attr, None)
        if isinstance(value, int):
            return value
    return 200


def _error_body(exc: Any) -> str:
    """Read the body of an :class:`urllib.error.HTTPError` without exploding."""
    try:
        raw = exc.read()
    except Exception:  # pragma: no cover - body already consumed
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw or "")


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.timeout, TimeoutError))


def _retryable_status(status: int) -> bool:
    """429 and 5xx are worth another go; every other 4xx is our own fault."""
    return status == 429 or status >= 500


def _server_error_detail(data: Any) -> str:
    """Pull a human message out of an OpenAI-style ``{"error": ...}`` body."""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or "").strip()
    if isinstance(err, str):
        return err.strip()
    detail = data.get("detail") or data.get("message")
    return str(detail).strip() if detail else ""


def iter_sse_payloads(chunks: Any) -> Iterator[str]:
    """Yield the ``data:`` payload of each Server-Sent Event in ``chunks``.

    ``chunks`` is any iterable of ``bytes`` -- whole lines from a file-like HTTP
    response, or arbitrary network-sized fragments that split a line down the
    middle.  Both are handled: bytes are buffered and only complete lines are
    interpreted.

    Per the SSE spec an event may carry several ``data:`` lines, which are
    joined with newlines and dispatched on the following blank line; ``:``
    comment lines are keep-alives and ignored.  Iteration stops at the
    ``[DONE]`` sentinel that OpenAI-compatible servers send last.
    """
    buffer = ""
    pending: List[str] = []
    finished = False

    def dispatch() -> Optional[str]:
        nonlocal pending, finished
        if not pending:
            return None
        payload = "\n".join(pending)
        pending = []
        if payload.strip() == "[DONE]":
            finished = True
            return None
        return payload

    for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", "replace")
        else:
            buffer += str(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                payload = dispatch()
                if finished:
                    return
                if payload is not None:
                    yield payload
                continue
            if line.startswith(":"):
                continue  # keep-alive comment
            if line.startswith("data:"):
                pending.append(line[len("data:"):].lstrip(" "))

    tail = buffer.rstrip("\r")
    if tail.startswith("data:"):
        pending.append(tail[len("data:"):].lstrip(" "))
    payload = dispatch()
    if not finished and payload is not None:
        yield payload


# --------------------------------------------------------------------------- #
#  Backend
# --------------------------------------------------------------------------- #
class OpenAICompatBackend(BaseLLM):
    """Chat client for any server exposing the OpenAI ``/v1`` API.

    Safe to share between threads: a single instance is intended to serve an
    entire tree of concurrent agents.
    """

    name = "openai-compat"

    #: Total attempts per request (1 disables retrying).
    max_attempts: int = 4
    #: Backoff is ``backoff_base * 2**(attempt-1)``, clamped to ``backoff_cap``.
    backoff_base: float = 0.5
    backoff_cap: float = 8.0

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(cfg)
        if name:
            self.name = name
        self._base_url = normalise_base_url(
            base_url or config_value(cfg, "openai_base_url", "openai_api_base", "vllm_host")
        )
        self._model = (model or config_value(cfg, "openai_model", "model")).strip()
        resolved_key = api_key if api_key is not None else config_value(
            cfg, "openai_api_key", "vllm_api_key", "api_key"
        )
        self._api_key = (resolved_key or os.environ.get(API_KEY_ENV, "")).strip()
        self._timeout = float(getattr(cfg, "request_timeout", 600.0) or 600.0)

        limit = int(getattr(cfg, "max_concurrent_requests", 0) or 0)
        self._max_concurrent = max(0, limit)
        self._semaphore: Optional[threading.Semaphore] = (
            threading.Semaphore(self._max_concurrent) if self._max_concurrent > 0 else None
        )

        self._probe_lock = threading.Lock()
        self._probe_cache: Optional[tuple] = None  # (monotonic_ts, bool)
        self._load_lock = threading.Lock()
        self._served_model: Optional[str] = None

    # -- introspection ------------------------------------------------------ #
    @property
    def base_url(self) -> str:
        """The normalised API base, e.g. ``http://127.0.0.1:8000/v1``."""
        return self._base_url

    @property
    def api_key(self) -> str:
        """The bearer token in use, or ``""`` when the endpoint is open."""
        return self._api_key

    @property
    def model(self) -> str:
        """The model id sent with requests (resolved by :meth:`load`)."""
        return self._served_model or self._model

    @property
    def max_concurrent_requests(self) -> int:
        """In-flight request cap; ``0`` means unlimited."""
        return self._max_concurrent

    # -- HTTP plumbing ------------------------------------------------------ #
    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{path}"

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        stream: bool = False,
    ) -> Any:
        """Build a fresh Request. Never cached -- that is what makes us reentrant."""
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        return _urlrequest.Request(
            self._url(path),
            data=data,
            headers=self._headers(stream=stream),
            method=method,
        )

    def _open_once(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        req = self._build_request(method, path, payload, stream=stream)
        return _urlrequest.urlopen(req, timeout=float(timeout or self._timeout))

    def _backoff_delay(self, attempt: int) -> float:
        """Capped exponential backoff with jitter, in seconds.

        Jitter matters with an agent tree: without it every subagent that hit
        the same ``503`` retries in lockstep and re-creates the thundering herd.
        """
        base = min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempt - 1)))
        return base + random.uniform(0.0, base * 0.25)

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self._backoff_delay(attempt))

    def _open_with_retries(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
        attempts: Optional[int] = None,
    ) -> Any:
        """Open a response, retrying connection errors / 429 / 5xx.

        The caller owns the returned response object and must close it; that is
        what lets :meth:`stream` consume the body incrementally.  ``attempts``
        overrides :attr:`max_attempts` for calls where a fast answer beats a
        patient one -- a health check, say.
        """
        url = self._url(path)
        effective = float(timeout or self._timeout)
        attempts = max(1, int(self.max_attempts if attempts is None else attempts))
        last_detail = ""

        for attempt in range(1, attempts + 1):
            try:
                return self._open_once(
                    method, path, payload, stream=stream, timeout=effective
                )
            except _urlerror.HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                body = _error_body(exc)
                if not _retryable_status(status) or attempt >= attempts:
                    raise self._http_error(url, status, body) from exc
                last_detail = f"HTTP {status}"
                logger.debug(
                    "%s: %s from %s, retry %d/%d", self.name, last_detail, url,
                    attempt, attempts,
                )
            except (_urlerror.URLError, OSError) as exc:
                if _is_timeout(exc):
                    raise RuntimeError(
                        f"{self.name}: request to {url} timed out after "
                        f"{effective:g}s. Raise llm.request_timeout, or check "
                        f"that the server is not swapping."
                    ) from exc
                reason = getattr(exc, "reason", exc)
                if attempt >= attempts:
                    raise RuntimeError(
                        f"{self.name}: could not reach {url}: {reason}. "
                        f"Is the server running and is base_url correct?"
                    ) from exc
                last_detail = str(reason)
                logger.debug(
                    "%s: %s reaching %s, retry %d/%d", self.name, last_detail, url,
                    attempt, attempts,
                )
            self._sleep_backoff(attempt)

        # Unreachable: the loop either returns or raises on its last attempt.
        raise RuntimeError(f"{self.name}: request to {url} failed: {last_detail}")

    def _http_error(self, url: str, status: int, body: str) -> RuntimeError:
        """A message that names the URL, the status, and what the server said."""
        detail = ""
        try:
            detail = _server_error_detail(json.loads(body))
        except Exception:
            detail = ""
        suffix = f" -- {detail}" if detail else ""
        return RuntimeError(
            f"{self.name}: {url} returned HTTP {status}{suffix}; "
            f"body: {_snippet(body)}"
        )

    def _body_error(self, url: str, status: int, body: str, why: str) -> RuntimeError:
        return RuntimeError(
            f"{self.name}: {why} from {url} (HTTP {status}); body: {_snippet(body)}"
        )

    @contextlib.contextmanager
    def _slot(self) -> Iterator[None]:
        """Hold one of the ``max_concurrent_requests`` in-flight slots."""
        if self._semaphore is None:
            yield
            return
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    # -- availability ------------------------------------------------------- #
    def is_available(self) -> bool:
        """True when ``GET {base}/models`` answers. Never raises.

        The verdict is cached for :data:`PROBE_CACHE_SECONDS`; a tree of agents
        would otherwise probe once per generation.
        """
        cached = self._probe_cache
        if cached is not None and (time.monotonic() - cached[0]) < PROBE_CACHE_SECONDS:
            return bool(cached[1])
        with self._probe_lock:
            cached = self._probe_cache
            if cached is not None and (time.monotonic() - cached[0]) < PROBE_CACHE_SECONDS:
                return bool(cached[1])
            ok = False
            try:
                resp = self._open_once("GET", "/models", timeout=PROBE_TIMEOUT)
                try:
                    resp.read(1)
                finally:
                    with contextlib.suppress(Exception):
                        resp.close()
                ok = True
            except Exception:
                ok = False
            self._probe_cache = (time.monotonic(), ok)
            return ok

    def invalidate_probe(self) -> None:
        """Forget the cached availability verdict (e.g. after starting a server)."""
        self._probe_cache = None

    # -- load --------------------------------------------------------------- #
    def load(self) -> None:
        """Idempotent and thread-safe: 24 agents starting at once resolve once."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._do_load()
            self._loaded = True

    def _do_load(self) -> None:
        # Only one quick retry here: the generation call that follows does the
        # patient backing-off, and duplicating it would double every startup.
        try:
            served = self.list_models(attempts=2)
        except Exception as exc:
            if not self._model:
                raise  # No configured id to fall back on -- surface the reason.
            logger.debug("%s: could not list models: %s", self.name, exc)
            self._served_model = self._model
            return

        if not served:
            self._served_model = self._model
            return

        chosen = self._match_model(self._model, served)
        if chosen != self._model:
            logger.info(
                "%s: configured model %r is not served by %s; using %r (served: %s)",
                self.name, self._model or "<unset>", self._base_url, chosen,
                ", ".join(served[:8]),
            )
        else:
            logger.info("%s: using served model %r at %s", self.name, chosen, self._base_url)
        self._served_model = chosen

    @staticmethod
    def _match_model(wanted: str, served: Sequence[str]) -> str:
        """Resolve ``wanted`` against the ids the server actually serves.

        Servers frequently rename what you asked for: a local path becomes its
        basename, or an org prefix is dropped.  Exact match wins, then a
        case-insensitive basename match, then simply the first served id.
        """
        if not served:
            return wanted
        if wanted and wanted in served:
            return wanted
        if wanted:
            tail = wanted.replace("\\", "/").rsplit("/", 1)[-1].lower()
            for candidate in served:
                if candidate.replace("\\", "/").rsplit("/", 1)[-1].lower() == tail:
                    return candidate
        return served[0]

    def unload(self) -> None:
        self._served_model = None
        super().unload()

    # -- payload ------------------------------------------------------------ #
    def _request_model(self) -> str:
        model = self._served_model or self._model
        if not model:
            raise RuntimeError(
                f"{self.name}: no model id configured and {self._base_url}/models "
                f"listed none. Set llm.model to a served model id."
            )
        return model

    def _payload(
        self,
        messages: Sequence[Message],
        gen: GenerationConfig,
        *,
        stream: bool,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": self._request_model(),
            "messages": [m.to_dict() for m in messages],
            "stream": bool(stream),
            "max_tokens": int(gen.max_new_tokens),
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
        }
        if gen.stop:
            payload["stop"] = list(gen.stop)
        if gen.seed is not None:
            payload["seed"] = int(gen.seed)
        # top_k is a vLLM / llama.cpp extension, not part of the OpenAI schema.
        # Strict hosted endpoints reject unknown fields -- set llm.top_k = 0 to
        # omit it when pointing this client at one of those.
        if gen.top_k and int(gen.top_k) > 0:
            payload["top_k"] = int(gen.top_k)
        return payload

    # -- generate ----------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResult:
        """One non-streaming chat completion. Safe to call from many threads."""
        self.load()
        gen = self._gen_config(config)
        payload = self._payload(messages, gen, stream=False)
        url = self._url("/chat/completions")

        with self._slot():
            resp = self._open_with_retries("POST", "/chat/completions", payload)
            try:
                status = _response_status(resp)
                body = resp.read().decode("utf-8", "replace")
            finally:
                with contextlib.suppress(Exception):
                    resp.close()

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise self._body_error(url, status, body, "invalid JSON") from exc

        detail = _server_error_detail(data)
        if detail and not (isinstance(data, dict) and data.get("choices")):
            raise self._body_error(url, status, body, f"server error: {detail}")

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise self._body_error(url, status, body, "response had no choices")

        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        else:
            text = str(choice.get("text") or "")

        text = strip_thinking(text)
        text = apply_stop_strings(text, gen.stop)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return LLMResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            raw=data,
        )

    # -- stream ------------------------------------------------------------- #
    def stream(
        self,
        messages: Sequence[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Yield content deltas from a streamed chat completion.

        Deltas are yielded raw (thinking blocks included) so a caller that wants
        them can see them; :meth:`generate` strips them.  Stop strings are still
        honoured -- the partial delta before the stop is emitted, then the
        iterator ends and the connection is closed.
        """
        self.load()
        gen = self._gen_config(config)
        payload = self._payload(messages, gen, stream=True)
        stops = tuple(gen.stop or ())

        with self._slot():
            resp = self._open_with_retries("POST", "/chat/completions", payload, stream=True)
            try:
                buffer = ""
                for raw in iter_sse_payloads(resp):
                    try:
                        obj = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    delta = _stream_delta(obj)
                    if not delta:
                        continue
                    previous = len(buffer)
                    buffer += delta
                    if stops:
                        clipped = apply_stop_strings(buffer, stops)
                        if len(clipped) < len(buffer):
                            remainder = clipped[previous:]
                            if remainder:
                                yield remainder
                            return
                    yield delta
            finally:
                with contextlib.suppress(Exception):
                    resp.close()

    # -- models ------------------------------------------------------------- #
    def list_models(self, *, attempts: Optional[int] = None) -> List[str]:
        """Return the model ids the server reports at ``GET {base}/models``.

        Raises :class:`RuntimeError` naming the URL when the server cannot be
        reached or answers with something that is not a model list.
        """
        url = self._url("/models")
        resp = self._open_with_retries(
            "GET",
            "/models",
            timeout=min(self._timeout, MODELS_TIMEOUT),
            attempts=attempts,
        )
        try:
            status = _response_status(resp)
            body = resp.read().decode("utf-8", "replace")
        finally:
            with contextlib.suppress(Exception):
                resp.close()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise self._body_error(url, status, body, "invalid JSON") from exc
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        return [
            str(item["id"])
            for item in entries
            if isinstance(item, dict) and item.get("id")
        ]


def _stream_delta(obj: Any) -> str:
    """Extract ``choices[0].delta.content`` from one streamed SSE frame."""
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        return str(delta.get("content") or "")
    # Some servers echo the non-streaming shape on the final frame.
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


__all__ = [
    "OpenAICompatBackend",
    "config_value",
    "iter_sse_payloads",
    "normalise_base_url",
    "DEFAULT_BASE_URL",
    "API_KEY_ENV",
]
