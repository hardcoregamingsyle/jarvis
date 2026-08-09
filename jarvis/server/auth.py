"""Who may CONNECT: shared tokens, browser sessions, constant-time comparison.

This module places no limit whatsoever on what JARVIS may do to the machine it
runs on.  The owner deliberately removed those limits and this package respects
that everywhere.  What it does do is keep *other people* out of the port.

The distinction matters.  Locally, JARVIS is an unrestricted shell, filesystem
and process agent with the owner's privileges.  Publishing that on a network
socket without a secret means anyone who can reach the socket owns the machine —
including whatever else is on the coffee-shop Wi-Fi.  So a shared token is
mandatory the moment the listening address stops being loopback, and every
comparison against it runs in constant time so the token cannot be recovered a
byte at a time by timing the failures.

Nothing here ever logs, returns or interpolates the token into a message.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


#: Environment variable an operator sets to pin the shared token.
TOKEN_ENV_VAR = "JARVIS_SERVER_TOKEN"

#: File under the JARVIS data directory where a generated token is persisted.
TOKEN_FILE_NAME = "server_token"

#: How long a browser session stays valid without being used.
DEFAULT_SESSION_TTL = 12 * 60 * 60.0

#: Sessions kept at once.  A cap keeps a login-spamming client from growing the
#: table without bound; the oldest idle session is evicted first.
DEFAULT_MAX_SESSIONS = 64

#: Host names the operating system resolves to the loopback interface.  Checked
#: before :mod:`ipaddress` because they are names, not addresses.
LOOPBACK_HOST_NAMES = frozenset(
    {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
)


# --------------------------------------------------------------------------- #
#  Tokens
# --------------------------------------------------------------------------- #
def generate_token() -> str:
    """Return a fresh URL-safe shared token with 256 bits of entropy."""
    return secrets.token_urlsafe(32)


def token_path(config: Any) -> Optional[Path]:
    """Where a generated token is persisted for ``config``.

    Returns ``None`` when the config object cannot tell us about a data
    directory, which is the failure type callers expect rather than an
    exception from deep inside path handling.
    """
    getter = getattr(config, "path", None)
    if callable(getter):
        try:
            return Path(getter(TOKEN_FILE_NAME))
        except Exception:  # noqa: BLE001 - a broken config must not stop the server
            log.debug("config.path(%r) failed", TOKEN_FILE_NAME, exc_info=True)
    home = getattr(config, "home", None)
    if callable(home):
        try:
            return Path(home()) / TOKEN_FILE_NAME
        except Exception:  # noqa: BLE001
            log.debug("config.home() failed", exc_info=True)
    return None


def _configured_token(config: Any) -> str:
    """Read ``config.server.token`` if such a section exists.

    Read defensively: the shipped :class:`~jarvis.core.config.Config` has no
    ``server`` section today, and this must keep working if one is added later.
    """
    if config is None:
        return ""
    section: Any = None
    if isinstance(config, dict):
        section = config.get("server")
    else:
        section = getattr(config, "server", None)
    if section is None:
        return ""
    value = section.get("token") if isinstance(section, dict) else getattr(section, "token", "")
    return str(value or "").strip()


def _read_token_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        log.debug("could not read the token file at %s", path, exc_info=True)
        return ""


def _write_token_file(path: Path, token: str) -> bool:
    """Persist ``token`` with owner-only permissions where the platform has them.

    Windows has no POSIX mode bits, so ``0o600`` is advisory there; the file
    still lands in the user's own profile directory.  Returns success.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.debug("could not create %s", path.parent, exc_info=True)
        return False
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    try:
        handle = os.open(str(path), flags, 0o600)
    except OSError:
        log.debug("could not create the token file at %s", path, exc_info=True)
        return False
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError:
        log.debug("could not write the token file at %s", path, exc_info=True)
        return False
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        # Best effort: some filesystems (and Windows) will not take the mode.
        log.debug("could not chmod 0600 %s", path, exc_info=True)
    return True


def load_or_create_token(
    config: Any = None,
    *,
    environ: Optional[Dict[str, str]] = None,
    create: bool = True,
) -> str:
    """Resolve the shared token, generating and persisting one when absent.

    Order of precedence, highest first:

    1. ``JARVIS_SERVER_TOKEN`` in the environment,
    2. a ``server.token`` entry in the configuration, if that section exists,
    3. the file at ``config.path("server_token")``,
    4. a freshly generated token, written to that file (unless ``create`` is
       false, in which case the empty string comes back).

    The token itself is never logged.  Only its storage path is, so an operator
    can read it with ``cat`` — that is why the caller, not this module, decides
    whether to display it.
    """
    env = environ if environ is not None else os.environ
    try:
        from_env = str(env.get(TOKEN_ENV_VAR) or "").strip()
    except Exception:  # noqa: BLE001 - a hostile mapping is not worth a crash
        from_env = ""
    if from_env:
        log.info("using the shared token from %s", TOKEN_ENV_VAR)
        return from_env

    from_config = _configured_token(config)
    if from_config:
        log.info("using the shared token from the configuration file")
        return from_config

    path = token_path(config)
    if path is None:
        if not create:
            return ""
        log.warning(
            "no data directory is available, so the generated access token "
            "cannot be persisted; it will change on every restart"
        )
        return generate_token()

    existing = _read_token_file(path)
    if existing:
        return existing
    if not create:
        return ""

    token = generate_token()
    if _write_token_file(path, token):
        log.warning(
            "generated a new remote-access token and stored it at %s "
            "(read it with: cat %s) — it is not written to the log",
            path,
            path,
        )
    else:
        log.warning(
            "generated a remote-access token but could not store it at %s; "
            "it will change on every restart",
            path,
        )
    return token


def check_token(supplied: Any, expected: Any) -> bool:
    """Constant-time comparison of a supplied token against the expected one.

    Tolerant of ``None`` and empty values on either side: both are refusals, so
    an unconfigured server never accidentally accepts an empty token.  The
    comparison always goes through :func:`hmac.compare_digest` — ``==`` on
    strings short-circuits on the first differing byte and leaks the token one
    character at a time to anyone who can time the responses.
    """
    if expected is None or supplied is None:
        return False
    try:
        expected_bytes = str(expected).encode("utf-8")
        supplied_bytes = str(supplied).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return False
    if not expected_bytes or not supplied_bytes:
        return False
    try:
        return bool(hmac.compare_digest(supplied_bytes, expected_bytes))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
#  Addresses
# --------------------------------------------------------------------------- #
def is_loopback(host: Any) -> bool:
    """True when ``host`` names only this machine's loopback interface.

    The empty string and ``0.0.0.0`` / ``::`` are deliberately **not** loopback:
    :mod:`http.server` treats them as "every interface", which is precisely the
    case a shared token has to be mandatory for.
    """
    if host is None:
        return False
    text = str(host).strip()
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text.lower() in LOOPBACK_HOST_NAMES:
        return True
    # An IPv6 literal may carry a zone id, e.g. "fe80::1%eth0".
    candidate = text.split("%", 1)[0]
    try:
        return bool(ipaddress.ip_address(candidate).is_loopback)
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
#  Sessions
# --------------------------------------------------------------------------- #
class Sessions:
    """Short-lived session ids handed out after a successful token login.

    Thread-safe, because the HTTP server is threaded and two requests from the
    same phone routinely land on different threads.  Expiry is sliding: using a
    session renews it, so an active conversation is never logged out mid-turn,
    while an abandoned tab stops working after ``expire_after`` seconds.
    """

    def __init__(
        self,
        *,
        expire_after: float = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self.expire_after = max(1.0, float(expire_after))
        self.max_sessions = max(1, int(max_sessions))
        self._lock = threading.RLock()
        self._sessions: Dict[str, float] = {}

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _now() -> float:
        # Monotonic: a clock adjustment must not resurrect an expired session.
        return time.monotonic()

    def _prune(self, now: float) -> None:
        cutoff = now - self.expire_after
        for sid in [s for s, seen in self._sessions.items() if seen <= cutoff]:
            self._sessions.pop(sid, None)

    # -- public API --------------------------------------------------------- #
    def create(self) -> str:
        """Mint a new session id.  Never reuses one."""
        now = self._now()
        sid = secrets.token_urlsafe(24)
        with self._lock:
            self._prune(now)
            while len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key])
                self._sessions.pop(oldest, None)
            self._sessions[sid] = now
        return sid

    def validate(self, session_id: Any, *, renew: bool = True) -> bool:
        """True when ``session_id`` is a live session; renews it by default."""
        if not session_id or not isinstance(session_id, str):
            return False
        now = self._now()
        with self._lock:
            self._prune(now)
            seen = self._sessions.get(session_id)
            if seen is None:
                return False
            if renew:
                self._sessions[session_id] = now
            return True

    def touch(self, session_id: Any) -> bool:
        """Renew a session without asserting anything else about it."""
        return self.validate(session_id, renew=True)

    def drop(self, session_id: Any) -> bool:
        """Invalidate one session.  True when it existed."""
        if not session_id or not isinstance(session_id, str):
            return False
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        """Invalidate every session — used on shutdown."""
        with self._lock:
            self._sessions.clear()

    def active(self) -> List[str]:
        """Ids of the sessions that have not expired."""
        now = self._now()
        with self._lock:
            self._prune(now)
            return list(self._sessions)

    def __len__(self) -> int:
        return len(self.active())


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_SESSION_TTL",
    "LOOPBACK_HOST_NAMES",
    "TOKEN_ENV_VAR",
    "TOKEN_FILE_NAME",
    "Sessions",
    "check_token",
    "generate_token",
    "is_loopback",
    "load_or_create_token",
    "token_path",
]
