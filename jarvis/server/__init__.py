"""Remote access: JARVIS in a browser on any device, with the brain left here.

The assistant does not move.  The model, the tools, the memory and the machine
control all stay on the box this package runs on; a phone, a laptop or a tablet
becomes a thin client with nothing installed on it but a browser.

Two properties make that useful rather than merely possible:

* **Audio belongs to the client.**  The microphone and speakers wired to the
  JARVIS machine are in whichever room that machine is in, so a remote session
  records with ``getUserMedia`` in the browser, uploads the clip to ``/api/stt``
  and plays the reply that ``/api/tts`` returns.  No route opens a server-side
  audio device.
* **Only the owner gets in.**  Nothing here constrains what JARVIS may do to its
  own machine — that stays wide open, as intended.  It constrains who may reach
  the port, because the port fronts an unrestricted agent: loopback by default,
  a mandatory shared token on any other address, constant-time comparison, no
  CORS, and a refusal to start on a public interface with no secret.

Built entirely on the standard library, so remote access costs no new
dependency on either end.

    from jarvis.server import serve
    serve(config)                       # http://127.0.0.1:8765
    serve(config, host="0.0.0.0")       # LAN, token required
"""

from __future__ import annotations

from .app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_AUDIO_BYTES,
    MAX_JSON_BYTES,
    PUBLIC_ROUTES,
    SESSION_COOKIE,
    InsecureBindError,
    JarvisHTTPRequestHandler,
    JarvisServer,
    client_page,
    serve,
    transcode_to_wav,
)
from .auth import (
    TOKEN_ENV_VAR,
    Sessions,
    check_token,
    generate_token,
    is_loopback,
    load_or_create_token,
    token_path,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_AUDIO_BYTES",
    "MAX_JSON_BYTES",
    "PUBLIC_ROUTES",
    "SESSION_COOKIE",
    "TOKEN_ENV_VAR",
    "InsecureBindError",
    "JarvisHTTPRequestHandler",
    "JarvisServer",
    "Sessions",
    "check_token",
    "client_page",
    "generate_token",
    "is_loopback",
    "load_or_create_token",
    "serve",
    "token_path",
    "transcode_to_wav",
]
