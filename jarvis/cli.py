"""Command-line entry point.

    jarvis                 # voice mode if a mic is available, else text chat
    jarvis chat            # text conversation
    jarvis voice           # hands-free
    jarvis say "..."       # one-shot: speak a line (voice check)
    jarvis ask "..."       # one-shot: answer and exit
    jarvis doctor          # what is installed, what is missing, how to fix it
    jarvis config          # show or write the effective configuration
    jarvis tools           # list registered tools
    jarvis memory          # inspect / search long-term memory
    jarvis setup           # download the voice model and check dependencies
    jarvis serve           # browser client, so another device can talk to this JARVIS
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import secrets
import socket
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .core.config import Config, load_config
from .core.logging_setup import setup_logging
from .core.platform_utils import system_summary

log = logging.getLogger(__name__)

BANNER = r"""
     ____             _         _  _____
    |  _ \           | |       | |/ ____|
    | |_) |          | |       | | (___
    |  _ <           | |       | |\___ \
  J | |_) | A  R  V  | | S     | |____) |
    |____/           |_|        \_____/
"""


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _out(text: str = "") -> None:
    """Write to stdout, tolerating consoles that cannot encode the character set."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"))


def _confirm(prompt: str) -> bool:
    """Interactive yes/no used as the security gate's confirmation callback."""
    try:
        answer = input(f"\n  [confirm] {prompt}\n  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _out()
        return False
    return answer in ("y", "yes")


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "verbose", False):
        cfg.log_level = "DEBUG"
    if getattr(args, "no_speech", False):
        cfg.tts.enabled = False
    if getattr(args, "model", None):
        cfg.llm.model = args.model
    if getattr(args, "backend", None):
        cfg.llm.backend = args.backend
    setup_logging(cfg.log_level, log_file=cfg.logs_dir() / "jarvis.log")
    return cfg


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    """Report on every optional dependency and what it unlocks."""
    cfg = _load(args)
    _out(BANNER)
    _out("Environment")
    for key, value in system_summary().items():
        _out(f"  {key:<10} {value}")
    _out(f"  {'data dir':<10} {cfg.home()}")

    groups = [
        ("Language model", [
            ("airllm", "AirLLM — run a 30B model without loading it all", "pip install airllm"),
            ("torch", "tensor backend (required by airllm/transformers)", "pip install torch"),
            ("transformers", "tokenisers and the HF backend", "pip install transformers"),
        ]),
        ("Speech to text", [
            ("faster_whisper", "fast offline transcription (recommended)", "pip install faster-whisper"),
            ("whisper", "reference Whisper implementation", "pip install openai-whisper"),
            ("vosk", "very light offline STT", "pip install vosk"),
        ]),
        ("Text to speech", [
            ("piper", "offline neural British voice (recommended)", "pip install piper-tts"),
            ("edge_tts", "online neural British voice, highest quality", "pip install edge-tts"),
            ("pyttsx3", "operating-system voices", "pip install pyttsx3"),
        ]),
        ("Audio devices", [
            ("sounddevice", "microphone capture and playback", "pip install sounddevice"),
            ("numpy", "audio buffers (strongly recommended)", "pip install numpy"),
        ]),
        ("Memory", [
            ("sentence_transformers", "semantic recall embeddings", "pip install sentence-transformers"),
        ]),
        ("System control", [
            ("psutil", "processes, CPU, memory, battery", "pip install psutil"),
            ("mss", "screenshots", "pip install mss"),
        ]),
    ]

    import importlib.util

    missing = []
    for title, entries in groups:
        _out(f"\n{title}")
        for module, purpose, install in entries:
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            mark = "OK  " if found else "--  "
            _out(f"  {mark}{module:<22} {purpose}")
            if not found:
                missing.append(install)

    _out("\nSubsystems")
    try:
        from . import app as app_module

        subsystems = app_module.build(cfg, configure_logging=False)
        for key, value in subsystems.status().items():
            _out(f"  {key:<15} {value}")
        app_module.shutdown(subsystems)
    except Exception as exc:  # noqa: BLE001
        _out(f"  build failed: {exc}")
        log.exception("doctor: build failed")

    if missing:
        _out("\nTo install what is missing:")
        for line in dict.fromkeys(missing):
            _out(f"  {line}")
        _out("\nOr install a complete profile:")
        _out("  pip install -r requirements-full.txt")
    else:
        _out("\nEverything is installed.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.write:
        path = cfg.save(None if args.write is True else Path(args.write))
        _out(f"Wrote configuration to {path}")
        return 0
    _out(json.dumps(cfg.to_dict(), indent=2))
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from .core.security import SecurityGate
    from .tools import create_registry

    registry = create_registry(cfg, SecurityGate(cfg.security))
    registry.load_generated()
    specs = sorted(registry.list(), key=lambda s: s.name)
    _out(f"{len(specs)} tools registered\n")
    for spec in specs:
        flag = " [dangerous]" if spec.dangerous else ""
        params = ", ".join(
            f"{p.name}" + ("" if p.required else "?") for p in spec.params
        )
        _out(f"  {spec.name}({params}){flag}")
        _out(textwrap.fill(spec.description.strip().split("\n")[0],
                           width=88, initial_indent="      ", subsequent_indent="      "))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from .memory import create_memory

    cfg.memory.db_path = str(cfg.db_file())
    store = create_memory(cfg.memory)
    try:
        if args.query:
            hits = store.search(args.query, k=args.limit)
            if not hits:
                _out("Nothing found.")
            for hit in hits:
                _out(f"[{hit.score:.3f}] ({hit.kind}) {hit.text[:300]}")
        elif args.stats:
            _out(json.dumps(store.stats(), indent=2, default=str))
        elif args.export:
            count = store.export_jsonl(args.export)
            _out(f"Exported {count} records to {args.export}")
        else:
            for record in store.recent(k=args.limit):
                _out(f"({record.kind}) {record.text[:200]}")
    finally:
        store.close()
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    """Speak a line — the quickest way to audition the British voice."""
    cfg = _load(args)
    from .speech.tts import create_tts

    engine = create_tts(cfg.tts, voices_dir=cfg.voices_dir())
    _out(f"Voice engine: {getattr(engine, 'name', '?')}")
    text = args.text or "Good evening. All systems are nominal and at your disposal."
    if args.output:
        data = engine.synthesize(text)
        if not data:
            _out("The engine produced no audio. Run 'jarvis doctor' for a diagnosis.")
            return 1

        # Some engines (edge-tts) hand back MP3 when no converter is installed.
        # Correct the extension rather than writing an MP3 called ".wav".
        target = Path(args.output)
        fmt = getattr(engine, "output_format", "wav")
        if data[:4] == b"RIFF":
            fmt = "wav"
        if fmt != "wav" and target.suffix.lower() == ".wav":
            target = target.with_suffix(f".{fmt}")
            _out(f"Engine returned {fmt.upper()}; writing {target.name} instead.")
            _out("Install ffmpeg (or pydub) to get WAV output from this engine.")

        target.write_bytes(data)
        _out(f"Wrote {len(data)} bytes to {target}")
    else:
        engine.speak(text)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """One question, one answer, then exit."""
    cfg = _load(args)
    from . import app as app_module

    subsystems = app_module.build(cfg, configure_logging=False)
    try:
        if subsystems.orchestrator is None:
            _out("JARVIS could not start. Run 'jarvis doctor' to see why.")
            return 1
        reply = subsystems.orchestrator.chat(args.text)
        _out(f"\n{reply}\n")
    finally:
        app_module.shutdown(subsystems)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive text conversation."""
    cfg = _load(args)
    from . import app as app_module

    subsystems = app_module.build(cfg, configure_logging=False)
    # An interactive session can actually ask, so wire the confirmation prompt.
    subsystems.security.set_confirm_callback(_confirm)

    agent = subsystems.orchestrator
    if agent is None:
        _out("JARVIS could not start. Run 'jarvis doctor' to see why.")
        app_module.shutdown(subsystems)
        return 1

    _out(BANNER)
    status = subsystems.status()
    _out(f"  model {status['model']} via {status['llm']} | voice {status['tts']} "
         f"| {status['tools']} tools")
    _out("  Type your request. 'exit' to quit, 'tasks' to list background work.\n")
    _out(f"{cfg.agent.name}: {agent.greet()}\n")

    try:
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                _out()
                break
            if not line:
                continue
            lowered = line.lower()
            if lowered in ("exit", "quit", "bye", ":q"):
                break
            if lowered == "tasks":
                for task in agent.tasks.list():
                    _out(f"  {task.id}  {task.state.value:<10} {task.goal[:60]}")
                continue

            reply = agent.chat(line)
            _out(f"\n{cfg.agent.name}: {reply}\n")

            for update in agent.pending_updates():
                _out(f"[background] {update}\n")
    finally:
        _out(f"{cfg.agent.name}: Very good. Shutting down.")
        app_module.shutdown(subsystems)
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    """Hands-free conversation."""
    cfg = _load(args)
    from . import app as app_module
    from .voice import VoiceLoop

    subsystems = app_module.build(cfg, configure_logging=False)
    subsystems.security.set_confirm_callback(_confirm)

    agent = subsystems.orchestrator
    if agent is None:
        _out("JARVIS could not start. Run 'jarvis doctor' to see why.")
        app_module.shutdown(subsystems)
        return 1

    loop = VoiceLoop(
        cfg, agent, subsystems.stt, bus=subsystems.bus,
        on_transcript=lambda t: _out(f"You: {t}"),
        on_reply=lambda r: _out(f"{cfg.agent.name}: {r}\n"),
    )

    _out(BANNER)
    if not loop.available():
        _out("No usable microphone or speech-to-text engine was found.")
        _out("Run 'jarvis doctor' for details, or use 'jarvis chat' for text mode.")
        app_module.shutdown(subsystems)
        return 1

    wake = ", ".join(cfg.voice.wake_words)
    _out(f"  Listening. Say '{wake}' to address me. Ctrl+C to stop.\n")
    agent.greet()

    try:
        loop.run()
    except KeyboardInterrupt:
        _out("\nStopping.")
    finally:
        loop.stop()
        app_module.shutdown(subsystems)
    return 0


# --------------------------------------------------------------------------- #
#  Remote access
# --------------------------------------------------------------------------- #
DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 8765
TOKEN_ENV_VAR = "JARVIS_SERVER_TOKEN"
TOKEN_FILE_NAME = "server_token"

# How a token was obtained, phrased for the operator.  The value is never
# printed for any of these — only for a token generated during this run.
_TOKEN_SOURCES = {
    "flag": "--token",
    "environment": f"${TOKEN_ENV_VAR}",
    "config": "the configuration file",
    "file": "the saved token file",
}


def _is_loopback(host: str) -> bool:
    """True when *host* can only be reached from this machine.

    Anything that is not demonstrably loopback — a wildcard bind, a LAN
    address, a hostname this function cannot resolve to an IP literal — is
    reported as reachable from elsewhere, because guessing the other way would
    silently drop the token requirement.
    """
    candidate = (host or "").strip().strip("[]")
    if not candidate:
        return False
    if candidate.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _lan_address() -> Optional[str]:
    """Best-effort LAN address of this machine, or ``None``.

    Asks the kernel which local address it would use to reach an off-link
    destination.  The socket is never sent on and the address used is RFC 5737
    documentation space, so this neither transmits anything nor needs the
    network to be up.
    """
    address = ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
    except OSError:
        address = ""
    if address and not address.startswith("127."):
        return address
    try:
        resolved = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    return resolved if resolved and not resolved.startswith("127.") else None


def _url_host(host: str) -> str:
    """Turn a bind address into something that can go in a URL.

    ``0.0.0.0`` is a bind address, not a destination: nothing can connect to
    it, so it becomes the LAN address when one can be found and loopback
    otherwise.  IPv6 literals gain the brackets a URL requires.
    """
    candidate = (host or "").strip().strip("[]")
    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError:
        return candidate or DEFAULT_SERVE_HOST
    if parsed.is_unspecified:
        return _lan_address() or DEFAULT_SERVE_HOST
    if parsed.version == 6:
        return f"[{candidate}]"
    return candidate


def _serve_url(host: str, port: int, tls: bool = False) -> str:
    """The URL a browser should open for this bind."""
    return f"{'https' if tls else 'http'}://{_url_host(host)}:{port}"


def _token_file(cfg: Config) -> Path:
    return cfg.path(TOKEN_FILE_NAME)


def _resolve_token(
    cfg: Config,
    requested: Optional[str] = None,
    *,
    environ: Optional[dict] = None,
) -> Tuple[str, str]:
    """Find the shared token, generating and saving one if there is none.

    Returns ``(token, source)``.  ``source`` is ``"generated"`` only when the
    token was created during this call, which is the one case where printing it
    is the whole point; every other source is already known to the operator and
    the value stays out of the terminal and the logs.

    An explicitly empty ``--token`` means "no token", and is honoured — the
    caller decides whether that is acceptable for the chosen bind address.
    """
    if requested is not None:
        stripped = requested.strip()
        return (stripped, "flag") if stripped else ("", "none")

    environ = environ if environ is not None else os.environ
    from_env = (environ.get(TOKEN_ENV_VAR) or "").strip()
    if from_env:
        return from_env, "environment"

    server_cfg = getattr(cfg, "server", None)
    from_cfg = (getattr(server_cfg, "token", "") or "").strip()
    if from_cfg:
        return from_cfg, "config"

    path = _token_file(cfg)
    try:
        saved = path.read_text(encoding="utf-8").strip()
    except OSError:
        saved = ""
    if saved:
        return saved, "file"

    token = secrets.token_urlsafe(32)
    try:
        # Created 0600 by os.open rather than written and then chmod'ed: this
        # token is full control of the machine, and the second form leaves it
        # world-readable for the window between the two calls.
        handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Windows and some filesystems will not take the mode; the file is
            # still inside the owner's own profile directory.
            log.debug("could not chmod 0600 %s", path, exc_info=True)
    except OSError as exc:  # noqa: BLE001
        # A token that cannot be saved still works for this run.
        log.warning("could not save the server token to %s: %s", path, exc)
    return token, "generated"


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the browser client, so another device can talk to *this* JARVIS.

    The assistant, its tools, its memory and its machine control all stay here.
    The other device runs nothing but a browser: it captures microphone audio
    locally and plays the reply locally, which is the only arrangement where
    speaking into a laptop in another room reaches the right microphone.
    """
    host = (args.host or DEFAULT_SERVE_HOST).strip()
    port = int(args.port)

    if bool(args.cert) != bool(args.key):
        _out("TLS needs both --cert and --key; only one was given.")
        return 2
    tls = bool(args.cert and args.key)

    if args.print_url:
        # Scripting hook: say where the server would be, bind nothing, exit 0.
        _out(_serve_url(host, port, tls))
        return 0

    cfg = _load(args)
    loopback = _is_loopback(host)
    token, source = _resolve_token(cfg, args.token)

    if not token and not loopback:
        _out(f"Refusing to bind {host}:{port} with no token.")
        _out("")
        _out("JARVIS has full run of this machine by design — any file, any command,")
        _out("the desktop itself. On a loopback bind that reach stops at this machine.")
        _out(f"On {host} it does not: whatever can open the port gets the same access")
        _out("you have. A shared token is what keeps other people out.")
        _out("")
        _out("Either drop the empty --token, or keep the bind on 127.0.0.1 and reach it")
        _out("through SSH from the other device:")
        _out(f"  ssh -N -L {port}:127.0.0.1:{port} user@this-machine")
        _out("")
        _out("docs/REMOTE_ACCESS.md walks through both.")
        return 2

    try:
        from .server import serve as serve_forever
    except ImportError as exc:
        _out("The remote-access server (jarvis.server) is not available in this copy.")
        _out(f"  {exc}")
        _out("")
        _out("Until it is, another device can still reach JARVIS over plain SSH:")
        _out("  ssh user@this-machine jarvis chat")
        _out("See docs/REMOTE_ACCESS.md.")
        return 1

    _out(BANNER)
    _out(f"  URL     {_serve_url(host, port, tls)}")
    _out(f"  bind    {host}:{port}" + ("  (loopback — this machine only)" if loopback else ""))
    if tls:
        _out(f"  TLS     {args.cert}")

    if not token:
        _out("  token   none — loopback bind, so nothing off this machine can connect")
    elif source == "generated":
        _out(f"  token   {token}")
        _out(f"          saved to {_token_file(cfg)}; the same one is reused next time")
    else:
        _out(f"  token   set, from {_TOKEN_SOURCES.get(source, source)}")

    if not loopback:
        lan = _lan_address()
        if lan:
            _out(f"  LAN     {'https' if tls else 'http'}://{lan}:{port}")
        _out("")
        _out("  This port is this machine. Whoever reaches it can read and write any")
        _out("  file and run any command, exactly as you can. Prefer an SSH tunnel or")
        _out("  Tailscale to opening a port on the router:")
        _out(f"    ssh -N -L {port}:127.0.0.1:{port} user@{lan or 'this-machine'}")
        if not tls:
            _out("  A tunnel also makes the microphone work: browsers only grant it on")
            _out("  https or localhost, and the tunnel makes this localhost on the client.")
        _out("  docs/REMOTE_ACCESS.md")

    # The subsystems are built inside serve(), after this banner: on a CPU-only
    # box that is tens of seconds during which the port refuses connections.
    # Saying so here is the difference between "still loading" and "broken".
    _out("\n  Starting the subsystems; the port refuses connections until that finishes.")
    _out("  Ctrl+C to stop.\n")

    try:
        serve_forever(cfg, host=host, port=port, token=token,
                      certfile=args.cert, keyfile=args.key)
    except KeyboardInterrupt:
        _out("\nStopping.")
    except OSError as exc:  # noqa: BLE001
        from .llm.models import redact

        _out(f"\nCould not serve on {host}:{port}: {redact(exc)}")
        if getattr(exc, "errno", None) is not None:
            _out("Another process may already hold that port; try --port.")
        return 1
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Fetch the pieces that need downloading, and verify the result."""
    cfg = _load(args)
    _out("Preparing JARVIS...\n")

    _out(f"  data directory: {cfg.home()}")
    for directory in (cfg.home(), cfg.tools_dir(), cfg.voices_dir(),
                      cfg.models_dir(), cfg.logs_dir()):
        _out(f"    {directory}")

    _out("\n  Voice model:")
    try:
        from .speech.tts import PiperTTS

        piper = PiperTTS(cfg.tts, voices_dir=cfg.voices_dir())
        result = piper.ensure_voice(download=not args.no_download)
        _out(f"    {result}")
    except Exception as exc:  # noqa: BLE001
        _out(f"    could not prepare the Piper voice: {exc}")
        _out("    (edge-tts or the system voice will be used instead)")

    _out("\n  Checking subsystems:")
    from . import app as app_module

    subsystems = app_module.build(cfg, configure_logging=False)
    for key, value in subsystems.status().items():
        _out(f"    {key:<15} {value}")
    app_module.shutdown(subsystems)

    _out("\nDone. Try:  jarvis say   /   jarvis chat   /   jarvis voice")
    return 0


# --------------------------------------------------------------------------- #
#  Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — a local voice assistant with full access to this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("-c", "--config", help="path to a config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--model", help="override the language model id")
    parser.add_argument("--backend", help="override the LLM backend "
                                          "(airllm, ollama, transformers, stub)")
    parser.add_argument("--no-speech", action="store_true", help="disable the voice")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive text conversation").set_defaults(func=cmd_chat)
    sub.add_parser("voice", help="hands-free conversation").set_defaults(func=cmd_voice)
    sub.add_parser("doctor", help="check dependencies and configuration").set_defaults(func=cmd_doctor)
    sub.add_parser("tools", help="list registered tools").set_defaults(func=cmd_tools)

    p_ask = sub.add_parser("ask", help="ask one question and exit")
    p_ask.add_argument("text", help="the question")
    p_ask.set_defaults(func=cmd_ask)

    p_say = sub.add_parser("say", help="speak a line (auditions the voice)")
    p_say.add_argument("text", nargs="?", help="what to say")
    p_say.add_argument("-o", "--output", help="write an audio file instead of playing")
    p_say.set_defaults(func=cmd_say)

    p_cfg = sub.add_parser("config", help="show or write the configuration")
    p_cfg.add_argument("--write", nargs="?", const=True, default=False,
                       metavar="PATH", help="write the effective config to disk")
    p_cfg.set_defaults(func=cmd_config)

    p_mem = sub.add_parser("memory", help="inspect long-term memory")
    p_mem.add_argument("query", nargs="?", help="search the memory")
    p_mem.add_argument("-n", "--limit", type=int, default=10)
    p_mem.add_argument("--stats", action="store_true")
    p_mem.add_argument("--export", metavar="PATH", help="export everything to JSONL")
    p_mem.set_defaults(func=cmd_memory)

    p_serve = sub.add_parser(
        "serve",
        help="serve the browser client so another device can talk to this JARVIS",
    )
    p_serve.add_argument(
        "--host", default=DEFAULT_SERVE_HOST,
        help=f"address to bind (default {DEFAULT_SERVE_HOST}: reachable only from "
             f"this machine; 0.0.0.0 exposes it to the LAN and then a token is required)",
    )
    p_serve.add_argument(
        "--port", type=int, default=DEFAULT_SERVE_PORT,
        help=f"TCP port (default {DEFAULT_SERVE_PORT})",
    )
    p_serve.add_argument(
        "--token",
        help=f"shared token clients must present (default: ${TOKEN_ENV_VAR}, the "
             f"config, the saved token file, or a freshly generated one)",
    )
    p_serve.add_argument("--cert", metavar="PEM", help="TLS certificate; needs --key")
    p_serve.add_argument("--key", metavar="PEM", help="TLS private key; needs --cert")
    p_serve.add_argument(
        "--print-url", action="store_true",
        help="print the URL and exit, without binding anything",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_setup = sub.add_parser("setup", help="download models and verify the install")
    p_setup.add_argument("--no-download", action="store_true",
                         help="check only; do not fetch anything")
    p_setup.set_defaults(func=cmd_setup)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # Bare `jarvis` is the hands-free assistant: microphone in, voice out.
        # `jarvis chat` is the text-only conversation.
        args.command = "voice"
        args.func = cmd_voice
        reason = ""
        try:
            cfg = load_config(getattr(args, "config", None))
            from .speech.audio_io import AudioRecorder

            if not AudioRecorder(cfg.stt).is_available():
                reason = "no usable microphone was found"
        except Exception as exc:  # noqa: BLE001
            reason = f"the audio stack could not be initialised ({exc})"

        if reason:
            # Say why. Dropping silently into text mode looks like the voice
            # assistant simply ignoring the fact that it is a voice assistant.
            args.command = "chat"
            args.func = cmd_chat
            _out(f"Starting in text mode: {reason}.")
            _out("Run 'jarvis doctor' to diagnose audio, or 'jarvis voice' to force voice mode.\n")

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _out("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("command failed")
        _out(f"\nError: {exc}")
        _out("Run 'jarvis doctor' for a diagnosis, or -v for a full traceback.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
