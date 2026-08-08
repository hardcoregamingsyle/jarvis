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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence

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

    p_setup = sub.add_parser("setup", help="download models and verify the install")
    p_setup.add_argument("--no-download", action="store_true",
                         help="check only; do not fetch anything")
    p_setup.set_defaults(func=cmd_setup)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # Bare `jarvis`: prefer voice, fall back to text.
        args.command = "voice"
        args.func = cmd_voice
        try:
            cfg = load_config(getattr(args, "config", None))
            from .speech.audio_io import AudioRecorder

            if not AudioRecorder(cfg.stt).is_available():
                args.func = cmd_chat
        except Exception:  # noqa: BLE001
            args.func = cmd_chat

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
