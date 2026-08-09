"""Provisioning: everything JARVIS needs on disk before it can actually run.

The Python packages are the easy half.  A working JARVIS also needs an
inference runtime (Ollama), sixteen gigabytes of model weights, a Piper voice
and a Whisper model — and until this package existed the installer *printed the
commands* for those and left the owner to run them by hand.  This is the half
that makes ``./install.sh`` produce a working assistant, and makes running it
again update every piece rather than only the Python.

* :mod:`jarvis.runtime.ollama` — rootless Ollama: install, upgrade, serve as a
  systemd **user** service, and pull the model.  No ``sudo``, ever.
* :mod:`jarvis.runtime.assets` — the Piper voice, the faster-whisper model, and
  a report on espeak-ng.

``install.sh`` shells into these functions, which is the point of writing them
in Python: shell is not testable, and this is.  The three rules every public
function here obeys:

* **Safe to call twice.**  A second run downloads nothing that is current.
* **Never raises.**  Offline, out of disk, and nothing installed are ordinary
  states of the target machine, and each comes back as a dict with ``ok`` and a
  ``reason`` a human can act on.
* **Reports what it did**, not what it intended.

Sizes are checked *before* the download starts, never after: a 16 GB pull that
dies at 90% on a laptop is a genuinely bad afternoon.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from . import assets, ollama
from .assets import (
    COMPONENTS,
    disk_report,
    ensure_piper_voice,
    ensure_whisper_model,
    espeak_status,
    piper_voice_status,
    provision_all,
    whisper_model_status,
)
from .ollama import (
    ensure_model,
    install_plan,
    install_service,
    is_installed,
    is_serving,
    list_models,
    pull_plan,
    service_unit_text,
    start_server,
    stop_server,
)


def ensure_ollama(cfg: Any = None, *,
                  progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                  **kwargs: Any) -> Dict[str, Any]:
    """Install or upgrade the rootless Ollama runtime.  Idempotent."""
    return ollama.install(cfg, progress=progress, **kwargs)


def status(cfg: Any = None) -> Dict[str, Any]:
    """The whole picture, read-only — for ``jarvis doctor`` and the installer.

    No downloads, no daemons started, no files written.  Every component
    reports itself even when the others are broken, because the machine this
    runs on is usually the one where something is wrong.
    """
    runtime = assets._safe("ollama", lambda: ollama.status(cfg))
    speech = assets.status(cfg)
    voice = speech.get("voice", {})
    stt = speech.get("stt", {})
    espeak = speech.get("espeak", {})

    ready = bool(
        runtime.get("installed")
        and runtime.get("model_present")
        and voice.get("present")
        and stt.get("present")
    )
    missing = []
    if not runtime.get("installed"):
        missing.append("ollama")
    if not runtime.get("model_present"):
        missing.append("model")
    if not voice.get("present"):
        missing.append("voice")
    if not stt.get("present"):
        missing.append("stt")

    return {
        "ready": ready,
        "missing": missing,
        "ollama": runtime,
        "voice": voice,
        "stt": stt,
        "espeak": espeak,
        "disk": speech.get("disk", {}),
    }


def provision(cfg: Any, *,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None,
              skip: Optional[Sequence[str]] = None,
              with_service: bool = False) -> Dict[str, Any]:
    """Alias for :func:`jarvis.runtime.assets.provision_all` — the installer entry point.

    The keyword is ``with_service`` here rather than ``install_service`` only so
    that it does not shadow the :func:`jarvis.runtime.ollama.install_service`
    exported alongside it.
    """
    return provision_all(cfg, progress=progress, skip=skip,
                         install_service=with_service)


__all__ = [
    "assets", "ollama",
    "COMPONENTS",
    "disk_report",
    "ensure_model", "ensure_ollama", "ensure_piper_voice", "ensure_whisper_model",
    "espeak_status",
    "install_plan", "install_service", "is_installed", "is_serving", "list_models",
    "piper_voice_status", "provision", "provision_all", "pull_plan",
    "service_unit_text", "start_server", "status", "stop_server",
    "whisper_model_status",
]
