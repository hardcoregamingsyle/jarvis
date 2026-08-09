"""The speech assets: the voice JARVIS speaks with and the ears it listens through.

Three things have to be on disk before the first spoken exchange works, and
none of them ship with the Python packages:

* **The Piper voice** (``en_GB-alan-medium``, ~63 MB) — two files fetched from
  ``rhasspy/piper-voices`` on Hugging Face.  :class:`jarvis.speech.tts.PiperTTS`
  already knows how to fetch them; this module drives that rather than growing a
  second downloader that would drift out of step with the first.
* **The faster-whisper model** (``base.en``, ~150 MB) — pulled into the Hugging
  Face cache the first time a ``WhisperModel`` is constructed.  Left to itself
  that download happens *during the first sentence somebody speaks*, and the
  assistant sits mute for a minute or two with no indication it is doing
  anything.  Pre-fetching it at install time is the entire reason this function
  exists.
* **espeak-ng** — the last-resort voice, a system package.  It is only ever
  *reported* here: installing it needs root, and the installer's standing
  promise is that it never runs ``sudo``.

Everything is idempotent, everything reports what it found rather than what it
intended, and nothing raises into the caller — an absent dependency, an
unreachable Hugging Face and a full disk are all ordinary conditions on the
target machine.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..core import platform_utils

log = logging.getLogger(__name__)

_GB = 1_000_000_000.0

#: Where faster-whisper's size aliases actually live on the Hub.  ``Systran``
#: is the current publisher; ``guillaumekln`` is where the same models lived
#: before, and a machine that has been running for a while may hold either.
_WHISPER_REPO_OWNERS = ("Systran", "guillaumekln")

#: A CTranslate2 Whisper directory is only usable with all of these present.
_WHISPER_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")

#: Rough on-disk sizes, for a plan that has to render before anything is
#: downloaded.  Approximate on purpose: their job is to answer "will this fit",
#: not to predict the last megabyte.
_WHISPER_SIZE_GB = {
    "tiny": 0.08, "tiny.en": 0.08,
    "base": 0.15, "base.en": 0.15,
    "small": 0.5, "small.en": 0.5,
    "medium": 1.6, "medium.en": 1.6,
    "large-v2": 3.1, "large-v3": 3.1, "large": 3.1,
}
_PIPER_SIZE_GB = 0.07


def _which(name: str) -> Optional[str]:
    """``shutil.which`` via platform_utils.  Patched in tests."""
    return platform_utils.which(name)


def _notify(progress: Optional[Callable[[Dict[str, Any]], None]],
            event: Dict[str, Any]) -> None:
    if progress is None:
        return
    try:
        progress(event)
    except Exception as exc:  # noqa: BLE001 - the callback is the caller's problem
        log.debug("progress callback raised %s", type(exc).__name__)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _dir_size(path: Path, *, limit: int = 100_000) -> int:
    """Bytes under ``path``; bounded, symlink-safe, never raises."""
    total = 0
    seen = 0
    try:
        for root, _dirs, files in os.walk(str(path), followlinks=False):
            for name in files:
                seen += 1
                if seen > limit:
                    return total
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        log.debug("could not size %s", path)
    return total


# --------------------------------------------------------------------------- #
#  Disk
# --------------------------------------------------------------------------- #
def _existing_ancestor(path: Path) -> Optional[Path]:
    """Nearest ancestor of ``path`` that exists — a target need not yet be there."""
    try:
        current = Path(path)
    except (TypeError, ValueError):
        return None
    for _ in range(64):
        try:
            if current.exists():
                return current
        except OSError:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return None


def disk_report(paths: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Free space at each target, keyed by path.

    Every large download in the installer is preceded by one of these, because
    the failure it prevents — a 16 GB pull that dies at 90% — costs an
    afternoon, and the check costs a syscall.
    """
    report: Dict[str, Any] = {}
    for raw in list(paths or []):
        try:
            path = Path(raw)
        except (TypeError, ValueError):
            continue
        key = str(path)
        anchor = _existing_ancestor(path)
        if anchor is None:
            report[key] = {
                "path": key, "exists": False, "ok": False,
                "total_bytes": 0, "used_bytes": 0, "free_bytes": -1,
                "total_gb": None, "used_gb": None, "free_gb": None,
                "reason": "no existing parent directory to measure",
            }
            continue
        try:
            usage = shutil.disk_usage(str(anchor))
        except (OSError, ValueError) as exc:
            report[key] = {
                "path": key, "exists": True, "ok": False,
                "total_bytes": 0, "used_bytes": 0, "free_bytes": -1,
                "total_gb": None, "used_gb": None, "free_gb": None,
                "reason": f"disk_usage failed: {exc}",
            }
            continue
        report[key] = {
            "path": key,
            "exists": True,
            "ok": True,
            "measured_at": str(anchor),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "total_gb": round(usage.total / _GB, 2),
            "used_gb": round(usage.used / _GB, 2),
            "free_gb": round(usage.free / _GB, 2),
            "reason": "",
        }
    return report


# --------------------------------------------------------------------------- #
#  Piper voice
# --------------------------------------------------------------------------- #
def _piper_engine(cfg: Any) -> Any:
    """A :class:`PiperTTS` bound to this config, or ``None`` if unimportable."""
    try:
        from ..speech.tts import PiperTTS
    except ImportError as exc:  # pragma: no cover - first-party module
        log.debug("PiperTTS unavailable: %s", exc)
        return None
    try:
        return PiperTTS(cfg.tts, voices_dir=cfg.voices_dir())
    except Exception as exc:  # noqa: BLE001 - a bad config is a result, not a crash
        log.debug("could not construct PiperTTS: %s", exc)
        return None


def _piper_result(cfg: Any, info: Optional[Dict[str, Any]], *,
                  downloaded: bool = False) -> Dict[str, Any]:
    voice = str(getattr(getattr(cfg, "tts", None), "piper_voice", "") or "")
    if info is None:
        return {
            "component": "piper-voice", "ok": False, "present": False,
            "voice": voice, "model_path": None, "config_path": None,
            "size_bytes": 0, "size_gb": 0.0, "downloaded": False,
            "expected_size_gb": _PIPER_SIZE_GB,
            "reason": "jarvis.speech.tts could not be loaded, so the Piper voice "
                      "could not be checked.",
        }
    model = Path(info.get("model_path") or "")
    config = Path(info.get("config_path") or "")
    present = bool(model.is_file() and config.is_file())
    size = _file_size(model) + _file_size(config)
    errors = [str(e) for e in (info.get("errors") or [])]
    return {
        "component": "piper-voice",
        "ok": present,
        "present": present,
        "voice": voice,
        "model_path": str(model),
        "config_path": str(config),
        "size_bytes": size,
        "size_gb": round(size / _GB, 3),
        "expected_size_gb": _PIPER_SIZE_GB,
        "downloaded": bool(info.get("downloaded")) or downloaded,
        "reason": (
            f"{voice} is present ({round(size / 1e6, 1)} MB)." if present
            else "; ".join(errors) or f"{voice} is not downloaded."
        ),
    }


def piper_voice_status(cfg: Any) -> Dict[str, Any]:
    """Is the Piper voice on disk?  Never touches the network."""
    engine = _piper_engine(cfg)
    if engine is None:
        return _piper_result(cfg, None)
    try:
        info = engine.ensure_voice(download=False)
    except Exception as exc:  # noqa: BLE001 - a status probe never raises
        log.debug("piper ensure_voice(status) failed: %s", exc)
        return _piper_result(cfg, None)
    return _piper_result(cfg, info)


def ensure_piper_voice(cfg: Any, *,
                       progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                       ) -> Dict[str, Any]:
    """Download the Piper voice if it is missing.  Safe to call twice.

    The download itself belongs to :class:`jarvis.speech.tts.PiperTTS` — it owns
    the ``rhasspy/piper-voices`` URL layout, and duplicating that here would
    mean two copies to fix when the layout changes.
    """
    current = piper_voice_status(cfg)
    if current["present"]:
        _notify(progress, {"phase": "voice", "message": current["reason"]})
        return current

    engine = _piper_engine(cfg)
    if engine is None:
        return current
    _notify(progress, {"phase": "voice",
                       "message": f"downloading the {current['voice']} Piper voice "
                                  f"(~{int(_PIPER_SIZE_GB * 1000)} MB)"})
    try:
        info = engine.ensure_voice(download=True)
    except Exception as exc:  # noqa: BLE001 - a failed download is a result
        log.debug("piper ensure_voice(download) failed: %s", exc)
        current["reason"] = f"voice download failed: {type(exc).__name__}: {exc}"
        return current
    result = _piper_result(cfg, info, downloaded=True)
    _notify(progress, {"phase": "voice", "message": result["reason"]})
    return result


# --------------------------------------------------------------------------- #
#  faster-whisper
# --------------------------------------------------------------------------- #
def _whisper_model_name(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "stt", None), "model", "") or "base.en").strip()


def whisper_repo_candidates(name: str) -> List[str]:
    """Hub repositories that could hold the CTranslate2 build of ``name``.

    A bare size alias (``base.en``) is what ``faster_whisper`` expands; an
    explicit ``org/repo`` is taken as given, because pinning a specific
    conversion is a legitimate thing to configure.
    """
    text = (name or "").strip()
    if not text:
        return []
    if "/" in text:
        return [text]
    return [f"{owner}/faster-whisper-{text}" for owner in _WHISPER_REPO_OWNERS]


def _hf_cache_dir() -> Optional[Path]:
    try:
        from ..llm.models import hf_cache_dir
    except ImportError:  # pragma: no cover - first-party module
        return None
    try:
        return hf_cache_dir()
    except Exception:  # noqa: BLE001
        return None


def _cache_dir_for(repo: str, cache: Path) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def _looks_complete(root: Path) -> bool:
    """True when a cached snapshot holds the files CTranslate2 actually loads.

    An interrupted download leaves the directory *there* but missing
    ``model.bin``, and "the folder exists" would then report a model that
    cannot be loaded.
    """
    try:
        if not root.is_dir():
            return False
        names = {p.name for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    except OSError:
        return False
    return all(required in names for required in _WHISPER_REQUIRED_FILES)


def whisper_model_status(cfg: Any) -> Dict[str, Any]:
    """Where the STT model is (or would be), and whether it is usable.

    Never imports ``faster_whisper`` and never touches the network: this is the
    question ``jarvis doctor`` asks, and it must be answerable on a machine
    where nothing is installed.
    """
    name = _whisper_model_name(cfg)
    expected = _WHISPER_SIZE_GB.get(name, 0.0)
    result: Dict[str, Any] = {
        "component": "whisper-model",
        "ok": False,
        "present": False,
        "model": name,
        "path": None,
        "size_bytes": 0,
        "size_gb": 0.0,
        "expected_size_gb": expected,
        "cache_dir": None,
        "candidates": whisper_repo_candidates(name),
        "reason": "",
    }

    # An explicit directory in the config wins: faster-whisper accepts a local
    # CTranslate2 folder in place of a size alias.
    local = Path(name).expanduser() if name else None
    if local is not None and local.is_dir():
        size = _dir_size(local)
        result.update({
            "ok": True, "present": True, "path": str(local),
            "size_bytes": size, "size_gb": round(size / _GB, 3),
            "reason": f"using the local model directory {local}.",
        })
        return result

    cache = _hf_cache_dir()
    if cache is None:
        result["reason"] = "the Hugging Face cache location could not be determined."
        return result
    result["cache_dir"] = str(cache)

    for repo in result["candidates"]:
        path = _cache_dir_for(repo, cache)
        if _looks_complete(path):
            size = _dir_size(path)
            result.update({
                "ok": True, "present": True, "path": str(path), "repo": repo,
                "size_bytes": size, "size_gb": round(size / _GB, 3),
                "reason": f"{repo} is cached ({round(size / 1e6, 1)} MB).",
            })
            return result

    result["reason"] = (
        f"the {name} speech-to-text model is not cached; it is about "
        f"{int(expected * 1000)} MB and would otherwise download during the first "
        "spoken sentence."
        if expected else
        f"the {name} speech-to-text model is not cached."
    )
    return result


def ensure_whisper_model(cfg: Any, *,
                         progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                         ) -> Dict[str, Any]:
    """Pre-fetch the faster-whisper model so the first utterance is not silent.

    Constructing a ``WhisperModel`` is what triggers the download; there is no
    separate "fetch" entry point in the library, so the model is built and then
    dropped.  ``faster_whisper`` is imported lazily and its absence is reported,
    not raised — a machine can perfectly well run JARVIS with a different STT.
    """
    current = whisper_model_status(cfg)
    if current["present"]:
        _notify(progress, {"phase": "stt", "message": current["reason"]})
        return current

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        current["reason"] = (
            "faster-whisper is not installed, so the speech-to-text model was not "
            "pre-fetched (pip install faster-whisper)."
        )
        current["action"] = "unavailable"
        return current
    except Exception as exc:  # noqa: BLE001 - a broken install is a result
        current["reason"] = f"faster-whisper could not be imported: {type(exc).__name__}: {exc}"
        current["action"] = "unavailable"
        return current

    name = current["model"]
    stt = getattr(cfg, "stt", None)
    compute = str(getattr(stt, "compute_type", "") or "int8")
    _notify(progress, {"phase": "stt",
                       "message": f"fetching the {name} speech-to-text model "
                                  f"(~{int(current['expected_size_gb'] * 1000)} MB)"})
    try:
        model = WhisperModel(name, device="cpu", compute_type=compute)
        del model
    except Exception as exc:  # noqa: BLE001 - download/convert failures are results
        current["reason"] = f"could not fetch the {name} model: {type(exc).__name__}: {exc}"
        current["action"] = "failed"
        return current

    refreshed = whisper_model_status(cfg)
    refreshed["action"] = "fetched" if refreshed["present"] else "unverified"
    if not refreshed["present"]:
        # The constructor succeeded, so the model loaded from somewhere the
        # cache scan does not know about. That is a reporting gap, not a
        # failure, and saying so is more useful than claiming it is missing.
        refreshed["ok"] = True
        refreshed["reason"] = (
            f"the {name} model loaded successfully but was not found in the "
            "Hugging Face cache; it may be cached elsewhere."
        )
    _notify(progress, {"phase": "stt", "message": refreshed["reason"]})
    return refreshed


# --------------------------------------------------------------------------- #
#  espeak-ng
# --------------------------------------------------------------------------- #
def espeak_status() -> Dict[str, Any]:
    """Is the last-resort voice available?

    Only ever reported.  ``espeak-ng`` is a system package, and installing one
    needs root — so the command is printed for the owner to run, exactly as the
    rest of the installer does.
    """
    for name in ("espeak-ng", "espeak"):
        found = _which(name)
        if found:
            return {
                "component": "espeak", "ok": True, "present": True,
                "binary": name, "path": found,
                "reason": f"{name} is available at {found}.",
                "install_hint": "",
            }
    return {
        "component": "espeak", "ok": False, "present": False,
        "binary": None, "path": None,
        "reason": "espeak-ng is not installed. It is only the last-resort voice "
                  "(Piper is the default), so this is not fatal.",
        "install_hint": "sudo apt-get install -y espeak-ng   # or your distro's equivalent",
    }


# --------------------------------------------------------------------------- #
#  Everything at once
# --------------------------------------------------------------------------- #
#: Names accepted by ``provision_all(skip=...)``.
COMPONENTS = ("ollama", "model", "voice", "stt", "espeak")


def _safe(name: str, call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Run one provisioning step; turn any escape into a reported failure.

    ``provision_all`` is the function the installer runs unattended.  One
    component blowing up must not take the other four with it.
    """
    try:
        value = call()
    except Exception as exc:  # noqa: BLE001 - this is the containment boundary
        log.debug("provisioning step %s raised %s", name, type(exc).__name__, exc_info=True)
        return {
            "component": name, "ok": False, "present": False,
            "reason": f"{name} could not be provisioned: {type(exc).__name__}: {exc}",
        }
    if not isinstance(value, dict):  # pragma: no cover - defensive
        return {"component": name, "ok": False, "reason": f"{name} returned {type(value).__name__}"}
    value.setdefault("component", name)
    return value


def _target_paths(cfg: Any) -> List[Path]:
    """Every location the installer is about to write something large into."""
    paths: List[Path] = []
    try:
        from . import ollama as ollama_runtime
        paths.append(ollama_runtime.runtime_dir(cfg))
        paths.append(ollama_runtime.models_cache_dir())
    except Exception:  # noqa: BLE001 - a disk report is best-effort
        log.debug("could not resolve the ollama paths for the disk report")
    try:
        paths.append(Path(cfg.voices_dir()))
    except Exception:  # noqa: BLE001
        log.debug("could not resolve the voices directory for the disk report")
    try:
        from ..llm.models import hf_cache_dir
        cache = hf_cache_dir()
        if cache is not None:
            paths.append(cache)
    except Exception:  # noqa: BLE001
        log.debug("could not resolve the HF cache for the disk report")
    return paths


def provision_all(cfg: Any, *,
                  progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                  skip: Optional[Sequence[str]] = None,
                  install_service: bool = False,
                  start_server: bool = True,
                  refresh: bool = False) -> Dict[str, Any]:
    """Install everything JARVIS needs to actually run, and say what happened.

    Idempotent from end to end: on a provisioned machine every step reports
    "already present" and nothing is downloaded.  Nothing here raises — with
    every single component missing and no network at all, this still returns a
    dict describing five failures.

    ``skip`` takes any of :data:`COMPONENTS`; the big ones (``ollama``,
    ``model``, ``stt``) are the ones worth skipping on a metered connection.

    ``start_server`` brings the Ollama daemon up before the model is pulled,
    because installing the binary does not start it and a pull has nowhere to
    go otherwise.  Pass ``False`` when something else owns the daemon's
    lifetime, or in a test that must not spawn one.

    ``refresh`` re-checks upstream for models and speech assets that are
    already present, which is what an *update* run wants; the default only
    ensures presence.
    """
    skipped = {str(s).strip().lower() for s in (skip or [])}
    components: Dict[str, Dict[str, Any]] = {}

    def wanted(name: str) -> bool:
        return name not in skipped

    try:  # pragma: no cover - a first-party import that cannot normally fail
        from . import ollama as ollama_runtime
    except Exception as exc:  # noqa: BLE001 - even this must not abort provisioning
        ollama_runtime = None  # type: ignore[assignment]
        log.debug("jarvis.runtime.ollama unavailable: %s", exc)

    if ollama_runtime is None:
        components["ollama"] = {
            "component": "ollama", "ok": False,
            "reason": "the ollama runtime module could not be imported.",
        }
        components["model"] = dict(components["ollama"], component="model")
    else:
        if wanted("ollama"):
            components["ollama"] = _safe(
                "ollama", lambda: ollama_runtime.install(cfg, progress=progress)
            )
            if install_service and components["ollama"].get("ok"):
                components["service"] = _safe(
                    "service", lambda: ollama_runtime.install_service(cfg)
                )
        else:
            components["ollama"] = _safe(
                "ollama", lambda: dict(ollama_runtime.status(cfg), component="ollama",
                                       ok=True, action="skipped",
                                       reason="skipped by request.")
            )

        if wanted("model"):
            # The daemon has to be up before a pull can reach it. Installing the
            # binary does not start it, so without this a freshly provisioned
            # machine got as far as "no Ollama server answered" and stopped —
            # the model could never be fetched through this function at all.
            # install.sh worked only because it starts the server itself.
            if start_server:
                components["server"] = _safe(
                    "server", lambda: ollama_runtime.start_server(cfg)
                )
            components["model"] = _safe(
                "model",
                lambda: ollama_runtime.ensure_model(
                    cfg, progress=progress, refresh=refresh
                ),
            )
        else:
            components["model"] = {
                "component": "model", "ok": True, "action": "skipped",
                "reason": "skipped by request.",
            }

    if wanted("voice"):
        components["voice"] = _safe(
            "voice", lambda: ensure_piper_voice(cfg, progress=progress)
        )
    else:
        components["voice"] = _safe("voice", lambda: piper_voice_status(cfg))
        components["voice"]["action"] = "skipped"

    if wanted("stt"):
        components["stt"] = _safe(
            "stt", lambda: ensure_whisper_model(cfg, progress=progress)
        )
    else:
        components["stt"] = _safe("stt", lambda: whisper_model_status(cfg))
        components["stt"]["action"] = "skipped"

    if wanted("espeak"):
        components["espeak"] = _safe("espeak", espeak_status)
    else:
        components["espeak"] = {"component": "espeak", "ok": True, "action": "skipped",
                                "reason": "skipped by request."}

    disk = _safe("disk", lambda: {"component": "disk", "ok": True,
                                  "targets": disk_report(_target_paths(cfg))})

    # espeak is advisory: Piper is the default voice and a machine without
    # espeak-ng is fully functional.
    #
    # "server" is advisory for a different reason: it is a means, not a
    # deliverable. Starting the daemon matters only so the model can be pulled,
    # so its success is already implied by the model step — and its failure is
    # already reported there, in terms the reader can act on. Counting it twice
    # would fail a provision whose every actual artefact is present.
    advisory = ("espeak", "server")
    required = [name for name in components if name not in advisory]
    failures = [name for name in required if not components[name].get("ok")]

    summary: List[str] = []
    for name in ("ollama", "service", "model", "voice", "stt", "espeak"):
        entry = components.get(name)
        if entry is None:
            continue
        mark = "ok" if entry.get("ok") else "!!"
        summary.append(f"[{mark}] {name}: {entry.get('reason') or '(no detail)'}")

    return {
        "ok": not failures,
        "components": components,
        "disk": disk,
        "skipped": sorted(skipped),
        "failed": failures,
        "summary": summary,
    }


def status(cfg: Any) -> Dict[str, Any]:
    """Read-only picture of the speech assets.  No downloads, no side effects."""
    return {
        "voice": _safe("voice", lambda: piper_voice_status(cfg)),
        "stt": _safe("stt", lambda: whisper_model_status(cfg)),
        "espeak": _safe("espeak", espeak_status),
        "disk": _safe("disk", lambda: {"component": "disk", "ok": True,
                                       "targets": disk_report(_target_paths(cfg))}),
    }


__all__ = [
    "COMPONENTS",
    "disk_report",
    "piper_voice_status", "ensure_piper_voice",
    "whisper_model_status", "ensure_whisper_model", "whisper_repo_candidates",
    "espeak_status",
    "provision_all", "status",
]
