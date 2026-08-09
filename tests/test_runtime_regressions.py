"""Defects found by adversarially verifying jarvis.runtime, pinned so they stay fixed.

Each of these was CONFIRMED against the real code — by running it, not by
reading it — and then repaired. They live in their own module because they are
about specific past failures rather than the module's general contract, and each
docstring records what actually went wrong.

The common thread is that every one of them was invisible in normal use: the
functions returned success, the installer printed a cheerful summary, and the
thing the owner asked for silently did not happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jarvis.core.config import load_config
from jarvis.runtime import assets, ollama


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every notion of "home" redirected into tmp_path."""
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    (home / ".config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.delenv("XDG_BIN_HOME", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return home


@pytest.fixture
def cfg(isolated_home: Path, fake_home: Path):
    configuration = load_config(use_env=False)
    configuration.data_dir = str(isolated_home)
    return configuration


PRESENT_PLAN: Dict[str, Any] = {
    "ok": True, "action": "present", "host": "http://h",
    "tag": "m:1", "reason": "m:1 is already pulled", "size_gb": 1,
}


def _all_provision_steps_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "install", lambda *a, **k: {"ok": True, "reason": "x"})
    monkeypatch.setattr(ollama, "ensure_model", lambda *a, **k: {"ok": True, "reason": "x"})
    monkeypatch.setattr(assets, "ensure_piper_voice", lambda *a, **k: {"ok": True, "reason": "x"})
    monkeypatch.setattr(assets, "ensure_whisper_model", lambda *a, **k: {"ok": True, "reason": "x"})


# --------------------------------------------------------------------------- #
#  A re-run must actually update the weights
# --------------------------------------------------------------------------- #
def test_a_re_run_refreshes_a_model_that_is_already_present(cfg, monkeypatch):
    """ensure_model returned "present" on an /api/tags hit and never pulled.

    So re-running the installer refreshed the code and the Python packages but
    left the weights at whatever was fetched the first time. An Ollama tag is a
    moving target — ``qwen3.6:27b`` is re-pointed upstream when the model is
    re-quantised or fixed — so "present" and "current" are different claims,
    and only the first one was being checked.
    """
    pulled: List[str] = []
    monkeypatch.setattr(ollama, "pull_plan", lambda *a, **k: dict(PRESENT_PLAN))
    monkeypatch.setattr(
        ollama, "pull",
        lambda tag, **k: (pulled.append(tag) or
                          {"ok": True, "downloaded": 3_000_000_000,
                           "reason": "pulled", "status": "success"}),
    )

    plain = ollama.ensure_model(cfg, tag="m:1")
    assert plain["action"] == "present"
    assert pulled == [], "a non-refresh call must not touch the network"

    refreshed = ollama.ensure_model(cfg, tag="m:1", refresh=True)
    assert pulled == ["m:1"], "refresh did not issue a pull"
    assert refreshed["ok"] is True
    assert refreshed["action"] == "updated"
    assert "3.0 GB" in refreshed["reason"]


def test_a_refresh_that_moves_no_bytes_reports_current(cfg, monkeypatch):
    """Ollama transfers only changed layers, so zero bytes means up to date."""
    monkeypatch.setattr(ollama, "pull_plan", lambda *a, **k: dict(PRESENT_PLAN))
    monkeypatch.setattr(
        ollama, "pull",
        lambda tag, **k: {"ok": True, "downloaded": 0, "reason": "ok", "status": "success"},
    )

    result = ollama.ensure_model(cfg, tag="m:1", refresh=True)
    assert result["action"] == "current"
    assert result["ok"] is True


def test_a_refresh_that_cannot_reach_the_registry_is_not_a_failure(cfg, monkeypatch):
    """Offline must not read as broken: the model is present and usable."""
    monkeypatch.setattr(ollama, "pull_plan", lambda *a, **k: dict(PRESENT_PLAN))
    monkeypatch.setattr(
        ollama, "pull",
        lambda tag, **k: {"ok": False, "downloaded": 0, "reason": "pull failed: offline"},
    )

    result = ollama.ensure_model(cfg, tag="m:1", refresh=True)
    assert result["ok"] is True
    assert result["action"] == "present"
    assert "could not check for updates" in result["reason"]


class _StreamingResponse:
    def __init__(self, frames: List[bytes]) -> None:
        self._frames = list(frames)

    def read(self, size: int = -1) -> bytes:
        return self._frames.pop(0) if self._frames else b""

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def test_pull_reports_the_bytes_it_actually_transferred(monkeypatch):
    """`downloaded` is the whole signal behind updated-vs-current.

    Ollama reports progress per layer and restarts the counter for each, so the
    last ``completed`` value is not the total. The peak is what distinguishes
    "moved data" from "moved none", which is all this needs to decide.
    """
    frames = [
        b'{"status":"pulling manifest"}\n',
        b'{"status":"pulling abc","completed":500,"total":1000}\n',
        b'{"status":"pulling abc","completed":1000,"total":1000}\n',
        b'{"status":"pulling def","completed":10,"total":10}\n',
        b'{"status":"success"}\n',
    ]
    monkeypatch.setattr(ollama, "_urlopen", lambda *a, **k: _StreamingResponse(frames))

    result = ollama.pull("m:1", host="http://h")
    assert result["ok"] is True
    assert result["downloaded"] == 1000, "the peak went backwards on the second layer"


def test_a_pull_that_downloads_nothing_reports_zero(monkeypatch):
    """The already-current case: manifest only, no completed fields at all."""
    frames = [
        b'{"status":"pulling manifest"}\n',
        b'{"status":"verifying sha256 digest"}\n',
        b'{"status":"success"}\n',
    ]
    monkeypatch.setattr(ollama, "_urlopen", lambda *a, **k: _StreamingResponse(frames))

    result = ollama.pull("m:1", host="http://h")
    assert result["ok"] is True
    assert result["downloaded"] == 0


# --------------------------------------------------------------------------- #
#  provision_all has to start the daemon it just installed
# --------------------------------------------------------------------------- #
def test_provision_all_starts_the_server_before_pulling(cfg, monkeypatch):
    """Installing the binary does not start it, and a pull needs a destination.

    Without this the model step on a fresh machine reported "no Ollama server
    answered" and stopped, so the model could never be fetched through this
    function at all. install.sh worked only because it starts the daemon itself
    — the public API did not.
    """
    order: List[str] = []
    monkeypatch.setattr(
        ollama, "install",
        lambda *a, **k: (order.append("install") or {"ok": True, "reason": "x"}))
    monkeypatch.setattr(
        ollama, "start_server",
        lambda *a, **k: (order.append("start") or {"ok": True, "reason": "x"}))
    monkeypatch.setattr(
        ollama, "ensure_model",
        lambda *a, **k: (order.append("model") or {"ok": True, "reason": "x"}))
    monkeypatch.setattr(assets, "ensure_piper_voice", lambda *a, **k: {"ok": True, "reason": "x"})
    monkeypatch.setattr(assets, "ensure_whisper_model", lambda *a, **k: {"ok": True, "reason": "x"})

    assets.provision_all(cfg)

    assert "start" in order, "the daemon was never started"
    assert order.index("start") < order.index("model"), f"wrong order: {order}"


def test_a_failed_server_start_does_not_fail_an_otherwise_complete_provision(cfg, monkeypatch):
    """Starting the daemon is a means, not a deliverable.

    Its failure is already reported by the model step in terms the reader can
    act on; counting it again would fail a provision whose every artefact is
    present.
    """
    _all_provision_steps_ok(monkeypatch)
    monkeypatch.setattr(ollama, "start_server",
                        lambda *a, **k: {"ok": False, "reason": "no binary"})

    result = assets.provision_all(cfg)

    assert result["ok"] is True
    assert "server" not in result["failed"]


def test_provision_all_can_be_told_not_to_spawn_a_daemon(cfg, monkeypatch):
    """Something else may own the daemon's lifetime — or it may be a test."""
    started: List[int] = []
    _all_provision_steps_ok(monkeypatch)
    monkeypatch.setattr(ollama, "start_server",
                        lambda *a, **k: (started.append(1) or {"ok": True, "reason": "x"}))

    assets.provision_all(cfg, start_server=False)

    assert started == []


def test_provision_all_passes_refresh_down_to_the_model(cfg, monkeypatch):
    seen: Dict[str, Any] = {}
    _all_provision_steps_ok(monkeypatch)
    monkeypatch.setattr(ollama, "start_server", lambda *a, **k: {"ok": True, "reason": "x"})
    monkeypatch.setattr(
        ollama, "ensure_model",
        lambda *a, **k: (seen.update(k) or {"ok": True, "reason": "x"}))

    assets.provision_all(cfg, refresh=True)

    assert seen.get("refresh") is True


# --------------------------------------------------------------------------- #
#  The systemd unit is someone's hand-tuned file until proven otherwise
# --------------------------------------------------------------------------- #
def test_install_service_refuses_to_overwrite_a_unit_it_did_not_write(cfg):
    """It compared content and overwrote on any difference, with no ownership
    check — unlike link_binary, which has one. A systemd unit is exactly the
    kind of file someone tunes once and forgets about.
    """
    path = ollama.unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = "[Unit]\nDescription=my own careful configuration\n"
    path.write_text(mine, encoding="utf-8")

    result = ollama.install_service(cfg, enable=False)

    assert result["ok"] is False
    assert result.get("action") == "conflict"
    assert path.read_text(encoding="utf-8") == mine, "the owner's unit was overwritten"
    assert "not written by JARVIS" in result["reason"]


def test_install_service_rewrites_its_own_unit_freely(cfg):
    """The refusal must not amount to the installer never updating anything."""
    path = ollama.unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ollama.UNIT_MARKER + "\nstale content\n", encoding="utf-8")

    result = ollama.install_service(cfg, enable=False)

    assert result["ok"] is True
    assert result["written"] is True
    assert "stale content" not in path.read_text(encoding="utf-8")


def test_the_generated_unit_carries_the_ownership_marker(cfg):
    """If the marker ever drifts, every re-run becomes a refusal instead."""
    assert ollama.service_unit_text(cfg).lstrip().startswith(ollama.UNIT_MARKER)


# --------------------------------------------------------------------------- #
#  Reporting the version of the binary that will actually run
# --------------------------------------------------------------------------- #
def test_an_external_ollama_wins_even_when_a_managed_tree_exists(cfg, monkeypatch):
    """The external branch was gated on "no managed tree", which is the one
    case where the report went wrong.

    With both present, PATH resolves the external binary — so that is what runs
    — but the plan fell through to the release comparison and reported the
    EXTERNAL binary's version as the state of the MANAGED install: "already the
    latest release", while a stale tree sat in the data directory, unused and
    never upgraded.
    """
    external = Path(str(cfg.home())).parent / "usr_bin" / "ollama"
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_text("#!/bin/sh\n", encoding="utf-8")

    # A managed tree is identified by its marker file, not merely by a binary
    # sitting in the expected place.
    runtime_root = ollama.runtime_dir(cfg)
    (runtime_root / "bin").mkdir(parents=True, exist_ok=True)
    (runtime_root / "bin" / ollama.BIN_NAME).write_text("#!/bin/sh\n", encoding="utf-8")
    (runtime_root / ollama.MANAGED_MARKER).write_text("0.5.0\n", encoding="utf-8")
    assert ollama.is_managed(cfg), "test setup failed to create a managed tree"

    monkeypatch.setattr(ollama, "is_installed", lambda *a, **k: external)
    monkeypatch.setattr(ollama, "installed_version", lambda *a, **k: "0.32.6")
    monkeypatch.setattr(ollama, "_is_external", lambda binary, c=None: True)

    plan = ollama.install_plan(cfg)

    assert plan["action"] == "external", (
        f"reported {plan['action']!r}: the managed tree suppressed the external "
        f"branch, so the wrong binary's version was reported as current"
    )
    assert "0.32.6" in plan["reason"]
    assert "unused" in plan["reason"], (
        "the orphaned managed tree should be mentioned — it is dead weight the "
        "owner cannot otherwise know about"
    )
