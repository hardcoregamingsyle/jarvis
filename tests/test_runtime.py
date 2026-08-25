"""Provisioning: rootless Ollama, the model, and the speech assets.

Nothing here touches the network, spawns a daemon, binds a socket or downloads
a byte.  The outside world is replaced at three seams and no others:

* ``ollama._urlopen``  — every HTTP call in the module goes through it
* ``ollama._which`` / ``assets._which`` — the PATH lookup
* ``shutil.disk_usage`` — free space

Home directories are redirected into ``tmp_path`` by the ``fake_home`` fixture
before any of these functions are called; the real ``~/.local/bin`` and the
real ``~/.ollama`` are never named as a target, let alone written to.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jarvis.core.config import LLMConfig, STTConfig, load_config
from jarvis.runtime import assets, ollama
from jarvis.runtime import status as runtime_status


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every notion of "home" into tmp_path.

    ``~`` is resolved through ``os.path.expanduser``, which reads ``HOME`` on
    POSIX and ``USERPROFILE`` on Windows; both are set so the suite behaves
    identically on the development box and the target laptop.  The XDG
    variables are set too, because that is what the module actually prefers.
    """
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    (home / ".config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.delenv("XDG_BIN_HOME", raising=False)
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return home


@pytest.fixture
def cfg(isolated_home: Path, fake_home: Path):
    """A default config rooted in the isolated JARVIS home."""
    configuration = load_config(use_env=False)
    configuration.data_dir = str(isolated_home)
    return configuration


@pytest.fixture(autouse=True)
def no_real_binary(monkeypatch: pytest.MonkeyPatch):
    """Nothing is on PATH unless a test says so."""
    monkeypatch.setattr(ollama, "_which", lambda name: None)
    monkeypatch.setattr(assets, "_which", lambda name: None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch):
    """Any un-stubbed HTTP call is a test bug, not a slow test.

    Both seams are closed. ``ollama._urlopen`` is the module's own entry point
    and is what individual tests replace with a stub; ``urllib.request.urlopen``
    is closed underneath it so that a path this file does not know about — the
    Piper downloader inside :mod:`jarvis.speech.tts`, say — cannot quietly
    reach Hugging Face from the test-suite.
    """
    def explode(request: Any, *args: Any, **kwargs: Any):
        url = getattr(request, "full_url", request)
        raise AssertionError(f"unstubbed network call to {url}")

    import urllib.request

    monkeypatch.setattr(ollama, "_urlopen", explode)
    monkeypatch.setattr(urllib.request, "urlopen", explode)


@pytest.fixture(autouse=True)
def no_faster_whisper(monkeypatch: pytest.MonkeyPatch):
    """``faster_whisper`` is absent unless a test installs a fake.

    It is genuinely installed on the development machine, so without this an
    ``ensure_whisper_model`` test would construct a real ``WhisperModel`` — and
    on a machine with a cold cache that is a 150 MB download from the test
    suite. ``None`` in ``sys.modules`` is what makes ``import`` raise
    ImportError, which is the state this code has to handle anyway.
    """
    import sys

    monkeypatch.setitem(sys.modules, "faster_whisper", None)


class FakeResponse:
    """Minimal ``urlopen`` return value: a context manager with ``read``."""

    def __init__(self, body: Any = b"", *, status: int = 200,
                 headers: Optional[Dict[str, str]] = None,
                 chunks: Optional[List[bytes]] = None) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._chunks = list(chunks) if chunks is not None else ([body] if body else [])
        self.status = status
        self.headers = dict(headers or {})

    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def clone(self) -> "FakeResponse":
        """A fresh, unread copy — a response body can only be consumed once.

        ``type(self)`` rather than ``FakeResponse`` so that a subclass which
        overrides ``read`` to fail keeps failing after being cloned.
        """
        return type(self)(status=self.status, headers=dict(self.headers),
                          chunks=list(self._chunks))

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def route(monkeypatch: pytest.MonkeyPatch, table: Dict[str, Any]) -> List[str]:
    """Serve ``_urlopen`` from a {url-substring: response-or-callable} table.

    A :class:`FakeResponse` in the table is cloned per request rather than
    handed out directly: a response body reads once, so a shared instance would
    silently return empty on a second call and a test asserting idempotence
    would pass for the wrong reason.

    Returns a list that records every URL requested, so a test can assert that
    a code path stayed offline.
    """
    seen: List[str] = []

    def fake(request: Any, timeout: float = 0.0):
        url = getattr(request, "full_url", str(request))
        seen.append(url)
        for needle, value in table.items():
            if needle in url:
                if callable(value):
                    return value(request)
                return value.clone()
        raise AssertionError(f"no stub for {url}")

    monkeypatch.setattr(ollama, "_urlopen", fake)
    return seen


def release_payload(version: str = "0.5.7", size: int = 1_000_000,
                    arch: str = "amd64", suffix: str = ".tgz",
                    extra: Any = None) -> bytes:
    """A GitHub "latest release" body carrying one asset (plus any extras)."""
    listed = [{
        "name": f"ollama-linux-{arch}{suffix}",
        "size": size,
        "browser_download_url":
            f"https://github.com/ollama/ollama/releases/download/v{version}/"
            f"ollama-linux-{arch}{suffix}",
    }]
    listed.extend(extra or [])
    return json.dumps({"tag_name": f"v{version}", "assets": listed}).encode("utf-8")


def make_tarball(entries: Dict[str, bytes], mode: str = "w:gz") -> bytes:
    """An in-memory archive, so extraction is exercised without a fixture file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class FakeUsage:
    """The (total, used, free) triple ``shutil.disk_usage`` returns."""

    def __init__(self, total: int, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


def fake_disk(monkeypatch: pytest.MonkeyPatch, free: int, total: int = 500_000_000_000):
    """Report a fixed amount of free space on every filesystem."""
    import shutil as shutil_module

    monkeypatch.setattr(
        shutil_module, "disk_usage",
        lambda path: FakeUsage(total, total - free, free),
    )


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #
def test_is_installed_finds_a_binary_on_path(cfg, tmp_path, monkeypatch):
    binary = tmp_path / "usr-bin" / "ollama"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(ollama, "_which", lambda name: str(binary) if name == "ollama" else None)

    assert ollama.is_installed(cfg) == binary


def test_is_installed_finds_a_binary_in_local_bin(cfg, fake_home):
    """~/.local/bin is not always on PATH in a non-interactive shell."""
    binary = fake_home / ".local" / "bin" / "ollama"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    assert ollama.is_installed(cfg) == binary


def test_is_installed_finds_the_managed_binary(cfg):
    binary = ollama.runtime_dir(cfg) / "bin" / "ollama"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    assert ollama.is_installed(cfg) == binary


def test_is_installed_returns_none_when_nothing_is_there(cfg):
    assert ollama.is_installed(cfg) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ollama version is 0.5.7", "0.5.7"),
        ("ollama version 0.12.3", "0.12.3"),
        ("0.9.1", "0.9.1"),
        ("v0.9.1", "0.9.1"),
        ("Warning: client version is 0.4.0-rc1", "0.4.0-rc1"),
        # The version is not necessarily on the first line.
        ("Warning: could not connect to a running Ollama instance\n"
         "ollama version is 0.6.2", "0.6.2"),
    ],
)
def test_parse_version_reads_real_output(text, expected):
    assert ollama.parse_version(text) == expected


@pytest.mark.parametrize(
    "garbage",
    ["", "   ", None, "command not found", "Segmentation fault (core dumped)",
     "bash: ollama: No such file or directory", "3 files, 1.5 MB"],
)
def test_parse_version_returns_none_for_garbage(garbage):
    """A decimal number in unrelated text must not be mistaken for a release."""
    assert ollama.parse_version(garbage) is None


def test_installed_version_survives_a_binary_that_fails(cfg, tmp_path, monkeypatch):
    binary = tmp_path / "ollama"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(ollama, "_which", lambda name: str(binary))

    from jarvis.core.platform_utils import CommandResult
    monkeypatch.setattr(
        ollama.platform_utils, "run_command",
        lambda *a, **k: CommandResult(127, "", "not executable"),
    )
    assert ollama.installed_version(cfg) is None


def test_installed_version_reads_a_client_that_exits_nonzero(cfg, tmp_path, monkeypatch):
    """ollama exits non-zero when no server is running but still prints its version."""
    binary = tmp_path / "ollama"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(ollama, "_which", lambda name: str(binary))

    from jarvis.core.platform_utils import CommandResult
    monkeypatch.setattr(
        ollama.platform_utils, "run_command",
        lambda *a, **k: CommandResult(1, "ollama version is 0.7.0\n",
                                      "could not connect to ollama app"),
    )
    assert ollama.installed_version(cfg) == "0.7.0"


def test_target_arch_maps_the_machines_ollama_ships_for():
    assert ollama.target_arch("x86_64") == "amd64"
    assert ollama.target_arch("AMD64") == "amd64"
    assert ollama.target_arch("aarch64") == "arm64"
    assert ollama.target_arch("arm64") == "arm64"
    assert ollama.target_arch("riscv64") is None


# --------------------------------------------------------------------------- #
#  Releases
# --------------------------------------------------------------------------- #
def test_latest_release_parses_the_github_tag(monkeypatch):
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("0.5.7"))})
    assert ollama.latest_release() == "0.5.7"


def test_latest_release_is_none_when_the_api_errors(monkeypatch):
    import urllib.error

    def boom(request, timeout=0.0):
        raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(ollama, "_urlopen", boom)
    assert ollama.latest_release() is None


def test_latest_release_is_none_when_the_api_times_out(monkeypatch):
    import socket
    import urllib.error

    def hang(request, timeout=0.0):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(ollama, "_urlopen", hang)
    assert ollama.latest_release() is None


def test_latest_release_is_none_for_malformed_json(monkeypatch):
    route(monkeypatch, {"api.github.com": FakeResponse(b"<html>not json</html>")})
    assert ollama.latest_release() is None


def test_release_asset_url_uses_the_current_naming():
    """Ollama moved Linux releases from .tgz to zstd-compressed tar."""
    url = ollama.release_asset_url("0.32.6", "amd64")
    assert url == (
        "https://github.com/ollama/ollama/releases/download/v0.32.6/"
        "ollama-linux-amd64.tar.zst"
    )
    assert ollama.release_asset_url("v0.32.6", "arm64").endswith("ollama-linux-arm64.tar.zst")
    assert ollama.release_asset_url("0.5.7", "amd64", ".tgz").endswith("ollama-linux-amd64.tgz")


def test_pick_asset_ignores_the_gpu_and_jetson_variants():
    """The variant builds sit next to the plain one and are wrong for a CPU laptop.

    The variants are listed **before** the plain asset, which is the order
    GitHub actually returns: ``-`` (0x2D) sorts before ``.`` (0x2E), so
    ``ollama-linux-amd64-rocm.tar.zst`` comes first.  With the plain asset first
    a prefix match would pass this test while shipping a gigabyte of ROCm to a
    machine with no AMD GPU, so the order here is the whole point.
    """
    release = {
        "tag_name": "v0.32.6",
        "assets": [
            {"name": "ollama-linux-amd64-mlx.tar.zst", "size": 220_000_000},
            {"name": "ollama-linux-amd64-rocm.tar.zst", "size": 1_050_000_000},
            {"name": "ollama-linux-arm64-jetpack6.tar.zst", "size": 270_000_000},
            {"name": "ollama-darwin.tgz", "size": 150_000_000},
            {
                "name": "ollama-linux-amd64.tar.zst",
                "size": 1_420_000_000,
                "browser_download_url":
                    "https://github.com/ollama/ollama/releases/download/v0.32.6/"
                    "ollama-linux-amd64.tar.zst",
            },
        ],
    }

    asset = ollama._pick_asset(release, "amd64", "0.32.6")

    assert asset["name"] == "ollama-linux-amd64.tar.zst"
    assert asset["size"] == 1_420_000_000
    assert "rocm" not in asset["url"] and "mlx" not in asset["url"]


def test_pick_asset_prefers_a_format_this_machine_can_unpack(monkeypatch):
    """Without zstd, a .tgz alongside it is the one to take."""
    release = json.loads(release_payload(
        "1.0.0", 1_400_000_000, suffix=".tar.zst",
        extra=[{"name": "ollama-linux-amd64.tgz", "size": 900_000_000}],
    ).decode("utf-8"))

    monkeypatch.setattr(ollama, "zstd_support", lambda: None)
    asset = ollama._pick_asset(release, "amd64", "1.0.0")

    assert asset["name"] == "ollama-linux-amd64.tgz"
    assert asset["extractable"] is True


def test_pick_asset_falls_back_to_the_convention_when_offline():
    asset = ollama._pick_asset(None, "arm64", "0.32.6")
    assert asset["name"] == "ollama-linux-arm64.tar.zst"
    assert asset["size"] == 0


def test_install_plan_refuses_zst_when_this_python_cannot_read_it(cfg, monkeypatch):
    """Python 3.9-3.13 has no zstd; say so, and name every fix that needs no root."""
    route(monkeypatch, {"api.github.com": FakeResponse(
        release_payload("0.32.6", 1_420_000_000, suffix=".tar.zst"))})
    monkeypatch.setattr(ollama, "zstd_support", lambda: None)
    fake_disk(monkeypatch, free=400_000_000_000)

    plan = ollama.install_plan(cfg)

    assert plan["ok"] is False
    assert plan["extractable"] is False
    assert "pip install zstandard" in plan["reason"]
    assert "never uses sudo" in plan["reason"]


def test_install_stops_before_downloading_what_it_cannot_unpack(cfg, monkeypatch):
    seen = route(monkeypatch, {"api.github.com": FakeResponse(
        release_payload("0.32.6", 1_420_000_000, suffix=".tar.zst"))})
    monkeypatch.setattr(ollama, "zstd_support", lambda: None)
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is False
    assert not any("releases/download" in url for url in seen), (
        "1.4 GB must not be downloaded when it cannot be unpacked"
    )


# --------------------------------------------------------------------------- #
#  install_plan
# --------------------------------------------------------------------------- #
def _install_managed(cfg, version: str = "0.5.7") -> Path:
    """Put a managed install in place, exactly as install() leaves it."""
    root = ollama.runtime_dir(cfg)
    binary = root / "bin" / "ollama"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    (root / ollama.MANAGED_MARKER).write_text(f"ollama {version}\n", encoding="utf-8")
    return binary


def _report_version(monkeypatch, version: str) -> None:
    from jarvis.core.platform_utils import CommandResult
    monkeypatch.setattr(
        ollama.platform_utils, "run_command",
        lambda *a, **k: CommandResult(0, f"ollama version is {version}\n", ""),
    )


def test_install_plan_says_already_current_when_versions_match(cfg, monkeypatch):
    _install_managed(cfg, "0.5.7")
    _report_version(monkeypatch, "0.5.7")
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("0.5.7"))})

    plan = ollama.install_plan(cfg)
    assert plan["action"] == "current"
    assert plan["ok"] is True
    assert plan["installed"] == "0.5.7"
    assert plan["latest"] == "0.5.7"


def test_install_plan_says_upgrade_when_versions_differ(cfg, monkeypatch):
    _install_managed(cfg, "0.5.0")
    _report_version(monkeypatch, "0.5.0")
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("0.9.9"))})

    plan = ollama.install_plan(cfg)
    assert plan["action"] == "upgrade"
    assert plan["installed"] == "0.5.0"
    assert plan["latest"] == "0.9.9"
    assert plan["url"].endswith("v0.9.9/ollama-linux-amd64.tgz")


def test_install_plan_says_install_when_nothing_is_present(cfg, monkeypatch):
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("0.9.9"))})

    plan = ollama.install_plan(cfg)
    assert plan["action"] == "install"
    assert plan["installed"] is None
    assert plan["size_bytes"] == 1_000_000


def test_install_plan_defers_to_a_system_wide_ollama(cfg, tmp_path, monkeypatch):
    """A distro-packaged ollama is used, not duplicated -- and no API call is made."""
    system = tmp_path / "usr" / "local" / "bin" / "ollama"
    system.parent.mkdir(parents=True)
    system.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(ollama, "_which", lambda name: str(system))
    _report_version(monkeypatch, "0.4.2")
    seen = route(monkeypatch, {"api.github.com": FakeResponse(release_payload())})

    plan = ollama.install_plan(cfg)
    assert plan["action"] == "external"
    assert plan["ok"] is True
    assert str(system) in plan["reason"]
    assert seen == [], "an existing install must not trigger a release lookup"


def test_install_plan_reports_unsupported_architecture(cfg, monkeypatch):
    monkeypatch.setattr(ollama.platform, "machine", lambda: "riscv64")

    plan = ollama.install_plan(cfg)
    assert plan["action"] == "unsupported"
    assert plan["ok"] is False
    assert "riscv64" in plan["reason"]


def test_install_plan_is_honest_about_being_offline(cfg, monkeypatch):
    import urllib.error

    def offline(request, timeout=0.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(ollama, "_urlopen", offline)
    plan = ollama.install_plan(cfg)
    assert plan["action"] == "unknown"
    assert plan["ok"] is False
    assert "github.com" in plan["reason"]


def test_install_plan_reports_free_space_at_the_target(cfg, monkeypatch):
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("1.0.0", 2_000_000_000))})
    fake_disk(monkeypatch, free=400_000_000_000)

    plan = ollama.install_plan(cfg)
    assert plan["target_dir"] == str(ollama.runtime_dir(cfg))
    assert plan["free_gb"] == 400.0
    assert plan["enough_space"] is True
    assert plan["required_bytes"] > plan["size_bytes"], (
        "the archive and the unpacked tree both need room"
    )


# --------------------------------------------------------------------------- #
#  install
# --------------------------------------------------------------------------- #
def test_install_refuses_when_free_space_is_short_and_names_the_shortfall(cfg, monkeypatch):
    route(monkeypatch, {"api.github.com": FakeResponse(release_payload("1.0.0", 2_000_000_000))})
    fake_disk(monkeypatch, free=1_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is False
    assert result["shortfall_gb"] > 0
    assert f"{result['shortfall_gb']} GB short" in result["reason"]
    assert "1.0 GB free" in result["reason"]
    assert not ollama.runtime_dir(cfg).exists(), "a refusal must not create the target"


def test_install_unpacks_links_and_marks_the_tree(cfg, fake_home, monkeypatch):
    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\necho ollama\n",
                            "lib/ollama/libggml.so": b"\x00\x01"})
    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(tarball))),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    events: List[Dict[str, Any]] = []
    result = ollama.install(cfg, progress=events.append)

    assert result["ok"] is True, result["reason"]
    assert result["installed_now"] is True
    assert result["installed"] == "0.9.9"

    binary = ollama.runtime_dir(cfg) / "bin" / "ollama"
    assert binary.is_file()
    assert (ollama.runtime_dir(cfg) / ollama.MANAGED_MARKER).is_file()
    assert ollama.is_managed(cfg) is True
    assert result["link"]["ok"] is True
    assert Path(result["link"]["path"]) == fake_home / ".local" / "bin" / "ollama"
    assert {e["phase"] for e in events} >= {"download", "extract"}


@pytest.mark.skipif(not ollama._stdlib_zstd(),
                    reason="this interpreter cannot write .tar.zst")
def test_install_unpacks_a_zstd_archive(cfg, monkeypatch):
    """The format ollama actually publishes for Linux today."""
    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"}, mode="w:zst")
    route(monkeypatch, {
        "api.github.com": FakeResponse(
            release_payload("0.32.6", len(tarball), suffix=".tar.zst")),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is True, result["reason"]
    assert result["asset"] == "ollama-linux-amd64.tar.zst"
    assert (ollama.runtime_dir(cfg) / "bin" / "ollama").is_file()


@pytest.mark.skipif(not ollama._stdlib_zstd(),
                    reason="this interpreter cannot write .tar.zst")
def test_zstd_is_decompressed_through_a_temporary_tar_on_older_pythons(cfg, monkeypatch):
    """On Python 3.9-3.13 tarfile cannot read zstd, so the CLI is shelled out to.

    The zstd binary is simulated with this interpreter's own decompressor, which
    exercises the real intermediate-file path: create it, extract from it, and
    -- the part worth asserting -- delete it afterwards.
    """
    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"}, mode="w:zst")
    route(monkeypatch, {
        "api.github.com": FakeResponse(
            release_payload("0.32.6", len(tarball), suffix=".tar.zst")),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    monkeypatch.setattr(ollama, "_stdlib_zstd", lambda: False)
    monkeypatch.setattr(ollama, "zstd_support", lambda: "cli")
    monkeypatch.setattr(ollama, "_which",
                        lambda name: "/usr/bin/zstd" if name == "zstd" else None)

    from compression import zstd as stdlib_zstd
    from jarvis.core.platform_utils import CommandResult

    invocations: List[List[str]] = []

    def fake_zstd(argv, **kwargs):
        invocations.append(list(argv))
        out = Path(argv[argv.index("-o") + 1])
        src = Path(argv[-1])
        out.write_bytes(stdlib_zstd.decompress(src.read_bytes()))
        return CommandResult(0, "", "")

    monkeypatch.setattr(ollama.platform_utils, "run_command", fake_zstd)

    result = ollama.install(cfg)

    assert result["ok"] is True, result["reason"]
    assert invocations, "the zstd command should have been used"
    assert (ollama.runtime_dir(cfg) / "bin" / "ollama").is_file()
    leftovers = list(ollama.download_dir(cfg).glob("*.tar"))
    assert leftovers == [], f"the intermediate tar was not cleaned up: {leftovers}"


def test_extract_reports_missing_zstd_support_with_every_rootless_fix(cfg, tmp_path,
                                                                     monkeypatch):
    archive = tmp_path / "ollama.tar.zst"
    archive.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 32)
    monkeypatch.setattr(ollama, "_stdlib_zstd", lambda: False)
    monkeypatch.setattr(ollama, "zstd_support", lambda: None)

    result = ollama._extract(archive, tmp_path / "staged")

    assert result["ok"] is False
    assert "pip install zstandard" in result["reason"]
    assert not (tmp_path / "staged").exists()


def test_install_never_overwrites_a_foreign_local_bin_ollama(cfg, fake_home, monkeypatch):
    """Somebody else's ~/.local/bin/ollama is left exactly as it was."""
    foreign = fake_home / ".local" / "bin" / "ollama"
    foreign.write_text("#!/bin/sh\n# hand-written by the owner\n", encoding="utf-8")
    before = foreign.read_text(encoding="utf-8")
    # is_installed() will find it, so keep the suite from trying to execute it.
    _report_version(monkeypatch, "0.1.0")

    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"})
    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(tarball))),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert foreign.read_text(encoding="utf-8") == before
    assert not foreign.is_symlink()
    assert result["link"]["mode"] == "conflict"
    assert "not created by JARVIS" in result["link"]["reason"]
    # The runtime itself still installed; only the convenience link was skipped.
    assert result["ok"] is True
    assert str(foreign) in result["reason"]


def test_a_failed_extraction_leaves_no_partial_directory(cfg, monkeypatch):
    corrupt = b"this is not a gzip stream at all"
    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(corrupt))),
        "releases/download": FakeResponse(corrupt),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is False
    assert "could not extract" in result["reason"]
    target = ollama.runtime_dir(cfg)
    assert not target.exists()
    leftovers = list(target.parent.glob(f"{target.name}.new-*"))
    assert leftovers == [], f"a staging directory survived a failure: {leftovers}"


def test_a_corrupt_archive_is_deleted_so_the_retry_is_not_poisoned(cfg, monkeypatch):
    """A corrupt archive of the *right length* would resume to nothing for ever."""
    corrupt = b"this is not a gzip stream at all"
    attempts: List[str] = []

    def serve(request):
        attempts.append(getattr(request, "full_url", ""))
        return FakeResponse(corrupt)

    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(corrupt))),
        "releases/download": serve,
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    first = ollama.install(cfg)
    assert first["ok"] is False
    assert list(ollama.download_dir(cfg).glob("*.tgz")) == []
    assert "deleted so a retry re-fetches it" in first["reason"]

    # The retry must actually go back to the network rather than resume a file
    # that can never extract.
    ollama.install(cfg)
    assert len(attempts) == 2


def test_install_refuses_to_replace_an_unmanaged_directory(cfg, monkeypatch):
    """A tree without our marker was put there by somebody else."""
    target = ollama.runtime_dir(cfg)
    target.mkdir(parents=True)
    (target / "precious.txt").write_text("do not delete", encoding="utf-8")

    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"})
    seen = route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(tarball))),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is False
    assert result["action"] == "conflict"
    assert (target / "precious.txt").read_text(encoding="utf-8") == "do not delete"
    # Whether the target is somebody else's is knowable from the filesystem
    # alone, so the refusal must come *before* the gigabyte, not after it.
    assert not any("releases/download" in url for url in seen), (
        "a refusal that is decidable offline must not download the archive first"
    )
    assert list(ollama.download_dir(cfg).glob("*")) == []


def test_a_successful_install_does_not_leave_the_tarball_behind(cfg, monkeypatch):
    """1.4 GB per release accumulating in <data>/downloads is a slow disk leak."""
    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"})
    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(tarball))),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is True, result["reason"]
    leftovers = list(ollama.download_dir(cfg).glob("*"))
    assert leftovers == [], f"the archive was never cleaned up: {leftovers}"
    # The binary it was unpacked from is what has to survive, not the archive.
    assert (ollama.runtime_dir(cfg) / "bin" / "ollama").is_file()


def test_install_is_a_no_op_when_already_current(cfg, monkeypatch):
    _install_managed(cfg, "0.9.9")
    _report_version(monkeypatch, "0.9.9")
    seen = route(monkeypatch, {"api.github.com": FakeResponse(release_payload("0.9.9"))})

    result = ollama.install(cfg)

    assert result["ok"] is True
    assert result["action"] == "current"
    assert result["installed_now"] is False
    assert not any("releases/download" in url for url in seen), "nothing may be downloaded"


def test_install_rejects_a_truncated_download_and_clears_the_partial(cfg, monkeypatch):
    """A wrong-sized archive must not be left behind to poison the retry."""
    tarball = make_tarball({"bin/ollama": b"#!/bin/sh\n"})
    route(monkeypatch, {
        "api.github.com": FakeResponse(release_payload("0.9.9", len(tarball) + 5000)),
        "releases/download": FakeResponse(tarball),
    })
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.install(cfg)

    assert result["ok"] is False
    assert "bytes" in result["reason"]
    assert list(ollama.download_dir(cfg).glob("*.tgz")) == []


def test_download_resumes_from_a_partial_file(cfg, monkeypatch):
    dest = ollama.download_dir(cfg) / "ollama.tgz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"AAAA")

    captured: Dict[str, Any] = {}

    def serve(request):
        captured["range"] = request.headers.get("Range")
        return FakeResponse(b"BBBB", status=206)

    route(monkeypatch, {"example": serve})

    result = ollama._download("https://example/ollama.tgz", dest, expected_size=8)

    assert captured["range"] == "bytes=4-"
    assert result["ok"] is True
    assert result["resumed"] is True
    assert dest.read_bytes() == b"AAAABBBB"


def test_download_skips_a_file_that_is_already_complete(cfg, monkeypatch):
    dest = ollama.download_dir(cfg) / "ollama.tgz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"12345678")
    seen = route(monkeypatch, {})

    result = ollama._download("https://example/ollama.tgz", dest, expected_size=8)

    assert result["ok"] is True
    assert seen == [], "a complete download must not be fetched again"


def test_link_binary_replaces_its_own_symlink(cfg, fake_home, tmp_path):
    """Re-running the installer updates the link it made last time."""
    old = ollama.runtime_dir(cfg) / "bin" / "ollama"
    old.parent.mkdir(parents=True)
    old.write_text("old", encoding="utf-8")
    link = fake_home / ".local" / "bin" / "ollama"

    first = ollama.link_binary(old, cfg=cfg)
    if first["mode"] != "symlink":
        pytest.skip("this filesystem does not support symlinks")

    second = ollama.link_binary(old, cfg=cfg)
    assert second["ok"] is True
    assert link.is_symlink()


# --------------------------------------------------------------------------- #
#  The systemd user unit
# --------------------------------------------------------------------------- #
def test_service_unit_sets_the_concurrency_ollama_needs_for_a_subagent_tree(cfg):
    cfg.llm.max_concurrent_requests = 6
    text = ollama.service_unit_text(cfg, binary=Path("/home/o/.local/bin/ollama"))

    assert "Environment=OLLAMA_NUM_PARALLEL=6" in text
    assert "Environment=OLLAMA_MAX_LOADED_MODELS=" in text
    assert 'ExecStart="/home/o/.local/bin/ollama" serve' in text


def ini_sections(text: str) -> Dict[str, List[str]]:
    """Split a unit file into its sections, the way systemd reads it.

    Line-based on purpose: the unit carries comments that mention ``[Unit]``
    and ``[Service]`` in prose, and a naive ``str.split`` would cut the file at
    a comment rather than at a header.
    """
    sections: Dict[str, List[str]] = {}
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections.setdefault(current, [])
        elif current and not stripped.startswith("#"):
            sections.setdefault(current, []).append(stripped)
    return sections


def test_service_unit_puts_startlimit_in_the_unit_section(cfg):
    """systemd ignores StartLimit* under [Service] without a word of warning."""
    sections = ini_sections(ollama.service_unit_text(cfg, binary=Path("/usr/bin/ollama")))

    assert "StartLimitIntervalSec=120" in sections["Unit"]
    assert "StartLimitBurst=5" in sections["Unit"]
    assert not any(line.startswith("StartLimit") for line in sections["Service"])


def test_service_unit_is_a_user_unit(cfg):
    text = ollama.service_unit_text(cfg, binary=Path("/usr/bin/ollama"))
    assert "WantedBy=default.target" in text
    assert "User=" not in text, "a user unit must not name a User"


def test_unlimited_concurrency_becomes_a_real_number(cfg):
    """OLLAMA_NUM_PARALLEL=0 would mean 'let ollama decide', not 'unlimited'."""
    cfg.llm.max_concurrent_requests = 0
    text = ollama.service_unit_text(cfg, binary=Path("/usr/bin/ollama"))
    assert "Environment=OLLAMA_NUM_PARALLEL=8" in text


def test_install_service_writes_into_the_xdg_user_unit_dir(cfg, fake_home):
    result = ollama.install_service(cfg, enable=False)

    written = fake_home / ".config" / "systemd" / "user" / ollama.UNIT_NAME
    assert result["ok"] is True
    assert Path(result["path"]) == written
    assert written.is_file()
    assert "OLLAMA_NUM_PARALLEL" in written.read_text(encoding="utf-8")

    # Second run: same content, nothing rewritten.
    again = ollama.install_service(cfg, enable=False)
    assert again["ok"] is True
    assert again["written"] is False


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #
def tags_response(*names: str, size: int = 16_100_000_000) -> FakeResponse:
    return FakeResponse(json.dumps({
        "models": [{"name": name, "size": size} for name in names]
    }).encode("utf-8"))


def test_model_tag_comes_from_the_config(cfg):
    assert ollama.model_tag(cfg) == LLMConfig().ollama_model
    cfg.llm.ollama_model = "qwen3:4b-instruct-2507-q4_K_M"
    assert ollama.model_tag(cfg) == "qwen3:4b-instruct-2507-q4_K_M"


def test_normalise_tag_makes_ollamas_implicit_latest_explicit():
    assert ollama.normalise_tag("qwen3.6") == "qwen3.6:latest"
    assert ollama.normalise_tag("qwen3.6:27b") == "qwen3.6:27b"
    assert ollama.normalise_tag("") == ""


def test_pull_plan_says_present_for_a_tag_already_in_api_tags(cfg, monkeypatch):
    route(monkeypatch, {"/api/tags": lambda r: tags_response("qwen3.6:27b")})
    fake_disk(monkeypatch, free=400_000_000_000)

    plan = ollama.pull_plan("qwen3.6:27b", cfg=cfg)

    assert plan["action"] == "present"
    assert plan["present"] is True
    assert plan["ok"] is True
    assert plan["present_size_bytes"] == 16_100_000_000


def test_pull_plan_reports_the_size_before_downloading(cfg, monkeypatch):
    route(monkeypatch, {"/api/tags": lambda r: tags_response("qwen3:4b")})
    fake_disk(monkeypatch, free=400_000_000_000)

    plan = ollama.pull_plan("qwen3.6:27b", cfg=cfg)

    assert plan["present"] is False
    assert plan["action"] == "pull"
    # 16.1 GB from the catalogue, quoted before a byte moves.
    assert plan["size_gb"] == pytest.approx(16.1, abs=0.2)
    assert "16.1 GB" in plan["reason"]


def test_pull_plan_refuses_when_the_model_will_not_fit(cfg, monkeypatch):
    route(monkeypatch, {"/api/tags": lambda r: tags_response()})
    fake_disk(monkeypatch, free=5_000_000_000)

    plan = ollama.pull_plan("qwen3.6:27b", cfg=cfg)

    assert plan["ok"] is False
    assert plan["shortfall_gb"] > 0
    assert "GB short" in plan["reason"]
    assert "5.0 GB free" in plan["reason"]


def test_pull_plan_reports_a_missing_server_rather_than_an_empty_catalogue(cfg, monkeypatch):
    import urllib.error

    def refused(request, timeout=0.0):
        raise urllib.error.URLError(ConnectionRefusedError("connection refused"))

    monkeypatch.setattr(ollama, "_urlopen", refused)
    fake_disk(monkeypatch, free=400_000_000_000)

    plan = ollama.pull_plan("qwen3.6:27b", cfg=cfg)

    assert plan["action"] == "no-server"
    assert plan["ok"] is False
    assert plan["server"] is False
    assert "ollama serve" in plan["reason"]


def test_list_models_is_empty_when_the_server_is_down(cfg, monkeypatch):
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    assert ollama.list_models(cfg) == []


# --------------------------------------------------------------------------- #
#  NDJSON streaming
# --------------------------------------------------------------------------- #
def test_ndjson_reassembles_a_line_torn_across_chunks():
    stream = ollama.NDJSONStream()

    assert stream.feed(b'{"status":"pulling ') == []
    assert stream.feed(b'manifest"}\n{"stat') == [{"status": "pulling manifest"}]
    assert stream.feed(b'us":"done"}\n') == [{"status": "done"}]
    assert stream.flush() == []


def test_ndjson_holds_back_a_trailing_line_until_flush():
    stream = ollama.NDJSONStream()
    assert stream.feed(b'{"status":"success"}') == []
    assert stream.flush() == [{"status": "success"}]


def test_ndjson_survives_a_split_multibyte_character():
    """A chunk boundary inside a UTF-8 sequence must not corrupt the text."""
    # ensure_ascii=False, or json escapes the character to é and there is
    # no multi-byte sequence left to tear.
    payload = json.dumps({"status": "pulling éclair"}, ensure_ascii=False).encode("utf-8")
    payload += b"\n"
    split = payload.index(b"\xc3") + 1
    stream = ollama.NDJSONStream()

    first = stream.feed(payload[:split])
    second = stream.feed(payload[split:])

    assert first == []
    assert second == [{"status": "pulling éclair"}]


def test_ndjson_ignores_unparsable_lines():
    stream = ollama.NDJSONStream()
    assert stream.feed(b'not json\n{"status":"ok"}\n') == [{"status": "ok"}]


def test_pull_reports_progress_across_torn_chunks(cfg, monkeypatch):
    chunks = [
        b'{"status":"pulling manifest"}\n{"status":"downloading","comple',
        b'ted":500,"total":1000}\n',
        b'{"status":"success"}\n',
    ]
    route(monkeypatch, {"/api/pull": FakeResponse(chunks=chunks)})

    events: List[Dict[str, Any]] = []
    result = ollama.pull("qwen3.6:27b", cfg=cfg, progress=events.append)

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["completed"] == 500
    assert result["total"] == 1000
    assert events[-1]["percent"] == 50.0
    assert result["events"] == 3


def test_pull_reports_a_final_error_object_without_raising(cfg, monkeypatch):
    chunks = [
        b'{"status":"pulling manifest"}\n',
        b'{"error":"model \'nope:1b\' not found"}\n',
    ]
    route(monkeypatch, {"/api/pull": FakeResponse(chunks=chunks)})

    result = ollama.pull("nope:1b", cfg=cfg)

    assert result["ok"] is False
    assert "not found" in result["error"]
    assert "refused to pull" in result["reason"]


def test_pull_returns_a_result_when_the_network_dies_mid_stream(cfg, monkeypatch):
    class Dying(FakeResponse):
        def read(self, size: int = -1) -> bytes:
            raise OSError("connection reset by peer")

    route(monkeypatch, {"/api/pull": Dying()})

    result = ollama.pull("qwen3.6:27b", cfg=cfg)

    assert result["ok"] is False
    assert "connection reset" in result["reason"]


def test_a_broken_progress_callback_does_not_abort_the_pull(cfg, monkeypatch):
    route(monkeypatch, {"/api/pull": FakeResponse(chunks=[b'{"status":"success"}\n'])})

    def hostile(event):
        raise RuntimeError("the caller's printer is broken")

    result = ollama.pull("qwen3.6:27b", cfg=cfg, progress=hostile)
    assert result["ok"] is True


# --------------------------------------------------------------------------- #
#  ensure_model
# --------------------------------------------------------------------------- #
class FakeDaemon:
    """An ollama server that starts empty and remembers what was pulled."""

    def __init__(self) -> None:
        self.models: List[str] = []
        self.pulls: List[str] = []

    def __call__(self, request: Any, timeout: float = 0.0) -> FakeResponse:
        url = getattr(request, "full_url", str(request))
        if url.endswith("/api/tags"):
            return FakeResponse(json.dumps({
                "models": [{"name": n, "size": 16_100_000_000} for n in self.models]
            }).encode("utf-8"))
        if url.endswith("/api/pull"):
            body = json.loads((request.data or b"{}").decode("utf-8"))
            tag = body["model"]
            self.pulls.append(tag)
            self.models.append(tag)
            return FakeResponse(chunks=[
                b'{"status":"pulling manifest"}\n',
                b'{"status":"success"}\n',
            ])
        raise AssertionError(f"unexpected request to {url}")


def test_ensure_model_pulls_once_and_is_a_no_op_the_second_time(cfg, monkeypatch):
    daemon = FakeDaemon()
    monkeypatch.setattr(ollama, "_urlopen", daemon)
    fake_disk(monkeypatch, free=400_000_000_000)

    first = ollama.ensure_model(cfg)
    assert first["ok"] is True
    assert first["action"] == "pulled"
    assert daemon.pulls == [LLMConfig().ollama_model]

    second = ollama.ensure_model(cfg)
    assert second["ok"] is True
    assert second["action"] == "present"
    assert daemon.pulls == [LLMConfig().ollama_model], (
        "the model must not be pulled twice"
    )


def test_ensure_model_does_not_pull_when_space_is_short(cfg, monkeypatch):
    daemon = FakeDaemon()
    monkeypatch.setattr(ollama, "_urlopen", daemon)
    fake_disk(monkeypatch, free=3_000_000_000)

    result = ollama.ensure_model(cfg)

    assert result["ok"] is False
    assert daemon.pulls == [], "nothing may be downloaded when it cannot fit"
    assert "GB short" in result["reason"]


def test_ensure_model_reports_a_dead_server_without_raising(cfg, monkeypatch):
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    fake_disk(monkeypatch, free=400_000_000_000)

    result = ollama.ensure_model(cfg)
    assert result["ok"] is False
    assert result["action"] == "no-server"


# --------------------------------------------------------------------------- #
#  Serving
# --------------------------------------------------------------------------- #
def test_is_serving_is_true_when_the_api_answers(cfg, monkeypatch):
    route(monkeypatch, {"/api/tags": lambda r: tags_response()})
    assert ollama.is_serving(cfg) is True


def test_is_serving_is_false_when_nothing_listens(cfg, monkeypatch):
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    assert ollama.is_serving(cfg) is False


def test_start_server_reports_an_already_running_daemon_without_spawning(cfg, monkeypatch):
    route(monkeypatch, {"/api/tags": lambda r: tags_response()})

    def forbidden(*args, **kwargs):
        raise AssertionError("no process may be spawned")

    monkeypatch.setattr(ollama.subprocess, "Popen", forbidden)

    result = ollama.start_server(cfg)
    assert result["ok"] is True
    assert result["action"] == "already-running"


def test_start_server_refuses_when_ollama_is_not_installed(cfg, monkeypatch):
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    monkeypatch.setattr(ollama.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("nothing to spawn"))

    result = ollama.start_server(cfg)
    assert result["ok"] is False
    assert "not installed" in result["reason"]


def test_stop_server_kills_and_reaps_the_child_it_owns(cfg, monkeypatch):
    class FakeProc:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False
            self.pid = 4242

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    proc = FakeProc()
    monkeypatch.setattr(ollama, "_SERVER", proc)

    result = ollama.stop_server(cfg)

    assert result["ok"] is True
    assert proc.terminated and proc.waited, "the child must be killed *and* reaped"
    assert ollama._SERVER is None


def test_host_url_and_authority_track_the_config(cfg):
    cfg.llm.ollama_host = "http://192.168.1.9:11434"
    assert ollama.host_url(cfg) == "http://192.168.1.9:11434"
    assert ollama.host_authority(cfg) == "192.168.1.9:11434"

    cfg.llm.ollama_host = "10.0.0.4:11434"
    assert ollama.host_url(cfg) == "http://10.0.0.4:11434"


# --------------------------------------------------------------------------- #
#  Disk report
# --------------------------------------------------------------------------- #
def test_disk_report_is_correct_against_a_faked_disk_usage(tmp_path, monkeypatch):
    fake_disk(monkeypatch, free=12_000_000_000, total=100_000_000_000)

    report = assets.disk_report([tmp_path])
    entry = report[str(tmp_path)]

    assert entry["ok"] is True
    assert entry["free_bytes"] == 12_000_000_000
    assert entry["total_bytes"] == 100_000_000_000
    assert entry["used_bytes"] == 88_000_000_000
    assert entry["free_gb"] == 12.0
    assert entry["total_gb"] == 100.0


def test_disk_report_measures_a_target_that_does_not_exist_yet(tmp_path, monkeypatch):
    """The whole point is to check space *before* creating the directory."""
    fake_disk(monkeypatch, free=7_000_000_000)
    target = tmp_path / "not" / "yet" / "created"

    entry = assets.disk_report([target])[str(target)]

    assert entry["ok"] is True
    assert entry["free_gb"] == 7.0
    assert entry["measured_at"] == str(tmp_path)


def test_disk_report_survives_an_unreadable_filesystem(tmp_path, monkeypatch):
    import shutil as shutil_module

    def boom(path):
        raise OSError("device not configured")

    monkeypatch.setattr(shutil_module, "disk_usage", boom)

    entry = assets.disk_report([tmp_path])[str(tmp_path)]
    assert entry["ok"] is False
    assert entry["free_bytes"] == -1
    assert "disk_usage failed" in entry["reason"]


def test_disk_report_of_nothing_is_empty():
    assert assets.disk_report([]) == {}
    assert assets.disk_report(None) == {}


# --------------------------------------------------------------------------- #
#  Speech assets
# --------------------------------------------------------------------------- #
def test_piper_voice_status_reports_a_missing_voice(cfg):
    result = assets.piper_voice_status(cfg)

    assert result["present"] is False
    assert result["voice"] == "en_GB-alan-medium"
    assert result["ok"] is False


def test_piper_voice_status_reports_a_present_voice(cfg):
    voices = Path(cfg.voices_dir())
    model = voices / "en_GB-alan-medium.onnx"
    model.write_bytes(b"x" * 2048)
    model.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")

    result = assets.piper_voice_status(cfg)

    assert result["present"] is True
    assert result["ok"] is True
    assert result["size_bytes"] == 2050
    assert "en_GB-alan-medium" in result["reason"]


def test_ensure_piper_voice_never_re_downloads_a_present_voice(cfg, monkeypatch):
    voices = Path(cfg.voices_dir())
    (voices / "en_GB-alan-medium.onnx").write_bytes(b"x" * 16)
    (voices / "en_GB-alan-medium.onnx.json").write_text("{}", encoding="utf-8")

    from jarvis.speech import tts as tts_module

    asked: List[bool] = []

    def recording(self, download=False):
        asked.append(bool(download))
        model = self.voices_dir / f"{self.cfg.piper_voice}.onnx"
        return {
            "voice": self.cfg.piper_voice,
            "model_path": model,
            "config_path": model.with_suffix(".onnx.json"),
            "downloaded": False,
            "errors": [],
        }

    monkeypatch.setattr(tts_module.PiperTTS, "ensure_voice", recording, raising=True)

    result = assets.ensure_piper_voice(cfg)

    assert result["present"] is True
    assert asked == [False], "a present voice must only be probed, never fetched"


def test_ensure_piper_voice_reports_a_failed_download(cfg, monkeypatch):
    from jarvis.speech import tts as tts_module

    def failing(self, download=False):
        return {
            "voice": self.cfg.piper_voice,
            "model_path": self.voices_dir / f"{self.cfg.piper_voice}.onnx",
            "config_path": self.voices_dir / f"{self.cfg.piper_voice}.onnx.json",
            "downloaded": False,
            "errors": ["download failed: no route to host"],
        }

    monkeypatch.setattr(tts_module.PiperTTS, "ensure_voice", failing, raising=True)

    result = assets.ensure_piper_voice(cfg)
    assert result["ok"] is False
    assert "no route to host" in result["reason"]


def test_whisper_repo_candidates_expand_a_size_alias():
    assert assets.whisper_repo_candidates("base.en") == [
        "Systran/faster-whisper-base.en",
        "guillaumekln/faster-whisper-base.en",
    ]
    assert assets.whisper_repo_candidates("org/custom-build") == ["org/custom-build"]
    assert assets.whisper_repo_candidates("") == []


#: The STT model the shipped config actually asks for. Derived rather than
#: written down, so changing STTConfig.model does not silently rot these tests.
DEFAULT_STT_MODEL = STTConfig().model
DEFAULT_STT_REPO = f"Systran/faster-whisper{DEFAULT_STT_MODEL and '-' + DEFAULT_STT_MODEL}"


def _seed_whisper_cache(root: Path, repo: str = DEFAULT_STT_REPO) -> Path:
    snapshot = root / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (snapshot / name).write_bytes(b"x" * 1000)
    return snapshot


def test_whisper_model_status_finds_a_cached_model(cfg, tmp_path, monkeypatch):
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    _seed_whisper_cache(cache)
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: cache)

    result = assets.whisper_model_status(cfg)

    assert result["present"] is True
    assert result["model"] == DEFAULT_STT_MODEL
    assert result["repo"] == DEFAULT_STT_REPO
    assert result["size_bytes"] == 3000


def test_whisper_model_status_rejects_an_interrupted_download(cfg, tmp_path, monkeypatch):
    """A directory missing model.bin is not a usable model, however present it looks."""
    cache = tmp_path / "hf-cache"
    snapshot = cache / ("models--" + DEFAULT_STT_REPO.replace("/", "--")) / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: cache)

    result = assets.whisper_model_status(cfg)

    assert result["present"] is False
    assert "not cached" in result["reason"]
    expected_mb = int(assets._WHISPER_SIZE_GB[DEFAULT_STT_MODEL] * 1000)
    assert f"{expected_mb} MB" in result["reason"], (
        "the size must be quoted before the download"
    )


def test_ensure_whisper_model_says_so_cleanly_when_faster_whisper_is_absent(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: tmp_path / "empty-cache")
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", None)

    result = assets.ensure_whisper_model(cfg)

    assert result["ok"] is False
    assert result["action"] == "unavailable"
    assert "faster-whisper is not installed" in result["reason"]


def test_ensure_whisper_model_prefetches_through_the_library(cfg, tmp_path, monkeypatch):
    """Constructing a WhisperModel is what triggers the download; nothing else does."""
    import sys
    import types

    cache = tmp_path / "hf-cache"
    cache.mkdir()
    built: List[Dict[str, Any]] = []

    def fake_model(name, device="cpu", compute_type="int8"):
        built.append({"name": name, "device": device, "compute_type": compute_type})
        _seed_whisper_cache(cache)
        return object()

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = fake_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: cache)

    result = assets.ensure_whisper_model(cfg)

    assert result["ok"] is True
    assert result["action"] == "fetched"
    assert built == [
        {"name": DEFAULT_STT_MODEL, "device": "cpu", "compute_type": "int8"}
    ]

    # Second run: already cached, so the library is never touched again.
    again = assets.ensure_whisper_model(cfg)
    assert again["present"] is True
    assert len(built) == 1


def test_ensure_whisper_model_reports_a_download_failure(cfg, tmp_path, monkeypatch):
    import sys
    import types

    def fake_model(name, device="cpu", compute_type="int8"):
        raise OSError("connection reset while downloading model.bin")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = fake_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: tmp_path / "cache")

    result = assets.ensure_whisper_model(cfg)

    assert result["ok"] is False
    assert result["action"] == "failed"
    assert "connection reset" in result["reason"]


def test_espeak_status_finds_espeak_ng(monkeypatch):
    monkeypatch.setattr(
        assets, "_which",
        lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None,
    )
    result = assets.espeak_status()

    assert result["present"] is True
    assert result["binary"] == "espeak-ng"
    assert result["install_hint"] == ""


def test_espeak_status_prints_the_command_rather_than_running_it():
    """The installer never runs sudo; it says what to run."""
    result = assets.espeak_status()

    assert result["present"] is False
    assert "sudo apt-get install" in result["install_hint"]
    assert "not fatal" in result["reason"]


# --------------------------------------------------------------------------- #
#  provision_all / status
# --------------------------------------------------------------------------- #
def test_provision_all_aggregates_and_never_raises_when_all_is_missing(
    cfg, tmp_path, monkeypatch
):
    """The worst case: no network, no binaries, no models, nothing cached."""
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("no network")),
    )
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: tmp_path / "empty-cache")

    from jarvis.speech import tts as tts_module
    monkeypatch.setattr(
        tts_module.PiperTTS, "ensure_voice",
        lambda self, download=False: {
            "voice": self.cfg.piper_voice,
            "model_path": self.voices_dir / "v.onnx",
            "config_path": self.voices_dir / "v.onnx.json",
            "downloaded": False,
            "errors": ["download failed: no network"],
        },
        raising=True,
    )

    result = assets.provision_all(cfg)

    assert result["ok"] is False
    assert set(result["failed"]) >= {"ollama", "model", "voice"}
    assert len(result["summary"]) >= 5
    assert all(isinstance(line, str) for line in result["summary"])
    assert result["disk"]["ok"] is True


def test_provision_all_skips_the_big_downloads_on_request(cfg, monkeypatch):
    """--no-model / --no-voice have to actually skip, not just log."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a skipped component must not be provisioned")

    monkeypatch.setattr(ollama, "install", forbidden)
    monkeypatch.setattr(ollama, "ensure_model", forbidden)
    monkeypatch.setattr(assets, "ensure_whisper_model", forbidden)
    monkeypatch.setattr(assets, "ensure_piper_voice", forbidden)
    monkeypatch.setattr(ollama, "status", lambda c: {"installed": False})

    result = assets.provision_all(cfg, skip=["ollama", "model", "voice", "stt"])

    assert result["skipped"] == ["model", "ollama", "stt", "voice"]
    assert result["components"]["model"]["action"] == "skipped"
    assert result["components"]["voice"]["action"] == "skipped"


def test_provision_all_contains_an_exploding_component(cfg, monkeypatch):
    """One component blowing up must not take the other four with it."""
    def explode(*args, **kwargs):
        raise RuntimeError("the disk caught fire")

    monkeypatch.setattr(ollama, "install", explode)
    monkeypatch.setattr(ollama, "ensure_model", lambda *a, **k: {"ok": True, "reason": "fine"})
    monkeypatch.setattr(assets, "ensure_piper_voice", lambda *a, **k: {"ok": True, "reason": "fine"})
    monkeypatch.setattr(assets, "ensure_whisper_model", lambda *a, **k: {"ok": True, "reason": "fine"})

    result = assets.provision_all(cfg)

    assert result["ok"] is False
    assert result["failed"] == ["ollama"]
    assert "the disk caught fire" in result["components"]["ollama"]["reason"]
    assert result["components"]["model"]["ok"] is True


def test_provision_all_is_ok_when_only_espeak_is_missing(cfg, monkeypatch):
    """Piper is the default voice; espeak-ng is advisory, not required."""
    monkeypatch.setattr(ollama, "install", lambda *a, **k: {"ok": True, "reason": "current"})
    monkeypatch.setattr(ollama, "ensure_model", lambda *a, **k: {"ok": True, "reason": "present"})
    monkeypatch.setattr(assets, "ensure_piper_voice", lambda *a, **k: {"ok": True, "reason": "present"})
    monkeypatch.setattr(assets, "ensure_whisper_model", lambda *a, **k: {"ok": True, "reason": "cached"})

    result = assets.provision_all(cfg)

    assert result["ok"] is True
    assert result["components"]["espeak"]["ok"] is False


def test_status_gives_the_whole_picture_without_side_effects(cfg, tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setattr(
        ollama, "_urlopen",
        lambda r, timeout=0.0: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: tmp_path / "empty-cache")

    before = sorted(p.name for p in Path(cfg.home()).iterdir())
    result = runtime_status(cfg)
    after = sorted(p.name for p in Path(cfg.home()).iterdir())

    assert result["ready"] is False
    assert set(result["missing"]) == {"ollama", "model", "voice", "stt"}
    assert result["ollama"]["installed"] is False
    assert result["ollama"]["serving"] is False
    assert result["ollama"]["model"] == LLMConfig().ollama_model
    assert result["espeak"]["present"] is False
    assert "disk" in result
    # voices_dir() is created on access by Config; nothing else may appear.
    assert set(after) - set(before) <= {"voices"}


def test_status_reports_ready_when_everything_is_in_place(cfg, tmp_path, monkeypatch):
    _install_managed(cfg, "0.9.9")
    _report_version(monkeypatch, "0.9.9")
    route(monkeypatch, {"/api/tags": lambda r: tags_response(LLMConfig().ollama_model)})

    voices = Path(cfg.voices_dir())
    (voices / "en_GB-alan-medium.onnx").write_bytes(b"x" * 16)
    (voices / "en_GB-alan-medium.onnx.json").write_text("{}", encoding="utf-8")

    cache = tmp_path / "hf-cache"
    cache.mkdir()
    _seed_whisper_cache(cache)
    monkeypatch.setattr(assets, "_hf_cache_dir", lambda: cache)

    result = runtime_status(cfg)

    assert result["ready"] is True
    assert result["missing"] == []
    assert result["ollama"]["version"] == "0.9.9"
    assert result["ollama"]["model_present"] is True
