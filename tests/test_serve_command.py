"""The `jarvis serve` subcommand — the CLI half of remote access.

Nothing here binds a socket. The server module is replaced with a recording
stand-in, so what is under test is exactly what the CLI is responsible for:
which address it chose, whether a token is required, whether the token value
ever reaches the terminal, and whether the operator is told what a non-loopback
bind actually means.

The distinction that shapes these tests: JARVIS may do anything it likes to the
machine it runs on — that is the point of the project. Who is allowed to *reach*
it over the network is a separate question, and the answers to that question are
what is asserted below.
"""

from __future__ import annotations

import ipaddress
import logging
import sys
import types
from pathlib import Path

import pytest

from jarvis import cli


ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "REMOTE_ACCESS.md"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
class RecordingServer:
    """Stands in for ``jarvis.server.serve`` and binds nothing."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, config, *, host, port, token, certfile, keyfile):
        self.calls.append({
            "config": config,
            "host": host,
            "port": port,
            "token": token,
            "certfile": certfile,
            "keyfile": keyfile,
        })
        return 0

    @property
    def last(self) -> dict:
        assert self.calls, "the server was never started"
        return self.calls[-1]


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> RecordingServer:
    """Install a `jarvis.server` module whose serve() returns immediately."""
    recorder = RecordingServer()
    module = types.ModuleType("jarvis.server")
    module.serve = recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis.server", module)
    return recorder


@pytest.fixture
def serve_config(config, monkeypatch: pytest.MonkeyPatch):
    """Make `jarvis serve` load the isolated test config, never the real one."""
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    return config


@pytest.fixture
def no_lan(monkeypatch: pytest.MonkeyPatch):
    """Pin the reported LAN address; never probe the real network."""
    monkeypatch.setattr(cli, "_lan_address", lambda: "192.168.1.42")
    return "192.168.1.42"


def run(argv) -> int:
    """Parse `argv` and dispatch, the way `main()` does."""
    args = cli.build_parser().parse_args(argv)
    return int(args.func(args) or 0)


# --------------------------------------------------------------------------- #
#  Registration and flag parsing
# --------------------------------------------------------------------------- #
def test_serve_is_registered():
    args = cli.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.func is cli.cmd_serve


def test_serve_appears_in_the_top_level_help():
    assert "serve" in cli.build_parser().format_help()


def test_serve_defaults_to_loopback_on_8765():
    args = cli.build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.token is None
    assert args.cert is None
    assert args.key is None
    assert args.print_url is False


def test_every_serve_flag_parses():
    args = cli.build_parser().parse_args([
        "serve",
        "--host", "0.0.0.0",
        "--port", "9100",
        "--token", "abc",
        "--cert", "/etc/ssl/jarvis.crt",
        "--key", "/etc/ssl/jarvis.key",
        "--print-url",
    ])
    assert (args.host, args.port, args.token) == ("0.0.0.0", 9100, "abc")
    assert args.cert == "/etc/ssl/jarvis.crt"
    assert args.key == "/etc/ssl/jarvis.key"
    assert args.print_url is True


def test_port_must_be_a_number():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--port", "not-a-port"])


def test_adding_serve_did_not_disturb_the_other_subcommands():
    parser = cli.build_parser()
    for name, expected in (
        ("chat", cli.cmd_chat),
        ("voice", cli.cmd_voice),
        ("doctor", cli.cmd_doctor),
        ("tools", cli.cmd_tools),
        ("setup", cli.cmd_setup),
    ):
        assert parser.parse_args([name]).func is expected


# --------------------------------------------------------------------------- #
#  --print-url
# --------------------------------------------------------------------------- #
def test_print_url_emits_a_url_and_exits_zero_without_binding(capsys, monkeypatch):
    # Any attempt to import the server would be a bug: --print-url binds nothing.
    monkeypatch.setitem(sys.modules, "jarvis.server", None)

    assert run(["serve", "--print-url"]) == 0

    printed = capsys.readouterr().out.strip()
    assert printed == "http://127.0.0.1:8765"


def test_print_url_honours_host_and_port(capsys, no_lan):
    assert run(["serve", "--print-url", "--port", "9100"]) == 0
    assert capsys.readouterr().out.strip() == "http://127.0.0.1:9100"


def test_print_url_uses_https_when_tls_is_configured(capsys):
    code = run(["serve", "--print-url", "--cert", "c.pem", "--key", "k.pem"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "https://127.0.0.1:8765"


def test_print_url_turns_a_wildcard_bind_into_something_connectable(capsys, no_lan):
    """0.0.0.0 is a bind address; no browser can open it."""
    assert run(["serve", "--print-url", "--host", "0.0.0.0"]) == 0
    assert capsys.readouterr().out.strip() == "http://192.168.1.42:8765"


def test_print_url_never_leaks_the_token(capsys, serve_config, monkeypatch):
    monkeypatch.setenv("JARVIS_SERVER_TOKEN", "SEKRIT-env-token")
    assert run(["serve", "--print-url"]) == 0
    assert "SEKRIT-env-token" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  A missing server module is an explanation, not a traceback
# --------------------------------------------------------------------------- #
def test_missing_server_module_is_reported_cleanly(capsys, serve_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "jarvis.server", None)

    code = run(["serve"])

    out = capsys.readouterr().out
    assert code == 1
    assert "Traceback" not in out
    assert "jarvis.server" in out
    # It should still say how to reach JARVIS from elsewhere in the meantime.
    assert "ssh" in out
    assert "REMOTE_ACCESS.md" in out


def test_main_reports_a_missing_server_module_without_a_traceback(capsys, serve_config,
                                                                 monkeypatch):
    monkeypatch.setitem(sys.modules, "jarvis.server", None)

    code = cli.main(["serve"])

    out = capsys.readouterr().out
    assert code == 1
    assert "Traceback" not in out
    assert "ModuleNotFoundError" not in out


# --------------------------------------------------------------------------- #
#  Loopback detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "localhost", "::1", "[::1]"])
def test_loopback_addresses_are_recognised(host):
    assert cli._is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.42", "100.101.102.103", "10.0.0.5",
     "jarvis-box", "jarvis-box.tailnet.ts.net", ""],
)
def test_anything_not_provably_loopback_is_treated_as_reachable(host):
    """Guessing the other way would silently drop the token requirement."""
    assert cli._is_loopback(host) is False


# --------------------------------------------------------------------------- #
#  URL construction
# --------------------------------------------------------------------------- #
def test_wildcard_falls_back_to_loopback_when_there_is_no_lan_address(monkeypatch):
    monkeypatch.setattr(cli, "_lan_address", lambda: None)
    assert cli._url_host("0.0.0.0") == "127.0.0.1"


def test_ipv6_literals_are_bracketed_for_urls():
    assert cli._serve_url("::1", 8765) == "http://[::1]:8765"


def test_hostnames_pass_through_unchanged():
    assert cli._serve_url("jarvis-box", 8765) == "http://jarvis-box:8765"


def test_lan_probe_never_contacts_a_real_host(monkeypatch):
    """The probe must use documentation space and must not transmit."""
    seen = {}

    class FakeSocket:
        def __init__(self, family, kind):
            seen["family"] = family
            seen["kind"] = kind

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def settimeout(self, value):
            seen["timeout"] = value

        def connect(self, target):
            seen["target"] = target

        def getsockname(self):
            return ("192.168.1.42", 51234)

    monkeypatch.setattr(cli.socket, "socket", FakeSocket)

    assert cli._lan_address() == "192.168.1.42"
    assert seen["kind"] == cli.socket.SOCK_DGRAM, "a datagram socket sends nothing"
    address, _port = seen["target"]
    assert ipaddress.ip_address(address) in ipaddress.ip_network("192.0.2.0/24"), (
        "the probe target must stay inside RFC 5737 documentation space"
    )


def test_lan_probe_falls_back_to_none_when_there_is_no_route(monkeypatch):
    class DeadSocket:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def settimeout(self, value):
            pass

        def connect(self, target):
            raise OSError("network is unreachable")

        def getsockname(self):  # pragma: no cover - never reached
            return ("", 0)

    monkeypatch.setattr(cli.socket, "socket", DeadSocket)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "jarvis-box")
    monkeypatch.setattr(cli.socket, "gethostbyname", lambda name: "127.0.1.1")

    assert cli._lan_address() is None


# --------------------------------------------------------------------------- #
#  Token resolution
# --------------------------------------------------------------------------- #
def test_a_token_is_generated_and_saved_when_there_is_none(config):
    token, source = cli._resolve_token(config, None, environ={})

    assert source == "generated"
    assert len(token) >= 32
    saved = cli._token_file(config).read_text(encoding="utf-8").strip()
    assert saved == token


def test_a_saved_token_is_reused_on_the_next_run(config):
    first, _ = cli._resolve_token(config, None, environ={})
    second, source = cli._resolve_token(config, None, environ={})

    assert second == first
    assert source == "file", "a re-generated token would lock out every paired device"


def test_the_command_line_token_wins(config):
    cli._token_file(config).write_text("from-the-file\n", encoding="utf-8")
    token, source = cli._resolve_token(
        config, "from-the-flag", environ={cli.TOKEN_ENV_VAR: "from-the-env"},
    )
    assert (token, source) == ("from-the-flag", "flag")


def test_the_environment_beats_the_config_and_the_file(config):
    config.server = types.SimpleNamespace(token="from-the-config")
    cli._token_file(config).write_text("from-the-file\n", encoding="utf-8")

    token, source = cli._resolve_token(config, None, environ={cli.TOKEN_ENV_VAR: "from-env"})
    assert (token, source) == ("from-env", "environment")


def test_the_config_beats_the_saved_file(config):
    config.server = types.SimpleNamespace(token="from-the-config")
    cli._token_file(config).write_text("from-the-file\n", encoding="utf-8")

    token, source = cli._resolve_token(config, None, environ={})
    assert (token, source) == ("from-the-config", "config")


def test_an_explicitly_empty_token_means_no_token(config):
    token, source = cli._resolve_token(config, "   ", environ={})
    assert token == ""
    assert source == "none"


def test_generating_a_token_does_not_overwrite_an_existing_one(config):
    path = cli._token_file(config)
    path.write_text("  kept-across-restarts  \n", encoding="utf-8")

    token, source = cli._resolve_token(config, None, environ={})

    assert (token, source) == ("kept-across-restarts", "file")
    assert "kept-across-restarts" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  What reaches the terminal
# --------------------------------------------------------------------------- #
def test_a_generated_token_is_printed_once(capsys, serve_config, fake_server):
    assert run(["serve"]) == 0

    out = capsys.readouterr().out
    token = fake_server.last["token"]
    assert token, "the server was started without a token"
    assert token in out, "a generated token must be shown, or nothing can pair with it"
    assert str(cli._token_file(serve_config)) in out


def test_a_token_from_the_config_is_never_printed(capsys, serve_config, fake_server):
    serve_config.server = types.SimpleNamespace(token="SEKRIT-config-token")

    assert run(["serve"]) == 0

    out = capsys.readouterr().out
    assert fake_server.last["token"] == "SEKRIT-config-token"
    assert "SEKRIT-config-token" not in out
    assert "configuration file" in out


def test_a_token_from_the_environment_is_never_printed(capsys, serve_config, fake_server,
                                                       monkeypatch):
    monkeypatch.setenv(cli.TOKEN_ENV_VAR, "SEKRIT-env-token")

    assert run(["serve"]) == 0

    out = capsys.readouterr().out
    assert fake_server.last["token"] == "SEKRIT-env-token"
    assert "SEKRIT-env-token" not in out


def test_a_token_from_the_command_line_is_never_echoed(capsys, serve_config, fake_server):
    assert run(["serve", "--token", "SEKRIT-flag-token"]) == 0

    out = capsys.readouterr().out
    assert fake_server.last["token"] == "SEKRIT-flag-token"
    assert "SEKRIT-flag-token" not in out


def test_a_saved_token_is_not_reprinted_on_the_second_run(capsys, serve_config, fake_server):
    assert run(["serve"]) == 0
    generated = fake_server.last["token"]
    capsys.readouterr()

    assert run(["serve"]) == 0

    out = capsys.readouterr().out
    assert fake_server.last["token"] == generated
    assert generated not in out


def test_a_token_that_cannot_be_saved_still_works_and_is_not_logged(config, caplog,
                                                                    tmp_path, monkeypatch):
    """The one place the CLI logs about a token must not log the token itself."""
    blocked = tmp_path / "server_token"
    blocked.mkdir()          # writing a file over a directory fails on every OS
    monkeypatch.setattr(cli, "_token_file", lambda cfg: blocked)

    with caplog.at_level(logging.WARNING, logger="jarvis.cli"):
        token, source = cli._resolve_token(config, None, environ={})

    assert source == "generated"
    assert token, "an unsaveable token must still work for this run"
    assert caplog.records, "failing to save the token should be reported"
    assert not any(token in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
#  The non-loopback warning
# --------------------------------------------------------------------------- #
def test_a_loopback_bind_gets_no_exposure_warning(capsys, serve_config, fake_server):
    assert run(["serve"]) == 0

    out = capsys.readouterr().out
    assert "This port is this machine." not in out
    assert "loopback" in out
    assert fake_server.last["host"] == "127.0.0.1"


def test_a_non_loopback_bind_says_what_the_port_actually_is(capsys, serve_config,
                                                            fake_server, no_lan):
    assert run(["serve", "--host", "0.0.0.0", "--token", "shared"]) == 0

    out = capsys.readouterr().out
    assert "This port is this machine." in out
    assert "ssh -N -L 8765:127.0.0.1:8765" in out
    assert "Tailscale" in out
    # The owner cannot reach the box without knowing which address to open.
    assert "192.168.1.42" in out


def test_a_non_loopback_bind_explains_the_dead_microphone_before_it_happens(
        capsys, serve_config, fake_server, no_lan):
    assert run(["serve", "--host", "192.168.1.42", "--token", "shared"]) == 0

    out = capsys.readouterr().out
    assert "A tunnel also makes the microphone work" in out
    assert "https or localhost" in out


def test_tls_suppresses_the_microphone_note_but_not_the_exposure_note(
        capsys, serve_config, fake_server, no_lan, tmp_path):
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    code = run(["serve", "--host", "0.0.0.0", "--token", "shared",
                "--cert", str(cert), "--key", str(key)])

    out = capsys.readouterr().out
    assert code == 0
    assert "This port is this machine." in out
    assert "browsers only grant it on" not in out
    assert "https://192.168.1.42:8765" in out


# --------------------------------------------------------------------------- #
#  Refusing to expose the machine anonymously
# --------------------------------------------------------------------------- #
def test_a_non_loopback_bind_with_no_token_refuses_and_says_why(capsys, serve_config,
                                                                fake_server, no_lan):
    code = run(["serve", "--host", "0.0.0.0", "--token", ""])

    out = capsys.readouterr().out
    assert code != 0
    assert not fake_server.calls, "nothing may be served without a token off loopback"
    assert "Refusing to bind 0.0.0.0:8765" in out
    assert "token" in out
    assert "ssh -N -L 8765:127.0.0.1:8765" in out


def test_an_empty_token_is_fine_on_loopback(capsys, serve_config, fake_server):
    """Restricting the owner on his own machine is not the point of any of this."""
    assert run(["serve", "--token", ""]) == 0

    assert fake_server.last["token"] == ""
    assert "none" in capsys.readouterr().out


def test_a_hostname_bind_still_requires_a_token(capsys, serve_config, fake_server):
    code = run(["serve", "--host", "jarvis-box", "--token", ""])

    assert code != 0
    assert not fake_server.calls
    assert "Refusing to bind jarvis-box:8765" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  TLS arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv", [
    ["serve", "--cert", "c.pem"],
    ["serve", "--key", "k.pem"],
])
def test_half_a_tls_configuration_is_refused(capsys, argv, fake_server):
    code = run(argv)

    assert code != 0
    assert not fake_server.calls
    assert "--cert" in capsys.readouterr().out


def test_tls_paths_are_handed_to_the_server(serve_config, fake_server, no_lan):
    assert run(["serve", "--token", "shared",
                "--cert", "/etc/ssl/j.crt", "--key", "/etc/ssl/j.key"]) == 0

    assert fake_server.last["certfile"] == "/etc/ssl/j.crt"
    assert fake_server.last["keyfile"] == "/etc/ssl/j.key"


# --------------------------------------------------------------------------- #
#  What is handed to the server
# --------------------------------------------------------------------------- #
def test_the_server_receives_the_loaded_config_and_the_chosen_bind(serve_config,
                                                                   fake_server, no_lan):
    assert run(["serve", "--host", "0.0.0.0", "--port", "9100", "--token", "shared"]) == 0

    call = fake_server.last
    assert call["config"] is serve_config
    assert call["host"] == "0.0.0.0"
    assert call["port"] == 9100
    assert call["token"] == "shared"
    assert call["certfile"] is None and call["keyfile"] is None


def test_a_bind_failure_is_reported_without_a_traceback(capsys, serve_config, monkeypatch):
    def refuse(config, **kwargs):
        raise OSError(98, "Address already in use")

    module = types.ModuleType("jarvis.server")
    module.serve = refuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis.server", module)

    code = run(["serve", "--port", "9100"])

    out = capsys.readouterr().out
    assert code == 1
    assert "Traceback" not in out
    assert "Could not serve on 127.0.0.1:9100" in out
    assert "--port" in out


def test_ctrl_c_stops_cleanly(capsys, serve_config, monkeypatch):
    def interrupt(config, **kwargs):
        raise KeyboardInterrupt

    module = types.ModuleType("jarvis.server")
    module.serve = interrupt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis.server", module)

    assert run(["serve"]) == 0
    assert "Stopping" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  The documentation is part of the feature
# --------------------------------------------------------------------------- #
def test_the_remote_access_document_exists():
    assert DOC.is_file(), (
        "docs/REMOTE_ACCESS.md is the answer to 'how do I connect from another "
        "device'; the serve command is not usable without it"
    )


@pytest.fixture(scope="module")
def doc_text() -> str:
    if not DOC.is_file():
        pytest.skip("docs/REMOTE_ACCESS.md is missing")
    return DOC.read_text(encoding="utf-8")


def test_the_document_gives_the_ssh_forwarding_command(doc_text):
    assert "ssh -N -L 8765:127.0.0.1:8765" in doc_text


def test_the_document_covers_tailscale(doc_text):
    lowered = doc_text.lower()
    assert "tailscale" in lowered
    assert "magicdns" in lowered, "the hostname is the whole point of MagicDNS here"


def test_the_document_explains_the_secure_context_microphone_rule(doc_text):
    lowered = doc_text.lower()
    assert "secure context" in lowered
    assert "getusermedia" in lowered
    assert "localhost" in lowered


def test_the_document_covers_all_three_client_operating_systems(doc_text):
    lowered = doc_text.lower()
    for name in ("linux", "macos", "windows"):
        assert name in lowered
    assert "openssh" in lowered, "Windows users need to know ssh is already installed"
    assert "putty" in lowered


def test_the_document_covers_the_lan_bind_and_its_trade_off(doc_text):
    assert "--host 0.0.0.0" in doc_text
    assert "ufw" in doc_text and "firewall-cmd" in doc_text


def test_the_document_is_honest_about_voice_over_ssh(doc_text):
    """`jarvis voice` over SSH listens in the wrong room; say so."""
    assert "jarvis voice" in doc_text
    assert "ssh -t user@jarvis-box jarvis voice" in doc_text
    assert "server's microphone" in doc_text.replace("’", "'")


def test_the_document_covers_persistence_and_lingering(doc_text):
    assert "loginctl enable-linger" in doc_text
    assert "OPERATIONS.md" in doc_text


def test_the_document_covers_the_troubleshooting_cases(doc_text):
    lowered = doc_text.lower()
    for topic in ("connection refused", "ffmpeg", "ios"):
        assert topic in lowered


def test_the_document_states_the_security_position(doc_text):
    lowered = doc_text.lower()
    assert "unrestricted" in lowered
    assert "127.0.0.1" in doc_text
