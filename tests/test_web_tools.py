"""Tests for :mod:`jarvis.tools.web_tools`.

All network calls are stubbed; nothing here opens a real socket except the
``check_internet`` test, which monkeypatches ``socket.socket`` entirely.
"""

from __future__ import annotations

import io
import socket
from typing import Optional

import pytest

from jarvis.core.contracts import ToolResult
from jarvis.tools.web_tools import (
    _check_internet,
    _make_download_file,
    _make_http_get,
    _make_web_search,
    _validate_url,
    extract_text_from_html,
)


# --------------------------------------------------------------------------- #
#  Security fakes
# --------------------------------------------------------------------------- #
class _AllowSecurity:
    class _Cfg:
        allow_network = True

    cfg = _Cfg()


class _DenySecurity:
    class _Cfg:
        allow_network = False

    cfg = _Cfg()


class _DuckSecurity:
    """Duck-typed security without .cfg — should still work."""

    allow_network = True


# --------------------------------------------------------------------------- #
#  URL validation
# --------------------------------------------------------------------------- #
class TestValidateUrl:

    @pytest.mark.parametrize("url", [
        "http://example.com",
        "https://example.com/path?x=1",
        "HTTP://Example.com",
    ])
    def test_accepts_http_https(self, url):
        ok, err = _validate_url(url)
        assert ok is True, err

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://example.com",
        "javascript:alert(1)",
        "data:text/plain,hi",
        "//example.com",
        "",
        "not a url",
    ])
    def test_rejects_bad(self, url):
        ok, err = _validate_url(url)
        assert ok is False, f"should reject: {url!r}"
        assert err


# --------------------------------------------------------------------------- #
#  Fake urllib response
# --------------------------------------------------------------------------- #
class _FakeHeaders:
    def __init__(self, items):
        self._items = list(items)

    def items(self):
        return list(self._items)

    def get_content_charset(self):
        for k, v in self._items:
            if k.lower() == "content-type" and "charset=" in v.lower():
                return v.split("charset=")[-1].strip()
        return None


class _FakeResponse:
    def __init__(self, body: bytes, url: str = "http://example.com/",
                 status: int = 200, headers=None):
        self._buf = io.BytesIO(body)
        self._url = url
        self._status = status
        self.status = status
        self.headers = _FakeHeaders(headers or [("Content-Type", "text/html; charset=utf-8")])

    def read(self, size: Optional[int] = None) -> bytes:
        if size is None:
            return self._buf.read()
        return self._buf.read(int(size))

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, response):
        self._response = response
        self.opened = []

    def open(self, request, timeout=None):
        self.opened.append((request.full_url, timeout))
        return self._response


# --------------------------------------------------------------------------- #
#  http_get
# --------------------------------------------------------------------------- #
class TestHttpGet:

    def test_returns_body_and_status(self, monkeypatch):
        body = "<html><body>Hello</body></html>".encode("utf-8")
        resp = _FakeResponse(body, url="http://example.com/hi")

        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        get = _make_http_get(_AllowSecurity())
        result = get(url="http://example.com/hi", timeout=5.0, max_bytes=1000)
        assert result.ok is True
        assert result.output["status"] == 200
        assert result.output["text"] == body.decode("utf-8")
        assert result.output["truncated"] is False
        assert result.output["bytes_read"] == len(body)

    def test_body_larger_than_cap_reports_truncated(self, monkeypatch):
        body = b"x" * 5000
        resp = _FakeResponse(body)
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        get = _make_http_get(_AllowSecurity())
        result = get(url="http://example.com/big", timeout=5.0, max_bytes=1000)
        assert result.ok is True
        assert result.output["bytes_read"] == 1000
        assert result.output["truncated"] is True

    def test_body_exactly_at_cap_is_not_truncated(self, monkeypatch):
        body = b"y" * 1000
        resp = _FakeResponse(body)
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        get = _make_http_get(_AllowSecurity())
        result = get(url="http://example.com/exact", timeout=5.0, max_bytes=1000)
        assert result.ok is True
        assert result.output["bytes_read"] == 1000
        assert result.output["truncated"] is False, (
            "A body of exactly max_bytes must NOT be flagged as truncated"
        )

    def test_denied_when_network_off(self):
        get = _make_http_get(_DenySecurity())
        result = get(url="http://example.com/", timeout=5.0, max_bytes=100)
        assert result.ok is False
        assert "network" in (result.error or "").lower()

    def test_rejects_non_http_scheme(self, monkeypatch):
        # Even if opener would work, scheme check must fire first.
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(
            wt, "_do_open",
            lambda req, timeout: (_ for _ in ()).throw(AssertionError("should not open")),
        )
        get = _make_http_get(_AllowSecurity())
        result = get(url="file:///etc/passwd", timeout=5.0, max_bytes=100)
        assert result.ok is False

    def test_url_error_becomes_failure(self, monkeypatch):
        import urllib.error
        import jarvis.tools.web_tools as wt

        def _raise(req, timeout):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(wt, "_do_open", _raise)
        get = _make_http_get(_AllowSecurity())
        result = get(url="http://example.com/", timeout=5.0, max_bytes=100)
        assert result.ok is False
        assert "URL error" in (result.error or "")


# --------------------------------------------------------------------------- #
#  HTML extractor
# --------------------------------------------------------------------------- #
_HTML_FIXTURE = """<!doctype html>
<html><head>
<title>Ignore me title</title>
<script>alert('bad');</script>
<style>body { color: red; }</style>
</head>
<body>
<nav>navigation should be hidden</nav>
<header>site header</header>
<main>
<h1>Café &amp; Résumé</h1>
<p>First paragraph with <b>bold</b> and <i>italic</i>.</p>
<p>Second paragraph — with an em-dash and an entity: &copy; 2026.</p>
<ul><li>one</li><li>two</li></ul>
<script>more bad js</script>
</main>
<footer>site footer</footer>
</body></html>
"""


class TestExtractor:

    def test_strips_script_and_style(self):
        text = extract_text_from_html(_HTML_FIXTURE)
        assert "alert('bad');" not in text
        assert "body { color: red" not in text
        assert "more bad js" not in text

    def test_drops_nav_header_footer(self):
        text = extract_text_from_html(_HTML_FIXTURE)
        assert "navigation should be hidden" not in text
        assert "site header" not in text
        assert "site footer" not in text

    def test_decodes_entities_and_unicode(self):
        text = extract_text_from_html(_HTML_FIXTURE)
        assert "Café & Résumé" in text
        assert "© 2026" in text or "© 2026" in text

    def test_handles_nested_tags(self):
        text = extract_text_from_html(_HTML_FIXTURE)
        assert "First paragraph with bold and italic." in text
        assert "one" in text and "two" in text

    def test_empty_input(self):
        assert extract_text_from_html("") == ""
        assert extract_text_from_html(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
#  web_search
# --------------------------------------------------------------------------- #
_DDG_FIXTURE = """<html><body>
<div class="results">
<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">First &amp; Foremost</a>
<a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">A first <b>snippet</b> here.</a>
<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Second Link</a>
<a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Second snippet.</a>
</div>
</body></html>
"""


class TestWebSearch:

    def test_parses_results(self, monkeypatch):
        resp = _FakeResponse(_DDG_FIXTURE.encode("utf-8"),
                             url="https://duckduckgo.com/html/?q=hi")
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        search = _make_web_search(_AllowSecurity())
        result = search(query="hi", max_results=5)
        assert result.ok is True
        results = result.output["results"]
        assert len(results) == 2
        assert results[0]["title"] == "First & Foremost"
        assert results[0]["url"] == "https://example.com/one"
        assert "first" in results[0]["snippet"].lower()
        assert results[1]["url"] == "https://example.com/two"

    def test_no_results_returns_failure(self, monkeypatch):
        resp = _FakeResponse(b"<html><body>no results</body></html>")
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)
        search = _make_web_search(_AllowSecurity())
        result = search(query="obscure", max_results=3)
        assert result.ok is False

    def test_empty_query_rejected(self):
        search = _make_web_search(_AllowSecurity())
        assert search(query="", max_results=3).ok is False


# --------------------------------------------------------------------------- #
#  download_file
# --------------------------------------------------------------------------- #
class TestDownload:

    def test_streams_ok(self, monkeypatch, tmp_path):
        body = b"BINARYDATA" * 100
        resp = _FakeResponse(body)
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        dest = tmp_path / "out.bin"
        download = _make_download_file(_AllowSecurity())
        result = download(url="http://example.com/x", dest=str(dest), max_mb=1)
        assert result.ok is True
        assert dest.exists() and dest.stat().st_size == len(body)

    def test_removes_partial_when_cap_exceeded(self, monkeypatch, tmp_path):
        body = b"z" * (2 * 1024 * 1024)  # 2 MiB
        resp = _FakeResponse(body)
        import jarvis.tools.web_tools as wt
        monkeypatch.setattr(wt, "_do_open", lambda req, timeout: resp)

        dest = tmp_path / "huge.bin"
        download = _make_download_file(_AllowSecurity())
        result = download(url="http://example.com/big", dest=str(dest), max_mb=1)
        assert result.ok is False
        assert "exceeded" in (result.error or "").lower()
        assert not dest.exists(), "partial file must be removed on overflow"

    def test_denied_when_network_off(self, tmp_path):
        download = _make_download_file(_DenySecurity())
        result = download(
            url="http://example.com/", dest=str(tmp_path / "d.bin"), max_mb=1
        )
        assert result.ok is False


# --------------------------------------------------------------------------- #
#  check_internet
# --------------------------------------------------------------------------- #
class TestCheckInternet:

    def test_reports_online_when_connect_succeeds(self, monkeypatch):
        class _OKSocket:
            def __init__(self, *a, **kw):
                pass

            def settimeout(self, t):
                pass

            def connect(self, addr):
                return None

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", _OKSocket)
        result = _check_internet(timeout=1.0)
        assert result.ok is True
        assert result.output["online"] is True

    def test_reports_offline_when_all_targets_fail(self, monkeypatch):
        class _FailSocket:
            def __init__(self, *a, **kw):
                pass

            def settimeout(self, t):
                pass

            def connect(self, addr):
                raise OSError("no route")

            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", _FailSocket)
        result = _check_internet(timeout=1.0)
        assert result.ok is True
        assert result.output["online"] is False


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
class TestBuildTools:

    def test_registers_expected(self):
        from jarvis.tools.web_tools import build_tools

        class _Ctx:
            security = _AllowSecurity()

        tools = {t.name: t for t in build_tools(_Ctx())}
        for expected in ("http_get", "http_post_json", "fetch_page_text",
                         "web_search", "download_file", "open_url", "check_internet"):
            assert expected in tools, f"missing tool {expected}"
        assert tools["download_file"].spec.dangerous is True
        assert tools["http_get"].spec.dangerous is False
