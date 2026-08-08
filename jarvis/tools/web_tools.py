"""HTTP + web-scraping tools built on the stdlib.

Deliberately dependency-free: ``requests`` and ``bs4`` are *used* when
importable, but the module never requires them.  Every call goes through
``ctx.security.allow_network`` before it hits the socket, a User-Agent is
always set, timeouts are always bounded, response bodies are always capped,
non-http(s) schemes are rejected, redirects are capped, and downloads that
exceed their size cap leave nothing on disk.
"""

from __future__ import annotations

import html
import json
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import (
    is_drive_relative,
    is_filesystem_root,
    open_path,
    resolve_path,
)
from .registry import FunctionTool, safe_truncate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
_UA = "JARVIS/1.0 (+https://example.local)"
_MAX_REDIRECTS = 5
_DEFAULT_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _network_allowed(security: Any) -> Tuple[bool, str]:
    """Ask the security layer whether outbound HTTP is permitted."""
    if security is None:
        return True, ""
    cfg = getattr(security, "cfg", None)
    allow = getattr(cfg, "allow_network", None) if cfg is not None else None
    if allow is None:
        allow = getattr(security, "allow_network", True)
    if not bool(allow):
        return False, "network access is disabled by policy"
    return True, ""


def _validate_url(url: str) -> Tuple[bool, str]:
    """Accept only ``http`` / ``https`` schemes with a host."""
    if not isinstance(url, str) or not url.strip():
        return False, "url is required"
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"scheme {scheme!r} is not permitted (only http/https)"
    if not parsed.netloc:
        return False, "url has no host"
    return True, ""


def _build_headers(extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    if extra:
        for key, value in extra.items():
            if key is None or value is None:
                continue
            headers[str(key)] = str(value)
    return headers


def _read_bounded(response: Any, max_bytes: int) -> Tuple[bytes, bool]:
    """Read *response* up to ``max_bytes``.

    Returns ``(body, truncated)`` where ``truncated`` is True only when a
    subsequent byte was available beyond the cap.  A body of exactly
    ``max_bytes`` is NOT truncated.
    """
    if max_bytes <= 0:
        return response.read(), False
    limit = int(max_bytes)
    chunk = response.read(limit)
    peek = response.read(1)
    if peek:
        return chunk, True
    return chunk, False


def _do_open(request: urllib.request.Request, timeout: float):
    """Open *request* with a bounded redirect count."""
    opener = urllib.request.build_opener(
        _BoundedRedirectHandler(max_redirects=_MAX_REDIRECTS)
    )
    return opener.open(request, timeout=timeout)


class _BoundedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A HTTPRedirectHandler that stops after ``max_redirects`` hops."""

    def __init__(self, max_redirects: int = 5) -> None:
        self.max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        # Count redirects that have already happened on this Request object.
        seen = int(getattr(req, "_jarvis_redirects", 0))
        if seen >= self.max_redirects:
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                f"exceeded {self.max_redirects} redirect(s)",
                headers,
                fp,
            )
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new._jarvis_redirects = seen + 1  # type: ignore[attr-defined]
        return new


# --------------------------------------------------------------------------- #
#  HTML extraction — a small, dependency-free stripper
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Collect visible text from an HTML document.

    Skips script/style/nav/header/footer/noscript spans, drops all attributes,
    decodes character entities via :mod:`html`, and inserts a newline for
    block-level tags so the output can be split back into paragraphs.
    """

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg"})
    _BLOCKY = frozenset(
        {
            "p", "div", "section", "article", "header", "footer", "nav",
            "aside", "main", "li", "ul", "ol", "table", "tr", "td", "th",
            "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
            "blockquote", "form", "figure", "figcaption",
        }
    )
    _INVISIBLE = frozenset({"nav", "header", "footer", "aside", "form"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP or tag in self._INVISIBLE:
            self._skip_stack.append(tag)
            return
        if tag in self._BLOCKY:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            return
        if tag in self._BLOCKY:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in ("br", "hr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        joined = "".join(self._parts)
        # Collapse runs of whitespace inside a line; keep paragraph breaks.
        lines: List[str] = []
        for raw in joined.splitlines():
            clean = re.sub(r"[\t  ]+", " ", raw).strip()
            if clean:
                lines.append(clean)
        return "\n".join(lines)


def extract_text_from_html(source: str) -> str:
    """Turn an HTML document into readable plain text."""
    if not source:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(source)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not raise
        pass
    return parser.get_text()


# --------------------------------------------------------------------------- #
#  Core HTTP entry points
# --------------------------------------------------------------------------- #
def _decode_body(body: bytes, response: Any) -> Tuple[str, str]:
    """Decode *body* using the response's declared charset (fallback utf-8)."""
    encoding = "utf-8"
    headers = getattr(response, "headers", None)
    if headers is not None:
        get_content_charset = getattr(headers, "get_content_charset", None)
        if callable(get_content_charset):
            try:
                found = get_content_charset()
                if found:
                    encoding = str(found)
            except Exception:  # noqa: BLE001
                pass
    try:
        return body.decode(encoding, errors="replace"), encoding
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace"), "utf-8"


def _headers_to_dict(response: Any) -> Dict[str, str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k): str(v) for k, v in headers.items()}
    except Exception:  # noqa: BLE001
        return {}


def _make_http_get(security: Any):
    def _http_get(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 20.0,
        max_bytes: int = 500000,
    ) -> ToolResult:
        """HTTP GET.  Body capped at ``max_bytes``; redirects capped at 5."""
        allowed, reason = _network_allowed(security)
        if not allowed:
            return ToolResult.failure(reason)
        ok, err = _validate_url(url)
        if not ok:
            return ToolResult.failure(err)
        request = urllib.request.Request(
            url,
            headers=_build_headers(headers),
            method="GET",
        )
        try:
            with _do_open(request, float(timeout)) as response:
                body, truncated = _read_bounded(response, int(max_bytes))
                status = getattr(response, "status", None) or response.getcode()
                text, encoding = _decode_body(body, response)
                hdrs = _headers_to_dict(response)
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            return ToolResult.failure(
                f"HTTP {exc.code} {exc.reason} for {url}"
            )
        except urllib.error.URLError as exc:
            return ToolResult.failure(f"URL error: {exc.reason}")
        except (socket.timeout, TimeoutError):
            return ToolResult.failure(f"request timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"network error: {exc}")
        return ToolResult.success(
            output={
                "url": final_url,
                "status": int(status),
                "encoding": encoding,
                "headers": hdrs,
                "text": text,
                "bytes_read": len(body),
                "truncated": truncated,
            }
        )

    return _http_get


def _make_http_post_json(security: Any):
    def _http_post_json(
        url: str,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 20.0,
        max_bytes: int = 500000,
    ) -> ToolResult:
        """HTTP POST with a JSON body."""
        allowed, reason = _network_allowed(security)
        if not allowed:
            return ToolResult.failure(reason)
        ok, err = _validate_url(url)
        if not ok:
            return ToolResult.failure(err)
        try:
            payload = json.dumps({} if data is None else data).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return ToolResult.failure(f"payload is not JSON-serialisable: {exc}")
        merged = _build_headers(headers)
        merged.setdefault("Content-Type", "application/json; charset=utf-8")
        merged["Content-Length"] = str(len(payload))
        request = urllib.request.Request(
            url, data=payload, headers=merged, method="POST"
        )
        try:
            with _do_open(request, float(timeout)) as response:
                body, truncated = _read_bounded(response, int(max_bytes))
                status = getattr(response, "status", None) or response.getcode()
                text, encoding = _decode_body(body, response)
                hdrs = _headers_to_dict(response)
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            return ToolResult.failure(
                f"HTTP {exc.code} {exc.reason} for {url}"
            )
        except urllib.error.URLError as exc:
            return ToolResult.failure(f"URL error: {exc.reason}")
        except (socket.timeout, TimeoutError):
            return ToolResult.failure(f"request timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"network error: {exc}")

        parsed_json: Any = None
        try:
            parsed_json = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed_json = None

        return ToolResult.success(
            output={
                "url": final_url,
                "status": int(status),
                "encoding": encoding,
                "headers": hdrs,
                "text": text,
                "json": parsed_json,
                "bytes_read": len(body),
                "truncated": truncated,
            }
        )

    return _http_post_json


# --------------------------------------------------------------------------- #
#  Higher-level readers
# --------------------------------------------------------------------------- #
def _make_fetch_page_text(security: Any):
    getter = _make_http_get(security)

    def _fetch_page_text(url: str, timeout: float = 20.0) -> ToolResult:
        """Fetch a URL and return the visible text (script/style stripped)."""
        result = getter(url=url, timeout=timeout, max_bytes=1_000_000)
        if not result.ok:
            return result
        data = result.output or {}
        text = extract_text_from_html(data.get("text") or "")
        return ToolResult.success(
            output={
                "url": data.get("url"),
                "status": data.get("status"),
                "text": text,
                "chars": len(text),
                "truncated": data.get("truncated"),
            }
        )

    return _fetch_page_text


_DDG_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _clean_html_snippet(snippet: str) -> str:
    text = re.sub(r"<[^>]+>", "", snippet)
    return html.unescape(text).strip()


def _resolve_ddg_href(href: str) -> str:
    """DuckDuckGo wraps outbound links in ``/l/?uddg=<encoded>``.  Unwrap it."""
    try:
        parsed = urllib.parse.urlparse(href)
    except ValueError:
        return href
    if parsed.path.endswith("/l/") or "uddg" in parsed.query:
        qs = urllib.parse.parse_qs(parsed.query)
        real = qs.get("uddg") or qs.get("u")
        if real:
            return urllib.parse.unquote(real[0])
    return href


def _make_web_search(security: Any):
    getter = _make_http_get(security)

    def _web_search(query: str, max_results: int = 5) -> ToolResult:
        """DuckDuckGo HTML search — best effort, no JS."""
        if not query or not str(query).strip():
            return ToolResult.failure("query is required")
        try:
            cap = max(1, int(max_results))
        except (TypeError, ValueError):
            cap = 5
        params = urllib.parse.urlencode({"q": str(query)})
        url = f"https://duckduckgo.com/html/?{params}"
        result = getter(url=url, timeout=15.0, max_bytes=500000)
        if not result.ok:
            return result
        body = (result.output or {}).get("text") or ""
        links = _DDG_LINK_RE.findall(body)
        snippets = _DDG_SNIPPET_RE.findall(body)
        if not links:
            return ToolResult.failure("could not parse DuckDuckGo results")
        items: List[Dict[str, Any]] = []
        for i, (href, title_html) in enumerate(links[:cap]):
            snippet = snippets[i] if i < len(snippets) else ""
            items.append(
                {
                    "title": _clean_html_snippet(title_html),
                    "url": _resolve_ddg_href(html.unescape(href)),
                    "snippet": _clean_html_snippet(snippet),
                }
            )
        return ToolResult.success(
            output={"query": str(query), "results": items, "count": len(items)}
        )

    return _web_search


# --------------------------------------------------------------------------- #
#  Download / open / connectivity
# --------------------------------------------------------------------------- #
def _make_download_file(security: Any):
    def _download_file(url: str, dest: str, max_mb: int = 200) -> ToolResult:
        """Stream a URL to *dest*; abort and remove the file if it exceeds the cap."""
        allowed, reason = _network_allowed(security)
        if not allowed:
            return ToolResult.failure(reason)
        ok, err = _validate_url(url)
        if not ok:
            return ToolResult.failure(err)
        if not dest or (isinstance(dest, str) and not dest.strip()):
            return ToolResult.failure("dest is required")
        if isinstance(dest, str) and is_drive_relative(dest):
            return ToolResult.failure(
                f"refusing drive-relative dest {dest!r}"
            )
        try:
            resolved = resolve_path(str(dest))
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if is_filesystem_root(resolved):
            return ToolResult.failure(
                f"refusing to write to a filesystem root: {resolved}"
            )
        try:
            cap_mb = float(max_mb)
        except (TypeError, ValueError):
            cap_mb = 200.0
        cap_bytes = int(max(1.0, cap_mb) * 1024 * 1024)

        request = urllib.request.Request(url, headers=_build_headers(None), method="GET")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult.failure(f"cannot create destination: {exc}")

        try:
            response = _do_open(request, 60.0)
        except urllib.error.HTTPError as exc:
            return ToolResult.failure(f"HTTP {exc.code} {exc.reason} for {url}")
        except urllib.error.URLError as exc:
            return ToolResult.failure(f"URL error: {exc.reason}")
        except (socket.timeout, TimeoutError):
            return ToolResult.failure("download connection timed out")
        except OSError as exc:
            return ToolResult.failure(f"network error: {exc}")

        written = 0
        chunk_size = 65536
        try:
            with response, open(resolved, "wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    if written + len(chunk) > cap_bytes:
                        raise _DownloadTooBig(cap_bytes)
                    fh.write(chunk)
                    written += len(chunk)
        except _DownloadTooBig as exc:
            _safe_unlink(resolved)
            return ToolResult.failure(
                f"download exceeded {exc.cap} bytes; partial file removed"
            )
        except (OSError, socket.timeout, TimeoutError) as exc:
            _safe_unlink(resolved)
            return ToolResult.failure(f"download failed: {exc}")
        return ToolResult.success(
            output={
                "url": url,
                "path": str(resolved),
                "bytes_written": written,
            }
        )

    return _download_file


class _DownloadTooBig(Exception):
    def __init__(self, cap: int) -> None:
        super().__init__(f"cap={cap}")
        self.cap = cap


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass


def _open_url(url: str) -> ToolResult:
    """Open ``url`` with the OS default browser."""
    ok, err = _validate_url(url)
    if not ok:
        return ToolResult.failure(err)
    result = open_path(url)
    if not result.ok:
        return ToolResult.failure(
            f"open_url failed: {safe_truncate(result.stderr or '', 300)}"
        )
    return ToolResult.success(output={"url": url})


def _check_internet(timeout: float = 3.0) -> ToolResult:
    """TCP-connect to a well-known DNS server to prove we have connectivity."""
    try:
        wait = float(timeout)
    except (TypeError, ValueError):
        wait = 3.0
    targets = (("1.1.1.1", 53), ("8.8.8.8", 53))
    last_error = ""
    for host, port in targets:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(wait)
        try:
            sock.connect((host, port))
            sock.close()
            return ToolResult.success(output={"online": True, "target": f"{host}:{port}"})
        except OSError as exc:
            last_error = f"{host}:{port}: {exc}"
        finally:
            try:
                sock.close()
            except OSError:
                pass
    return ToolResult.success(
        output={"online": False, "error": last_error}
    )


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    """Return the built-in web tools bound to *ctx*."""
    security = getattr(ctx, "security", None)
    tools: List[Tool] = [
        FunctionTool(
            _make_http_get(security),
            name="http_get",
            description="HTTP GET a URL; capped body, capped redirects.",
        ),
        FunctionTool(
            _make_http_post_json(security),
            name="http_post_json",
            description="HTTP POST a JSON body.",
        ),
        FunctionTool(
            _make_fetch_page_text(security),
            name="fetch_page_text",
            description="Fetch a URL and return the visible text.",
        ),
        FunctionTool(
            _make_web_search(security),
            name="web_search",
            description="Simple web search via DuckDuckGo HTML endpoint.",
        ),
        FunctionTool(
            _make_download_file(security),
            name="download_file",
            description="Stream a URL to a local path (size-capped).",
            dangerous=True,
        ),
        FunctionTool(
            _open_url,
            name="open_url",
            description="Open a URL in the OS default browser.",
        ),
        FunctionTool(
            _check_internet,
            name="check_internet",
            description="Quick TCP probe to verify internet connectivity.",
        ),
    ]
    return tools


__all__ = [
    "build_tools",
    "extract_text_from_html",
]
