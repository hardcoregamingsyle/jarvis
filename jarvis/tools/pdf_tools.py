"""Write a real PDF report to disk. Zero dependencies.

Not a text file renamed ``.pdf`` -- this emits actual PDF syntax (objects,
a content stream per page, an xref table, a trailer) using one of the 14
standard fonts (Helvetica) that every PDF viewer can render without the font
being embedded. No ``reportlab``, no ``fpdf``, nothing to install: the whole
point is that "write a report" never again has nothing to fulfil it, on any
machine, offline, the moment JARVIS boots.

Markup understood in the body text (deliberately minimal, matching what a
model naturally produces for a report):
    # Heading        -> a larger, bold line
    ## Subheading    -> a smaller bold line
    blank line       -> paragraph break
    anything else    -> word-wrapped body text

Word-wrap uses Helvetica's real per-character advance widths (the AFM metrics
for the standard 14 fonts are public and fixed -- no font file needed to know
them), not a fixed-width guess, so wrapping is accurate rather than
approximate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..core.contracts import Tool, ToolResult
from ..core.platform_utils import is_filesystem_root, resolve_path
from .registry import FunctionTool

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helvetica metrics (Adobe's published AFM widths, per 1000 em) -- exact,
#  not approximated, for the 95 printable ASCII characters. Bold uses the
#  Helvetica-Bold AFM. Anything outside this table (e.g. an em-dash a model
#  likes to use) falls back to the space-width average rather than crashing.
# --------------------------------------------------------------------------- #
_HELV_WIDTHS = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667,
    "'": 191, '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333,
    '.': 278, '/': 278, '0': 556, '1': 556, '2': 556, '3': 556, '4': 556,
    '5': 556, '6': 556, '7': 556, '8': 556, '9': 556, ':': 278, ';': 278,
    '<': 584, '=': 584, '>': 584, '?': 556, '@': 1015, 'A': 667, 'B': 667,
    'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778, 'H': 722, 'I': 278,
    'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722, 'O': 778, 'P': 667,
    'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 278, '\\': 278, ']': 278, '^': 469,
    '_': 556, '`': 333, 'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556,
    'f': 278, 'g': 556, 'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222,
    'm': 833, 'n': 556, 'o': 556, 'p': 556, 'q': 556, 'r': 333, 's': 500,
    't': 278, 'u': 556, 'v': 500, 'w': 722, 'x': 500, 'y': 500, 'z': 500,
    '{': 334, '|': 260, '}': 334, '~': 584,
}
_HELVB_WIDTHS = dict(_HELV_WIDTHS, **{
    'A': 722, 'B': 722, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 556, 'K': 722, 'L': 611, 'M': 833, 'N': 722,
    'O': 778, 'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722,
    'V': 667, 'W': 944, 'X': 667, 'Y': 667, 'Z': 611, 'a': 556, 'b': 611,
    'c': 556, 'd': 611, 'e': 556, 'f': 333, 'g': 611, 'h': 611, 'i': 278,
    'j': 278, 'k': 556, 'l': 278, 'm': 889, 'n': 611, 'o': 611, 'p': 611,
    'q': 611, 'r': 389, 's': 556, 't': 333, 'u': 611, 'v': 556, 'w': 778,
    'x': 556, 'y': 556, 'z': 500, ',': 278, '.': 278, ':': 333, ';': 333,
})
_DEFAULT_WIDTH = 556   # used for any character not in either table above

# Page geometry: US Letter, 1 inch margins (72 points/inch).
PAGE_W, PAGE_H = 612.0, 792.0
MARGIN = 72.0
_USABLE_W = PAGE_W - 2 * MARGIN

_FONT_SIZES = {"body": 11.0, "h1": 20.0, "h2": 14.0}
_LEADING = {"body": 15.0, "h1": 26.0, "h2": 19.0}


def _text_width(text: str, size: float, bold: bool) -> float:
    table = _HELVB_WIDTHS if bold else _HELV_WIDTHS
    return sum(table.get(ch, _DEFAULT_WIDTH) for ch in text) * size / 1000.0


def _wrap(text: str, size: float, bold: bool, max_width: float) -> List[str]:
    """Greedy word-wrap using real glyph widths, not a character count guess."""
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if _text_width(candidate, size, bold) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def _escape_pdf_text(text: str) -> str:
    """Escape the three characters that are syntactically special inside a
    PDF string literal, and drop anything outside Latin-1 (the standard 14
    fonts have no way to render it without embedding a Unicode font, and
    silently mangling the byte stream is worse than a visible substitution)."""
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("latin-1", "replace").decode("latin-1")


def _parse_blocks(body: str) -> List[Tuple[str, str]]:
    """-> [(kind, text), ...] where kind is 'h1' | 'h2' | 'body' | 'blank'."""
    blocks: List[Tuple[str, str]] = []
    for raw_line in body.replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            blocks.append(("blank", ""))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
        else:
            blocks.append(("body", line.strip()))
    return blocks


class _PdfWriter:
    """Builds the object graph for a simple multi-page text PDF, then
    serialises it with a correct xref table. One purpose, kept in one place
    rather than spread across the tool function -- the byte-offset
    bookkeeping is the part that is easy to get subtly wrong."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: List[List[str]] = []   # one list of content-stream lines per page
        self._new_page()

    def _new_page(self) -> None:
        self.pages.append([])
        self._y = PAGE_H - MARGIN

    def _ensure_room(self, leading: float) -> None:
        if self._y - leading < MARGIN:
            self._new_page()

    def add_lines(self, lines: List[str], *, size: float, leading: float, bold: bool) -> None:
        font = "/F2" if bold else "/F1"
        for line in lines:
            self._ensure_room(leading)
            self._y -= leading
            if line:
                escaped = _escape_pdf_text(line)
                self.pages[-1].append(
                    f"BT {font} {size:g} Tf {MARGIN:g} {self._y:g} Td ({escaped}) Tj ET"
                )

    def add_gap(self, points: float) -> None:
        self._ensure_room(points)
        self._y -= points

    # -- serialisation -------------------------------------------------- #
    def render(self) -> bytes:
        objects: List[bytes] = []   # 1-indexed by position + 1

        def add_object(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        n_pages = len(self.pages)
        font1_id = add_object(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        )
        font2_id = add_object(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        )

        # Pages tree needs its Kids array before the pages exist, so reserve
        # its object number now and fill the body in once the page ids exist.
        pages_tree_id = len(objects) + 1
        objects.append(b"")   # placeholder

        page_ids: List[int] = []
        for content_lines in self.pages:
            stream = "\n".join(content_lines).encode("latin-1", "replace")
            content_id = add_object(
                b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
            )
            page_id = add_object(
                (
                    "<< /Type /Page /Parent %d 0 R "
                    "/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> "
                    "/Contents %d 0 R >>"
                    % (pages_tree_id, font1_id, font2_id, content_id)
                ).encode("ascii")
            )
            page_ids.append(page_id)

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[pages_tree_id - 1] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} "
            f"/MediaBox [0 0 {PAGE_W:g} {PAGE_H:g}] >>"
        ).encode("ascii")

        catalog_id = add_object(
            f"<< /Type /Catalog /Pages {pages_tree_id} 0 R >>".encode("ascii")
        )
        title_escaped = _escape_pdf_text(self.title)
        info_id = add_object(
            f"<< /Title ({title_escaped}) /Producer (JARVIS) >>".encode("ascii")
        )

        # -- assemble the file, tracking byte offsets for the xref table -- #
        out = bytearray()
        out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        offsets = [0] * (len(objects) + 1)   # index 0 is the free-list head
        for i, body in enumerate(objects, start=1):
            offsets[i] = len(out)
            out += f"{i} 0 obj\n".encode("ascii")
            out += body
            out += b"\nendobj\n"

        xref_offset = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
        out += b"0000000000 65535 f \n"
        for i in range(1, len(objects) + 1):
            out += f"{offsets[i]:010d} 00000 n \n".encode("ascii")
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R "
            f"/Info {info_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
        return bytes(out)


def _default_desktop_path(filename: str) -> Path:
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return Path.home() / "Desktop" / filename


def _make_write_pdf(security: Any):
    def _write_pdf(title: str, body: str, path: str = "") -> ToolResult:
        """Write a real PDF report -- not a text file, actual PDF syntax any
        viewer can open. `body` is plain text; start a line with '# ' for a
        heading or '## ' for a subheading, and use a blank line between
        paragraphs. Long paragraphs are word-wrapped automatically across as
        many pages as needed. Omit `path` (or give just a filename) to save
        to the Desktop; give a full path to save elsewhere.
        """
        if not str(title or "").strip():
            return ToolResult.failure("title is required")
        if not str(body or "").strip():
            return ToolResult.failure("body is required")

        raw_path = str(path or "").strip()
        try:
            if not raw_path or ("/" not in raw_path and "\\" not in raw_path):
                target = _default_desktop_path(raw_path or _slugify(title))
            else:
                if not raw_path.lower().endswith(".pdf"):
                    raw_path += ".pdf"
                target = resolve_path(raw_path)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if is_filesystem_root(target):
            return ToolResult.failure(f"refusing to write to a filesystem root: {target}")
        checker = getattr(security, "is_protected", None)
        if callable(checker):
            try:
                if checker(str(target)):
                    return ToolResult.failure(f"path is protected: {target}")
            except Exception:  # noqa: BLE001
                pass

        writer = _PdfWriter(title)
        writer.add_lines(_wrap(title, _FONT_SIZES["h1"], True, _USABLE_W),
                          size=_FONT_SIZES["h1"], leading=_LEADING["h1"], bold=True)
        writer.add_gap(10)

        for kind, text in _parse_blocks(body):
            if kind == "blank":
                writer.add_gap(_LEADING["body"] * 0.5)
            elif kind == "h1":
                writer.add_gap(6)
                writer.add_lines(_wrap(text, _FONT_SIZES["h1"], True, _USABLE_W),
                                  size=_FONT_SIZES["h1"], leading=_LEADING["h1"], bold=True)
                writer.add_gap(4)
            elif kind == "h2":
                writer.add_gap(4)
                writer.add_lines(_wrap(text, _FONT_SIZES["h2"], True, _USABLE_W),
                                  size=_FONT_SIZES["h2"], leading=_LEADING["h2"], bold=True)
                writer.add_gap(2)
            else:
                writer.add_lines(_wrap(text, _FONT_SIZES["body"], False, _USABLE_W),
                                  size=_FONT_SIZES["body"], leading=_LEADING["body"], bold=False)

        pdf_bytes = writer.render()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pdf_bytes)
        except OSError as exc:
            return ToolResult.failure(f"write failed: {exc}")

        return ToolResult.success(output={
            "path": str(target),
            "bytes_written": len(pdf_bytes),
            "pages": len(writer.pages),
        })

    return _write_pdf


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    return (slug or "report") + ".pdf"


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_tools(ctx: Any) -> List[Tool]:
    security = getattr(ctx, "security", None)
    return [
        FunctionTool(
            _make_write_pdf(security),
            name="write_pdf",
            description=(
                "Write a real PDF report to disk (not a renamed text file). "
                "'# heading' / '## subheading' lines and blank-line paragraph "
                "breaks are recognised; long text wraps and paginates "
                "automatically. Defaults to the Desktop when no path is given."
            ),
        ),
    ]


__all__ = ["build_tools"]
