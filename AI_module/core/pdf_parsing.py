"""
PDF layout parsing with PyMuPDF: blocks → lines → spans, font-size heuristics,
bbox-based vertical gaps for paragraph/title boundaries, and chapter tree extraction
(:class:`ChapterSegment`). Does not build embedding chunks; see :mod:`chunking`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Numbered outline: "1 Title", "1.2 Subtitle", "2.3.1 X"
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*((?:\d+\.)*\d+)\s*[.\s]+\s*(.+)$",
)

_FONT_EPS = 0.01
_TITLE_MERGE_MAX_LINES = 4
_VERTICAL_GAP_PT = 6.0
# Bbox vertical gap (PDF points): paragraph / title separation scales with modal body font size.
_BODY_GAP_SIZE_MULT = 0.7
_TITLE_MIN_DELTA_PT = 0.5
# Band from page top/bottom (pt) for “starts new page” / “ends page” title heuristics.
_PAGE_EDGE_BAND_PT = 56.0


def collapse_line_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t]+", " ", text.strip())


def _normalize_heading_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_numbered_heading(first_line: str) -> tuple[tuple[int, ...], str] | None:
    """
    Parse '1.2.3 Some title' -> ((1, 2, 3), 'Some title').
    """
    m = _NUMBERED_HEADING_RE.match(first_line.strip())
    if not m:
        return None
    num_str, rest = m.group(1), m.group(2).strip()
    try:
        parts = tuple(int(x) for x in num_str.split("."))
    except ValueError:
        return None
    return parts, _normalize_heading_title(rest)


def format_numbered_heading(parts: tuple[int, ...], title: str) -> str:
    """Format numbered heading for path/payload: (1, 2) + 'Chapter' -> '1.2. Chapter'."""
    num = ".".join(str(p) for p in parts)
    t = _normalize_heading_title(title)
    if not t:
        return f"{num}."
    return f"{num}. {t}"


class EmptyPdfError(Exception):
    """Raised when a PDF contains no extractable text (e.g. image-only or empty)."""

    def __init__(self, message: str, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)
        super().__init__(message)


def _mode_font_size(sizes: list[float]) -> float:
    if not sizes:
        return 10.0
    rounded = [round(s, 2) for s in sizes]
    return Counter(rounded).most_common(1)[0][0]


@dataclass
class RawBlock:
    page: int
    text: str
    max_font_size: float
    first_line: str


def _significant_vertical_gap_pt(dy: float, body_size: float) -> bool:
    """
    True if vertical distance ``dy`` (top of lower line minus bottom of upper line, in pt)
    is large enough for a new paragraph or for a title line after body text.
    """
    if dy <= 0:
        return False
    threshold = max(_VERTICAL_GAP_PT, body_size * _BODY_GAP_SIZE_MULT)
    return dy >= threshold


def is_significant_vertical_gap(dy: float, body_size: float) -> bool:
    """Public wrapper for tests and callers that need the bbox gap rule explicitly."""
    return _significant_vertical_gap_pt(dy, body_size)


@dataclass
class PdfSpanRec:
    text: str
    size: float
    bbox: tuple[float, float, float, float]


@dataclass
class PdfLineRec:
    page: int
    spans: list[PdfSpanRec]
    bbox: tuple[float, float, float, float]
    page_height: float

    @property
    def line_text(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class PdfBlockRec:
    page: int
    block_index: int
    bbox: tuple[float, float, float, float]
    lines: list[PdfLineRec]


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _span_from_pymupdf(span: dict[str, Any]) -> PdfSpanRec | None:
    text = span.get("text") or ""
    sz = span.get("size")
    bb = span.get("bbox")
    if sz is None or not bb or len(bb) < 4:
        return None
    return PdfSpanRec(
        text=str(text),
        size=float(sz),
        bbox=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
    )


def _line_from_pymupdf(page_one: int, page_height: float, line: dict[str, Any]) -> PdfLineRec | None:
    spans: list[PdfSpanRec] = []
    for sp in line.get("spans", []):
        rec = _span_from_pymupdf(sp)
        if rec is not None:
            spans.append(rec)
    if not spans:
        return None
    return PdfLineRec(
        page=page_one,
        spans=spans,
        bbox=_bbox_union([s.bbox for s in spans]),
        page_height=float(page_height),
    )


def _block_from_pymupdf(page_one: int, page_height: float, block_index: int, block: dict[str, Any]) -> PdfBlockRec | None:
    bb = block.get("bbox")
    if not bb or len(bb) < 4:
        bbox = (0.0, 0.0, 0.0, 0.0)
    else:
        bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    lines: list[PdfLineRec] = []
    for line in block.get("lines", []):
        ln = _line_from_pymupdf(page_one, page_height, line)
        if ln is not None:
            lines.append(ln)
    if not lines:
        return None
    return PdfBlockRec(page=page_one, block_index=block_index, bbox=bbox, lines=lines)


def _line_max_span_size(line: PdfLineRec) -> float:
    return max((s.size for s in line.spans), default=0.0)


def every_span_above_body_font(line: PdfLineRec, body_size: float) -> bool:
    """
    True if every span that contributes non-whitespace text uses a font size strictly
    larger than the modal body size (same rule for all spans on the line).
    """
    threshold = body_size + _FONT_EPS
    saw_content = False
    for s in line.spans:
        if not (s.text or "").strip():
            continue
        saw_content = True
        if s.size <= threshold:
            return False
    return saw_content


def _line_looks_title_sized(line: PdfLineRec, body_size: float) -> bool:
    """
    Potential title rule: max span size is greater than modal body by ``_TITLE_MIN_DELTA_PT``.
    """
    return _line_max_span_size(line) > body_size + _TITLE_MIN_DELTA_PT


def _line_is_body_sized(line: PdfLineRec, body_size: float) -> bool:
    """Complement of title-sized: max span not above body + title delta."""
    return _line_max_span_size(line) <= body_size + _TITLE_MIN_DELTA_PT + _FONT_EPS


def _line_near_page_top(line: PdfLineRec, margin_pt: float | None = None) -> bool:
    m = _PAGE_EDGE_BAND_PT if margin_pt is None else margin_pt
    return line.bbox[1] <= m


def _line_near_page_bottom(line: PdfLineRec, margin_pt: float | None = None) -> bool:
    m = _PAGE_EDGE_BAND_PT if margin_pt is None else margin_pt
    return (line.page_height - line.bbox[3]) <= m


def _heading_starts_upper_or_digit(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    c = t[0]
    return bool(c.isdigit() or (c.isalpha() and c.isupper()))


@dataclass
class ChapterSegment:
    """Accumulated body for one leaf in the chapter tree."""

    path_titles: list[str]  # [doc, ch1, ch2, ...]
    leaf_title: str
    parent_title: str
    page_start: int = 0
    page_end: int = 0
    body_parts: list[str] = field(default_factory=list)

    def append_body(self, text: str, page: int, *, paragraph_break_before: bool = False) -> None:
        t = text.strip()
        if not t:
            return
        if self.body_parts and paragraph_break_before:
            # "\n".join(["a", "", "b"]) -> "a\n\nb" for split_into_paragraphs
            self.body_parts.append("")
        self.body_parts.append(t)
        if self.page_start == 0:
            self.page_start = page
            self.page_end = page
        else:
            self.page_start = min(self.page_start, page)
            self.page_end = max(self.page_end, page)

    @property
    def body(self) -> str:
        # Single newlines between PyMuPDF blocks preserve line flow without treating
        # every block as a separate paragraph; true breaks use \n\n, tabs, or pilcrow.
        return "\n".join(self.body_parts)


def extract_pdf_layout(path: Path, is_multicolumn: bool) -> tuple[list[PdfBlockRec], list[float]]:
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PDF layout extraction requires PyMuPDF. Install with: pip install pymupdf") from e

    all_sizes: list[float] = []
    blocks_out: list[PdfBlockRec] = []
    block_index = 0
    with fitz.open(path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = page.get_text("dict", sort=True) if is_multicolumn else page.get_text("dict")
            page_one = page_num + 1
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                rec = _block_from_pymupdf(page_one, float(page.rect.height), block_index, block)
                if rec is None:
                    continue
                blocks_out.append(rec)
                for ln in rec.lines:
                    for sp in ln.spans:
                        all_sizes.append(sp.size)
                block_index += 1
    return blocks_out, all_sizes


def extract_raw_blocks(path: Path, is_multicolumn: bool) -> tuple[list[RawBlock], list[float]]:
    """Flatten layout blocks to :class:`RawBlock` (for tests / diagnostics)."""
    layout, all_sizes = extract_pdf_layout(path, is_multicolumn)
    blocks: list[RawBlock] = []
    for b in layout:
        line_strs = [ln.line_text for ln in b.lines]
        block_text = "\n".join(line_strs)
        if not block_text.strip():
            continue
        first = block_text.split("\n", 1)[0].strip()
        block_max = max((sp.size for ln in b.lines for sp in ln.spans), default=0.0)
        blocks.append(
            RawBlock(
                page=b.page,
                text=block_text,
                max_font_size=block_max,
                first_line=first,
            )
        )
    return blocks, all_sizes


def font_title_gap_ok(
    first_title_line: PdfLineRec,
    last_body_line: PdfLineRec | None,
    last_emitted_was_body: bool,
    body_size: float,
) -> bool:
    """
    Require a bbox-significant vertical gap only after body text (not between headings).
    On page transitions, use effective gap: bottom whitespace of previous page
    + top whitespace on next page.

    Exception (title starting at top of next page): if the previous line was body-sized
    on the prior page and the title line sits near the top of the following page, the
    page break substitutes for a large visual gap (same checks otherwise).
    """
    if not last_emitted_was_body or last_body_line is None:
        return True
    # Case 2: title at page start after body-only previous page — page transition replaces gap.
    if (
        first_title_line.page == last_body_line.page + 1
        and _line_is_body_sized(last_body_line, body_size)
        and _line_near_page_top(first_title_line)
    ):
        return True
    if first_title_line.page != last_body_line.page:
        bottom_ws = max(0.0, last_body_line.page_height - last_body_line.bbox[3])
        top_ws = max(0.0, first_title_line.bbox[1])
        dy = bottom_ws + top_ws
    else:
        dy = first_title_line.bbox[1] - last_body_line.bbox[3]
    return _significant_vertical_gap_pt(dy, body_size)


def paragraph_break_before_line(
    line: PdfLineRec,
    prev_line: PdfLineRec | None,
    body_size: float,
) -> bool:
    """
    Use bbox vertical distance to insert an extra blank line (paragraph break) in body.
    On page transitions, use effective gap: bottom whitespace of previous page
    + top whitespace on next page.

    Consecutive pages with **body-sized** lines on both sides always merge into one
    paragraph (no break), regardless of whitespace—same spirit as a title split across
    pages (case 3).
    """
    if prev_line is None:
        return False
    if (
        line.page == prev_line.page + 1
        and _line_is_body_sized(prev_line, body_size)
        and _line_is_body_sized(line, body_size)
    ):
        return False
    if line.page != prev_line.page:
        bottom_ws = max(0.0, prev_line.page_height - prev_line.bbox[3])
        top_ws = max(0.0, line.bbox[1])
        dy = bottom_ws + top_ws
    else:
        dy = line.bbox[1] - prev_line.bbox[3]
    return _significant_vertical_gap_pt(dy, body_size)


def outline_push_numbered(
    outline: list[tuple[int, str, tuple[int, ...] | None]],
    parts: tuple[int, ...],
    title: str,
    heading_tier: int,
) -> None:
    """
    Insert a numbered heading: structure among numbered headings follows **digits** only
    (pop trailing unnumbered, then pop numbered until ``parent_key == parts[:-1]``).
    """
    parent_key = parts[:-1]
    while outline and outline[-1][2] is None:
        outline.pop()
    while outline and outline[-1][2] is not None and outline[-1][2] != parent_key:
        outline.pop()
    outline.append((heading_tier, title, parts))


def outline_push_unnumbered_by_font(
    outline: list[tuple[int, str, tuple[int, ...] | None]],
    title: str,
    tier: int,
) -> None:
    """
    Insert an unnumbered heading: placement vs the whole outline uses **font tier only**
    (same rule as a font-only document; numbered entries compare by stored tier).
    """
    while outline and outline[-1][0] >= tier:
        outline.pop()
    outline.append((tier, title, None))


def append_merged_lines_as_body(
    current: ChapterSegment,
    merged: list[PdfLineRec],
    body_size: float,
) -> None:
    prev: PdfLineRec | None = None
    for ln in merged:
        t = collapse_line_whitespace(ln.line_text)
        if not t:
            prev = ln
            continue
        para = paragraph_break_before_line(ln, prev, body_size)
        current.append_body(t, ln.page, paragraph_break_before=para)
        prev = ln


def extract_chapter_segments(
    path: Path,
    doc_name: str,
    *,
    is_multicolumn: bool,
    title_max_tokens: int,
    token_count: Callable[[str], int],
) -> list[ChapterSegment]:
    """
    Build chapter segments from PDF layout (font hierarchy, numbered headings, bbox gaps).

    **Outline model** (single stack of ``(font_tier, title, num_key | None)``):

    - Among **numbered** headings, hierarchy follows **numeric keys** only (1, 1.1, 2, …).
    - **Unnumbered** headings and **cross-type** placement (numbered vs unnumbered) use
      **font tier** only—the same tier scale as if the document were font-only: pop while
      ``top.tier >= new_tier``, then push. Numbered entries participate in that comparison
      by their stored tier, not by their digit structure.

    **Page / title flow**

    - Consecutive pages: **body-sized** line at end of one page and **body-sized** line at
      start of the next always merge into one paragraph (no automatic break), like a title
      split across pages.
    - Title at the **top** of the next page after a **body-sized** line on the previous page
      may use the page break in place of a large visual gap before the title.
    - A title may **span** two pages when the first fragment is near the page bottom and the
      next is near the top of the following page; all fragments must pass title font rules.
    """
    layout, all_sizes = extract_pdf_layout(path, is_multicolumn)
    if not layout or not all_sizes:
        return []

    body_size = _mode_font_size(all_sizes)
    title_sizes_desc = sorted(
        {round(s, 2) for s in all_sizes if s > body_size + _TITLE_MIN_DELTA_PT},
        reverse=True,
    )

    def tier_from_font_size(size: float) -> int:
        """
        Tier index by document title-size ladder (> body + 1pt), largest first.
        Sizes in (body+0.5, body+1] still qualify as potential titles but map to the
        deepest tier.
        """
        if not title_sizes_desc:
            return 0
        sr = round(size, 2)
        for idx, sz in enumerate(title_sizes_desc):
            if sr >= sz - _FONT_EPS:
                return idx
        return len(title_sizes_desc)

    flat: list[tuple[PdfBlockRec, PdfLineRec]] = []
    for bl in layout:
        for ln in bl.lines:
            flat.append((bl, ln))

    # (font_tier, display_title, numbered_parts_or_none). num_key None = unnumbered heading.
    outline: list[tuple[int, str, tuple[int, ...] | None]] = []

    def outline_path() -> list[str]:
        return [doc_name] + [entry[1] for entry in outline]

    segments: list[ChapterSegment] = []
    current: ChapterSegment | None = None
    last_emitted_was_body = False
    last_body_line_rec: PdfLineRec | None = None

    def flush() -> None:
        nonlocal current
        if current and current.body.strip():
            segments.append(current)
        current = None

    def start_segment(path_titles: list[str]) -> None:
        nonlocal current, last_body_line_rec
        flush()
        if len(path_titles) < 2:
            leaf = path_titles[0]
            parent = ""
        else:
            leaf = path_titles[-1]
            parent = path_titles[-2]
        current = ChapterSegment(
            path_titles=list(path_titles),
            leaf_title=leaf,
            parent_title=parent,
        )
        last_body_line_rec = None

    start_segment([doc_name])

    i = 0
    while i < len(flat):
        bl0, line0 = flat[i]
        lt0 = line0.line_text
        if not (lt0 or "").strip():
            i += 1
            continue

        first_col = collapse_line_whitespace(lt0)
        numbered = parse_numbered_heading(first_col)
        if numbered:
            parts, title = numbered
            max_sz = max((s.size for s in line0.spans), default=body_size)
            # Numbered text is still a title only if it matches the font-size title rule.
            if max_sz <= body_size + _TITLE_MIN_DELTA_PT:
                numbered = None
            else:
                has_numbered_already = any(entry[2] is not None for entry in outline)
                if has_numbered_already:
                    # With existing numbered hierarchy, use numeric depth for tier.
                    heading_tier = max(0, len(parts) - 1)
                else:
                    # First numbered heading uses the same font-size tiering as unnumbered titles.
                    heading_tier = tier_from_font_size(max_sz)
                title_display = format_numbered_heading(parts, title)
                outline_push_numbered(outline, parts, title_display, heading_tier)
                start_segment(outline_path())
                last_emitted_was_body = False
                i += 1
                continue

        if not _line_looks_title_sized(line0, body_size):
            if current is None:
                start_segment([doc_name])
            assert current is not None
            prev_body = (
                last_body_line_rec
                if last_emitted_was_body and current.body_parts
                else None
            )
            para = paragraph_break_before_line(line0, prev_body, body_size)
            current.append_body(
                collapse_line_whitespace(lt0),
                line0.page,
                paragraph_break_before=para,
            )
            last_emitted_was_body = True
            last_body_line_rec = line0
            i += 1
            continue

        merged_lines: list[PdfLineRec] = [line0]
        j = i + 1
        cur_block = bl0
        cur_page = line0.page
        while len(merged_lines) < _TITLE_MERGE_MAX_LINES and j < len(flat):
            blj, lnj = flat[j]
            if not lnj.line_text.strip():
                break
            last_ln = merged_lines[-1]
            same_block_page = blj.block_index == cur_block.block_index and blj.page == cur_page
            cross_page_title = (
                lnj.page == last_ln.page + 1
                and _line_near_page_bottom(last_ln)
                and _line_near_page_top(lnj)
                and _line_looks_title_sized(lnj, body_size)
            )
            if same_block_page:
                if not _line_looks_title_sized(lnj, body_size):
                    break
                merged_lines.append(lnj)
                j += 1
                continue
            if cross_page_title:
                merged_lines.append(lnj)
                cur_block = blj
                cur_page = lnj.page
                j += 1
                continue
            break

        title_full = "\n".join(
            collapse_line_whitespace(x.line_text) for x in merged_lines if x.line_text.strip()
        )

        gap_ok = font_title_gap_ok(
            merged_lines[0],
            last_body_line_rec,
            last_emitted_was_body,
            body_size,
        )
        tok_ok = token_count(title_full) <= title_max_tokens
        start_ok = _heading_starts_upper_or_digit(first_col)

        max_sz = max(s.size for ln in merged_lines for s in ln.spans)
        tier = tier_from_font_size(max_sz)

        if gap_ok and start_ok and tok_ok:
            title_display = _normalize_heading_title(title_full.replace("\n", " "))
            outline_push_unnumbered_by_font(outline, title_display, tier)
            start_segment(outline_path())
            last_emitted_was_body = False
        else:
            if current is None:
                start_segment([doc_name])
            assert current is not None
            append_merged_lines_as_body(current, merged_lines, body_size)
            last_emitted_was_body = True
            last_body_line_rec = merged_lines[-1]

        i = j

    flush()
    return [s for s in segments if s.body.strip()]
