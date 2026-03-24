"""
Unit tests for pdf_parsing.py: bbox gaps, title/body rules, chapter segments helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest

from AI_module.core.pdf_parsing import (
    ChapterSegment,
    PdfLineRec,
    PdfSpanRec,
    append_merged_lines_as_body,
    every_span_above_body_font,
    font_title_gap_ok,
    is_significant_vertical_gap,
    outline_push_numbered,
    outline_push_unnumbered_by_font,
    paragraph_break_before_line,
    parse_numbered_heading,
)


def _line(
    page: int,
    y0: float,
    y1: float,
    text: str,
    size: float = 12.0,
    page_height: float = 200.0,
) -> PdfLineRec:
    """Horizontal bbox; y0/y1 are PDF-ish bottom/top (PyMuPDF: y increases down)."""
    bbox = (0.0, y0, 100.0, y1)
    return PdfLineRec(
        page=page,
        spans=[PdfSpanRec(text=text, size=size, bbox=bbox)],
        bbox=bbox,
        page_height=page_height,
    )


def test_is_significant_vertical_gap_threshold():
    body = 10.0
    # threshold = max(6, 10 * 0.7) = 7
    assert is_significant_vertical_gap(0.0, body) is False
    assert is_significant_vertical_gap(-1.0, body) is False
    assert is_significant_vertical_gap(6.9, body) is False
    assert is_significant_vertical_gap(7.0, body) is True
    # large body: max(6, 20*0.7)=14
    assert is_significant_vertical_gap(9.0, 20.0) is False
    assert is_significant_vertical_gap(14.0, 20.0) is True


def test_paragraph_break_before_line_cross_page_body_always_merges():
    a = _line(1, 10.0, 20.0, "first")
    b = _line(2, 10.0, 20.0, "second")
    # Large effective gap, but both body-sized on consecutive pages → no paragraph break
    assert paragraph_break_before_line(b, a, body_size=12.0) is False


def test_paragraph_break_before_line_same_page_small_gap():
    a = _line(1, 100.0, 110.0, "first")
    b = _line(1, 112.0, 122.0, "second")  # dy = 2
    assert paragraph_break_before_line(b, a, body_size=12.0) is False


def test_paragraph_break_before_line_same_page_large_gap():
    a = _line(1, 100.0, 110.0, "first")
    b = _line(1, 125.0, 135.0, "second")  # dy = 15 >= 6
    assert paragraph_break_before_line(b, a, body_size=12.0) is True


def test_paragraph_break_before_line_no_prev():
    b = _line(1, 10.0, 20.0, "only")
    assert paragraph_break_before_line(b, None, body_size=12.0) is False


def test_font_title_gap_ok_skips_when_no_prior_body():
    title = _line(1, 200.0, 220.0, "Title", size=18.0)
    assert font_title_gap_ok(title, None, False, 12.0) is True
    assert font_title_gap_ok(title, _line(1, 100.0, 110.0, "x"), False, 12.0) is True


def test_font_title_gap_ok_requires_gap_after_body_same_page():
    body = _line(1, 100.0, 110.0, "body text")
    # title immediately below: dy small
    title_tight = _line(1, 112.0, 130.0, "Title", size=18.0)
    assert font_title_gap_ok(title_tight, body, True, 12.0) is False
    # title with large vertical gap
    title_far = _line(1, 130.0, 150.0, "Title", size=18.0)
    assert font_title_gap_ok(title_far, body, True, 12.0) is True


def test_font_title_gap_ok_different_page_uses_effective_gap():
    body = _line(1, 100.0, 110.0, "body")
    title = _line(2, 10.0, 30.0, "Title", size=18.0)
    assert font_title_gap_ok(title, body, True, 12.0) is True


def test_cross_page_body_continuous_small_gap_title_at_page_top_gets_gap_exception():
    body = _line(1, 190.0, 198.0, "body", page_height=200.0)
    nxt = _line(2, 0.5, 8.0, "next", page_height=200.0)
    title = _line(2, 0.5, 20.0, "Title", size=18.0, page_height=200.0)
    # Body-sized on consecutive pages → always merged (no paragraph break)
    assert paragraph_break_before_line(nxt, body, body_size=12.0) is False
    # Case 2: title at top of page 2 after body-sized line on page 1 — page break replaces gap rule
    assert font_title_gap_ok(title, body, True, 12.0) is True


def test_font_title_gap_ok_cross_page_title_not_at_top_uses_effective_gap():
    body = _line(1, 100.0, 110.0, "body", page_height=200.0)
    # y0=80 > default top band → case 2 does not apply; effective gap is large
    title = _line(2, 80.0, 100.0, "Title", size=18.0, page_height=200.0)
    assert font_title_gap_ok(title, body, True, 12.0) is True


def test_font_title_gap_ok_page_top_exception_skipped_if_prev_not_body_sized():
    """Case 2 applies only when the previous line is body-sized."""
    prev = _line(1, 190.0, 199.5, "big", size=18.0, page_height=200.0)
    title = _line(2, 5.0, 25.0, "Title", size=18.0, page_height=200.0)
    # dy = (200-199.5)+5 = 5.5 < threshold(7 for body 12)
    assert font_title_gap_ok(title, prev, True, body_size=12.0) is False


def test_chapter_segment_append_body_paragraph_break_inserts_blank_part():
    s = ChapterSegment(path_titles=["d", "c"], leaf_title="c", parent_title="d")
    s.append_body("Line one", 1)
    s.append_body("Line two", 1, paragraph_break_before=True)
    assert s.body == "Line one\n\nLine two"


def test_append_merged_lines_as_body_inserts_paragraph_on_bbox_gap():
    seg = ChapterSegment(path_titles=["d"], leaf_title="d", parent_title="")
    merged = [
        _line(1, 50.0, 60.0, "First line", size=11.0),
        _line(1, 80.0, 90.0, "Second line", size=11.0),  # dy=20
    ]
    append_merged_lines_as_body(seg, merged, body_size=11.0)
    assert "\n\n" in seg.body
    assert "First line" in seg.body and "Second line" in seg.body


def test_parse_numbered_heading_basic():
    assert parse_numbered_heading("1.2 Introduction") == ((1, 2), "Introduction")
    assert parse_numbered_heading("3  Methods") == ((3,), "Methods")
    assert parse_numbered_heading("not a heading") is None


def test_outline_numbered_among_themselves_uses_digits_only():
    o: list[tuple[int, str, tuple[int, ...] | None]] = []
    outline_push_numbered(o, (1,), "Chapter One", heading_tier=0)
    outline_push_numbered(o, (1, 1), "Section A", heading_tier=2)
    assert [x[1] for x in o] == ["Chapter One", "Section A"]
    # Top-level "2" has parent (); pops 1.1 and 1 — siblings by numbering, not font.
    outline_push_numbered(o, (2,), "Chapter Two", heading_tier=0)
    assert [x[1] for x in o] == ["Chapter Two"]


def test_outline_unnumbered_uses_font_tier_against_numbered_heads():
    o: list[tuple[int, str, tuple[int, ...] | None]] = []
    outline_push_numbered(o, (1,), "Chapter One", heading_tier=0)
    outline_push_numbered(o, (1, 1), "Section A", heading_tier=2)
    # Same tier as Section → pop section; chapter tier 0 remains
    outline_push_unnumbered_by_font(o, "Aside", tier=2)
    assert [x[1] for x in o] == ["Chapter One", "Aside"]
    # Tier-0 unnumbered competes with chapter tier 0 → replaces whole outline
    outline_push_unnumbered_by_font(o, "Appendix", tier=0)
    assert [x[1] for x in o] == ["Appendix"]


def test_outline_numbered_after_unnumbered_pops_trailing_unnumbered():
    o: list[tuple[int, str, tuple[int, ...] | None]] = []
    outline_push_numbered(o, (1,), "Chapter One", heading_tier=0)
    outline_push_unnumbered_by_font(o, "Note", tier=3)
    outline_push_numbered(o, (1, 1), "First section", heading_tier=2)
    assert [x[1] for x in o] == ["Chapter One", "First section"]


def test_every_span_above_body_all_spans_must_exceed_body():
    body = 11.0
    bbox = (0.0, 0.0, 100.0, 10.0)
    ok_line = PdfLineRec(
        page=1,
        spans=[
            PdfSpanRec(text="Big ", size=14.0, bbox=bbox),
            PdfSpanRec(text="Title", size=14.0, bbox=bbox),
        ],
        bbox=bbox,
        page_height=200.0,
    )
    assert every_span_above_body_font(ok_line, body) is True

    mixed = PdfLineRec(
        page=1,
        spans=[
            PdfSpanRec(text="Big", size=14.0, bbox=bbox),
            PdfSpanRec(text="small", size=11.0, bbox=bbox),
        ],
        bbox=bbox,
        page_height=200.0,
    )
    assert every_span_above_body_font(mixed, body) is False

    whitespace_only_span = PdfLineRec(
        page=1,
        spans=[
            PdfSpanRec(text="   ", size=10.0, bbox=bbox),
            PdfSpanRec(text="OK", size=14.0, bbox=bbox),
        ],
        bbox=bbox,
        page_height=200.0,
    )
    assert every_span_above_body_font(whitespace_only_span, body) is True


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
