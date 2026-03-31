"""
Integration tests for chunking.py: real PDFs and real tokenizer (PyMuPDF, transformers).
For unit tests with mocked extraction see chunking_unit_test.py.
Run directly: python AI_module/tests/core_tests/chunking_integration_test.py
Or: python -m pytest AI_module/tests/core_tests/chunking_integration_test.py -v
"""

import sys
import re
import tempfile
from pathlib import Path

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunking import PdfChunker, get_embedding_tokenizer


def _make_pdf_with_chapters(
    path: Path,
    body_fontsize: float = 11.0,
    chapter_fontsize: float = 18.0,
    chapter1_title: str = "Chapter One",
    chapter2_title: str = "Chapter Two",
    body_words: list[str] | None = None,
) -> None:
    try:
        import fitz
    except ImportError:
        raise AssertionError("PyMuPDF (fitz) is required for chunking integration tests. Install it (pip install pymupdf).")
    if body_words is None:
        body_words = ["word"] * 400
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # insert_text (point) reliably shows up in get_text(); tight insert_textbox rects can omit titles
    y = 60.0
    page.insert_text((50, y), chapter1_title, fontsize=chapter_fontsize, fontname="helv")
    y += float(chapter_fontsize) * 2.0
    body_text = " ".join(body_words)
    rect = fitz.Rect(50, y, 545, 380)
    page.insert_textbox(rect, body_text, fontsize=body_fontsize, fontname="helv")
    y = 400.0
    page.insert_text((50, y), chapter2_title, fontsize=chapter_fontsize, fontname="helv")
    y += float(chapter_fontsize) * 2.0
    rect = fitz.Rect(50, y, 545, 800)
    page.insert_textbox(rect, body_text, fontsize=body_fontsize, fontname="helv")
    doc.save(path)
    doc.close()


def _make_single_block_pdf(path: Path, words: list[str], fontsize: float = 11.0) -> None:
    try:
        import fitz
    except ImportError:
        raise AssertionError("PyMuPDF (fitz) is required for chunking integration tests. Install it (pip install pymupdf).")
    text = " ".join(words)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    rect = fitz.Rect(50, 50, 545, 800)
    page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv")
    doc.save(path)
    doc.close()


def _get_tokenizer_or_skip():
    try:
        return get_embedding_tokenizer()
    except ImportError:
        raise AssertionError(
            "transformers (and tokenizer config) is required for chunking integration tests. "
            "Install it (pip install transformers)."
        )


def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t\n\r]+", " ", text).strip()


def _payload_body_only(text: str) -> str:
    """Strip ``Chapter:`` and optional ``Section:`` header lines."""
    lines = text.splitlines()
    if not lines:
        return ""
    i = 0
    if lines[i].lower().startswith("chapter:"):
        i += 1
    if i < len(lines) and lines[i].lower().startswith("section:"):
        i += 1
    return "\n".join(lines[i:]).strip()


def test_real_pdf_no_chunk_exceeds_max_tokens():
    tokenizer = _get_tokenizer_or_skip()
    # Limit applies to full chunk text (headers + body); allow headroom beyond body-only 32.
    chunker = PdfChunker(max_tokens=96, min_tokens=1, overlap_tokens=6, tokenizer=tokenizer)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_pdf_with_chapters(pdf_path, body_words=["term"] * 300)
        chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 2
        for c in chunks:
            tok_ids = tokenizer.encode(c.payload["text"], add_special_tokens=False)
            assert len(tok_ids) <= 96
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_pdf_overlap_between_consecutive_chunks():
    tokenizer = _get_tokenizer_or_skip()
    chunker = PdfChunker(max_tokens=96, min_tokens=1, overlap_tokens=6, tokenizer=tokenizer)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_pdf_with_chapters(pdf_path, body_words=["term"] * 300)
        chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            if chunks[i].payload.get("path") != chunks[i + 1].payload.get("path"):
                continue
            cur = tokenizer.encode(_payload_body_only(chunks[i].payload["text"]), add_special_tokens=False)
            nxt = tokenizer.encode(_payload_body_only(chunks[i + 1].payload["text"]), add_special_tokens=False)
            if len(cur) < 6 or len(nxt) < 6:
                continue
            assert tuple(cur[-6:]) == tuple(nxt[:6])
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_pdf_chapter_detection_and_chapter_change_mid_content():
    tokenizer = _get_tokenizer_or_skip()
    chunker = PdfChunker(max_tokens=120, min_tokens=1, overlap_tokens=8, tokenizer=tokenizer)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_pdf_with_chapters(
            pdf_path,
            chapter1_title="First Chapter",
            chapter2_title="Second Chapter",
            body_words=["item"] * 300,
        )
        chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 2
        second_chapter_chunks = [
            c for c in chunks if c.payload.get("path") and "Second Chapter" in c.payload["path"]
        ]
        has_any_path = any(c.payload.get("path") for c in chunks)
        if has_any_path:
            assert len(second_chapter_chunks) >= 1
        for c in second_chapter_chunks:
            assert "Second Chapter" in c.payload["path"]
    finally:
        pdf_path.unlink(missing_ok=True)


def test_joined_chunk_texts_match_original_real_pdf():
    # No overlap: naive join of chunk bodies must equal extractable PDF text (overlap duplicates words).
    chunker = PdfChunker(max_tokens=160, overlap_tokens=0, min_tokens=1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_single_block_pdf(pdf_path, ["item"] * 200)
        blocks, _ = chunker._extract_raw_blocks(pdf_path)
        original = _normalize_whitespace(" ".join(b.text for b in blocks))
        chunks = chunker.chunk_document(pdf_path)

        def strip_headers(text: str) -> str:
            lines = text.splitlines()
            if not lines:
                return ""
            i = 0
            if lines[i].lower().startswith("chapter:"):
                i += 1
            if i < len(lines) and lines[i].lower().startswith("section:"):
                i += 1
            return "\n".join(lines[i:]).strip()

        reconstructed = _normalize_whitespace(" ".join(strip_headers(c.payload["text"]) for c in chunks))
        assert original == reconstructed
    finally:
        pdf_path.unlink(missing_ok=True)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
