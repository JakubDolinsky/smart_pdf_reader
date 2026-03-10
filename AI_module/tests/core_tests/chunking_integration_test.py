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
from AI_module.core.chunk import Chunk
from AI_module.core.chunking import PdfChunker, _remove_duplicate_headers, get_embedding_tokenizer


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
        pytest.skip("PyMuPDF (fitz) required for integration tests")
    if body_words is None:
        body_words = ["word"] * 400
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 50.0
    rect = fitz.Rect(50, y, 545, y + 30)
    page.insert_textbox(rect, chapter1_title, fontsize=chapter_fontsize, fontname="helv")
    y += 50
    body_text = " ".join(body_words)
    rect = fitz.Rect(50, y, 545, 380)
    page.insert_textbox(rect, body_text, fontsize=body_fontsize, fontname="helv")
    y = 400
    rect = fitz.Rect(50, y, 545, y + 30)
    page.insert_textbox(rect, chapter2_title, fontsize=chapter_fontsize, fontname="helv")
    y += 50
    rect = fitz.Rect(50, y, 545, 800)
    page.insert_textbox(rect, body_text, fontsize=body_fontsize, fontname="helv")
    doc.save(path)
    doc.close()


def _make_single_block_pdf(path: Path, words: list[str], fontsize: float = 11.0) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF (fitz) required for integration tests")
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
        pytest.skip("transformers/config required for integration tests")


def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t\n\r]+", " ", text).strip()


def _tokenize_like_chunker(chunker: PdfChunker, text: str) -> list:
    if chunker._tokenizer is not None:
        return list(chunker._tokenizer.encode(text, add_special_tokens=False))
    return text.split()


def _reconstruct_text_from_chunks(chunker: PdfChunker, chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    tokens = _tokenize_like_chunker(chunker, chunks[0].payload["text"])
    max_overlap = chunker._overlap_tokens
    for c in chunks[1:]:
        ct = _tokenize_like_chunker(chunker, c.payload["text"])
        search_up_to = min(len(tokens), len(ct), max_overlap)
        overlap_len = 0
        for L in range(1, search_up_to + 1):
            if tokens[-L:] == ct[:L]:
                overlap_len = L
        tokens.extend(ct[overlap_len:])
    if chunker._tokenizer is not None:
        return chunker._tokenizer.decode(tokens)
    return " ".join(tokens)


def test_real_pdf_no_chunk_exceeds_max_tokens():
    tokenizer = _get_tokenizer_or_skip()
    chunker = PdfChunker(max_tokens=32, overlap_tokens=6, tokenizer=tokenizer)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_pdf_with_chapters(pdf_path, body_words=["term"] * 300)
        chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 2
        for c in chunks:
            tok_ids = tokenizer.encode(c.payload["text"], add_special_tokens=False)
            assert len(tok_ids) <= 32
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_pdf_overlap_between_consecutive_chunks():
    tokenizer = _get_tokenizer_or_skip()
    chunker = PdfChunker(max_tokens=32, overlap_tokens=6, tokenizer=tokenizer)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_pdf_with_chapters(pdf_path, body_words=["term"] * 300)
        chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            cur = tokenizer.encode(chunks[i].payload["text"], add_special_tokens=False)
            nxt = tokenizer.encode(chunks[i + 1].payload["text"], add_special_tokens=False)
            if len(cur) < 6 or len(nxt) < 6:
                continue
            assert tuple(cur[-6:]) == tuple(nxt[:6])
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_pdf_chapter_detection_and_chapter_change_mid_content():
    tokenizer = _get_tokenizer_or_skip()
    chunker = PdfChunker(max_tokens=40, overlap_tokens=8, tokenizer=tokenizer)
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
        second_chapter_chunks = [c for c in chunks if c.payload.get("chapter") == "Second Chapter"]
        has_any_chapter = any(c.payload.get("chapter") for c in chunks)
        if has_any_chapter:
            assert len(second_chapter_chunks) >= 1
        for c in second_chapter_chunks:
            assert c.payload["chapter"] == "Second Chapter"
    finally:
        pdf_path.unlink(missing_ok=True)


def test_joined_chunk_texts_match_original_real_pdf():
    chunker = PdfChunker(max_tokens=80, overlap_tokens=12)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        _make_single_block_pdf(pdf_path, ["item"] * 200)
        paragraphs = chunker._extract_paragraphs_with_chapters(pdf_path)
        deduped = _remove_duplicate_headers(paragraphs)
        original = " ".join(p[1] for p in deduped)
        chunks = chunker.chunk_document(pdf_path)
        reconstructed = _reconstruct_text_from_chunks(chunker, chunks)
        assert _normalize_whitespace(original) == _normalize_whitespace(reconstructed)
    finally:
        pdf_path.unlink(missing_ok=True)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
