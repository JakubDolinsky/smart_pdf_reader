"""
Unit tests for chunking.py (PdfChunker, chunk_directory, EmptyPdfError). Extraction mocked; no real PDFs.
For real-PDF and tokenizer integration tests see chunking_integration_test.py.
Run directly: python AI_module/tests/core_tests/chunking_unit_test.py
Or: python -m pytest AI_module/tests/core_tests/chunking_unit_test.py -v
"""

import sys
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunk import Chunk
from AI_module.core.chunking import (
    EmptyPdfError,
    PdfChunker,
    _remove_duplicate_headers,
    chunk_directory,
    get_embedding_tokenizer,
)


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


def test_pdf_chunker_init_valid():
    c = PdfChunker(max_tokens=100, overlap_tokens=10)
    assert c._max_tokens == 100 and c._overlap_tokens == 10


def test_pdf_chunker_init_invalid_raises():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PdfChunker(max_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens in"):
        PdfChunker(max_tokens=50, overlap_tokens=50)
    with pytest.raises(ValueError):
        PdfChunker(max_tokens=10, overlap_tokens=15)


def test_chunk_document_non_pdf_extension_raises():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    with pytest.raises(ValueError, match="Expected a PDF file"):
        chunker.chunk_document(Path("/nonexistent/file.txt"))


def test_chunk_document_missing_file_raises():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    with pytest.raises(FileNotFoundError):
        chunker.chunk_document(Path("/nonexistent/doc.pdf"))


def test_chunk_document_empty_pdf_raises():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=[]):
            with pytest.raises(EmptyPdfError, match="no extractable text"):
                chunker.chunk_document(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)


def test_chunk_document_returns_chunks_with_payload():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    mock_paragraphs = [(1, "First block of text.", ""), (1, "Second block.", "")]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=mock_paragraphs):
            chunks = chunker.chunk_document(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Chunk) and c.vector is None and c.id and "text" in c.payload
        assert c.payload["source"] == pdf_path.stem and "page" in c.payload and "chunk_index" in c.payload and "chapter" in c.payload


def test_chunk_document_document_name_used_in_payload():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=[(1, "Some text.", "")]):
            chunks = chunker.chunk_document(pdf_path, document_name="my_doc")
    finally:
        pdf_path.unlink(missing_ok=True)
    assert len(chunks) == 1 and chunks[0].payload["source"] == "my_doc"


def test_empty_pdf_error_has_pdf_path_and_message():
    err = EmptyPdfError("No text.", "/path/to/file.pdf")
    assert str(err) == "No text." and err.pdf_path == "/path/to/file.pdf"


def test_chunk_directory_nonexistent_dir_returns_empty():
    assert chunk_directory(pdf_dir=Path("/nonexistent_folder_xyz")) == []


def test_chunk_directory_empty_dir_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert chunk_directory(pdf_dir=tmp) == []


def test_chunk_directory_iterates_only_pdf_files():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.pdf").touch()
        (Path(tmp) / "b.PDF").touch()
        (Path(tmp) / "c.txt").touch()
        with patch("AI_module.core.chunking.PdfChunker") as mock_cls:
            mock_chunker = mock_cls.return_value
            mock_chunker.chunk_document.return_value = []
            chunk_directory(pdf_dir=tmp, chunker=mock_chunker)
            assert mock_chunker.chunk_document.call_count == 2


def test_chunk_directory_skips_empty_pdf_when_skip_empty_true():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "empty.pdf").touch()
        with patch("AI_module.core.chunking.PdfChunker") as mock_cls:
            mock_chunker = mock_cls.return_value
            mock_chunker.chunk_document.side_effect = EmptyPdfError("No text.", Path(tmp) / "empty.pdf")
            result = chunk_directory(pdf_dir=tmp, chunker=mock_chunker, skip_empty=True)
        assert result == []


def test_chunk_directory_reraises_empty_pdf_when_skip_empty_false():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "empty.pdf").touch()
        with patch("AI_module.core.chunking.PdfChunker") as mock_cls:
            mock_chunker = mock_cls.return_value
            mock_chunker.chunk_document.side_effect = EmptyPdfError("No text.", Path(tmp) / "empty.pdf")
            with pytest.raises(EmptyPdfError):
                chunk_directory(pdf_dir=tmp, chunker=mock_chunker, skip_empty=False)


def test_chunker_token_count_fallback_without_tokenizer():
    chunker = PdfChunker(max_tokens=5, overlap_tokens=1)
    mock_paragraphs = [(1, "one two three four five six seven", "")]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=mock_paragraphs):
            chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 1
        assert sum(len(c.payload["text"].split()) for c in chunks) >= 7
    finally:
        pdf_path.unlink(missing_ok=True)


def test_chapter_change_in_middle_of_chunk_mocked():
    chunker = PdfChunker(max_tokens=200, overlap_tokens=20)
    mock_paragraphs = [
        (1, "First chapter content. " * 30, "Chapter One"),
        (1, "More text under chapter one. " * 20, ""),
        (1, "Chapter Two", "Chapter Two"),
        (1, "Content under chapter two. " * 15, ""),
    ]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=mock_paragraphs):
            chunks = chunker.chunk_document(pdf_path)
        chunk_with_ch2 = next((c for c in chunks if "Chapter Two" in c.payload["text"]), None)
        assert chunk_with_ch2 is not None and chunk_with_ch2.payload["chapter"] == "Chapter Two"
    finally:
        pdf_path.unlink(missing_ok=True)


def test_joined_chunk_texts_match_original_mocked():
    chunker = PdfChunker(max_tokens=50, overlap_tokens=8)
    mock_paragraphs = [
        (1, "First paragraph with some content.", ""),
        (1, "Second paragraph continues the flow.", ""),
        (1, "Third paragraph and more text here.", ""),
        (2, "Page two starts with this block.", ""),
    ]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_paragraphs_with_chapters", return_value=mock_paragraphs):
            chunks = chunker.chunk_document(pdf_path)
        deduped = _remove_duplicate_headers(mock_paragraphs)
        original = " ".join(p[1] for p in deduped)
        reconstructed = _reconstruct_text_from_chunks(chunker, chunks)
        assert _normalize_whitespace(original) == _normalize_whitespace(reconstructed)
    finally:
        pdf_path.unlink(missing_ok=True)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
