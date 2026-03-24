"""
Unit tests for chunking.py (PdfChunker, chunk_directory, EmptyPdfError). Extraction mocked; no real PDFs.
PDF layout / bbox tests: pdf_parsing_unit_test.py. Real-PDF tests: chunking_integration_test.py.
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
    ChapterSegment,
    EmptyPdfError,
    PdfChunker,
    chunk_directory,
    get_embedding_tokenizer,
    split_into_paragraphs,
)


def _make_segment(path_titles: list[str], body: str, page: int = 1) -> ChapterSegment:
    leaf = path_titles[-1]
    parent = path_titles[-2] if len(path_titles) >= 2 else ""
    s = ChapterSegment(path_titles=list(path_titles), leaf_title=leaf, parent_title=parent)
    s.body_parts.append(body)
    s.page_start = s.page_end = page
    return s


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


def _strip_chunk_headers(text: str) -> str:
    """Drop ``Chapter:`` and optional ``Section:`` header lines."""
    lines = text.splitlines()
    if not lines:
        return ""
    i = 0
    if lines[i].lower().startswith("chapter:"):
        i += 1
    if i < len(lines) and lines[i].lower().startswith("section:"):
        i += 1
    return "\n".join(lines[i:]).strip()


def test_split_into_paragraphs_double_newline_and_pilcrow():
    text = "First para line.\n\nSecond paragraph\nwith two lines.\n\n¶\nThird block."
    paras = split_into_paragraphs(text)
    assert len(paras) >= 2
    assert "First para" in paras[0]
    assert any("Second paragraph" in p for p in paras)


def test_split_into_paragraphs_tab_starts_new_paragraph():
    text = "First block line one.\n\tIndented new paragraph line."
    paras = split_into_paragraphs(text)
    assert len(paras) == 2
    assert "First block" in paras[0]
    assert "Indented new paragraph" in paras[1]


def test_chapter_body_joins_blocks_without_extra_paragraph_breaks():
    """Single newlines between PDF blocks → one paragraph unless \\n\\n, tab, or pilcrow."""
    s = ChapterSegment(
        path_titles=["doc", "Ch1"],
        leaf_title="Ch1",
        parent_title="doc",
    )
    s.body_parts.append("Line a\nLine b")
    s.body_parts.append("Line c")
    assert "\n\n" not in s.body
    paras = split_into_paragraphs(s.body)
    assert len(paras) == 1
    assert "Line a" in paras[0] and "Line c" in paras[0]


def test_normalize_copyright_and_dot_runs_in_chunk_text():
    chunker = PdfChunker(max_tokens=256, min_tokens=1, overlap_tokens=0, tokenizer=None)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    stem = pdf_path.stem
    seg = _make_segment(
        [stem, "Chapter....", "Section © Copyright TEST...."],
        "Body © Copyright TEST.... end.\n\nNext paragraph TEST.....",
        1,
    )
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[seg]):
            chunks = chunker.chunk_document(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)

    combined = "\n".join(c.payload["text"] for c in chunks)
    assert "©" not in combined
    assert "Copyright" not in combined
    assert "...." not in combined  # no 4+ dot runs should remain
    assert "TEST..." in combined


def test_split_long_text_sentence_safe_and_balanced():
    """Mandatory splits keep full sentences and avoid tiny tail chunks."""
    chunker = PdfChunker(max_tokens=35, min_tokens=4, overlap_tokens=0, tokenizer=None)
    leaf, parent = "Ab", "Cd"
    sentences = [f"Sentence number {i} has six words." for i in range(1, 13)]
    text = " ".join(sentences)
    assert chunker._full_chunk_token_count(leaf, parent, text) > chunker._max_tokens

    pieces = chunker._split_long_text(leaf, parent, text)
    assert len(pieces) >= 2

    toks = [chunker._full_chunk_token_count(leaf, parent, p) for p in pieces]
    assert all(t <= chunker._max_tokens for t in toks)
    assert max(toks) - min(toks) <= 10, f"token lengths too uneven: {toks}"

    # Each piece should end with sentence punctuation and contain whole source sentences.
    for p in pieces:
        assert p.strip().endswith((".", "!", "?"))
    for s in sentences:
        assert any(s in p for p in pieces), f"missing full sentence: {s}"


def test_pdf_chunker_init_valid():
    c = PdfChunker(max_tokens=100, min_tokens=5, overlap_tokens=10)
    assert c._max_tokens == 100 and c._min_tokens == 5 and c._overlap_tokens == 10


def test_pdf_chunker_init_invalid_raises():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PdfChunker(max_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens in"):
        PdfChunker(max_tokens=50, overlap_tokens=50)
    with pytest.raises(ValueError):
        PdfChunker(max_tokens=10, overlap_tokens=15)
    with pytest.raises(ValueError, match="min_tokens"):
        PdfChunker(max_tokens=20, min_tokens=50, overlap_tokens=2)


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
        with patch.object(chunker, "_extract_chapter_segments", return_value=[]):
            with pytest.raises(EmptyPdfError, match="no extractable text"):
                chunker.chunk_document(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)


def test_chunk_document_returns_chunks_with_payload():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    stem = pdf_path.stem
    seg = _make_segment(
        [stem, "Part A"],
        "First block of text.\n\nSecond block.",
        1,
    )
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[seg]):
            chunks = chunker.chunk_document(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Chunk) and c.vector is None and c.id and "text" in c.payload
        assert (
            c.payload["source"] == pdf_path.stem
            and "page" in c.payload
            and "chunk_index" in c.payload
            and "path" in c.payload
        )
        assert " - " in c.payload["path"] or c.payload["path"]


def test_chunk_document_document_name_used_in_payload():
    chunker = PdfChunker(max_tokens=256, overlap_tokens=50)
    seg = _make_segment(["my_doc"], "Some text.", 1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[seg]):
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
    # Header adds several words; limits apply to full chunk (header + body).
    chunker = PdfChunker(max_tokens=35, min_tokens=1, overlap_tokens=2)
    seg = _make_segment(["tmp"], "one two three four five six seven", 1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[seg]):
            chunks = chunker.chunk_document(pdf_path)
        assert len(chunks) >= 1
        assert sum(len(c.payload["text"].split()) for c in chunks) >= 7
    finally:
        pdf_path.unlink(missing_ok=True)


def test_chapter_path_in_payload_mocked():
    chunker = PdfChunker(max_tokens=200, overlap_tokens=20)
    stem = "docx"
    s1 = _make_segment([stem, "Chapter One"], "First chapter content. " * 30, 1)
    s2 = _make_segment([stem, "Chapter One", "Chapter Two"], "Content under chapter two. " * 15, 1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[s1, s2]):
            chunks = chunker.chunk_document(pdf_path, document_name=stem)
        ch2 = [c for c in chunks if "Chapter Two" in c.payload["path"]]
        assert len(ch2) >= 1
        assert " - " in ch2[0].payload["path"]
    finally:
        pdf_path.unlink(missing_ok=True)


def test_joined_chunk_texts_match_original_mocked():
    """Bodies round-trip after stripping chapter/section headers."""
    # Whole-chunk word limit includes headers; 50 is tight — use a higher cap for this body.
    chunker = PdfChunker(max_tokens=120, min_tokens=5, overlap_tokens=0)
    stem = "docstem"
    body = (
        "First paragraph with some content.\n\n"
        "Second paragraph continues the flow.\n\n"
        "Third paragraph and more text here.\n\n"
        "Page two starts with this block."
    )
    seg = _make_segment([stem], body, 1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)
    try:
        with patch.object(chunker, "_extract_chapter_segments", return_value=[seg]):
            chunks = chunker.chunk_document(pdf_path, document_name=stem)
        original = _normalize_whitespace(body)
        reconstructed = _normalize_whitespace(
            " ".join(_strip_chunk_headers(c.payload["text"]) for c in chunks)
        )
        assert original == reconstructed
    finally:
        pdf_path.unlink(missing_ok=True)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
