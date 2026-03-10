"""
PDF chunking: extract text, normalize, split into chunks by paragraphs (max tokens + overlap).
Chapter detection by font size; optional multicolumn pipeline.
Produces Chunk instances (id, payload, vector=None) ready for the embedding step.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from .chunk import Chunk

from AI_module.config import (
CHUNK_MAX_TOKENS,
CHUNK_OVERLAP_TOKENS,
IS_MULTICOLUMN,
PDF_INPUT_DIR,
CHAPTER_FONT_SIZE_MULTIPLIER,
EMBEDDING_MODEL_NAME
)

logger = logging.getLogger(__name__)


def get_embedding_tokenizer() -> Any:
    """
    Return the HuggingFace tokenizer for the embedding model (from config).
    Use with PdfChunker(tokenizer=...) for accurate token-based chunking (e.g. 256/50 tokens).
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    except ImportError as e:
        raise ImportError("Token-based chunking requires transformers. Install with: pip install transformers") from e


class EmptyPdfError(Exception):
    """
    Raised when a PDF contains no extractable text (e.g. image-only or empty).
    Logged by the chunker; API layer can catch and return this message to the user.
    """

    def __init__(self, message: str, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)
        super().__init__(message)


def _normalize_text(text: str) -> str:
    """Collapse redundant whitespace, normalize line breaks, strip."""
    if not text:
        return ""
    text = re.sub(r"[ \t\n\r]+", " ", text)
    return text.strip()


def _token_count_fallback(text: str) -> int:
    """Fallback when no tokenizer: use word count as approximate token count."""
    return len(text.split())


def _make_chunk_id(document_name: str, chunk_index: int) -> str:
    """Stable id from hash of document name and chunk index."""
    raw = f"{document_name}_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _remove_duplicate_headers(
    paragraphs: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    """Drop consecutive duplicate paragraphs (e.g. repeated headers); preserve chapter."""
    if not paragraphs:
        return []
    out: list[tuple[int, str, str]] = [paragraphs[0]]
    for i in range(1, len(paragraphs)):
        page, para, chapter = paragraphs[i]
        _, prev_para, _ = out[-1]
        if para.strip() and para.strip() == prev_para.strip():
            continue
        if len(para.strip()) <= 3 and prev_para.strip() == para.strip():
            continue
        out.append((page, para, chapter))
    return out


# (page_no, paragraph_text, chapter_heading_or_empty)
_ParaWithChapter = tuple[int, str, str]


class PdfChunker:
    """
    Creates chunks from a PDF: read text, normalize, split by paragraphs with
    max tokens and overlap (token-based), detect chapters by font size, build Chunk objects.
    Supports single-column and multicolumn (configurable) extraction.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        is_multicolumn: bool | None = None,
        tokenizer: Any = None,
    ) -> None:
        """
        Args:
            max_tokens: Max tokens per chunk (from config if None).
            overlap_tokens: Overlap tokens between chunks (from config if None).
            is_multicolumn: Use multicolumn extraction (from config if None).
            tokenizer: HuggingFace tokenizer for counting tokens; if None, word count is used.
        """
        default_max, default_overlap, default_multi = (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, IS_MULTICOLUMN)
        max_tokens = max_tokens if max_tokens is not None else default_max
        overlap_tokens = overlap_tokens if overlap_tokens is not None else default_overlap
        is_multicolumn = is_multicolumn if is_multicolumn is not None else default_multi
        if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("max_tokens must be positive and overlap_tokens in [0, max_tokens)")
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._is_multicolumn = is_multicolumn
        self._tokenizer = tokenizer

    def _token_count(self, text: str) -> int:
        """Return token count using tokenizer if set, else word count."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return _token_count_fallback(text)

    def chunk_document(
        self,
        pdf_path: str | Path,
        document_name: str | None = None,
        base_path: str | Path | None = None,
    ) -> list[Chunk]:
        """
        Read PDF, normalize, split into chunks (max tokens + overlap), assign id and payload.
        Payload: text, source, page, page_start, page_end, chunk_index, chapter (font-detected).
        Resolves relative pdf_path against base_path when given (default base_path from config PDF_INPUT_DIR).

        Args:
            pdf_path: Path to the PDF file (or relative to base_path).
            document_name: Name for chunk ids and payload source; if None, uses pdf_path stem.
            base_path: If set and pdf_path is relative, resolve as base_path / pdf_path. If None, uses config PDF_INPUT_DIR.

        Returns:
            List of Chunk with id and payload set; vector is None.

        Raises:
            EmptyPdfError: When the PDF has no extractable text (logged; report to user).
        """
        path = Path(pdf_path)
        base = base_path if base_path is not None else PDF_INPUT_DIR
        if base is not None and not path.is_absolute():
            path = Path(base) / path
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got {path.suffix!r}")
        if not path.exists():
            raise FileNotFoundError(str(path))

        name = (document_name or path.stem).strip() or path.stem

        paragraphs_with_chapter = self._extract_paragraphs_with_chapters(path)
        paragraphs_with_chapter = _remove_duplicate_headers(paragraphs_with_chapter)

        if not paragraphs_with_chapter:
            msg = f"PDF contains no extractable text: {path}"
            logger.error(msg)
            raise EmptyPdfError(msg, path)

        chunk_specs = self._build_chunk_specs(paragraphs_with_chapter)

        chunks: list[Chunk] = []
        for idx, (chunk_text, page_start, page_end, chapter) in enumerate(chunk_specs):
            chunk_id = _make_chunk_id(name, idx)
            payload: dict[str, Any] = {
                "text": chunk_text,
                "source": name,
                "page": page_start if page_start == page_end else f"{page_start}-{page_end}",
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": idx,
                "chapter": chapter or "",
            }
            chunks.append(Chunk(id=chunk_id, payload=payload, vector=None))
        return chunks

    def _extract_paragraphs_with_chapters(self, path: Path) -> list[_ParaWithChapter]:
        """
        Extract text per page with chapter detection (larger font = heading).
        Returns list of (page_no, paragraph_text, chapter_heading).
        """
        try:
            import fitz
        except ImportError as e:
            raise ImportError("PdfChunker requires PyMuPDF. Install with: pip install pymupdf") from e

        # Single pass: collect (page, block_text, max_font_size_in_block) per block
        blocks_with_size: list[tuple[int, str, float]] = []
        all_sizes: list[float] = []

        with fitz.open(path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict", sort=True) if self._is_multicolumn else page.get_text("dict")
                page_one_based = page_num + 1
                for block in blocks.get("blocks", []):
                    block_parts: list[str] = []
                    block_max_size: float = 0.0
                    for line in block.get("lines", []):
                        line_text_parts = []
                        for span in line.get("spans", []):
                            line_text_parts.append(span.get("text", ""))
                            s = span.get("size")
                            if s is not None:
                                size_f = float(s)
                                all_sizes.append(size_f)
                                block_max_size = max(block_max_size, size_f)
                        block_parts.append("".join(line_text_parts))
                    block_text = _normalize_text(" ".join(block_parts))
                    if not block_text:
                        continue
                    blocks_with_size.append((page_one_based, block_text, block_max_size))

        if not blocks_with_size:
            return []

        # Chapter = blocks with font size >= median * config multiplier
        median_size = float(sorted(all_sizes)[len(all_sizes) // 2]) if all_sizes else 0.0
        multiplier = CHAPTER_FONT_SIZE_MULTIPLIER
        threshold = median_size * multiplier if median_size > 0 else float("inf")

        current_chapter = ""
        result: list[_ParaWithChapter] = []
        for page, text, size in blocks_with_size:
            if size >= threshold:
                current_chapter = _normalize_text(text)
            result.append((page, text, current_chapter))
        return result

    def _build_chunk_specs(
        self,
        paragraphs_with_chapter: list[_ParaWithChapter],
    ) -> list[tuple[str, int, int, str]]:
        """
        Group paragraphs into chunks of at most max_tokens with overlap_tokens.
        Returns list of (chunk_text, page_start, page_end, chapter).
        """
        if not paragraphs_with_chapter:
            return []

        specs: list[tuple[str, int, int, str]] = []
        current: list[tuple[int, str]] = []
        current_token_count = 0
        last_chapter = ""

        def flush() -> None:
            nonlocal current, current_token_count, last_chapter
            if not current:
                return
            texts = [p[1] for p in current]
            page_start = current[0][0]
            page_end = current[-1][0]
            chunk_text = " ".join(texts)
            specs.append((chunk_text, page_start, page_end, last_chapter))
            overlap_remaining = self._overlap_tokens
            overlap_paras: list[tuple[int, str]] = []
            for i in range(len(current) - 1, -1, -1):
                page, para = current[i]
                tc = self._token_count(para)
                if overlap_remaining <= 0:
                    break
                overlap_paras.append((page, para))
                overlap_remaining -= tc
            current = list(reversed(overlap_paras))
            current_token_count = sum(self._token_count(p[1]) for p in current)
            last_chapter = ""

        for (page, para, chapter) in paragraphs_with_chapter:
            if chapter:
                last_chapter = chapter
            tc = self._token_count(para)
            if tc > self._max_tokens:
                flush()
                words = para.split()
                start = 0
                while start < len(words):
                    segment: list[str] = []
                    while start + len(segment) < len(words):
                        candidate = segment + [words[start + len(segment)]]
                        if self._token_count(" ".join(candidate)) > self._max_tokens and segment:
                            break
                        segment = candidate
                    if not segment:
                        segment = [words[start]]
                        start += 1
                    else:
                        start += len(segment)
                    specs.append((" ".join(segment), page, page, last_chapter))
                    last_chapter = ""
                    if start < len(words):
                        overlap_start = start
                        while overlap_start > 0 and self._token_count(" ".join(words[overlap_start:start])) < self._overlap_tokens:
                            overlap_start -= 1
                        start = overlap_start
                continue
            current.append((page, para))
            current_token_count += tc
            if current_token_count >= self._max_tokens:
                flush()

        if current:
            flush()

        return specs


def chunk_directory(
    pdf_dir: str | Path | None = None,
    chunker: PdfChunker | None = None,
    base_path: str | Path | None = None,
    *,
    skip_empty: bool = True,
) -> list[Chunk]:
    """
    Iterate over all PDFs in a directory and chunk each; return combined list of Chunk.
    Use this to parse all PDF documents found in the configured (or given) PDF directory.

    Args:
        pdf_dir: Directory to scan for *.pdf files. If None, uses config PDF_INPUT_DIR.
        chunker: PdfChunker instance; if None, creates one from config (no tokenizer).
        base_path: Base path for resolving relative paths; if None, uses pdf_dir or config.
        skip_empty: If True, log and skip PDFs that raise EmptyPdfError; if False, re-raise.

    Returns:
        List of Chunk from all PDFs (each chunk's payload has "source" = document name).
    """
    directory = Path(pdf_dir) if pdf_dir is not None else PDF_INPUT_DIR
    if directory is None:
        logger.warning("chunk_directory: pdf_dir not set and config PDF_INPUT_DIR not available")
        return []
    if not directory.is_dir():
        logger.warning("chunk_directory: not a directory or does not exist: %s", directory)
        return []
    if chunker is None:
        chunker = PdfChunker()
    base = base_path if base_path is not None else directory
    all_chunks: list[Chunk] = []
    pdf_paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        logger.info("chunk_directory: no PDFs found in %s", directory)
        return []
    for path in pdf_paths:
        try:
            chunks = chunker.chunk_document(path, base_path=base)
            all_chunks.extend(chunks)
            logger.info("chunk_directory: %s -> %d chunks", path.name, len(chunks))
        except EmptyPdfError as e:
            logger.warning("chunk_directory: skipping empty PDF %s: %s", path, e)
            if not skip_empty:
                raise
    return all_chunks
