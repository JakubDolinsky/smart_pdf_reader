"""
Turn :class:`ChapterSegment` bodies into :class:`Chunk` records (token min/max, overlap).

Layout and chapter tree: :mod:`pdf_parsing` via :func:`~pdf_parsing.extract_chapter_segments`.
Each PDF becomes a list of segments (one leaf of the outline per segment); **all splitting,
merging, and packing happen only inside that segment** — nothing is merged across chapters
or across segments.

**Paragraphs → chunks**

- Body text is split with :func:`split_into_paragraphs`, then **short** paragraphs are
  **merged** with the next until each run is at least ``min_tokens`` (and the merge stays
  ≤ ``max_tokens``). **Long** paragraphs are **split** (sentences first, then balanced
  bins, then greedy / word fallback) so every piece fits under ``max_tokens`` with headers.
- Packed chunks add overlap by repeating trailing paragraphs between consecutive chunks
  in the same segment.

**Chunk text shape**

- Header lines (then body): ``Chapter: <title>,\n`` or ``Chapter: <parent>,\n`` +
  ``Section: <leaf>,\n``. Body is segment text only; outline is payload ``path``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from .chunk import Chunk
from .pdf_parsing import ChapterSegment, extract_chapter_segments

# Re-export for ``from AI_module.core.chunking import ChapterSegment`` / ``AI_module.core``.
__all__ = [
    "ChapterSegment",
    "EmptyPdfError",
    "PdfChunker",
    "chunk_directory",
    "get_embedding_tokenizer",
    "split_into_paragraphs",
]

from AI_module.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TITLE_MAX_TOKENS,
    EMBEDDING_MODEL_NAME,
    IS_MULTICOLUMN,
    PDF_INPUT_DIR,
)

logger = logging.getLogger(__name__)

_embedding_tokenizer: Any | None = None

_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)", re.S)
_COPYRIGHT_RE = re.compile(r"(?:©|\bCopyright\b\.?)", re.IGNORECASE)
_DOT_RUN_RE = re.compile(r"\.{4,}")
# Paragraph boundaries: blank line(s), pilcrow, U+2029, or newline before tab-indented line.
_PARA_BREAK_RE = re.compile(r"(?:\n\s*){2,}|[¶\u2029]+|\n(?=\s*\t)")


def get_embedding_tokenizer() -> Any:
    """
    Return the HuggingFace tokenizer for the embedding model (from config).
    Cached per process for performance (prompt fitting and chunking call it often).
    """
    global _embedding_tokenizer
    if _embedding_tokenizer is not None:
        return _embedding_tokenizer
    try:
        from transformers import AutoTokenizer

        _embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        return _embedding_tokenizer
    except ImportError as e:
        raise ImportError("Token-based chunking requires transformers. Install with: pip install transformers") from e


class EmptyPdfError(Exception):
    """Raised when a PDF yields no chapter segments / no extractable structured text."""

    def __init__(self, message: str, pdf_path: str | Path) -> None:
        self.pdf_path = str(pdf_path)
        super().__init__(message)


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split chapter body text into paragraphs: double newlines, pilcrow / U+2029,
    or a newline followed by a tab-indented line.
    """
    if not text or not text.strip():
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = _PARA_BREAK_RE.split(t)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_copyright_and_dot_runs(text: str) -> str:
    if not text:
        return ""
    t = _COPYRIGHT_RE.sub("", text)
    t = _DOT_RUN_RE.sub("...", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    return t.strip()


def _make_chunk_id(document_name: str, chunk_index: int) -> str:
    raw = f"{document_name}_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _format_headers(segment: ChapterSegment) -> str:
    """Title-case labels, comma after each heading line; body has no path (path is payload)."""
    pts = segment.path_titles
    if len(pts) <= 1:
        return ""
    if len(pts) == 2:
        t = _normalize_copyright_and_dot_runs(segment.leaf_title)
        return f"Chapter: {t},\n"
    p = _normalize_copyright_and_dot_runs(segment.parent_title)
    l = _normalize_copyright_and_dot_runs(segment.leaf_title)
    return f"Chapter: {p},\nSection: {l},\n"


def _format_path(segment: ChapterSegment) -> str:
    """
    Full outline path: document (root) then each heading, e.g.
    ``DocName - Chapter one - Sub chapter``.
    """
    pts = segment.path_titles
    if not pts:
        return ""
    return " - ".join(p.strip() for p in pts if p and str(p).strip())


class PdfChunker:
    """
    Chunks each :class:`ChapterSegment` separately: paragraph split → merge shorts for
    ``min_tokens`` → split longs for ``max_tokens`` → pack with overlap. No cross-segment
    merges. Optional tokenizer for true token counts; otherwise word counts approximate tokens.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        overlap_tokens: int | None = None,
        is_multicolumn: bool | None = None,
        tokenizer: Any = None,
        title_max_tokens: int | None = None,
    ) -> None:
        mt = max_tokens if max_tokens is not None else CHUNK_MAX_TOKENS
        mn = min_tokens if min_tokens is not None else CHUNK_MIN_TOKENS
        ov = overlap_tokens if overlap_tokens is not None else CHUNK_OVERLAP_TOKENS
        multi = is_multicolumn if is_multicolumn is not None else IS_MULTICOLUMN
        if mt < 1:
            raise ValueError("max_tokens must be positive")
        if ov < 0 or ov >= mt:
            raise ValueError("Invalid overlap_tokens in [0, max_tokens); must be strictly less than max_tokens")
        if mn > mt:
            raise ValueError("min_tokens must not exceed max_tokens")
        self._max_tokens = mt
        self._min_tokens = mn
        self._overlap_tokens = ov
        self._is_multicolumn = multi
        self._tokenizer = tokenizer
        self._title_max_tokens = title_max_tokens if title_max_tokens is not None else CHUNK_TITLE_MAX_TOKENS

    def _token_count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _headers_for_pair(self, parent: str, leaf: str) -> str:
        """Same prefix shape as :func:`_format_headers`."""
        if not leaf:
            return ""
        if not parent:
            return f"Chapter: {leaf},\n"
        return f"Chapter: {parent},\nSection: {leaf},\n"

    def _full_chunk_token_count(self, leaf: str, parent: str, body: str) -> int:
        return self._token_count(self._headers_for_pair(parent, leaf) + body)

    def _chunk_full_tokens(self, segment: ChapterSegment, body: str) -> int:
        return self._token_count(_format_headers(segment) + body)

    def _split_long_text(self, leaf: str, parent: str, text: str) -> list[str]:
        sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        joined = " ".join(sentences)
        if self._full_chunk_token_count(leaf, parent, joined) <= self._max_tokens:
            return [joined]

        n = 2
        best: list[str] | None = None
        while n <= len(sentences):
            parts = self._pack_sentences_into_n_bins(leaf, parent, sentences, n)
            if not parts:
                n += 1
                continue
            if all(self._full_chunk_token_count(leaf, parent, p) <= self._max_tokens for p in parts):
                toks = [self._full_chunk_token_count(leaf, parent, p) for p in parts]
                spread = max(toks) - min(toks)
                if spread <= 10:
                    return parts
                if best is None or spread < max(
                    self._full_chunk_token_count(leaf, parent, x) for x in best
                ) - min(self._full_chunk_token_count(leaf, parent, x) for x in best):
                    best = parts
            n += 1

        if best is not None:
            return best

        pieces: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            nonlocal current
            if current:
                pieces.append(" ".join(current))
                current = []

        for s in sentences:
            trial = " ".join(current + [s])
            if self._full_chunk_token_count(leaf, parent, trial) <= self._max_tokens:
                current.append(s)
                continue
            flush_current()
            if self._full_chunk_token_count(leaf, parent, s) <= self._max_tokens:
                current = [s]
            else:
                pieces.extend(self._hard_word_split_leaf(leaf, parent, s))
                current = []
        flush_current()
        return pieces

    def _pack_sentences_into_n_bins(self, leaf: str, parent: str, sentences: list[str], n: int) -> list[str]:
        """Assign each sentence to one of n bins (min-sum greedy) and return non-empty joined parts."""
        if n < 1 or not sentences:
            return []
        bins: list[list[str]] = [[] for _ in range(n)]
        sums = [0] * n
        for s in sentences:
            j = min(range(n), key=lambda i: sums[i])
            bins[j].append(s)
            sums[j] += self._token_count(s)
        return [" ".join(b) for b in bins if b]

    def _hard_word_split_leaf(self, leaf: str, parent: str, text: str) -> list[str]:
        """Last resort: pack words into pieces under ``max_tokens`` (headers + piece)."""
        words = text.split()
        if not words:
            return []
        out: list[str] = []
        seg: list[str] = []
        for w in words:
            trial = " ".join(seg + [w])
            if not seg:
                seg = [w]
                continue
            if self._full_chunk_token_count(leaf, parent, trial) <= self._max_tokens:
                seg.append(w)
            else:
                out.append(" ".join(seg))
                seg = [w]
        if seg:
            out.append(" ".join(seg))
        return [p for p in out if p.strip()]

    def _merge_paragraphs_for_min_tokens(self, paragraphs: list[str]) -> list[str]:
        if not paragraphs:
            return []
        out: list[str] = []
        i = 0
        while i < len(paragraphs):
            cur = paragraphs[i]
            while i + 1 < len(paragraphs) and self._token_count(cur) < self._min_tokens:
                nxt = paragraphs[i + 1]
                trial = cur + "\n\n" + nxt
                if self._token_count(trial) > self._max_tokens:
                    break
                cur = trial
                i += 1
            out.append(cur)
            i += 1
        return out

    def _expand_oversized_paragraphs(self, segment: ChapterSegment, paragraphs: list[str]) -> list[str]:
        leaf, parent = segment.leaf_title, segment.parent_title
        flat: list[str] = []
        for p in paragraphs:
            if self._full_chunk_token_count(leaf, parent, p) <= self._max_tokens:
                flat.append(p)
            else:
                flat.extend(self._split_long_text(leaf, parent, p))
        return flat

    def _overlap_paragraphs_suffix(self, buf: list[str]) -> list[str]:
        if not buf or self._overlap_tokens <= 0:
            return []
        overlap_remaining = self._overlap_tokens
        overlap_paras: list[str] = []
        for k in range(len(buf) - 1, -1, -1):
            piece = buf[k]
            tc = self._token_count(piece)
            if overlap_remaining <= 0 and overlap_paras:
                break
            overlap_paras.insert(0, piece)
            overlap_remaining -= tc
        return overlap_paras

    def _pack_paragraphs_to_chunk_bodies(self, segment: ChapterSegment, paragraphs: list[str]) -> list[str]:
        """Return list of body strings (no header); each fits within max_tokens with header."""
        if not paragraphs:
            return []
        paras = list(paragraphs)
        n = len(paras)
        bodies: list[str] = []
        i = 0
        while i < n:
            j = i
            current: list[str] = []
            while j < n:
                trial = current + [paras[j]]
                body = "\n\n".join(trial)
                if self._chunk_full_tokens(segment, body) <= self._max_tokens:
                    current = trial
                    j += 1
                else:
                    break
            if not current:
                p = paras[i]
                subs = self._split_long_text(segment.leaf_title, segment.parent_title, p)
                if len(subs) == 1 and self._chunk_full_tokens(segment, subs[0]) > self._max_tokens:
                    subs = self._hard_word_split_segment(segment, subs[0])
                paras[i : i + 1] = subs
                n = len(paras)
                continue
            bodies.append("\n\n".join(current))
            if j >= n:
                break
            ov = self._overlap_paragraphs_suffix(current)
            len_ov = len(ov)
            # If overlap is empty or swallowed the whole chunk (chunk token sum < overlap budget),
            # advance past this chunk only — otherwise we never increment and loop forever.
            if len_ov == 0 or len_ov >= len(current):
                i = j
            else:
                i = j - len_ov
        return bodies

    def _hard_word_split_segment(self, segment: ChapterSegment, text: str) -> list[str]:
        """Word-boundary fallback: each piece satisfies header + piece ≤ ``max_tokens``."""
        hdr = _format_headers(segment)
        words = text.split()
        if not words:
            return []
        out: list[str] = []
        seg: list[str] = []
        for w in words:
            trial = " ".join(seg + [w])
            if not seg:
                seg = [w]
                continue
            if self._token_count(hdr + trial) <= self._max_tokens:
                seg.append(w)
            else:
                out.append(" ".join(seg))
                seg = [w]
        if seg:
            out.append(" ".join(seg))
        return [x for x in out if x.strip()]

    def _extract_chapter_segments(self, path: Path, document_name: str) -> list[ChapterSegment]:
        return extract_chapter_segments(
            path,
            document_name,
            is_multicolumn=self._is_multicolumn,
            title_max_tokens=self._title_max_tokens,
            token_count=self._token_count,
        )

    def _extract_raw_blocks(self, path: Path) -> tuple[list[Any], list[float]]:
        """Delegate to :func:`pdf_parsing.extract_raw_blocks` (tests / diagnostics)."""
        from .pdf_parsing import extract_raw_blocks

        return extract_raw_blocks(path, self._is_multicolumn)

    def chunk_document(
        self,
        pdf_path: str | Path,
        document_name: str | None = None,
        base_path: str | Path | None = None,
    ) -> list[Chunk]:
        path = Path(pdf_path)
        base = base_path if base_path is not None else PDF_INPUT_DIR
        if base is not None and not path.is_absolute():
            path = Path(base) / path
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got {path.suffix!r}")
        if not path.exists():
            raise FileNotFoundError(str(path))

        name = (document_name or path.stem).strip() or path.stem
        segments = self._extract_chapter_segments(path, name)
        if not segments:
            msg = f"PDF contains no extractable text: {path}"
            logger.error(msg)
            raise EmptyPdfError(msg, path)

        chunks: list[Chunk] = []
        global_idx = 0
        for segment in segments:
            raw_body = segment.body
            norm_body = _normalize_copyright_and_dot_runs(raw_body)
            paras = split_into_paragraphs(norm_body)
            paras = self._merge_paragraphs_for_min_tokens(paras)
            paras = self._expand_oversized_paragraphs(segment, paras)
            bodies = self._pack_paragraphs_to_chunk_bodies(segment, paras)
            hdr = _format_headers(segment)
            path_str = _format_path(segment)
            ps, pe = segment.page_start, segment.page_end
            page_val: int | str = ps if ps == pe else f"{ps}-{pe}"
            for body in bodies:
                full_text = hdr + body if hdr else body
                chunk_id = _make_chunk_id(name, global_idx)
                payload: dict[str, Any] = {
                    "text": full_text,
                    "source": name,
                    "page": page_val,
                    "page_start": ps,
                    "page_end": pe,
                    "chunk_index": global_idx,
                    "path": path_str,
                }
                chunks.append(Chunk(id=chunk_id, payload=payload, vector=None))
                global_idx += 1
        return chunks


def chunk_directory(
    pdf_dir: str | Path | None = None,
    chunker: PdfChunker | None = None,
    base_path: str | Path | None = None,
    *,
    skip_empty: bool = True,
) -> list[Chunk]:
    """
    Iterate over all PDFs in a directory and chunk each; return combined list of Chunk.
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
    pdf_paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
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
