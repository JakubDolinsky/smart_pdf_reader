"""
Reranking in RAG pipeline: prepare (query, chunk) inputs for the reranker and return top-k chunks.
Accepts a dictionary with "chunks" (list of Chunk from DBManager.search_similar). Builds query/chunk
pairs, calls the infra RerankingClient to score them, and returns the reranked chunks (no scores
in output). Logs the top chunks with scores for analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from AI_module.config import RERANK_TOP_K
from AI_module.core.chunk import Chunk
from AI_module.infra_layer.reranking_client import RerankingClient

logger = logging.getLogger(__name__)

_EMPTY_RESULT: dict[str, Any] = {"chunks": []}

# After sorting by score descending: drop chunks with score <= RERANK_MIN_SCORE; keep the best
# chunk and others only if (top_score - score) <= RERANK_MAX_SCORE_GAP. Cap count at top_k.
RERANK_MIN_SCORE: float = -2.0
RERANK_MAX_SCORE_GAP: float = 3.0


def _select_filtered_reranked(
    indexed: list[tuple[int, float]],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    indexed: (original_index, score) sorted by score descending.
    Returns a sublist in the same order, at most top_k items.
    """
    if not indexed or top_k <= 0:
        return []
    top_score = indexed[0][1]
    if top_score <= RERANK_MIN_SCORE:
        return []
    out: list[tuple[int, float]] = []
    for i, score in indexed:
        if score <= RERANK_MIN_SCORE:
            continue
        if top_score - score > RERANK_MAX_SCORE_GAP:
            continue
        out.append((i, score))
        if len(out) >= top_k:
            break
    return out


class RerankingService:
    """
    Core service for reranking chunks by relevance to a query. Delegates to RerankingClient.
    Use this in the application layer; inject client for tests.
    """

    def __init__(self, *, client: RerankingClient | None = None) -> None:
        """
        Args:
            client: Optional RerankingClient; if None, a default instance is used.
        """
        self._client = client if client is not None else RerankingClient()

    def rerank(
        self,
        query: str,
        chunks_dict: dict[str, Any],
        top_k: int = RERANK_TOP_K,
    ) -> dict[str, Any]:
        """
        Rerank chunks from DBManager.search_similar result to the top-k most relevant to the query.

        chunks_dict: Dict with "chunks" (list of Chunk, e.g. from DBManager.search_similar).
        Query must be non-empty; at least one chunk must have payload with non-empty text.
        Invalid chunks are skipped. If there is no valid pair, returns {"chunks": []}.
        Scores are not returned; the top chunks and their scores are logged for analysis.
        Chunks with score <= RERANK_MIN_SCORE (-4) are dropped. The highest-scoring chunk is kept
        only if its score > RERANK_MIN_SCORE; additional chunks are kept if their score is within
        RERANK_MAX_SCORE_GAP (3) of that top score. At most top_k chunks are returned.

        Returns:
            Dict with "chunks" (list of Chunk, ordered by relevance, vector=None).
        """
        if chunks_dict is None or not isinstance(chunks_dict, dict):
            return _EMPTY_RESULT.copy()
        if query is None or not _is_valid_text(query):
            return _EMPTY_RESULT.copy()

        ids, metadatas = _chunks_to_ids_metadatas(chunks_dict)
        if not ids or not metadatas:
            return _EMPTY_RESULT.copy()

        pairs, valid_indices = prepare_pairs(query, ids, metadatas)
        if not pairs:
            return _EMPTY_RESULT.copy()

        scores = self._client.score_pairs(pairs)
        if len(scores) != len(pairs):
            return _EMPTY_RESULT.copy()

        indexed = list(zip(valid_indices, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        top_indexed = _select_filtered_reranked(indexed, top_k)
        if not top_indexed:
            return _EMPTY_RESULT.copy()

        out_ids = [ids[i] for i, _ in top_indexed]
        out_metadatas = [metadatas[i] for i, _ in top_indexed]

        # Log selected chunks + scores for analysis when behaviour is weird
        for rank, (i, score) in enumerate(top_indexed[:3], start=1):
            meta = metadatas[i]
            text_preview = (meta.get("text") or "")[:80].replace("\n", " ")
            if len((meta.get("text") or "")) > 80:
                text_preview += "..."
            logger.info(
                "Rerank top-%d: id=%s score=%.4f text=%r",
                rank,
                ids[i],
                score,
                text_preview,
            )

        out_chunks = [
            Chunk(id=cid, payload=dict(meta), vector=None)
            for cid, meta in zip(out_ids, out_metadatas)
        ]
        return {"chunks": out_chunks}


def _chunks_to_ids_metadatas(chunks_dict: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Extract ids and metadatas from a dict with "chunks" (list of Chunk-like).
    Returns ([], []) if chunks missing or invalid.
    """
    chunks = chunks_dict.get("chunks")
    if not chunks or not isinstance(chunks, list):
        return [], []
    ids = []
    metadatas = []
    for c in chunks:
        pid = getattr(c, "id", None)
        payload = getattr(c, "payload", None)
        if pid is not None and isinstance(payload, dict):
            ids.append(pid)
            metadatas.append(payload)
    return ids, metadatas


def _is_valid_text(value: Any) -> bool:
    """True if value is a non-empty string after stripping whitespace."""
    if value is None:
        return False
    s = str(value).strip()
    return len(s) > 0


def prepare_pairs(
    query: str,
    ids: list[str],
    metadatas: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], list[int]]:
    """
    Build (query, chunk_text) pairs for chunks that have valid metadata with non-empty text.
    Skips chunks whose text is missing or empty/whitespace.

    Returns:
        (pairs, valid_indices). pairs only includes valid chunks; may be empty.
    """
    if not ids or not metadatas or len(ids) != len(metadatas):
        return [], []

    query_clean = str(query).strip() if query else ""
    if not query_clean:
        return [], []

    pairs: list[tuple[str, str]] = []
    valid_indices: list[int] = []
    for i, meta in enumerate(metadatas):
        if not isinstance(meta, dict):
            continue
        text = meta.get("text")
        if not _is_valid_text(text):
            continue
        pairs.append((query_clean, str(text).strip()))
        valid_indices.append(i)
    return pairs, valid_indices

