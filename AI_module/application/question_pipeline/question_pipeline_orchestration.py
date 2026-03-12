"""
Question pipeline orchestration: embed question -> search similar chunks -> rerank top-k -> LLM answer.
Uses core EmbeddingService, DBManager, RerankingService, LLMChatter. Input is a question about PDF content;
output is the LLM answer string (or PROMPT_INCOMPLETE_RESPONSE when query/chunks are invalid).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from AI_module.config import RERANK_TOP_K, TOP_K_SIMILAR_CHUNKS, VECTOR_COLLECTION_NAME
from AI_module.core.db_manager import DBManager
from AI_module.core.embedding import EmbeddingService
from AI_module.core.llm_chatter import LLMChatter, PROMPT_INCOMPLETE_RESPONSE
from AI_module.core.reranking import RerankingService

if TYPE_CHECKING:
    from AI_module.core.chunk import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_END = ".!?"


def _trim_to_last_sentence(text: str) -> str:
    """
    If text does not end with a sentence-ending punctuation (. ! ?), truncate to the last
    complete sentence to avoid showing a cut-off sentence when the LLM hits the token limit.
    """
    if not text:
        return text
    s = text.rstrip()
    if not s:
        return text
    if s[-1] in _SENTENCE_END:
        return s
    last_end = -1
    for char in _SENTENCE_END:
        idx = s.rfind(char)
        if idx > last_end:
            last_end = idx
    if last_end == -1:
        return s
    return s[: last_end + 1].rstrip()


def answer_question(
    question: str,
    *,
    collection_name: str | None = None,
    top_k_similar: int = TOP_K_SIMILAR_CHUNKS,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_service: EmbeddingService | None = None,
    db_manager: DBManager | None = None,
    reranking_service: RerankingService | None = None,
    llm_chatter: LLMChatter | None = None,
    host: str | None = None,
    port: int | None = None,
    history: list[dict[str, str]] | None = None,
    history_last_four_messages_reversed: list[dict[str, str]] | None = None,
) -> str:
    """
    Answer a question about PDF content using RAG: embed question, search DB for similar chunks
    (cosine), rerank to top-k, then build prompt (with optional conversation history) and call LLM.

    Args:
        question: User question about the PDF content.
        collection_name: Vector DB collection. If None, uses config VECTOR_COLLECTION_NAME.
        top_k_similar: Number of similar chunks to fetch from DB (default from config).
        rerank_top_k: Number of chunks to keep after reranking (default 3).
        embedding_service: Optional; if None, a default EmbeddingService is used.
        db_manager: Optional; if None, a DBManager is created (uses host/port when given).
        reranking_service: Optional; if None, a default RerankingService is used.
        llm_chatter: Optional; if None, a default LLMChatter is used.
        host: Qdrant host when creating DBManager (e.g. for tests).
        port: Qdrant port when creating DBManager.
        history: Optional list of {"role": "user"|"assistant", "content": str}; last 4 messages
            are included in the prompt (chronological). Used for LLM when history_last_four_messages_reversed is not provided.
        history_last_four_messages_reversed: Optional pre-built list of last 4 messages with
            first = most recent (last in conversation). If provided, used for embedding (to append
            prior context when question contains reference words) and reversed for the LLM prompt.
            If empty or question has no reference words, embedding uses the question only.

    Returns:
        LLM answer string, or PROMPT_INCOMPLETE_RESPONSE if question is empty or no valid chunks.
    """
    query_clean = (question or "").strip()
    if not query_clean:
        logger.debug("Question pipeline: empty question, returning incomplete response.")
        return PROMPT_INCOMPLETE_RESPONSE

    # History for LLM (chronological: oldest first). For embedding we pass reversed (first = most recent).
    if history_last_four_messages_reversed is not None:
        history_for_llm = list(reversed(history_last_four_messages_reversed))
        embed_history = history_last_four_messages_reversed if history_last_four_messages_reversed else None
    else:
        history_last_four = (history[-4:] if history else [])
        history_for_llm = history_last_four
        embed_history = list(reversed(history_last_four)) if history_last_four else None

    coll = collection_name or VECTOR_COLLECTION_NAME
    embedder = embedding_service if embedding_service is not None else EmbeddingService()
    db = db_manager if db_manager is not None else DBManager(
        collection_name=coll, host=host, port=port
    )
    reranker = reranking_service if reranking_service is not None else RerankingService()
    chatter = llm_chatter if llm_chatter is not None else LLMChatter()

    query_embedding = embedder.embed_query(query_clean, history=embed_history)
    search_result: dict[str, Any] = db.search_similar(
        query_embedding, top_k=top_k_similar, include_scores=True
    )
    chunks_from_db = search_result.get("chunks") or []
    if not chunks_from_db:
        logger.debug("Question pipeline: no similar chunks found, returning incomplete response.")
        return PROMPT_INCOMPLETE_RESPONSE

    reranked = reranker.rerank(query_clean, {"chunks": chunks_from_db}, top_k=rerank_top_k)
    top_chunks: list[Chunk] = reranked.get("chunks") or []
    if not top_chunks:
        logger.debug("Question pipeline: rerank returned no chunks, returning incomplete response.")
        return PROMPT_INCOMPLETE_RESPONSE

    raw_answer = chatter.chat({"chunks": top_chunks}, query_clean, history=history_for_llm)
    if raw_answer == PROMPT_INCOMPLETE_RESPONSE:
        return raw_answer
    return _trim_to_last_sentence(raw_answer)
