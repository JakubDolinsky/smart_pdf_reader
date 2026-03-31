"""
Question pipeline orchestration: embed question -> search similar chunks -> rerank top-k -> LLM answer.
Uses core EmbeddingService, DBManager, RerankingService, LLMChatter. Input is a question about PDF content;
output is the LLM answer string, RAG_NO_INFORMATION_IN_DOCUMENT when reranking returns no chunks,
or PROMPT_INCOMPLETE_RESPONSE when the question is empty or vector search returns nothing.
"""

from __future__ import annotations

import logging
import re
from math import sqrt
from typing import TYPE_CHECKING, Any

from AI_module.config import (
    REFERENCE_WORDS,
    RAG_NO_INFORMATION_IN_DOCUMENT,
    RERANK_TOP_K,
    TOP_K_SIMILAR_CHUNKS,
    VECTOR_COLLECTION_NAME,
)
from AI_module.core.db_manager import DBManager
from AI_module.core.embedding import EmbeddingService
from AI_module.core.llm_chatter import LLMChatter, PROMPT_INCOMPLETE_RESPONSE
from AI_module.core.reranking import RerankingService

if TYPE_CHECKING:
    from AI_module.core.chunk import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_END = ".!?"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _reference_words_pattern(words: list[str]) -> re.Pattern[str]:
    phrases = sorted({p.strip() for p in words if p and str(p).strip()}, key=len, reverse=True)
    if not phrases:
        return re.compile(r"(?!x)x")
    escaped = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<!\w)(?:{escaped})(?!\w)", re.IGNORECASE)


def _question_contains_reference_word(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    return _reference_words_pattern(REFERENCE_WORDS).search(text) is not None


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two vectors; returns 0.0 when one norm is zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_sentences(text: str) -> list[str]:
    """Split text into simple sentences; keep non-empty trimmed pieces."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s and s.strip()]


def _select_top_assistant_sentences(
    assistant_text: str,
    current_question: str,
    embedder: EmbeddingService,
    max_sentences: int = 3,
) -> str:
    """
    Keep up to max_sentences from assistant_text by cosine similarity to current_question.
    Sentences are returned in descending relevance order.
    """
    sentences = _split_sentences(assistant_text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences).strip()

    q_vec = embedder.embed_query(current_question)
    scored: list[tuple[str, float]] = []
    for sentence in sentences:
        s_vec = embedder.embed_query(sentence)
        scored.append((sentence, _cosine_similarity(q_vec, s_vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return " ".join(s for s, _ in scored[:max_sentences]).strip()


def _build_modified_history_for_context(
    history_for_llm: list[dict[str, str]],
    current_question: str,
    embedder: EmbeddingService,
) -> list[dict[str, str]]:
    """
    Keep full user messages; reduce assistant messages to top 3 sentences most relevant
    to current_question.
    """
    out: list[dict[str, str]] = []
    for msg in history_for_llm:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "assistant":
            reduced = _select_top_assistant_sentences(content, current_question, embedder, max_sentences=3)
            if reduced:
                out.append({"role": "assistant", "content": reduced})
        elif role == "user":
            out.append({"role": "user", "content": content})
    return out


def _build_query_with_context(current_question: str, history_for_context: list[dict[str, str]]) -> str:
    """Build merged embedding input: question + explicit context block from history."""
    if not history_for_context:
        return current_question
    lines = [current_question, "", "Context:"]
    for msg in history_for_context:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"user: {content}")
        elif role == "assistant":
            lines.append(f"assistant: {content}")
    return "\n".join(lines).strip()


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
    history_last_two_messages_reversed: list[dict[str, str]] | None = None,
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
        history: Optional list of {"role": "user"|"assistant", "content": str}; last 2 messages
            are included in the prompt (chronological).
        history_last_two_messages_reversed: Optional pre-built list of last 2 messages with
            first = most recent (last in conversation); reversed for chronological LLM history.

    Returns:
        LLM answer string; RAG_NO_INFORMATION_IN_DOCUMENT if DB returned chunks but reranking returned
        none (LLM not called); PROMPT_INCOMPLETE_RESPONSE if question is empty or vector search
        returned no chunks.
    """
    query_clean = (question or "").strip()
    if not query_clean:
        logger.debug("Question pipeline: empty question, returning incomplete response.")
        return PROMPT_INCOMPLETE_RESPONSE

    # History for LLM (chronological: oldest first).
    if history_last_two_messages_reversed is not None:
        history_for_llm = list(reversed(history_last_two_messages_reversed))
    else:
        history_last_two = (history[-2:] if history else [])
        history_for_llm = history_last_two

    coll = collection_name or VECTOR_COLLECTION_NAME
    embedder = embedding_service if embedding_service is not None else EmbeddingService()
    db = db_manager if db_manager is not None else DBManager(
        collection_name=coll, host=host, port=port
    )
    reranker = reranking_service if reranking_service is not None else RerankingService()
    chatter = llm_chatter if llm_chatter is not None else LLMChatter()

    if _question_contains_reference_word(query_clean):
        history_for_embed_and_prompt = _build_modified_history_for_context(
            history_for_llm, query_clean, embedder
        )
    else:
        history_for_embed_and_prompt = []
    query_for_embed = _build_query_with_context(query_clean, history_for_embed_and_prompt)
    query_embedding = embedder.embed_query(query_for_embed)
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
        logger.debug(
            "Question pipeline: rerank left no chunks, returning no-information response without LLM."
        )
        return RAG_NO_INFORMATION_IN_DOCUMENT

    raw_answer = chatter.chat({"chunks": top_chunks}, query_clean, history=history_for_embed_and_prompt)

    if raw_answer == PROMPT_INCOMPLETE_RESPONSE:
        return raw_answer
    return _trim_to_last_sentence(raw_answer)
