"""
Core embedding: embed a query (for similarity search) or a batch of chunks (for ingestion).

Uses AI_module.infra_layer.embedding_client. Outputs are ready for:
- Query: vector to pass to DBManager.search_similar.
- Batch: chunks with vector set, ready to pass to DBManager.insert_chunks.

When embedding a query, if conversation history is provided and the question contains
reference words (e.g. "this", "that", "the former"), the most recent message from history
is appended to the question for embedding to improve retrieval.
"""

import logging
import re
from typing import TYPE_CHECKING

from .chunk import Chunk

if TYPE_CHECKING:
    from AI_module.infra_layer.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

# Words that often refer to prior context; if present in the question and history is non-empty,
# we append the first (most recent) message from history to the question for embedding.
REFERENCE_WORDS = [
    "this", "that", "these", "those",
    "he", "she", "they", "him", "her", "them",
    "his", "her", "their", "its",
    "the former", "the latter", "one", "ones",
]


def _question_contains_reference_word(query: str) -> bool:
    """True if query contains any of REFERENCE_WORDS as a whole word/phrase (case-insensitive)."""
    if not query or not query.strip():
        return False
    text = query.lower().strip()
    for word in REFERENCE_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def _get_embedding_client() -> "EmbeddingClient":
    """Lazy import to avoid loading sentence-transformers at module load."""
    from AI_module.infra_layer.embedding_client import EmbeddingClient
    return EmbeddingClient()


class EmbeddingService:
    """
    Core service for embedding queries and chunks. Delegates to EmbeddingClient.
    Use this in the application layer; inject client for tests.
    """

    def __init__(self, *, client: "EmbeddingClient | None" = None) -> None:
        """
        Args:
            client: Optional EmbeddingClient; if None, a default instance is used.
        """
        self._client = client if client is not None else _get_embedding_client()

    def embed_query(self, query: str, history: list[dict[str, str]] | None = None) -> list[float]:
        """
        Embed a single query for similarity search.

        If history is non-empty and the query contains any REFERENCE_WORDS (e.g. "this", "that"),
        the content of the first (most recent) message in history is appended to the query before
        embedding to improve retrieval for referential questions.

        Args:
            query: User question or search text.
            history: Optional list of messages (e.g. last 4 reversed: first = most recent).
                     Each item: {"role": "user"|"assistant", "content": str}. Only the first
                     message's content is used when appending.

        Returns:
            Embedding vector (e.g. 384-dim). Ready to pass to
            DBManager.search_similar(query_embedding, top_k=...).
        """
        text_to_embed = query.strip()
        if not text_to_embed:
            return self._client.embed_query(query)

        if history and _question_contains_reference_word(query):
            first_msg = history[0] if history else None
            if isinstance(first_msg, dict):
                content = (first_msg.get("content") or "").strip()
                if content:
                    text_to_embed = f"{text_to_embed} {content}"
        return self._client.embed_query(text_to_embed)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed a batch of chunks; returns chunks with vector set.

        Args:
            chunks: Chunks from chunking (id and payload set, vector None).

        Returns:
            New list of Chunk with the same id and payload and vector filled.
        """
        if not chunks:
            return []
        texts = [c.payload.get("text", "") for c in chunks]
        vectors = self._client.embed_batch(texts)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"embed_batch returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        return [chunk.with_vector(vec) for chunk, vec in zip(chunks, vectors)]
