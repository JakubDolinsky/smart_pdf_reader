"""
Core embedding: embed a query (for similarity search) or a batch of chunks (for ingestion).

Uses AI_module.infra_layer.embedding_client. Outputs are ready for:
- Query: vector to pass to DBManager.search_similar.
- Batch: chunks with vector set, ready to pass to DBManager.insert_chunks.
"""

import logging
from typing import TYPE_CHECKING

from .chunk import Chunk

if TYPE_CHECKING:
    from AI_module.infra_layer.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


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

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query for similarity search.

        Args:
            query: User question or search text.

        Returns:
            Embedding vector (e.g. 384-dim). Ready to pass to
            DBManager.search_similar(query_embedding, top_k=...).
        """
        q = (query or "").strip()
        return self._client.embed_query(q)

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
