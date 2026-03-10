"""
DB manager for the RAG pipeline: Chunk-level abstraction over the vector DB.
Works only with Chunk or list[Chunk]; delegates to AI_module.infra_layer.db_client.VectorDBClient.
Application code should use DBManager; only DBManager and db_client tests use VectorDBClient directly.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from AI_module.config import (
    TOP_K_SIMILAR_CHUNKS,
    VECTOR_COLLECTION_NAME,
)
from AI_module.core.chunk import Chunk

if TYPE_CHECKING:
    from AI_module.infra_layer.db_client import VectorDBClient

logger = logging.getLogger(__name__)


def _get_client(
    collection_name: str = VECTOR_COLLECTION_NAME,
    host: str | None = None,
    port: int | None = None,
    **kwargs: Any,
) -> "VectorDBClient":
    """Create VectorDBClient; used when client is not injected."""
    from AI_module.infra_layer.db_client import VectorDBClient
    if host is not None and port is not None:
        return VectorDBClient(host=host, port=port, collection_name=collection_name, **kwargs)
    return VectorDBClient(collection_name=collection_name, **kwargs)


class DBManager:
    """
    Chunk-oriented interface to the vector DB. All operations take or return Chunk instances.
    Delegates to VectorDBClient for actual storage and search.
    """

    def __init__(
        self,
        collection_name: str = VECTOR_COLLECTION_NAME,
        *,
        client: "VectorDBClient | None" = None,
        host: str | None = None,
        port: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            collection_name: Name of the collection.
            client: If provided, use this VectorDBClient; otherwise create one from config or host/port.
            host: Qdrant host when not using an injected client (e.g. for tests).
            port: Qdrant port when not using an injected client.
        """
        if client is not None:
            self._client = client
        else:
            self._client = _get_client(collection_name=collection_name, host=host, port=port, **kwargs)

    def insert_chunks(self, chunks: list[Chunk] | Chunk) -> None:
        """
        Insert chunks into the collection. Each chunk must have vector set.
        Call delete_all first if you need to replace the collection contents.

        Args:
            chunks: One or more Chunk instances with vector set (e.g. from EmbeddingService().embed_chunks).
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        if not chunks:
            raise ValueError("chunks must be non-empty")
        ids = [c.id for c in chunks]
        vectors = [c.vector for c in chunks]
        if any(v is None for v in vectors):
            raise ValueError("all chunks must have vector set for insert_chunks")
        embeddings = [v for v in vectors if v is not None]
        metadatas = [dict(c.payload) for c in chunks]
        self._client.insert_chunks(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger.info("DBManager: inserted %d chunks into %r", len(chunks), self._client._collection_name)

    def delete_all(self) -> None:
        """Remove all chunks from the collection."""
        self._client.delete_all()

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_SIMILAR_CHUNKS,
        *,
        include_scores: bool = True,
    ) -> dict[str, Any]:
        """
        Return the top-k chunks most similar to the query embedding.

        Args:
            query_embedding: Embedding vector of the query (e.g. from EmbeddingService().embed_query).
            top_k: Number of similar chunks to return.
            include_scores: Whether to include similarity scores in the result.

        Returns:
            Dict with "chunks" (list of Chunk with vector=None) and optionally "scores" (list of float).
            Chunk ids and payloads come from the stored metadata.
        """
        raw = self._client.search_similar(
            query_embedding,
            top_k=top_k,
            include_metadatas=True,
            include_distances=include_scores,
        )
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        scores = raw.get("distances") if include_scores else []
        if len(metadatas) != len(ids):
            metadatas = [{}] * len(ids)
        chunks = []
        for i, cid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            payload = dict(meta)
            chunks.append(Chunk(id=str(cid), payload=payload, vector=None))
        out: dict[str, Any] = {"chunks": chunks}
        if include_scores and scores is not None:
            out["scores"] = list(scores)
        return out
