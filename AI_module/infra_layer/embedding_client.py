"""
Embedding client for RAG: embed document chunks (ingestion) and user queries (search).
Uses intfloat/multilingual-e5-small (384 dimensions) with query/passage prefixes.
"""

import logging
from typing import Any

from AI_module.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# E5 models expect "query: " / "passage: " prefixes for retrieval
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class EmbeddingClient:
    """
    Client for computing text embeddings with multilingual-e5-small (dim 384).
    - embed_chunks: batch embed document chunks; returns list of vectors (ids/metadatas built in ingestion layer).
    - embed_query: embed a single question for similarity search.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id (default: intfloat/multilingual-e5-small).
            dimension: Expected embedding dimension (384 for multilingual-e5-small).
        """
        self._model_name = model_name
        self._dimension = dimension
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model and verify dimension matches config."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "EmbeddingClient requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e
            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: config EMBEDDING_DIMENSION is {self._dimension}, "
                    f"but model {self._model_name!r} has dimension {actual_dim}. "
                    "Update EMBEDDING_DIMENSION in config to match the model, or use a different model."
                )
        return self._model

    def embed_batch(self, chunks: list[str]) -> list[list[float]]:
        """
        Embed a batch of document chunks for ingestion. Returns only vectors;
        ids and metadatas are created in the ingestion layer when building DB insert payloads.

        Args:
            chunks: List of text chunks (e.g. from chunking a PDF).

        Returns:
            List of 384-dim vectors, one per chunk, in the same order as chunks.
        """
        if not chunks:
            return []

        model = self._get_model()
        prefixed = [f"{PASSAGE_PREFIX}{t}" for t in chunks]
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=len(chunks) > 50,
        )
        vectors = [emb.tolist() for emb in embeddings]
        logger.info("Embedded %d chunks (model=%s)", len(chunks), self._model_name)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query (e.g. user question) for similarity search against
        chunk embeddings produced by embed_chunks.

        Args:
            query: Question or search text.

        Returns:
            Single 384-dim vector; use with VectorDBClient.search_similar.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        model = self._get_model()
        prefixed = f"{QUERY_PREFIX}{query.strip()}"
        embedding = model.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Embedding dimension (384 for multilingual-e5-small)."""
        return self._dimension
