"""
Reranking client for RAG: score (query, chunk text) pairs with a cross-encoder.
Used by core.reranking to produce relevance scores; core layer prepares pairs and builds the output dict.
Model: e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 (higher score = more relevant).
"""

from __future__ import annotations

import logging
from typing import Any

from AI_module.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)


class RerankingClient:
    """
    Client for scoring (query, text) pairs with a cross-encoder.
    Input: list of (query, chunk_text) pairs prepared by core.reranking.
    Output: list of relevance scores in the same order as the pairs.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder model id (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
        """
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers CrossEncoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "RerankingClient requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e
            logger.info("Loading reranker model %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score each (query, chunk_text) pair with the cross-encoder.
        Higher score means more relevant to the query.

        Args:
            pairs: List of (query, chunk_text) tuples as prepared by core.reranking.

        Returns:
            List of relevance scores, one per pair, in the same order as pairs.
        """
        if not pairs:
            return []

        model = self._get_model()
        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        else:
            scores = list(scores)
        return [float(s) for s in scores]
