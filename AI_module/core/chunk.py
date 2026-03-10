"""
Chunk model for the RAG pipeline.

Represents a document chunk used across ingestion, pipeline, and infrastructure.
- id: set during PDF processing when cutting the document into chunks.
- vector: set during embedding (optional until embedded).
- payload: metadata from PDF processing (e.g. text, source, page).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    """
    A single chunk in the pipeline: id and payload from PDF processing,
    vector filled in by the embedding step.
    """

    id: str
    payload: dict[str, Any]
    vector: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("chunk id must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError("chunk payload must be a dict")

    def with_vector(self, vector: list[float]) -> Chunk:
        """Return a new Chunk with the same id and payload and the given vector."""
        return Chunk(id=self.id, payload=dict(self.payload), vector=vector)
