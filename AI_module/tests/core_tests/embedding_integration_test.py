"""
Integration tests for embedding.py (EmbeddingService) with real EmbeddingClient.

- Uses real sentence-transformers model (may download on first run). Slow.
- For fast unit tests with mocks see embedding_unit_test.py.

Run directly: python AI_module/tests/core_tests/embedding_integration_test.py
Or: python -m pytest AI_module/tests/core_tests/embedding_integration_test.py -v
"""

import sys
from pathlib import Path

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest

from AI_module.core.chunk import Chunk
from AI_module.core.embedding import EmbeddingService


def _make_chunk(chunk_id: str, text: str, **payload_extra) -> Chunk:
    payload = {"text": text, "source": "test_doc", "page": 1, "chunk_index": 0, "chapter": ""}
    payload.update(payload_extra)
    return Chunk(id=chunk_id, payload=payload, vector=None)


# ---------------------------------------------------------------------------
# Real EmbeddingClient (no mock)
# ---------------------------------------------------------------------------


def test_embed_query_real_client_returns_384_dim_vector():
    """EmbeddingService without client uses real EmbeddingClient and returns 384-dim vector."""
    service = EmbeddingService()
    result = service.embed_query("A short question.")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_embed_chunks_real_client_fills_vectors():
    """EmbeddingService.embed_chunks without client uses real EmbeddingClient and fills chunk vectors."""
    service = EmbeddingService()
    chunks = [_make_chunk("c1", "A single chunk for embedding.")]
    result = service.embed_chunks(chunks)
    assert len(result) == 1
    assert result[0].id == "c1"
    assert result[0].vector is not None
    assert len(result[0].vector) == 384
    assert all(isinstance(x, float) for x in result[0].vector)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
