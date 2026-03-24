"""
Unit tests for core.db_manager (DBManager). VectorDBClient mocked; only DBManager logic is tested.
DBManager is the Chunk-level abstraction; it delegates to VectorDBClient.
For integration tests with real client see db_manager_integration_test.py.
Run: python -m pytest AI_module/tests/core_tests/db_manager_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunk import Chunk
from AI_module.core.db_manager import DBManager


def _make_chunk(chunk_id: str, text: str, vector: list[float] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        payload={"text": text, "source": "test", "page": 1, "chunk_index": 0, "path": ""},
        vector=vector,
    )


@pytest.fixture
def mock_client():
    """VectorDBClient mock; DBManager delegates to it."""
    client = MagicMock()
    client._collection_name = "test_coll"
    client.search_similar.return_value = {"ids": [], "metadatas": [], "distances": []}
    return client


def test_insert_chunks_calls_client_insert_chunks(mock_client):
    """DBManager.insert_chunks extracts ids, vectors, metadatas and calls client.insert_chunks."""
    chunks = [
        _make_chunk("a", "text a", [0.1] * 384),
        _make_chunk("b", "text b", [0.2] * 384),
    ]
    manager = DBManager(client=mock_client)
    manager.insert_chunks(chunks)
    mock_client.insert_chunks.assert_called_once()
    call_kw = mock_client.insert_chunks.call_args[1]
    assert call_kw["ids"] == ["a", "b"]
    assert len(call_kw["embeddings"]) == 2
    assert call_kw["embeddings"][0] == [0.1] * 384
    assert call_kw["metadatas"][0]["text"] == "text a"
    assert call_kw["metadatas"][1]["text"] == "text b"


def test_insert_chunks_single_chunk(mock_client):
    """insert_chunks accepts a single Chunk (wraps in list)."""
    one = _make_chunk("only", "content", [0.0] * 384)
    manager = DBManager(client=mock_client)
    manager.insert_chunks(one)
    mock_client.insert_chunks.assert_called_once()
    assert mock_client.insert_chunks.call_args[1]["ids"] == ["only"]


def test_insert_chunks_empty_raises(mock_client):
    """insert_chunks with empty list raises."""
    manager = DBManager(client=mock_client)
    with pytest.raises(ValueError, match="chunks must be non-empty"):
        manager.insert_chunks([])


def test_insert_chunks_missing_vector_raises(mock_client):
    """insert_chunks when a chunk has no vector raises."""
    chunks = [_make_chunk("a", "x", [0.0] * 384), _make_chunk("b", "y", None)]
    manager = DBManager(client=mock_client)
    with pytest.raises(ValueError, match="all chunks must have vector set"):
        manager.insert_chunks(chunks)


def test_delete_all_calls_client_delete_all(mock_client):
    """DBManager.delete_all delegates to client.delete_all."""
    manager = DBManager(client=mock_client)
    manager.delete_all()
    mock_client.delete_all.assert_called_once()


def test_search_similar_calls_client_and_returns_chunks_with_scores(mock_client):
    """search_similar delegates to client.search_similar and returns Chunk list + scores."""
    mock_client.search_similar.return_value = {
        "ids": ["id1", "id2"],
        "metadatas": [{"text": "first", "chunk_id": "id1"}, {"text": "second", "chunk_id": "id2"}],
        "distances": [0.9, 0.7],
    }
    manager = DBManager(client=mock_client)
    result = manager.search_similar([0.1] * 384, top_k=2)
    mock_client.search_similar.assert_called_once()
    call_args = mock_client.search_similar.call_args
    assert call_args[0][0] == [0.1] * 384
    assert call_args[1]["top_k"] == 2
    assert call_args[1]["include_metadatas"] is True
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "id1" and result["chunks"][0].payload["text"] == "first"
    assert result["chunks"][1].id == "id2" and result["chunks"][1].payload["text"] == "second"
    assert result["scores"] == [0.9, 0.7]
    assert result["chunks"][0].vector is None


def test_search_similar_empty_result(mock_client):
    """search_similar with no hits returns empty chunks and scores."""
    mock_client.search_similar.return_value = {"ids": [], "metadatas": [], "distances": []}
    manager = DBManager(client=mock_client)
    result = manager.search_similar([0.0] * 384, top_k=5, include_scores=True)
    assert result["chunks"] == []
    assert result["scores"] == []


def test_search_similar_include_scores_false(mock_client):
    """When include_scores=False, client is called with include_distances=False and result has no scores."""
    mock_client.search_similar.return_value = {"ids": ["x"], "metadatas": [{"text": "y"}]}
    manager = DBManager(client=mock_client)
    result = manager.search_similar([0.0] * 384, top_k=1, include_scores=False)
    assert mock_client.search_similar.call_args[1]["include_distances"] is False
    assert "chunks" in result
    assert "scores" not in result or result.get("scores") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
