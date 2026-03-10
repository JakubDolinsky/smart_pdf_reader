"""
Unit tests for VectorDBClient using in-memory backend only (no Qdrant server).
For integration tests against real Qdrant server see db_client_integration_test.py.
Run directly: python AI_module/tests/infra_layer_tests/db_client_unit_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/db_client_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.infra_layer.db_client import VectorDBClient


@pytest.fixture(autouse=True)
def force_memory_storage():
    """Force in-memory backend for all VectorDBClient instances in this module."""
    with patch("AI_module.infra_layer.db_client.STORAGE_TYPE", "memory"):
        yield


@pytest.fixture
def client():
    """In-memory VectorDBClient for each test (no server)."""
    return VectorDBClient()


def test_insert_chunks_success(client):
    ids = ["chunk_0"]
    embeddings = [[0.1, 0.2, 0.3, 0.4]]
    metadatas = [{"text": "hello world"}]
    client.insert_chunks(ids=ids, embeddings=embeddings, metadatas=metadatas)
    result = client.search_similar(embeddings[0], top_k=1)
    assert result["ids"] == ["chunk_0"]
    assert len(result["metadatas"]) == 1
    assert result["metadatas"][0]["text"] == "hello world"
    assert result["metadatas"][0]["chunk_id"] == "chunk_0"
    assert len(result["distances"]) == 1


def test_insert_chunks_without_metadatas(client):
    ids = ["a", "b"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    client.insert_chunks(ids=ids, embeddings=embeddings)
    result = client.search_similar([1.0, 0.0], top_k=2)
    assert result["ids"] == ["a", "b"]
    assert result["metadatas"][0]["chunk_id"] == "a"
    assert result["metadatas"][1]["chunk_id"] == "b"


def test_insert_chunks_length_mismatch_raises(client):
    with pytest.raises(ValueError, match="same length"):
        client.insert_chunks(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{}],
        )
    with pytest.raises(ValueError, match="same length"):
        client.insert_chunks(
            ids=["a"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_insert_chunks_empty_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.insert_chunks(ids=[], embeddings=[])
    with pytest.raises(ValueError, match="non-empty"):
        client.insert_chunks(ids=["a"], embeddings=[])


def test_insert_chunks_wrong_vector_dimension_raises(client):
    """Insert with one dimension, then insert with different dimension raises ValueError."""
    client.insert_chunks(ids=["a"], embeddings=[[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="vector size|embeddings of size"):
        client.insert_chunks(ids=["b"], embeddings=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="vector size|embeddings of size"):
        client.insert_chunks(ids=["c"], embeddings=[[1.0, 0.0, 0.0, 0.0]])


def test_delete_all_removes_all(client):
    ids = ["c1", "c2"]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    client.insert_chunks(ids=ids, embeddings=embeddings)
    client.delete_all()
    result = client.search_similar([1.0, 0.0, 0.0], top_k=10)
    assert result["ids"] == [] and result["metadatas"] == [] and result["distances"] == []


def test_delete_all_empty_collection_no_error(client):
    client.delete_all()


def test_insert_after_delete_all(client):
    client.insert_chunks(ids=["old"], embeddings=[[1.0, 0.0]])
    client.delete_all()
    client.insert_chunks(ids=["new"], embeddings=[[0.0, 1.0]])
    result = client.search_similar([0.0, 1.0], top_k=5)
    assert result["ids"] == ["new"]


def test_search_similar_returns_top_k(client):
    ids = ["near", "mid", "far"]
    embeddings = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    client.insert_chunks(ids=ids, embeddings=embeddings)
    result = client.search_similar([1.0, 0.0, 0.0], top_k=2)
    assert len(result["ids"]) == 2 and result["ids"][0] == "near"


def test_search_similar_respects_top_k(client):
    ids = [f"chunk_{i}" for i in range(5)]
    embeddings = [[float(i), 0.0, 0.0] for i in range(5)]
    client.insert_chunks(ids=ids, embeddings=embeddings)
    result = client.search_similar([4.0, 0.0, 0.0], top_k=2)
    assert len(result["ids"]) == 2


def test_search_similar_include_metadatas_false(client):
    client.insert_chunks(ids=["x"], embeddings=[[1.0, 0.0]], metadatas=[{"text": "secret"}])
    result = client.search_similar(
        [1.0, 0.0], top_k=1, include_metadatas=False, include_distances=True
    )
    assert "metadatas" not in result and "ids" in result and "distances" in result


def test_search_similar_include_distances_false(client):
    client.insert_chunks(ids=["y"], embeddings=[[0.0, 1.0]])
    result = client.search_similar(
        [0.0, 1.0], top_k=1, include_metadatas=True, include_distances=False
    )
    assert "distances" not in result and "ids" in result and "metadatas" in result


def test_search_similar_empty_collection(client):
    result = client.search_similar([1.0, 0.0, 0.0], top_k=5)
    assert result["ids"] == [] and result["metadatas"] == [] and result["distances"] == []


def test_search_similar_scores_higher_is_more_similar(client):
    client.insert_chunks(
        ids=["exact", "other"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    result = client.search_similar([1.0, 0.0, 0.0], top_k=2)
    assert result["ids"][0] == "exact" and result["distances"][0] >= result["distances"][1]


def test_storage_type_server_requires_host():
    with patch("AI_module.infra_layer.db_client.STORAGE_TYPE", "server"):
        with patch("AI_module.infra_layer.db_client.QDRANT_LOCAL_HOST", None):
            with pytest.raises(ValueError, match="QDRANT_LOCAL_HOST"):
                VectorDBClient()
        with patch("AI_module.infra_layer.db_client.QDRANT_LOCAL_HOST", ""):
            with pytest.raises(ValueError, match="QDRANT_LOCAL_HOST"):
                VectorDBClient()


def test_storage_type_path_requires_persist_directory():
    with patch("AI_module.infra_layer.db_client.STORAGE_TYPE", "path"):
        with patch("AI_module.infra_layer.db_client.QDRANT_PERSIST_DIRECTORY", None):
            with pytest.raises(ValueError, match="QDRANT_PERSIST_DIRECTORY"):
                VectorDBClient()
        with patch("AI_module.infra_layer.db_client.QDRANT_PERSIST_DIRECTORY", ""):
            with pytest.raises(ValueError, match="QDRANT_PERSIST_DIRECTORY"):
                VectorDBClient()


def test_custom_collection_name(client):
    custom = VectorDBClient(collection_name="my_collection")
    custom.insert_chunks(ids=["only_here"], embeddings=[[1.0, 0.0]])
    result = custom.search_similar([1.0, 0.0], top_k=1)
    assert result["ids"] == ["only_here"]
    default_result = client.search_similar([1.0, 0.0], top_k=1)
    assert default_result["ids"] == []


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
