"""
Integration tests for VectorDBClient against a real Qdrant server.
Runs bootstrap.ensure_db_ready() before tests (Docker + Qdrant start). Skips if server unavailable.
Uses test config (STORAGE_TYPE=server, QDRANT_HOST/PORT, VECTOR_COLLECTION_NAME_TEST).
For unit tests (in-memory only) see db_client_unit_test.py.
Run directly: python AI_module/tests/infra_layer_tests/db_client_integration_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/db_client_integration_test.py -v
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
from AI_module.tests.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_COLLECTION_NAME_TEST,
)
from AI_module.tests.db_bootstrap import ensure_db_ready, get_resolved_host_port, stop_qdrant_server
from AI_module.infra_layer.db_client import VectorDBClient


def _delete_test_collections_only():
    try:
        from qdrant_client import QdrantClient
        host, port = get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)
        if host is None:
            host, port = QDRANT_HOST, QDRANT_PORT
        c = QdrantClient(host=host, port=port)
        for col in c.get_collections().collections:
            if "test" in col.name.lower():
                c.delete_collection(col.name)
    except Exception:
        pass


def _recreate_test_collection():
    _delete_test_collections_only()


@pytest.fixture(scope="session", autouse=True)
def start_qdrant_server_before_tests():
    """Run bootstrap: ensure Docker and Qdrant server are up before any integration test; stop Qdrant container after tests (like stop_app_db.bat)."""
    ok = ensure_db_ready(host=QDRANT_HOST, port=QDRANT_PORT)
    assert ok, f"Qdrant server not reachable at {QDRANT_HOST}:{QDRANT_PORT} and could not be started."
    yield
    stop_qdrant_server()


@pytest.fixture(scope="session")
def qdrant_host_port():
    """Resolved (host, port) for Qdrant after bootstrap, or (None, None) if not reachable."""
    return get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)


@pytest.fixture(scope="session", autouse=True)
def cleanup_qdrant_after_test_cycle():
    yield
    _delete_test_collections_only()


@pytest.fixture
def client(qdrant_host_port):
    """VectorDBClient against real Qdrant server (test collection). Skips if server not reachable after bootstrap."""
    host, port = qdrant_host_port
    assert host is not None and port is not None, "Qdrant host/port could not be resolved after bootstrap."
    _recreate_test_collection()
    c = VectorDBClient(
        host=host,
        port=port,
        collection_name=VECTOR_COLLECTION_NAME_TEST,
    )
    yield c
    _recreate_test_collection()


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


def test_insert_chunks_length_mismatch_raises(client):
    with pytest.raises(ValueError, match="same length"):
        client.insert_chunks(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{}],
        )


def test_insert_chunks_empty_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.insert_chunks(ids=[], embeddings=[])


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


def test_custom_collection_name(client, qdrant_host_port):
    host, port = qdrant_host_port
    custom = VectorDBClient(
        host=host,
        port=port,
        collection_name="my_integration_test_collection",
    )
    custom.insert_chunks(ids=["only_here"], embeddings=[[1.0, 0.0]])
    result = custom.search_similar([1.0, 0.0], top_k=1)
    assert result["ids"] == ["only_here"]
    default_result = client.search_similar([1.0, 0.0], top_k=1)
    assert default_result["ids"] == []
    # custom collection name contains "test", so session cleanup will delete it


def test_persistence_after_reconnect(client, qdrant_host_port):
    """Data inserted with one client is visible to a new client (same collection) after reconnect."""
    client.insert_chunks(
        ids=["persisted_1", "persisted_2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"text": "first"}, {"text": "second"}],
    )
    host, port = qdrant_host_port
    client2 = VectorDBClient(
        host=host,
        port=port,
        collection_name=VECTOR_COLLECTION_NAME_TEST,
    )
    result = client2.search_similar([1.0, 0.0, 0.0], top_k=2)
    assert result["ids"] == ["persisted_1", "persisted_2"]
    assert result["metadatas"][0]["text"] == "first"
    assert result["metadatas"][1]["text"] == "second"


def test_cleanup_removes_all_data(client, qdrant_host_port):
    """After delete_all, a new client sees no data in the collection."""
    client.insert_chunks(ids=["to_remove"], embeddings=[[1.0, 0.0, 0.0]])
    result_before = client.search_similar([1.0, 0.0, 0.0], top_k=1)
    assert result_before["ids"] == ["to_remove"]
    client.delete_all()
    host, port = qdrant_host_port
    client2 = VectorDBClient(
        host=host,
        port=port,
        collection_name=VECTOR_COLLECTION_NAME_TEST,
    )
    result_after = client2.search_similar([1.0, 0.0, 0.0], top_k=1)
    assert result_after["ids"] == []
    assert result_after["metadatas"] == []
    assert result_after["distances"] == []


def main():
    # Run bootstrap (Docker + Qdrant) before pytest
    print("Ensuring Qdrant server is running (bootstrap.ensure_db_ready)...", flush=True)
    if ensure_db_ready(host=QDRANT_HOST, port=QDRANT_PORT):
        print("Qdrant server ready.", flush=True)
    else:
        print(
            f"Qdrant server not reachable at {QDRANT_HOST}:{QDRANT_PORT}; tests will be skipped.",
            flush=True,
        )
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
