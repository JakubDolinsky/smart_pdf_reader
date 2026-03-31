"""
Integration tests for core.db_manager with real VectorDBClient against Qdrant server.
Uses db_bootstrap to start Docker/Qdrant; DBManager delegates to VectorDBClient (host/port).
Run: python -m pytest AI_module/tests/core_tests/db_manager_integration_test.py -v
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
from AI_module.core.db_manager import DBManager
from AI_module.infra_layer.db_client import VectorDBClient
from AI_module.tests.config import QDRANT_HOST, QDRANT_PORT
from AI_module.tests.db_bootstrap import (
    ensure_db_ready,
    get_resolved_host_port,
    stop_qdrant_server,
)

COLLECTION_NAME = "test_db_manager"


def _delete_db_manager_collection():
    try:
        from qdrant_client import QdrantClient
        host, port = get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)
        if host is None:
            host, port = QDRANT_HOST, QDRANT_PORT
        c = QdrantClient(host=host, port=port)
        if c.collection_exists(COLLECTION_NAME):
            c.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def start_qdrant_server_before_tests():
    """Ensure Docker and Qdrant are up before tests; stop Qdrant after session."""
    ok = ensure_db_ready(host=QDRANT_HOST, port=QDRANT_PORT)
    assert ok, f"Qdrant server not reachable at {QDRANT_HOST}:{QDRANT_PORT} and could not be started."
    yield
    stop_qdrant_server()


@pytest.fixture(scope="session")
def qdrant_host_port():
    """Resolved (host, port) for Qdrant after bootstrap, or (None, None) if not reachable."""
    return get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)


@pytest.fixture
def client(qdrant_host_port):
    """VectorDBClient connected to real Qdrant (test collection). Skips if server not reachable."""
    host, port = qdrant_host_port
    assert host is not None and port is not None, "Qdrant host/port could not be resolved after bootstrap."
    _delete_db_manager_collection()
    c = VectorDBClient(
        host=host,
        port=port,
        collection_name=COLLECTION_NAME,
    )
    yield c
    _delete_db_manager_collection()


def _make_chunk(chunk_id: str, text: str, vector: list[float]) -> Chunk:
    return Chunk(
        id=chunk_id,
        payload={"text": text, "source": "test", "page": 1, "chunk_index": 0, "path": ""},
        vector=vector,
    )


def test_insert_chunks_then_search_returns_chunks(client):
    """DBManager with Qdrant client: insert_chunks then search_similar returns Chunk list with scores."""
    manager = DBManager(client=client)
    dim = 384
    chunks = [
        _make_chunk("c1", "First passage.", [1.0] + [0.0] * (dim - 1)),
        _make_chunk("c2", "Second passage.", [0.0] * (dim - 1) + [1.0]),
        _make_chunk("c3", "Third passage.", [0.0] * (dim // 2) + [1.0] + [0.0] * (dim // 2 - 1)),
    ]
    manager.insert_chunks(chunks)
    query = [1.0] + [0.0] * (dim - 1)
    result = manager.search_similar(query, top_k=2)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "c1"
    assert result["chunks"][0].payload["text"] == "First passage."
    assert result["chunks"][0].vector is None
    assert len(result["scores"]) == 2
    assert result["scores"][0] >= result["scores"][1]


def test_delete_all_then_search_empty(client):
    """delete_all clears the collection; search_similar returns empty chunks."""
    manager = DBManager(client=client)
    manager.insert_chunks(_make_chunk("x", "Only one.", [0.5] * 384))
    manager.delete_all()
    result = manager.search_similar([0.5] * 384, top_k=10)
    assert result["chunks"] == []
    assert result["scores"] == []


def test_insert_single_chunk(client):
    """insert_chunks accepts a single Chunk and stores it."""
    manager = DBManager(client=client)
    one = _make_chunk("single", "Single chunk.", [0.1] * 384)
    manager.insert_chunks(one)
    result = manager.search_similar([0.1] * 384, top_k=1)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "single"
    assert result["chunks"][0].payload["text"] == "Single chunk."


def test_search_similar_respects_top_k(client):
    """search_similar returns at most top_k chunks and scores."""
    manager = DBManager(client=client)
    dim = 384
    chunks = [
        _make_chunk(f"chunk_{i}", f"Text {i}.", [float(i)] + [0.0] * (dim - 1))
        for i in range(5)
    ]
    manager.insert_chunks(chunks)
    result = manager.search_similar([2.0] + [0.0] * (dim - 1), top_k=2)
    assert len(result["chunks"]) == 2
    assert len(result["scores"]) == 2
    assert all(isinstance(c, Chunk) for c in result["chunks"])
    assert all(c.vector is None for c in result["chunks"])
    assert all(isinstance(s, float) for s in result["scores"])


def test_search_similar_include_scores_false(client):
    """search_similar with include_scores=False returns chunks without scores."""
    manager = DBManager(client=client)
    manager.insert_chunks(_make_chunk("a", "Content.", [0.5] * 384))
    result = manager.search_similar([0.5] * 384, top_k=1, include_scores=False)
    assert "chunks" in result
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "a"
    assert result["chunks"][0].payload["text"] == "Content."
    assert "scores" not in result


def test_search_similar_empty_collection_returns_empty(client):
    """search_similar on empty collection returns empty chunks and scores."""
    manager = DBManager(client=client)
    result = manager.search_similar([0.1] * 384, top_k=10)
    assert result["chunks"] == []
    assert result["scores"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
