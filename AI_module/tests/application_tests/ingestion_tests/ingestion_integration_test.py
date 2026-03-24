"""
Integration tests for ingestion.run_ingestion with real Qdrant server and real embedding.
Use test config (STORAGE_TYPE=server, Qdrant host/port, INGESTION_TEST_COLLECTION); db_bootstrap starts Docker/Qdrant.
Chunking is mocked (no test PDFs).
Run: python -m pytest AI_module/tests/application_tests/ingestion_tests/ingestion_integration_test.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_file = Path(__file__).resolve()
_root = _file.parents[3]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunk import Chunk
from AI_module.core.db_manager import DBManager
from AI_module.application.ingestion.ingestion import run_ingestion
from AI_module.infra_layer.db_client import VectorDBClient
from AI_module.tests.config import (
    INGESTION_TEST_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    STORAGE_TYPE,
)
from AI_module.tests.db_bootstrap import (
    ensure_db_ready,
    get_resolved_host_port,
    stop_qdrant_server,
)


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        payload={"text": text, "source": "test_doc", "page": 1, "chunk_index": 0, "path": ""},
        vector=None,
    )


@pytest.fixture(scope="session", autouse=True)
def qdrant_server_and_config():
    """
    Ensure Qdrant is up via db_bootstrap (like db_client_integration_test).
    Patch ingestion and db_client to use test config STORAGE_TYPE so run_ingestion uses Qdrant DB.
    Restore and stop Qdrant after the session.
    """
    ensure_db_ready(host=QDRANT_HOST, port=QDRANT_PORT)

    patch_ingestion = patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", STORAGE_TYPE)
    patch_db_client = patch("AI_module.infra_layer.db_client.STORAGE_TYPE", STORAGE_TYPE)
    patch_ingestion.start()
    patch_db_client.start()
    try:
        yield
    finally:
        patch_ingestion.stop()
        patch_db_client.stop()
        stop_qdrant_server()


@pytest.fixture(scope="session")
def qdrant_host_port():
    """Resolved (host, port) for Qdrant after bootstrap, or (None, None) if not reachable."""
    return get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)


@pytest.fixture
def db_manager(qdrant_host_port):
    """DBManager over VectorDBClient for the ingestion test collection. Skips if Qdrant not reachable."""
    host, port = qdrant_host_port
    if host is None or port is None:
        pytest.skip("Qdrant server not reachable")
    client = VectorDBClient(
        host=host,
        port=port,
        collection_name=INGESTION_TEST_COLLECTION,
    )
    return DBManager(client=client)


@pytest.fixture(autouse=True)
def clear_ingestion_collection(qdrant_host_port):
    """Clear the ingestion test collection before each test."""
    yield
    try:
        from qdrant_client import QdrantClient
        host, port = get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)
        if host is None:
            host, port = QDRANT_HOST, QDRANT_PORT
        c = QdrantClient(host=host, port=port)
        if c.collection_exists(INGESTION_TEST_COLLECTION):
            c.delete_collection(INGESTION_TEST_COLLECTION)
    except Exception:
        pass


def test_run_ingestion_inserts_and_search_returns_data(db_manager):
    """Full pipeline: run_ingestion (mocked chunks) -> data is in DB and searchable via DBManager."""
    chunks = [
        _make_chunk("ing_chunk_1", "The capital of France is Paris."),
        _make_chunk("ing_chunk_2", "Photosynthesis uses sunlight and water."),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
            code = run_ingestion(pdf_dir=tmp, collection_name=INGESTION_TEST_COLLECTION)
    assert code == 0

    from AI_module.core.embedding import EmbeddingService
    embedder = EmbeddingService()
    query_vec = embedder.embed_query("What is the capital of France?")
    result = db_manager.search_similar(query_vec, top_k=2)
    assert len(result["chunks"]) >= 1
    chunk_texts = [c.payload.get("text", "") for c in result["chunks"]]
    chunk_ids = [c.id for c in result["chunks"]]
    assert any("Paris" in t or "France" in t for t in chunk_texts) or "ing_chunk_1" in chunk_ids


def test_run_ingestion_delete_then_insert_replaces_previous_batch(db_manager):
    """Second run_ingestion clears the collection and only the new batch is present."""
    batch1 = [
        _make_chunk("only_in_first", "This content is only in the first run."),
    ]
    batch2 = [
        _make_chunk("only_in_second", "This content is only in the second run."),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=batch1):
            code1 = run_ingestion(pdf_dir=tmp, collection_name=INGESTION_TEST_COLLECTION)
    assert code1 == 0

    with tempfile.TemporaryDirectory() as tmp2:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=batch2):
            code2 = run_ingestion(pdf_dir=tmp2, collection_name=INGESTION_TEST_COLLECTION)
    assert code2 == 0

    from AI_module.core.embedding import EmbeddingService
    embedder = EmbeddingService()
    query_vec = embedder.embed_query("This content is only in the second run.")
    result = db_manager.search_similar(query_vec, top_k=5)
    result_ids = [c.id for c in result["chunks"]]
    assert "only_in_second" in result_ids
    assert "only_in_first" not in result_ids


def test_run_ingestion_delete_all_before_insert_clears_collection(db_manager):
    """After run_ingestion, only the inserted batch exists (delete_all was called first)."""
    chunks = [
        _make_chunk("single_chunk", "Unique sentence for deletion test."),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
            run_ingestion(pdf_dir=tmp, collection_name=INGESTION_TEST_COLLECTION)

    from AI_module.core.embedding import EmbeddingService
    embedder = EmbeddingService()
    query_vec = embedder.embed_query("Unique sentence for deletion test.")
    result = db_manager.search_similar(query_vec, top_k=10)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "single_chunk"


def test_run_ingestion_metadatas_persisted(db_manager):
    """Inserted chunks have correct metadata (text, source) in DB; query via DBManager returns Chunks."""
    chunks = [
        _make_chunk("meta_1", "Metadata check content."),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
            run_ingestion(pdf_dir=tmp, collection_name=INGESTION_TEST_COLLECTION)

    from AI_module.core.embedding import EmbeddingService
    embedder = EmbeddingService()
    query_vec = embedder.embed_query("Metadata check content.")
    result = db_manager.search_similar(query_vec, top_k=1)
    assert len(result["chunks"]) == 1
    c = result["chunks"][0]
    assert c.id == "meta_1"
    assert c.payload.get("text") == "Metadata check content."
    assert c.payload.get("source") == "test_doc"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
