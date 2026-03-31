"""
Integration tests for question_pipeline.answer_question with real Qdrant, embedding, reranker, and LLM.
Uses test collection; db_bootstrap starts Docker/Qdrant. Chunks are inserted with real embedding, then answer_question is called.
Run: python -m pytest AI_module/tests/application_tests/question_pipeline_tests/question_pipeline_integration_test.py -v
"""

import sys
from pathlib import Path

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
from AI_module.core.embedding import EmbeddingService
from AI_module.application.question_pipeline.question_pipeline_orchestration import answer_question
from AI_module.core.llm_chatter import PROMPT_INCOMPLETE_RESPONSE
from AI_module.infra_layer.db_client import VectorDBClient
from AI_module.tests.config import QDRANT_HOST, QDRANT_PORT, VECTOR_COLLECTION_NAME_TEST
from AI_module.tests.db_bootstrap import (
    ensure_db_ready,
    get_resolved_host_port,
    stop_qdrant_server,
)


def _resolve_qdrant_host_port_or_fail() -> tuple[str, int]:
    # Give Docker/Qdrant more time on cold starts (Windows + Docker Desktop).
    ok = ensure_db_ready(host=QDRANT_HOST, port=QDRANT_PORT, wait_qdrant_seconds=60.0)
    assert ok, f"Qdrant server not reachable at {QDRANT_HOST}:{QDRANT_PORT} and could not be started."
    host, port = get_resolved_host_port(QDRANT_HOST, QDRANT_PORT)
    assert host is not None and port is not None, "Qdrant host/port could not be resolved after bootstrap."
    return host, port


def _make_chunk(chunk_id: str, text: str, source: str = "test_doc", path: str = "Ch1", page: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        # Include both `path` and legacy `chapter` to match different parts of the pipeline.
        payload={"text": text, "source": source, "path": path, "chapter": path, "page": page},
        vector=None,
    )


@pytest.fixture(scope="session", autouse=True)
def qdrant_server():
    """Ensure Qdrant is up via db_bootstrap; stop after session."""
    _resolve_qdrant_host_port_or_fail()
    try:
        yield
    finally:
        stop_qdrant_server()


@pytest.fixture(scope="session")
def qdrant_host_port():
    """Resolved (host, port) for Qdrant after bootstrap."""
    return _resolve_qdrant_host_port_or_fail()


@pytest.fixture
def db_manager(qdrant_host_port):
    """DBManager for question pipeline test collection. Skips if Qdrant not reachable."""
    host, port = qdrant_host_port
    assert host is not None and port is not None, "Qdrant host/port could not be resolved after bootstrap."
    client = VectorDBClient(
        host=host,
        port=port,
        collection_name=VECTOR_COLLECTION_NAME_TEST,
    )
    return DBManager(client=client)


@pytest.fixture(autouse=True)
def clear_question_pipeline_collection(qdrant_host_port):
    """Clear the question pipeline test collection before each test."""
    from qdrant_client import QdrantClient
    host, port = _resolve_qdrant_host_port_or_fail()
    c = QdrantClient(host=host, port=port)
    if c.collection_exists(VECTOR_COLLECTION_NAME_TEST):
        c.delete_collection(VECTOR_COLLECTION_NAME_TEST)
    yield
    # Ensure cleanup after each test too.
    if c.collection_exists(VECTOR_COLLECTION_NAME_TEST):
        c.delete_collection(VECTOR_COLLECTION_NAME_TEST)


def test_answer_question_empty_returns_incomplete_response():
    """Empty question returns PROMPT_INCOMPLETE_RESPONSE (no DB/LLM needed)."""
    result = answer_question("")
    assert result == PROMPT_INCOMPLETE_RESPONSE


def test_answer_question_no_chunks_in_db_returns_incomplete_response(db_manager, qdrant_host_port):
    """When the collection is empty, answer_question returns PROMPT_INCOMPLETE_RESPONSE."""
    host, port = qdrant_host_port
    result = answer_question(
        "What is the main topic?",
        collection_name=VECTOR_COLLECTION_NAME_TEST,
        host=host,
        port=port,
    )
    assert result == PROMPT_INCOMPLETE_RESPONSE


def test_answer_question_with_chunks_returns_llm_answer(db_manager, qdrant_host_port):
    """Insert chunks with real embedding, then answer_question returns a non-empty LLM answer."""
    host, port = qdrant_host_port
    chunks = [
        _make_chunk("qp_1", "The capital of France is Paris. It is known for the Eiffel Tower.", "doc.pdf", "Europe", 1),
        _make_chunk("qp_2", "Berlin is the capital of Germany. It has the Brandenburg Gate.", "doc.pdf", "Europe", 2),
    ]
    embedder = EmbeddingService()
    embedded = embedder.embed_chunks(chunks)
    db_manager.delete_all()
    db_manager.insert_chunks(embedded)

    result = answer_question(
        "What is the capital of France?",
        collection_name=VECTOR_COLLECTION_NAME_TEST,
        host=host,
        port=port,
    )
    assert result != PROMPT_INCOMPLETE_RESPONSE
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # LLM should mention Paris or France given the source chunks
    assert "Paris" in result or "France" in result or "capital" in result.lower()


def test_answer_question_reranks_and_uses_top_chunks(db_manager, qdrant_host_port):
    """Pipeline uses reranking: more relevant chunk should influence the answer."""
    host, port = qdrant_host_port
    chunks = [
        _make_chunk("irrelevant", "The weather today is sunny.", "doc.pdf", "Misc", 1),
        _make_chunk("relevant", "Photosynthesis converts sunlight into chemical energy in plants.", "doc.pdf", "Biology", 2),
    ]
    embedder = EmbeddingService()
    embedded = embedder.embed_chunks(chunks)
    db_manager.delete_all()
    db_manager.insert_chunks(embedded)

    result = answer_question(
        "What is photosynthesis?",
        collection_name=VECTOR_COLLECTION_NAME_TEST,
        host=host,
        port=port,
    )
    assert result != PROMPT_INCOMPLETE_RESPONSE
    # Answer should reflect the relevant chunk (plants, sunlight, energy, etc.)
    result_lower = result.lower()
    assert any(word in result_lower for word in ["photosynthesis", "plant", "sunlight", "energy", "convert"])


def test_answer_question_with_history_returns_llm_answer(db_manager, qdrant_host_port):
    """answer_question with history=... still runs the full pipeline and returns a non-empty answer."""
    host, port = qdrant_host_port
    chunks = [
        _make_chunk("qp_1", "The capital of France is Paris.", "doc.pdf", "Europe", 1),
        _make_chunk("qp_2", "Berlin is the capital of Germany.", "doc.pdf", "Europe", 2),
    ]
    embedder = EmbeddingService()
    embedded = embedder.embed_chunks(chunks)
    db_manager.delete_all()
    db_manager.insert_chunks(embedded)

    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital."},
        {"role": "user", "content": "And Germany?"},
        {"role": "assistant", "content": "Berlin."},
    ]
    result = answer_question(
        "Summarize both capitals.",
        collection_name=VECTOR_COLLECTION_NAME_TEST,
        host=host,
        port=port,
        history=history,
    )
    assert result != PROMPT_INCOMPLETE_RESPONSE
    assert isinstance(result, str)
    assert len(result.strip()) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
