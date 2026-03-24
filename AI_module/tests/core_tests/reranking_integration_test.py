"""
Integration tests for core.reranking with real RerankingClient (cross-encoder model).
Uses real sentence-transformers CrossEncoder; may download model on first run. Slow.
Invalid input must return empty collections without calling the model.
For fast unit tests with mocks see reranking_unit_test.py.
Run: python -m pytest AI_module/tests/core_tests/reranking_integration_test.py -v
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
from AI_module.core.reranking import RerankingService

_EMPTY = {"chunks": []}


def _dbmanager_result(ids: list[str], texts: list[str]) -> dict:
    """Simulate DBManager.search_similar output: chunks (list of Chunk)."""
    chunks = [
        Chunk(id=cid, payload={"text": t, "source": "test", "page": 1}, vector=None)
        for cid, t in zip(ids, texts)
    ]
    return {"chunks": chunks}


# ---------- Invalid input: return empty (no model call in practice) ----------


def test_rerank_real_client_chunks_dict_none_returns_empty():
    service = RerankingService()
    result = service.rerank("query", None)
    assert result == _EMPTY


def test_rerank_real_client_query_none_returns_empty():
    chunks = _dbmanager_result(["a"], ["some text"])
    service = RerankingService()
    result = service.rerank(None, chunks)
    assert result == _EMPTY


def test_rerank_real_client_empty_query_returns_empty():
    chunks = _dbmanager_result(["a", "b"], ["first", "second"])
    service = RerankingService()
    result = service.rerank("", chunks, top_k=2)
    assert result == _EMPTY


def test_rerank_real_client_metadata_empty_text_returns_empty():
    chunks = _dbmanager_result(["x"], [""])
    service = RerankingService()
    result = service.rerank("question", chunks)
    assert result == _EMPTY


def test_rerank_real_client_metadata_missing_text_returns_empty():
    chunks = {"chunks": [Chunk(id="y", payload={"source": "doc"}, vector=None)]}
    service = RerankingService()
    result = service.rerank("question", chunks)
    assert result == _EMPTY


# ---------- Valid input: real model returns top-k with scores ----------


def test_rerank_real_client_returns_top_k_with_scores():
    """RerankingService without client uses real RerankingClient and returns top_k chunks (no scores in output)."""
    chunks = _dbmanager_result(
        ["c1", "c2", "c3"],
        [
            "The capital of France is Paris.",
            "Photosynthesis uses sunlight and water.",
            "Paris is a major European city and tourist destination.",
        ],
    )
    service = RerankingService()
    result = service.rerank("What is the capital of France?", chunks, top_k=2)
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert all(hasattr(c, "payload") for c in result["chunks"])
    assert "scores" not in result


def test_rerank_real_client_ranks_relevant_chunk_higher():
    """Chunks more relevant to the query are ordered first (Paris/France > photosynthesis)."""
    chunks = _dbmanager_result(
        ["about_paris", "about_photosynthesis"],
        [
            "Paris is the capital of France and a center of culture.",
            "Photosynthesis is a process in plants that uses sunlight.",
        ],
    )
    service = RerankingService()
    result = service.rerank("What is the capital of France?", chunks, top_k=2)
    assert len(result["chunks"]) == 2
    top_text = result["chunks"][0].payload.get("text", "")
    assert "Paris" in top_text or "France" in top_text


def test_rerank_real_client_single_chunk():
    """Single chunk returns single chunk in result."""
    chunks = _dbmanager_result(["only"], ["Single passage about anything in particular."])
    service = RerankingService()
    result = service.rerank("anything", chunks, top_k=3)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "only"
    assert result["chunks"][0].payload["text"] == "Single passage about anything in particular."


def test_rerank_real_client_output_ready_for_prompt():
    """Output chunks are suitable for llm_chatter (Chunk with payload text)."""
    chunks = _dbmanager_result(
        ["a", "b"],
        ["First context sentence.", "Second context sentence."],
    )
    service = RerankingService()
    result = service.rerank("User question?", chunks, top_k=2)
    assert isinstance(result, dict)
    assert "chunks" in result
    for c in result["chunks"]:
        assert "text" in c.payload
        assert c.id in ["a", "b"]


def test_rerank_real_client_accepts_dbmanager_chunks_output():
    """RerankingService accepts DBManager.search_similar output (dict with 'chunks' list of Chunk); returns chunks only."""
    dbmanager_out = _dbmanager_result(
        ["c1", "c2", "c3"],
        [
            "The capital of France is Paris.",
            "Photosynthesis uses sunlight and water.",
            "Paris is a major European city.",
        ],
    )
    service = RerankingService()
    result = service.rerank("What is the capital of France?", dbmanager_out, top_k=2)
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert "scores" not in result
    top_text = result["chunks"][0].payload.get("text", "")
    assert "Paris" in top_text or "France" in top_text


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
