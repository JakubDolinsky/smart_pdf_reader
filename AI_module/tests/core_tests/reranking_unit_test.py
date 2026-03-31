"""
Unit tests for core.reranking (prepare_pairs and rerank). RerankingClient mocked; no real model.
For integration tests with real cross-encoder see reranking_integration_test.py.
Run: python -m pytest AI_module/tests/core_tests/reranking_unit_test.py -v
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
from AI_module.core.reranking import RerankingService, prepare_pairs

_EMPTY = {"chunks": []}


def _dbmanager_result(chunk_ids: list[str], texts: list[str], scores: list[float] | None = None) -> dict:
    """Simulate DBManager.search_similar output: chunks (list of Chunk)."""
    chunks = [
        Chunk(id=cid, payload={"text": t, "source": "test", "page": 1}, vector=None)
        for cid, t in zip(chunk_ids, texts)
    ]
    out = {"chunks": chunks}
    if scores is not None:
        out["scores"] = scores
    return out


# ---------- prepare_pairs ----------


def test_prepare_pairs_valid_returns_pairs_and_indices():
    """prepare_pairs returns (query, text) pairs and valid_indices for chunks with non-empty text."""
    ids = ["a", "b"]
    metadatas = [{"text": "First text.", "source": "test"}, {"text": "Second text.", "source": "test"}]
    pairs, valid_indices = prepare_pairs(" my query ", ids, metadatas)
    assert pairs == [("my query", "First text."), ("my query", "Second text.")]
    assert valid_indices == [0, 1]


def test_prepare_pairs_skips_empty_text():
    """Chunks with empty or whitespace-only text are skipped."""
    ids = ["x", "y", "z"]
    metadatas = [{"text": "ok"}, {"text": ""}, {"text": "  \n "}]
    pairs, valid_indices = prepare_pairs("q", ids, metadatas)
    assert pairs == [("q", "ok")]
    assert valid_indices == [0]


def test_prepare_pairs_skips_missing_text():
    """Chunks without 'text' in metadata are skipped."""
    ids = ["a", "b"]
    metadatas = [{"text": "yes"}, {"source": "doc"}]
    pairs, valid_indices = prepare_pairs("q", ids, metadatas)
    assert pairs == [("q", "yes")]
    assert valid_indices == [0]


def test_prepare_pairs_skips_non_dict_metadata():
    """Non-dict metadata entries are skipped."""
    ids = ["a", "b"]
    metadatas = [{"text": "ok"}, "not a dict"]
    pairs, valid_indices = prepare_pairs("q", ids, metadatas)
    assert pairs == [("q", "ok")]
    assert valid_indices == [0]


def test_prepare_pairs_empty_query_returns_no_pairs():
    """Empty or whitespace-only query yields no pairs."""
    ids = ["a"]
    metadatas = [{"text": "some text"}]
    pairs, valid_indices = prepare_pairs("", ids, metadatas)
    assert pairs == []
    assert valid_indices == []

    pairs2, _ = prepare_pairs("  \t\n ", ids, metadatas)
    assert pairs2 == []


def test_prepare_pairs_empty_ids_returns_no_pairs():
    """Missing or empty ids/metadatas returns no pairs."""
    pairs, valid_indices = prepare_pairs("q", [], [])
    assert pairs == [] and valid_indices == []

    pairs2, _ = prepare_pairs("q", ["a"], [])
    assert pairs2 == []


def test_prepare_pairs_mismatched_length_returns_no_pairs():
    """When ids and metadatas length differ, no pairs are built."""
    ids = ["a", "b"]
    metadatas = [{"text": "only one"}]
    pairs, valid_indices = prepare_pairs("q", ids, metadatas)
    assert pairs == [] and valid_indices == []


# ---------- rerank: invalid input returns empty ----------


def test_rerank_chunks_dict_none_returns_empty():
    service = RerankingService()
    result = service.rerank("query", None)
    assert result == _EMPTY


def test_rerank_chunks_dict_not_dict_returns_empty():
    service = RerankingService()
    result = service.rerank("query", [])
    assert result == _EMPTY


def test_rerank_query_none_returns_empty():
    chunks = _dbmanager_result(["a"], ["text"])
    service = RerankingService()
    result = service.rerank(None, chunks)
    assert result == _EMPTY


def test_rerank_query_empty_returns_empty():
    chunks = _dbmanager_result(["a"], ["text"])
    service = RerankingService()
    result = service.rerank("", chunks)
    assert result == _EMPTY


def test_rerank_no_valid_pairs_returns_empty():
    chunks = _dbmanager_result(["x"], [""])
    service = RerankingService()
    result = service.rerank("q", chunks)
    assert result == _EMPTY


# ---------- rerank: mocked client ----------


@pytest.fixture
def mock_reranking_client():
    """RerankingClient mock; rerank calls score_pairs on it when client is passed."""
    client = MagicMock()
    return client


def test_rerank_calls_client_score_pairs_with_pairs(mock_reranking_client):
    """rerank builds pairs and calls client.score_pairs with them; returns chunks only."""
    mock_reranking_client.score_pairs.return_value = [0.85, 0.75]
    chunks_in = _dbmanager_result(["id1", "id2"], ["Chunk one.", "Chunk two."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("user query", chunks_in, top_k=2)
    mock_reranking_client.score_pairs.assert_called_once()
    call_pairs = mock_reranking_client.score_pairs.call_args[0][0]
    assert call_pairs == [("user query", "Chunk one."), ("user query", "Chunk two.")]
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "id1" and result["chunks"][0].payload["text"] == "Chunk one."
    assert result["chunks"][1].id == "id2" and result["chunks"][1].payload["text"] == "Chunk two."
    assert "scores" not in result


def test_rerank_orders_by_score_descending(mock_reranking_client):
    """rerank sorts by score descending and returns top_k chunks."""
    mock_reranking_client.score_pairs.return_value = [0.3, 0.9, 0.5]
    chunks_in = _dbmanager_result(["a", "b", "c"], ["A.", "B.", "C."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=2)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "b" and result["chunks"][0].payload["text"] == "B."
    assert result["chunks"][1].id == "c" and result["chunks"][1].payload["text"] == "C."


def test_rerank_top_k_limits_results(mock_reranking_client):
    """rerank returns at most top_k chunks."""
    mock_reranking_client.score_pairs.return_value = [0.71, 0.72, 0.73, 0.74]
    chunks_in = _dbmanager_result(["a", "b", "c", "d"], ["A.", "B.", "C.", "D."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=2)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "d" and result["chunks"][1].id == "c"  # scores 0.74, 0.73


def test_rerank_client_returns_wrong_scores_length_returns_empty(mock_reranking_client):
    """If client returns different number of scores than pairs, rerank returns empty."""
    mock_reranking_client.score_pairs.return_value = [0.5]
    chunks_in = _dbmanager_result(["a", "b"], ["One.", "Two."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=2)
    assert result == _EMPTY


def test_rerank_single_chunk(mock_reranking_client):
    """Single valid chunk returns single chunk."""
    mock_reranking_client.score_pairs.return_value = [0.99]
    chunks_in = _dbmanager_result(["only"], ["Single passage."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("query", chunks_in, top_k=5)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "only"
    assert result["chunks"][0].payload["text"] == "Single passage."


def test_rerank_skips_invalid_chunks_still_reranks_valid(mock_reranking_client):
    """Chunks with empty text are skipped; only valid chunks are sent to client and reranked."""
    mock_reranking_client.score_pairs.return_value = [0.7]
    chunks_in = _dbmanager_result(["good", "bad"], ["Valid.", ""])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=2)
    call_pairs = mock_reranking_client.score_pairs.call_args[0][0]
    assert call_pairs == [("q", "Valid.")]
    assert len(result["chunks"]) == 1
    assert result["chunks"][0].id == "good"


def test_rerank_accepts_dbmanager_chunks_output(mock_reranking_client):
    """rerank accepts DBManager.search_similar output (dict with 'chunks' list of Chunk); returns chunks only."""
    mock_reranking_client.score_pairs.return_value = [0.9, 0.7]
    dbmanager_out = _dbmanager_result(
        ["c1", "c2"],
        ["First passage.", "Second passage."],
        scores=[0.8, 0.6],
    )
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("user question?", dbmanager_out, top_k=2)
    mock_reranking_client.score_pairs.assert_called_once()
    call_pairs = mock_reranking_client.score_pairs.call_args[0][0]
    assert call_pairs == [("user question?", "First passage."), ("user question?", "Second passage.")]
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "c1" and result["chunks"][0].payload["text"] == "First passage."
    assert result["chunks"][1].id == "c2" and result["chunks"][1].payload["text"] == "Second passage."
    assert "scores" not in result


def test_rerank_keeps_chunks_within_gap_and_above_min(mock_reranking_client):
    """Chunks with score > -4 and within 3 of the top score are kept (sorted descending)."""
    mock_reranking_client.score_pairs.return_value = [0.65, 0.85]
    chunks_in = _dbmanager_result(["low", "high"], ["Low.", "High."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=5)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "high"
    assert result["chunks"][1].id == "low"


def test_rerank_low_scores_still_return_top_k(mock_reranking_client):
    mock_reranking_client.score_pairs.return_value = [0.1, 0.2, 0.35]
    chunks_in = _dbmanager_result(["a", "b", "c"], ["A.", "B.", "C."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=3)
    assert len(result["chunks"]) == 3
    assert result["chunks"][0].id == "c" and result["chunks"][1].id == "b"


def test_rerank_returns_empty_when_best_score_not_above_min(mock_reranking_client):
    """If the top score is <= -4, no chunks are returned."""
    mock_reranking_client.score_pairs.return_value = [-4.5, -6.0]
    chunks_in = _dbmanager_result(["a", "b"], ["A.", "B."])
    service = RerankingService(client=mock_reranking_client)
    assert service.rerank("q", chunks_in, top_k=3) == _EMPTY


def test_rerank_drops_chunks_at_or_below_min_score(mock_reranking_client):
    """Chunks with score <= -4 are excluded even if within score gap of the top."""
    mock_reranking_client.score_pairs.return_value = [-3.0, -4.5, -2.0]
    chunks_in = _dbmanager_result(["a", "b", "c"], ["A.", "B.", "C."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=5)
    # Sorted: c -2.0, a -3.0, b -4.5 -> top -2.0; a gap 1.0 OK; b <= -4 dropped
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "c"
    assert result["chunks"][1].id == "a"


def test_rerank_drops_chunks_outside_score_gap(mock_reranking_client):
    """Only the top chunk is kept when the next-best is more than 3 below the top."""
    mock_reranking_client.score_pairs.return_value = [1.0, 5.0, 3.0]
    chunks_in = _dbmanager_result(["a", "b", "c"], ["A.", "B.", "C."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=5)
    # Sorted: b 5.0, c 3.0 (gap 3.0 OK), a 1.0 (gap 4.0 > 2) excluded
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "b"
    assert result["chunks"][1].id == "c"


def test_rerank_gap_boundary_includes_chunk_exactly_two_below_top(mock_reranking_client):
    """Chunk with score exactly (top - 2) is included: gap 2 is allowed."""
    mock_reranking_client.score_pairs.return_value = [3.0, 5.0, 1.0]
    chunks_in = _dbmanager_result(["a", "b", "c"], ["A.", "B.", "C."])
    service = RerankingService(client=mock_reranking_client)
    result = service.rerank("q", chunks_in, top_k=5)
    assert len(result["chunks"]) == 2
    assert result["chunks"][0].id == "b"
    assert result["chunks"][1].id == "a"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
