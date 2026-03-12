"""
Unit tests for embedding.py (EmbeddingService). EmbeddingClient mocked.
For real-model tests see embedding_integration_test.py.
Run directly: python AI_module/tests/core_tests/embedding_unit_test.py
Or: python -m pytest AI_module/tests/core_tests/embedding_unit_test.py -v
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
from AI_module.core.embedding import EmbeddingService


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.embed_query.return_value = [0.1] * 384
    client.embed_batch.return_value = [[0.2] * 384, [0.3] * 384]
    return client


def _make_chunk(chunk_id: str, text: str, **payload_extra) -> Chunk:
    payload = {"text": text, "source": "test_doc", "page": 1, "chunk_index": 0, "chapter": ""}
    payload.update(payload_extra)
    return Chunk(id=chunk_id, payload=payload, vector=None)


def test_embed_query_returns_vector_from_client(mock_client):
    mock_client.embed_query.return_value = [0.5] * 384
    service = EmbeddingService(client=mock_client)
    result = service.embed_query("What is the main topic?")
    mock_client.embed_query.assert_called_once_with("What is the main topic?")
    assert result == [0.5] * 384 and len(result) == 384


def test_embed_query_empty_raises(mock_client):
    mock_client.embed_query.side_effect = ValueError("query must be non-empty")
    service = EmbeddingService(client=mock_client)
    with pytest.raises(ValueError, match="query must be non-empty"):
        service.embed_query("")


def test_embed_query_whitespace_only_raises(mock_client):
    mock_client.embed_query.side_effect = ValueError("query must be non-empty")
    service = EmbeddingService(client=mock_client)
    with pytest.raises(ValueError, match="query must be non-empty"):
        service.embed_query("   \n\t  ")


def test_embed_chunks_empty_returns_empty_list(mock_client):
    service = EmbeddingService(client=mock_client)
    result = service.embed_chunks([])
    assert result == []
    mock_client.embed_batch.assert_not_called()


def test_embed_chunks_returns_chunks_with_vector_set(mock_client):
    chunks = [_make_chunk("id1", "First text.", chunk_index=0), _make_chunk("id2", "Second text.", chunk_index=1)]
    service = EmbeddingService(client=mock_client)
    result = service.embed_chunks(chunks)
    assert len(result) == 2 and result[0].id == "id1" and result[0].vector == [0.2] * 384
    assert result[1].id == "id2" and result[1].vector == [0.3] * 384


def test_embed_chunks_preserves_order_and_payload(mock_client):
    chunks = [_make_chunk("a", "Alpha", source="doc1", page=1, chapter="Ch1"), _make_chunk("b", "Beta", source="doc1", page=2, chapter="")]
    service = EmbeddingService(client=mock_client)
    result = service.embed_chunks(chunks)
    assert result[0].payload["source"] == "doc1" and result[0].payload["chapter"] == "Ch1"
    assert result[1].payload["page"] == 2


def test_embed_chunks_passes_text_from_payload_to_embed_batch(mock_client):
    chunks = [_make_chunk("x", "Hello world."), _make_chunk("y", "Another passage.")]
    service = EmbeddingService(client=mock_client)
    service.embed_chunks(chunks)
    mock_client.embed_batch.assert_called_once_with(["Hello world.", "Another passage."])


def test_embed_chunks_missing_text_uses_empty_string(mock_client):
    chunk = Chunk(id="z", payload={"source": "doc"}, vector=None)
    mock_client.embed_batch.return_value = [[0.0] * 384]
    service = EmbeddingService(client=mock_client)
    result = service.embed_chunks([chunk])
    mock_client.embed_batch.assert_called_once_with([""])
    assert result[0].vector is not None


def test_embed_chunks_length_mismatch_raises(mock_client):
    chunks = [_make_chunk("id1", "One."), _make_chunk("id2", "Two.")]
    mock_client.embed_batch.return_value = [[0.0] * 384]
    service = EmbeddingService(client=mock_client)
    with pytest.raises(ValueError, match="embed_batch returned 1 vectors for 2 chunks"):
        service.embed_chunks(chunks)


def test_embed_chunks_output_ready_for_insert_chunks(mock_client):
    chunks = [_make_chunk("id_a", "Text A."), _make_chunk("id_b", "Text B.")]
    service = EmbeddingService(client=mock_client)
    embedded = service.embed_chunks(chunks)
    ids = [c.id for c in embedded]
    embeddings = [c.vector for c in embedded]
    metadatas = [c.payload for c in embedded]
    assert ids == ["id_a", "id_b"] and len(embeddings) == 2 and metadatas[0]["text"] == "Text A."


def test_embed_query_output_ready_for_search_similar(mock_client):
    mock_client.embed_query.return_value = [0.1] * 384
    service = EmbeddingService(client=mock_client)
    vector = service.embed_query("Find similar chunks.")
    assert isinstance(vector, list) and len(vector) == 384 and all(isinstance(x, float) for x in vector)


def test_embed_query_with_history_and_reference_word_appends_first_message(mock_client):
    """When history is non-empty and query contains a reference word (e.g. 'that'), first message content is appended for embedding."""
    mock_client.embed_query.return_value = [0.1] * 384
    service = EmbeddingService(client=mock_client)
    history = [{"role": "assistant", "content": "Paris is the capital of France."}]
    service.embed_query("What about that?", history=history)
    mock_client.embed_query.assert_called_once_with(
        "What about that? Paris is the capital of France."
    )


def test_embed_query_with_history_no_reference_word_uses_query_only(mock_client):
    """When history is provided but query has no reference word, client is called with query only."""
    mock_client.embed_query.return_value = [0.1] * 384
    service = EmbeddingService(client=mock_client)
    history = [{"role": "user", "content": "What is the capital?"}]
    service.embed_query("What is the population?", history=history)
    mock_client.embed_query.assert_called_once_with("What is the population?")


def test_embed_query_with_empty_history_uses_query_only(mock_client):
    """When history is empty list, no appending."""
    mock_client.embed_query.return_value = [0.1] * 384
    service = EmbeddingService(client=mock_client)
    service.embed_query("What is this?", history=[])
    mock_client.embed_query.assert_called_once_with("What is this?")


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
