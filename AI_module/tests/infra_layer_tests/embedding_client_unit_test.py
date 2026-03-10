"""
Unit tests for EmbeddingClient. Model is mocked (no sentence-transformers load).
For integration tests with real model see embedding_client_integration_test.py.
Run directly: python AI_module/tests/infra_layer_tests/embedding_client_unit_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/embedding_client_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import numpy as np
import pytest
from AI_module.config import EMBEDDING_DIMENSION
from AI_module.infra_layer.embedding_client import EmbeddingClient


@pytest.fixture
def client():
    return EmbeddingClient()


def _make_mock_model():
    m = MagicMock()
    dim = EMBEDDING_DIMENSION

    # generator of random numbers, result number is the same in every itteration
    rng = np.random.default_rng(42)

    def encode(texts, **kwargs):
        if isinstance(texts, str):
            # query: offset, should be different from batch
            return (rng.random(dim) + 0.1).astype(np.float32)
        else:
            # batch: the same seed
            return rng.random((len(texts), dim), dtype=np.float32)

    m.encode.side_effect = encode
    return m


def test_dimension(client):
    assert client.dimension == EMBEDDING_DIMENSION == 384


def test_embed_batch_empty_returns_empty_list(client):
    assert client.embed_batch([]) == []


def test_embed_batch_single_chunk_returns_one_vector(client):
    mock_model = _make_mock_model()
    with patch.object(client, "_get_model", return_value=mock_model):
        vectors = client.embed_batch(["A single sentence for embedding."])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in vectors[0])
    call_args = mock_model.encode.call_args[0][0]
    assert len(call_args) == 1 and call_args[0].startswith("passage: ")


def test_embed_batch_multiple_chunks_returns_same_count(client):
    mock_model = _make_mock_model()
    chunks = ["First chunk.", "Second chunk.", "Third chunk."]
    with patch.object(client, "_get_model", return_value=mock_model):
        vectors = client.embed_batch(chunks)
    assert len(vectors) == len(chunks)
    for v in vectors:
        assert len(v) == EMBEDDING_DIMENSION and all(isinstance(x, float) for x in v)


def test_embed_batch_preserves_order(client):
    mock_model = _make_mock_model()
    chunks = ["Alpha", "Beta", "Gamma"]
    with patch.object(client, "_get_model", return_value=mock_model):
        vectors = client.embed_batch(chunks)
    assert len(vectors) == 3
    call_args = mock_model.encode.call_args[0][0]
    assert call_args[0].endswith("Alpha") and call_args[1].endswith("Beta") and call_args[2].endswith("Gamma")


def test_embed_query_returns_vector(client):
    mock_model = _make_mock_model()
    with patch.object(client, "_get_model", return_value=mock_model):
        vec = client.embed_query("What is the main topic of the document?")
    assert isinstance(vec, list) and len(vec) == EMBEDDING_DIMENSION and all(isinstance(x, float) for x in vec)
    call_arg = mock_model.encode.call_args[0][0]
    assert call_arg.startswith("query: ")


def test_embed_query_empty_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.embed_query("")


def test_embed_query_whitespace_only_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.embed_query("   \t\n ")


def test_embed_query_and_embed_batch_same_text_differ(client):
    mock_model = _make_mock_model()
    with patch.object(client, "_get_model", return_value=mock_model):
        query_vec = client.embed_query("Same text.")
        batch_vecs = client.embed_batch(["Same text."])
    assert len(batch_vecs) == 1
    assert query_vec != batch_vecs[0]


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
