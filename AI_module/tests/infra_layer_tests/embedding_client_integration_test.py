"""
Integration tests for EmbeddingClient with real sentence-transformers model (first run may download).
For unit tests (mocked) see embedding_client_unit_test.py.
Run directly: python AI_module/tests/infra_layer_tests/embedding_client_integration_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/embedding_client_integration_test.py -v
"""

import sys
from pathlib import Path
from typing import List

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.config import EMBEDDING_DIMENSION
from AI_module.infra_layer.embedding_client import EmbeddingClient


@pytest.fixture(scope="session")
def client():
    return EmbeddingClient()


def test_dimension(client):
    assert client.dimension == EMBEDDING_DIMENSION == 384


def test_embed_batch_empty_returns_empty_list(client):
    assert client.embed_batch([]) == []


def test_embed_batch_single_chunk_returns_one_vector(client):
    vectors = client.embed_batch(["A single sentence for embedding."])
    assert len(vectors) == 1 and len(vectors[0]) == EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_batch_multiple_chunks_returns_same_count(client):
    chunks = ["First chunk.", "Second chunk.", "Third chunk."]
    vectors = client.embed_batch(chunks)
    assert len(vectors) == len(chunks)
    for v in vectors:
        assert len(v) == EMBEDDING_DIMENSION and all(isinstance(x, float) for x in v)


def test_embed_batch_preserves_order(client):
    chunks = ["Alpha", "Beta", "Gamma"]
    vectors = client.embed_batch(chunks)
    assert len(vectors) == 3
    assert vectors[0] != vectors[1] != vectors[2] != vectors[0]


def test_embed_query_returns_vector(client):
    vec = client.embed_query("What is the main topic of the document?")
    assert isinstance(vec, list) and len(vec) == EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in vec)


def test_embed_query_empty_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.embed_query("")


def test_embed_query_whitespace_only_raises(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.embed_query("   \t\n ")


def test_embed_query_and_embed_batch_same_text_differ(client):
    text = "Same text for both query and passage."
    query_vec = client.embed_query(text)
    batch_vecs = client.embed_batch([text])
    assert len(batch_vecs) == 1 and query_vec != batch_vecs[0]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity (vectors are already normalized by EmbeddingClient)."""
    return sum(x * y for x, y in zip(a, b))


def test_semantic_similarity_query_and_matching_passage(client):
    """Query and passage with similar meaning should have high cosine similarity."""
    query = "What is the capital of France?"
    passage = "Paris is the capital of France."
    query_vec = client.embed_query(query)
    (passage_vec,) = client.embed_batch([passage])
    similarity = _cosine_similarity(query_vec, passage_vec)
    assert similarity > 0.5, (
        f"Semantically related query and passage should have similarity > 0.5, got {similarity:.4f}"
    )


def test_semantic_similarity_related_pair_higher_than_unrelated(client):
    """Similar-meaning (query, passage) pair should have higher similarity than an unrelated pair."""
    query = "How does photosynthesis work?"
    related_passage = "Plants convert light into energy through photosynthesis."
    unrelated_passage = "The train departed from platform three at noon."
    query_vec = client.embed_query(query)
    related_vec, = client.embed_batch([related_passage])
    unrelated_vec, = client.embed_batch([unrelated_passage])
    sim_related = _cosine_similarity(query_vec, related_vec)
    sim_unrelated = _cosine_similarity(query_vec, unrelated_vec)
    assert sim_related > sim_unrelated, (
        f"Related pair similarity ({sim_related:.4f}) should exceed unrelated ({sim_unrelated:.4f})"
    )


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
