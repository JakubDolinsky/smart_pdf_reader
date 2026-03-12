"""
Integration tests for AI_api.main (FastAPI app). They call the real RAG pipeline (no mocks).
Requires Qdrant at default host/port; tests are skipped if Qdrant is not reachable.
Run from repo root: python -m pytest AI_api/tests/main_integration_test.py -v
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if _root not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_root))

import pytest
from fastapi.testclient import TestClient

from AI_module.config import QDRANT_LOCAL_HOST, QDRANT_LOCAL_PORT
from AI_module.infra_layer import check_db_ready

from AI_api.main import app

client = TestClient(app)


def _qdrant_ready() -> bool:
    """True if Qdrant is reachable at default host/port."""
    return check_db_ready(host=QDRANT_LOCAL_HOST, port=QDRANT_LOCAL_PORT)


def test_integration_health():
    """GET /health returns 200 and status ok (no external deps)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not _qdrant_ready(), reason="Qdrant not reachable at default host/port")
def test_integration_ask_returns_200_and_answer():
    """POST /ask calls real RAG pipeline; returns 200 and answer string (may be incomplete if no chunks)."""
    response = client.post(
        "/ask",
        json={"question": "What is the main topic?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


@pytest.mark.skipif(not _qdrant_ready(), reason="Qdrant not reachable at default host/port")
def test_integration_ask_with_history_returns_200():
    """POST /ask with history calls real RAG pipeline; returns 200 and answer string."""
    response = client.post(
        "/ask",
        json={
            "question": "And the second one?",
            "history": [
                {"role": "user", "content": "What is the first topic?"},
                {"role": "assistant", "content": "The first topic is X."},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


@pytest.mark.skipif(not _qdrant_ready(), reason="Qdrant not reachable at default host/port")
def test_integration_ask_invalid_body_returns_422():
    """POST /ask with missing question returns 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_integration_ask_invalid_history_role_returns_422():
    """POST /ask with history entry where role is not 'user' or 'assistant' returns 422 (no Qdrant needed)."""
    response = client.post(
        "/ask",
        json={
            "question": "Follow-up?",
            "history": [
                {"role": "user", "content": "First."},
                {"role": "system", "content": "Invalid role."},
            ],
        },
    )
    assert response.status_code == 422
