"""
Unit tests for AI_api.main (FastAPI app). RAG get_answer is mocked.
Run from repo root: python -m pytest AI_api/tests/main_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

_root = Path(__file__).resolve().parent.parent.parent
if _root not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_root))

import pytest
from fastapi.testclient import TestClient

from AI_api.main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_from_rag():
    """POST /ask with question only calls get_answer(question) and returns answer."""
    with patch("AI_api.main.get_answer") as mock_get_answer:
        mock_get_answer.return_value = "Paris is the capital of France."
        response = client.post(
            "/ask",
            json={"question": "What is the capital of France?"},
        )
        assert response.status_code == 200
        assert response.json() == {"answer": "Paris is the capital of France."}
        mock_get_answer.assert_called_once_with("What is the capital of France?", history=None)


def test_ask_with_history_passes_history_to_rag():
    """POST /ask with history in body passes it to get_answer(question, history=...)."""
    with patch("AI_api.main.get_answer") as mock_get_answer:
        mock_get_answer.return_value = "Berlin is the capital of Germany."
        body = {
            "question": "And Germany?",
            "history": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is the capital of France."},
                {"role": "user", "content": "What about Germany?"},
                {"role": "assistant", "content": "Berlin."},
            ],
        }
        response = client.post("/ask", json=body)
        assert response.status_code == 200
        assert response.json() == {"answer": "Berlin is the capital of Germany."}
        mock_get_answer.assert_called_once()
        call_args = mock_get_answer.call_args
        assert call_args[0][0] == "And Germany?"
        assert call_args[1]["history"] == [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."},
            {"role": "user", "content": "What about Germany?"},
            {"role": "assistant", "content": "Berlin."},
        ]


def test_ask_with_empty_history_calls_rag_with_none():
    """POST /ask with history: [] results in get_answer(question, history=None) (no history)."""
    with patch("AI_api.main.get_answer") as mock_get_answer:
        mock_get_answer.return_value = "Answer."
        response = client.post(
            "/ask",
            json={"question": "What is it?", "history": []},
        )
        assert response.status_code == 200
        mock_get_answer.assert_called_once_with("What is it?", history=None)


def test_ask_missing_question_returns_422():
    """POST /ask without question returns 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_empty_question_returns_422():
    """POST /ask with empty question string returns 422."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_invalid_history_role_returns_422():
    """POST /ask with history entry where role is not 'user' or 'assistant' returns 422."""
    response = client.post(
        "/ask",
        json={
            "question": "Follow-up?",
            "history": [
                {"role": "user", "content": "First question."},
                {"role": "Jozef", "content": "Invalid role."},
            ],
        },
    )
    assert response.status_code == 422


def test_ask_history_role_assistant_accepted():
    """POST /ask with role 'assistant' in history is accepted."""
    with patch("AI_api.main.get_answer") as mock_get_answer:
        mock_get_answer.return_value = "Answer."
        response = client.post(
            "/ask",
            json={
                "question": "Next?",
                "history": [{"role": "assistant", "content": "Previous answer."}],
            },
        )
        assert response.status_code == 200
        mock_get_answer.assert_called_once()
        assert mock_get_answer.call_args[1]["history"] == [{"role": "assistant", "content": "Previous answer."}]
