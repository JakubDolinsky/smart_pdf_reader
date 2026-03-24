"""Unit tests for core.rewriting."""

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

from AI_module.core.rewriting import (
    question_contains_reference_word,
    rewrite_question_for_embedding,
)


def test_question_contains_reference_word_detects_that():
    assert question_contains_reference_word("What about that?") is True


def test_question_contains_reference_word_detects_phrase_the_former():
    assert question_contains_reference_word("Compare A and B. Who was the former?") is True


def test_question_contains_reference_word_false_for_substrings():
    assert question_contains_reference_word("What is theremin?") is False


def test_question_contains_reference_word_false_hexadecimal_not_he():
    """'he' must not match inside unrelated words like hexadecimal."""
    assert question_contains_reference_word("What is hexadecimal?") is False


def test_question_contains_reference_word_true_standalone_he():
    assert question_contains_reference_word("Did he agree?") is True


def test_rewrite_skips_without_reference_word():
    mock_rw = MagicMock()
    hist = [
        {"role": "assistant", "content": "Ans."},
        {"role": "user", "content": "Q?"},
    ]
    out = rewrite_question_for_embedding(
        "What is the population?",
        hist,
        rewriter_client=mock_rw,
    )
    assert out == "What is the population?"
    mock_rw.rewrite.assert_not_called()


def test_rewrite_skips_empty_history():
    mock_rw = MagicMock()
    out = rewrite_question_for_embedding(
        "What about that?",
        None,
        rewriter_client=mock_rw,
    )
    assert out == "What about that?"
    mock_rw.rewrite.assert_not_called()


def test_rewrite_calls_client_when_reference_and_full_history():
    mock_rw = MagicMock()
    mock_rw.rewrite.return_value = "Rewritten question?"
    hist = [
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "Capital of France?"},
    ]
    out = rewrite_question_for_embedding(
        "What about that city?",
        hist,
        rewriter_client=mock_rw,
    )
    assert out == "Rewritten question?"
    mock_rw.rewrite.assert_called_once()
    prompt = mock_rw.rewrite.call_args[0][0]
    assert "Capital of France?" in prompt
    assert "Paris." in prompt
    assert "What about that city?" in prompt


def test_rewrite_falls_back_when_client_returns_empty():
    mock_rw = MagicMock()
    mock_rw.rewrite.return_value = ""
    hist = [
        {"role": "assistant", "content": "A."},
        {"role": "user", "content": "Q?"},
    ]
    out = rewrite_question_for_embedding(
        "Explain that.",
        hist,
        rewriter_client=mock_rw,
    )
    assert out == "Explain that."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
