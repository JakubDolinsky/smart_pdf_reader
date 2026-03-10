"""
Unit tests for LlmClient (Mistral 7B via Ollama). Ollama API mocked via _call_ollama.
For real LLM integration tests see llm_client_integration_test.py.
Run directly: python AI_module/tests/infra_layer_tests/llm_client_unit_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/llm_client_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.infra_layer.llm_client import LlmClient


@pytest.fixture
def client():
    return LlmClient()


def test_answer_empty_prompt_raises(client):
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        client.answer("")
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        client.answer("   \t\n ")


def test_answer_returns_string(client):
    with patch.object(client, "_call_ollama", return_value="This is a simulated model response."):
        result = client.answer("Question?")
    assert result == "This is a simulated model response."


def test_answer_strips_inst_section(client):
    expected = "The main idea of the document is as follows."
    with patch.object(client, "_call_ollama", return_value=expected):
        result = client.answer("What is the main idea?")
    assert result == expected and "[/INST]" not in result


def test_answer_question_prompt(client):
    question = "What is the purpose of this document?"
    answer = "The purpose of the document is to explain the procedure."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer(question)
    assert result == answer and isinstance(result, str) and len(result) > 0


def test_answer_when_decode_has_no_inst_returns_stripped_full_output(client):
    full_output = "Unexpected output format without [/INST] marker."
    with patch.object(client, "_call_ollama", return_value=full_output):
        result = client.answer("Question?")
    assert result == full_output


def test_answer_rag_style_prompt(client):
    prompt = "Context:\nThis document describes the implementation procedure.\n\nQuestion: What is the first step?"
    answer = "The first step is analysis."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer(prompt)
    assert result == answer and ("analysis" in result or "step" in result.lower())


def test_answer_long_prompt(client):
    long_context = "Paragraph of text. " * 200
    prompt = f"{long_context}\n\nSummarize in one sentence."
    answer = "The text summarizes the main points."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer(prompt)
    assert result == answer and len(result) > 0


def test_answer_model_returns_empty_after_inst(client):
    with patch.object(client, "_call_ollama", return_value=""):
        result = client.answer("This question has no answer.")
    assert result == ""


def test_answer_model_returns_inst_in_response_takes_last_segment(client):
    with patch.object(client, "_call_ollama", return_value="This is the actual answer."):
        result = client.answer("Question?")
    assert result == "This is the actual answer."


def test_answer_prompt_with_newlines_and_whitespace(client):
    prompt = "  First line.\nSecond line.\n  Third.  "
    answer = "Answer to a multi-line question."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer(prompt)
    assert result == answer


def test_answer_unicode_prompt(client):
    question = "What does the term \"implementation\" mean in this context?"
    answer = "Implementation means carrying out the design."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer(question)
    assert result == answer and ("implementation" in result or "carrying out" in result)


def test_answer_factual_question_scenario(client):
    with patch.object(client, "_call_ollama", return_value="The document describes three main steps."):
        result = client.answer("How many steps does the document describe?")
    assert result and len(result) >= 5


def test_answer_no_answer_in_context_scenario(client):
    answer = "The annual salary is not mentioned in the given context. I cannot answer."
    with patch.object(client, "_call_ollama", return_value=answer):
        result = client.answer("Context: This is about gardening. Question: What was the annual salary?")
    assert result == answer and isinstance(result, str)


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
