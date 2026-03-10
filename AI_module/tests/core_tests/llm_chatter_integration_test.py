"""
Integration tests for core.llm_chatter with real config (LM_PROMPT_TEMPLATE) and optional real LLM.
Full prompt is built from rerank-shaped output and user question; chat() uses LlmClient when RUN_LLM_TESTS.
Run: python -m pytest AI_module/tests/core_tests/llm_chatter_integration_test.py -v
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
from AI_module.core.llm_chatter import LLMChatter, PROMPT_INCOMPLETE_RESPONSE
from AI_module.config import RUN_LLM_TESTS


def _chunks(ids: list[str], metadatas: list[dict]) -> list[Chunk]:
    return [
        Chunk(id=cid, payload=dict(meta), vector=None)
        for cid, meta in zip(ids, metadatas)
    ]


def test_create_prompt_uses_real_config_template():
    """LLMChatter.create_prompt without template override uses LM_PROMPT_TEMPLATE from config."""
    chunks = _chunks(
        ["c1"],
        [{"text": "Paris is the capital of France.", "source": "Geography.pdf", "chapter": "Europe", "page": 3}],
    )
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "What is the capital of France?")
    assert "You are an assistant answering questions using the provided sources" in prompt
    assert "Use only the information from the sources" in prompt
    assert "Cite the source after each statement using the format (Chapter, Page)" in prompt
    assert "The information is not available in the provided document" in prompt
    assert "Sources:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "Paris is the capital of France." in prompt
    assert "Geography.pdf" in prompt
    assert "Europe" in prompt
    assert "3" in prompt
    assert "What is the capital of France?" in prompt


def test_build_context_realistic_rerank_output():
    """LLMChatter.build_context with realistic rerank chunks produces source blocks with PDF name and citation."""
    chunks = _chunks(
        ["id1", "id2"],
        [
            {"text": "First relevant passage.", "source": "Report.pdf", "chapter": "Introduction", "page": 1},
            {"text": "Second passage.", "source": "Report.pdf", "chapter": "Methods", "page": 5},
        ],
    )
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert "Source: Report.pdf" in ctx
    assert "Introduction" in ctx and "Methods" in ctx
    assert "Page: 1" in ctx and "Page: 5" in ctx
    assert "First relevant passage." in ctx
    assert "Second passage." in ctx


def test_full_prompt_ready_for_llm():
    """Full prompt has correct structure for LLM: rules, sources block, question, answer placeholder."""
    chunks = _chunks(
        ["a", "b"],
        [
            {"text": "Content A.", "source": "DocA.pdf", "page": 2},
            {"text": "Content B.", "source": "DocB.pdf", "chapter": "Ch1", "page": 1},
        ],
    )
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "User question here?")
    lines = prompt.split("\n")
    assert any("Sources:" in line for line in lines)
    assert any("Question:" in line for line in lines)
    assert any("Answer:" in line for line in lines)
    assert "DocA.pdf" in prompt and "DocB.pdf" in prompt
    assert "Content A." in prompt and "Content B." in prompt
    assert "User question here?" in prompt


def test_create_prompt_empty_sources_still_produces_valid_prompt():
    """Empty chunks still yields a valid prompt with empty context."""
    chatter = LLMChatter()
    prompt = chatter.create_prompt([], "Any question?")
    assert "Sources:" in prompt
    assert "Question:" in prompt
    assert "Any question?" in prompt
    assert prompt.endswith("Answer:\n") or "Answer:" in prompt


def test_chat_returns_incomplete_when_query_empty():
    """chat() with empty query returns PROMPT_INCOMPLETE_RESPONSE and does not call LLM."""
    chunks = _chunks(
        ["c1"],
        [{"text": "Paris is the capital.", "source": "Geography.pdf", "chapter": "Europe", "page": 3}],
    )
    chatter = LLMChatter()
    result = chatter.chat(chunks, "")
    assert result == PROMPT_INCOMPLETE_RESPONSE


def test_chat_returns_incomplete_when_no_chunks():
    """chat() with no chunks returns PROMPT_INCOMPLETE_RESPONSE and does not call LLM."""
    chatter = LLMChatter()
    result = chatter.chat([], "What is the capital of France?")
    assert result == PROMPT_INCOMPLETE_RESPONSE

    result2 = chatter.chat({"chunks": []}, "What is the capital of France?")
    assert result2 == PROMPT_INCOMPLETE_RESPONSE


def test_chat_returns_incomplete_when_chunk_metadata_incomplete():
    """chat() when all chunks have incomplete metadata returns PROMPT_INCOMPLETE_RESPONSE."""
    chatter = LLMChatter()
    # Missing chapter
    chunks_no_chapter = _chunks(
        ["c1"],
        [{"text": "Paris is the capital.", "source": "Geography.pdf", "chapter": "", "page": 3}],
    )
    result = chatter.chat(chunks_no_chapter, "What is the capital?")
    assert result == PROMPT_INCOMPLETE_RESPONSE

    # Missing source
    chunks_no_source = _chunks(
        ["c1"],
        [{"text": "Paris is the capital.", "source": "", "chapter": "Europe", "page": 3}],
    )
    result2 = chatter.chat(chunks_no_source, "What is the capital?")
    assert result2 == PROMPT_INCOMPLETE_RESPONSE


def test_chat_excludes_incomplete_chunks_and_calls_llm_with_remaining():
    """chat() with one complete and one incomplete chunk uses only the complete chunk and calls LLM."""
    mock_client = MagicMock()
    mock_client.answer.return_value = "Paris."
    chunks = _chunks(
        ["incomplete", "complete"],
        [
            {"text": "Junk.", "source": "x.pdf", "chapter": "", "page": 1},
            {"text": "Paris is the capital of France.", "source": "Geography.pdf", "chapter": "Europe", "page": 3},
        ],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "What is the capital of France?")
    assert result == "Paris."
    mock_client.answer.assert_called_once()
    call_prompt = mock_client.answer.call_args[0][0]
    assert "Paris is the capital of France." in call_prompt
    assert "Europe" in call_prompt
    assert "Geography.pdf" in call_prompt


@pytest.mark.skipif(not RUN_LLM_TESTS, reason="Real LLM tests skipped; set RUN_LLM_TESTS=True in config to run")
def test_chat_real_llm_returns_non_empty_answer():
    """LLMChatter.chat() with default client calls real LlmClient and returns non-empty answer."""
    chunks = _chunks(
        ["c1"],
        [{"text": "Paris is the capital of France.", "source": "Geography.pdf", "chapter": "Europe", "page": 3}],
    )
    chatter = LLMChatter()
    answer = chatter.chat(chunks, "What is the capital of France?")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    assert "Paris" in answer or "capital" in answer.lower() or "France" in answer


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
