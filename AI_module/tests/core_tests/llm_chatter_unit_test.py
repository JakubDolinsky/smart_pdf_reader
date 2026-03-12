"""
Unit tests for core.llm_chatter (LLMChatter). LlmClient mocked for chat().
For integration tests with real config/LLM see llm_chatter_integration_test.py.
Run: python -m pytest AI_module/tests/core_tests/llm_chatter_unit_test.py -v
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


def _chunks_list(ids, metadatas):
    """Build list of Chunk from ids and metadatas (rerank-style)."""
    return [
        Chunk(id=cid, payload=dict(meta), vector=None)
        for cid, meta in zip(ids, metadatas)
    ]


def test_build_context_empty_metadatas_returns_empty_string():
    chatter = LLMChatter()
    assert chatter.build_context([]) == ""
    assert chatter.build_context({"chunks": []}) == ""


def test_build_context_missing_chunks_key_returns_empty_string():
    chatter = LLMChatter()
    assert chatter.build_context({"ids": ["a"]}) == ""


def test_build_context_single_chunk_with_source_chapter_page():
    chunks = _chunks_list(
        ["c1"],
        [{"text": "Paris is the capital.", "source": "doc.pdf", "chapter": "Europe", "page": 3}],
    )
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert "Source: doc.pdf" in ctx
    assert "Chapter: Europe" in ctx
    assert "Page: 3" in ctx
    assert "Paris is the capital." in ctx
    assert "[Source: doc.pdf; Chapter: Europe, Page: 3]" in ctx


def test_build_context_uses_page_start_if_page_missing():
    chunks = _chunks_list(["id1"], [{"text": "Content.", "source": "x.pdf", "page_start": 5}])
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert "Page: 5" in ctx
    assert "Content." in ctx


def test_build_context_multiple_chunks_joined_by_double_newline():
    chunks = _chunks_list(
        ["c1", "c2"],
        [
            {"text": "First.", "source": "a.pdf", "page": 1},
            {"text": "Second.", "source": "b.pdf", "chapter": "Ch2", "page": 2},
        ],
    )
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert "First." in ctx and "Second." in ctx
    assert ctx.count("\n\n") >= 1
    assert "a.pdf" in ctx and "b.pdf" in ctx


def test_build_context_skips_chunk_with_empty_text():
    chunks = _chunks_list(
        ["a", "b", "c"],
        [
            {"text": "Keep.", "source": "s.pdf", "page": 1},
            {"text": "", "source": "s.pdf", "page": 2},
            {"text": "  \n ", "source": "s.pdf", "page": 3},
        ],
    )
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert "Keep." in ctx
    assert ctx.count("Keep.") == 1


def test_build_context_no_source_or_chapter_page_still_includes_text():
    chunks = _chunks_list(["x"], [{"text": "Only text."}])
    chatter = LLMChatter()
    ctx = chatter.build_context(chunks)
    assert ctx.strip() == "Only text."


def test_create_prompt_includes_context_and_query():
    chunks = _chunks_list(
        ["id1"],
        [{"text": "The answer is 42.", "source": "guide.pdf", "page": 1}],
    )
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "What is the answer?")
    assert "The answer is 42." in prompt
    assert "What is the answer?" in prompt
    assert "Sources:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "Conversation history" in prompt


def test_create_prompt_includes_history_when_provided():
    """When history is passed, last 4 messages appear in the prompt."""
    chunks = _chunks_list(["c1"], [{"text": "Content.", "source": "doc.pdf", "chapter": "Ch1", "page": 1}])
    history = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question?"},
        {"role": "assistant", "content": "Second answer."},
    ]
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "Third question?", history=history)
    assert "User: First question?" in prompt
    assert "Assistant: First answer." in prompt
    assert "User: Second question?" in prompt
    assert "Assistant: Second answer." in prompt
    assert "Third question?" in prompt
    assert "Content." in prompt


def test_create_prompt_history_takes_last_four_messages():
    """Only the last 4 history messages are included."""
    chunks = _chunks_list(["c1"], [{"text": "X.", "source": "s.pdf", "page": 1}])
    history = [
        {"role": "user", "content": "Old1"},
        {"role": "assistant", "content": "Old2"},
        {"role": "user", "content": "Recent1"},
        {"role": "assistant", "content": "Recent2"},
    ]
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "Q?", history=history)
    assert "Recent1" in prompt and "Recent2" in prompt
    assert "Old1" in prompt and "Old2" in prompt
    history_extra = history + [{"role": "user", "content": "Newest"}]
    prompt2 = chatter.create_prompt(chunks, "Q?", history=history_extra)
    assert "Newest" in prompt2
    assert "Old1" not in prompt2
    assert "Old2" in prompt2


def test_create_prompt_strips_query_whitespace():
    chunks = _chunks_list(["a"], [{"text": "X.", "source": "s.pdf"}])
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "  \n  Your question?  \n ")
    assert "Your question?" in prompt


def test_create_prompt_empty_chunks():
    template = "History: {history}\nContext: {context}\nQuery: {query}"
    chatter = LLMChatter()
    prompt = chatter.create_prompt([], "Q?", template=template)
    assert prompt == "History: \nContext: \nQuery: Q?"
    prompt2 = chatter.create_prompt({"chunks": []}, "Q?", template=template)
    assert prompt2 == "History: \nContext: \nQuery: Q?"


def test_create_prompt_custom_template():
    chunks = _chunks_list(["a"], [{"text": "C.", "source": "f.pdf"}])
    custom = "History: {history}\nSources:\n{context}\n\nQ: {query}\nA:"
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, "Ask", template=custom)
    assert "History:" in prompt
    assert "Sources:" in prompt
    assert "C." in prompt
    assert "f.pdf" in prompt
    assert "Q: Ask" in prompt
    assert "A:" in prompt


def test_create_prompt_none_query_formatted_as_empty():
    chunks = _chunks_list(["a"], [{"text": "T.", "source": "s.pdf"}])
    template = "H: {history}\nC: {context}\nQ: {query}"
    chatter = LLMChatter()
    prompt = chatter.create_prompt(chunks, None, template=template)
    assert "C:" in prompt
    assert "Q: " in prompt or "Q:\n" in prompt


def test_chat_builds_prompt_and_returns_client_answer():
    mock_client = MagicMock()
    mock_client.answer.return_value = "The capital is Paris."
    chunks = _chunks_list(
        ["c1"],
        [{"text": "Paris is the capital of France.", "source": "doc.pdf", "chapter": "Intro", "page": 1}],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "What is the capital of France?")
    assert result == "The capital is Paris."
    mock_client.answer.assert_called_once()
    call_prompt = mock_client.answer.call_args[0][0]
    assert "Paris" in call_prompt
    assert "What is the capital of France?" in call_prompt


def test_chat_returns_incomplete_when_no_chunks():
    mock_client = MagicMock()
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat([], "Any question?")
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_client.answer.assert_not_called()

    result2 = chatter.chat({"chunks": []}, "Any question?")
    assert result2 == PROMPT_INCOMPLETE_RESPONSE
    assert mock_client.answer.call_count == 0


@pytest.mark.parametrize("query", ["", None, "   \n\t  "])
def test_chat_returns_incomplete_when_query_empty_or_whitespace(query):
    """Empty, None, or whitespace-only query returns PROMPT_INCOMPLETE_RESPONSE without calling LLM."""
    mock_client = MagicMock()
    chunks = _chunks_list(
        ["c1"],
        [{"text": "Some content.", "source": "doc.pdf", "chapter": "Ch1", "page": 1}],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, query)
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_client.answer.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "source": "doc.pdf", "chapter": "Ch1", "page": 1},
        {"source": "doc.pdf", "chapter": "Ch1", "page": 1},
        {"text": "Content.", "source": "", "chapter": "Ch1", "page": 1},
        {"text": "Content.", "source": "doc.pdf", "chapter": "", "page": ""},
    ],
    ids=["empty_text", "missing_text", "empty_source", "no_chapter_no_page"],
)
def test_chat_returns_incomplete_when_single_chunk_metadata_incomplete(payload):
    """Single chunk with incomplete metadata (missing text, source, or both chapter and page) returns PROMPT_INCOMPLETE_RESPONSE."""
    mock_client = MagicMock()
    chunks = _chunks_list(["c1"], [payload])
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "Question?")
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_client.answer.assert_not_called()


def test_chat_uses_custom_template_when_passed():
    mock_client = MagicMock()
    mock_client.answer.return_value = "42"
    template = "History: {history}\nContext: {context}\nQ: {query}"
    chunks = _chunks_list(["a"], [{"text": "Answer is 42.", "source": "x.pdf", "chapter": "Ch1", "page": 1}])
    chatter = LLMChatter(client=mock_client)
    chatter.chat(chunks, "What is it?", template=template)
    call_prompt = mock_client.answer.call_args[0][0]
    assert "History:" in call_prompt and "Context:" in call_prompt and "Q:" in call_prompt
    assert "Answer is 42." in call_prompt and "What is it?" in call_prompt


def test_chat_passes_history_to_llm():
    """chat(..., history=[...]) passes formatted history in the prompt to the client."""
    mock_client = MagicMock()
    mock_client.answer.return_value = "Based on history: Third answer."
    chunks = _chunks_list(["c1"], [{"text": "Doc content.", "source": "doc.pdf", "chapter": "Ch1", "page": 1}])
    history = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Second answer."},
    ]
    chatter = LLMChatter(client=mock_client)
    chatter.chat(chunks, "Third?", history=history)
    call_prompt = mock_client.answer.call_args[0][0]
    assert "User: First?" in call_prompt
    assert "Assistant: First answer." in call_prompt
    assert "User: Second?" in call_prompt
    assert "Assistant: Second answer." in call_prompt
    assert "Third?" in call_prompt


def test_chat_excludes_incomplete_chunks_and_calls_llm_with_remaining():
    """When some chunks have incomplete metadata, only chunks with complete metadata are used; LLM is called."""
    mock_client = MagicMock()
    mock_client.answer.return_value = "Paris is the capital."
    chunks = _chunks_list(
        ["bad", "good"],
        [
            {"text": "", "source": "doc.pdf", "chapter": "Ch1", "page": 1},
            {"text": "Paris is the capital of France.", "source": "doc.pdf", "chapter": "Intro", "page": 1},
        ],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "What is the capital of France?")
    assert result == "Paris is the capital."
    mock_client.answer.assert_called_once()
    call_prompt = mock_client.answer.call_args[0][0]
    assert "Paris is the capital of France." in call_prompt
    assert "Intro" in call_prompt


def test_chat_returns_incomplete_when_all_chunks_incomplete():
    """When every chunk has incomplete metadata (e.g. missing source or neither chapter nor page), return PROMPT_INCOMPLETE_RESPONSE."""
    mock_client = MagicMock()
    chunks = _chunks_list(
        ["a", "b"],
        [
            {"text": "Content.", "source": "", "chapter": "Ch1", "page": 1},
            {"text": "More.", "source": "x.pdf", "chapter": "", "page": ""},
        ],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "Question?")
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_client.answer.assert_not_called()


def test_chat_calls_client_when_chunk_has_page_start_instead_of_page():
    mock_client = MagicMock()
    mock_client.answer.return_value = "OK"
    chunks = _chunks_list(
        ["c1"],
        [{"text": "Content.", "source": "doc.pdf", "chapter": "Ch1", "page_start": 5}],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "Question?")
    assert result == "OK"
    mock_client.answer.assert_called_once()


def test_chat_calls_llm_when_chunk_has_page_but_empty_chapter():
    """Chunk with page (or page_start) but empty chapter is still used when chapter cannot be detected."""
    mock_client = MagicMock()
    mock_client.answer.return_value = "Answer."
    chunks = _chunks_list(
        ["c1"],
        [{"text": "Content.", "source": "doc.pdf", "chapter": "", "page": 3}],
    )
    chatter = LLMChatter(client=mock_client)
    result = chatter.chat(chunks, "Question?")
    assert result == "Answer."
    mock_client.answer.assert_called_once()
    call_prompt = mock_client.answer.call_args[0][0]
    assert "Content." in call_prompt
    assert "Page: 3" in call_prompt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
