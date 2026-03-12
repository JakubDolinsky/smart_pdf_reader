"""
Unit tests for question_pipeline.answer_question. All core services (embedding, DB, reranker, LLM) are mocked.
For integration tests with real DB and LLM see question_pipeline_integration_test.py.
Run: python -m pytest AI_module/tests/application_tests/question_pipeline_tests/question_pipeline_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_file = Path(__file__).resolve()
_root = _file.parents[3]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunk import Chunk
from AI_module.application.question_pipeline.question_pipeline_orchestration import answer_question
from AI_module.core.llm_chatter import PROMPT_INCOMPLETE_RESPONSE


def _chunk(id_: str, text: str, source: str = "doc.pdf", chapter: str = "Ch1", page: int = 1) -> Chunk:
    return Chunk(
        id=id_,
        payload={"text": text, "source": source, "chapter": chapter, "page": page},
        vector=None,
    )


def test_answer_question_empty_returns_incomplete_response():
    """Empty or whitespace-only question returns PROMPT_INCOMPLETE_RESPONSE without calling services."""
    assert answer_question("") == PROMPT_INCOMPLETE_RESPONSE
    assert answer_question("   \n\t  ") == PROMPT_INCOMPLETE_RESPONSE


def test_answer_question_no_similar_chunks_returns_incomplete_response():
    """When search_similar returns no chunks, answer_question returns PROMPT_INCOMPLETE_RESPONSE."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": []}

    result = answer_question(
        "What is the main topic?",
        embedding_service=mock_embed,
        db_manager=mock_db,
    )
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_embed.embed_query.assert_called_once_with("What is the main topic?", history=None)
    mock_db.search_similar.assert_called_once()


def test_answer_question_rerank_returns_no_chunks_returns_incomplete_response():
    """When rerank returns no chunks, answer_question returns PROMPT_INCOMPLETE_RESPONSE."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "some text")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": []}

    result = answer_question(
        "What is it?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
    )
    assert result == PROMPT_INCOMPLETE_RESPONSE
    mock_reranker.rerank.assert_called_once()


def test_answer_question_full_flow_returns_llm_answer():
    """With mocks: embed -> search -> rerank -> chat; answer_question returns LLM answer."""
    query = "What is the capital?"
    query_embedding = [0.2] * 384
    chunks_from_db = [
        _chunk("c1", "Paris is the capital of France.", "doc.pdf", "Europe", 3),
        _chunk("c2", "Berlin is the capital of Germany.", "doc.pdf", "Europe", 4),
    ]
    reranked_chunks = [chunks_from_db[0], chunks_from_db[1]]
    llm_answer = "Paris is the capital of France."

    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = query_embedding
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": chunks_from_db}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": reranked_chunks}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = llm_answer

    result = answer_question(
        query,
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
    )
    assert result == llm_answer

    mock_embed.embed_query.assert_called_once_with(query, history=None)
    mock_db.search_similar.assert_called_once()
    call_kw = mock_db.search_similar.call_args[1]
    assert call_kw["top_k"] == 20
    assert call_kw["include_scores"] is True
    mock_reranker.rerank.assert_called_once_with(query, {"chunks": chunks_from_db}, top_k=3)
    mock_chatter.chat.assert_called_once()
    chat_chunks = mock_chatter.chat.call_args[0][0]
    assert chat_chunks == {"chunks": reranked_chunks}
    assert mock_chatter.chat.call_args[0][1] == query


def test_answer_question_passes_history_to_chatter():
    """When history is passed, it is forwarded to llm_chatter.chat(..., history=history)."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Answer with history."

    history = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question?"},
        {"role": "assistant", "content": "Second answer."},
    ]
    result = answer_question(
        "Third question?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
        history=history,
    )
    assert result == "Answer with history."
    mock_chatter.chat.assert_called_once()
    call_kw = mock_chatter.chat.call_args[1]
    assert call_kw.get("history") == history  # history_for_llm is last 4 chronological = history when len 4


def test_answer_question_calls_search_with_custom_top_k():
    """top_k_similar and rerank_top_k are passed to search_similar and rerank."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("x", "text")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("x", "text")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Answer"

    answer_question(
        "Q?",
        top_k_similar=10,
        rerank_top_k=2,
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
    )
    mock_db.search_similar.assert_called_once()
    assert mock_db.search_similar.call_args[1]["top_k"] == 10
    mock_reranker.rerank.assert_called_once()
    assert mock_reranker.rerank.call_args[1]["top_k"] == 2


def test_answer_question_uses_passed_collection_name():
    """When collection_name is passed, DBManager is created with that name (no mock DB)."""
    from unittest.mock import patch
    with patch("AI_module.application.question_pipeline.question_pipeline_orchestration.DBManager") as mock_db_cls:
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.0] * 384
        mock_db_instance = MagicMock()
        mock_db_instance.search_similar.return_value = {"chunks": []}
        mock_db_cls.return_value = mock_db_instance

        answer_question("Q?", collection_name="my_collection", embedding_service=mock_embed)
        mock_db_cls.assert_called_once_with(collection_name="my_collection", host=None, port=None)


def test_answer_question_strips_question_whitespace():
    """Question is stripped before embedding and passed to rerank/chat."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c", "t")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("c", "t")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Ok"

    answer_question("  What is it?  ", embedding_service=mock_embed, db_manager=mock_db,
                    reranking_service=mock_reranker, llm_chatter=mock_chatter)
    mock_embed.embed_query.assert_called_once_with("What is it?", history=None)
    assert mock_chatter.chat.call_args[0][1] == "What is it?"


def test_answer_question_with_reference_word_passes_history_to_embed():
    """When question contains a reference word (e.g. 'that') and history is passed, embed_query is called with history so it can append prior context."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Answer."

    history_reversed = [{"role": "assistant", "content": "Paris is the capital of France."}]
    answer_question(
        "What about that?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
        history_last_four_messages_reversed=history_reversed,
    )
    mock_embed.embed_query.assert_called_once()
    call_args = mock_embed.embed_query.call_args
    assert call_args[0][0] == "What about that?"
    assert call_args[1].get("history") == history_reversed


def test_answer_question_history_last_four_reversed_used_for_llm_chronological():
    """When history_last_four_messages_reversed is passed, chatter receives chronological (reversed) list."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.0] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "Text.")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("c1", "Text.")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Ok"

    # First = last message (most recent), so chronological for LLM = [oldest, ..., newest]
    history_reversed = [
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "First?"},
    ]
    answer_question(
        "Third?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
        history_last_four_messages_reversed=history_reversed,
    )
    call_kw = mock_chatter.chat.call_args[1]
    expected_chronological = list(reversed(history_reversed))
    assert call_kw.get("history") == expected_chronological


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
