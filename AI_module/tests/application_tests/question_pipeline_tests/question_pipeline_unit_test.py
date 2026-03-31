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
from AI_module.config import RAG_NO_INFORMATION_IN_DOCUMENT, TOP_K_SIMILAR_CHUNKS
from AI_module.core.llm_chatter import PROMPT_INCOMPLETE_RESPONSE


def _chunk(id_: str, text: str, source: str = "doc.pdf", path: str = "Ch1", page: int = 1) -> Chunk:
    return Chunk(
        id=id_,
        payload={"text": text, "source": source, "path": path, "page": page},
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
    mock_embed.embed_query.assert_called_once_with("What is the main topic?")
    mock_db.search_similar.assert_called_once()


def test_answer_question_rerank_returns_no_chunks_returns_no_information_without_llm():
    """When rerank returns no chunks, return RAG message and do not call LLM."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "some text")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": []}
    mock_chatter = MagicMock()

    result = answer_question(
        "What is it?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
    )
    assert result == RAG_NO_INFORMATION_IN_DOCUMENT
    mock_reranker.rerank.assert_called_once()
    mock_chatter.chat.assert_not_called()


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

    mock_embed.embed_query.assert_called_once_with(query)
    mock_db.search_similar.assert_called_once()
    call_kw = mock_db.search_similar.call_args[1]
    assert call_kw["top_k"] == TOP_K_SIMILAR_CHUNKS
    assert call_kw["include_scores"] is True
    mock_reranker.rerank.assert_called_once_with(query, {"chunks": chunks_from_db}, top_k=3)
    mock_chatter.chat.assert_called_once()
    chat_chunks = mock_chatter.chat.call_args[0][0]
    assert chat_chunks == {"chunks": reranked_chunks}
    assert mock_chatter.chat.call_args[0][1] == query


def test_answer_question_passes_history_to_chatter():
    """Without reference words in query, history passed to llm_chatter is empty."""
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
    assert call_kw.get("history") == []


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
    mock_embed.embed_query.assert_called_once_with("What is it?")
    assert mock_chatter.chat.call_args[0][1] == "What is it?"


def test_answer_question_embeds_current_question_with_context_from_history():
    """Embedding input is current question plus Context block with user + reduced assistant content."""
    mock_embed = MagicMock()
    query_vec = [0.0] * 384
    sent1 = [1.0, 0.0]
    sent2 = [0.8, 0.2]
    sent3 = [0.6, 0.4]
    sent4 = [0.0, 1.0]
    # First call: embedding query; then four sentence embeddings for assistant sentence scoring;
    # last call: final merged query embedding used for vector search.
    mock_embed.embed_query.side_effect = [
        query_vec,  # assistant ranking: query
        sent1, sent2, sent3, sent4,  # assistant sentences
        query_vec,  # final merged embedding
    ]
    mock_db = MagicMock()
    mock_db.search_similar.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = {"chunks": [_chunk("c1", "Some text.")]}
    mock_chatter = MagicMock()
    mock_chatter.chat.return_value = "Answer."

    history = [
        {"role": "user", "content": "What is the capital?"},
        {
            "role": "assistant",
            "content": "Paris is the capital. It is in Europe. It has the Eiffel Tower. Bananas are yellow.",
        },
    ]
    answer_question(
        "What about that?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
        history=history,
    )
    assert mock_embed.embed_query.call_count == 6
    final_embed_input = mock_embed.embed_query.call_args_list[-1].args[0]
    assert final_embed_input.startswith("What about that?\n\nContext:\n")
    assert "user: What is the capital?" in final_embed_input
    assert "assistant: Paris is the capital. It is in Europe. It has the Eiffel Tower." in final_embed_input
    assert "Bananas are yellow." not in final_embed_input


def test_answer_question_history_last_two_reversed_used_for_llm_chronological():
    """Without reference words in query, provided reversed history is ignored for LLM prompt history."""
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
        {"role": "assistant", "content": "Second answer. More detail here."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "First answer. More detail there."},
        {"role": "user", "content": "First?"},
    ]
    answer_question(
        "Third?",
        embedding_service=mock_embed,
        db_manager=mock_db,
        reranking_service=mock_reranker,
        llm_chatter=mock_chatter,
        history_last_two_messages_reversed=history_reversed,
    )
    call_kw = mock_chatter.chat.call_args[1]
    assert call_kw.get("history") == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
