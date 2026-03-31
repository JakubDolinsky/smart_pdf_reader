"""
RAG service: answer a single question (for FastAPI or programmatic use) or run the CLI question loop.
RAG tools (embedder, reranker, LLM chatter, DB manager) are initialized once per process and reused
for all calls when using the default host/port, so CLI and FastAPI avoid repeated model loading.
"""

from __future__ import annotations

import logging

from AI_module.config import (
    QDRANT_LOCAL_HOST,
    QDRANT_LOCAL_PORT,
    VECTOR_COLLECTION_NAME,
)
from AI_module.core.db_manager import DBManager
from AI_module.core.embedding import EmbeddingService
from AI_module.core.llm_chatter import LLMChatter
from AI_module.core.reranking import RerankingService
from AI_module.infra_layer import check_db_ready
from AI_module.application.question_pipeline import answer_question

logger = logging.getLogger(__name__)

# Lazy-initialized once per process; reused for default host/port.
_embedder: EmbeddingService | None = None
_reranker: RerankingService | None = None
_chatter: LLMChatter | None = None
_db_manager: DBManager | None = None


def _get_services(
    host: str | None = None,
    port: int | None = None,
) -> tuple[EmbeddingService, DBManager, RerankingService, LLMChatter]:
    """
    Return shared RAG pipeline services, creating them once when using the default host/port.
    When host/port are passed (non-default), embedder/reranker/chatter are still shared;
    only DBManager is created for the given host/port (not cached).
    """
    global _embedder, _reranker, _chatter, _db_manager
    h = host if host is not None else QDRANT_LOCAL_HOST
    p = port if port is not None else QDRANT_LOCAL_PORT
    use_default = host is None and port is None

    if _embedder is None:
        logger.info("RAG service: initializing embedding model (once per process).")
        _embedder = EmbeddingService()
    if _reranker is None:
        logger.info("RAG service: initializing reranker model (once per process).")
        _reranker = RerankingService()
    if _chatter is None:
        _chatter = LLMChatter()

    if use_default:
        if _db_manager is None:
            logger.info("RAG service: initializing DB manager for default host/port (once per process).")
            _db_manager = DBManager(collection_name=VECTOR_COLLECTION_NAME, host=h, port=p)
        return _embedder, _db_manager, _reranker, _chatter

    # Caller passed host/port (e.g. tests): reuse embedder/reranker/chatter, create DB for this call.
    db = DBManager(collection_name=VECTOR_COLLECTION_NAME, host=h, port=p)
    return _embedder, db, _reranker, _chatter


def get_answer(
    question: str,
    *,
    host: str | None = None,
    port: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Answer a single question using the RAG pipeline (embed -> search -> rerank -> LLM).

    Uses shared pipeline services when host/port are the defaults.
    Optional history: last 2 messages (1 user + 1 assistant) are included in the prompt.

    Returns:
        Answer string; config RAG_NO_INFORMATION_IN_DOCUMENT when reranking returns no chunks;
        PROMPT_INCOMPLETE_RESPONSE when question is empty or vector search returns no chunks.
    """
    h = host if host is not None else QDRANT_LOCAL_HOST
    p = port if port is not None else QDRANT_LOCAL_PORT
    embedder, db, reranker, chatter = _get_services(host=host, port=port)
    return answer_question(
        question,
        embedding_service=embedder,
        db_manager=db,
        reranking_service=reranker,
        llm_chatter=chatter,
        history=history,
    )


def run_cli_loop(
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    Run the interactive CLI: check DB, then loop prompting for questions and printing answers
    until the user enters an empty question or Ctrl+C / EOF. Uses shared RAG services (init once).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    h = host if host is not None else QDRANT_LOCAL_HOST
    p = port if port is not None else QDRANT_LOCAL_PORT

    if not check_db_ready(host=h, port=p):
        print(
            f"Qdrant DB is not reachable at {h}:{p}. "
            "Start the DB server first (e.g. dev_tools/start_app_db.bat)."
        )
        return

    prompt = "Ask a question about the ingested PDFs (empty to exit): "
    while True:
        try:
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            print("Exiting.")
            break
        if host is None and port is None:
            answer = get_answer(question)
        else:
            answer = get_answer(question, host=h, port=p)
        print("\nAnswer:\n")
        print(answer)
        print()
