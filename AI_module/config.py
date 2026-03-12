"""
Common configuration for AI_module (RAG application: infra layer, core, etc.).
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging (application logs under application/logs; config here)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR: Path = PROJECT_ROOT / "application" / "logs"
LOG_FILE: str | None = None
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_APP_LOG_FILENAME: str = "app.log"
ENABLE_LOGGING: bool = False

_INFRA_LOGGING_CONFIGURED = False


def configure_logging(
    log_file: str | None = LOG_FILE,
    level: str = LOG_LEVEL,
    fmt: str = LOG_FORMAT,
    enabled: bool | None = None,
) -> None:
    """Configure application logging to a file under LOG_DIR (application/logs). Idempotent.
    If enabled is False, no file handler is added and the logs directory is not created.
    """
    global _INFRA_LOGGING_CONFIGURED
    if _INFRA_LOGGING_CONFIGURED:
        return
    if not (enabled if enabled is not None else ENABLE_LOGGING):
        return
    logger = logging.getLogger("AI_module")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if fmt:
        formatter = logging.Formatter(fmt)
    else:
        formatter = logging.Formatter(LOG_FORMAT)
    path: Path
    if log_file:
        path = Path(log_file)
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / DEFAULT_APP_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    _INFRA_LOGGING_CONFIGURED = True


# ---------------------------------------------------------------------------
# Vector DB / retrieval
# ---------------------------------------------------------------------------

TOP_K_SIMILAR_CHUNKS: int = 20
SIMILARITY_METRIC: str = "cosine"
VECTOR_COLLECTION_NAME: str = "pdf_knowledge_base"
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K: int = 3

# ---------------------------------------------------------------------------
# PDF chunking and paths
# ---------------------------------------------------------------------------

PDF_INPUT_DIR = PROJECT_ROOT / "data" / "pdfs"

CHUNK_MAX_TOKENS: int = 256
CHUNK_OVERLAP_TOKENS: int = 70

IS_MULTICOLUMN: bool = False
CHAPTER_FONT_SIZE_MULTIPLIER: float = 1.25

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# LLM prompt (RAG: sources + question -> prompt for LM)
# ---------------------------------------------------------------------------

LM_PROMPT_TEMPLATE: str = """You are an assistant answering questions using the provided sources.

Rules:
- Use only the information from the sources.
- Cite the source after each statement using the format (Chapter, Page).
- If the answer is not in the sources, say: "The information is not available in the provided document."

Conversation history (last 2 user questions and 2 assistant answers, if any):
{history}

Sources:
{context}

Question:
{query}

Answer:
"""

# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------

LLM_OLLAMA_MODEL: str = "mistral:latest"
LLM_OLLAMA_HOST: str = "http://localhost:11434"
LLM_MAX_NEW_TOKENS: int = 300
LLM_LANGUAGE: str = "en"
LLM_LANGUAGE_CUSTOM_INSTRUCTION: str | None = None

_LLM_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "Respond in English.",
    "sk": "Odpovedz v slovenčine.",
}


def _get_llm_language_instruction() -> str:
    if LLM_LANGUAGE == "other" and LLM_LANGUAGE_CUSTOM_INSTRUCTION:
        return LLM_LANGUAGE_CUSTOM_INSTRUCTION
    return _LLM_LANGUAGE_INSTRUCTIONS.get(LLM_LANGUAGE, "Respond in English.")


LLM_LANGUAGE_INSTRUCTION: str = _get_llm_language_instruction()
RUN_LLM_TESTS: bool = True
AUTO_START_OLLAMA_SERVER: bool = True
OLLAMA_SERVE_READY_TIMEOUT: float = 15.0

# ---------------------------------------------------------------------------
# Vector DB storage
# ---------------------------------------------------------------------------
#storage type can be server or memory
STORAGE_TYPE: str = "server"
QDRANT_LOCAL_HOST: str | None = "localhost"
QDRANT_LOCAL_PORT: int = 6333
QDRANT_PERSIST_DIRECTORY: str | None = "data/qdrant"
AUTO_START_QDRANT_SERVER: bool = True
AUTO_START_DOCKER: bool = True
DOCKER_START_TIMEOUT: float = 120.0
