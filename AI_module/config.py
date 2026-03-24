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
LOG_LEVEL: str = "DEBUG"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_APP_LOG_FILENAME: str = "app.log"
ENABLE_LOGGING: bool = True

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

TOP_K_SIMILAR_CHUNKS: int = 7
SIMILARITY_METRIC: str = "cosine"
VECTOR_COLLECTION_NAME: str = "pdf_knowledge_base"
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K: int = 3

# Shown when the vector DB returned candidates but reranking returned no chunks (e.g. invalid input)
# or a similar failure. The main RAG LLM is not called in that case.
RAG_NO_INFORMATION_IN_DOCUMENT: str = (
    "The information is not available in the provided document."
)

# ---------------------------------------------------------------------------
# PDF chunking and paths
# ---------------------------------------------------------------------------

PDF_INPUT_DIR = PROJECT_ROOT / "data" / "pdfs"

CHUNK_MAX_TOKENS: int = 200
# Paragraphs shorter than this (in tokens) are merged with neighbors until >= min or max would be exceeded.
CHUNK_MIN_TOKENS: int = 30
CHUNK_OVERLAP_TOKENS: int = 70
# Max tokens for a line/group accepted as a font-based chapter title (excludes long paragraphs).
CHUNK_TITLE_MAX_TOKENS: int = 25

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
- Name the entity to which was the question related in every answer.
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

# Keep both model options available and allow switching via a single selector.
# Note: Ollama model tags are whatever you have locally (see `ollama list`).
LLM_OLLAMA_MODEL_LATEST: str = "mistral:latest"

# Quantized Mistral option (Q4_K). Adjust this string to the exact Ollama tag
# you have installed for the Q4_K model.
LLM_OLLAMA_MODEL_Q4_K: str = "mistral:7b-instruct-q4_K_S"

LLM_OLLAMA_MODEL_PHI_MINI: str = "phi3:mini"

LLM_OLLAMA_HOST: str = "http://localhost:11434"
LLM_MAX_NEW_TOKENS: int = 170

# Query rewriter (Ollama): separate from main RAG LLM; keep short generation for one-line output.
REWRITER_OLLAMA_MODEL: str = "phi3:mini"
REWRITER_MAX_NEW_TOKENS: int = 40

# Referential questions: trigger rewriter when any of these appear as whole words/phrases (see core.rewriting).
REFERENCE_WORDS: list[str] = [
    "this", "that", "these", "those",
    "he", "she", "they", "him", "her", "them",
    "his", "her", "their", "its",
    "the former", "the latter", "one", "ones",
    "there", "here", "it",
    "this thing", "that thing",
    "this event", "that event",
    "this situation", "that situation",
    "someone", "something", "someone else", "something else",
    "the first", "the second", "the third",
    "the previous", "the next",
    "the same", "the other", "another",
    "then", "at that time", "during that time",
    "after that", "before that",
    "such", "such a", "such an",
    "what about", "how about", "and what about"
]

REWRITE_PROMPT_TEMPLATE: str = """You are a query rewriting assistant.

Rewrite the user's question so that it is self-contained and explicit.
Use the context from the previous question and its answer to resolve any pronouns or ambiguous references.

Previous question: {prev_question}
Previous answer: {prev_answer}
User's current question: {current_question}

Output only the rewritten question.
"""

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
