"""
Common configuration for AI_module (RAG application: infra layer, core, etc.).
"""

import logging
import os
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

# Hugging Face Hub id (used when local snapshot is missing; run dev_tools/download_embedding_model.py).
EMBEDDING_MODEL_HUB_ID: str = "intfloat/multilingual-e5-small"
# Full model snapshot for offline use (ignored by git under AI_module/data/).
EMBEDDING_MODEL_LOCAL_DIR: Path = PROJECT_ROOT / "data" / "models" / "multilingual-e5-small"


def _resolve_embedding_model_name() -> str:
    """Use project-local snapshot if present (offline-friendly); otherwise the Hub id."""
    local = EMBEDDING_MODEL_LOCAL_DIR.resolve()
    if local.is_dir() and (local / "config.json").exists():
        return str(local)
    return EMBEDDING_MODEL_HUB_ID


# Path string or Hub id passed to SentenceTransformer / AutoTokenizer.
EMBEDDING_MODEL_NAME: str = _resolve_embedding_model_name()
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# LLM prompt (RAG: sources + question -> prompt for LM)
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE: str = """You are an assistant answering a user's question using provided source chunks and recent conversation context.

INPUT:
- Question: {query}
- Previous conversation:
{history}
- Sources (numbered chunks):
{context}

INSTRUCTIONS:

1. If the Question contains references (like "it", "they", "there", "that"), resolve them using the previous conversation (1 user message + 1 assistant message).

2. Answer the Question using ONLY the provided Sources. Do NOT use outside knowledge.

3. Every sentence MUST include a citation in parentheses:
   (Source: <source>, Chapter: <chapter>, Page: <page>)

4. A sentence without a citation is INVALID and must NOT be included.

5. If a paragraph/bullet uses multiple chunks, include multiple citations.

6. If NONE of the provided Sources contain EXACT information that directly answers the Question,
output EXACTLY:
The information is not available in the provided document.

If at least one Source contains relevant information, you MUST provide the answer and MUST NOT output the fallback sentence.

7. Keep the answer concise and include only information that directly answers the question.

OUTPUT FORMAT:

<your answer with citations inline>
"""

PHI_MINI_LLM_PROMPT_TEMPLATE: str = """You are an assistant answering a user's question using provided sources and recent conversation context.

Question:
{query}

Previous conversation:
{history}

Chunks:
{context}

Instructions:

1. If the question contains words like "it", "they", "there", use the previous conversation to understand them.

2. Use ONLY the provided chunks.

3. Write only sentences that directly answer the question.

4. Every sentence MUST end with a citation in this format:
(Source: <source>, Chapter: <chapter>, Page: <page>)

5. Do not write any sentence without a citation.

6. If NONE of the provided Sources contain EXACT information that directly answers the Question,
output EXACTLY:
The information is not available in the provided document.

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

# RAG prompt template for LLMChatter (must include {history}, {context}, {query} placeholders).
# Point this at ``LM_PROMPT_TEMPLATE`` or ``PHI_MINI_LLM_PROMPT_TEMPLATE`` (or a custom string).
llm_model_prompt_template: str = PHI_MINI_LLM_PROMPT_TEMPLATE

# Ollama model tag for LlmClient (e.g. ``LLM_OLLAMA_MODEL_PHI_MINI`` or ``LLM_OLLAMA_MODEL_LATEST``).
llm_model: str = LLM_OLLAMA_MODEL_PHI_MINI

LLM_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MAX_NEW_TOKENS: int = 170

# Reference words used to decide whether prompt history is needed.
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
    "what about", "how about", "and what about",
]

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
QDRANT_LOCAL_HOST: str | None = os.getenv("QDRANT_HOST", "localhost")
QDRANT_LOCAL_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_PERSIST_DIRECTORY: str | None = "data/qdrant"
AUTO_START_QDRANT_SERVER: bool = True
AUTO_START_DOCKER: bool = True
DOCKER_START_TIMEOUT: float = 120.0
