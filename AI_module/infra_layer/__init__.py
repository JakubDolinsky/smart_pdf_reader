"""Infrastructure layer: vector DB, embeddings, LLM client, Ollama lifecycle, DB health check. Config: AI_module.config."""

from .db_health import check_db_ready, is_qdrant_server_running
from .llm_client import LlmClient
from .ollama_lifecycle import (
    is_ollama_running,
    managed as ollama_managed,
    register_application_exit_handlers,
    start_ollama,
    stop_ollama,
)

__all__ = [
    "check_db_ready",
    "is_ollama_running",
    "is_qdrant_server_running",
    "LlmClient",
    "ollama_managed",
    "register_application_exit_handlers",
    "start_ollama",
    "stop_ollama",
]
