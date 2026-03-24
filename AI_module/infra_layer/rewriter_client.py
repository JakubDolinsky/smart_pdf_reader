"""
HTTP client for query rewriting via Ollama. Uses a dedicated small model (e.g. phi3:mini),
separate from the main RAG LLM model configured on LlmClient.
"""

from __future__ import annotations

import logging
from typing import Any

from AI_module.config import (
    LLM_OLLAMA_HOST,
    REWRITER_MAX_NEW_TOKENS,
    REWRITER_OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class RewriterClient:
    """
    Sends rewriting prompts to Ollama; output should be a single self-contained question line.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        max_new_tokens: int = REWRITER_MAX_NEW_TOKENS,
    ) -> None:
        self._model = model if model is not None else REWRITER_OLLAMA_MODEL
        self._host = host if host is not None else LLM_OLLAMA_HOST
        self._max_new_tokens = max_new_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as e:
                raise ImportError(
                    "RewriterClient requires the ollama package. Install with: pip install ollama"
                ) from e
            self._client = Client(host=self._host)
            logger.info(
                "Ollama rewriter client ready (host=%s, model=%s)",
                self._host,
                self._model,
            )
        return self._client

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        response = client.chat(
            model=self._model,
            messages=messages,
            options={"num_predict": self._max_new_tokens},
        )
        content = (response.get("message") or {}).get("content") or ""
        return content.strip()

    def rewrite(self, prompt: str) -> str:
        """
        Send the full rewriting instructions + context as the user message; return the model text.

        Args:
            prompt: Non-empty rewriting prompt (built by core.rewriting).

        Returns:
            Trimmed model output (expected: one rewritten question).
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        normalized = prompt.strip()
        messages = [{"role": "user", "content": normalized}]
        logger.debug(
            "Sending rewriter prompt (model=%s, len=%d)",
            self._model,
            len(normalized),
        )
        out = self._call_ollama(messages)
        logger.debug("Rewriter output generated (model=%s)", self._model)
        return out
