"""
LLM client for RAG: send a prompt to Mistral 7B Instruct via Ollama and get an answer.
Communication language is configurable via LLM_LANGUAGE / LLM_LANGUAGE_INSTRUCTION in config.
"""

import logging
from typing import Any

from AI_module.config import (
    LLM_LANGUAGE_INSTRUCTION,
    LLM_MAX_NEW_TOKENS,
    LLM_OLLAMA_HOST,
    LLM_OLLAMA_MODEL_PHI_MINI,
)

logger = logging.getLogger(__name__)


class LlmClient:
    """
    Client for question-answering with Mistral 7B Instruct via Ollama.
    Uses Ollama's API (default http://localhost:11434); run: ollama pull mistral:7b-instruct
    - answer(prompt): send a prompt (question), get the model's answer.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        language_instruction: str = LLM_LANGUAGE_INSTRUCTION,
    ) -> None:
        """
        Args:
            model: Ollama model name (e.g. mistral:7b-instruct). If None, uses LLM_OLLAMA_MODEL_Q4_K from config.
            host: Ollama server URL. If None, uses LLM_OLLAMA_HOST from config.
            max_new_tokens: Maximum tokens to generate for the answer.
            language_instruction: Instruction so the model responds in the desired language (e.g. English or Slovak).
        """
        self._model = model if model is not None else LLM_OLLAMA_MODEL_PHI_MINI
        self._host = host if host is not None else LLM_OLLAMA_HOST
        self._max_new_tokens = max_new_tokens
        self._language_instruction = language_instruction
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init Ollama client (avoids import at module load)."""
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as e:
                raise ImportError(
                    "LlmClient requires the ollama package. Install with: pip install ollama"
                ) from e
            self._client = Client(host=self._host)
            logger.info("Ollama client ready (host=%s, model=%s)", self._host, self._model)
        return self._client

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        """
        Send messages to Ollama chat API and return the assistant message content.
        Extracted so tests can mock this method.
        """
        client = self._get_client()
        response = client.chat(
            model=self._model,
            messages=messages,
            options={"num_predict": self._max_new_tokens},
        )
        # Response shape: {"message": {"role": "assistant", "content": "..."}, ...}
        content = (response.get("message") or {}).get("content") or ""
        return content.strip()

    def answer(self, prompt: str) -> str:
        """
        Send a prompt (question) to the LLM and return the generated answer.
        The answer language follows config LLM_LANGUAGE_INSTRUCTION.

        Args:
            prompt: User question or prompt.

        Returns:
            The model's answer as a string (trimmed).
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        normalized_prompt = prompt.strip()

        messages = [
            {"role": "system", "content": self._language_instruction.strip()},
            {"role": "user", "content": normalized_prompt},
        ]

        # Logging the actual prompt helps debugging RAG prompt construction issues.
        # Keep it bounded to avoid huge logs (prompts may include large chunk text).
        max_logged_chars = 2000
        prompt_for_log = (
            normalized_prompt[:max_logged_chars] + "…(truncated)"
            if len(normalized_prompt) > max_logged_chars
            else normalized_prompt
        )
        logger.debug(
            "Sending LLM prompt (model=%s, prompt_len=%d): %s",
            self._model,
            len(normalized_prompt),
            prompt_for_log,
        )

        answer = self._call_ollama(messages)
        logger.debug(
            "LLM answer generated (model=%s, prompt_len=%d)",
            self._model,
            len(normalized_prompt),
        )
        return answer
