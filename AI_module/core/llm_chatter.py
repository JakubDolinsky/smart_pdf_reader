"""
LLM chatter: build RAG prompt from reranking chunks and user query, then call LLM client for answer.
Sources from chunks (e.g. RerankingService.rerank output); prompt template from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from AI_module.config import LM_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from AI_module.infra_layer.llm_client import LlmClient

# Chunk-like: has .payload (dict)
ChunkLike = Any

# Returned by chat() when query/chunks/metadata are incomplete; LLM is not called.
PROMPT_INCOMPLETE_RESPONSE: str = (
    "The prompt is not complete and the LLM was not called. "
    "Please provide a non-empty question and at least one source chunk with complete metadata (text, source PDF name, chapter, and page)."
)


def _format_source(metadata: dict[str, Any]) -> str:
    """Format a single source block for the prompt (source PDF, Chapter, Page and text)."""
    text = (metadata.get("text") or "").strip()
    source_name = metadata.get("source") or ""
    chapter = metadata.get("chapter") or ""
    page = metadata.get("page") or metadata.get("page_start") or ""
    if not text:
        return ""
    parts = []
    if source_name:
        parts.append(f"Source: {source_name}")
    if chapter or page:
        sub = []
        if chapter:
            sub.append(f"Chapter: {chapter}")
        if page:
            sub.append(f"Page: {page}")
        parts.append(", ".join(sub))
    citation = "; ".join(parts).strip()
    if citation:
        return f"[{citation}]\n{text}"
    return text


def _chunks_list(chunks: list[ChunkLike] | dict[str, Any]) -> list[ChunkLike]:
    """Normalize to list of chunk-like objects (with .payload)."""
    if isinstance(chunks, dict):
        return chunks.get("chunks") or []
    if isinstance(chunks, list):
        return chunks
    return []


def _is_non_empty(value: Any) -> bool:
    """True if value is not None and has non-empty string representation after strip."""
    if value is None:
        return False
    return len(str(value).strip()) > 0


def _chunk_has_complete_metadata(payload: dict[str, Any]) -> bool:
    """
    True if payload has non-empty text, source (PDF name), chapter, and page (or page_start).
    Required for reliable citations and LLM response.
    """
    if not isinstance(payload, dict):
        return False
    text = payload.get("text")
    if not _is_non_empty(text):
        return False
    source = payload.get("source")
    if not _is_non_empty(source):
        return False
    chapter = payload.get("chapter")
    if not _is_non_empty(chapter):
        return False
    page = payload.get("page") if payload.get("page") is not None else payload.get("page_start")
    if not _is_non_empty(page):
        return False
    return True


def _filter_chunks_with_complete_metadata(
    chunks: list[ChunkLike] | dict[str, Any],
) -> list[ChunkLike]:
    """
    Return only chunks whose payload has complete metadata (text, source, chapter, page).
    Incomplete chunks are excluded so the LLM can be called with at least one reliable source.
    """
    chunk_list = _chunks_list(chunks)
    result = []
    for c in chunk_list:
        payload = getattr(c, "payload", None) if not isinstance(c, dict) else c
        if _chunk_has_complete_metadata(payload):
            result.append(c)
    return result


def _get_llm_client() -> "LlmClient":
    """Lazy import to avoid loading ollama at module load."""
    from AI_module.infra_layer.llm_client import LlmClient
    return LlmClient()


class LLMChatter:
    """
    Build RAG prompts from reranking chunks and user query; optionally call LLM client for answer.
    Use in the application layer; inject client for tests.
    """

    def __init__(self, *, client: "LlmClient | None" = None) -> None:
        """
        Args:
            client: Optional LlmClient; if None, a default instance is used for chat().
        """
        self._client = client if client is not None else None

    def _get_client(self) -> "LlmClient":
        if self._client is not None:
            return self._client
        return _get_llm_client()

    def build_context(self, chunks: list[ChunkLike] | dict[str, Any]) -> str:
        """
        Build the Sources section from reranking chunks.

        Args:
            chunks: List of Chunk (or chunk-like with .payload), or dict with "chunks" key.

        Returns:
            Formatted context string: one block per chunk, with optional [Source, Chapter, Page] citation.
        """
        chunk_list = _chunks_list(chunks)
        if not chunk_list:
            return ""
        parts = []
        for c in chunk_list:
            payload = getattr(c, "payload", None) if not isinstance(c, dict) else c
            if not isinstance(payload, dict):
                continue
            block = _format_source(payload)
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    def create_prompt(
        self,
        chunks: list[ChunkLike] | dict[str, Any],
        query: str,
        *,
        template: str | None = None,
    ) -> str:
        """
        Create the full LLM prompt from reranking chunks and the user's question.

        Args:
            chunks: List of Chunk (or dict with "chunks" key) from RerankingService.rerank.
            query: The user's question.
            template: Optional override for the prompt template (default: LM_PROMPT_TEMPLATE).

        Returns:
            Full prompt string with Sources and Question filled in.
        """
        context = self.build_context(chunks)
        query_clean = (query or "").strip()
        tpl = template if template is not None else LM_PROMPT_TEMPLATE
        return tpl.format(context=context, query=query_clean)

    def chat(
        self,
        chunks: list[ChunkLike] | dict[str, Any],
        query: str,
        *,
        template: str | None = None,
    ) -> str:
        """
        Build prompt from chunks and query, send it to the LLM client, return the answer.
        Chunks with incomplete metadata (missing or empty text, source, chapter, or page) are excluded.
        If query is empty/None or no chunk with complete metadata remains, the LLM is not called
        and PROMPT_INCOMPLETE_RESPONSE is returned.

        Args:
            chunks: List of Chunk (or dict with "chunks" key) from RerankingService.rerank.
            query: The user's question.
            template: Optional override for the prompt template (default: LM_PROMPT_TEMPLATE).

        Returns:
            The LLM's answer string, or PROMPT_INCOMPLETE_RESPONSE if query is empty or no chunk has complete metadata.
        """
        if not _is_non_empty(query):
            return PROMPT_INCOMPLETE_RESPONSE
        filtered = _filter_chunks_with_complete_metadata(chunks)
        if not filtered:
            return PROMPT_INCOMPLETE_RESPONSE
        prompt = self.create_prompt(filtered, query, template=template)
        return self._get_client().answer(prompt)
