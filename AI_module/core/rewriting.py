"""
Query rewriting for retrieval: expand referential questions using prior Q&A, via RewriterClient (phi mini).
"""

from __future__ import annotations

import functools
import logging
import re

from AI_module import config
from AI_module.config import REWRITE_PROMPT_TEMPLATE
from AI_module.infra_layer.rewriter_client import RewriterClient

logger = logging.getLogger(__name__)

# Re-export for "from AI_module.core.rewriting import REFERENCE_WORDS"
REFERENCE_WORDS = config.REFERENCE_WORDS


@functools.lru_cache(maxsize=16)
def _reference_words_pattern(words_key: tuple[str, ...]) -> re.Pattern[str]:
    """
    Single regex: any REFERENCE_WORDS phrase as a standalone token (not inside a longer word).
    Uses (?<!\\w) / (?!\\w) so short items like "he" do not match inside "hexadecimal".
    Longest phrases first so alternation prefers multi-word matches where relevant.
    """
    phrases = sorted({p.strip() for p in words_key if p and str(p).strip()}, key=len, reverse=True)
    if not phrases:
        return re.compile(r"(?!x)x")  # never matches
    escaped = "|".join(re.escape(p) for p in phrases)
    # \w = Unicode letters/digits/underscore; must not touch word chars on either side of the phrase.
    return re.compile(rf"(?<!\w)(?:{escaped})(?!\w)", re.IGNORECASE)


def question_contains_reference_word(query: str) -> bool:
    """True if query contains any of REFERENCE_WORDS as a whole word/phrase (case-insensitive)."""
    if not query or not query.strip():
        return False
    text = query.strip()
    pattern = _reference_words_pattern(tuple(config.REFERENCE_WORDS))
    return pattern.search(text) is not None


def _extract_previous_q_and_a(
    embed_history: list[dict[str, str]],
) -> tuple[str, str]:
    """
    embed_history: most recent message first (reversed slice from orchestration).
    Returns (prev_user_question, prev_assistant_answer) from the last complete turn before current.
    """
    prev_question = ""
    prev_answer = ""
    for msg in embed_history:
        role = (msg.get("role") or "").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and not prev_answer:
            prev_answer = content
        elif role == "user" and not prev_question:
            prev_question = content
        if prev_question and prev_answer:
            break
    return prev_question, prev_answer


def rewrite_question_for_embedding(
    current_question: str,
    embed_history: list[dict[str, str]] | None,
    *,
    rewriter_client: RewriterClient | None = None,
) -> str:
    """
    If the question contains reference words and non-empty history yields a prior Q&A pair,
    call the rewriter model and return its output; otherwise return the stripped original question.
    """
    q = (current_question or "").strip()
    if not q:
        return q
    if not embed_history:
        return q
    if not question_contains_reference_word(q):
        return q

    prev_question, prev_answer = _extract_previous_q_and_a(embed_history)
    if not prev_question or not prev_answer:
        logger.debug(
            "Rewriting skipped: missing previous Q or A (have_q=%s have_a=%s)",
            bool(prev_question),
            bool(prev_answer),
        )
        return q

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        prev_question=prev_question,
        prev_answer=prev_answer,
        current_question=q,
    )

    client = rewriter_client if rewriter_client is not None else RewriterClient()
    try:
        rewritten = client.rewrite(prompt).strip()
    except Exception:
        logger.exception("Query rewriter call failed; using original question for embedding.")
        return q

    if not rewritten:
        logger.debug("Rewriter returned empty; using original question for embedding.")
        return q
    return rewritten
