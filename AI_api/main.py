"""
FastAPI transfer layer between RAG service and .NET REST API.
Exposes POST /ask (question -> answer) and GET /health.
Run from repo root: uvicorn AI_api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# Ensure repo root is on path so AI_module can be imported (e.g. when running from any cwd).
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
if _repo_root not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from AI_module.application.rag_service import get_answer

app = FastAPI(
    title="RAG API",
    description="Transfer layer for PDF RAG: ask a question, get an answer. Consumed by .NET REST API.",
    version="1.0.0",
)


class HistoryMessage(BaseModel):
    """One message in conversation history (role + content). Role must be 'user' or 'assistant'."""

    role: Literal["user", "assistant"] = Field(..., description="Must be 'user' or 'assistant'.")
    content: str = Field(..., description="Message text.")


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, description="Question about the PDF content.")
    history: list[HistoryMessage] | None = Field(
        default=None,
        description="Optional conversation history (last 2 messages used). Each: {role, content}.",
    )


class AskResponse(BaseModel):
    """Response for POST /ask."""

    answer: str = Field(..., description="RAG answer or message when no answer could be produced.")


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness check for .NET or load balancers."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Ask a question about the ingested PDFs. Uses the RAG pipeline (embed -> search -> rerank -> LLM).
    Optional history: last 2 messages (1 user + 1 assistant) are included for context and referential questions.
    Returns the answer or a message if the question is empty or no relevant chunks were found.
    """
    question = (request.question or "").strip()
    history_dicts: list[dict[str, str]] | None = None
    if request.history:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
    answer_text = get_answer(question, history=history_dicts)
    return AskResponse(answer=answer_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("AI_api.main:app", host="0.0.0.0", port=8000, reload=True)
