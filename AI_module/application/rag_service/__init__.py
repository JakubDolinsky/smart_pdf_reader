"""
RAG service: single-question answer and CLI loop.
Call get_answer(question) from FastAPI or run_cli_loop() from main.py.
"""

from .rag_service import get_answer, run_cli_loop

__all__ = ["get_answer", "run_cli_loop"]
