from __future__ import annotations

import logging

from AI_module.application.rag_service import get_answer
from AI_module.config import QDRANT_LOCAL_HOST, QDRANT_LOCAL_PORT
from AI_module.infra_layer import check_db_ready


def main() -> None:
    """Run the RAG CLI loop (prompt for questions, print answers until empty input)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    host, port = QDRANT_LOCAL_HOST, QDRANT_LOCAL_PORT

    if not check_db_ready(host=host, port=port):
        print(
            f"Qdrant DB is not reachable at {host}:{port}. "
            "Start the DB server first (e.g. dev_tools/start_app_db.bat)."
        )
        return

    conversation: list[dict[str, str]] = []
    prompt_text = "Ask a question about the ingested PDFs (empty to exit): "

    while True:
        try:
            question = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            print("Exiting.")
            break
        answer = get_answer(question, host=host, port=port, history=conversation)
        print("\nAnswer:\n")
        print(answer)
        print()
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
