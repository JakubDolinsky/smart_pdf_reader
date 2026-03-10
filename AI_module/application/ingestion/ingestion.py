"""
Ingestion pipeline: chunk PDFs from config PDF_INPUT_DIR, embed with core.embedding,
and upsert into the vector DB. Before every insert the collection is cleared (delete_all then insert).
Runnable directly: python -m AI_module.application.ingestion.ingestion
Or: AI_module/dev_tools/run_ingestion.bat
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from AI_module.config import (
    PDF_INPUT_DIR,
    STORAGE_TYPE,
    VECTOR_COLLECTION_NAME,
)
from AI_module.core.chunking import chunk_directory
from AI_module.core.db_manager import DBManager
from AI_module.core.embedding import EmbeddingService
from AI_module.infra_layer.db_health import check_db_ready

logger = logging.getLogger(__name__)


def run_ingestion(
    pdf_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> int:
    """
    Run full ingestion: chunk PDFs, embed chunks, clear DB collection, insert all chunks.

    Args:
        pdf_dir: Directory to scan for *.pdf files. If None, uses config PDF_INPUT_DIR.
        collection_name: Target collection name. If None, uses config VECTOR_COLLECTION_NAME.

    Returns:
        0 on success, 1 if no PDFs found or ingestion failed.
    """
    directory = Path(pdf_dir) if pdf_dir else PDF_INPUT_DIR
    if directory is None or not directory.is_dir():
        logger.error("Ingestion: PDF directory not set or not found. Set PDF_INPUT_DIR in AI_module.config or pass pdf_dir.")
        return 1

    coll_name = collection_name or VECTOR_COLLECTION_NAME

    if (STORAGE_TYPE or "").strip().lower() == "server":
        if not check_db_ready():
            logger.error(
                "Ingestion: Qdrant server is not reachable. Start it with AI_module/dev_tools/start_app_db.bat (or ensure STORAGE_TYPE/server is configured)."
            )
            return 1

    logger.info("Ingestion: chunking PDFs from %s", directory)
    chunks = chunk_directory(pdf_dir=directory, skip_empty=True)
    if not chunks:
        logger.warning("Ingestion: no chunks produced (no PDFs or all empty).")
        return 1

    logger.info("Ingestion: embedding %d chunks", len(chunks))
    embedder = EmbeddingService()
    embedded = embedder.embed_chunks(chunks)

    logger.info("Ingestion: clearing collection %r and inserting %d chunks", coll_name, len(embedded))
    db = DBManager(collection_name=coll_name)
    db.delete_all()
    db.insert_chunks(embedded)

    logger.info("Ingestion: done. Inserted %d chunks into %r.", len(embedded), coll_name)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    code = run_ingestion()
    return code


if __name__ == "__main__":
    sys.exit(main())
