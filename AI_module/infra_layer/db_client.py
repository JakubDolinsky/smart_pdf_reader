"""
Vector DB client for storing and querying document chunk embeddings.
Supports: insert full embedding set, delete all, and top-k similarity search.
Uses Qdrant as the backend. Can use a local server at http://localhost:6333.

The client does not start Docker or Qdrant. In the application, use db_health.check_db_ready()
to verify the DB is up; start the DB with AI_module/dev_tools/start_app_db.bat. In tests, the
db_client_integration_test fixture uses tests/db_bootstrap to start the server.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from AI_module.config import (
    QDRANT_LOCAL_HOST,
    QDRANT_LOCAL_PORT,
    QDRANT_PERSIST_DIRECTORY,
    SIMILARITY_METRIC,
    STORAGE_TYPE,
    TOP_K_SIMILAR_CHUNKS,
    VECTOR_COLLECTION_NAME,
    configure_logging,
)
from .db_health import is_qdrant_server_running as _check_server_running

logger = logging.getLogger(__name__)

# Qdrant distance names -> client enum
_QDRANT_DISTANCE = {
    "cosine": "Cosine",
    "l2": "Euclid",
    "ip": "Dot",
}


def _to_point_id(chunk_id: str) -> int:
    """Map string chunk id to a stable 63-bit positive int for Qdrant (collision-resistant)."""
    digest = hashlib.sha256(chunk_id.encode()).digest()
    n = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return n if n != 0 else 1


def _is_qdrant_server_running(host: str, port: int) -> bool:
    """Return True if a Qdrant server is reachable at host:port (tries 127.0.0.1 when host is localhost on Windows)."""
    return _check_server_running(host=host, port=port)


class VectorDBClient:
    """
    Client for a persistent vector database used in RAG (Qdrant backend).
    Operations: insert chunks, delete all, search top-k similar chunks.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = VECTOR_COLLECTION_NAME,
        similarity_metric: str = SIMILARITY_METRIC,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Args:
            persist_directory: Directory for DB persistence (local path). If set, uses path storage.
            collection_name: Name of the collection storing chunk embeddings.
            similarity_metric: One of "cosine", "l2", "ip".
            host: Qdrant server host. If set, uses server at host:port.
            port: Qdrant server port when using a server (default from config, 6333).

        The client does not start Docker or Qdrant. Use db_health.check_db_ready() to verify the DB
        is up; start the DB with dev_tools/start_app_db.bat before running the application.

        When both host and persist_directory are None, storage is taken from config STORAGE_TYPE
        ("server" | "memory" | "path"). For "server", QDRANT_LOCAL_HOST must be set; for "path",
        QDRANT_PERSIST_DIRECTORY must be set; otherwise an exception is raised.
        """
        if host is not None:
            self._host = host
            self._persist_directory = None
        elif persist_directory is not None:
            self._host = None
            self._persist_directory = str(persist_directory)
        else:
            mode = (STORAGE_TYPE or "memory").strip().lower().replace(" ", "")
            if mode == "inmemory":
                mode = "memory"
            if mode == "server":
                if not QDRANT_LOCAL_HOST or not str(QDRANT_LOCAL_HOST).strip():
                    raise ValueError(
                        "STORAGE_TYPE is 'server' but QDRANT_LOCAL_HOST is not set or empty. "
                        "Set QDRANT_LOCAL_HOST (and optionally QDRANT_LOCAL_PORT) in config, "
                        "or pass host= to VectorDBClient."
                    )
                self._host = str(QDRANT_LOCAL_HOST).strip()
                self._persist_directory = None
            elif mode == "path":
                if not QDRANT_PERSIST_DIRECTORY or not str(QDRANT_PERSIST_DIRECTORY).strip():
                    raise ValueError(
                        "STORAGE_TYPE is 'path' but QDRANT_PERSIST_DIRECTORY is not set or empty. "
                        "Set QDRANT_PERSIST_DIRECTORY in config, or pass persist_directory= to VectorDBClient."
                    )
                self._host = None
                self._persist_directory = str(QDRANT_PERSIST_DIRECTORY).strip()
            elif mode == "memory":
                self._host = None
                self._persist_directory = None
            else:
                raise ValueError(
                    f"Invalid STORAGE_TYPE '{STORAGE_TYPE}'. "
                    "Must be one of: 'server', 'memory', 'path' (or 'in memory')."
                )
        self._collection_name = collection_name
        self._similarity_metric = similarity_metric
        self._port = port if port is not None else QDRANT_LOCAL_PORT
        self._client: Any = None
        self._vector_size: int | None = None

    def _ensure_client(self) -> None:
        """Lazy-init Qdrant client. For server mode, only checks if server is reachable (does not start it)."""
        if self._client is not None:
            return
        configure_logging()
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "VectorDBClient requires qdrant-client. Install with: pip install qdrant-client"
            ) from e

        if self._host is not None:
            if not _is_qdrant_server_running(self._host, self._port):
                raise RuntimeError(
                    f"Qdrant server at http://{self._host}:{self._port} is not running. "
                    "Start the DB with dev_tools/start_app_db.bat and use db_health.check_db_ready() in your entrypoint before using VectorDBClient."
                )
            # On Windows, "localhost" can resolve to IPv6 and fail; use 127.0.0.1 when that works
            connect_host = self._host
            if (connect_host or "").strip().lower() in ("localhost", "::1", "::"):
                if _is_qdrant_server_running("127.0.0.1", self._port):
                    connect_host = "127.0.0.1"
            self._client = QdrantClient(host=connect_host, port=self._port)
            logger.info(
                "DB connection: connected to Qdrant server at http://%s:%s (remote server, data persisted on server)",
                self._host,
                self._port,
            )
        elif self._persist_directory:
            self._client = QdrantClient(path=self._persist_directory)
            logger.info(
                "DB connection: using Qdrant persistent storage at %s (local path, data persisted on disk)",
                self._persist_directory,
            )
        else:
            self._client = QdrantClient(":memory:")
            logger.info(
                "DB connection: using in-memory storage (no server, data not persisted; lost when process exits)",
            )

    def _distance_model(self) -> Any:
        """Resolve similarity_metric to Qdrant Distance enum."""
        from qdrant_client.models import Distance

        name = _QDRANT_DISTANCE.get(
            self._similarity_metric.lower(), self._similarity_metric
        )
        return getattr(Distance, name.upper(), Distance.COSINE)

    def _ensure_collection(self, vector_size: int) -> None:
        """Create collection if it does not exist (vector_size from first insert)."""
        if self._client.collection_exists(self._collection_name):
            info = self._client.get_collection(self._collection_name)
            existing_size = None
            try:
                vec_params = info.config.params.vectors
                existing_size = vec_params.size if hasattr(vec_params, "size") else None
            except AttributeError:
                pass
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection {self._collection_name!r} has vector size {existing_size}, "
                    f"but got embeddings of size {vector_size}. "
                    "Call delete_all() first to reset, or use a different collection_name."
                )
            return
        from qdrant_client.models import VectorParams

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self._distance_model(),
            ),
        )
        self._vector_size = vector_size

    def insert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Insert a full set of chunk embeddings (e.g. from an embedding procedure).
        Call delete_all first if you need to avoid duplicates when re-inserting.

        Args:
            ids: Unique identifiers for each chunk (e.g. chunk_0, doc_1_chunk_2).
            embeddings: Embedding vectors, one per chunk.
            metadatas: Optional metadata per chunk (e.g. {"text": "...", "source": "..."}).
        """
        if not ids or not embeddings:
            raise ValueError("ids and embeddings must be non-empty")
        self._ensure_client()
        if metadatas is None:
            metadatas = [{}] * len(ids)
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, and metadatas must have the same length")

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_to_point_id(cid),
                vector=emb,
                payload={"chunk_id": cid, **meta},
            )
            for cid, emb, meta in zip(ids, embeddings, metadatas)
        ]
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            "Inserted %d chunks into collection %r",
            len(ids),
            self._collection_name,
        )

    def delete_all(self) -> None:
        """Remove all embeddings from the collection efficiently."""
        self._ensure_client()
        if not self._client.collection_exists(self._collection_name):
            logger.debug("delete_all: collection %r does not exist", self._collection_name)
            return

        # Delete entire collection directly (much faster than scrolling)
        self._client.delete_collection(self._collection_name)
        self._vector_size = None
        logger.info("Deleted all embeddings from collection %r", self._collection_name)

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_SIMILAR_CHUNKS,
        *,
        include_metadatas: bool = True,
        include_distances: bool = True,
    ) -> dict[str, Any]:
        """
        Return the top-k chunks most similar to the query embedding.

        Args:
            query_embedding: Embedding vector of the query chunk.
            top_k: Number of similar chunks to return (default from config).
            include_metadatas: Whether to return chunk metadata (payload).
            include_distances: Whether to return similarity/distance scores.

        Returns:
            Dict with "ids" (chunk_id from payload), "metadatas" (if requested),
            and "distances" (if requested). For cosine/dot: higher score = more similar.
        """
        self._ensure_client()
        if not self._client.collection_exists(self._collection_name):
            out = {"ids": []}
            if include_metadatas:
                out["metadatas"] = []
            if include_distances:
                out["distances"] = []
            return out

        count = self._client.get_collection(self._collection_name).points_count
        limit = min(top_k, count) if count else 0
        if limit == 0:
            out = {"ids": []}
            if include_metadatas:
                out["metadatas"] = []
            if include_distances:
                out["distances"] = []
            return out

        # Use query_points (search() was removed in newer qdrant-client; use query_points)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=include_metadatas,
        )
        hits = response.points

        chunk_ids = []
        for h in hits:
            payload = h.payload or {}
            chunk_ids.append(payload.get("chunk_id", h.id))

        out: dict[str, Any] = {"ids": chunk_ids}
        if include_metadatas:
            out["metadatas"] = [h.payload or {} for h in hits]
        if include_distances:
            out["distances"] = [h.score for h in hits]
        logger.debug(
            "search_similar: returned %d results from collection %r",
            len(chunk_ids),
            self._collection_name,
        )
        return out
