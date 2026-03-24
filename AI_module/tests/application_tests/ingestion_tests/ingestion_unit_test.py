"""
Unit tests for ingestion.run_ingestion. Chunking, embedding, and DB are mocked.
For integration tests with real DB and embedding see ingestion_integration_test.py.
Run: python -m pytest AI_module/tests/application_tests/ingestion_tests/ingestion_unit_test.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_file = Path(__file__).resolve()
_root = _file.parents[3]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.core.chunk import Chunk
from AI_module.application.ingestion.ingestion import run_ingestion
from AI_module.tests.config import STORAGE_TYPE


def _make_chunk(chunk_id: str, text: str, vector: list[float] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        payload={"text": text, "source": "test", "page": 1, "chunk_index": 0, "path": ""},
        vector=vector,
    )


def test_run_ingestion_missing_pdf_dir_returns_1():
    """When pdf_dir is None and PDF_INPUT_DIR is empty, run_ingestion returns 1."""
    with patch("AI_module.application.ingestion.ingestion.PDF_INPUT_DIR", None):
        assert run_ingestion(pdf_dir=None) == 1


def test_run_ingestion_nonexistent_dir_returns_1():
    """When pdf_dir does not exist or is not a directory, run_ingestion returns 1."""
    assert run_ingestion(pdf_dir=Path("/nonexistent_pdf_dir_xyz")) == 1


def test_run_ingestion_empty_dir_returns_1():
    """When directory exists but chunk_directory returns no chunks, run_ingestion returns 1."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=[]):
            assert run_ingestion(pdf_dir=tmp) == 1


def test_run_ingestion_server_mode_db_not_ready_returns_1():
    """When STORAGE_TYPE is server and check_db_ready is False, run_ingestion returns 1."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=[
            _make_chunk("c1", "hello"),
        ]):
            with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", STORAGE_TYPE):
                with patch("AI_module.application.ingestion.ingestion.check_db_ready", return_value=False):
                    assert run_ingestion(pdf_dir=tmp) == 1


def test_run_ingestion_calls_delete_all_then_insert_chunks():
    """run_ingestion calls db.delete_all() then db.insert_chunks with correct args."""
    import tempfile
    chunks = [
        _make_chunk("id1", "text one"),
        _make_chunk("id2", "text two"),
    ]
    embedded = [
        _make_chunk("id1", "text one", [0.1] * 384),
        _make_chunk("id2", "text two", [0.2] * 384),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    mock_db = MagicMock()
                    with patch("AI_module.application.ingestion.ingestion.DBManager", return_value=mock_db):
                        result = run_ingestion(pdf_dir=tmp, collection_name="test_coll")
    assert result == 0
    assert mock_db.delete_all.call_count == 1
    assert mock_db.insert_chunks.call_count == 1
    call_args = mock_db.insert_chunks.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].id == "id1" and call_args[0].payload["text"] == "text one"
    assert call_args[1].id == "id2" and call_args[1].payload["text"] == "text two"


def test_run_ingestion_delete_all_called_before_insert():
    """delete_all is called before insert_chunks (order of calls)."""
    import tempfile
    chunks = [_make_chunk("x", "y")]
    embedded = [_make_chunk("x", "y", [0.0] * 384)]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    mock_db = MagicMock()
                    with patch("AI_module.application.ingestion.ingestion.DBManager", return_value=mock_db):
                        run_ingestion(pdf_dir=tmp)
    call_order = [c[0] for c in mock_db.method_calls]
    assert call_order == ["delete_all", "insert_chunks"]


def test_run_ingestion_uses_collection_name_from_config_when_not_passed():
    """When collection_name is not passed, DBManager is called with VECTOR_COLLECTION_NAME."""
    import tempfile
    chunks = [_make_chunk("a", "b")]
    embedded = [_make_chunk("a", "b", [0.0] * 384)]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    with patch("AI_module.application.ingestion.ingestion.DBManager") as mock_cls:
                        run_ingestion(pdf_dir=tmp)
    mock_cls.assert_called_once()
    assert mock_cls.call_args[1]["collection_name"] == "pdf_knowledge_base"


def test_run_ingestion_uses_passed_collection_name():
    """When collection_name is passed, DBManager is called with that name."""
    import tempfile
    chunks = [_make_chunk("a", "b")]
    embedded = [_make_chunk("a", "b", [0.0] * 384)]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    with patch("AI_module.application.ingestion.ingestion.DBManager") as mock_cls:
                        run_ingestion(pdf_dir=tmp, collection_name="my_collection")
    mock_cls.assert_called_once_with(collection_name="my_collection")


def test_run_ingestion_returns_0_on_success():
    """run_ingestion returns 0 when chunking and insert succeed."""
    import tempfile
    chunks = [_make_chunk("ok", "content")]
    embedded = [_make_chunk("ok", "content", [0.0] * 384)]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    with patch("AI_module.application.ingestion.ingestion.DBManager"):
                        assert run_ingestion(pdf_dir=tmp) == 0


def test_run_ingestion_embed_chunks_receives_chunks_from_chunk_directory():
    """EmbeddingService().embed_chunks is called with the list returned by chunk_directory."""
    import tempfile
    chunks = [_make_chunk("c1", "t1"), _make_chunk("c2", "t2")]
    embedded = [
        _make_chunk("c1", "t1", [0.1] * 384),
        _make_chunk("c2", "t2", [0.2] * 384),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with patch("AI_module.application.ingestion.ingestion.STORAGE_TYPE", "memory"):
            with patch("AI_module.application.ingestion.ingestion.chunk_directory", return_value=chunks):
                with patch("AI_module.application.ingestion.ingestion.EmbeddingService") as MockEmbeddingService:
                    MockEmbeddingService.return_value.embed_chunks.return_value = embedded
                    with patch("AI_module.application.ingestion.ingestion.DBManager"):
                        run_ingestion(pdf_dir=tmp)
    mock_embed_chunks = MockEmbeddingService.return_value.embed_chunks
    mock_embed_chunks.assert_called_once()
    assert mock_embed_chunks.call_args[0][0] == chunks


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
