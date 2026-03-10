"""
Test-only configuration for AI_module tests.
Use this config in test scenarios instead of AI_module.config so tests use Qdrant server
and test collection names without changing application defaults (e.g. STORAGE_TYPE is "server" here, "memory" in AI_module.config).
"""

from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# Vector DB: tests use Qdrant server (unlike default "memory" in AI_module.config)
# ---------------------------------------------------------------------------

STORAGE_TYPE: str = "server"

# Qdrant host/port (used by bootstrap and fixtures)
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333

# Collection names for integration tests (names contain "test" for cleanup)
VECTOR_COLLECTION_NAME_TEST: str = "test_pdf_knowledge_base"
INGESTION_TEST_COLLECTION: str = "test_ingestion_pipeline"

# ---------------------------------------------------------------------------
# Bootstrap (Docker / Qdrant start) – used by db_bootstrap when available
# ---------------------------------------------------------------------------

AUTO_START_DOCKER: bool = True
DOCKER_START_TIMEOUT: float = 120.0

# ---------------------------------------------------------------------------
# Test logging: logs written under tests/logs
# ---------------------------------------------------------------------------

TESTS_DIR: Path = Path(__file__).resolve().parent
LOG_DIR: Path = TESTS_DIR / "logs"
LOG_FILE: str | None = "tests.log"
LOG_LEVEL: str = "DEBUG"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
ENABLE_LOGGING: bool = False

_TEST_LOGGING_CONFIGURED = False


def configure_test_logging(
    log_dir: Path | None = None,
    log_file: str | None = LOG_FILE,
    level: str = LOG_LEVEL,
    fmt: str = LOG_FORMAT,
    enabled: bool | None = None,
) -> None:
    """
    Configure logging for tests: create tests/logs if needed and attach a file handler
    to the root logger. Idempotent. If enabled is False, no file handler is added
    and the logs directory is not created.
    """
    global _TEST_LOGGING_CONFIGURED
    if _TEST_LOGGING_CONFIGURED:
        return
    if not (enabled if enabled is not None else ENABLE_LOGGING):
        return
    directory = log_dir if log_dir is not None else LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.DEBUG)
    formatter = logging.Formatter(fmt)
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file:
        path = directory / log_file
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(log_level)
        root.addHandler(fh)
    _TEST_LOGGING_CONFIGURED = True
