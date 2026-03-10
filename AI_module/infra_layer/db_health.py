"""
DB health check for the application: check if Qdrant is running and responding.
Does not start Docker or Qdrant. For starting the DB, use dev_tools/start_app_db.bat.
"""

from __future__ import annotations

import logging
import urllib.request

from AI_module.config import QDRANT_LOCAL_HOST, QDRANT_LOCAL_PORT

logger = logging.getLogger(__name__)


def is_qdrant_server_running(
    host: str | None = None,
    port: int | None = None,
) -> bool:
    """Return True if a Qdrant server is reachable at host:port. On Windows, tries 127.0.0.1 when host is localhost and direct check fails (IPv6/localhost issue)."""
    h = host if host is not None else (QDRANT_LOCAL_HOST or "localhost")
    p = port if port is not None else QDRANT_LOCAL_PORT
    if not h or not str(h).strip():
        h = "localhost"
    for try_host in (h, "127.0.0.1"):
        if try_host != h and h not in ("localhost", "::1", "::"):
            break
        url = f"http://{try_host}:{p}/healthz"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"Qdrant server is running at {try_host}:{p}")
                    logger.debug("Qdrant server is running at %s:%s", try_host, p)
                    return True
        except OSError:
            print(f"Qdrant server is not reachable at {try_host}:{p}")
            logger.debug("Qdrant server not reachable at %s:%s", try_host, p)
    return False


def check_db_ready(
    host: str | None = None,
    port: int | None = None,
) -> bool:
    """Return True if Qdrant DB is running and responding (alias for is_qdrant_server_running)."""
    return is_qdrant_server_running(host=host, port=port)
