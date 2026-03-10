"""
Bootstrap for tests that need the vector DB: start Docker and Qdrant for db_client_integration_test
or other tests that require a running Qdrant server. Not used by the application.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

from AI_module.tests.config import (
    AUTO_START_DOCKER,
    DOCKER_START_TIMEOUT,
    QDRANT_HOST,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)


def is_qdrant_server_running(
    host: str | None = None,
    port: int | None = None,
) -> bool:
    """Return True if a Qdrant server is reachable at host:port. On Windows, tries 127.0.0.1 when host is localhost and direct check fails (IPv6/localhost issue)."""
    h = host if host is not None else (QDRANT_HOST or "localhost")
    p = port if port is not None else QDRANT_PORT
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
                    return True
        except OSError:
            logger.debug("Qdrant server not reachable at %s:%s", try_host, p)
    return False


def docker_available() -> bool:
    """Return True if Docker daemon is running and responsive."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_docker_running() -> bool:
    """If Docker is not available, try to start it. Return True if Docker is responsive afterward."""
    if docker_available():
        return True
    if not AUTO_START_DOCKER:
        logger.debug("Docker not running; AUTO_START_DOCKER is False")
        return False
    logger.info("Docker not running; attempting to start")
    system = platform.system()
    if system == "Windows":
        try:
            subprocess.Popen(
                ["docker", "desktop", "start"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except FileNotFoundError:
            for exe in [
                Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
                Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Docker" / "Docker" / "Docker Desktop.exe",
            ]:
                if exe.exists():
                    subprocess.Popen(
                        [str(exe)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                    )
                    break
    elif system == "Darwin":
        try:
            subprocess.Popen(
                ["open", "-a", "Docker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass
    elif system == "Linux":
        for cmd in [["systemctl", "start", "docker"], ["service", "docker", "start"]]:
            try:
                subprocess.run(cmd, capture_output=True, timeout=15, check=False)
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    deadline = time.monotonic() + DOCKER_START_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(3)
        if docker_available():
            logger.info("Docker is now running")
            return True
    logger.warning("Docker did not become ready within %.0f s", DOCKER_START_TIMEOUT)
    return False


def _qdrant_run_exists() -> bool:
    try:
        r = subprocess.run(
            ["qdrant", "run", "--help"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_qdrant_server(
    host: str | None = None,
    port: int | None = None,
) -> bool:
    """Try to start Qdrant (existing container, new container, then qdrant binary). Return True if running after attempts."""
    h = (host or QDRANT_HOST or "").strip() or "localhost"
    p = port if port is not None else QDRANT_PORT
    logger.info("Qdrant server not running at %s:%s; attempting to start", h, p)

    if not docker_available() and AUTO_START_DOCKER:
        ensure_docker_running()

    try:
        subprocess.run(
            ["docker", "start", "qdrant_local"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        time.sleep(2)
        if is_qdrant_server_running(h, p):
            logger.info("Started existing Qdrant Docker container qdrant_local")
            return True
    except FileNotFoundError:
        pass

    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{p}:6333",
                "--name", "qdrant_local",
                "qdrant/qdrant",
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        time.sleep(3)
        if is_qdrant_server_running(h, p):
            logger.info("Started new Qdrant Docker container qdrant_local")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        subprocess.Popen(
            ["qdrant", "run"] if _qdrant_run_exists() else ["qdrant"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(3)
        if is_qdrant_server_running(h, p):
            logger.info("Started Qdrant server via qdrant binary")
            return True
    except FileNotFoundError:
        pass

    logger.warning("Could not start Qdrant server at %s:%s", h, p)
    return False


def ensure_qdrant_server_running(
    host: str | None = None,
    port: int | None = None,
    wait_seconds: float = 16.0,
) -> bool:
    """If Qdrant is not reachable, try to start it; poll until ready or wait_seconds."""
    h = (host or QDRANT_HOST or "").strip() or "localhost"
    p = port if port is not None else QDRANT_PORT
    if is_qdrant_server_running(h, p):
        return True
    if start_qdrant_server(h, p):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(2)
            if is_qdrant_server_running(h, p):
                return True
    return False


def ensure_db_ready(
    host: str | None = None,
    port: int | None = None,
    wait_qdrant_seconds: float = 16.0,
) -> bool:
    """Ensure Docker and Qdrant are up for tests. Returns True if Qdrant is reachable afterward."""
    ensure_docker_running()
    return ensure_qdrant_server_running(host=host, port=port, wait_seconds=wait_qdrant_seconds)


def get_resolved_host_port(
    host: str | None = None,
    port: int | None = None,
) -> tuple[str | None, int | None]:
    """Return (host, port) that work for connecting to Qdrant (tries localhost then 127.0.0.1). (None, None) if not reachable."""
    h = (host or QDRANT_HOST or "").strip() or "localhost"
    p = port if port is not None else QDRANT_PORT
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(host=h, port=p)
        c.get_collections()
        return (h, p)
    except Exception:
        pass
    if h != "127.0.0.1":
        try:
            from qdrant_client import QdrantClient
            c = QdrantClient(host="127.0.0.1", port=p)
            c.get_collections()
            return ("127.0.0.1", p)
        except Exception:
            pass
    return (None, None)


def stop_qdrant_server(container_name: str = "qdrant_local") -> bool:
    """Stop the Qdrant Docker container (same as stop_app_db.bat). Returns True if Docker ran stop without error. Does not stop Docker Desktop."""
    try:
        r = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if r.returncode == 0:
            logger.info("Stopped Qdrant container %s", container_name)
            return True
        logger.debug("docker stop %s returned %s: %s", container_name, r.returncode, r.stderr.decode(errors="replace").strip() or r.stdout.decode(errors="replace").strip())
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("Could not stop Qdrant container: %s", e)
        return False
