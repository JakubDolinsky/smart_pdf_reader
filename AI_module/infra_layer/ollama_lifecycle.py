"""
Ollama server lifecycle: start and stop the Ollama server to avoid running it permanently.

- Production: call start_ollama() when the application starts and register_application_exit_handlers()
  so that stop_ollama() runs on normal exit, failure, or interrupt. Only starts Ollama if
  AUTO_START_OLLAMA_SERVER is True and the server is not already running; only stops the
  process that this module started (never kills an externally started Ollama).
- Tests: use the managed() context manager to start Ollama for the duration of test scenarios
  and stop it on exit.
"""

import atexit
import logging
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from shutil import which
from typing import Any

from AI_module.config import (
    AUTO_START_OLLAMA_SERVER,
    LLM_OLLAMA_HOST,
    OLLAMA_SERVE_READY_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Process we started (so we only stop our own, not a user's existing Ollama).
_our_process: subprocess.Popen[Any] | None = None


def _find_ollama_executable() -> str | None:
    """Return path to ollama executable, or None if not found. Prefer PATH; on Windows try default install dir."""
    path = which("ollama")
    if path:
        return path
    if os.name == "nt":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            for candidate in (
                Path(localappdata) / "Programs" / "Ollama" / "ollama.exe",
                Path(localappdata) / "Ollama" / "ollama.exe",
            ):
                if candidate.is_file():
                    return str(candidate)
    return None


def is_ollama_running(host: str | None = None) -> bool:
    """
    Return True if the Ollama server is reachable at the given host URL.

    Args:
        host: Base URL (e.g. http://localhost:11434). If None, uses LLM_OLLAMA_HOST from config.

    Returns:
        True if GET to host returns successfully.
    """
    url = (host or LLM_OLLAMA_HOST).rstrip("/") + "/"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except OSError:
        logger.debug("Ollama server not reachable at %s", url)
        return False


def start_ollama(
    host: str | None = None,
    *,
    auto_start: bool | None = None,
    ready_timeout: float | None = None,
) -> bool:
    """
    Ensure Ollama server is running: if already running, do nothing; otherwise start it
    when auto_start is True (default from AUTO_START_OLLAMA_SERVER).

    Only the process started by this function is tracked; stopping later will only
    terminate that process, not an externally started Ollama.

    Args:
        host: Base URL to check (e.g. http://localhost:11434). If None, uses LLM_OLLAMA_HOST.
        auto_start: If True, start the server when not running. If None, uses config AUTO_START_OLLAMA_SERVER.
        ready_timeout: Seconds to wait for server to become ready after starting. If None, uses OLLAMA_SERVE_READY_TIMEOUT.

    Returns:
        True if Ollama is running after this call (either was already running or we started it).
    """
    global _our_process
    base_url = host or LLM_OLLAMA_HOST
    do_start = auto_start if auto_start is not None else AUTO_START_OLLAMA_SERVER
    timeout = ready_timeout if ready_timeout is not None else OLLAMA_SERVE_READY_TIMEOUT

    if is_ollama_running(base_url):
        logger.debug("Ollama already running at %s", base_url)
        return True

    if not do_start:
        logger.debug("Ollama not running at %s; auto_start disabled", base_url)
        return False

    ollama_exe = _find_ollama_executable()
    if not ollama_exe:
        logger.warning(
            "Could not find Ollama executable (PATH or Windows %%LOCALAPPDATA%%\\Programs\\Ollama). "
            "Install Ollama or add it to PATH."
        )
        return False
    logger.info("Starting Ollama server (host=%s)", base_url)
    try:
        proc = subprocess.Popen(
            [ollama_exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        logger.warning("Could not start Ollama at %s", ollama_exe)
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if is_ollama_running(base_url):
            _our_process = proc
            logger.info("Ollama server started (PID=%s)", proc.pid)
            return True
        if proc.poll() is not None:
            logger.warning("Ollama process exited early with code %s", proc.returncode)
            return False

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    logger.warning("Ollama did not become ready within %.1fs", timeout)
    return False


def stop_ollama() -> None:
    """
    Stop the Ollama server process that was started by start_ollama() in this process.
    If Ollama was already running externally (or we never started it), this is a no-op.
    """
    global _our_process
    if _our_process is None:
        logger.debug("No Ollama process owned by this module; nothing to stop")
        return
    try:
        _our_process.terminate()
        try:
            _our_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _our_process.kill()
            _our_process.wait(timeout=5)
        logger.info("Ollama server stopped (was PID=%s)", _our_process.pid)
    except Exception as e:
        logger.warning("Error stopping Ollama process: %s", e)
    finally:
        _our_process = None


def register_application_exit_handlers() -> None:
    """
    Register atexit and signal handlers so that stop_ollama() is called when the
    application exits normally or on SIGINT/SIGTERM. Call this in production after
    start_ollama() so Ollama is stopped when the app is turned off or fails.
    """
    atexit.register(stop_ollama)

    def _signal_handler(signum: int, frame: Any) -> None:
        stop_ollama()
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except (AttributeError, ValueError):
        pass  # Windows may not have SIGINT in same way
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (AttributeError, ValueError):
        pass


class managed:
    """
    Context manager for test runs: start Ollama on enter and stop it on exit
    (only if we started it). Use in test .py files for scenarios that need a real Ollama.

    Example:
        with ollama_lifecycle.managed():
            # run tests that require Ollama
            client = LlmClient()
            assert client.answer("Hello?")
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        ready_timeout: float | None = None,
    ) -> None:
        self._host = host or LLM_OLLAMA_HOST
        self._ready_timeout = ready_timeout

    def __enter__(self) -> None:
        start_ollama(
            self._host,
            auto_start=True,
            ready_timeout=self._ready_timeout,
        )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        stop_ollama()
        return None
