"""
Pytest configuration for AI_module tests.
Configures test logging and logs pass/fail (and failure reason) for each test.
"""

import logging
import sys
import warnings

import pytest

from AI_module.tests.config import configure_test_logging

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Configure test logging and console warning/error output at session start."""
    configure_test_logging()
    # Print all Python warnings during tests and route them through logging.
    warnings.simplefilter("always")
    logging.captureWarnings(True)

    # Always print WARNING+ logs (including captured warnings) to console.
    root = logging.getLogger()
    has_console_warning_handler = any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) in (sys.stderr, sys.stdout)
        and h.level <= logging.WARNING
        for h in root.handlers
    )
    if not has_console_warning_handler:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        root.addHandler(ch)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Log test passed/failed/skipped and, on failure, the reason."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    name = item.nodeid
    if report.passed:
        logger.info("PASSED: %s", name)
    elif report.failed:
        reason = getattr(report, "longrepr", None)
        if reason is not None:
            reason_str = str(reason).split("\n")[0] if reason else "unknown"
        else:
            reason_str = "unknown"
        logger.error("FAILED: %s — %s", name, reason_str)
    elif report.skipped:
        logger.info("SKIPPED: %s", name)
