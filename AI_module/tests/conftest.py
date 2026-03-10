"""
Pytest configuration for AI_module tests.
Configures test logging and logs pass/fail (and failure reason) for each test.
"""

import logging

import pytest

from AI_module.tests.config import configure_test_logging

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Configure test logging at session start (creates tests/logs when ENABLE_LOGGING is True)."""
    configure_test_logging()


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
