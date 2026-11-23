"""With CI set, a recipe test may not skip for a missing package.

The recipe tests skip on a laptop without the extras (DuckDB, the notebook kernel),
which is right there and wrong in CI: a skipped recipe test is a green line that
tested nothing. CI installs `.[recipes,recipes-test]`; if that ever stops, this
turns each such skip into a failure that names the missing package. Skips a test
asks for by design (AVALOKA_HEAVY_TESTS) do not match and stay skips.
"""

from __future__ import annotations

import os
import re

import pytest

#: Skip reasons that mean "a package is missing": the install hints of
#: avaloka/recipes (engine.INSTALL_HINT, notebook_exec.INSTALL_HINT), pytest.importorskip,
#: and the parquet-engine probe in test_engine_and_run.
MISSING_PACKAGE = re.compile(
    r"\bpip install\b|\bnot installed\b|\bNo module named\b|\bcould not import\b|\bno parquet engine\b",
    re.I)


def _ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false")


def _skip_reason(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr or "")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not (_ci() and report.skipped and not hasattr(report, "wasxfail")):
        return
    reason = _skip_reason(report)
    if MISSING_PACKAGE.search(reason):
        report.outcome = "failed"
        report.longrepr = (f"AssertionError: CI is set, so a recipe test may not skip for a missing "
                           f"package; install .[recipes,recipes-test]. Skip reason: {reason}")


def _child_kernels() -> dict:
    import psutil
    found = {}
    for proc in psutil.Process().children(recursive=True):
        try:
            if "ipykernel_launcher" in " ".join(proc.cmdline()):
                found[proc.pid] = proc
        except psutil.Error:
            continue
    return found


@pytest.fixture(autouse=True)
def _stop_kernels_the_test_started(request):
    """Teardown safety net for `notebook` tests: stop any kernel this test started and
    left running, by its exact PID, so one leak cannot pile up across the suite.
    test_notebook_kernel_lifecycle.py asserts that execute() itself leaves none."""
    if request.node.get_closest_marker("notebook") is None:
        yield
        return
    before = set(_child_kernels())
    yield
    for pid, proc in _child_kernels().items():
        if pid not in before:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
