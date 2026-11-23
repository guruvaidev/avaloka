"""The server URLs the service-backed 1.6 release tests need.

A skip reads as a pass in a summary line. On a laptop that is fine: a developer
without a Postgres or MySQL server gets a skip naming the variable. On the run
that certifies the release (bitbucket-pipelines.yml release-1-6-integration,
where Bitbucket sets CI=true) a skip would mean the service path was never
exercised, so there it is a failure. Same rule as tests/test_lineage.py and
tests/telemetry_ingest/test_postgres.py.
"""

from __future__ import annotations

import os

import pytest


def ci_is_set() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false")


def require_url(var: str, what: str) -> str:
    """The value of ``var``; without it, skip on a laptop and FAIL when CI is set."""
    value = os.environ.get(var, "").strip()
    if value:
        return value
    message = f"{var} is not set ({what}); see bitbucket-pipelines.yml release-1-6-integration"
    if ci_is_set():
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)
