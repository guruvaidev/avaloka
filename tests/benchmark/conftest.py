"""
pytest configuration + fixtures for the Avaloka benchmark.

Markers
-------
    kaggle   : test hits real Kaggle (needs AVALOKA_BENCH_KAGGLE=1 + creds)
    cloud    : test hits a real cloud object store / DB
    heavy    : real deep-learning training (needs AVALOKA_BENCH_HEAVY=1 + torch)
    swarm    : multi-avaloka coordination test
    slow     : long-running

Heavy/cloud/kaggle tests auto-skip unless their env switch is set, mirroring the
``SkipTest`` gating in tests/test_e2e_gke_groupby_sum.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `app.agents...` resolves like the other tests.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.benchmark.adapters import get_client  # noqa: E402


def pytest_configure(config):
    for marker in (
        "kaggle: hits real Kaggle (AVALOKA_BENCH_KAGGLE=1 + creds)",
        "cloud: hits a real cloud object store / DB",
        "heavy: real deep-learning training (AVALOKA_BENCH_HEAVY=1 + torch)",
        "swarm: multi-avaloka coordination test",
        "slow: long-running test",
    ):
        config.addinivalue_line("markers", marker)


def _heavy_enabled() -> bool:
    return os.environ.get("AVALOKA_BENCH_HEAVY", "0") == "1"


def pytest_collection_modifyitems(config, items):
    skip_heavy = pytest.mark.skip(reason="set AVALOKA_BENCH_HEAVY=1 (and install torch) to run")
    for item in items:
        if "heavy" in item.keywords and not _heavy_enabled():
            item.add_marker(skip_heavy)


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("avaloka_bench")


@pytest.fixture(scope="session")
def client():
    """The Avaloka client under test (reference, or the real CLI via env)."""
    return get_client()


@pytest.fixture(scope="session")
def heavy_enabled() -> bool:
    return _heavy_enabled()


#: Source kinds whose local artifact is a .sqlite file. ``postgres`` falls back
#: to the sqlite fake unless AVALOKA_BENCH_PG_DSN is set (see sources.py).
SQLITE_BACKED_KINDS = frozenset({"sqlite", "postgres"})


@pytest.fixture
def sqlite_reader_for(monkeypatch):
    """Opt one test into the sampler's local SQLite reader, for sqlite-backed kinds only.

    The reader is off by default (``ENABLE_SQLITE=0``, sampling_agent.py) because
    it opens arbitrary local files; production routes SQL through the Postgres
    MCP. These benchmark fakes are .sqlite files, so the test that samples them
    turns the reader on for itself and the patch is undone afterwards. Never set
    ENABLE_SQLITE for the CI step or the session: tests/test_sqlite_reader_off_by_default.py
    checks both the default and the session.
    """
    def enable(*kinds: str) -> bool:
        if not SQLITE_BACKED_KINDS.intersection(kinds):
            return False
        import app.agents.sampling_agent as sampling_agent

        monkeypatch.setattr(sampling_agent, "ENABLE_SQLITE", True)
        return True

    return enable
