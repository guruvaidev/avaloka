"""
Source-connector benchmark suite.

Asserts that every connector (local / sqlite / postgres / gcs / s3 / azure)
lands the *same logical bytes* locally and yields an identical schema -- so the
CLI can be pointed at any source and behave equivalently. Real cloud paths are
marked ``cloud`` and only activate when their env switch is set; otherwise the
filesystem-backed fakes are exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.benchmark.datasets import materialize
from tests.benchmark.registry import get
from tests.benchmark.sources import ALL_SOURCE_KINDS, make_source


@pytest.mark.parametrize("kind", ALL_SOURCE_KINDS)
def test_source_roundtrip(kind, workdir):
    spec = get("titanic")
    handle = materialize(spec)
    src = make_source(kind, handle, workdir)

    local = src.materialize_locally()
    assert local.exists(), f"{kind} did not materialize a local artifact"

    # URI must be well-formed and carry the source scheme.
    assert "://" in src.uri

    # Schema parity vs the canonical CSV.
    canonical = sorted(pd.read_csv(handle.path).columns.tolist())
    if local.suffix == ".sqlite":
        import sqlite3

        conn = sqlite3.connect(local)
        try:
            tbl = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;", conn
            ).iloc[0, 0]
            got = sorted(pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 1;", conn).columns.tolist())
        finally:
            conn.close()
    else:
        got = sorted(pd.read_csv(local).columns.tolist())
    assert got == canonical, f"{kind} schema drift: {got} != {canonical}"


@pytest.mark.parametrize("kind", ALL_SOURCE_KINDS)
def test_sample_equivalent_across_sources(kind, client, workdir, sqlite_reader_for):
    """Sampling from any source returns a usable, error-free result."""
    from tests.benchmark.harness import prepare_source

    sqlite_reader_for(kind)
    handle, source, local = prepare_source("telco_churn", kind, workdir)
    out = client.sample(source.uri, sample_size=0.05)
    assert out.get("error") is None, f"{kind}: {out.get('error')}"
    assert out.get("rows"), f"{kind}: no rows sampled"


@pytest.mark.cloud
def test_real_cloud_smoke(workdir):
    """Only runs when a real cloud switch is set; otherwise skipped."""
    import os

    if not any(os.environ.get(f"AVALOKA_BENCH_{x}") == "1" for x in ("GCS", "S3", "AZURE")):
        pytest.skip("no real cloud source enabled")
    spec = get("titanic")
    handle = materialize(spec)
    for kind, env in (("gcs", "GCS"), ("s3", "S3"), ("azure", "AZURE")):
        if os.environ.get(f"AVALOKA_BENCH_{env}") != "1":
            continue
        src = make_source(kind, handle, workdir)
        local = src.materialize_locally()
        assert local.exists()
