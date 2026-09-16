"""Source-connector tests: local file, database and cloud object storage.

Proves the CLI can analyse/train from *any* source by resolving each to a local
file with honest provenance — the property the whole source-agnostic benchmark
relies on.
"""

import numpy as np
import pandas as pd
import pytest

from avaloka.io import materialize, parse_uri, route
from avaloka.io.sources import ResolvedSource, SourceError
from avaloka.workload import Lane


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=300), "b": rng.integers(0, 5, 300),
        "c": rng.choice(["x", "y"], 300),
    })


@pytest.fixture
def csv(tmp_path, frame):
    p = tmp_path / "data.csv"
    frame.to_csv(p, index=False)
    return p


# --- parsing --------------------------------------------------------------
def test_parse_uri_local_variants(tmp_path):
    assert parse_uri("/abs/path.csv")[0] == ""
    assert parse_uri("relative.csv")[0] == ""
    assert parse_uri("file:///abs/path.csv")[0] == ""   # file:// is normalised to local
    assert parse_uri("sqlite:///x.db#t")[0] == "sqlite"
    assert parse_uri("s3://bucket/key.csv")[0] == "s3"


# --- local ----------------------------------------------------------------
def test_materialize_local_in_place(csv):
    r = materialize(str(csv))
    assert isinstance(r, ResolvedSource)
    assert r.kind == "file" and r.materialized is False
    assert r.local_path == csv


def test_materialize_file_scheme(csv):
    r = materialize(f"file://{csv}")
    assert r.local_path.exists() and r.kind == "file"


def test_materialize_missing_local_raises():
    with pytest.raises(FileNotFoundError):
        materialize("/no/such/file.csv")


# --- database -------------------------------------------------------------
@pytest.fixture
def sqlite_url(tmp_path, frame):
    import sqlalchemy
    db = tmp_path / "app.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    frame.to_sql("records", engine, index=False)
    engine.dispose()
    return f"sqlite:///{db}"


def test_materialize_database_table(sqlite_url, frame):
    r = materialize(f"{sqlite_url}#records")
    assert r.kind == "database" and r.materialized
    assert r.detail["n_rows"] == len(frame) and r.detail["n_cols"] == frame.shape[1]
    back = pd.read_parquet(r.local_path)
    assert list(back.columns) == list(frame.columns)


def test_materialize_database_query(sqlite_url):
    r = materialize(f"{sqlite_url}#query=SELECT a, b FROM records WHERE b > 2")
    df = pd.read_parquet(r.local_path)
    assert list(df.columns) == ["a", "b"] and (df["b"] > 2).all()


def test_database_unknown_table_raises(sqlite_url):
    with pytest.raises(SourceError):
        materialize(f"{sqlite_url}#does_not_exist")


def test_database_requires_fragment(sqlite_url):
    with pytest.raises(SourceError):
        materialize(sqlite_url)


# --- object storage (fsspec) ---------------------------------------------
def test_materialize_memory_object(csv):
    import fsspec
    uri = "memory://tests/data.csv"
    with fsspec.open(uri, "wb") as fh:
        fh.write(csv.read_bytes())
    r = materialize(uri)
    assert r.kind == "object_storage" and r.materialized
    assert pd.read_csv(r.local_path).shape[0] == 300


def test_object_storage_missing_backend_raises():
    # s3fs is not installed in this environment → a clear SourceError, not a crash.
    pytest.importorskip("fsspec")
    try:
        import s3fs  # noqa: F401
        pytest.skip("s3fs installed; missing-backend path not exercised")
    except ImportError:
        pass
    with pytest.raises(SourceError):
        materialize("s3://bucket/key.csv")


# --- routing integration --------------------------------------------------
def test_route_sizes_any_source(sqlite_url):
    resolved, plan = route(f"{sqlite_url}#records")
    assert resolved.kind == "database"
    assert plan.lane is Lane.ONLINE and plan.n_rows == 300
