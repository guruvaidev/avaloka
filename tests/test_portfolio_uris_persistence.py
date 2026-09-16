"""
portfolio_sample_uris must survive persistence.

The background job computes portfolio_sample_uris (cloud URIs for each portfolio
sample) and passes them to persist_full_profile, but they were dropped — never
stored — so after a session restart the server's stored_profile.get(
"portfolio_sample_uris") was always None and Ray sample reuse broke.

This test drives persist_full_profile -> load_profile through an in-memory fake
Supabase client and asserts the URIs round-trip while profiling_result stays clean.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.agents.sampling_persistence as sp


class _Exec:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, table):
        self.store, self.table = store, table
        self._op = self._row = self._conflict = self._in = None
        self._filters = []

    def upsert(self, row, on_conflict=None):
        self._op, self._row, self._conflict = "upsert", row, on_conflict
        return self

    def select(self, *a, **k):
        self._op = "select"
        return self

    def eq(self, k, v):
        self._filters.append((k, v))
        return self

    def in_(self, k, vals):
        self._in = (k, list(vals))
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        rows = self.store.setdefault(self.table, [])
        if self._op == "upsert":
            keys = (self._conflict or "").split(",")
            keyof = lambda r: tuple(r.get(c) for c in keys)
            for i, r in enumerate(rows):
                if keyof(r) == keyof(self._row):
                    rows[i] = {**r, **self._row}
                    return _Exec([rows[i]])
            rows.append(dict(self._row))
            return _Exec([self._row])
        res = rows
        for k, v in self._filters:
            res = [r for r in res if r.get(k) == v]
        if self._in:
            k, vals = self._in
            res = [r for r in res if r.get(k) in vals]
        return _Exec(list(res))


class _Client:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        return _Query(self.store, name)


def _profiling_result():
    return {
        "column_statistics": {"a": {"type": "numeric"}},
        "data_quality": {"overall_completeness": 0.9},
        "data_shape": {"rows": 100, "columns": 1},
        "portfolio_metadata": {"display_sample_size": 50},
    }


def test_portfolio_sample_uris_round_trip(monkeypatch):
    store = {}
    monkeypatch.setattr(sp, "get_supabase_client", lambda: _Client(store))

    uris = {
        "random_baseline": "gs://bucket/_avaloka_sampling/ds1/portfolio/random_baseline.csv",
        "strat_region": "gs://bucket/_avaloka_sampling/ds1/portfolio/strat_region.csv",
    }
    sp.persist_full_profile(
        dataset_id="ds1",
        full_result={
            "schema": ["a"],
            "rows": [],
            "profiling_result": _profiling_result(),
            "portfolio_samples": {},
            "portfolio_sample_uris": uris,
        },
        source_path="gs://bucket/data.csv",
        source_type="csv",
    )

    loaded = sp.load_profile("ds1")

    assert loaded is not None
    assert loaded.get("portfolio_sample_uris") == uris          # recovered after "restart"
    assert "portfolio_sample_uris" not in loaded["profiling_result"]  # kept clean
    assert loaded["profiling_result"]["column_statistics"] == {"a": {"type": "numeric"}}


def test_missing_uris_does_not_add_key(monkeypatch):
    store = {}
    monkeypatch.setattr(sp, "get_supabase_client", lambda: _Client(store))

    sp.persist_full_profile(
        dataset_id="ds2",
        full_result={
            "schema": ["a"],
            "rows": [],
            "profiling_result": _profiling_result(),
            "portfolio_samples": {},
            # no portfolio_sample_uris
        },
        source_path="gs://bucket/data.csv",
        source_type="csv",
    )

    loaded = sp.load_profile("ds2")

    assert loaded is not None
    assert "portfolio_sample_uris" not in loaded
    assert "portfolio_sample_uris" not in loaded["profiling_result"]


# ── load_profile is defined once (no shadowed duplicate) ─────────────────────

def test_load_profile_defined_once_and_structured():
    import inspect

    src = Path(sp.__file__).read_text()
    assert src.count("def load_profile(") == 1                 # the duplicate is gone

    params = list(inspect.signature(sp.load_profile).parameters)
    assert params == ["dataset_id"]                            # the active structured API
