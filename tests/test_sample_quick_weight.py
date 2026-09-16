"""
The synthetic _weight column must not pollute quick-sample profiling.

sample_quick added a synthetic _weight column to base_sample *before* stats
aggregation, so _weight was profiled as a real (constant) column — polluting
column_statistics and data_quality.constant_columns and feeding the LLM profiler.

Fixes: sample_quick no longer pre-adds _weight (create_sample_portfolio adds it
per sample), and aggregate_statistics_from_portfolio defensively skips _weight.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import daft

import app.agents.sampling_agent_daft as sad
import app.agents.sampling_async as sa


def test_aggregate_skips_weight_column():
    """Even if base_sample carries _weight, it must not be profiled."""
    df = daft.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "z"]})
    base = df.with_column("_weight", daft.lit(1.0))

    stats = sad.aggregate_statistics_from_portfolio({"random_baseline": base}, base)

    assert "_weight" not in stats["column_statistics"]
    assert set(stats["column_statistics"]) == {"a", "b"}
    assert "_weight" not in stats["data_quality"]["constant_columns"]


def test_sample_quick_does_not_profile_weight(tmp_path, monkeypatch):
    """The large-dataset quick path must not surface _weight in profiling."""
    # Force the large (quick-preview) branch without a huge file.
    monkeypatch.setattr(sad, "_estimate_dataset_size", lambda *a, **k: (200_000, 10.0))

    csv = tmp_path / "q.csv"
    csv.write_text("a,b\n" + "\n".join(f"{i % 5},cat{i % 3}" for i in range(300)) + "\n")

    res = sa.sample_quick(path=str(csv), source_type="csv", use_ray=False)

    assert not res.get("error")
    pr = res["profiling_result"]
    assert pr["portfolio_metadata"]["quick_mode"] is True   # confirms the large path ran
    assert "_weight" not in pr["column_statistics"]
    assert "_weight" not in pr["data_quality"].get("constant_columns", [])
    # display rows never carry _weight either
    assert all("_weight" not in row for row in res.get("rows", []))
