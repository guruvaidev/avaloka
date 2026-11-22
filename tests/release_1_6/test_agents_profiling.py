"""1.6 release suite: dataset profiling ground truth (plan case AG-06).

Runs the real sampler/profiler (Daft native executor, no Ray) over a synthetic
CSV whose shape, dtypes and null counts are known exactly.
See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

import pandas as pd

from app.agents.sampling_agent_daft import sample_with_profiling


def test_ag06_profile_matches_the_fixture_ground_truth(tmp_path):
    """AG-06: row count, column count, dtypes and per-column null counts in the
    profile equal the fixture's ground truth exactly."""
    n = 40
    frame = pd.DataFrame({
        "id": range(1, n + 1),
        "amount": [1.5 if i % 4 else None for i in range(n)],      # 10 nulls
        "region": ["north", "south", None, "east"] * (n // 4),      # 10 nulls
        "units": list(range(n)),                                    # 0 nulls
    })
    path = tmp_path / "ground_truth.csv"
    frame.to_csv(path, index=False)

    result = sample_with_profiling(str(path), "csv", use_ray=False)

    assert not result.get("error"), result.get("error")
    assert result["total_rows_exact"] == n
    assert result["schema"] == {"id": "Int64", "amount": "Float64", "region": "String", "units": "Int64"}

    stats = result["profiling_result"]["column_statistics"]
    assert set(stats) == {"id", "amount", "region", "units"}
    expected_nulls = {c: int(frame[c].isna().sum()) for c in frame.columns}
    for col, nulls in expected_nulls.items():
        assert stats[col]["missing_count"] == nulls, (col, stats[col])
        assert stats[col]["count"] == n - nulls, (col, stats[col])
    assert stats["id"]["n_unique"] == n
    assert stats["amount"]["type"] == "numeric"
    assert stats["region"]["type"] == "categorical"
