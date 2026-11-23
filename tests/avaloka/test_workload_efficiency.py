"""Opt-in paired efficiency measurements over downloaded Kaggle datasets.

Run with:
  pytest tests/avaloka/test_workload_efficiency.py -m kaggle_efficiency -v -s
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from avaloka.benchmark.datasets import ensure_kaggle_dataset
from avaloka.benchmark.workload_efficiency import (
    benchmark_parquet_scan,
    choose_selective_filter,
)

pytestmark = [pytest.mark.kaggle, pytest.mark.kaggle_efficiency]

DATASETS = [
    ("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ("yasserh/titanic-dataset", "Titanic-Dataset.csv"),
    ("uciml/iris", "Iris.csv"),
]


@pytest.mark.parametrize("slug,preferred", DATASETS, ids=["telco", "titanic", "iris"])
def test_kaggle_optimizer_efficiency(slug, preferred, tmp_path, record_property):
    source = ensure_kaggle_dataset(slug, preferred)
    if source is None:
        pytest.skip(f"Kaggle dataset {slug} is not downloaded")
    frame = pd.read_csv(source)
    filter_column, filter_value = choose_selective_filter(frame)
    projections = [filter_column] + [c for c in frame.columns if c != filter_column][:2]
    parquet = tmp_path / f"{slug.replace('/', '__')}.parquet"
    frame.to_parquet(parquet, index=False, row_group_size=max(64, len(frame) // 16))

    result = benchmark_parquet_scan(
        parquet, filter_column=filter_column, filter_value=filter_value,
        projected_columns=projections, repeats=7,
    )
    report = result.as_dict()
    (tmp_path / "workload_efficiency.json").write_text(json.dumps(report, indent=2, default=str))
    record_property("workload_efficiency", report)
    print(json.dumps(report, indent=2, default=str))

    assert result.baseline.rows_returned == result.optimized.rows_returned
    assert result.baseline.columns_returned == result.optimized.columns_returned
    # Timing is recorded, not asserted: small datasets and shared CI hosts are noisy.
    assert result.baseline.cpu_seconds >= 0
    assert result.optimized.cpu_seconds >= 0
