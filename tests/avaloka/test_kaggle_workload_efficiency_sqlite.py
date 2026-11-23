"""Opt-in indexed workload efficiency benchmark over real Kaggle CSVs."""
import json
import pytest
from avaloka.benchmark.datasets import ensure_kaggle_dataset
from avaloka.benchmark.sqlite_efficiency import (
    benchmark_sqlite_rows, choose_filter, load_csv_rows,
)

pytestmark = [pytest.mark.kaggle, pytest.mark.kaggle_efficiency]
DATASETS = [
    ("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ("yasserh/titanic-dataset", "Titanic-Dataset.csv"),
    ("uciml/iris", "Iris.csv"),
]

@pytest.mark.parametrize("slug,preferred", DATASETS, ids=["telco", "titanic", "iris"])
def test_kaggle_indexed_filter_efficiency(slug, preferred, record_property):
    source = ensure_kaggle_dataset(slug, preferred)
    if source is None:
        pytest.skip(f"Kaggle dataset {slug} is not downloaded")
    columns, rows = load_csv_rows(source)
    filter_column, filter_value = choose_filter(columns, rows)
    projected = [filter_column] + [column for column in columns if column != filter_column][:2]
    result = benchmark_sqlite_rows(source.name, columns, rows, filter_column,
                                   filter_value, projected, repeats=7)
    record_property("workload_efficiency", result.as_dict())
    print(json.dumps(result.as_dict(), indent=2))
    assert result.baseline_vm_steps > result.optimized_vm_steps
    assert result.vm_step_reduction_pct > 0
