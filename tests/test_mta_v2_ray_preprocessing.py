import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from app.agents.mta_v2.local_trainer import fit_feature_preprocessing


TRAINING_IMAGE_ROOT = (
    Path(__file__).parents[1]
    / "app"
    / "agents"
    / "mta_v2"
    / "training_docker_image"
)


def _load_ray_job_module():
    module_name = "avaloka_mta_v2_ray_job_for_tests"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(TRAINING_IMAGE_ROOT))
    spec = importlib.util.spec_from_file_location(
        module_name,
        TRAINING_IMAGE_ROOT / "ray_job.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ray_job = _load_ray_job_module()


def test_init_ray_replaces_immutable_execution_resource_limits(monkeypatch):
    unlimited = object()

    class _ResourceLimits:
        @classmethod
        def for_limits(cls):
            return unlimited

    execution_options = SimpleNamespace(
        resource_limits=_ResourceLimits(),
        preserve_order=True,
        locality_with_output=False,
    )
    context = SimpleNamespace(
        target_max_block_size=0,
        execution_options=execution_options,
    )

    monkeypatch.setattr(ray_job.ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray_job.ray, "nodes", lambda: [{"Alive": True}])
    monkeypatch.setattr(ray_job.ray, "cluster_resources", lambda: {"CPU": 4})
    monkeypatch.setattr(ray_job.DataContext, "get_current", lambda: context)

    ray_job.init_ray()

    assert execution_options.resource_limits is unlimited
    assert execution_options.preserve_order is False
    assert execution_options.locality_with_output is True


def test_ray_training_batches_use_bounded_epoch_shuffle():
    source = inspect.getsource(ray_job.train_loop_per_worker)

    assert 'local_shuffle_buffer_size=max(1024, config["batch_size"] * 4)' in source
    assert 'local_shuffle_seed=config["random_seed"] + epoch' in source


def test_ray_pii_transform_matches_the_application_contract():
    from app.agents.pii_agent import pseudonym

    key = "ray-test-only-key"
    frame = pd.DataFrame({
        "customer_email": ["alice@example.com", None],
        "amount": [10.0, 20.0],
    })
    transformed = ray_job.transform_pii_batch(
        frame,
        transformations={
            "customer_email": {
                "kind": "email",
                "strategy": "pseudonymise",
            }
        },
        key=key,
    )

    assert transformed["customer_email"].iloc[0] == pseudonym(
        "alice@example.com",
        salt="customer_email",
        key=key.encode("utf-8"),
    )
    assert pd.isna(transformed["customer_email"].iloc[1])
    assert frame["customer_email"].iloc[0] == "alice@example.com"


class _Schema:
    names = ["category", "amount", "target"]
    types = [pa.string(), pa.float64(), pa.int64()]


class _FeatureDataset:
    def schema(self):
        return _Schema()


class _NumericStatsDataset:
    def mean(self, on):
        assert on == ["amount"]
        return {"mean(amount)": 10.0}

    def map_batches(self, *args, **kwargs):
        return self

    def std(self, on, ddof):
        assert on == ["amount"]
        assert ddof == 0
        return {"std(amount)": 2.0}


class _TrainingDataset(_FeatureDataset):
    def map_batches(self, *args, **kwargs):
        return _NumericStatsDataset()


class _RowsDataset:
    def __init__(self, rows):
        self._rows = rows

    def take_all(self):
        return self._rows


class _PandasGroupBy:
    def __init__(self, df: pd.DataFrame, column: str):
        self._df = df
        self._column = column

    def count(self):
        counts = self._df.groupby(self._column, dropna=False).size()
        return _RowsDataset(
            [
                {self._column: value, "count()": int(count)}
                for value, count in counts.items()
            ]
        )


class _PandasDataset:
    """Small in-memory stand-in that exercises the real Ray preprocessing code."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    def schema(self):
        return pa.Schema.from_pandas(self._df, preserve_index=False)

    def select_columns(self, columns):
        return _PandasDataset(self._df[list(columns)].copy())

    def groupby(self, column):
        return _PandasGroupBy(self._df, column)

    def map_batches(self, fn, *args, **kwargs):
        return _PandasDataset(fn(self._df.copy()))

    def mean(self, on):
        return {
            f"mean({column})": float(self._df[column].mean())
            for column in on
        }

    def std(self, on, ddof):
        return {
            f"std({column})": float(self._df[column].std(ddof=ddof))
            for column in on
        }


def test_ray_validation_retains_supported_categorical_features():
    result = ray_job.validate_feature_columns(
        _FeatureDataset(),
        ["category", "amount", "missing", "target"],
        "target",
    )

    assert result == ["category", "amount"]


def test_ray_preprocessing_fits_one_hot_imputation_and_scaling(monkeypatch):
    monkeypatch.setattr(
        ray_job,
        "_group_value_counts",
        lambda _dataset, column: {"apple": 4, "banana": 1} if column == "category" else {},
    )

    preprocessing = ray_job.fit_ray_feature_preprocessing(
        _TrainingDataset(),
        ["category", "amount"],
        "classification",
    )

    category = preprocessing["categorical_features"]["category"]
    assert preprocessing["version"] == 3
    assert category["encoding"] == "one_hot"
    assert category["categories"] == ["apple", "banana"]
    assert preprocessing["imputation"]["category"]["fill_value"] == "apple"
    assert preprocessing["imputation"]["amount"]["fill_value"] == 10.0
    assert preprocessing["feature_scaling"]["amount"]["scale"] == 2.0

    validation = pd.DataFrame(
        {
            "category": ["orange", None],
            "amount": [14.0, np.nan],
            "__label__": [0, 1],
        }
    )
    transformed = ray_job.transform_ray_feature_batch(
        validation,
        ["category", "amount"],
        preprocessing,
    )

    assert transformed[category["unknown_column"]].tolist() == [1.0, 0.0]
    apple_column = category["output_columns"][0]
    assert transformed[apple_column].tolist() == [0.0, 1.0]
    numeric_column = preprocessing["feature_scaling"]["amount"]["output_column"]
    assert transformed[numeric_column].tolist() == pytest.approx([2.0, 0.0])


def test_ray_and_local_preprocessing_have_identical_ordered_feature_contract():
    feature_cols = ["age", "fruit", "amount"]
    training_rows = pd.DataFrame(
        {
            "age": [10.0, 11.0, 12.0, 13.0],
            "fruit": ["banana", "apple", "banana", None],
            "amount": [100.0, 110.0, np.nan, 130.0],
            "__label__": [0, 1, 0, 1],
        }
    )

    local = fit_feature_preprocessing(training_rows, feature_cols)
    distributed = ray_job.fit_ray_feature_preprocessing(
        _PandasDataset(training_rows),
        feature_cols,
        "classification",
    )

    assert distributed["model_feature_names"] == local["model_feature_names"]
    assert distributed["model_feature_names"] == [
        "__mta_feature_0_numeric",
        "__mta_feature_1_category_0",
        "__mta_feature_1_category_1",
        "__mta_feature_1_category_unknown",
        "__mta_feature_2_numeric",
    ]
    assert distributed["categorical_features"] == local["categorical_features"]

    for column in ("age", "amount"):
        assert distributed["imputation"][column]["fill_value"] == pytest.approx(
            local["imputation"][column]["fill_value"]
        )
        assert distributed["feature_scaling"][column]["mean"] == pytest.approx(
            local["feature_scaling"][column]["mean"]
        )
        assert distributed["feature_scaling"][column]["scale"] == pytest.approx(
            local["feature_scaling"][column]["scale"]
        )


def test_ray_ordering_change_preserves_regression_target_scaling():
    training_rows = pd.DataFrame(
        {
            "age": [10.0, 11.0, 12.0, 13.0],
            "fruit": ["banana", "apple", "banana", None],
            "__label__": [100.0, 110.0, 120.0, 130.0],
        }
    )

    preprocessing = ray_job.fit_ray_feature_preprocessing(
        _PandasDataset(training_rows),
        ["age", "fruit"],
        "regression",
    )

    assert preprocessing["target_scaling"] == {
        "method": "standard",
        "mean": pytest.approx(115.0),
        "scale": pytest.approx(np.std([100.0, 110.0, 120.0, 130.0], ddof=0)),
    }


def test_ray_balanced_class_weights_upweight_minority_class():
    weights = ray_job.balanced_class_weights_from_counts({0: 4, 1: 1}, 2)

    assert weights == pytest.approx([0.625, 2.5])
    assert weights[1] > weights[0]


def test_ray_early_stopping_requires_configured_minimum_improvement():
    assert ray_job.validation_loss_improved(0.89, 1.0, 0.1)
    assert not ray_job.validation_loss_improved(0.91, 1.0, 0.1)
