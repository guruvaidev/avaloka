"""Tests for the benchmark harness itself (fast synthetic subset).

Verifies the runner executes real missions through the source connectors and the
scorers grade on-disk artifacts correctly — so a green benchmark is trustworthy.
"""

import numpy as np
import pandas as pd
import pytest

from avaloka.benchmark import datasets as ds
from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.scoring import score_exact_convergence
from avaloka.benchmark.spec import (BenchmarkTask, CheckResult, DatasetSpec,
                                    Expectations, Scorecard, TaskResult)
from avaloka.benchmark.suite import (all_tasks, data_engineering_tasks,
                                     data_science_tasks)


# --- dataset builders -----------------------------------------------------
def test_synthetic_builders_have_ground_truth(tmp_path):
    churn = pd.read_csv(ds.build_churn(tmp_path / "c.csv"))
    assert {"churned", "contract", "tenure"} <= set(churn.columns)
    leaky = pd.read_csv(ds.build_leaky(tmp_path / "l.csv"))
    assert (leaky["leak_score"] == leaky["target"]).all()   # planted leak
    dirty = pd.read_csv(ds.build_dirty(tmp_path / "d.csv"))
    assert dirty["income"].isna().mean() > 0.2               # planted holes
    assert dirty["status_flag"].nunique() == 1               # constant column


def test_ensure_kaggle_returns_none_without_cli(monkeypatch):
    monkeypatch.setattr(ds.shutil, "which", lambda _name: None)
    assert ds.ensure_kaggle_dataset("owner/nonexistent-xyz") is None


# --- scoring units --------------------------------------------------------
def test_score_exact_convergence_flags_mismatch():
    frame = pd.DataFrame({"v": np.arange(100.0)})
    good = {"n_rows": 100, "columns": [{"name": "v", "role": "numeric",
                                        "mean": frame["v"].mean(), "min": 0.0, "max": 99.0}]}
    checks = score_exact_convergence(good, frame, ["v"])
    assert all(c.passed for c in checks)
    bad = {"n_rows": 100, "columns": [{"name": "v", "role": "numeric",
                                       "mean": 999.0, "min": 0.0, "max": 99.0}]}
    assert not all(c.passed for c in score_exact_convergence(bad, frame, ["v"]))


def test_scorecard_aggregation():
    card = Scorecard()
    card.add(TaskResult("t1", "data_engineering", "analyze", "file",
                        [CheckResult("a", True), CheckResult("b", False)]))
    card.add(TaskResult("t2", "data_science", "train", "file", [CheckResult("c", True)]))
    assert card.overall_score == pytest.approx((0.5 + 1.0) / 2)
    assert card.by_family()["data_science"] == 1.0
    assert len(card.failures()) == 1


def test_checkresult_coerces_numpy_bool():
    c = CheckResult("x", np.bool_(True))
    assert c.passed is True and isinstance(c.passed, bool)


# --- suite integrity ------------------------------------------------------
def test_suite_has_both_families_and_all_sources():
    de, dsci = data_engineering_tasks(), data_science_tasks()
    assert de and dsci
    schemes = {s for t in de + dsci for s in t.source_schemes}
    assert {"file", "sqlite", "object_storage"} <= schemes
    assert any(t.coordinate_workers for t in dsci)          # a multi-Avaloka task exists
    assert any(t.kind == "batch" for t in de)               # a convergence task exists


# --- end-to-end runner (fast subset) -------------------------------------
def test_runner_scores_data_engineering_task(tmp_path):
    task = BenchmarkTask(
        name="de_smoke", family="data_engineering", kind="analyze",
        goal="Profile churn from file + database.",
        dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn),
        source_schemes=("file", "sqlite"),
        expectations=Expectations(expected_lane="online",
                                  expected_roles={"contract": "categorical"}))
    card = BenchmarkRunner(tmp_path).run([task])
    assert len(card.results) == 2
    assert all(r.passed for r in card.results), [r.as_dict() for r in card.failures()]


def test_runner_scores_leakage_and_coordination(tmp_path):
    task = BenchmarkTask(
        name="ds_smoke", family="data_science", kind="train",
        goal="Catch leakage; converge across a fleet.",
        dataset=DatasetSpec("leaky", "synthetic", builder=ds.build_leaky, target="target"),
        target="target", metric="roc_auc", source_schemes=("file",),
        coordinate_workers=3,
        expectations=Expectations(must_detect_leakage=["leak_score", "outcome_label"],
                                  expected_verdict="fail", expected_max_level=1))
    card = BenchmarkRunner(tmp_path).run([task])
    kinds = {r.kind for r in card.results}
    assert "train" in kinds and "train+coordinate" in kinds
    assert card.overall_score == pytest.approx(1.0), [r.as_dict() for r in card.failures()]
