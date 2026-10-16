"""Integration tests for the Avaloka CLI mission workflows.

Covers intermediate integration between the CLI interface and core mission execution:
- analyze mission end-to-end integration and deliverable artifact verification
- train mission model packaging and deployable container artifact generation
- coordinator fleet dispatching and aggregation across workers
- data source materialization and workload state binding
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from avaloka.coordinator import MainAvaloka
from avaloka.io import materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train
from avaloka.workload import Lane


def test_analyze_integration_deliverables_bundle(churn_csv, tmp_path):
    """run_analyze generates the complete defensible analysis deliverables bundle."""
    out = tmp_path / "bundle"
    ctx = MissionContext(
        kind=MissionKind.ANALYZE,
        goal="Identify key churn drivers",
        data_source=str(churn_csv),
        output_dir=out,
        budget=Budget(),
    )

    result = run_analyze(ctx)
    assert result is not None
    assert result.summary["verdict"] in {"pass", "warn"}

    # Verify all documented deliverable files exist
    expected_files = [
        "executive_report.html",
        "technical_report.html",
        "analysis.py",
        "analysis.ipynb",
        "lineage.json",
        "data_quality.json",
        "assumptions.yaml",
        "mission.json",
    ]
    for filename in expected_files:
        filepath = out / filename
        assert filepath.exists(), f"Missing deliverable: {filename}"

    # Verify lineage structure
    lineage = json.loads((out / "lineage.json").read_text())
    assert isinstance(lineage, dict)

    # Verify data quality report
    dq = json.loads((out / "data_quality.json").read_text())
    assert isinstance(dq, dict)


def test_train_integration_deployable_package(churn_csv, tmp_path):
    """run_train with deployable=True emits all production serving artifacts."""
    out = tmp_path / "model_pkg"
    ctx = MissionContext(
        kind=MissionKind.TRAIN,
        goal="Predict churn",
        data_source=str(churn_csv),
        output_dir=out,
        budget=Budget(),
        target="churned",
        metric="roc_auc",
        deployable=True,
    )

    result = run_train(ctx)
    assert result is not None
    assert "model" in result.summary

    # Verify deployment artifacts
    assert (out / "model" / "model.pkl").exists()
    assert (out / "Dockerfile").exists()
    assert (out / "service.py").exists()
    assert (out / "mission.json").exists()


def test_coordinator_fleet_integration(churn_csv, tmp_path):
    """MainAvaloka partitions work, runs worker clones, and synthesizes results."""
    out = tmp_path / "coord_out"
    main = MainAvaloka()
    res = main.coordinate(
        source=str(churn_csv),
        goal="Coordinated analysis of churn",
        output_dir=out,
        workers=2,
    )
    assert res is not None
    assert res.economics is not None
    assert (out / "coordination.json").exists()


def test_materialize_local_dataset_integration(churn_csv):
    """materialize cleanly loads local file, infers format, and binds workload."""
    resolved = materialize(str(churn_csv))
    assert resolved.local_path.exists()
    assert resolved.scheme in {"file", "local"} or resolved.local_path == churn_csv
