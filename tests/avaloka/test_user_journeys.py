"""End-to-end user journeys expressed as business-shaped swarm prompts.

These tests deliberately use the same prompts published in the user guide. They
prove that a decision-oriented request can travel through analysis, deployable
training, and multi-Avaloka coordination and finish as inspectable artifacts.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from avaloka.coordinator import MainAvaloka, converged_mean
from avaloka.swarm_limits import UNLIMITED
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train


CHURN_ANALYSIS_PROMPT = (
    "Analyze which customer segments are most likely to churn, quantify the "
    "strongest drivers, check whether missing or imbalanced data could mislead "
    "us, and recommend three retention actions supported by the data."
)

CHURN_TRAINING_PROMPT = (
    "Predict churn early enough for the retention team to intervene; prioritize "
    "reliable ranking, expose leakage and segment-level failure modes."
)

CHURN_FLEET_PROMPT = (
    "Build a full-population churn risk brief: reconcile segment patterns across "
    "shards, preserve minority behavior, and surface any validation warning "
    "instead of averaging it away."
)


def _context(kind, data, output, goal, **kwargs):
    return MissionContext(
        kind=kind,
        goal=goal,
        data_source=str(data),
        output_dir=output,
        budget=Budget(limit_usd=kwargs.pop("budget", None)),
        **kwargs,
    )


def test_retention_prompt_runs_analysis_then_deployable_training(churn_csv, tmp_path):
    analysis_dir = tmp_path / "retention-analysis"
    analysis = run_analyze(
        _context(MissionKind.ANALYZE, churn_csv, analysis_dir, CHURN_ANALYSIS_PROMPT)
    )

    assert analysis.summary["max_safe_deployment_level"] == 1
    assert (analysis_dir / "executive_report.html").exists()
    assert (analysis_dir / "validation_report.json").exists()
    assert (analysis_dir / "assumptions.yaml").exists()

    model_dir = tmp_path / "retention-model"
    training = run_train(
        _context(
            MissionKind.TRAIN,
            churn_csv,
            model_dir,
            CHURN_TRAINING_PROMPT,
            target="churned",
            metric="roc_auc",
            deployable=True,
            budget=25,
        )
    )

    assert training.summary["model"]["selected"]
    assert training.summary["verdict"] in {"pass", "warn", "fail"}
    assert (model_dir / "model" / "model.pkl").exists()
    assert (model_dir / "deployment" / "kubernetes.yaml").exists()
    assert (model_dir / "tests" / "example_request.json").exists()
    validation = json.loads((model_dir / "validation_report.json").read_text())
    assert "verdict" in validation and "leakage" in validation


def test_full_population_prompt_converges_exactly_across_sub_avalokas(churn_csv, tmp_path):
    source = pd.read_csv(churn_csv)
    result = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv),
        goal=CHURN_FLEET_PROMPT,
        output_dir=tmp_path / "retention-fleet",
        workers=4,
        kind=MissionKind.ANALYZE,
    )

    assert result.workers == 4
    assert result.n_rows == len(source)
    assert converged_mean(result.converged_profile, "tenure") == pytest.approx(
        source["tenure"].mean(), abs=1e-4
    )
    assert len(result.shards) == 4
    assert all(shard.manual_minutes > 0 for shard in result.shards)
    report = json.loads((tmp_path / "retention-fleet" / "coordination.json").read_text())
    assert report["goal"] == CHURN_FLEET_PROMPT
    assert report["n_rows"] == len(source)
