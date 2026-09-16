"""End-to-end mission tests: analyze, train, deploy and validation behaviour."""

import json

import pytest

from avaloka.deploy import plan_deployment
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train

ANALYZE_DELIVERABLES = [
    "executive_report.html", "technical_report.html", "analysis.ipynb", "analysis.py",
    "transformed_dataset.parquet", "data_quality.json", "assumptions.yaml",
    "validation_report.json", "planner_graph.json", "lineage.json", "environment.lock",
    "README.md",
]

TRAIN_EXTRA = [
    "model/model.pkl", "model_card.md", "evaluation_report.html", "feature_contract.yaml",
    "inference_schema.json", "requirements.lock", "Dockerfile", "service.py",
    "tests/test_service.py", "monitoring_config.yaml", "deployment/kubernetes.yaml",
    "deployment/rayservice.yaml", "deployment/local-compose.yaml", "mlflow/MLmodel",
]


def _ctx(kind, data, out, **kw):
    return MissionContext(kind=kind, goal=kw.pop("goal", "test"), data_source=str(data),
                          output_dir=out, budget=Budget(limit_usd=kw.pop("budget", None)), **kw)


def test_analyze_produces_all_deliverables(churn_csv, tmp_path):
    out = tmp_path / "ana"
    ctx = _ctx(MissionKind.ANALYZE, churn_csv, out, goal="Understand churn")
    result = run_analyze(ctx)
    for rel in ANALYZE_DELIVERABLES:
        assert (out / rel).exists(), f"missing deliverable: {rel}"
    # notebook is valid JSON
    json.loads((out / "analysis.ipynb").read_text())
    # economics are sane
    assert result.economics.estimated_manual_effort_hours > 0
    assert result.economics.economic_multiplier > 0
    # analysis missions are research artifacts, not deployable
    assert result.summary["max_safe_deployment_level"] == 1


def test_train_produces_model_and_package(churn_csv, tmp_path):
    out = tmp_path / "mdl"
    ctx = _ctx(MissionKind.TRAIN, churn_csv, out, goal="Predict churn",
               target="churned", metric="roc_auc", deployable=True)
    result = run_train(ctx)
    for rel in ANALYZE_DELIVERABLES + TRAIN_EXTRA:
        assert (out / rel).exists(), f"missing deliverable: {rel}"
    assert result.summary["model"]["selected"]
    assert result.summary["verdict"] in {"pass", "warn", "fail"}
    # model pickle actually loads and predicts
    import pickle
    import pandas as pd
    with (out / "model" / "model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    df = pd.read_parquet(out / "transformed_dataset.parquet").drop(columns=["churned"]).head(3)
    assert len(model.predict(df)) == 3


def test_validator_detects_leakage(leaky_csv, tmp_path):
    out = tmp_path / "leak"
    ctx = _ctx(MissionKind.TRAIN, leaky_csv, out, goal="Predict y", target="y", metric="roc_auc")
    result = run_train(ctx)
    report = json.loads((out / "validation_report.json").read_text())
    assert report["verdict"] == "fail"
    assert any(f["column"] == "leak" for f in report["leakage"])
    assert result.summary["max_safe_deployment_level"] == 1


def test_budget_scales_candidate_count(churn_csv, tmp_path):
    out = tmp_path / "tiny"
    # A tiny budget should force the planner to train fewer candidates.
    ctx = _ctx(MissionKind.TRAIN, churn_csv, out, goal="Predict churn",
               target="churned", metric="roc_auc", budget=0.06)
    run_train(ctx)
    plan = json.loads((out / "planner_graph.json").read_text())
    assert plan["compute_plan"]["max_candidates"] <= 2


def test_deploy_blocks_production_without_approval(churn_csv, tmp_path):
    out = tmp_path / "mdl"
    ctx = _ctx(MissionKind.TRAIN, churn_csv, out, goal="Predict churn",
               target="churned", metric="roc_auc", deployable=True)
    run_train(ctx)
    plan = plan_deployment(str(out), "kubernetes", requested_level=4, approved=False)
    assert plan.achieved_level < 4
    assert plan.blocked_reason is not None
    # staging is reachable
    staging = plan_deployment(str(out), "kubernetes", requested_level=3)
    assert staging.achieved_level == 3
