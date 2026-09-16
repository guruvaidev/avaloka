"""Scorers — turn mission/coordination artifacts into graded checks.

Every scorer reads the *artifacts a real user would inspect* (the written JSON
deliverables) rather than in-memory objects, so a green benchmark means the CLI
genuinely produced correct, on-disk outputs. Model quality is graded against a
naive baseline computed independently here, so "beats baseline" is measured, not
asserted by the system under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avaloka.benchmark.spec import CheckResult, Expectations


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Data-engineering scorers
# --------------------------------------------------------------------------
def score_workload(out: Path, exp: Expectations) -> list[CheckResult]:
    if not exp.expected_lane:
        return []
    wl = _read_json(out / "workload.json")
    lane = wl.get("lane")
    return [CheckResult(
        "workload_routing", lane == exp.expected_lane,
        f"router chose '{lane}', expected '{exp.expected_lane}'", weight=2.0)]


def score_roles(out: Path, exp: Expectations) -> list[CheckResult]:
    if not exp.expected_roles:
        return []
    profile = _read_json(out / "data_quality.json")
    roles = {c["name"]: c["role"] for c in profile["columns"]}
    checks = []
    for col, want in exp.expected_roles.items():
        got = roles.get(col)
        checks.append(CheckResult(
            f"role[{col}]", got == want, f"inferred '{got}', expected '{want}'"))
    return checks


def score_quality(out: Path, exp: Expectations) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not (exp.expect_quality_flags or exp.min_quality is not None or exp.max_quality is not None):
        return checks
    q = _read_json(out / "data_quality.json")["quality"]
    for key, cols in exp.expect_quality_flags.items():
        flagged = set(q.get(key, []))
        for col in cols:
            checks.append(CheckResult(
                f"quality[{key}:{col}]", col in flagged,
                f"'{col}' {'in' if col in flagged else 'MISSING from'} {key}={sorted(flagged)}"))
    if exp.min_quality is not None:
        checks.append(CheckResult("quality_min", q["score"] >= exp.min_quality,
                                  f"score {q['score']} >= {exp.min_quality}"))
    if exp.max_quality is not None:
        checks.append(CheckResult("quality_max", q["score"] <= exp.max_quality,
                                  f"score {q['score']} <= {exp.max_quality}"))
    return checks


def score_transformation(out: Path) -> list[CheckResult]:
    parquet = out / "transformed_dataset.parquet"
    ok = parquet.exists()
    rows = 0
    if ok:
        try:
            rows = len(pd.read_parquet(parquet))
        except Exception:
            ok = False
    return [CheckResult("transformation_reproducible", ok and rows > 0,
                        f"transformed_dataset.parquet exists with {rows} rows")]


# --------------------------------------------------------------------------
# Data-science scorers
# --------------------------------------------------------------------------
def score_leakage(out: Path, exp: Expectations) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not (exp.must_detect_leakage or exp.expected_verdict):
        return checks
    report = _read_json(out / "validation_report.json")
    flagged = {f["column"] for f in report.get("leakage", [])}
    for col in exp.must_detect_leakage:
        checks.append(CheckResult(
            f"leakage[{col}]", col in flagged,
            f"'{col}' {'flagged' if col in flagged else 'NOT flagged'}; leaks={sorted(flagged)}",
            weight=2.0))
    if exp.expected_verdict:
        checks.append(CheckResult(
            "verdict", report.get("verdict") == exp.expected_verdict,
            f"verdict '{report.get('verdict')}', expected '{exp.expected_verdict}'", weight=2.0))
    return checks


def score_deployment_level(out: Path, exp: Expectations) -> list[CheckResult]:
    if exp.expected_max_level is None:
        return []
    mission = _read_json(out / "mission.json")
    got = mission.get("max_safe_deployment_level")
    return [CheckResult("max_safe_deployment_level", got == exp.expected_max_level,
                        f"validator cap {got}, expected {exp.expected_max_level}", weight=2.0)]


def score_fireflies(out: Path, exp: Expectations) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not (exp.expected_fireflies or exp.forbidden_fireflies):
        return checks
    mission = _read_json(out / "mission.json")
    active = {u["firefly"] for u in mission.get("fireflies", [])}
    for name in exp.expected_fireflies:
        checks.append(CheckResult(f"firefly_active[{name}]", name in active,
                                  f"'{name}' {'activated' if name in active else 'DID NOT activate'}"))
    for name in exp.forbidden_fireflies:
        checks.append(CheckResult(f"firefly_dormant[{name}]", name not in active,
                                  f"'{name}' correctly dormant" if name not in active
                                  else f"'{name}' wrongly activated"))
    return checks


def score_economics(out: Path, exp: Expectations) -> list[CheckResult]:
    if not exp.require_positive_multiplier:
        return []
    econ = _read_json(out / "economics.json")
    mult = econ.get("economic_multiplier", 0)
    effort = econ.get("estimated_manual_effort_hours", 0)
    ok = mult is not None and 0 < mult < 100_000 and math.isfinite(mult) and effort > 0
    return [CheckResult("economics_sane", ok,
                        f"multiplier={mult}, manual_hours={effort}")]


def _baseline_score(frame: pd.DataFrame, target: str, task: str, metric: str) -> float:
    """A naive, model-free baseline the trained model must beat."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                  mean_squared_error, r2_score, roc_auc_score)

    y = frame[target]
    X = frame.drop(columns=[target])
    # deterministic holdout matching the mission's spirit (no leakage into baseline)
    n = len(frame)
    cut = int(n * 0.75)
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    if task == "regression":
        dr = DummyRegressor(strategy="mean").fit(X.iloc[:cut], ytr)
        pred = dr.predict(X.iloc[cut:])
        if metric == "r2":
            return float(r2_score(yte, pred))
        if metric == "mae":
            return float(mean_absolute_error(yte, pred))
        return float(math.sqrt(mean_squared_error(yte, pred)))  # rmse
    dc = DummyClassifier(strategy="prior").fit(X.iloc[:cut], ytr)
    if metric == "roc_auc":
        return 0.5  # a prior-only classifier has no ranking power
    pred = dc.predict(X.iloc[cut:])
    if metric in {"f1", "f1_macro"}:
        return float(f1_score(yte, pred, average="macro" if metric == "f1_macro" else "binary",
                              zero_division=0))
    return float(accuracy_score(yte, pred))


def score_model(out: Path, exp: Expectations, frame: pd.DataFrame, target: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    mission = _read_json(out / "mission.json")
    model = mission.get("model")
    if not model:
        if exp.min_score is not None or exp.min_gain_over_baseline is not None:
            checks.append(CheckResult("model_present", False, "no model in mission.json", weight=2.0))
        return checks
    metric = model["metric"]
    score = model["score"]
    task = mission.get("task", "")
    lower_better = metric in {"rmse", "mae"}

    if exp.min_score is not None:
        ok = score <= exp.min_score if lower_better else score >= exp.min_score
        checks.append(CheckResult("model_min_score", ok,
                                  f"{metric}={score:.4f} vs floor {exp.min_score}", weight=2.0))
    if exp.min_gain_over_baseline is not None:
        try:
            base = _baseline_score(frame, target, task, metric)
            gain = (base - score) if lower_better else (score - base)
            ok = gain >= exp.min_gain_over_baseline
            checks.append(CheckResult("beats_baseline", ok,
                                      f"{metric}: model={score:.4f}, baseline={base:.4f}, "
                                      f"gain={gain:+.4f} >= {exp.min_gain_over_baseline}", weight=2.0))
        except Exception as exc:  # pragma: no cover - defensive
            checks.append(CheckResult("beats_baseline", False, f"baseline error: {exc}"))
    return checks


# --------------------------------------------------------------------------
# Convergence scorers (batch + multi-Avaloka coordination)
# --------------------------------------------------------------------------
def score_exact_convergence(converged: dict[str, Any], frame: pd.DataFrame,
                            columns: list[str]) -> list[CheckResult]:
    """Converged mean/min/max must equal the single-pass truth (no sampling error)."""
    checks: list[CheckResult] = []
    n_ok = converged.get("n_rows") == len(frame)
    checks.append(CheckResult("converged_n_rows", n_ok,
                              f"converged {converged.get('n_rows')} vs {len(frame)}", weight=2.0))
    by_name = {c["name"]: c for c in converged.get("columns", [])}
    for col in columns:
        c = by_name.get(col)
        if c is None or c.get("role") != "numeric":
            checks.append(CheckResult(f"converge[{col}]", False, "column absent or non-numeric"))
            continue
        exact_mean = float(pd.to_numeric(frame[col], errors="coerce").mean())
        exact_min = float(pd.to_numeric(frame[col], errors="coerce").min())
        exact_max = float(pd.to_numeric(frame[col], errors="coerce").max())
        ok = (np.isclose(c["mean"], exact_mean, rtol=1e-5, atol=1e-4)
              and np.isclose(c["min"], exact_min, atol=1e-6)
              and np.isclose(c["max"], exact_max, atol=1e-6))
        checks.append(CheckResult(
            f"converge[{col}]", ok,
            f"mean {c['mean']:.6f}~{exact_mean:.6f}, min {c['min']}~{exact_min}, "
            f"max {c['max']}~{exact_max}", weight=2.0))
    return checks


def score_plan(out: Path, exp: Expectations) -> list[CheckResult]:
    """The task and split a correct plan must choose."""
    if not (exp.expected_task or exp.expected_split):
        return []
    plan = _read_json(out / "planner_graph.json")
    checks: list[CheckResult] = []
    if exp.expected_task:
        got = plan.get("task")
        checks.append(CheckResult("plan_task", got == exp.expected_task,
                                  f"task '{got}', expected '{exp.expected_task}'", weight=2.0))
    if exp.expected_split:
        got = (plan.get("split") or {}).get("strategy")
        checks.append(CheckResult("plan_split", got == exp.expected_split,
                                  f"split '{got}', expected '{exp.expected_split}'", weight=2.0))
    return checks


def score_split_was_performed(out: Path, exp: Expectations,
                              frame: pd.DataFrame) -> list[CheckResult]:
    """A temporal split must be the split the model was actually fitted with.

    The plan recorded ``time_based_holdout`` and the validator reported it as
    used while train_test_split shuffled at random. Reporting a split that did
    not happen is worse than choosing the wrong one, so this checks behaviour
    rather than narration.
    """
    if exp.expected_split != "time_based_holdout":
        return []
    profile = _read_json(out / "data_quality.json")
    time_cols = [c["name"] for c in profile.get("columns", [])
                 if c.get("role") == "datetime" and c["name"] in frame.columns]
    if not time_cols:
        return [CheckResult("temporal_split_performed", False,
                            "no datetime column was recognised, so no temporal split is possible",
                            weight=3.0)]
    try:
        from avaloka.modeling import _temporal_split

        column = time_cols[0]
        features = frame.drop(columns=[column])
        x_train, x_test, _, _ = _temporal_split(frame, features, frame.iloc[:, -1], column, 0.2)
        train_max = pd.to_datetime(frame.loc[x_train.index, column], errors="coerce").max()
        test_min = pd.to_datetime(frame.loc[x_test.index, column], errors="coerce").min()
        ok = bool(train_max < test_min) and len(x_test) > 0
        return [CheckResult("temporal_split_performed", ok,
                            f"train ends {train_max}, test starts {test_min}", weight=3.0)]
    except Exception as exc:                                   # pragma: no cover - defensive
        return [CheckResult("temporal_split_performed", False,
                            f"temporal split unavailable: {type(exc).__name__}: {exc}", weight=3.0)]


def score_baseline(out: Path, exp: Expectations) -> list[CheckResult]:
    """Avaloka's own baseline, not one the harness computes for her.

    The suite already measured a naive baseline itself and compared the model
    against it. That is a fine check of the model and no check at all of the
    product: Avaloka could report a score with nothing to compare it against
    and still pass.
    """
    if not (exp.require_baseline_reported or exp.expect_beats_baseline is not None):
        return []
    model = (_read_json(out / "mission.json") or {}).get("model") or {}
    checks: list[CheckResult] = []
    if exp.require_baseline_reported:
        base = model.get("baseline_score")
        reported = isinstance(base, (int, float)) and base == base
        checks.append(CheckResult("baseline_reported", reported,
                                  f"baseline_score={base!r} ({model.get('baseline_strategy')})",
                                  weight=2.0))
    if exp.expect_beats_baseline is not None:
        got = model.get("beats_baseline")
        checks.append(CheckResult("beats_baseline", got is exp.expect_beats_baseline,
                                  f"beats_baseline={got!r}, expected {exp.expect_beats_baseline!r}",
                                  weight=3.0))
    return checks


def score_features_used(out: Path, exp: Expectations) -> list[CheckResult]:
    """Columns that must survive into the model."""
    if not exp.must_use_features:
        return []
    experiments = _read_json(out / "model" / "experiments.json")
    used = set(experiments.get("feature_columns") or [])
    return [CheckResult(f"feature_used[{name}]", name in used,
                        f"{name} {'kept' if name in used else 'DROPPED'} "
                        f"(model saw: {sorted(used)})", weight=2.0)
            for name in exp.must_use_features]
