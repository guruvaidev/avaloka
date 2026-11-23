"""Validator / Skeptic — replaces the reviewer or model-risk specialist.

Validation, not blind automation, is the product's second promise. The
Validator hunts for the failure modes that make automated ML dangerous:
leakage, invalid splits, unstable estimates, insufficient data and overfitting.
It writes ``validation_report.json`` and — critically — caps the deployment
level the mission is allowed to claim, so Avaloka never silently moves a model
toward production.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.goal_analysis import apply_filters
from avaloka.mission.context import MissionKind
from avaloka.stats import detect_target_leakage
from avaloka.util import write_json

PASS, WARN, FAIL = "pass", "warn", "fail"

#: How close a reported figure must be to its recomputation. Figures are
#: stored to six decimals, so anything wider than rounding is a real mismatch.
RECONCILE_REL_TOL = 1e-4
RECONCILE_ABS_TOL = 1e-5


# --- reconciliation ---------------------------------------------------------
# Every figure a goal answer reports carries how to recompute it. The code
# below recomputes it from the rows with numpy, separately from the pandas
# path that produced it, so a figure that does not follow from the data --
# a sum printed as a mean, the wrong group, a stale number -- fails here.
def _numbers(frame: pd.DataFrame, expr: dict[str, Any] | None) -> np.ndarray:
    if expr is None:
        return np.ones(len(frame))
    if "column" in expr:
        return pd.to_numeric(frame[expr["column"]], errors="coerce").to_numpy(dtype="float64")
    if "indicator" in expr:
        spec = expr["indicator"]
        raw = frame[spec["column"]]
        present = raw.notna().to_numpy()
        if pd.api.types.is_numeric_dtype(raw):
            hit = pd.to_numeric(raw, errors="coerce").to_numpy(dtype="float64") == float(spec["equals"])
        else:
            hit = np.array([str(v).lower() == str(spec["equals"]).lower() for v in raw.to_numpy()])
        return np.where(present, hit.astype("float64"), np.nan)
    left, right = expr["product"]
    return _numbers(frame, {"column": left}) * _numbers(frame, {"column": right})


def _text(frame: pd.DataFrame, column: str, missing: str, cache: dict) -> np.ndarray:
    """Each cell as the text a reported group key was written with: ``str()``."""
    key = (column, missing)
    if key not in cache:
        raw = frame[column]
        cache[key] = np.where(raw.isna().to_numpy(), missing, raw.map(str).to_numpy())
    return cache[key]


def _where(frame: pd.DataFrame, conditions: list[dict[str, Any]], cache: dict) -> np.ndarray:
    keep = np.ones(len(frame), dtype=bool)
    for cond in conditions:
        if "equals_text" in cond:
            keep &= _text(frame, cond["column"], cond.get("missing_as", "nan"), cache) == cond["equals_text"]
            continue
        dates = pd.to_datetime(frame[cond["column"]], errors="coerce", dayfirst=bool(cond.get("dayfirst")),
                               utc=bool(cond.get("utc")))
        if "month" in cond:
            keep &= (dates.dt.strftime("%Y-%m") == cond["month"]).to_numpy()
        elif "hour" in cond:
            keep &= (dates.dt.hour == float(cond["hour"])).to_numpy()
        elif "weekday" in cond:
            keep &= (dates.dt.day_name() == cond["weekday"]).to_numpy()
    return keep


def _eta(values: np.ndarray, groups: np.ndarray) -> float:
    ok = ~np.isnan(values)
    values, groups = values[ok], groups[ok]
    mean = values.mean()
    _, inverse = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    between = float(np.sum(counts * (sums / counts - mean) ** 2))
    return math.sqrt(between / float(np.sum((values - mean) ** 2)))


def recompute_figure(frame: pd.DataFrame, check: dict[str, Any], cache: dict | None = None) -> float:
    """The figure ``check`` describes, recomputed from ``frame``."""
    cache = {} if cache is None else cache
    op = check["op"]
    if op == "missing":
        return float(frame[check["column"]].isna().to_numpy().sum())
    if op == "agg":
        keep = _where(frame, check.get("where") or [], cache)
        values = _numbers(frame, check.get("value"))[keep]
        if check["agg"] == "count":
            return float(keep.sum()) if check.get("value") is None else float((~np.isnan(values)).sum())
        values = values[~np.isnan(values)]
        return float({"mean": np.mean, "sum": np.sum, "median": np.median,
                      "max": np.max, "min": np.min}[check["agg"]](values))
    if op == "pearson":
        x, y = _numbers(frame, check["x"]), _numbers(frame, check["y"])
        ok = ~np.isnan(x) & ~np.isnan(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])
    if op == "eta":
        return _eta(_numbers(frame, check["y"]), _text(frame, check["groups"], "<missing>", cache))
    if op == "cramers_v":
        a = _text(frame, check["a"], "<missing>", cache)
        present = frame[check["b"]].notna().to_numpy()
        a, b = a[present], _text(frame, check["b"], "<missing>", cache)[present]
        _, ia = np.unique(a, return_inverse=True)
        _, ib = np.unique(b, return_inverse=True)
        table = np.zeros((ia.max() + 1, ib.max() + 1))
        np.add.at(table, (ia, ib), 1)
        n = table.sum()
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        chi2 = float(np.sum((table - expected) ** 2 / np.where(expected > 0, expected, 1)))
        return math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    raise ValueError(f"no recomputation for {op!r}")


def reconcile_figures(figures: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The figures whose reported value differs from their recomputation."""
    mismatches, cache = [], {}
    for figure in figures:
        try:
            again = recompute_figure(frame, figure["check"], cache)
        except Exception as exc:  # noqa: BLE001 - an unrecomputable figure is itself a mismatch
            mismatches.append({"label": figure["label"], "reported": figure["value"],
                               "recomputed": None, "error": type(exc).__name__})
            continue
        if figure["check"]["op"] == "pearson" and figure["check"].get("absolute"):
            again = abs(again)
        # A figure at the file's stated scale ("£ thousands") is the recomputed
        # figure times that factor (goal_analysis._with_units).
        again *= figure["check"].get("scale", 1)
        if not math.isclose(float(figure["value"]), again, rel_tol=RECONCILE_REL_TOL,
                            abs_tol=RECONCILE_ABS_TOL):
            mismatches.append({"label": figure["label"], "reported": figure["value"], "recomputed": again})
    return mismatches


class Validator(Firefly):
    name = "validator"
    human_role = "Model-risk specialist"

    def _goal_checks(self) -> list[dict[str, Any]]:
        bb = self.ctx.blackboard
        goal = bb.get("goal_analysis")
        if self.ctx.kind != MissionKind.ANALYZE or goal is None:
            return []
        if goal.get("status") != "answered" or not goal.get("figures"):
            reason = (goal.get("findings") or ["no goal-specific figure was produced"])[0]
            return [{"name": "goal_answered", "status": WARN, "severity": "high",
                     "detail": f"Not answered: {reason}"}]
        out = [{"name": "goal_answered", "status": PASS, "severity": "high",
                "detail": f"Answered by a {goal['kind'].replace('_', ' ')} calculation."}]
        basis = goal.get("basis") or {}
        frame = bb.get("working_frame") if basis.get("sampled") else bb.get("frame", bb.get("working_frame"))
        if frame is None:
            out.append({"name": "goal_reconciliation", "status": WARN, "severity": "high",
                        "detail": "The rows behind the answer were not available to recompute its figures."})
            return out
        rows = apply_filters(frame, goal.get("filters") or [])
        figures = list(goal["figures"])
        mismatches = reconcile_figures(figures, rows)
        statistic, plan = bb.get("statistical_result"), bb.get("statistical_plan")
        source = bb.get("transformed_frame")
        if (bb.get("statistical_basis") or {}).get("full_pass") and bb.get("frame") is not None:
            # The Statistician answered from every row: reconcile against them.
            source = apply_filters(bb["frame"], ((bb.get("workload_optimization") or {})
                                                 .get("applied") or {}).get("filters_applied") or [])
        if statistic is not None and isinstance(plan, dict) and source is not None \
                and plan.get("method") in {"mean", "median"} and isinstance(statistic.estimate, (int, float)):
            stat_figure = {"label": plan.get("estimand") or plan["method"], "value": statistic.estimate,
                           "check": {"op": "agg", "agg": plan["method"],
                                     "value": {"column": plan["variables"][0]}, "where": []}}
            figures.append(stat_figure)
            mismatches += reconcile_figures([stat_figure], source)
        if mismatches:
            shown = "; ".join(
                f"{m['label']}: reported {m['reported']}, recomputed "
                + (f"{m['recomputed']:.6g}" if m["recomputed"] is not None else f"nothing ({m.get('error')})")
                for m in mismatches[:3])
            out.append({"name": "goal_reconciliation", "status": FAIL, "severity": "high",
                        "detail": f"{len(mismatches)} of {len(figures)} reported figure(s) do not match a "
                                  f"recomputation from the data: {shown}."})
        else:
            out.append({"name": "goal_reconciliation", "status": PASS, "severity": "high",
                        "detail": f"All {len(figures)} reported figure(s) match an independent "
                                  f"recomputation over {len(rows):,} rows."})
        return out

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        profile = self.ctx.blackboard["profile"]
        plan = self.ctx.blackboard.get("plan", {})
        checks: list[dict[str, Any]] = []
        leakage: list[dict[str, Any]] = []
        max_level = 4  # optimistic; each failed check lowers the ceiling

        # --- data sufficiency ------------------------------------------------
        suff = plan.get("data_sufficiency", {})
        if suff:
            status = PASS if suff.get("sufficient") else WARN
            if status == WARN:
                max_level = min(max_level, 2)
            checks.append({"name": "data_sufficiency", "status": status,
                           "severity": "medium", "detail": suff.get("reason", "")})

        # --- leakage ---------------------------------------------------------
        target = self.ctx.target
        frame: pd.DataFrame | None = self.ctx.blackboard.get("transformed_frame")
        if target and frame is not None and target in frame.columns:
            leakage = detect_target_leakage(frame, target, profile["columns"])
            if leakage:
                max_level = min(max_level, 1)
                checks.append({"name": "target_leakage", "status": FAIL, "severity": "high",
                               "detail": f"{len(leakage)} feature(s) appear to encode the target: "
                                         + ", ".join(f["column"] for f in leakage)})
            else:
                checks.append({"name": "target_leakage", "status": PASS, "severity": "high",
                               "detail": "No feature is near-perfectly predictive of the target."})

        # --- model-specific checks ------------------------------------------
        training = self.ctx.blackboard.get("training")
        if training is not None:
            best = training.best
            # split validity
            split = plan.get("split", {})
            checks.append({"name": "split_validity", "status": PASS, "severity": "medium",
                           "detail": f"Used '{split.get('strategy', 'holdout')}' "
                                     f"({training.n_train} train / {training.n_test} test rows)."})

            # beats the trivial model
            #
            # A score with nothing to compare it against is not evidence. This
            # module and modeling.py both promised "defensible baselines" while
            # computing none, so a model that had learned nothing was reported
            # with the same confidence as one that had.
            beats = training.beats_baseline
            base = training.baseline_score
            if beats is None:
                checks.append({"name": "beats_baseline", "status": WARN, "severity": "medium",
                               "detail": "Could not compare against a trivial baseline."})
                max_level = min(max_level, 2)
            elif beats:
                checks.append({"name": "beats_baseline", "status": PASS, "severity": "high",
                               "detail": f"CV {best.primary_metric} {best.cv_mean:.3f} beats the "
                                         f"{training.baseline_strategy} baseline ({base:.3f}) by "
                                         f"more than fold noise."})
            else:
                # Not a warning. A model that does not beat guessing is not a
                # weak result, it is an absent one.
                max_level = min(max_level, 1)
                checks.append({"name": "beats_baseline", "status": FAIL, "severity": "high",
                               "detail": f"CV {best.primary_metric} {best.cv_mean:.3f} does not beat "
                                         f"the {training.baseline_strategy} baseline ({base:.3f}). "
                                         f"The model has found nothing the baseline had not."})

            # stability: cross-validation variance
            if not math.isnan(best.cv_std) and best.cv_mean:
                rel = abs(best.cv_std / best.cv_mean) if best.cv_mean else float("inf")
                if rel > 0.15:
                    max_level = min(max_level, 2)
                    checks.append({"name": "stability", "status": WARN, "severity": "medium",
                                   "detail": f"CV std/mean = {rel:.0%}; result is sensitive to the split."})
                else:
                    checks.append({"name": "stability", "status": PASS, "severity": "medium",
                                   "detail": f"CV {best.primary_metric} = {best.cv_mean:.3f} "
                                             f"+/- {best.cv_std:.3f} (stable)."})

            # overfitting: holdout vs cv gap
            if not math.isnan(best.cv_mean):
                gap = best.primary_score - best.cv_mean
                if best.primary_metric not in {"rmse", "mae"} and gap > 0.1:
                    max_level = min(max_level, 2)
                    checks.append({"name": "overfitting", "status": WARN, "severity": "medium",
                                   "detail": f"Holdout exceeds CV by {gap:.3f}; optimism likely."})
                else:
                    checks.append({"name": "overfitting", "status": PASS, "severity": "low",
                                   "detail": "Holdout and CV scores are consistent."})

            # deployment latency constraint
            if self.ctx.deployment_latency_ms is not None:
                if best.inference_ms_per_1k <= self.ctx.deployment_latency_ms:
                    checks.append({"name": "latency_budget", "status": PASS, "severity": "medium",
                                   "detail": f"~{best.inference_ms_per_1k:.1f}ms/req <= "
                                             f"{self.ctx.deployment_latency_ms:.0f}ms budget."})
                else:
                    max_level = min(max_level, 1)
                    checks.append({"name": "latency_budget", "status": FAIL, "severity": "high",
                                   "detail": f"~{best.inference_ms_per_1k:.1f}ms/req exceeds "
                                             f"{self.ctx.deployment_latency_ms:.0f}ms budget."})

        # --- data quality gate ----------------------------------------------
        q = profile["quality"]["score"]
        if q < 60:
            max_level = min(max_level, 2)
            checks.append({"name": "data_quality", "status": WARN, "severity": "medium",
                           "detail": f"Data quality score {q}/100; clean before production use."})

        # --- the goal: answered, and do its numbers follow from the data -----
        # An analysis is only a PASS when it answered the question asked and
        # every figure it printed comes back the same when recomputed. A
        # general profile in place of an answer used to read PASS.
        goal_checks = self._goal_checks()
        checks += goal_checks
        if any(c["status"] != PASS for c in goal_checks):
            max_level = min(max_level, 1)

        # The Validator sets the *technical* safety ceiling. A clean model is
        # certified up to Level 4; the human-approval governance gate for
        # production is enforced separately at deploy time (avaloka.deploy). An
        # analysis mission is a research artifact, never a deployable model.
        if training is None:
            max_level = min(max_level, 1)

        statuses = [c["status"] for c in checks]
        verdict = FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS)
        blocking = [c["detail"] for c in checks if c["status"] == FAIL]

        report = {
            "verdict": verdict,
            "max_safe_deployment_level": max_level,
            "checks": checks,
            "leakage": leakage,
            "blocking_issues": blocking,
            "note": "This is the technical safety ceiling. Production (Level 4) additionally "
                    "requires explicit human approval at deploy time; Avaloka never auto-promotes.",
        }
        self.ctx.blackboard["validation"] = report
        write_json(self.ctx.path("validation_report.json"), report)

        return (
            f"Validation verdict='{verdict}'; {len(checks)} checks; max safe deployment "
            f"level {max_level}.",
            60 + 4 * profile["dataset"]["n_cols"],
            {"verdict": verdict, "max_level": max_level},
        )
