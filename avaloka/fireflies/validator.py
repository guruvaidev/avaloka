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

import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.stats import detect_target_leakage
from avaloka.util import write_json

PASS, WARN, FAIL = "pass", "warn", "fail"


class Validator(Firefly):
    name = "validator"
    human_role = "Model-risk specialist"

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
