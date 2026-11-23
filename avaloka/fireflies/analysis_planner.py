"""Analysis Planner — replaces the senior data scientist.

This is the firefly closest to Avaloka's moat: it converts a mission (goal,
target, metric, budget) into an efficient, *defensible* execution plan and
answers the hard questions — what task should be performed, is the data
sufficient, which split strategy is valid, is sampling appropriate, which
algorithms to consider, how much computation is economically justified, and
what a human must approve. It emits ``planner_graph.json``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.goal_analysis import analyze_goal
from avaloka.mission.context import MissionKind
from avaloka.specialist_plans import (conformal_config_for,
                                      experiment_plan_from_goal,
                                      statistical_plan_from_goal)
from avaloka.util import write_json


def infer_task(frame: pd.DataFrame, target: str | None) -> tuple[str, dict[str, Any]]:
    """Decide the modelling task from the target column's nature."""
    if not target:
        return "exploratory_analysis", {"reason": "No --target supplied; mission is analysis-only."}
    if target not in frame.columns:
        return "exploratory_analysis", {
            "reason": f"Target {target!r} is not a column; falling back to analysis.",
        }
    y = frame[target].dropna()
    n_unique = y.nunique()
    if n_unique <= 1:
        return "exploratory_analysis", {"reason": "Target has <=1 distinct value."}
    if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
        return "regression", {"reason": f"Numeric target with {n_unique} distinct values."}
    if n_unique == 2:
        return "binary_classification", {"reason": "Target has exactly 2 classes."}
    if n_unique <= 20:
        return "multiclass_classification", {"reason": f"Categorical target with {n_unique} classes."}
    return "regression", {"reason": "High-cardinality numeric target."}


# Per-candidate compute cost is tiny on local CPU; this is the unit the planner
# uses to scale model count to the declared budget.
COST_PER_CANDIDATE_USD = 0.05


class AnalysisPlanner(Firefly):
    name = "analysis_planner"
    human_role = "Senior data scientist"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        frame: pd.DataFrame = self.ctx.blackboard["transformed_frame"]
        profile = self.ctx.blackboard["profile"]
        sampling = self.ctx.blackboard["sampling"]
        target = self.ctx.target

        task, task_reason = infer_task(frame, target)
        is_classification = task.endswith("classification")

        # Split strategy validity.
        has_datetime = any(c["role"] == "datetime" for c in profile["columns"])
        if task == "regression" and has_datetime:
            split = {"strategy": "time_based_holdout",
                     "reason": "Datetime column present; a temporal split avoids look-ahead leakage."}
        elif is_classification:
            split = {"strategy": "stratified_train_test", "test_size": 0.2,
                     "reason": "Stratified split preserves class balance in train and test."}
        elif task == "regression":
            split = {"strategy": "random_train_test", "test_size": 0.2,
                     "reason": "Random holdout is valid for i.i.d. tabular regression."}
        else:
            split = {"strategy": "none", "reason": "Analysis-only mission; no model split required."}

        # Candidate algorithms by task.
        algos = {
            "binary_classification": ["logistic_regression", "random_forest", "gradient_boosting"],
            "multiclass_classification": ["logistic_regression", "random_forest", "gradient_boosting"],
            "regression": ["linear_regression", "random_forest", "gradient_boosting"],
        }.get(task, [])

        # Budget-aware compute plan: how many candidates can we justify? The
        # ensemble counts as one more candidate -- it refits every member per
        # fold, so it costs about as much as the panel it votes over -- and it
        # is the last one a stated budget pays for. With no budget stated it
        # does not run; a user can force it either way with --ensemble/--no-ensemble.
        remaining = self.ctx.budget.remaining()
        if remaining == float("inf"):
            max_candidates = len(algos)
            affordable = None
            budget_note = "Unconstrained budget; all candidate algorithms considered."
        else:
            affordable = int(remaining // COST_PER_CANDIDATE_USD)
            max_candidates = max(1, min(len(algos), affordable)) if algos else 0
            budget_note = (
                f"${remaining:,.2f} budget headroom supports ~{max_candidates} candidate(s) "
                f"at ~${COST_PER_CANDIDATE_USD:.2f}/candidate on local CPU."
            )
        selected_algos = algos[:max_candidates]
        ensemble, ensemble_reason = self._ensemble_decision(selected_algos, affordable)

        # Data sufficiency for the chosen task.
        n_rows = profile["dataset"]["n_rows"]
        sufficiency = self._sufficiency(task, n_rows, frame, target)

        # Hypotheses worth testing, grounded in measured correlations.
        hypotheses = self._hypotheses(profile, target)

        # Which analytical specialists the goal and the task call for. The
        # plans go on the blackboard, where the mission slots the Statistician,
        # Experimenter and ConformalCalibrator in ahead of the Validator; a
        # plan a caller seeded beforehand is kept as is.
        specialists = self._specialists(frame, task)

        # Human approval gates — the system never silently ships to production.
        gates = ["Confirm the target and metric match the intended decision."]
        if task != "exploratory_analysis":
            gates.append("Approve the model before any Level 4 (production) deployment.")
        if profile["quality"]["high_missing_columns"]:
            gates.append("Decide an imputation policy for high-missingness columns.")
        if not sufficiency["sufficient"]:
            gates.append("Acknowledge low data sufficiency before trusting model metrics.")

        plan = {
            "mission_id": self.ctx.mission_id,
            "goal": self.ctx.goal,
            "task": task,
            "task_reason": task_reason["reason"],
            "target": target,
            "metric": self.ctx.metric,
            "split": split,
            "sampling": {"sampled": sampling["sampled"], "strategy": sampling["strategy"]},
            "candidate_algorithms": selected_algos,
            "compute_plan": {
                "execution_mode": self.ctx.execution_mode.value,
                "device": "cpu",
                "max_candidates": max_candidates,
                "cost_per_candidate_usd": COST_PER_CANDIDATE_USD,
                "budget_note": budget_note,
                "ensemble": ensemble,
                "ensemble_reason": ensemble_reason,
            },
            "data_sufficiency": sufficiency,
            "workload_optimization": self.ctx.blackboard.get("workload_optimization", {}),
            "hypotheses": hypotheses,
            "human_approval_gates": gates,
            "refuse_if": [
                "Target leakage is detected and unresolved.",
                "Data is insufficient for a stable estimate of the chosen metric.",
                "A requested deployment latency cannot be met by any candidate.",
            ],
            "specialists": specialists,
            "execution_graph": self._graph(task, specialists),
        }

        self.ctx.blackboard["plan"] = plan
        if self.ctx.kind == MissionKind.ANALYZE:
            # The answer is calculated over the rows the goal named: the same
            # predicates the Data Engineer applied, so the figure and the
            # prepared data agree.
            # When every source row is already in memory the figures are
            # computed over all of them; a sample is only the fallback, and
            # the result says which it used.
            applied = (self.ctx.blackboard.get("workload_optimization") or {}).get("applied") or {}
            whole = self.ctx.blackboard.get("frame")
            if whole is not None and len(whole) != int(profile["dataset"].get("n_rows") or -1):
                whole = None
            goal_result = analyze_goal(
                self.ctx.blackboard["working_frame"], self.ctx.goal, profile,
                sampled=bool(sampling.get("sampled")),
                filters=applied.get("filters_applied") or None,
                population=whole,
            )
            self.ctx.blackboard["goal_analysis"] = goal_result
            plan["goal_analysis_status"] = goal_result["status"]
            plan["goal_analysis_kind"] = goal_result["kind"]
            write_json(self.ctx.path("goal_analysis.json"), goal_result)
        write_json(self.ctx.path("planner_graph.json"), plan)

        return (
            f"Planned task='{task}' with {len(selected_algos)} candidate(s); "
            f"split='{split['strategy']}'; {len(gates)} human approval gate(s).",
            60.0,
            {"task": task, "n_candidates": len(selected_algos)},
        )

    def _ensemble_decision(self, selected_algos: list[str],
                           affordable: int | None) -> tuple[bool, str]:
        """Whether the panel also votes as an ensemble, and why.

        The ensemble roughly doubles a train mission (every member is refit
        per fold), which is a cost on a laptop as much as in dollars. So the
        default path -- no --max-cost, no flag -- skips it; it runs when the
        user says --ensemble, or when a stated budget has room for one more
        candidate after the panel. ``affordable`` is None for no budget.
        """
        forced = self.ctx.ensemble
        if len(selected_algos) < 2:
            return False, "An ensemble needs at least two candidates to vote."
        if forced is False:
            return False, "Disabled by --no-ensemble."
        if forced is True:
            return True, "Requested with --ensemble."
        if affordable is None:
            return False, ("No budget was stated, and the ensemble roughly doubles training "
                           "time; pass --ensemble, or a --max-cost with room for it, to train it.")
        if affordable >= len(selected_algos) + 1:
            return True, ("The budget has room for one more candidate after the panel, so the "
                          "panel also votes as an ensemble (roughly doubles training time).")
        return False, (f"The budget pays for {affordable} candidate(s), not the panel plus an "
                       "ensemble; pass --ensemble to train it anyway.")

    def _specialists(self, frame: pd.DataFrame, task: str) -> dict[str, Any]:
        """Decide, and write, the specialist plans the goal and task call for.

        Each entry records whether a specialist was planned and why, so the
        planner graph says "no experiment: no arm column" rather than
        silently running without one.
        """
        bb = self.ctx.blackboard
        out: dict[str, Any] = {}

        plan, rationale = experiment_plan_from_goal(self.ctx.goal, frame)
        if plan is not None and bb.get("experiment_plan") is None:
            bb["experiment_plan"] = plan
        out["experiment"] = {**rationale, "plan": bb.get("experiment_plan") if plan else None}

        plan, rationale = statistical_plan_from_goal(self.ctx.goal, frame)
        if plan is not None and bb.get("statistical_plan") is None:
            bb["statistical_plan"] = plan
        out["statistic"] = {**rationale, "plan": bb.get("statistical_plan") if plan else None}

        config = conformal_config_for(task) if self.ctx.kind == MissionKind.TRAIN else None
        if config is not None and bb.get("conformal_config") is None:
            bb["conformal_config"] = config
        out["conformal"] = {
            "planned": config is not None,
            "reason": (f"Every train mission calibrates a {config['target_coverage']:.0%} "
                       f"{'interval' if config['task'] == 'regression' else 'label set'} "
                       "on the winner's holdout." if config else
                       "Conformal calibration needs a trained model."),
            "config": ({k: v for k, v in config.items()} if config else None),
        }
        return out

    def _sufficiency(self, task: str, n_rows: int, frame: pd.DataFrame, target: str | None) -> dict[str, Any]:
        if task == "exploratory_analysis":
            ok = n_rows >= 30
            return {"sufficient": ok, "n_rows": n_rows,
                    "reason": "EDA needs only a modest sample." if ok else "Too few rows for stable EDA."}
        # Rule of thumb: >= ~50 rows per feature, and per class for classification.
        n_features = max(1, frame.shape[1] - (1 if target in frame.columns else 0))
        needed = 50 * n_features
        ok = n_rows >= needed
        detail = {"sufficient": ok, "n_rows": n_rows, "rows_recommended": needed,
                  "reason": f"~50 rows/feature suggests >= {needed} rows for {n_features} features."}
        if task.endswith("classification") and target in frame.columns:
            counts = frame[target].value_counts()
            min_class = int(counts.min()) if len(counts) else 0
            detail["min_class_count"] = min_class
            if min_class < 20:
                detail["sufficient"] = False
                detail["reason"] += f" Smallest class has only {min_class} examples (<20)."
        return detail

    def _hypotheses(self, profile: dict[str, Any], target: str | None) -> list[str]:
        hyps: list[str] = []
        for pair in profile.get("correlations", [])[:3]:
            if pair["corr"] >= 0.5:
                hyps.append(
                    f"'{pair['a']}' and '{pair['b']}' move together (|corr|={pair['corr']}); "
                    "consider redundancy or a shared driver."
                )
        if target:
            hyps.append(f"Identify which features most influence '{target}' and quantify the effect.")
        else:
            hyps.append("Surface the dominant segments and the variables that separate them.")
        if not hyps:
            hyps.append("No strong linear relationships detected; explore non-linear structure.")
        return hyps

    def _graph(self, task: str, specialists: dict[str, Any] | None = None) -> dict[str, Any]:
        nodes = [
            {"id": "data_scout", "role": "Data analyst", "produces": "data_quality.json"},
            {"id": "sampling_specialist", "role": "Data scientist", "produces": "working sample"},
            {"id": "workload_optimizer", "role": "Query optimization engineer", "produces": "workload_optimization.json"},
            {"id": "data_engineer", "role": "Data engineer", "produces": "transformed_dataset.parquet"},
            {"id": "analysis_planner", "role": "Senior data scientist", "produces": "planner_graph.json"},
        ]
        edges = [
            {"from": "data_scout", "to": "sampling_specialist"},
            {"from": "sampling_specialist", "to": "workload_optimizer"},
            {"from": "workload_optimizer", "to": "data_engineer"},
            {"from": "data_engineer", "to": "analysis_planner"},
        ]
        # The specialists sit between the planner (or the model scientist,
        # whose holdout the calibrator needs) and the validator.
        specialists = specialists or {}
        before_validator = "analysis_planner"
        if task != "exploratory_analysis":
            nodes.append({"id": "model_scientist", "role": "ML scientist", "produces": "model + experiments"})
            edges.append({"from": "analysis_planner", "to": "model_scientist"})
            before_validator = "model_scientist"
        for key, node in (
            ("statistic", {"id": "statistician", "role": "Statistical analysis specialist",
                           "produces": "statistical_result.json"}),
            ("experiment", {"id": "experimenter", "role": "Experimentation scientist",
                            "produces": "experiment_result.json"}),
            ("conformal", {"id": "conformal_calibrator", "role": "Predictive uncertainty specialist",
                           "produces": "conformal_result.json"}),
        ):
            if specialists.get(key, {}).get("planned"):
                nodes.append(node)
                edges.append({"from": before_validator, "to": node["id"]})
                before_validator = node["id"]
        nodes.append({"id": "validator", "role": "Model-risk specialist",
                      "produces": "validation_report.json"})
        edges.append({"from": before_validator, "to": "validator"})
        if task != "exploratory_analysis":
            nodes += [
                {"id": "finops", "role": "Infrastructure engineer", "produces": "cost-quality options"},
                {"id": "ml_engineer", "role": "ML platform engineer", "produces": "deployment package"},
            ]
            edges += [
                {"from": "validator", "to": "finops"},
                {"from": "finops", "to": "ml_engineer"},
                {"from": "ml_engineer", "to": "reporter"},
            ]
        else:
            edges.append({"from": "validator", "to": "reporter"})
        nodes.append({"id": "reporter", "role": "Analyst / consultant",
                      "produces": "executive & technical reports"})
        return {"nodes": nodes, "edges": edges}
