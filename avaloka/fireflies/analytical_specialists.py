"""Executable Firefly adapters for analytical specialist engines."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from avaloka.experiments import ExperimentPlan, analyze_ab_test
from avaloka.features import FeatureRecipe, TransformStep, execute_feature_recipe
from avaloka.fireflies.base import Firefly
from avaloka import goal_analysis
from avaloka.statistics import StatisticalPlan, compute_statistic
from avaloka.uncertainty import (split_conformal_classification,
                                 split_conformal_regression)
from avaloka.util import write_json


def _feature_recipe(value) -> FeatureRecipe:
    if isinstance(value, FeatureRecipe):
        return value
    value = value or {}
    steps = tuple(
        step if isinstance(step, TransformStep) else TransformStep(
            operation=step["operation"], output=step.get("output"),
            inputs=tuple(step.get("inputs") or ()),
            parameters=dict(step.get("parameters") or {}),
        ) for step in value.get("steps", ())
    )
    return FeatureRecipe(
        steps=steps, output_grain=tuple(value.get("output_grain") or ()),
        assumptions=tuple(value.get("assumptions") or ()),
    )


class FeatureEngineer(Firefly):
    name = "feature_engineer"
    human_role = "Feature engineering specialist"

    def perform(self):
        frame: pd.DataFrame = self.ctx.blackboard["transformed_frame"]
        recipe = _feature_recipe(self.ctx.blackboard.get("feature_recipe"))
        result = execute_feature_recipe(frame, recipe)
        self.ctx.blackboard["transformed_frame"] = result.frame
        self.ctx.blackboard["feature_result"] = result
        payload = {
            "recipe": asdict(recipe), "recipe_hash": result.recipe_hash,
            "diagnostics": list(result.diagnostics),
            "shape": list(result.frame.shape),
        }
        write_json(self.ctx.path("feature_recipe.json"), payload)
        return (
            f"Applied {len(recipe.steps)} evidence-preserving feature step(s).",
            45 + 8 * len(recipe.steps),
            {"steps": len(recipe.steps), "recipe_hash": result.recipe_hash},
        )


class Statistician(Firefly):
    name = "statistician"
    human_role = "Statistical analysis specialist"

    def perform(self):
        frame: pd.DataFrame = self.ctx.blackboard["transformed_frame"]
        plan = self.ctx.blackboard.get("statistical_plan")
        if isinstance(plan, dict):
            plan = StatisticalPlan(
                method=plan["method"], variables=tuple(plan.get("variables") or ()),
                grouping=tuple(plan.get("grouping") or ()), estimand=plan.get("estimand", ""),
                unit_of_analysis=tuple(plan.get("unit_of_analysis") or ()),
                weights=plan.get("weights"), missing_policy=plan.get("missing_policy", "drop_required"),
                parameters=dict(plan.get("parameters") or {}),
            )
        if not isinstance(plan, StatisticalPlan):
            raise ValueError("Statistician requires a typed statistical_plan")
        population = self._population(plan)
        if population is not None:
            # Every source row is in memory: the figure is the population's,
            # not the working sample's. The bootstrap would be billions of
            # draws at that size, so a mean takes its analytic interval.
            frame = population
            if plan.method == "mean" and plan.parameters.get("bootstrap"):
                plan = replace(plan, parameters={**plan.parameters, "interval": "analytic"})
        result = compute_statistic(frame, plan)
        self.ctx.blackboard["statistical_result"] = result
        self.ctx.blackboard["statistical_basis"] = {
            "full_pass": population is not None,
            "sampled": population is None and bool((self.ctx.blackboard.get("sampling") or {}).get("sampled")),
            "rows_used": int(len(frame)),
        }
        write_json(self.ctx.path("statistical_result.json"),
                   {**asdict(result), "basis": self.ctx.blackboard["statistical_basis"]})
        return (f"Computed '{plan.method}' for {result.sample_size} valid rows.", 60.0,
                {"method": plan.method, "plan_hash": result.plan_hash})

    def _population(self, plan: StatisticalPlan) -> pd.DataFrame | None:
        """Every source row, restricted to the goal's filters, when the
        working frame is a sample of a frame held in memory -- the same rule
        the goal answer follows (avaloka.goal_analysis.analyze_goal); else
        None, and the figure stays the sample's and says so."""
        bb = self.ctx.blackboard
        whole = bb.get("frame")
        if whole is None or not (bb.get("sampling") or {}).get("sampled"):
            return None
        if len(whole) != int((bb.get("profile") or {}).get("dataset", {}).get("n_rows") or -1) \
                or len(whole) > goal_analysis.FULL_PASS_MAX_ROWS:
            return None
        needed = [*plan.variables, *plan.grouping] + ([plan.weights] if plan.weights else [])
        if any(column not in whole.columns for column in needed):
            return None
        applied = (bb.get("workload_optimization") or {}).get("applied") or {}
        return goal_analysis.apply_filters(whole, applied.get("filters_applied") or [])


class Experimenter(Firefly):
    name = "experimenter"
    human_role = "Experimentation scientist"

    def perform(self):
        frame: pd.DataFrame = self.ctx.blackboard["transformed_frame"]
        plan = self.ctx.blackboard.get("experiment_plan")
        if isinstance(plan, dict):
            fields = ExperimentPlan.__dataclass_fields__
            plan = ExperimentPlan(**{k: v for k, v in plan.items() if k in fields})
        if not isinstance(plan, ExperimentPlan):
            raise ValueError("Experimenter requires a typed experiment_plan")
        result = analyze_ab_test(frame, plan)
        self.ctx.blackboard["experiment_result"] = result
        write_json(self.ctx.path("experiment_result.json"), {**asdict(result), "plan": asdict(plan)})
        if result.effect is None:
            output = f"Experiment analysis finished with status '{result.status}'."
        else:
            low, high = result.confidence_interval
            output = (f"'{plan.treatment}' vs '{plan.control}' on '{plan.outcome}': lift "
                      f"{result.effect:+.4f} (95% CI {low:+.4f} to {high:+.4f}); status '{result.status}'.")
        return (output, 90.0,
                {"status": result.status, "control_n": result.control_n, "treatment_n": result.treatment_n})


class ConformalCalibrator(Firefly):
    name = "conformal_calibrator"
    human_role = "Predictive uncertainty specialist"

    def perform(self):
        config = dict(self.ctx.blackboard.get("conformal_config") or {})
        task = config.get("task")
        coverage = float(config.get("target_coverage", 0.9))
        # The planner asks for calibration on every train mission; the Model
        # Scientist supplies the scores when the winner can provide them. When
        # it could not, say so rather than calibrating nothing.
        scores_key = "calibration_prediction" if task == "regression" else "calibration_probabilities"
        if scores_key not in config:
            reason = config.get("skipped_reason") or "no calibration scores were supplied"
            self.ctx.blackboard["conformal_result"] = None
            write_json(self.ctx.path("conformal_result.json"),
                       {"task": task, "target_coverage": coverage, "skipped": True, "reason": reason})
            return (f"Skipped conformal calibration: {reason}.", 5.0,
                    {"task": task, "target_coverage": coverage, "skipped": True})
        if task == "regression":
            result = split_conformal_regression(
                np.asarray(config["calibration_y"]), np.asarray(config["calibration_prediction"]),
                np.asarray(config["test_prediction"]), target_coverage=coverage,
                test_y=np.asarray(config["test_y"]) if "test_y" in config else None,
            )
        elif task == "classification":
            result = split_conformal_classification(
                np.asarray(config["calibration_y"]), np.asarray(config["calibration_probabilities"]),
                np.asarray(config["test_probabilities"]), target_coverage=coverage,
                test_y=np.asarray(config["test_y"]) if "test_y" in config else None,
            )
        else:
            raise ValueError("ConformalCalibrator requires task regression or classification")
        self.ctx.blackboard["conformal_result"] = result
        payload = asdict(result)
        payload["predictions"] = np.asarray(result.predictions, dtype=object).tolist()
        write_json(self.ctx.path("conformal_result.json"), payload)
        return (f"Calibrated {task} uncertainty at target coverage {coverage:.1%}.", 60.0,
                {"task": task, "target_coverage": coverage, "efficiency": result.efficiency})
