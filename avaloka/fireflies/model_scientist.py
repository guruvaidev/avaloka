"""Model Scientist — replaces the ML scientist.

Economic output: defensible baselines, a candidate panel and a selected model.
It trains every candidate the planner authorised (budget-bounded) behind a
leakage-resistant pipeline, persists the winner, and hands the full result set
to the Validator and FinOps fireflies. It never claims a model is good — it
reports what was measured.
"""

from __future__ import annotations

import pickle
from typing import Any

import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.modeling import train_candidates
from avaloka.util import write_json


class ModelScientist(Firefly):
    name = "model_scientist"
    human_role = "ML scientist"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        plan = self.ctx.blackboard["plan"]
        frame: pd.DataFrame = self.ctx.blackboard["transformed_frame"]
        target = self.ctx.target
        task = plan["task"]
        algorithms = plan["candidate_algorithms"]

        # Pass the planned split through so it is performed, not merely
        # reported. The datetime column comes from the profile rather than the
        # plan, because the plan records the strategy and not which column
        # orders it.
        split = plan.get("split", {}) or {}
        time_column = None
        if split.get("strategy") == "time_based_holdout":
            time_column = next(
                (c["name"] for c in self.ctx.blackboard["profile"]["columns"]
                 if c.get("role") == "datetime" and c["name"] in frame.columns),
                None,
            )
        outcome = train_candidates(
            frame, target, task,
            metric=self.ctx.metric, algorithms=algorithms,
            split_strategy=split.get("strategy"), time_column=time_column,
        )

        # Persist the winning pipeline.
        model_path = self.ctx.path("model", "model.pkl")
        with model_path.open("wb") as fh:
            pickle.dump(outcome.best.pipeline, fh)

        summary = {
            "task": outcome.task,
            "metric": outcome.metric,
            "baseline_score": outcome.baseline_score,
            "baseline_strategy": outcome.baseline_strategy,
            "beats_baseline": outcome.beats_baseline,
            "n_train": outcome.n_train,
            "n_test": outcome.n_test,
            "holdout_test_size": outcome.holdout_test_size,
            "feature_columns": outcome.X_columns,
            "candidates": [c.public() for c in outcome.candidates],
            "selected": outcome.best.name,
            "selected_metrics": outcome.best.metrics,
        }

        self.ctx.blackboard["training"] = outcome
        self.ctx.blackboard["training_summary"] = summary
        write_json(self.ctx.path("model", "experiments.json"), summary)

        # Effort scales with the number of candidates actually trained.
        manual = 180 + 90 * len(outcome.candidates)
        return (
            f"Trained {len(outcome.candidates)} candidate(s); selected "
            f"'{outcome.best.name}' ({outcome.metric}={outcome.best.primary_score:.4f}).",
            manual,
            {"selected": outcome.best.name, "score": round(outcome.best.primary_score, 4),
             "n_candidates": len(outcome.candidates)},
        )
