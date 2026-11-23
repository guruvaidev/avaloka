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

import numpy as np
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
            ensemble=bool(plan.get("compute_plan", {}).get("ensemble", False)),
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
            "selection_basis": outcome.selection_basis,
            "selected_metrics": outcome.best.metrics,
            "selected_classification": outcome.best.classification,
            "label_names": outcome.label_names,
            "classes": outcome.classes,
            "ensemble_vs_single": outcome.ensemble_vs_single,
        }

        self.ctx.blackboard["training"] = outcome
        self.ctx.blackboard["training_summary"] = summary
        write_json(self.ctx.path("model", "experiments.json"), summary)
        self._conformal_inputs(outcome)

        # Effort scales with the number of candidates actually trained.
        manual = 180 + 90 * len(outcome.candidates)
        if outcome.ensemble_vs_single:
            manual += 30  # the ensemble-versus-single comparison is its own judgement
        # Say which number chose the winner. The line used to quote the
        # winner's holdout score after a choice made on CV, so a reader saw
        # the ensemble "selected at 0.4887" beside a single at 0.5278.
        best = outcome.best
        if outcome.selection_basis == "cv":
            chosen = (f"on CV {outcome.metric} {best.cv_mean:.4f} "
                      f"(holdout {best.primary_score:.4f})")
        else:
            chosen = f"on holdout {outcome.metric} {best.primary_score:.4f} (CV failed)"
        return (
            f"Trained {len(outcome.candidates)} candidate(s); selected "
            f"'{best.name}' {chosen}.",
            manual,
            {"selected": outcome.best.name, "score": round(outcome.best.primary_score, 4),
             "n_candidates": len(outcome.candidates)},
        )


    def _conformal_inputs(self, outcome) -> None:
        """Complete the planner's ``conformal_config`` with the winner's holdout.

        The planner asks for a 90% interval (or label set) on every train
        mission; only the trained model can supply the calibration scores.
        The holdout is split in two: the first half calibrates the quantile,
        the second measures the coverage the calibrator then reports. Neither
        half was seen in fitting, which is the one assumption split conformal
        needs. When the winner cannot provide what the method needs (no class
        probabilities, too few rows) the reason is written instead of an
        array, and the calibrator says it skipped rather than guessing.
        """
        config = self.ctx.blackboard.get("conformal_config")
        if not isinstance(config, dict) or outcome.X_test is None:
            return
        X_test, y_test = outcome.X_test, np.asarray(outcome.y_test)
        n = len(y_test)
        if n < 4:
            config["skipped_reason"] = f"holdout has {n} rows; split conformal needs at least 4"
            return
        half = n // 2
        pipe = outcome.best.pipeline
        try:
            if outcome.task == "regression":
                pred = np.asarray(pipe.predict(X_test), dtype=float)
                config.update(task="regression",
                              calibration_y=y_test[:half].astype(float).tolist(),
                              calibration_prediction=pred[:half].tolist(),
                              test_prediction=pred[half:].tolist(),
                              test_y=y_test[half:].astype(float).tolist())
            else:
                if not hasattr(pipe, "predict_proba"):
                    config["skipped_reason"] = (f"'{outcome.best.name}' exposes no class "
                                                "probabilities; label sets need them")
                    return
                proba = np.asarray(pipe.predict_proba(X_test), dtype=float)
                # The calibrator indexes probability columns by label, so
                # each label becomes its column's position in ``classes_``.
                # A numeric target such as {1, 2, 3, 4} is not encoded, and
                # this used to require classes_ == [0..k-1] and skip it with
                # "class codes are not contiguous from zero".
                classes = list(getattr(pipe, "classes_", range(proba.shape[1])))
                column = {c: i for i, c in enumerate(classes)}
                if len(classes) != proba.shape[1] or any(c not in column for c in y_test):
                    config["skipped_reason"] = ("the holdout holds a class the model "
                                                "has no probability column for")
                    return
                y_pos = np.asarray([column[c] for c in y_test], dtype=int)
                name_of = dict(zip(outcome.classes or [], outcome.label_names or []))
                config.update(task="classification",
                              calibration_y=y_pos[:half].tolist(),
                              calibration_probabilities=proba[:half].tolist(),
                              test_probabilities=proba[half:].tolist(),
                              test_y=y_pos[half:].tolist(),
                              label_names=[str(name_of.get(c, c)) for c in classes])
        except Exception as exc:  # the card must not cost the training run
            config["skipped_reason"] = f"could not score the holdout: {exc}"
