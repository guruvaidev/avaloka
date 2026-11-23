"""Classification detail and the ensemble candidate in the CLI trainer.

The panel reported accuracy and F1 and nothing that could show *which* class
a classifier was missing; and it never tried the panel voting together, so
"ensemble versus single" was a question the bundle could not answer. Both
are measured here, and the ensemble is selected only when its gain over the
best single clears fold noise on the cross-validation interval.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from avaloka import modeling
from avaloka.modeling import classification_detail, ensemble_verdict, train_candidates


def _three_class_frame(n=450, seed=5):
    rng = np.random.default_rng(seed)
    colour = rng.choice(["red", "green", "blue"], n)
    centre = {"red": (0.0, 0.0), "green": (3.0, 0.0), "blue": (0.0, 3.0)}
    x = np.array([centre[c] for c in colour]) + rng.normal(0, 0.9, (n, 2))
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "colour": colour})


class TestClassificationDetail:
    def test_three_class_matrix_rows_sum_to_the_class_counts(self):
        """The selected classifier reports a labelled 3x3 matrix whose rows
        add up to each class's holdout count. A mutation that drops a class
        fails on the labels and on the sums."""
        outcome = train_candidates(_three_class_frame(), "colour", "multiclass_classification")
        assert outcome.label_names == ["blue", "green", "red"]
        counts = pd.Series(np.asarray(outcome.y_test)).value_counts()
        for candidate in outcome.candidates:
            table = candidate.classification["confusion_matrix"]
            assert table[0] == ["actual \\ predicted", "blue", "green", "red"]
            assert [row[0] for row in table[1:]] == ["blue", "green", "red"]
            for code, row in enumerate(table[1:]):
                assert len(row[1:]) == 3
                assert sum(row[1:]) == counts[code], f"{candidate.name}: row {row[0]}"
            assert set(candidate.classification["per_class"]) == {"blue", "green", "red"}
            assert 0.0 <= candidate.metrics["precision_macro"] <= 1.0
            assert 0.0 <= candidate.metrics["recall_macro"] <= 1.0

    def test_a_class_the_model_never_predicts_still_has_a_row(self):
        detail = classification_detail([0, 1, 2, 2], [0, 1, 1, 0], ["a", "b", "c"])
        assert detail["confusion_matrix"][3] == ["c", 1, 1, 0]
        assert detail["per_class"]["c"]["recall"] == 0.0
        assert detail["per_class"]["c"]["support"] == 2

    def test_regression_carries_no_matrix(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        frame["y"] = frame.a * 2 + frame.b + rng.normal(0, 0.1, 200)
        outcome = train_candidates(frame, "y", "regression")
        assert all(c.classification is None for c in outcome.candidates)
        assert outcome.label_names is None


class _Column(BaseEstimator, TransformerMixin):
    """Keep one column of the preprocessed matrix, so each candidate sees a
    different third of the signal and the vote adds up to more than any one."""

    def __init__(self, index=0):
        self.index = index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, [self.index]]


def _additive_frame(n=600, seed=11):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    frame["y"] = frame.x1 + frame.x2 + frame.x3 + rng.normal(0, 0.05, n)
    return frame


class TestEnsembleVersusSingle:
    def test_the_ensemble_is_one_more_candidate_on_the_real_panel(self):
        outcome = train_candidates(_three_class_frame(), "colour", "multiclass_classification")
        names = [c.name for c in outcome.candidates]
        assert names.count("ensemble") == 1
        ensemble = next(c for c in outcome.candidates if c.name == "ensemble")
        assert not np.isnan(ensemble.cv_mean)
        assert ensemble.classification["confusion_matrix"][0][1:] == ["blue", "green", "red"]
        verdict = outcome.ensemble_vs_single
        assert verdict["best_single"] in names and verdict["best_single"] != "ensemble"
        assert verdict["cv_winner"] in {"ensemble", "single", "tie"}
        assert verdict["selected"] in {"ensemble", "single"}
        assert (outcome.best.name == "ensemble") == (verdict["selected"] == "ensemble")

    def test_the_ensemble_wins_when_the_members_each_hold_a_third_of_the_signal(self, monkeypatch):
        """Three one-feature linear models on an additive target: each scores
        about a third of the variance, their average about five ninths, and
        the folds barely disagree. The verdict must say the ensemble won by
        more than fold noise and select it."""
        monkeypatch.setattr(modeling, "_estimators", lambda task: {
            f"only_x{i + 1}": make_pipeline(_Column(i), LinearRegression()) for i in range(3)
        })
        outcome = train_candidates(_additive_frame(), "y", "regression")
        verdict = outcome.ensemble_vs_single
        assert verdict["cv_winner"] == "ensemble"
        assert verdict["exceeds_fold_noise"] is True
        assert verdict["selected"] == "ensemble"
        assert verdict["cv_margin"] > 0.15
        assert outcome.best.name == "ensemble"
        assert "beats the best single" in verdict["reason"]

    def test_the_single_wins_when_one_member_is_exact_and_another_is_noise(self, monkeypatch):
        """Averaging an exact model with a mean-predictor halves the signal:
        the single is ahead by far more than fold noise and stays selected."""
        monkeypatch.setattr(modeling, "_estimators", lambda task: {
            "exact": LinearRegression(), "nothing": DummyRegressor(),
        })
        outcome = train_candidates(_additive_frame(), "y", "regression")
        verdict = outcome.ensemble_vs_single
        assert verdict["cv_winner"] == "single"
        assert verdict["exceeds_fold_noise"] is True
        assert verdict["selected"] == "single"
        assert verdict["best_single"] == "exact"
        assert outcome.best.name == "exact"
        assert "not selected" in verdict["reason"]

    def test_a_gain_inside_fold_noise_keeps_the_single(self):
        """The verdict reuses the baseline rule: the interval, not the mean."""
        ens = modeling.CandidateResult("ensemble", "ensemble", "r2", 0.80, {}, 0.80, 0.05, 1.0, 1.0)
        single = modeling.CandidateResult("rf", "rf", "r2", 0.78, {}, 0.78, 0.02, 1.0, 1.0)
        verdict = ensemble_verdict(ens, single, "r2")
        assert verdict["cv_winner"] == "ensemble"
        assert verdict["exceeds_fold_noise"] is False
        assert verdict["selected"] == "single"
        assert "fold noise" in verdict["reason"]
        # lower-is-better metrics flip the direction
        ens = modeling.CandidateResult("ensemble", "ensemble", "rmse", 1.0, {}, 1.0, 0.01, 1.0, 1.0)
        single = modeling.CandidateResult("rf", "rf", "rmse", 1.2, {}, 1.2, 0.01, 1.0, 1.0)
        assert ensemble_verdict(ens, single, "rmse")["selected"] == "ensemble"

    def test_a_single_candidate_has_no_ensemble(self):
        frame = _three_class_frame()
        outcome = train_candidates(frame, "colour", "multiclass_classification",
                                   algorithms=["logistic_regression"])
        assert [c.name for c in outcome.candidates] == ["logistic_regression"]
        assert outcome.ensemble_vs_single is None


class TestEnsembleGating:
    """The ensemble roughly doubles a train mission, so it is budgeted like
    one more candidate: the default path (no budget, no flag) skips it, a
    stated budget runs it only when it has room, and the user can force it
    either way. A mutation that ignores the budget fails the tiny-budget
    case; one that votes by default fails the default-path case."""

    def test_train_candidates_can_be_told_not_to_vote(self):
        outcome = train_candidates(_three_class_frame(), "colour", "multiclass_classification",
                                   ensemble=False)
        assert "ensemble" not in [c.name for c in outcome.candidates]
        assert outcome.ensemble_vs_single is None

    @staticmethod
    def _train(churn_csv, tmp_path, name, **kw):
        from avaloka.mission.budget import Budget
        from avaloka.mission.context import MissionContext, MissionKind
        from avaloka.missions import run_train
        out = tmp_path / name
        ctx = MissionContext(kind=MissionKind.TRAIN, goal="Predict churn", data_source=str(churn_csv),
                             output_dir=out, budget=Budget(limit_usd=kw.pop("budget", None)),
                             target="churned", metric="roc_auc", **kw)
        run_train(ctx)
        plan = json.loads((out / "planner_graph.json").read_text())["compute_plan"]
        summary = json.loads((out / "model" / "experiments.json").read_text())
        return plan, [c["name"] for c in summary["candidates"]], summary["ensemble_vs_single"]

    def test_a_budget_with_no_room_after_the_panel_skips_the_ensemble(self, churn_csv, tmp_path):
        # $0.11 pays for two candidates; the ensemble would be a third.
        plan, names, verdict = self._train(churn_csv, tmp_path, "tight", budget=0.11)
        assert plan["max_candidates"] == 2
        assert plan["ensemble"] is False and "budget" in plan["ensemble_reason"]
        assert "ensemble" not in names and verdict is None

    def test_the_default_path_skips_the_ensemble(self, churn_csv, tmp_path):
        # No --max-cost and no flag: what `avaloka train data.csv -t y` does.
        plan, names, verdict = self._train(churn_csv, tmp_path, "default")
        assert plan["max_candidates"] == 3
        assert plan["ensemble"] is False and "--ensemble" in plan["ensemble_reason"]
        assert "ensemble" not in names and verdict is None

    def test_a_budget_with_room_after_the_panel_votes(self, churn_csv, tmp_path):
        # $1.00 pays for twenty candidates: the panel of three plus the ensemble.
        plan, names, verdict = self._train(churn_csv, tmp_path, "roomy", budget=1.0)
        assert plan["ensemble"] is True and "room" in plan["ensemble_reason"]
        assert "ensemble" in names and verdict["reason"]

    def test_the_user_can_force_it_either_way(self, churn_csv, tmp_path):
        plan, names, _ = self._train(churn_csv, tmp_path, "off", ensemble=False)
        assert plan["ensemble"] is False and "--no-ensemble" in plan["ensemble_reason"]
        assert "ensemble" not in names
        plan, names, verdict = self._train(churn_csv, tmp_path, "on", budget=0.11, ensemble=True)
        assert plan["ensemble"] is True and "--ensemble" in plan["ensemble_reason"]
        assert "ensemble" in names and verdict["selected"] in {"ensemble", "single"}

    @pytest.mark.parametrize("flags, expected", [([], False), (["--no-ensemble"], False),
                                                 (["--ensemble"], True)])
    def test_the_cli_flag_reaches_the_plan(self, churn_csv, tmp_path, flags, expected):
        # [] is the default `avaloka train`: it must not pay for the ensemble.
        from typer.testing import CliRunner
        from avaloka.cli import app
        out = tmp_path / "cli"
        result = CliRunner().invoke(app, ["train", str(churn_csv), "--target", "churned",
                                          *flags, "--quiet", "--output", str(out)])
        assert result.exit_code == 0, result.output
        plan = json.loads((out / "planner_graph.json").read_text())["compute_plan"]
        assert plan["ensemble"] is expected
        names = [c["name"] for c in
                 json.loads((out / "model" / "experiments.json").read_text())["candidates"]]
        assert ("ensemble" in names) is expected


class TestBundle:
    def test_experiments_json_and_the_cards_carry_the_matrix_and_the_verdict(self, trained_package):
        summary = json.loads((trained_package / "model" / "experiments.json").read_text())
        assert summary["label_names"] == ["0", "1"]
        verdict = summary["ensemble_vs_single"]
        assert verdict["ensemble"] == "ensemble" and verdict["reason"]
        assert {"cv_winner", "exceeds_fold_noise", "selected", "ensemble_cv", "single_cv"} <= set(verdict)
        detail = summary["selected_classification"]
        assert detail["confusion_matrix"][0] == ["actual \\ predicted", "0", "1"]
        assert len(detail["confusion_matrix"]) == 3
        assert summary["selected_metrics"]["roc_auc"] > 0.5
        for candidate in summary["candidates"]:
            assert candidate["classification"]["labels"] == ["0", "1"]
            assert "precision_macro" in candidate["metrics"] and "recall_macro" in candidate["metrics"]
        card = (trained_package / "model_card.md").read_text()
        assert "Confusion matrix" in card and "Ensemble versus single" in card
        assert "| Class | Precision | Recall | Support |" in card
        html = (trained_package / "evaluation_report.html").read_text()
        assert "Confusion matrix" in html and "Ensemble versus single" in html
