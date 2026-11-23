"""One selection rule for every candidate, and a matrix labelled by class.

The e2e run on nyc_taxi (``train --ensemble``, target ``payment_type``)
selected the ensemble at holdout f1_macro 0.4887 and reported that it "beats
the best single 'logistic_regression' (0.5133)". The best single had been
picked on the holdout (0.5278) and then compared with the ensemble on CV,
where gradient boosting (0.566) and random forest (0.555) both beat the
ensemble's 0.534; cost_quality.json, ranking on the holdout, recommended
logistic regression without saying why. The model card's matrix read
"2 | 3 | 4 | 4" for classes 1-4, and conformal calibration skipped the same
target as "not contiguous from zero".
"""

import json

import numpy as np
import pandas as pd
import pytest

from avaloka import modeling
from avaloka.modeling import classification_detail, train_candidates

# The nyc_taxi leaderboard on base/2.0-stack (ce126765), reproduced from
# `avaloka train yellow_tripdata_2015-01_dataset_50k.csv --target payment_type
# --ensemble`: holdout f1_macro, CV mean.
TAXI = {
    "logistic_regression": (0.5278, 0.5133),
    "random_forest": (0.4887, 0.5548),
    "gradient_boosting": (0.5226, 0.5660),
    "ensemble": (0.4887, 0.5343),
}


def _four_class_frame(n=480, seed=3, labels=(1, 2, 3, 4)):
    rng = np.random.default_rng(seed)
    y = rng.choice(list(labels), n)
    centre = {lab: (2.5 * (i % 2), 2.5 * (i // 2)) for i, lab in enumerate(labels)}
    x = np.array([centre[v] for v in y]) + rng.normal(0, 1.0, (n, 2))
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})


def _scripted(monkeypatch, scores, std=0.005):
    """Train the real panel, then overwrite each candidate's holdout and CV
    score with ``scores[name]`` -- the selection logic is what is under test,
    and real estimators do not produce a chosen ordering on demand."""
    real = modeling._evaluate_candidate

    def fake(name, *args, **kwargs):
        result = real(name, *args, **kwargs)
        holdout, cv = scores[name]
        result.primary_score, result.cv_mean, result.cv_std = holdout, cv, std
        result.metrics[result.primary_metric] = holdout
        return result

    monkeypatch.setattr(modeling, "_evaluate_candidate", fake)


class TestOneSelectionRule:
    def test_a_single_that_beats_the_ensemble_on_cv_is_selected(self, monkeypatch):
        """(a) The taxi ordering. Ranking the singles on the holdout made
        logistic regression the "best single"; the ensemble's CV beat *its*
        CV and was selected. On one rule gradient boosting is best."""
        _scripted(monkeypatch, TAXI)
        outcome = train_candidates(_four_class_frame(), "y", "multiclass_classification")
        verdict = outcome.ensemble_vs_single
        assert outcome.selection_basis == "cv"
        assert verdict["best_single"] == "gradient_boosting"
        assert verdict["selected"] == "single"
        assert outcome.best.name == "gradient_boosting"

    def test_an_ensemble_that_beats_every_single_on_cv_is_selected(self, monkeypatch):
        """(b) The ensemble clears the CV-best single by more than fold
        noise -- even though a single has the best holdout -- and wins."""
        scores = dict(TAXI, ensemble=(0.5000, 0.6200))
        _scripted(monkeypatch, scores)
        outcome = train_candidates(_four_class_frame(), "y", "multiclass_classification")
        verdict = outcome.ensemble_vs_single
        assert verdict["best_single"] == "gradient_boosting"
        assert verdict["exceeds_fold_noise"] is True
        assert verdict["selected"] == "ensemble"
        assert outcome.best.name == "ensemble"

    def test_every_candidate_is_scored_on_the_same_folds(self, monkeypatch):
        seen = []
        real = modeling.cross_val_score

        def spy(estimator, X, y, cv=None, **kw):
            seen.append(cv)
            return real(estimator, X, y, cv=cv, **kw)

        monkeypatch.setattr(modeling, "cross_val_score", spy)
        train_candidates(_four_class_frame(), "y", "multiclass_classification")
        assert len(seen) == 4 and all(cv is seen[0] for cv in seen)
        # The rarest of {1..4} has ~90 training rows, so five folds, not the
        # two that bincount's phantom class 0 forced.
        assert seen[0].get_n_splits() == 5


class TestReportedNumbersAreTheLeaderboard:
    """(c) The verdict, the scientist's line and cost_quality.json quote the
    leaderboard's own numbers, and cost_quality ranks as the selection does."""

    @pytest.fixture
    def bundle(self, tmp_path, monkeypatch):
        from avaloka.mission.budget import Budget
        from avaloka.mission.context import MissionContext, MissionKind
        from avaloka.missions import run_train

        _scripted(monkeypatch, TAXI)
        data = tmp_path / "taxi_like.csv"
        _four_class_frame().rename(columns={"y": "payment_type"}).to_csv(data, index=False)
        out = tmp_path / "out"
        ctx = MissionContext(kind=MissionKind.TRAIN, goal="Predict payment type",
                             data_source=str(data), output_dir=out, budget=Budget(limit_usd=None),
                             target="payment_type", metric="f1_macro", ensemble=True)
        run_train(ctx)
        return out, ctx

    def test_the_comparison_pair_is_the_leaderboard_pair(self, bundle):
        out, ctx = bundle
        summary = json.loads((out / "model" / "experiments.json").read_text())
        board = {c["name"]: c for c in summary["candidates"]}
        verdict = summary["ensemble_vs_single"]
        singles = [c for c in summary["candidates"] if c["name"] != "ensemble"]
        assert verdict["best_single"] == max(singles, key=lambda c: c["cv_mean"])["name"]
        assert verdict["ensemble_cv"]["mean"] == board["ensemble"]["cv_mean"]
        assert verdict["single_cv"]["mean"] == board[verdict["best_single"]]["cv_mean"]
        assert f"{board['gradient_boosting']['cv_mean']:.4f}" in verdict["reason"]
        assert f"{board['ensemble']['cv_mean']:.4f}" in verdict["reason"]
        assert summary["selected"] == "gradient_boosting"

        line = next(u.output for u in ctx.ledger.units if u.firefly == "model_scientist")
        assert "selected 'gradient_boosting' on CV f1_macro 0.5660 (holdout 0.5226)" in line

    def test_cost_quality_ranks_on_the_same_numbers_and_explains_a_difference(self, bundle):
        out, _ = bundle
        cq = json.loads((out / "cost_quality.json").read_text())
        for row in cq["all_candidates"]:
            assert row["score_basis"] == "cv"
            assert row["validation_score"] == round(TAXI[row["candidate"]][1], 4)
            assert row["holdout_score"] == round(TAXI[row["candidate"]][0], 4)
        assert cq["options"]["Maximum quality"]["candidate"] == "gradient_boosting"
        rec = cq["recommendation"]
        assert rec["selected"] == "gradient_boosting"
        assert rec["agrees_with_selection"] is (rec["candidate"] == "gradient_boosting")
        if not rec["agrees_with_selection"]:
            assert "Selection picks 'gradient_boosting'" in rec["selection_note"]
            assert f"'{rec['candidate']}'" in rec["selection_note"]

    def test_the_card_labels_classes_one_to_four_and_calibration_runs(self, bundle):
        """(d) end to end, plus the conformal skip on the same target."""
        out, _ = bundle
        card = (out / "model_card.md").read_text()
        assert "| actual \\ predicted | 1 | 2 | 3 | 4 |" in card
        rows = [line.split(" | ")[0].strip("| ") for line in card.splitlines()
                if line.startswith("| ") and line.count("|") == 6 and "predicted" not in line]
        assert rows == ["1", "2", "3", "4"]
        conformal = json.loads((out / "conformal_result.json").read_text())
        assert not conformal.get("skipped"), conformal.get("reason")


class TestMatrixLabels:
    """(d) Labels come from the class values, in order."""

    @pytest.mark.parametrize("labels", [(1, 2, 3, 4), (2, 5, 7), ("a", "b")])
    def test_trained_matrix_headers_are_the_classes(self, labels):
        task = "binary_classification" if len(labels) == 2 else "multiclass_classification"
        outcome = train_candidates(_four_class_frame(labels=labels), "y", task, ensemble=False)
        names = [str(v) for v in labels]
        counts = pd.Series(np.asarray(outcome.y_test)).value_counts()
        for c in outcome.candidates:
            table = c.classification["confusion_matrix"]
            assert table[0] == ["actual \\ predicted", *names]
            assert [row[0] for row in table[1:]] == names
            for code, row in zip(outcome.classes, table[1:]):
                assert sum(row[1:]) == counts[code]
            assert list(c.classification["per_class"]) == names

    def test_detail_with_values_reads_names_by_value(self):
        detail = classification_detail([1, 2, 3, 4, 4], [1, 2, 4, 4, 3],
                                       ["1", "2", "3", "4"], classes=[1, 2, 3, 4])
        assert detail["confusion_matrix"][0] == ["actual \\ predicted", "1", "2", "3", "4"]
        assert detail["confusion_matrix"][4] == ["4", 0, 0, 1, 1]
