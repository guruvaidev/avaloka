"""Explaining a model the agent just trained.

Avaloka could train a model and then not explain it: there was no shap, no
lime, no permutation importance and no coefficient anywhere in `app/`. When a
user asked *why* a prediction came out as it did, the honest answer was that
nothing in the system knew.
"""
import numpy as np
import pandas as pd
import pytest

from app.agents.mta_v2 import estimators, model_card

FEATURES = ["tenure", "monthly", "support_calls", "noise"]


def _frame(n=600, seed=0, regression=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n).astype(float),
        "monthly": rng.normal(70, 20, n),
        "support_calls": rng.poisson(2, n).astype(float),
        "noise": rng.normal(0, 1, n),          # deliberately carries no signal
    })
    signal = 0.05 * df.tenure - 0.04 * df.monthly + 0.6 * df.support_calls
    df["__label__"] = (signal + rng.normal(0, 0.5, n) if regression
                       else (signal + rng.normal(0, 1, n) > signal.mean()).astype(int))
    return df.iloc[:450], df.iloc[450:]


def _fitted(algorithm="gradient_boosting", regression=False):
    train, val = _frame(regression=regression)
    model_type = "regression" if regression else "classification"
    model, _, metrics = estimators.fit_sklearn(train, val, {
        "algorithm": algorithm, "feature_cols": FEATURES,
        "model_type": model_type, "random_seed": 42,
    })
    return model, train, val, metrics, model_type


class TestImportances:
    def test_finds_the_features_that_carry_the_signal(self):
        """`noise` is generated independently of the label, so a working
        measure must rank it below the three features that built it."""
        model, train, val, metrics, mt = _fitted()
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        ranked = [d["feature"] for d in card["importances"]]
        assert ranked, "no importances produced at all"
        assert ranked[0] in {"tenure", "support_calls", "monthly"}
        if "noise" in ranked:
            assert ranked.index("noise") == len(ranked) - 1

    def test_works_for_regression_too(self):
        model, train, val, metrics, mt = _fitted(regression=True)
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        assert card["importances"]

    def test_a_model_without_predict_does_not_raise(self):
        """A failed model card must never cost a successful training run."""
        card = model_card.build_model_card(
            object(), algorithm="mlp", model_type="classification",
            train_df=_frame()[0], val_df=_frame()[1],
            feature_cols=FEATURES, metrics={})
        assert card["importances"] == []
        assert isinstance(card["summary_text"], str)


class TestGlassboxBaseline:
    def test_fits_a_linear_model_and_a_shallow_tree(self):
        model, train, val, metrics, mt = _fitted()
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        baseline = card["baseline"]
        assert "linear" in baseline and "tree" in baseline
        assert baseline["linear"]["coefficients"]
        assert baseline["tree"]["rules"].strip()

    def test_linear_coefficients_recover_the_signs_of_the_signal(self):
        """tenure and support_calls push the label up, monthly pushes it down."""
        model, train, val, metrics, mt = _fitted()
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        signs = {c["feature"]: c["coefficient"] for c in card["baseline"]["linear"]["coefficients"]}
        assert signs["tenure"] > 0
        assert signs["support_calls"] > 0
        assert signs["monthly"] < 0

    def test_can_be_skipped(self):
        model, train, val, metrics, mt = _fitted()
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics,
            with_baseline=False)
        assert card["baseline"] == {}


class TestSummaryText:
    def test_fits_in_a_prompt(self):
        """This text exists to be carried in a conversation. A 40 kB dump of
        every coefficient would be dropped by whatever reads it next."""
        model, train, val, metrics, mt = _fitted()
        card = model_card.build_model_card(
            model, algorithm="gradient_boosting", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        assert 0 < len(card["summary_text"]) <= model_card.MAX_SUMMARY_CHARS

    def test_names_the_algorithm_and_the_score(self):
        model, train, val, metrics, mt = _fitted("random_forest")
        card = model_card.build_model_card(
            model, algorithm="random_forest", model_type=mt,
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=metrics)
        text = card["summary_text"]
        assert "random_forest" in text
        assert "Validation accuracy" in text

    def test_truncation_marks_itself(self):
        text = model_card.render_summary(
            algorithm="mlp", model_type="classification",
            metrics={"val_accuracy": 0.9, "val_f1_score": 0.9},
            importances=[{"feature": f"f{i}", "importance": 0.1} for i in range(8)],
            baseline={"tree": {"score_name": "accuracy", "score": 0.8,
                               "rules": "x\n" * 4000}},
            feature_count=8, row_count=100, max_chars=300)
        assert len(text) <= 300 + len("\n[summary truncated]")
        assert "truncated" in text

    def test_says_when_the_simple_model_was_nearly_as_good(self):
        """A model that barely beats a three-question tree is telling you the
        signal is simple, and that is worth saying out loud."""
        text = model_card.render_summary(
            algorithm="random_forest", model_type="classification",
            metrics={"val_accuracy": 0.805, "val_f1_score": 0.80},
            importances=[], baseline={"tree": {"score_name": "accuracy", "score": 0.80,
                                               "rules": ""}},
            feature_count=4, row_count=100)
        assert "simple" in text.lower()

    def test_says_when_the_complexity_earned_its_place(self):
        text = model_card.render_summary(
            algorithm="random_forest", model_type="classification",
            metrics={"val_accuracy": 0.95, "val_f1_score": 0.95},
            importances=[], baseline={"tree": {"score_name": "accuracy", "score": 0.60,
                                               "rules": ""}},
            feature_count=4, row_count=100)
        assert "earning its place" in text


class TestTorchModel:
    """The MLP is the one family with no ``predict``, so it needs an adapter.

    This path is where the interesting bug was: ``permutation_importance``
    demands ``.score()`` when no scoring is passed, a wrapped torch module has
    none, and the best-effort wrapper turned the resulting TypeError into an
    empty list. The MLP silently had no importances at all and nothing said so.
    """

    def _torch_bits(self):
        torch = pytest.importorskip("torch", reason="MLP path needs torch")
        np_ = pytest.importorskip("numpy")
        try:
            # torch built against numpy 1.x on a numpy 2.x runtime imports fine
            # and then fails at the boundary; skip rather than report a failure
            # that is the environment's, not the code's.
            torch.tensor(np_.zeros((2, 2)), dtype=torch.float32).numpy()
        except Exception as exc:                              # pragma: no cover
            pytest.skip(f"torch/numpy ABI mismatch in this environment: {exc}")
        return torch

    def test_permutation_importance_works_on_a_torch_module(self):
        torch = self._torch_bits()
        import torch.nn as nn

        train, val = _frame()

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

            def forward(self, x):
                return self.net(x)

        model = TinyMLP()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()
        X = torch.tensor(train[FEATURES].values, dtype=torch.float32)
        y = torch.tensor(train["__label__"].values, dtype=torch.long)
        for _ in range(120):
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

        card = model_card.build_model_card(
            model, algorithm="mlp", model_type="classification",
            train_df=train, val_df=val, feature_cols=FEATURES,
            metrics={"val_accuracy": 0.8, "val_f1_score": 0.8})

        assert card["importances"], (
            "the MLP produced no importances -- the torch adapter is broken "
            "again, and the best-effort wrapper is hiding it"
        )
        ranked = [d["feature"] for d in card["importances"]]
        assert ranked[0] in {"tenure", "support_calls", "monthly"}
