"""Saying what the model learned, in a form the conversational agent can read.

Avaloka could train a model and then not explain it. There is no shap, no lime,
no permutation importance and no coefficient anywhere in ``app/`` -- so when a
user asked *why* a prediction came out as it did, the honest answer was that
nothing in the system knew.

Two things here, and the cheap one matters most:

* A glass-box baseline fitted alongside the real model. A regularised linear
  model and a depth-3 tree cost a second or two and answer a question the
  primary model cannot: is any of this hard? A forest that barely beats a
  three-question tree is telling you the signal is simple, which is worth
  knowing before anyone ships a forest.

* A compact summary, deliberately sized for a prompt rather than a dashboard.
  The target is under 2 kB, because this text exists to be pasted into a
  conversation, and a 40 kB dump of every coefficient would be dropped by
  whatever reads it next.

Everything here is best-effort. A model card that raises is worse than no model
card, so every entry point returns something usable on failure and logs why.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Prompt-sized, not dashboard-sized. The whole point is that a conversational
#: agent can carry this in context and quote from it.
MAX_SUMMARY_CHARS = 2000

#: How many features are worth naming. Past a dozen the list stops being a
#: summary and starts being the data.
TOP_FEATURES = 8


def _safe(fn, what: str, default=None):
    try:
        return fn()
    except Exception as exc:
        logger.warning("Model card: %s unavailable (%s)", what, exc)
        return default


def permutation_importances(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    is_classification: bool,
    n_repeats: int = 5,
    random_state: int = 42,
) -> List[Dict[str, float]]:
    """Which features the model actually leans on, ranked.

    Permutation importance rather than anything model-specific, because it is
    the one measure that works for every family here including the MLP -- it
    only needs ``predict``. A torch module is wrapped so it looks like one.
    """
    def _run():
        from sklearn.inspection import permutation_importance
        estimator = model
        if not hasattr(model, "predict"):
            estimator = _torch_adapter(model, is_classification=is_classification)
        # Scoring is passed explicitly. Left to default, permutation_importance
        # demands the estimator expose .score(), which a wrapped torch module
        # does not -- it raised TypeError and the best-effort wrapper turned
        # that into an empty list, so the MLP silently had no importances at
        # all. Being explicit also means every family is ranked by the same
        # measure, which is the only way the numbers compare across them.
        result = permutation_importance(
            estimator, X, y,
            scoring="accuracy" if is_classification else "r2",
            n_repeats=n_repeats, random_state=random_state, n_jobs=1,
        )
        order = np.argsort(result.importances_mean)[::-1]
        return [
            {"feature": str(feature_names[i]),
             "importance": round(float(result.importances_mean[i]), 6),
             "std": round(float(result.importances_std[i]), 6)}
            for i in order[:TOP_FEATURES]
            if result.importances_mean[i] > 0
        ]
    return _safe(_run, "permutation importance", default=[]) or []


def _torch_adapter(module: Any, *, is_classification: bool):
    """Wrap a torch module so scikit-learn's inspection tools accept it.

    Built here rather than declared at module scope so it can inherit from
    sklearn's own base classes. That matters for portability: sklearn 1.9
    replaced the old ``_estimator_type`` attribute with a ``__sklearn_tags__``
    protocol, and an adapter that hand-rolls either one breaks on the other
    version. Inheriting gets both, plus a ``score`` method, so the default
    scoring path works as well as the explicit one.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

    mixin = ClassifierMixin if is_classification else RegressorMixin

    class _TorchPredictAdapter(mixin, BaseEstimator):
        def __init__(self, torch_module=None):
            self.torch_module = torch_module

        def fit(self, X, y=None):                            # pragma: no cover
            return self

        def __sklearn_is_fitted__(self) -> bool:
            # Already trained by the time a model card is built; without this
            # sklearn's check_is_fitted rejects it for having no trailing-
            # underscore attributes.
            return True

        def predict(self, X):
            import torch
            self.torch_module.eval()
            with torch.no_grad():
                out = self.torch_module(torch.tensor(np.asarray(X), dtype=torch.float32))
            arr = out.detach().cpu().numpy()
            if arr.ndim == 2 and arr.shape[1] > 1:
                return arr.argmax(axis=1)
            return arr.ravel()

    adapter = _TorchPredictAdapter(torch_module=module)
    if is_classification:
        adapter.classes_ = np.array([0, 1])
    return adapter


def fit_glassbox_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    *,
    is_classification: bool,
    random_state: int = 42,
) -> Dict[str, Any]:
    """A linear model and a shallow tree, fitted for comparison and for reading.

    Not a candidate model -- a yardstick. The number that matters is how close
    it gets to the primary model, because that difference is the only honest
    measure of what the complicated model bought.
    """
    def _run():
        from sklearn.metrics import accuracy_score, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

        out: Dict[str, Any] = {}

        if is_classification:
            from sklearn.linear_model import LogisticRegression
            linear = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=random_state),
            )
            score = accuracy_score
            score_name = "accuracy"
        else:
            from sklearn.linear_model import RidgeCV
            linear = make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))
            score = r2_score
            score_name = "r2"

        linear.fit(X_train, y_train)
        out["linear"] = {
            "score_name": score_name,
            "score": round(float(score(y_val, linear.predict(X_val))), 4),
            "coefficients": _linear_coefficients(linear, feature_names),
        }

        tree_cls = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
        tree = tree_cls(max_depth=3, random_state=random_state)
        tree.fit(X_train, y_train)
        out["tree"] = {
            "score_name": score_name,
            "score": round(float(score(y_val, tree.predict(X_val))), 4),
            "rules": _safe(
                lambda: export_text(tree, feature_names=list(feature_names), max_depth=3),
                "tree rules", default="",
            ),
        }
        return out

    return _safe(_run, "glass-box baseline", default={}) or {}


def _linear_coefficients(pipeline: Any, feature_names: Sequence[str]) -> List[Dict[str, float]]:
    """Signed standardised coefficients, largest magnitude first.

    Standardised, so the numbers are comparable across features measured in
    different units -- the sign is the direction, the magnitude is the strength.
    """
    def _run():
        est = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline
        coef = np.asarray(getattr(est, "coef_"))
        if coef.ndim > 1:
            coef = coef[0] if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
        order = np.argsort(np.abs(coef))[::-1]
        return [{"feature": str(feature_names[i]), "coefficient": round(float(coef[i]), 4)}
                for i in order[:TOP_FEATURES]]
    return _safe(_run, "linear coefficients", default=[]) or []


def render_summary(
    *,
    algorithm: str,
    model_type: str,
    metrics: Dict[str, Any],
    importances: Sequence[Dict[str, float]],
    baseline: Dict[str, Any],
    feature_count: int,
    row_count: int,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """The model card, as text, under ``max_chars``.

    Ordered by what a reader needs first: what was trained, how well it did,
    what it leaned on, and whether a far simpler model would have done. Trimmed
    from the bottom, so truncation costs the least important part.
    """
    is_classification = "classification" in str(model_type)
    headline = "accuracy" if is_classification else "val_loss"
    primary = metrics.get("val_accuracy") if is_classification else metrics.get("val_loss")

    lines: List[str] = [
        f"Model: {algorithm} ({model_type}), {feature_count} features, {row_count} rows.",
    ]
    if primary is not None:
        lines.append(f"Validation {headline}: {primary:.4f}"
                     + (f", weighted F1 {metrics['val_f1_score']:.4f}"
                        if is_classification and metrics.get("val_f1_score") is not None else "")
                     + (f", macro precision {metrics['val_precision_macro']:.4f}, "
                        f"macro recall {metrics['val_recall_macro']:.4f}"
                        if is_classification and metrics.get("val_precision_macro") is not None
                        and metrics.get("val_recall_macro") is not None else ""))
    lines.extend(_evaluation_lines(metrics))

    if importances:
        named = ", ".join(f"{d['feature']} ({d['importance']:.3f})" for d in importances[:5])
        lines.append(f"Most influential features, by permutation importance: {named}.")

    linear = baseline.get("linear") or {}
    if linear.get("coefficients"):
        top = ", ".join(
            f"{c['feature']} {'+' if c['coefficient'] >= 0 else '-'}{abs(c['coefficient']):.3f}"
            for c in linear["coefficients"][:5]
        )
        lines.append(f"A linear model scores {linear['score_name']} "
                     f"{linear['score']:.4f}; its strongest standardised "
                     f"coefficients are {top}.")

    tree = baseline.get("tree") or {}
    if tree.get("score") is not None:
        lines.append(f"A depth-3 decision tree scores {tree['score_name']} {tree['score']:.4f}.")
        if primary is not None and is_classification:
            gap = primary - float(tree["score"])
            lines.append(
                f"The trained model is {gap:+.4f} {headline} against that tree — "
                + ("most of the signal is simple, and the simpler model may be the "
                   "better answer." if gap < 0.02 else
                   "the extra complexity is earning its place.")
            )

    rules = (tree.get("rules") or "").strip()
    if rules:
        lines.append("Tree rules:\n" + rules)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit("\n", 1)[0]
    return trimmed + "\n[summary truncated]"


def _evaluation_lines(metrics: Dict[str, Any]) -> List[str]:
    """How the score was validated, and the confusion matrix for a classifier.

    A cross-validated mean with its spread and the fold strategy when the
    trainer k-folded; "one holdout" when it did not, so the reader is never
    left to assume. The matrix is written in full for up to ten classes --
    past that it stops being a summary.
    """
    lines: List[str] = []
    cv = metrics.get("cross_validation")
    strategy = metrics.get("evaluation_strategy")
    if isinstance(cv, dict) and cv.get("cv_mean") is not None:
        primary = cv.get("metrics", {}).get(cv.get("primary_metric"), {}) or {}
        ci = ""
        if primary.get("ci_low") is not None and primary.get("ci_high") is not None:
            ci = f" (95% CI {primary['ci_low']:.4f}–{primary['ci_high']:.4f})"
        lines.append(
            f"Cross-validation: {cv.get('primary_metric')} {cv['cv_mean']:.4f} "
            f"± {cv.get('cv_std') or 0.0:.4f}{ci} over {cv.get('n_folds')}-fold "
            f"{cv.get('splitter')} on the training split"
            + (f"; {cv['splitter_reason']}" if cv.get("splitter_reason") else "") + "."
        )
        if cv.get("beats_baseline") is False:
            lines.append("The folds do not beat the trivial baseline on the confidence interval.")
    elif strategy:
        lines.append(f"Validation: one {strategy} split; the score above was not cross-validated.")

    detail = metrics.get("classification")
    if isinstance(detail, dict) and detail.get("confusion_matrix"):
        labels = list(detail.get("labels") or [])
        if len(labels) <= 10:
            rows = detail["confusion_matrix"][1:]
            body = "; ".join(f"{row[0]}: {list(row[1:])}" for row in rows)
            lines.append(f"Confusion matrix (rows actual, columns predicted {labels}): {body}.")
            per_class = detail.get("per_class") or {}
            named = ", ".join(f"{name} P={pc['precision']:.3f} R={pc['recall']:.3f}"
                              for name, pc in per_class.items())
            if named:
                lines.append(f"Per-class precision/recall: {named}.")
        else:
            lines.append(f"Confusion matrix over {len(labels)} classes is in the evaluation report.")
        if detail.get("roc_auc") is not None:
            lines.append(f"ROC AUC: {detail['roc_auc']:.4f}.")
    return lines


def build_model_card(
    model: Any,
    *,
    algorithm: str,
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    metrics: Dict[str, Any],
    random_state: int = 42,
    with_baseline: bool = True,
) -> Dict[str, Any]:
    """Everything explainable about a finished model. Never raises."""
    is_classification = "classification" in str(model_type)
    try:
        X_train = train_df[list(feature_cols)].to_numpy(dtype=float)
        X_val = val_df[list(feature_cols)].to_numpy(dtype=float)
        y_train = train_df["__label__"].to_numpy()
        y_val = val_df["__label__"].to_numpy()
        if is_classification:
            y_train, y_val = y_train.astype(int), y_val.astype(int)
    except Exception as exc:
        logger.warning("Model card: could not assemble matrices (%s)", exc)
        return {"summary_text": "", "importances": [], "baseline": {}}

    importances = permutation_importances(
        model, X_val, y_val, list(feature_cols),
        is_classification=is_classification, random_state=random_state,
    )
    baseline = (fit_glassbox_baseline(
        X_train, y_train, X_val, y_val, list(feature_cols),
        is_classification=is_classification, random_state=random_state,
    ) if with_baseline else {})

    summary = render_summary(
        algorithm=algorithm, model_type=model_type, metrics=metrics or {},
        importances=importances, baseline=baseline,
        feature_count=len(feature_cols), row_count=len(train_df),
    )
    metrics = metrics or {}
    return {
        "summary_text": summary, "importances": importances, "baseline": baseline,
        # Structured, not only prose: the fold report and the labelled matrix
        # as the trainer produced them, for anything that reads JSON.
        "evaluation": {
            "strategy": metrics.get("evaluation_strategy") or "holdout",
            "cross_validation": metrics.get("cross_validation"),
            "classification": metrics.get("classification"),
        },
    }
