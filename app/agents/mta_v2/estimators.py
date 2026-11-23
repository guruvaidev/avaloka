"""Model families other than the MLP.

Until this existed the training agent had exactly one model: ``DynamicMLP``.
``TrainingPlan`` carried no field naming an algorithm -- ``model_type`` is
``"classification" | "regression"``, which is the *task* -- and
``_execute_training`` chose only a backend, local or Ray, never a family. So a
user asking Avaloka to predict a column got a neural network whether or not one
was a sensible answer, and on tabular data it usually is not: gradient-boosted
trees beat an MLP on most tables, which is why they are what a data scientist
reaches for first.

This module supplies the other families and keeps them behind the same
``(model, history, final_metrics)`` contract ``train_local`` already returns, so
nothing downstream has to learn a second shape. What downstream *does* have to
learn is that a model is not always a torch module -- see :func:`is_torch_model`
and the branching in ``local_trainer`` around saving and MLflow logging.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Algorithm = Literal["mlp", "gradient_boosting", "random_forest", "linear"]

#: Every family the planner may choose. "mlp" stays first and stays the default:
#: it is what every existing plan means, and a plan written before this field
#: existed must keep training the same model it trained yesterday.
ALGORITHMS: Tuple[str, ...] = ("mlp", "gradient_boosting", "random_forest", "linear")

DEFAULT_ALGORITHM = "mlp"

#: The dtype a non-MLP family is fitted and validated on (``to_numpy(dtype=...)``
#: below). The logged signature takes the same dtype (model_signature), because
#: MLflow refuses pyfunc input whose dtype differs from the signature's.
SKLEARN_FEATURE_DTYPE = "float64"

#: Hyperparameters that mean something to each family. The planner emits only
#: these for the family it picked; sending MLP keys to a random forest is how
#: you end up with a plan that reads as if it configured something it did not.
FAMILY_HYPERPARAMETERS: Dict[str, Tuple[str, ...]] = {
    "mlp": ("learning_rate", "epochs", "batch_size", "optimizer_name", "random_seed",
            "early_stopping_patience", "hidden_layer_sizes", "activation",
            "dropout_rate", "batch_norm"),
    "gradient_boosting": ("learning_rate", "max_iter", "max_depth", "max_leaf_nodes",
                          "l2_regularization", "early_stopping_patience", "random_seed"),
    "random_forest": ("n_estimators", "max_depth", "min_samples_leaf",
                      "max_features", "random_seed"),
    "linear": ("alphas", "max_iter", "random_seed"),
}

#: The MLP's defaults: the template the planner extracts hyperparameters
#: from, and what LocalTrainer fills in for an MLP plan that leaves a key out.
MLP_DEFAULTS: Dict[str, Any] = {
    "learning_rate": 0.001, "epochs": 30, "batch_size": 32, "optimizer_name": "adam",
    "random_seed": 42, "early_stopping_patience": 5, "hidden_layer_sizes": [128, 64],
    "activation": "relu", "dropout_rate": 0.2, "batch_norm": True,
}

#: Every other family's own defaults: scikit-learn's (learning rate 0.1,
#: early stopping after 10 rounds without improvement), with 200 boosting
#: rounds and 300 trees as before. Never the MLP's.
FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gradient_boosting": {"learning_rate": 0.1, "max_iter": 200, "max_depth": None,
                          "max_leaf_nodes": 31, "l2_regularization": 0.0,
                          "early_stopping_patience": 10, "random_seed": 42},
    "random_forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                      "max_features": "sqrt", "random_seed": 42},
    "linear": {"alphas": [0.1, 1.0, 10.0, 100.0], "max_iter": 1000, "random_seed": 42},
}

#: Why a human would pick each one. Carried into the plan so the reply can say
#: something truer than "a model was trained".
FAMILY_RATIONALE: Dict[str, str] = {
    "mlp": "a neural network; useful when features interact in ways trees "
           "approximate badly, and when you expect to fine-tune it later",
    "gradient_boosting": "gradient-boosted trees; the usual strongest baseline "
                         "on tabular data, handles mixed scales without fuss",
    "random_forest": "a random forest; close to boosting in accuracy, harder to "
                     "overfit, and its feature importances are easy to explain",
    "linear": "a regularised linear model; the most interpretable option and a "
              "fair yardstick -- if it matches the others, the signal is simple",
}


#: Types the skops audit flags in models these families produce, and that we
#: trust because training produced them in this process. mlflow.sklearn saves
#: in skops format and reloads the file through the audit before logging it;
#: since skops 0.16 the audit rejects both of these (out-of-range node indices
#: in a hostile file can crash .predict()), so every gradient-boosting and
#: random-forest training failed in a freshly built image while CI, on skops
#: 0.15, passed. The list is passed as ``skops_trusted_types``, MLflow's
#: documented switch, which also records it in the MLmodel for loading. The
#: audit stays on: any type outside this list still fails the save, and
#: tests/test_mta_skops_trust.py fails first, naming the family that needs it.
SKOPS_TRUSTED_TYPES: Tuple[str, ...] = (
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",  # gradient_boosting
    "sklearn.tree._tree.Tree",                                           # random_forest
)


def normalise_algorithm(value: Any) -> str:
    """Coerce whatever the planner emitted into a family we implement.

    An unknown name falls back to the default rather than raising: a plan that
    names a model we do not have should still train something and say what it
    trained, not fail at the last step of a long job.
    """
    name = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "neural_network": "mlp", "nn": "mlp", "dynamicmlp": "mlp", "mlp": "mlp",
        "gbm": "gradient_boosting", "gbt": "gradient_boosting",
        "xgboost": "gradient_boosting", "lightgbm": "gradient_boosting",
        "boosting": "gradient_boosting", "hist_gradient_boosting": "gradient_boosting",
        "forest": "random_forest", "rf": "random_forest",
        "ridge": "linear", "logistic": "linear", "linear_model": "linear",
    }
    resolved = aliases.get(name, name if name in ALGORITHMS else "")
    if not resolved:
        if name:
            logger.warning("Unknown algorithm %r; training %s instead.", value, DEFAULT_ALGORITHM)
        return DEFAULT_ALGORITHM
    return resolved


def is_torch_model(model: Any) -> bool:
    """Whether this model is a torch module.

    Used at every point that previously assumed one: saving .pth/.onnx, and
    ``mlflow.pytorch.log_model``. Imports torch lazily so a sklearn-only path
    does not pay for it.
    """
    try:
        import torch.nn as nn
    except Exception:                                        # pragma: no cover
        return False
    return isinstance(model, nn.Module)


def _int(config: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _float(config: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def build_estimator(algorithm: str, model_type: str, config: Dict[str, Any]):
    """The scikit-learn estimator for a family. Never called for ``"mlp"``."""
    is_classification = "classification" in str(model_type)
    seed = _int(config, "random_seed", 42)

    if algorithm == "gradient_boosting":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        kwargs = dict(
            learning_rate=_float(config, "learning_rate", 0.1),
            # Not ``epochs``: LocalTrainer always passes the MLP's 30, which
            # made every boosted model stop at 30 rounds. An epochs request
            # meant for this family is mapped in family_hyperparameters.
            max_iter=_int(config, "max_iter", FAMILY_DEFAULTS["gradient_boosting"]["max_iter"]),
            max_depth=config.get("max_depth") or None,
            max_leaf_nodes=_int(config, "max_leaf_nodes", 31),
            l2_regularization=_float(config, "l2_regularization", 0.0),
            early_stopping=True,
            n_iter_no_change=_int(config, "early_stopping_patience", 10),
            random_state=seed,
        )
        return (HistGradientBoostingClassifier(**kwargs) if is_classification
                else HistGradientBoostingRegressor(**kwargs))

    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        kwargs = dict(
            n_estimators=_int(config, "n_estimators", 300),
            max_depth=config.get("max_depth") or None,
            min_samples_leaf=_int(config, "min_samples_leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=seed,
        )
        return (RandomForestClassifier(**kwargs) if is_classification
                else RandomForestRegressor(**kwargs))

    if algorithm == "linear":
        # Scaled, because a penalised linear model on raw columns penalises
        # whichever feature happens to be measured in small units.
        from sklearn.linear_model import LogisticRegressionCV, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if is_classification:
            est = LogisticRegressionCV(
                max_iter=_int(config, "max_iter", 1000), n_jobs=-1, random_state=seed,
            )
        else:
            est = RidgeCV(alphas=config.get("alphas") or (0.1, 1.0, 10.0, 100.0))
        return make_pipeline(StandardScaler(), est)

    raise ValueError(f"no estimator for algorithm {algorithm!r}")


_FAMILY_ONLY_KEYS = ("max_iter", "max_depth", "max_leaf_nodes", "l2_regularization",
                     "n_estimators", "min_samples_leaf", "max_features", "alphas")


def default_for(algorithm: Any, key: str) -> Any:
    """The value ``key`` takes for this family when nobody set it."""
    family = normalise_algorithm(algorithm)
    defaults = MLP_DEFAULTS if family == "mlp" else FAMILY_DEFAULTS.get(family, {})
    return defaults.get(key)


def family_hyperparameters(algorithm: Any, hyperparameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The plan's hyperparameters, made the chosen family's own.

    The planner extracts hyperparameters from a template of MLP defaults
    before it picks a family, so a gradient-boosted plan arrived carrying
    ``learning_rate=0.001`` and ``epochs=30`` and trained with both: thirty
    boosting rounds at a hundredth of the usual rate, which on rc3's stroke
    and diabetes tasks never beat the majority-class baseline. A value equal
    to the MLP template's, or missing, takes the family's own default; any
    other value is an explicit choice and is kept. Keys the family does not
    use are dropped (FAMILY_HYPERPARAMETERS). An MLP plan is returned as is.
    """
    family = normalise_algorithm(algorithm)
    hp = dict(hyperparameters or {})
    if family == "mlp":
        return hp
    out: Dict[str, Any] = {}
    for key in FAMILY_HYPERPARAMETERS[family]:
        value = hp.get(key)
        if value is None or (key in MLP_DEFAULTS and value == MLP_DEFAULTS[key]):
            value = FAMILY_DEFAULTS[family].get(key)
        out[key] = value
    epochs = hp.get("epochs")
    if (family == "gradient_boosting" and hp.get("max_iter") is None
            and epochs is not None and epochs != MLP_DEFAULTS["epochs"]):
        out["max_iter"] = epochs           # "train for 100 epochs" means 100 rounds here
    return out


def plan_hyperparameters(algorithm: Any, model_type: str,
                         hyperparameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The hyperparameters this family will really train with, for the plan text.

    The plan once read "30 epochs, Adam optimizer" for a gradient-boosted
    model, and "None epochs, None optimizer" when the plan carried only the
    family's own keys. This builds the config the way LocalTrainer does and,
    for a scikit-learn family, reads the values back off the estimator
    ``build_estimator`` makes, so the text cannot drift from what trains.
    No value in the result is None: an unlimited depth is ``"no limit"``.
    """
    family = normalise_algorithm(algorithm)
    hp = dict(hyperparameters or {})
    if family == "mlp":
        def _or_default(key: str) -> Any:
            value = hp.get(key)
            return MLP_DEFAULTS[key] if value is None else value
        return {
            "algorithm": "mlp",
            "epochs": _or_default("epochs"),
            "optimizer": _or_default("optimizer_name"),
            "learning_rate": _or_default("learning_rate"),
            "batch_size": _or_default("batch_size"),
        }

    # As LocalTrainer builds it: a shared key the plan leaves out takes the
    # family's default, not the MLP's.
    config = {key: hp.get(key, default_for(family, key))
              for key in ("learning_rate", "early_stopping_patience", "random_seed")}
    config.update({key: hp[key] for key in _FAMILY_ONLY_KEYS if hp.get(key) is not None})
    estimator = build_estimator(family, model_type, config)
    if hasattr(estimator, "steps"):          # linear: scaler + model pipeline
        estimator = estimator.steps[-1][1]
    params = estimator.get_params()

    def _depth(value: Any) -> Any:
        return "no limit" if value is None else value

    if family == "gradient_boosting":
        return {
            "algorithm": family,
            "estimators": params["max_iter"],
            "max_depth": _depth(params["max_depth"]),
            "learning_rate": params["learning_rate"],
        }
    if family == "random_forest":
        return {
            "algorithm": family,
            "estimators": params["n_estimators"],
            "max_depth": _depth(params["max_depth"]),
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"] if params["max_features"] is not None else "all",
        }
    if "max_iter" in params:                 # LogisticRegressionCV
        return {"algorithm": family, "max_iter": params["max_iter"]}
    return {"algorithm": family, "alphas": list(params["alphas"])}


def _metrics(y_true, y_pred, is_classification: bool) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                                 precision_score, recall_score)
    if is_classification:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            # Macro, so a class the model never predicts pulls the number
            # down; weighted F1 and accuracy both let the majority hide it.
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "loss": float(1.0 - accuracy_score(y_true, y_pred)),
        }
    mse = float(mean_squared_error(y_true, y_pred))
    return {"accuracy": 0.0, "f1_score": 0.0, "loss": mse}


def classification_detail(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    y_score=None,
) -> Dict[str, Any]:
    """The confusion matrix and per-class precision/recall, labelled.

    Rows are the actual class, columns the predicted one, and each row sums
    to that class's validation count -- the check a reader can do by eye. The
    first row and column carry the class names so the JSON reads on its own.
    Every class present in the truth or the predictions gets a row: a class
    the model never predicted is a row of misses, not an omission.

    ``y_score`` is the positive-class probability for a binary task; ROC AUC
    is reported only when it is given and both classes occur.
    """
    from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                                 roc_auc_score)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    codes = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    names = [str(class_names[int(c)]) if class_names and 0 <= int(c) < len(class_names)
             else str(c) for c in codes]
    matrix = confusion_matrix(y_true, y_pred, labels=codes)
    precision = precision_score(y_true, y_pred, labels=codes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=codes, average=None, zero_division=0)
    table: List[List[Any]] = [["actual \\ predicted", *names]]
    for name, row in zip(names, matrix):
        table.append([name, *[int(v) for v in row]])
    detail: Dict[str, Any] = {
        "labels": names,
        "confusion_matrix": table,
        "per_class": {
            name: {"precision": round(float(p), 6), "recall": round(float(r), 6),
                   "support": int(matrix[i].sum())}
            for i, (name, p, r) in enumerate(zip(names, precision, recall))
        },
        "precision_macro": round(float(np.mean(precision)), 6) if len(codes) else 0.0,
        "recall_macro": round(float(np.mean(recall)), 6) if len(codes) else 0.0,
        "roc_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            detail["roc_auc"] = round(float(roc_auc_score(y_true, np.asarray(y_score, dtype=float))), 6)
        except ValueError:
            pass
    return detail


def predict_with_scores(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Class labels and, where the model has them, class probabilities.

    One call for every family: a torch module is run forward and softmaxed,
    a scikit-learn estimator answers through ``predict``/``predict_proba``.
    """
    if is_torch_model(model):
        import torch
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(np.asarray(X), dtype=torch.float32))
        logits = out.detach().cpu().numpy()
        if logits.ndim == 2 and logits.shape[1] > 1:
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            return logits.argmax(axis=1), shifted / shifted.sum(axis=1, keepdims=True)
        return logits.ravel(), None
    labels = np.asarray(model.predict(X))
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X), dtype=float)
        except Exception:                                    # pragma: no cover
            proba = None
    return labels, proba


def validation_detail(
    model: Any,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """The classification detail on the validation split, or None for regression."""
    if "classification" not in str(config.get("model_type")):
        return None
    X_val = val_df[config["feature_cols"]].to_numpy(dtype=SKLEARN_FEATURE_DTYPE)
    y_val = val_df["__label__"].to_numpy().astype(int)
    y_pred, proba = predict_with_scores(model, X_val)
    y_score = proba[:, 1] if proba is not None and proba.ndim == 2 and proba.shape[1] == 2 else None
    return classification_detail(y_val, y_pred, config.get("class_names"), y_score)


def fit_sklearn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    """Fit a non-MLP family and return train_local's exact contract.

    ``history`` is a single entry rather than one per epoch. These families do
    not train in epochs, and inventing a curve for them would put a shape in
    the UI that means nothing. One honest row beats thirty fabricated ones.
    """
    algorithm: str = config["algorithm"]
    feature_cols: List[str] = config["feature_cols"]
    model_type: str = config["model_type"]
    is_classification = "classification" in model_type

    X_train = train_df[feature_cols].to_numpy(dtype=SKLEARN_FEATURE_DTYPE)
    X_val = val_df[feature_cols].to_numpy(dtype=SKLEARN_FEATURE_DTYPE)
    y_train = train_df["__label__"].to_numpy()
    y_val = val_df["__label__"].to_numpy()
    if is_classification:
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)

    model = build_estimator(algorithm, model_type, config)
    logger.info("[Train] Fitting %s (%s) on %d rows x %d features",
                type(model).__name__, algorithm, len(X_train), len(feature_cols))
    model.fit(X_train, y_train)

    train_m = _metrics(y_train, model.predict(X_train), is_classification)
    val_m = _metrics(y_val, model.predict(X_val), is_classification)

    record = {
        "epoch": 1,
        "train_loss": round(train_m["loss"], 6),
        "val_loss": round(val_m["loss"], 6),
        "accuracy": round(train_m["accuracy"], 6),
        "val_accuracy": round(val_m["accuracy"], 6),
        "f1_score": round(train_m["f1_score"], 6),
        "val_f1_score": round(val_m["f1_score"], 6),
    }
    if is_classification:
        record.update({
            "precision_macro": round(train_m["precision_macro"], 6),
            "val_precision_macro": round(val_m["precision_macro"], 6),
            "recall_macro": round(train_m["recall_macro"], 6),
            "val_recall_macro": round(val_m["recall_macro"], 6),
        })
    logger.info("[Train] %s: val_loss=%.4f acc=%.4f f1=%.4f",
                algorithm, record["val_loss"], record["val_accuracy"], record["val_f1_score"])
    return model, [record], record
