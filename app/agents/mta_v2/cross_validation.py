"""k-fold cross-validation of a trained model's recipe, with a confusion matrix per fold.

Runs after training on the **training split only** (the holdout stays the honest
out-of-sample number). Each fold re-fits the whole recipe -- feature
preprocessing, scaling and the model -- on that fold's fit rows, exactly as
``training_main`` does, so nothing fitted on the whole data leaks into a fold
(TI-31/TI-32). The MLP re-trains with the plan's hyperparameters and a capped
epoch count; the scikit-learn families (1.7) clone-fit through
``estimators.fit_sklearn``.

Pure parts (no data loading, no MLflow) are separate so the oracle fixture of
TI-36 and the fold-count rule can be tested without training anything:

- :func:`k_eff` -- ``max(2, min(k, n // 2))``, the rule from evaluation_agent.
- :func:`fold_predictions` -- the fold loop over an injected ``fit_predict``.
- :func:`confusion` / :func:`aggregate` -- counts, row-normalised, accuracy, f1.

Telemetry about a CV run carries only k, row counts, duration and status.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_K = 5
#: Below this, fold estimates are too noisy to be reported (evaluation_agent.MIN_ROWS_FOR_CV).
MIN_ROWS_FOR_CV = 30
#: Refitting an MLP per fold must stay cheap: epochs are capped here.
MAX_FOLD_EPOCHS = 20

FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (train_idx, val_idx) -> y_pred for val_idx


# --------------------------------------------------------------------------- #
# Pure rules
# --------------------------------------------------------------------------- #

def k_eff(n_rows: int, k: int = DEFAULT_K) -> int:
    """The fold count actually used: ``max(2, min(k, n // 2))``.

    n=3 -> 2, n=4 -> 2, n=9 -> 4, n=10 -> 5, n=11 -> 5 (k=5). With the 30-row
    floor every run that is cross-validated gets k itself; the rule matters for
    callers that lower the floor.
    """
    return max(2, min(int(k), int(n_rows) // 2))


def choose_splitter(task: str, *, time_ordered: bool, k: int, seed: int):
    """StratifiedKFold for classification, KFold for regression, TimeSeriesSplit when time-ordered."""
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

    if time_ordered:
        return "TimeSeriesSplit", TimeSeriesSplit(n_splits=k)
    if task == "classification":
        return "StratifiedKFold", StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return "KFold", KFold(n_splits=k, shuffle=True, random_state=seed)


def fold_predictions(fit_predict: FitPredict, y: np.ndarray, splitter: Any) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run the fold loop. Returns ``[(val_idx, y_true, y_pred), ...]`` in fold order.

    ``fit_predict(train_idx, val_idx)`` fits on the first and predicts the second;
    the caller owns what "fit" means (a full recipe refit, or a lookup in a test).
    """
    y = np.asarray(y)
    X_dummy = np.zeros((len(y), 1))
    out = []
    for train_idx, val_idx in splitter.split(X_dummy, y):
        y_pred = np.asarray(fit_predict(train_idx, val_idx))
        if len(y_pred) != len(val_idx):
            raise ValueError(f"fit_predict returned {len(y_pred)} predictions for {len(val_idx)} rows")
        out.append((val_idx, y[val_idx], y_pred))
    return out


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> Dict[str, Any]:
    """Counts (rows = actual, columns = predicted) in ``labels`` order, plus row-normalised
    values, accuracy, macro F1 and per-class precision/recall."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    codes = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=codes)
    row_tot = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_tot, out=np.zeros_like(matrix, dtype=float), where=row_tot > 0)
    return {
        "labels": [str(l) for l in labels],
        "matrix": [[int(v) for v in row] for row in matrix],
        "normalized": [[round(float(v), 4) for v in row] for row in normalized],
        "n": int(matrix.sum()),
        "accuracy": round(float((y_true == y_pred).mean()) if len(y_true) else 0.0, 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=codes, average="macro", zero_division=0)), 6),
        "precision": [round(float(v), 6) for v in precision_score(y_true, y_pred, labels=codes, average=None, zero_division=0)],
        "recall": [round(float(v), 6) for v in recall_score(y_true, y_pred, labels=codes, average=None, zero_division=0)],
    }


def regression_scores(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return {"rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
            "r2": round(float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0, 6), "n": int(len(y_true))}


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return (round(float(arr.mean()), 6), round(float(arr.std(ddof=0)), 6)) if arr.size else (0.0, 0.0)


def aggregate(folds: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Sum the fold matrices (classification) or average the fold metrics (regression)."""
    if task == "classification":
        labels = folds[0]["labels"]
        total = np.zeros((len(labels), len(labels)), dtype=int)
        for f in folds:
            total += np.asarray(f["matrix"], dtype=int)
        row_tot = total.sum(axis=1, keepdims=True)
        normalized = np.divide(total, row_tot, out=np.zeros_like(total, dtype=float), where=row_tot > 0)
        acc_m, acc_s = _mean_std(f["accuracy"] for f in folds)
        f1_m, f1_s = _mean_std(f["f1_macro"] for f in folds)
        # Pooled scores on the summed matrix (TI-36's accuracy 9/12, f1_macro 0.75):
        # the fold means above describe the spread, these describe the whole split.
        diag = np.diag(total).astype(float)
        col_tot = total.sum(axis=0).astype(float)
        precision = np.divide(diag, col_tot, out=np.zeros_like(diag), where=col_tot > 0)
        recall = np.divide(diag, row_tot.reshape(-1).astype(float), out=np.zeros_like(diag), where=row_tot.reshape(-1) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diag), where=(precision + recall) > 0)
        return {"labels": labels, "matrix": total.tolist(), "normalized": [[round(float(v), 4) for v in r] for r in normalized],
                "n": int(total.sum()), "k": len(folds), "primary_metric": "accuracy",
                "accuracy": round(float(diag.sum() / total.sum()) if total.sum() else 0.0, 6),
                "f1_macro": round(float(f1.mean()) if len(f1) else 0.0, 6),
                "precision": [round(float(v), 6) for v in precision], "recall": [round(float(v), 6) for v in recall],
                "metric_mean": acc_m, "metric_std": acc_s, "accuracy_mean": acc_m, "accuracy_std": acc_s,
                "f1_macro_mean": f1_m, "f1_macro_std": f1_s}
    out: Dict[str, Any] = {"k": len(folds), "n": int(sum(f["n"] for f in folds)), "primary_metric": "rmse", "confusion_matrix": None}
    for key in ("rmse", "mae", "r2"):
        m, s = _mean_std(f[key] for f in folds)
        out[f"{key}_mean"], out[f"{key}_std"] = m, s
    out["metric_mean"], out["metric_std"] = out["rmse_mean"], out["rmse_std"]
    return out


def run_folds(y: np.ndarray, *, task: str, labels: Optional[Sequence[Any]], fit_predict: FitPredict,
              k: int = DEFAULT_K, seed: int = 42, time_ordered: bool = False, min_rows: int = MIN_ROWS_FOR_CV) -> Dict[str, Any]:
    """The whole report over an injected ``fit_predict``: the piece the oracle test drives."""
    n = len(y)
    if n < min_rows:
        return {"status": "skipped", "k": None, "splitter": None, "folds": [], "aggregate": None,
                "reason": f"only {n} rows; at least {min_rows} are needed for cross-validated estimates"}
    k_used = k_eff(n, k)
    if task == "classification":
        _, counts = np.unique(np.asarray(y), return_counts=True)
        if counts.min() < k_used and not time_ordered:
            return {"status": "skipped", "k": k_used, "splitter": "StratifiedKFold", "folds": [], "aggregate": None,
                    "reason": f"a class has only {int(counts.min())} rows, fewer than the {k_used} folds; stratified folds would leave it empty"}
    name, splitter = choose_splitter(task, time_ordered=time_ordered, k=k_used, seed=seed)
    folds = []
    for i, (val_idx, y_true, y_pred) in enumerate(fold_predictions(fit_predict, y, splitter)):
        entry = confusion(y_true, y_pred, labels) if task == "classification" else regression_scores(y_true, y_pred)
        entry["fold"] = i
        entry["val_index"] = [int(v) for v in val_idx]
        folds.append(entry)
    report = {"status": "succeeded", "k": k_used, "splitter": name, "folds": folds, "aggregate": aggregate(folds, task),
              "reason": None if k_used == k else f"k reduced from {k} to {k_used} for {n} rows (max(2, min(k, n // 2)))"}
    return report


# --------------------------------------------------------------------------- #
# The real recipe: refit per fold with the trainers' own functions
# --------------------------------------------------------------------------- #

def _recipe_fit_predict(df: Any, plan: Dict[str, Any], *, task: str, class_names: List[str], feature_cols: List[str],
                        model_type: str, algorithm: str) -> FitPredict:
    from app.agents.mta_v2 import local_trainer as lt

    hp = plan.get("hyperparameter_config") or {}

    def fit_predict(train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        train_df = df.iloc[train_idx].copy(); val_df = df.iloc[val_idx].copy()
        preprocessing = lt.fit_feature_preprocessing(train_df, feature_cols)              # fitted inside the fold (TI-32)
        train_df = lt.transform_feature_dataframe(train_df, feature_cols, preprocessing)
        val_df = lt.transform_feature_dataframe(val_df, feature_cols, preprocessing)
        model_feature_cols = list(preprocessing["model_feature_names"])
        train_df, val_df, preprocessing = lt.apply_standard_scaling(train_df, val_df, model_feature_cols, model_type, preprocessing)
        config: Dict[str, Any] = {
            "feature_cols": model_feature_cols, "model_type": model_type, "class_names": class_names,
            "num_classes": len(class_names) if class_names else 1, "output_dim": len(class_names) if class_names else 1,
            "hidden_sizes": list(hp.get("hidden_layer_sizes", [128, 64])), "activation": hp.get("activation", "relu"),
            "dropout": float(hp.get("dropout_rate", 0.2)), "batch_norm": bool(hp.get("batch_norm", True)),
            "learning_rate": float(hp.get("learning_rate", 0.001)),
            "epochs": int(min(int(hp.get("epochs", 30)), MAX_FOLD_EPOCHS)), "batch_size": int(hp.get("batch_size", 32)),
            "optimizer": hp.get("optimizer_name", "adam"), "random_seed": int(hp.get("random_seed", 42)),
            "early_stopping_patience": int(hp.get("early_stopping_patience", 5)),
            "early_stopping_min_delta": float(hp.get("early_stopping_min_delta", 1e-4)), "use_gpu": False,
            "target_scaling": preprocessing.get("target_scaling", {}), "class_weights": preprocessing.get("class_weights"),
            "algorithm": algorithm, "time_ordered": bool((plan.get("data_config") or {}).get("time_column")),
        }
        if algorithm == "mlp":
            model, _, _ = lt.train_local(train_df, val_df, config)
        else:
            from app.agents.mta_v2 import estimators
            model, _, _ = estimators.fit_sklearn(train_df, val_df, config)
        X_val = val_df[model_feature_cols].to_numpy(dtype=float)
        if task == "classification":
            from app.agents.mta_v2.estimators import predict_with_scores
            y_pred, _ = predict_with_scores(model, X_val)
            return np.asarray(y_pred).astype(int)
        # regression: the recipe standardises the target; undo it so folds are in target units
        from app.agents.mta_v2.estimators import is_torch_model
        if is_torch_model(model):
            import torch
            model.eval()
            with torch.no_grad():
                raw = model(torch.tensor(X_val, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
        else:
            raw = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
        return np.asarray(lt.inverse_standard_scaling(raw.astype(float).tolist(), preprocessing.get("target_scaling") or {}))

    return fit_predict


def cross_validate_plan(plan: Dict[str, Any], *, df: Any = None, k: Optional[int] = None, seed: Optional[int] = None,
                        fit_predict: Optional[FitPredict] = None) -> Dict[str, Any]:
    """Cross-validate a training plan's recipe on its training split.

    ``df`` may be given (tests); otherwise the plan's dataset is loaded with the
    trainer's loader. The holdout split is removed first with the trainer's own
    ``split_train_val`` so CV sees exactly the training split.
    """
    from app.agents.mta_v2 import local_trainer as lt

    t0 = time.time()
    data_cfg = plan.get("data_config") or {}
    hp = plan.get("hyperparameter_config") or {}
    model_type = str(plan.get("model_type") or "classification")
    task = "regression" if "regression" in model_type else "classification"
    feature_cols = [str(c) for c in data_cfg.get("feature_columns") or []]
    target_col = str(data_cfg.get("target_column") or "")
    time_column = data_cfg.get("time_column") or ""
    algorithm = str(plan.get("algorithm") or hp.get("algorithm") or "mlp").lower()
    k = int(k or (plan.get("cross_validation") or {}).get("k") or DEFAULT_K)
    seed = int(seed if seed is not None else hp.get("random_seed", 42))

    if df is None:
        df = lt.load_dataframe(str(data_cfg.get("dataset_uri") or ""))
    df = df[feature_cols + [target_col] + ([time_column] if time_column and time_column in df.columns else [])].copy()
    if task == "classification":
        class_names = [str(v) for v in (data_cfg.get("class_names") or sorted(df[target_col].astype(str).unique()))]
        df["__label__"] = df[target_col].astype(str).apply(lambda v: class_names.index(v)).astype(np.int64)
    else:
        class_names = []
        df["__label__"] = df[target_col].astype(np.float32)
    if time_column and time_column in df.columns:
        df = df.sort_values(time_column).reset_index(drop=True)
    train_df, _holdout = lt.split_train_val(df, time_ordered=bool(time_column))
    train_df = train_df.reset_index(drop=True)
    y = train_df["__label__"].to_numpy()
    fp = fit_predict or _recipe_fit_predict(train_df, plan, task=task, class_names=class_names, feature_cols=feature_cols,
                                            model_type=model_type, algorithm=algorithm)
    report = run_folds(y, task=task, labels=class_names if task == "classification" else None, fit_predict=fp,
                       k=k, seed=seed, time_ordered=bool(time_column))
    report.update({"task": task, "n_rows": int(len(y)), "duration_s": round(time.time() - t0, 3), "requested_k": k, "seed": seed})
    return report


def fold_artifacts(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{artifact path: json}`` for MLflow: one file per fold and the aggregate."""
    out: Dict[str, Dict[str, Any]] = {}
    for f in report.get("folds") or []:
        out[f"cv/fold_{f['fold']}/confusion_matrix.json" if report.get("task") == "classification" else f"cv/fold_{f['fold']}/metrics.json"] = f
    if report.get("aggregate") is not None:
        out["cv/confusion_matrix.json" if report.get("task") == "classification" else "cv/metrics.json"] = report["aggregate"]
    return out


def run_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    agg = report.get("aggregate") or {}
    if report.get("task") == "classification":
        keys = ("accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std")
    else:
        keys = ("rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std")
    return {f"cv_{k}": float(agg[k]) for k in keys if k in agg}
