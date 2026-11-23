"""
Training paths for the benchmark.

Two implementations, both deterministic:

* ``train_baseline`` — a dependency-free numpy model (logistic / softmax /
  ridge regression via closed-form or gradient descent). Always available, fast,
  and used to compute the *oracle baseline* every CLI model is scored against.

* ``train_deep`` — a real PyTorch tabular MLP (the "heavy" path the user asked
  for). Trains for real on CPU/GPU. Raises ``HeavyDepsUnavailable`` if torch is
  not installed so the suite can skip gracefully.

All metrics use a fixed train/test split (seeded) so scores are reproducible.
``task_kind`` is one of: "binary_classification", "multiclass_classification",
"regression".
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

SEED = 1729


class HeavyDepsUnavailable(RuntimeError):
    """Raised when the heavy DL stack (torch) is not importable."""


# --------------------------------------------------------------------------- #
# Shared prep
# --------------------------------------------------------------------------- #


def _encode_target(y: pd.Series, task_kind: str):
    if task_kind == "regression":
        return pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype=float), None
    cats = sorted(y.astype("string").fillna("__na__").unique().tolist())
    idx = {c: i for i, c in enumerate(cats)}
    yi = y.astype("string").fillna("__na__").map(idx).to_numpy()
    return yi, cats


def _split(X: np.ndarray, y: np.ndarray, test_frac=0.25):
    rng = np.random.default_rng(SEED)
    n = len(y)
    perm = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    return X[tr], X[te], y[tr], y[te]


def _metrics_classification(y_true, proba, n_classes) -> Dict[str, float]:
    pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)
    acc = float((pred == y_true).mean())
    # macro F1
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    # log loss
    if proba.ndim == 1:
        p = np.clip(proba, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    else:
        p = np.clip(proba, 1e-7, 1.0)
        ll = float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))
    return {"accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "logloss": round(ll, 4)}


def _metrics_regression(y_true, pred) -> Dict[str, float]:
    err = pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "r2": round(1 - ss_res / ss_tot, 4)}


# --------------------------------------------------------------------------- #
# Baseline (numpy, always available) — also the oracle
# --------------------------------------------------------------------------- #


def _gd_logistic(Xtr, ytr, n_classes, epochs=300, lr=0.1):
    n, d = Xtr.shape
    if n_classes == 2:
        w = np.zeros(d)
        b = 0.0
        for _ in range(epochs):
            z = Xtr @ w + b
            p = 1 / (1 + np.exp(-z))
            g = p - ytr
            w -= lr * (Xtr.T @ g / n + 1e-3 * w)
            b -= lr * g.mean()
        return ("binary", w, b)
    W = np.zeros((d, n_classes))
    bb = np.zeros(n_classes)
    Y = np.eye(n_classes)[ytr]
    for _ in range(epochs):
        z = Xtr @ W + bb
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        g = p - Y
        W -= lr * (Xtr.T @ g / n + 1e-3 * W)
        bb -= lr * g.mean(axis=0)
    return ("multi", W, bb)


def _predict_logistic(model, X):
    kind = model[0]
    if kind == "binary":
        _, w, b = model
        return 1 / (1 + np.exp(-(X @ w + b)))
    _, W, bb = model
    z = X @ W + bb
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_baseline(X: np.ndarray, y: pd.Series, task_kind: str) -> Tuple[Dict[str, float], str]:
    yenc, cats = _encode_target(y, task_kind)
    Xtr, Xte, ytr, yte = _split(X, yenc)
    if task_kind == "regression":
        # Ridge closed form.
        A = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ ytr)
        pred = Xte @ w
        return _metrics_regression(yte, pred), "ridge_baseline"
    n_classes = len(cats)
    model = _gd_logistic(Xtr, ytr, n_classes)
    proba = _predict_logistic(model, Xte)
    return _metrics_classification(yte, proba, n_classes), "logistic_baseline"


def compute_baseline_oracle(df: pd.DataFrame, target: str, task_kind: str,
                            encode) -> Dict[str, float]:
    """Convenience used by the harness to populate ``ctx.extras['baseline']``."""
    X, _ = encode(df, target)
    metrics, _ = train_baseline(X, df[target], task_kind)
    return metrics


# --------------------------------------------------------------------------- #
# Heavy deep learning path (PyTorch)
# --------------------------------------------------------------------------- #


def train_deep(X: np.ndarray, y: pd.Series, task_kind: str,
               time_budget_s: int = 60) -> Tuple[Dict[str, float], str]:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # torch not installed
        raise HeavyDepsUnavailable(str(e))

    import time

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    yenc, cats = _encode_target(y, task_kind)
    Xtr, Xte, ytr, yte = _split(X.astype(np.float32), yenc)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_out = 1 if task_kind == "regression" else max(len(cats), 2)
    is_reg = task_kind == "regression"

    model = nn.Sequential(
        nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_out),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    if is_reg:
        # Standardize the target so the MLP converges on large-magnitude targets
        # (e.g. house prices in the millions); metrics are computed on the
        # original scale below.
        y_mean = float(ytr.mean())
        y_std = float(ytr.std()) or 1.0
        ytr_t = torch.tensor(((ytr - y_mean) / y_std).astype(np.float32),
                             device=device).view(-1, 1)
    else:
        y_mean, y_std = 0.0, 1.0
        ytr_t = torch.tensor(ytr.astype(np.int64), device=device)

    start = time.time()
    model.train()
    epoch = 0
    # Full-batch training: let the wall-clock budget govern convergence rather
    # than an arbitrarily small epoch cap (which underfits regression targets).
    while epoch < 5000 and (time.time() - start) < time_budget_s:
        opt.zero_grad()
        out = model(Xtr_t)
        loss = loss_fn(out, ytr_t)
        loss.backward()
        opt.step()
        epoch += 1

    model.eval()
    with torch.no_grad():
        out = model(Xte_t).cpu().numpy()
    if is_reg:
        pred = out.reshape(-1) * y_std + y_mean  # back to original scale
        return _metrics_regression(yte, pred), f"torch_mlp(epochs={epoch})"
    # softmax probabilities
    z = out - out.max(axis=1, keepdims=True)
    e = np.exp(z)
    proba = e / e.sum(axis=1, keepdims=True)
    return _metrics_classification(yte, proba, n_out), f"torch_mlp(epochs={epoch})"
