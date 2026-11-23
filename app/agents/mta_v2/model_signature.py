"""Model signature and input example for ``mlflow.pytorch`` / ``mlflow.sklearn``.

A registered model without a signature cannot tell a caller what it expects,
and the registry UI shows nothing. The signature here is the *tensor*
contract the trainers actually save: the encoded feature columns
(``preprocessing["model_feature_names"]``) in the dtype the model was trained
on, and its raw output. For the torch MLP that is float32 in (the trainer's
tensors) and float32 out (logits or the scaled target); for a scikit-learn
family it is float64 in (``estimators.SKLEARN_FEATURE_DTYPE``, what
``fit_sklearn`` fits on). MLflow enforces a tensor signature's dtype exactly,
so a float32 signature on a model fitted on float64 refuses an ordinary
float64 frame. Building it never fails training: on any problem the caller
logs the model without a signature.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EXAMPLE_ROWS = 5


def example_matrix(rows: Iterable[Any], feature_cols: Sequence[str], limit: int = EXAMPLE_ROWS,
                   dtype: str = "float32") -> "Any":
    """A ``(n, len(feature_cols))`` array of ``dtype`` from dict rows or a DataFrame."""
    import numpy as np

    if hasattr(rows, "to_numpy"):  # a DataFrame
        frame = rows[list(feature_cols)].head(limit)
        return frame.to_numpy(dtype=dtype)
    out: List[List[float]] = []
    for row in rows:
        out.append([float(row[c]) for c in feature_cols])
        if len(out) >= limit:
            break
    return np.asarray(out, dtype=dtype).reshape(-1, len(feature_cols))


def build_signature(model: Any, rows: Iterable[Any], feature_cols: Sequence[str]) -> Tuple[Optional[Any], Optional[Any]]:
    """``(signature, input_example)`` or ``(None, None)`` when it cannot be built."""
    try:
        import numpy as np
        import torch
        from mlflow.models import infer_signature

        X = example_matrix(rows, feature_cols)
        if X.size == 0:
            return None, None
        model.eval()
        with torch.no_grad():
            y = model(torch.as_tensor(X)).detach().cpu().numpy().astype("float32")
        return infer_signature(X, y), np.asarray(X[:EXAMPLE_ROWS])
    except Exception as exc:  # noqa: BLE001 - a signature must never fail training
        logger.warning("[mlflow] signature not built: %s", exc)
        return None, None


def build_sklearn_signature(model: Any, rows: Iterable[Any], feature_cols: Sequence[str]) -> Tuple[Optional[Any], Optional[Any]]:
    """``(signature, input_example)`` for a scikit-learn estimator (1.7 families), or ``(None, None)``."""
    try:
        import numpy as np
        from mlflow.models import infer_signature

        from app.agents.mta_v2.estimators import SKLEARN_FEATURE_DTYPE

        X = example_matrix(rows, feature_cols, dtype=SKLEARN_FEATURE_DTYPE)
        if X.size == 0:
            return None, None
        y = np.asarray(model.predict(X))
        return infer_signature(X, y), np.asarray(X[:EXAMPLE_ROWS])
    except Exception as exc:  # noqa: BLE001 - a signature must never fail training
        logger.warning("[mlflow] sklearn signature not built: %s", exc)
        return None, None
