"""Model signature and input example for ``mlflow.pytorch.log_model``.

A registered model without a signature cannot tell a caller what it expects,
and the registry UI shows nothing. The signature here is the *tensor*
contract the trainers actually save: the encoded feature columns
(``preprocessing["model_feature_names"]``) as float32 in, the network's raw
output (logits or the scaled target) as float32 out. Building it never fails
training: on any problem the caller logs the model without a signature.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EXAMPLE_ROWS = 5


def example_matrix(rows: Iterable[Any], feature_cols: Sequence[str], limit: int = EXAMPLE_ROWS) -> "Any":
    """A float32 ``(n, len(feature_cols))`` array from dict rows or a DataFrame."""
    import numpy as np

    if hasattr(rows, "to_numpy"):  # a DataFrame
        frame = rows[list(feature_cols)].head(limit)
        return frame.to_numpy(dtype="float32")
    out: List[List[float]] = []
    for row in rows:
        out.append([float(row[c]) for c in feature_cols])
        if len(out) >= limit:
            break
    return np.asarray(out, dtype="float32").reshape(-1, len(feature_cols))


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
