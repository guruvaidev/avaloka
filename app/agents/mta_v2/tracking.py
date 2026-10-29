"""Where MLflow runs are recorded when nobody said.

The trainer writes runs and the model endpoints read them, so both must
resolve the same store. They used to default to ``mlruns`` -- MLflow's file
store, which the pinned MLflow refuses to write to ("the filesystem tracking
backend is in maintenance mode"). A local API run without
``MLFLOW_TRACKING_URI`` therefore failed every training turn and returned 500
from ``GET /api/models``. The default is now a SQLite file in the working
directory, the backend MLflow itself recommends for local use.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """``explicit``, then MLFLOW_TRACKING_URI, then MLFLOW_BACKEND_STORE_URI,
    then ``sqlite:///<cwd>/mlflow.db``."""
    return (
        (explicit or "").strip()
        or os.getenv("MLFLOW_TRACKING_URI", "").strip()
        or os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
        or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    )
