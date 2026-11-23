"""
The Avaloka client contract + reference implementation.

CONTRACT (the Python entrypoint the CLI package must expose)
------------------------------------------------------------
The benchmark talks to Avaloka through the ``AvalokaClient`` protocol. The real
CLI package is expected to ship a factory ``avaloka.benchmark_client()`` (or a
class ``avaloka.AvalokaClient``) returning an object implementing these methods.
Every method takes a source *URI* (as produced by ``sources.DataSource.uri``)
plus keyword args, and returns a plain ``dict`` whose shape is fixed below so the
same scoring checks apply to any implementation.

    sample(uri, *, stratify_by=None, sample_size=5)
        -> {schema: [str], rows: [dict], ddl_schema: str, table_name: str|None, error}
    analyze_features(uri, *, target=None)
        -> {features: [{name,dtype,role,missing_fraction,cardinality}], target, error}
    build_features(uri, *, target=None, recipe=None)
        -> {feature_names: [str], n_features: int, matrix_path: str, error}
    transform(uri, *, operations=[...])
        -> {output_path: str, rows: int, columns: [str], error}
    train(uri, *, target, task_kind, heavy=False, time_budget_s=60)
        -> {metrics: {...}, model_path: str, model_kind: str, error}
    serve(model_ref, *, platform="k8s", replicas=1)
        -> {endpoint: str, status: str, ready: bool, predict_url: str, error}

RESOLUTION
----------
``get_client()`` returns the real CLI client if the ``avaloka`` package exposes
one and ``AVALOKA_BENCH_CLIENT != reference``; otherwise it returns the built-in
``ReferenceClient`` that drives the *existing* LangGraph agents
(``app.agents.sampling_agent``) and deterministic numpy/pandas implementations.
This lets the benchmark run today and stay valid once the CLI lands.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import ml_train


@runtime_checkable
class AvalokaClient(Protocol):
    name: str

    def sample(self, uri: str, *, stratify_by: Optional[str] = None,
               sample_size: float = 5) -> Dict[str, Any]: ...

    def analyze_features(self, uri: str, *, target: Optional[str] = None) -> Dict[str, Any]: ...

    def build_features(self, uri: str, *, target: Optional[str] = None,
                       recipe: Optional[dict] = None) -> Dict[str, Any]: ...

    def transform(self, uri: str, *, operations: List[dict]) -> Dict[str, Any]: ...

    def train(self, uri: str, *, target: str, task_kind: str,
              heavy: bool = False, time_budget_s: int = 60) -> Dict[str, Any]: ...

    def serve(self, model_ref: str, *, platform: str = "k8s",
              replicas: int = 1) -> Dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _local_path_from_uri(uri: str) -> Path:
    """Resolve a benchmark source URI to a readable local CSV path.

    The benchmark always materializes data locally before invoking the client
    (see ``sources.DataSource.materialize_locally``); for fake object stores the
    path is encoded in the URI. ``file://`` and bare paths are handled directly.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path or uri)
    # For sqlite URIs the path precedes the '#table' fragment.
    if parsed.scheme == "sqlite":
        return Path(uri[len("sqlite://"):].split("#", 1)[0])
    # Object-store fakes: the benchmark passes a resolved local path via env map.
    mapping = json.loads(os.environ.get("AVALOKA_BENCH_URI_MAP", "{}"))
    if uri in mapping:
        return Path(mapping[uri])
    # Last resort: treat the path component as local.
    return Path(parsed.path.lstrip("/"))


def _read_frame(uri: str) -> pd.DataFrame:
    p = _local_path_from_uri(uri)
    if p.suffix == ".sqlite":
        import sqlite3

        conn = sqlite3.connect(p)
        try:
            tbl = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;", conn
            ).iloc[0, 0]
            return pd.read_sql_query(f"SELECT * FROM {tbl};", conn)
        finally:
            conn.close()
    return pd.read_csv(p, on_bad_lines="skip")


def _artifact_dir() -> Path:
    root = Path(os.environ.get("AVALOKA_BENCH_DATA_DIR", Path.home() / ".avaloka-bench"))
    d = Path(os.environ.get("AVALOKA_BENCH_ARTIFACTS", root / "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Reference client — drives existing agents + deterministic implementations
# --------------------------------------------------------------------------- #


class ReferenceClient:
    """A correct, deterministic baseline implementation of the contract.

    It is the *oracle* the real CLI is scored against, and also keeps the
    benchmark runnable before the CLI exists. Where the repo already has a real
    agent (sampling), it delegates to it; otherwise it uses pandas/numpy.
    """

    name = "reference"

    # --- sample: delegate to the existing sampling_agent when possible ----- #
    def sample(self, uri, *, stratify_by=None, sample_size=5):
        path = _local_path_from_uri(uri)
        source_type = "sql" if path.suffix == ".sqlite" else "csv"
        try:
            from app.agents.sampling_agent import sample_data_from_source

            out = sample_data_from_source(
                str(path), source_type, stratify_by=stratify_by, sample_size=sample_size
            )
            # sampling_agent already returns the contract shape (+ trace on error)
            out.setdefault("error", None)
            return out
        except Exception:
            # Fall back to a pandas-only sampler with a DDL generator so the
            # benchmark survives environments without Spark/Java.
            return self._pandas_sample(path, stratify_by, sample_size)

    def _pandas_sample(self, path: Path, stratify_by, sample_size):
        try:
            df = _read_frame(f"file://{path}")
            if df.empty:
                return {"error": "Dataframe is empty.", "rows": []}
            if stratify_by and stratify_by in df.columns and df[stratify_by].notna().any():
                frac = sample_size if isinstance(sample_size, float) else None
                n = None if frac else int(sample_size)
                sampled = df.groupby(stratify_by, group_keys=False).apply(
                    lambda g: g.sample(
                        n=min(n, len(g)) if n else max(int(len(g) * frac), 1),
                        random_state=42,
                    )
                )
            else:
                n = int(sample_size) if isinstance(sample_size, int) else max(int(len(df) * sample_size), 1)
                sampled = df.sample(n=min(n, len(df)), random_state=42)
            ddl = self._ddl(df, "sample_table")
            return {
                "schema": df.columns.tolist(),
                "rows": sampled.to_dict(orient="records"),
                "table_name": None,
                "ddl_schema": ddl,
                "error": None,
            }
        except Exception as e:
            return {"error": f"Failed to sample data: {e}", "rows": []}

    @staticmethod
    def _ddl(df: pd.DataFrame, table: str) -> str:
        type_map = {"int64": "INT", "float64": "DOUBLE", "bool": "BOOLEAN",
                    "object": "VARCHAR", "datetime64[ns]": "TIMESTAMP"}
        cols = [f"{c} {type_map.get(str(t), 'TEXT')}" for c, t in df.dtypes.items()]
        return f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols) + "\n);"

    # --- analyze_features --------------------------------------------------- #
    def analyze_features(self, uri, *, target=None):
        try:
            df = _read_frame(uri)
        except Exception as e:
            return {"error": f"read failed: {e}", "features": []}
        features = []
        for col in df.columns:
            s = df[col]
            is_num = pd.api.types.is_numeric_dtype(s)
            role = "target" if col == target else ("numeric" if is_num else "categorical")
            features.append({
                "name": col,
                "dtype": str(s.dtype),
                "role": role,
                "missing_fraction": round(float(s.isna().mean()), 4),
                "cardinality": int(s.nunique(dropna=True)),
            })
        return {"features": features, "target": target, "error": None}

    # --- build_features ----------------------------------------------------- #
    def build_features(self, uri, *, target=None, recipe=None):
        try:
            df = _read_frame(uri)
        except Exception as e:
            return {"error": f"read failed: {e}", "feature_names": [], "n_features": 0}
        X, names = self._encode(df, target)
        out = _artifact_dir() / f"features_{abs(hash(uri)) % (10**8)}.npy"
        np.save(out, X)
        return {
            "feature_names": names,
            "n_features": len(names),
            "matrix_path": str(out),
            "rows": int(X.shape[0]),
            "error": None,
        }

    @staticmethod
    def _encode(df: pd.DataFrame, target: Optional[str]):
        feat = df.drop(columns=[target]) if target and target in df.columns else df
        pieces, names = [], []
        for col in feat.columns:
            s = feat[col]
            if pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce").fillna(s.median() if s.notna().any() else 0.0)
                std = v.std() or 1.0
                pieces.append(((v - v.mean()) / std).to_numpy().reshape(-1, 1))
                names.append(col)
            else:
                # One-hot the top-k levels to keep width bounded.
                top = s.astype("string").fillna("__na__").value_counts().head(12).index
                for level in top:
                    pieces.append((s.astype("string").fillna("__na__") == level)
                                  .astype(float).to_numpy().reshape(-1, 1))
                    names.append(f"{col}={level}")
        if not pieces:
            return np.zeros((len(df), 0)), []
        return np.hstack(pieces), names

    # --- transform ---------------------------------------------------------- #
    def transform(self, uri, *, operations):
        try:
            df = _read_frame(uri)
        except Exception as e:
            return {"error": f"read failed: {e}"}
        try:
            for op in operations:
                df = self._apply_op(df, op)
        except Exception as e:
            return {"error": f"transform failed: {e}"}
        out = _artifact_dir() / f"transform_{abs(hash(uri)) % (10**8)}.csv"
        df.to_csv(out, index=False)
        return {"output_path": str(out), "rows": int(len(df)),
                "columns": df.columns.tolist(), "error": None}

    @staticmethod
    def _apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
        kind = op["op"]
        if kind == "dropna":
            return df.dropna(subset=op.get("columns")) if op.get("columns") else df.dropna()
        if kind == "fillna":
            return df.fillna(op.get("value", 0))
        if kind == "cast":
            return df.astype({op["column"]: op["dtype"]}, errors="ignore")
        if kind == "to_numeric":
            df = df.copy()
            df[op["column"]] = pd.to_numeric(df[op["column"]], errors="coerce")
            return df
        if kind == "groupby_agg":
            return (df.groupby(op["by"])[op["value"]].agg(op.get("agg", "sum"))
                    .reset_index())
        if kind == "filter":
            return df.query(op["expr"])
        if kind == "derive":  # add a column from an expression over existing cols
            df = df.copy()
            df[op["column"]] = df.eval(op["expr"])
            return df
        if kind == "select":
            return df[op["columns"]]
        raise ValueError(f"Unknown op {kind!r}")

    # --- train -------------------------------------------------------------- #
    def train(self, uri, *, target, task_kind, heavy=False, time_budget_s=60):
        try:
            df = _read_frame(uri)
        except Exception as e:
            return {"error": f"read failed: {e}", "metrics": {}}
        if target not in df.columns:
            return {"error": f"target '{target}' not in columns", "metrics": {}}
        X, _ = self._encode(df, target)
        y = df[target]
        try:
            if heavy:
                metrics, kind = ml_train.train_deep(X, y, task_kind, time_budget_s)
            else:
                metrics, kind = ml_train.train_baseline(X, y, task_kind)
        except ml_train.HeavyDepsUnavailable as e:
            return {"error": f"heavy deps unavailable: {e}", "metrics": {}}
        except Exception as e:
            return {"error": f"train failed: {e}", "metrics": {}}
        model_path = _artifact_dir() / f"model_{abs(hash(uri)) % (10**8)}.json"
        model_path.write_text(json.dumps({"kind": kind, "metrics": metrics}))
        return {"metrics": metrics, "model_path": str(model_path),
                "model_kind": kind, "error": None}

    # --- serve (k8s inference-as-a-service; faked endpoint) ----------------- #
    def serve(self, model_ref, *, platform="k8s", replicas=1):
        if not Path(model_ref).exists():
            return {"error": f"model_ref not found: {model_ref}", "ready": False}
        svc = abs(hash(model_ref)) % 9000 + 1000
        endpoint = f"http://avaloka-infer-{svc}.default.svc.cluster.local:8080"
        return {
            "endpoint": endpoint,
            "predict_url": f"{endpoint}/predict",
            "status": "deployed",
            "ready": True,
            "platform": platform,
            "replicas": replicas,
            "error": None,
        }


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


def get_client() -> AvalokaClient:
    """Return the Avaloka client under test.

    Honors ``AVALOKA_BENCH_CLIENT``:
        "reference" (default-if-unset) -> built-in ReferenceClient
        "<module>:<factory>"           -> import and call that factory
    Falls back to ReferenceClient if a real client can't be resolved.
    """
    spec = os.environ.get("AVALOKA_BENCH_CLIENT", "reference")
    if spec == "reference":
        return ReferenceClient()
    try:
        if ":" in spec:
            mod_name, factory = spec.split(":", 1)
        else:  # try the conventional package entrypoint
            mod_name, factory = spec, "benchmark_client"
        mod = importlib.import_module(mod_name)
        client = getattr(mod, factory)()
        if not isinstance(client, AvalokaClient):
            raise TypeError(f"{spec} does not satisfy AvalokaClient protocol")
        return client
    except Exception as e:
        if os.environ.get("AVALOKA_BENCH_STRICT_CLIENT") == "1":
            raise
        # Soft fallback keeps the suite runnable; the report records the swap.
        ref = ReferenceClient()
        ref.name = f"reference(fallback:{spec}:{type(e).__name__})"
        return ref
