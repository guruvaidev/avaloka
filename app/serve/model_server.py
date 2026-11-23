"""One model-server Deployment that serves many models (the default inference backend).

Why a plain FastAPI/uvicorn process and not Ray Serve: a RayService keeps a
Ray head pod running whether or not a model is loaded, which fails the
owner's "idle cost about zero" rule; this process is one Deployment that the
infra layer scales 0 <-> 1 (``app/infra/model_server_scaler.py``) and an HPA
scales 1 <-> N. Ray Serve multiplexing stays available as the opt-in
``INFERENCE_BACKEND=rayserve`` for GPU-heavy or very-high-QPS installs.

Routes (ClusterIP only; the API is the sole client and does the owner check):

    GET  /healthz                    liveness
    GET  /readyz                     readiness (process up; models load lazily)
    GET  /models                     loaded set: run_id, bytes, last_used, flavor
    POST /models/{run_id}/load       warm one model (body: model_uri | registered_name+version)
    POST /models/{run_id}/unload
    POST /models/{run_id}/predict    body {"features": [...] | {...}, "model_uri"?: ...}

Models load from the MLflow registry on first use (``models:/{name}/{version}``
when the API passes the registry record, else the run's ONNX artifact) and
live in an LRU bounded by bytes (``MODEL_SERVER_MEMORY_BUDGET_BYTES``) and by
count (``MODEL_SERVER_MAX_MODELS``). Credentials come from the environment;
nothing in a request body is a credential. No cloud SDK is imported here.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.serve.inference import _Model, _materialize, _to_vector

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_FRACTION = float(os.getenv("MODEL_SERVER_MEMORY_BUDGET_FRACTION", "0.7"))
DEFAULT_MAX_MODELS = int(os.getenv("MODEL_SERVER_MAX_MODELS", "16"))
ONNX_ARTIFACT = "model_files/model.onnx"


def _cgroup_memory_limit() -> Optional[int]:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = open(path).read().strip()
        except OSError:
            continue
        if text.isdigit():
            value = int(text)
            if 0 < value < 1 << 60:
                return value
    return None


def default_budget_bytes() -> int:
    """``MODEL_SERVER_MEMORY_BUDGET_BYTES``, else 70 % of the container limit, else 1.5 GiB."""
    explicit = os.getenv("MODEL_SERVER_MEMORY_BUDGET_BYTES", "").strip()
    if explicit.isdigit():
        return int(explicit)
    limit = _cgroup_memory_limit()
    if limit:
        return int(limit * DEFAULT_BUDGET_FRACTION)
    return 1536 * 1024 * 1024


# --------------------------------------------------------------------------- #
# LRU bounded by bytes and count -- pure, tested without a server
# --------------------------------------------------------------------------- #

@dataclass
class CacheEntry:
    model: Any
    bytes: int
    flavor: str = "onnx"
    model_uri: Optional[str] = None
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0


class ModelCache:
    """Least-recently-used models, evicted until under the byte budget and the count cap."""

    def __init__(self, budget_bytes: int, max_models: int = DEFAULT_MAX_MODELS) -> None:
        if budget_bytes <= 0 or max_models <= 0:
            raise ValueError("budget_bytes and max_models must be positive")
        self.budget_bytes = int(budget_bytes)
        self.max_models = int(max_models)
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self.evictions: List[str] = []

    def bytes(self) -> int:
        return sum(e.bytes for e in self._entries.values())

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    def get(self, run_id: str, *, now: Optional[float] = None) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._entries.get(run_id)
            if entry is None:
                return None
            self._entries.move_to_end(run_id)
            entry.last_used = now if now is not None else time.time()
            entry.hits += 1
            return entry

    def put(self, run_id: str, entry: CacheEntry) -> List[str]:
        """Insert (or replace) and evict LRU entries until within budget. Returns evicted ids."""
        evicted: List[str] = []
        with self._lock:
            self._entries.pop(run_id, None)
            self._entries[run_id] = entry
            self._entries.move_to_end(run_id)
            while (self.bytes() > self.budget_bytes or len(self._entries) > self.max_models) and len(self._entries) > 1:
                victim, _ = self._entries.popitem(last=False)
                evicted.append(victim)
            self.evictions.extend(evicted)
        return evicted

    def evict(self, run_id: str) -> bool:
        with self._lock:
            return self._entries.pop(run_id, None) is not None

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"run_id": k, "bytes": e.bytes, "flavor": e.flavor, "model_uri": e.model_uri,
                     "loaded_at": e.loaded_at, "last_used": e.last_used, "hits": e.hits}
                    for k, e in self._entries.items()]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

class ModelLoadError(RuntimeError):
    pass


def resolve_model_uri(run_id: str, *, model_uri: Optional[str] = None,
                      registered_name: Optional[str] = None, version: Optional[Any] = None) -> str:
    """The registry version when the API passes it, else the run's ONNX artifact."""
    if model_uri:
        return model_uri
    if registered_name and version not in (None, ""):
        return f"models:/{registered_name}/{version}"
    return f"runs:/{run_id}/{ONNX_ARTIFACT}"


def load_model(run_id: str, model_uri: str, tracking_uri: Optional[str] = None) -> CacheEntry:
    """Materialise and open the artifact. ONNX by default; a registry ``models:/`` URI that
    resolves to an MLflow model directory is loaded through onnxruntime when it holds an
    ONNX file, else through ``mlflow.pyfunc`` (1.7's scikit-learn families)."""
    local = _materialize(model_uri, tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or None)
    path = local
    if os.path.isdir(local):
        onnx_candidates = [os.path.join(r, f) for r, _, fs in os.walk(local) for f in fs if f.endswith(".onnx")]
        if onnx_candidates:
            path = onnx_candidates[0]
        else:
            import mlflow.pyfunc

            pyfunc = mlflow.pyfunc.load_model(local)
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local) for f in fs)
            return CacheEntry(model=_PyfuncModel(pyfunc, model_uri), bytes=int(size * 1.2) + 1, flavor="pyfunc",
                              model_uri=model_uri)
    model = _Model(path)
    if not model.loaded:
        raise ModelLoadError(model.load_error or f"could not load {model_uri}")
    model.model_uri = model_uri
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    return CacheEntry(model=model, bytes=int(size * 1.2) + 1, flavor="onnx", model_uri=model_uri)


def _signature_input_dtype(pyfunc: Any, default: str = "float32") -> str:
    """The dtype the model's tensor signature declares for its input.

    MLflow enforces a tensor signature's dtype exactly: the torch MLP is logged
    as float32, a scikit-learn family as float64 (what it was fitted on), and a
    request cast to the other one is refused. A model with no tensor signature
    is not enforced, so it keeps the float32 this server always sent.
    """
    try:
        schema = pyfunc.metadata.get_input_schema()
        if schema is not None and schema.is_tensor_spec():
            dtype = schema.numpy_types()[0]
            if dtype is not None:
                return str(dtype)
    except Exception:  # noqa: BLE001 - an unreadable signature is no signature
        logger.warning("[model_server] could not read the input signature", exc_info=True)
    return default


class _PyfuncModel:
    """Adapter giving an mlflow.pyfunc model the ``_Model.predict`` shape."""

    loaded = True
    backend = "mlflow.pyfunc"

    def __init__(self, pyfunc: Any, model_uri: str) -> None:
        self._pyfunc = pyfunc
        self.model_uri = model_uri
        self.input_dtype = _signature_input_dtype(pyfunc)

    def predict(self, features: Any) -> Dict[str, Any]:
        import numpy as np

        vec = _to_vector(features)
        out = self._pyfunc.predict(np.asarray([vec], dtype=self.input_dtype))
        arr = np.asarray(out)
        scores = arr.tolist()
        prediction = arr.reshape(-1)[0].item() if arr.size else None
        return {"prediction": prediction, "scores": scores, "model_uri": self.model_uri,
                "model_loaded": True, "backend": self.backend}


# --------------------------------------------------------------------------- #
# The server
# --------------------------------------------------------------------------- #

def create_app(cache: Optional[ModelCache] = None, loader: Optional[Callable[..., CacheEntry]] = None):
    from fastapi import Body, FastAPI, HTTPException

    # `cache or ...` would be wrong: ModelCache defines __len__, so an empty
    # cache is falsy and a fresh one would silently replace the one passed in.
    cache = cache if cache is not None else ModelCache(default_budget_bytes(), DEFAULT_MAX_MODELS)
    loader = loader if loader is not None else load_model
    app = FastAPI(title="Avaloka model server", docs_url=None, redoc_url=None)
    app.state.cache = cache
    load_lock = threading.Lock()

    def _ensure(run_id: str, body: Dict[str, Any]) -> CacheEntry:
        entry = cache.get(run_id)
        if entry is not None:
            return entry
        uri = resolve_model_uri(run_id, model_uri=body.get("model_uri"),
                                registered_name=body.get("registered_name"), version=body.get("version"))
        with load_lock:
            entry = cache.get(run_id)
            if entry is not None:
                return entry
            try:
                entry = loader(run_id, uri)
            except ModelLoadError as exc:
                raise HTTPException(status_code=422, detail={"error": str(exc), "model_uri": uri})
            except Exception as exc:  # noqa: BLE001 - the reason goes back to the worker, not a stack
                logger.warning("[model-server] load failed for %s: %s", run_id, exc)
                raise HTTPException(status_code=422, detail={"error": str(exc).splitlines()[0][:300], "model_uri": uri})
            evicted = cache.put(run_id, entry)
            if evicted:
                logger.info("[model-server] evicted %s to fit %s", evicted, run_id)
            return entry

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        return {"status": "ready", "models": len(cache), "bytes": cache.bytes(), "budget_bytes": cache.budget_bytes}

    @app.get("/models")
    def models():
        return {"models": cache.snapshot(), "bytes": cache.bytes(), "budget_bytes": cache.budget_bytes,
                "max_models": cache.max_models}

    @app.post("/models/{run_id}/load")
    def load(run_id: str, body: Dict[str, Any] = Body(default={})):
        entry = _ensure(run_id, body or {})
        return {"loaded": True, "run_id": run_id, "bytes": entry.bytes, "flavor": entry.flavor, "model_uri": entry.model_uri}

    @app.post("/models/{run_id}/unload")
    def unload(run_id: str):
        return {"unloaded": cache.evict(run_id), "run_id": run_id}

    @app.post("/models/{run_id}/predict")
    def predict(run_id: str, body: Dict[str, Any] = Body(...)):
        entry = _ensure(run_id, body or {})
        features = body.get("features", body)
        if isinstance(features, dict) and not features:
            raise HTTPException(status_code=400, detail="features are required")
        result = entry.model.predict(features)
        result["run_id"] = run_id
        return result

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("MODEL_SERVER_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("MODEL_SERVER_LOG_LEVEL", "info"))
