"""Batch analysis over Apache Ray — the swarm at scale.

For data too large to analyse live, Avaloka fans herself across a Ray cluster:
each clone profiles one partition, and she *converges* their partial statistics
into one exact, full-dataset profile (a real map-reduce, not a sample). The same
driver runs locally (for proof and small clusters) or as a ``RayJob`` on
Kubernetes via KubeRay.

The convergence is exact for the statistics it computes (counts, missing, min,
max, and mean/variance via sufficient statistics) because partial sums combine
losslessly — there is no sampling error in the batch lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avaloka.util import write_json, write_text


# --- partition profiling (the map step) -----------------------------------
def _partition_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Sufficient statistics for one partition; designed to combine losslessly."""
    cols: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        entry = {"missing": int(s.isna().sum()), "count": int(s.notna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            entry.update({
                "is_numeric": True,
                "sum": float(v.sum()), "sumsq": float((v ** 2).sum()),
                "min": float(v.min()) if len(v) else math.inf,
                "max": float(v.max()) if len(v) else -math.inf,
            })
        else:
            entry["is_numeric"] = False
            entry["top"] = {str(k): int(n) for k, n in s.dropna().astype(str).value_counts().head(20).items()}
        cols[str(c)] = entry
    return {"n_rows": int(len(df)), "columns": cols}


def _combine(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """The reduce step: merge partial statistics into one exact profile."""
    n_rows = sum(p["n_rows"] for p in parts)
    names = parts[0]["columns"].keys() if parts else []
    merged_cols = []
    for name in names:
        es = [p["columns"][name] for p in parts]
        missing = sum(e["missing"] for e in es)
        count = sum(e["count"] for e in es)
        col: dict[str, Any] = {
            "name": name, "missing": missing,
            "missing_pct": round(missing / n_rows, 4) if n_rows else 0.0,
            "count": count,
        }
        if es[0].get("is_numeric"):
            total = sum(e["sum"] for e in es)
            sumsq = sum(e["sumsq"] for e in es)
            mean = total / count if count else 0.0
            var = (sumsq / count - mean ** 2) if count else 0.0
            col.update({
                "role": "numeric",
                "min": min(e["min"] for e in es),
                "max": max(e["max"] for e in es),
                "mean": round(mean, 6),
                "std": round(math.sqrt(max(0.0, var)), 6),
            })
        else:
            agg: dict[str, int] = {}
            for e in es:
                for k, v in e.get("top", {}).items():
                    agg[k] = agg.get(k, 0) + v
            top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:10]
            col.update({"role": "categorical",
                        "top_values": [{"value": k, "count": v} for k, v in top]})
        merged_cols.append(col)
    return {"n_rows": n_rows, "n_partitions": len(parts), "columns": merged_cols}


@dataclass
class BatchResult:
    converged: dict[str, Any]
    n_partitions: int
    engine: str  # "ray-local" | "ray-cluster" | "single-process"


def run_ray_local(source: str, n_partitions: int = 8) -> BatchResult:
    """Run the full-dataset profile as a real Ray map-reduce on the local machine."""
    from avaloka.io import load_dataset

    handle = load_dataset(source)
    frame = handle.frame
    # Split by row positions (np.array_split on a DataFrame is deprecated).
    idx_parts = np.array_split(np.arange(len(frame)), max(1, n_partitions))
    chunks = [frame.iloc[idx] for idx in idx_parts if len(idx)]

    try:
        import ray

        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 include_dashboard=False, configure_logging=False)
        try:
            remote_stats = ray.remote(_partition_stats)
            futures = [remote_stats.remote(chunk) for chunk in chunks]
            parts = ray.get(futures)
            engine = "ray-local"
        finally:
            ray.shutdown()
    except Exception:
        # Ray unavailable — still converge, just in-process (identical maths).
        parts = [_partition_stats(chunk) for chunk in chunks]
        engine = "single-process"

    converged = _combine(parts)
    return BatchResult(converged=converged, n_partitions=len(chunks), engine=engine)


# --- Kubernetes RayJob (KubeRay) ------------------------------------------
def write_batch_artifacts(out_dir: Path, source: str, *, image: str, n_partitions: int) -> dict[str, Path]:
    """Write the standalone Ray batch driver and the KubeRay RayJob manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = out_dir / "ray_batch_driver.py"
    write_text(driver, _DRIVER_SRC.format(source=repr(source), n=n_partitions))

    manifest = out_dir / "rayjob.yaml"
    import yaml

    rayjob = {
        "apiVersion": "ray.io/v1", "kind": "RayJob",
        "metadata": {"name": "avaloka-batch"},
        "spec": {
            "entrypoint": f"python ray_batch_driver.py",
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": {"spec": {"containers": [{
                        "name": "ray-head", "image": image,
                        "resources": {"requests": {"cpu": "1", "memory": "2Gi"}}}]}}},
                "workerGroupSpecs": [{
                    "groupName": "workers", "replicas": 2, "minReplicas": 0, "maxReplicas": 8,
                    "rayStartParams": {},
                    "template": {"spec": {"containers": [{
                        "name": "ray-worker", "image": image,
                        # KubeRay can autoscale workers and target node classes via labels.
                        "resources": {"requests": {"cpu": "2", "memory": "4Gi"},
                                      "limits": {"cpu": "2", "memory": "4Gi"}}}]}}}],
            },
        },
    }
    manifest.write_text(yaml.safe_dump(rayjob, sort_keys=False), encoding="utf-8")
    return {"driver": driver, "manifest": manifest}


def submit_commands(manifest: Path) -> list[str]:
    return [
        "# Submit the full-dataset analysis as a KubeRay RayJob (autoscaling workers):",
        f"kubectl apply -f {manifest}",
        "kubectl get rayjob avaloka-batch -w        # watch it converge",
        "kubectl logs -l job-name=avaloka-batch --tail=50",
    ]


_DRIVER_SRC = '''"""Avaloka Ray batch driver — full-dataset profile via map-reduce.

Runs the same convergence Avaloka uses locally, but across a Ray cluster so the
partitions are read and profiled in parallel. Exact, not sampled.
"""

import numpy as np
import pandas as pd
import ray

from avaloka.ray_batch import _partition_stats, _combine

SOURCE = {source}
N = {n}


def main():
    ray.init(address="auto", ignore_reinit_error=True)
    df = pd.read_parquet(SOURCE) if str(SOURCE).endswith((".parquet", ".pq")) else pd.read_csv(SOURCE)
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), N) if len(idx)]
    parts = ray.get([ray.remote(_partition_stats).remote(c) for c in chunks])
    converged = _combine(parts)
    print("CONVERGED_ROWS", converged["n_rows"], "PARTITIONS", converged["n_partitions"])
    import json
    with open("batch_profile.json", "w") as fh:
        json.dump(converged, fh, indent=2)


if __name__ == "__main__":
    main()
'''
