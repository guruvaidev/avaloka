"""
Glue that runs a single BenchmarkTask against an AvalokaClient.

It materializes the dataset + source, builds the per-task ``TaskContext``
(including the baseline oracle for DS training tasks), dispatches the right
client method for the task's stage, and scores the result.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from . import ml_train
from .adapters import AvalokaClient, ReferenceClient
from .datasets import DatasetHandle, materialize
from .registry import get as get_dataset
from .scoring import TaskResult, run_checks
from .sources import DataSource, make_source
from .tasks import BenchmarkTask, Stage, TaskContext


def _register_uri(source: DataSource, local: Path) -> None:
    """Expose fake-object-store URIs -> local path for the adapter resolver."""
    mapping = json.loads(os.environ.get("AVALOKA_BENCH_URI_MAP", "{}"))
    mapping[source.uri] = str(local)
    os.environ["AVALOKA_BENCH_URI_MAP"] = json.dumps(mapping)


def prepare_source(dataset_key: str, source_kind: str, workdir: Path
                   ) -> Tuple[DatasetHandle, DataSource, Path]:
    spec = get_dataset(dataset_key)
    handle = materialize(spec)
    source = make_source(source_kind, handle, workdir)
    local = source.materialize_locally()
    _register_uri(source, local)
    return handle, source, local


def build_context(task: BenchmarkTask, handle: DatasetHandle, source: DataSource,
                  local: Path) -> TaskContext:
    spec = handle.spec
    extras: Dict[str, Any] = {}
    df = pd.read_csv(local) if local.suffix != ".sqlite" else None
    if df is not None:
        extras["n_rows"] = len(df)
    # Baseline oracle for training tasks so 'beats_baseline' checks have a target.
    if task.stage == Stage.TRAIN and spec.target and df is not None:
        try:
            extras["baseline"] = ml_train.compute_baseline_oracle(
                df, spec.target, spec.task_kind.value, ReferenceClient._encode
            )
        except Exception:
            extras["baseline"] = {}
    return TaskContext(
        dataset_key=spec.key,
        target=spec.target,
        local_path=str(local),
        source_kind=source.kind,
        extras=extras,
    )


def dispatch(task: BenchmarkTask, client: AvalokaClient, source: DataSource,
             ctx: TaskContext) -> Dict[str, Any]:
    uri = source.uri
    target = ctx.target
    if task.stage == Stage.SAMPLE:
        return client.sample(uri, stratify_by=target, sample_size=0.1)
    if task.stage == Stage.ANALYZE:
        return client.analyze_features(uri, target=target)
    if task.stage == Stage.BUILD:
        return client.build_features(uri, target=target)
    if task.stage == Stage.TRANSFORM:
        ops = _transform_ops_for(task, ctx)
        return client.transform(uri, operations=ops)
    if task.stage == Stage.TRAIN:
        kind = get_dataset(task.dataset_key).task_kind.value
        return client.train(uri, target=target, task_kind=kind, heavy=task.heavy,
                            time_budget_s=30)
    if task.stage == Stage.SERVE:
        # serve depends on a model: train one first (reference path) to get a ref.
        kind = get_dataset(task.dataset_key).task_kind.value
        trained = client.train(uri, target=target, task_kind=kind, heavy=False,
                                time_budget_s=20)
        model_ref = trained.get("model_path", "__no_model__")
        return client.serve(model_ref, platform="k8s", replicas=2)
    raise ValueError(f"no dispatch for stage {task.stage}")


def _transform_ops_for(task: BenchmarkTask, ctx: TaskContext):
    if task.task_id == "de.transform.clean":
        return [{"op": "to_numeric", "column": "TotalCharges"},
                {"op": "fillna", "value": 0}]
    if task.task_id == "de.transform.aggregate":
        return [{"op": "groupby_agg", "by": "PULocationID",
                 "value": "total_amount", "agg": "sum"}]
    return [{"op": "fillna", "value": 0}]


def run_task(task: BenchmarkTask, client: AvalokaClient, source_kind: str,
             workdir: Path) -> TaskResult:
    handle, source, local = prepare_source(task.dataset_key, source_kind, workdir)
    ctx = build_context(task, handle, source, local)
    start = time.time()
    try:
        result = dispatch(task, client, source, ctx)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    dur = time.time() - start
    return run_checks(task, result, ctx, client_name=getattr(client, "name", "?"),
                      duration_s=dur)
