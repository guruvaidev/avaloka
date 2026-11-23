"""Data-science benchmark suite (feature engineering, training, serving)."""

from __future__ import annotations

import pytest

from tests.benchmark.catalog import DS_TASKS
from tests.benchmark.harness import run_task

_LIGHT = [t for t in DS_TASKS if not t.heavy]
_HEAVY = [t for t in DS_TASKS if t.heavy]


@pytest.mark.parametrize("task", _LIGHT, ids=[t.task_id for t in _LIGHT])
def test_ds_task(task, client, workdir):
    result = run_task(task, client, "local", workdir)
    failed = [o for o in result.outcomes if not o.passed]
    detail = "; ".join(f"{o.name}({o.severity.value}): {o.detail}" for o in failed)
    assert result.passed, (
        f"{task.task_id} scored {result.score} (client={result.client}); "
        f"failures: {detail}"
    )


@pytest.mark.heavy
@pytest.mark.slow
@pytest.mark.parametrize("task", _HEAVY, ids=[t.task_id for t in _HEAVY])
def test_ds_task_heavy(task, client, workdir):
    pytest.importorskip("torch", reason="heavy DL path needs PyTorch")
    result = run_task(task, client, "local", workdir)
    failed = [o for o in result.outcomes if not o.passed]
    detail = "; ".join(f"{o.name}({o.severity.value}): {o.detail}" for o in failed)
    assert result.passed, (
        f"{task.task_id} scored {result.score} (client={result.client}); "
        f"failures: {detail}"
    )


def test_baseline_oracle_is_meaningful(workdir):
    """Sanity: the baseline oracle should learn signal on synthetic data."""
    import pandas as pd

    from tests.benchmark import ml_train
    from tests.benchmark.adapters import ReferenceClient
    from tests.benchmark.datasets import materialize
    from tests.benchmark.registry import get

    spec = get("titanic")
    handle = materialize(spec)
    df = pd.read_csv(handle.path)
    metrics = ml_train.compute_baseline_oracle(
        df, spec.target, spec.task_kind.value, ReferenceClient._encode
    )
    assert metrics["accuracy"] >= 0.55, f"oracle too weak: {metrics}"
