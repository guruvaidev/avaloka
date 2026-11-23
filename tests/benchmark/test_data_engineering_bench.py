"""Data-engineering benchmark suite."""

from __future__ import annotations

import pytest

from tests.benchmark.catalog import DE_TASKS
from tests.benchmark.harness import run_task

# Expand (task x source_kind) into individual parametrized cases.
_CASES = [
    (task, sk)
    for task in DE_TASKS
    for sk in task.source_kinds
]


@pytest.mark.parametrize(
    "task,source_kind",
    _CASES,
    ids=[f"{t.task_id}[{sk}]" for t, sk in _CASES],
)
def test_de_task(task, source_kind, client, workdir, sqlite_reader_for):
    sqlite_reader_for(source_kind)
    result = run_task(task, client, source_kind, workdir)
    failed = [o for o in result.outcomes if not o.passed]
    detail = "; ".join(f"{o.name}({o.severity.value}): {o.detail}" for o in failed)
    assert result.passed, (
        f"{task.task_id} on {source_kind} scored {result.score} "
        f"(client={result.client}); failures: {detail}"
    )


def test_multisource_schema_parity(client, workdir, sqlite_reader_for):
    """Same dataset, different sources -> same column schema from sampling."""
    from tests.benchmark.catalog import DE_TASKS as tasks

    task = next(t for t in tasks if t.task_id == "de.sample.multisource")
    kinds = ("local", "sqlite", "gcs")
    sqlite_reader_for(*kinds)
    schemas = {}
    for sk in kinds:
        from tests.benchmark.harness import prepare_source, build_context, dispatch

        handle, source, local = prepare_source(task.dataset_key, sk, workdir)
        ctx = build_context(task, handle, source, local)
        out = dispatch(task, client, source, ctx)
        schemas[sk] = sorted(out.get("schema") or [])
    base = schemas["local"]
    assert base, "local schema empty"
    for sk, sch in schemas.items():
        assert sch == base, f"schema drift on {sk}: {sch} != {base}"
