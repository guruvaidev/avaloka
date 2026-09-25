"""Tests for the measurement harness itself.

A before/after number is only worth reporting if the yardstick is sound, so the
yardstick is tested first:

  * every task's acceptance criteria must REJECT the untransformed input frame
    -- otherwise the task can be "passed" by doing nothing;
  * every task's criteria must be SATISFIED by a hand-written correct solution
    -- otherwise the criteria are impossible and a low score means nothing.

Those two tests are deterministic and need no LLM. The end-to-end pipeline test
is marked ``integration`` because it spends real model calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.blueprint_contract import tasks as task_mod
from tests.blueprint_contract.reference_solutions import REFERENCE_SOLUTIONS
from tests.blueprint_contract.tasks import TASKS, TASKS_BY_ID


def _run_solution(source: str, df: pd.DataFrame) -> pd.DataFrame:
    scope: dict = {}
    exec("import pandas as pd\n" + source, scope)  # noqa: S102 - test fixture code
    return scope["main"](df.copy())


def _missing_datasets() -> list[str]:
    seen = []
    for t in TASKS:
        for path in [t.dataset, *t.extra_datasets.values()]:
            if not path.exists() and str(path) not in seen:
                seen.append(str(path))
    return seen


pytestmark = pytest.mark.skipif(
    bool(_missing_datasets()),
    reason=f"task fixtures not present: {_missing_datasets()}",
)


@pytest.fixture(scope="module")
def frames():
    return {t.id: t.load() for t in TASKS}


def test_every_task_has_a_reference_solution():
    missing = [t.id for t in TASKS if t.id not in REFERENCE_SOLUTIONS]
    assert not missing, f"tasks without a reference solution: {missing}"


def test_every_task_has_at_least_one_check(frames):
    for task in TASKS:
        df = frames[task.id]
        checks = task.check(df.copy(), df)
        assert checks, f"{task.id} defines no acceptance checks"


@pytest.mark.parametrize("task_id", [t.id for t in TASKS])
def test_criteria_reject_the_untransformed_input(task_id, frames):
    """If the raw input passes, the task measures nothing."""
    task = TASKS_BY_ID[task_id]
    df = frames[task_id]
    checks = task.check(df.copy(), df)
    failed = [name for name, ok, _ in checks if not ok]
    assert failed, (
        f"{task_id}: every acceptance check passed on the untransformed input, "
        "so this task cannot distinguish a correct answer from doing nothing"
    )


@pytest.mark.parametrize("task_id", [t.id for t in TASKS])
def test_criteria_are_satisfied_by_a_correct_solution(task_id, frames):
    """If correct code cannot pass, the criteria are impossible."""
    task = TASKS_BY_ID[task_id]
    df = frames[task_id]
    result = _run_solution(REFERENCE_SOLUTIONS[task_id], df)
    checks = task.check(result, df)
    failed = [f"{name}: {detail}" for name, ok, detail in checks if not ok]
    assert not failed, f"{task_id} reference solution failed:\n" + "\n".join(failed)


def test_task_families_cover_the_complex_operations():
    """The corpus is meant to exercise operations the simple-prompt path never
    reaches. Keep that true as tasks are added."""
    families = set(task_mod.families())
    for required in {"join", "window", "pivot", "timeseries", "groupby_transform", "chain"}:
        assert required in families, f"no task covers {required}"


def test_fixtures_are_present_and_non_trivial(frames):
    for task in TASKS:
        df = frames[task.id]
        assert len(df) > 100, f"{task.id}: fixture too small to be meaningful"
        assert len(df.columns) >= 3


# ---------------------------------------------------------------------------
# end-to-end (live LLM)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pipeline_runs_a_complex_task_end_to_end(tmp_path):
    """Smoke test of the real coding subgraph on one complex task.

    Marked ``integration``: it needs a configured coder/validator LLM and makes
    real model calls. The full corpus run lives in
    ``tests/blueprint_contract/harness.py``, not in the test suite, because it
    is a measurement rather than an assertion.
    """
    from app.agents import coder as coder_mod

    if coder_mod.coder_llm is None:
        pytest.skip("no coder LLM configured (GROQ_API_KEY_CODING_AGENT)")

    from tests.blueprint_contract.harness import run_task

    task = TASKS_BY_ID["hc_abnormal_rate_by_hospital"]
    outcome = run_task(task, tmp_path)

    assert outcome.coder_invocations >= 1
    assert outcome.graph_error is None
    # Not asserting success: the model is stochastic and this is a smoke test.
    # What must hold is that the run produced a decision rather than crashing.
    assert outcome.failure_stage is not None or outcome.success


@pytest.mark.integration
def test_blueprint_emits_a_parseable_contract():
    """The contract is only useful if the model actually produces one."""
    from app.agents import coder as coder_mod

    if coder_mod.coder_llm is None:
        pytest.skip("no coder LLM configured (GROQ_API_KEY_CODING_AGENT)")

    from app.agents.blueprint_contract import parse_blueprint

    task = TASKS_BY_ID["hc_abnormal_rate_by_hospital"]
    df = task.load()
    state = {
        "user_prompt": task.prompt,
        "plan": task.prompt,
        "schema": {str(c): str(t) for c, t in df.dtypes.items()},
        "uploaded_csv_preview": df.head(20).to_dict(orient="records"),
    }
    raw = coder_mod._generate_pseudocode(state)
    steps, contract, problems = parse_blueprint(raw, schema=state["schema"])

    assert steps, "blueprint produced no prose steps"
    assert contract is not None, f"no usable contract: {problems}"
    assert contract.result_kind == "aggregate"
