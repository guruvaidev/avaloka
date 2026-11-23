"""The notebook reproduces the report (design §2.4).

For every golden case that produces a report, the recipe's generated notebook
is executed in a fresh kernel on the case's data (a CSV in a temporary
directory, passed through the ``parameters`` cell) and every figure of the
report must match the notebook's evidence leaf at its ``evidence_ref``, by
the report's own rounding rule. Skipped, not failed, when the
``avaloka[recipes-test]`` extra is absent.
"""

from __future__ import annotations

import json

import pytest

from avaloka.outcomes import spec as _spec
from avaloka.recipes import notebook_exec
from tests.recipes._golden import assert_notebook_parity, case_id, cases, execute_notebook, make_frame, run_case

MISSING = notebook_exec.missing_dependency()
pytestmark = [pytest.mark.notebook, pytest.mark.skipif(MISSING is not None, reason=MISSING or "")]


def reported_cases():
    """Cases whose expected verdict carries a report, so a notebook exists."""
    return [c for c in cases() if json.loads((c / "expected.json").read_text())["verdict"] in ("pass", "qualified")]


def test_every_recipe_has_a_reported_case():
    from avaloka.recipes import RECIPE_IDS
    assert {c.parent.name for c in reported_cases()} == set(RECIPE_IDS)


@pytest.mark.parametrize("case", reported_cases(), ids=case_id)
def test_the_notebook_reproduces_the_report(case):
    assert_notebook_parity(case)


def test_the_parameters_cell_is_what_the_notebook_reads():
    """The same notebook on a two-row copy of the data reports two rows: the
    overridden ``DATA_SOURCE`` is what ran, not the source the recipe saw."""
    case = next(c for c in cases() if case_id(c) == "describe/clean_by_dimension")
    _, result = run_case(case)
    evidence = execute_notebook(case, result.notebook, frame=make_frame(case).head(2))
    assert evidence["measure"]["n"]["value"] == 2.0
    assert result.report["evidence"]["measure"]["n"]["value"] != 2.0


# ── a sampled run ──────────────────────────────────────────────────────────

def _frame(n=3_000, seed=19):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=730, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"region": rng.choice(["north", "south", "east", "west"], n),
                         "date": rng.choice(days, n),
                         "revenue": np.round(rng.lognormal(4, 0.5, n), 2)})


def _execute_on(notebook: dict, csv) -> dict:
    import os
    from tests.recipes._golden import REPO
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    return notebook_exec.execute(notebook, parameters={"DATA_SOURCE": str(csv)}, env={"PYTHONPATH": path})


@pytest.mark.parametrize("recipe_id, params", [
    ("describe", {"measure": "revenue"}),
    ("trend", {"measure": "revenue", "time": "date", "grain": "month"}),
])
def test_a_sampled_runs_notebook_reproduces_the_sampled_report(tmp_path, recipe_id, params):
    """A run over the budget is computed on a sample; the notebook it ships
    takes the same sample (``SAMPLE_ROWS`` / ``SAMPLE_SEED`` in its
    ``parameters`` cell, ``engine.SAMPLE_SQL`` in its load cell), so every
    figure of the sampled report is reproduced — and reproduced again."""
    from avaloka.recipes import Intent, Verdict, get
    from avaloka.recipes.budget import Budget
    from avaloka.recipes.run import load_and_run
    from tests.recipes._golden import assert_figures_reproduced

    csv = tmp_path / "data.csv"
    _frame().to_csv(csv, index=False)
    budget = Budget(max_rows=1_000, sample_rows=1_000)
    result = load_and_run(get(recipe_id), Intent("q", recipe_id, params, 1.0, "fixture"), csv, budget=budget)
    assert result.verdict is Verdict.QUALIFIED, (result.verdict, result.reason, result.refusal, result.ask)
    assert result.report["method"]["sample"] == {"rows_analyzed": 1_000, "total_rows": 3_000, "is_sampled": True}
    assert result.notebook is not None

    first = _execute_on(result.notebook, csv)
    assert_figures_reproduced(result.report, first, f"{recipe_id} (sampled)")
    # The sample's row count itself is in the evidence: the notebook read 1,000 rows, not 3,000.
    assert first["rows"]["raw"]["value"] == 1_000.0 == result.report["evidence"]["rows"]["raw"]["value"]

    # The parameters cell the kernel ran (DATA_SOURCE overridden, the rest kept) carries the sample.
    executed = notebook_exec.apply_parameters(result.notebook, {"DATA_SOURCE": str(csv)})
    cell = next(c for c in executed["cells"] if "parameters" in c["metadata"].get("tags", []))
    lines = "".join(cell["source"]).splitlines()
    assert "SAMPLE_ROWS = 1000" in lines and f"SAMPLE_SEED = {budget.seed}" in lines, lines

    second = _execute_on(result.notebook, csv)
    assert_figures_reproduced(result.report, second, f"{recipe_id} (sampled, second execution)")
    for fig in result.report["figures"]:
        pointer = fig["evidence_ref"]
        a, b = _spec.evidence_leaf(first, pointer)[0], _spec.evidence_leaf(second, pointer)[0]
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), (pointer, a, b)
