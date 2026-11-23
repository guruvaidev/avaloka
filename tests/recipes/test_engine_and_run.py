"""The engine, the report builder and the runner, apart from the goldens."""

from __future__ import annotations

import builtins
import json

import numpy as np
import pandas as pd
import pytest

import avaloka.recipes.library as lib
from avaloka import notebook as nb
from avaloka.outcomes import semantics as _sem
from avaloka.recipes import Intent, Verdict, get
from avaloka.recipes import engine
from avaloka.recipes.report import ReportInvalid, verify_report
from avaloka.recipes.run import build_profile, load_and_run, run


def _frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": rng.choice(["a", "b", "c"], n),
                         "date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                         "revenue": np.round(rng.lognormal(4, 0.5, n), 2)})


def test_connect_registers_a_frame_as_t_and_query_returns_a_frame():
    con = engine.connect(_frame(), memory_limit="256MB", threads=1)
    out = engine.query(con, "SELECT count(*) AS n FROM t")
    assert isinstance(out, pd.DataFrame) and int(out["n"][0]) == 120


def test_connect_reads_csv_and_parquet_by_extension(tmp_path):
    df = _frame(30)
    csv, pq = tmp_path / "d.csv", tmp_path / "d.parquet"
    df.to_csv(csv, index=False)
    assert engine.query(engine.connect(csv), "SELECT count(*) FROM t").iloc[0, 0] == 30
    try:
        df.to_parquet(pq)
    except ImportError:
        pytest.skip("no parquet engine in this environment")
    assert engine.query(engine.connect(pq), "SELECT count(*) FROM t").iloc[0, 0] == 30


def test_missing_duckdb_is_reported_in_plain_words(monkeypatch):
    real_import = builtins.__import__

    def no_duckdb(name, *a, **k):
        if name == "duckdb":
            raise ImportError("No module named 'duckdb'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_duckdb)
    assert "pip install 'avaloka[recipes]'" in engine.missing_dependency()
    with pytest.raises(engine.RecipesDependencyError, match="avaloka\\[recipes\\]"):
        engine.connect(_frame())
    result = load_and_run(lib.DESCRIBE, Intent("q", "describe", {"measure": "revenue"}, 1.0, "fixture"), _frame())
    assert result.verdict is Verdict.REFUSE and result.reason == "missing_dependency"


def test_missing_dependency_is_none_when_duckdb_is_present():
    assert engine.missing_dependency() is None


def test_library_registers_the_three_recipes():
    assert [r.spec.id for r in (get("profile"), get("describe"), get("trend"))] == ["profile", "describe", "trend"]


def test_library_registers_all_ten_recipes():
    from avaloka.recipes import RECIPE_IDS, all_recipes
    assert tuple(r.spec.id for r in all_recipes()) == RECIPE_IDS
    assert all(r.spec.version == "1.0" for r in all_recipes())
    # ab_test is the one recipe that measures a tested effect; every other one reads a pattern.
    assert {r.spec.id for r in all_recipes() if r.spec.evidence_kind == "experimental"} == {"ab_test"}
    assert all(r.spec.evidence_kind == "associational" for r in all_recipes() if r.spec.id not in ("profile", "ab_test"))


@pytest.mark.parametrize("recipe_id, params, module_sql", [
    ("compare_periods", {"measure": "revenue", "time": "date"}, "SQL_WINDOWS"),
    ("segment", {"measure": "revenue", "dimension": "region"}, "SQL_GROUPS"),
    ("correlate", {"measures": ["revenue", "cost"]}, "SQL_VALUES"),
    ("regress", {"outcome": "revenue", "measures": ["cost"]}, "SQL_VALUES"),
    ("anomalies", {"measure": "revenue", "time": "date"}, "SQL_RESIDUALS"),
])
def test_each_new_notebook_carries_the_same_sql_the_recipe_ran(recipe_id, params, module_sql):
    import importlib
    module = importlib.import_module(f"avaloka.recipes.library.{recipe_id}")
    df = _frame(400).assign(cost=lambda d: d["revenue"] * 0.5 + np.random.default_rng(5).normal(0, 5, len(d)))
    result = load_and_run(get(recipe_id), Intent("q", recipe_id, params, 1.0, "fixture"), df)
    assert result.verdict in (Verdict.PASS, Verdict.QUALIFIED), (result.reason, result.refusal, result.ask)
    sources = ["".join(c["source"]) for c in result.notebook["cells"]]
    assert any(repr(getattr(module, module_sql)) in s for s in sources)
    params_cells = [c for c in result.notebook["cells"] if c["metadata"].get("tags") == ["parameters"]]
    assert len(params_cells) == 1 and f"RECIPE = {recipe_id!r}" in "".join(params_cells[0]["source"])


@pytest.mark.parametrize("recipe_id, case, module_sql", [
    ("cohort_retention", "clean_monthly", "SQL_MATRIX"),
    ("ab_test", "binary_clear_lift", "SQL_ARMS"),
])
def test_experiment_notebooks_carry_the_same_sql_the_recipe_ran(recipe_id, case, module_sql):
    """The cohort and experiment recipes need an id, a date or an arm column
    the shared frame lacks, so their golden generators stand in for it."""
    import importlib
    from tests.recipes._golden import GOLDEN, make_frame
    module = importlib.import_module(f"avaloka.recipes.library.{recipe_id}")
    folder = GOLDEN / recipe_id / case
    intent = json.loads((folder / "intent.json").read_text())
    result = load_and_run(get(recipe_id), Intent(**intent), make_frame(folder))
    assert result.verdict is Verdict.PASS, (result.reason, result.refusal, result.ask)
    sources = ["".join(c["source"]) for c in result.notebook["cells"]]
    assert any(repr(getattr(module, module_sql)) in s for s in sources)
    params_cells = [c for c in result.notebook["cells"] if c["metadata"].get("tags") == ["parameters"]]
    assert len(params_cells) == 1 and f"RECIPE = {recipe_id!r}" in "".join(params_cells[0]["source"])


def test_every_reported_notebook_ends_with_an_evidence_cell():
    """The parameters cell is the first code cell and the evidence cell the
    last, tagged, and prints its dict as JSON: the shape the parity test and
    any runner read."""
    from tests.recipes._golden import cases, run_case
    seen = 0
    for case in cases():
        _, result = run_case(case)
        if result.notebook is None:
            continue
        code = [c for c in result.notebook["cells"] if c["cell_type"] == "code"]
        assert code[0]["metadata"].get("tags") == ["parameters"], case
        assert code[-1]["metadata"].get("tags") == ["evidence"], case
        assert "print(json.dumps(evidence, default=float))" in "".join(code[-1]["source"]), case
        assert all("id" in c for c in result.notebook["cells"]), case
        seen += 1
    assert seen >= 30


def test_run_marks_a_sampled_report():
    df = _frame()
    result = run(lib.DESCRIBE, Intent("q", "describe", {"measure": "revenue"}, 1.0, "fixture"),
                 engine.connect(df), _sem.infer(df), build_profile(df), sampled=True)
    # Sampled figures are estimates, so the run is qualified even with every gate passed.
    assert result.verdict is Verdict.QUALIFIED
    assert result.gates is not None and result.gates.verdict is Verdict.PASS
    assert result.report["method"]["sample"]["is_sampled"] is True
    assert result.report["caveats"][0]["kind"] == "sampling"
    assert result.report["provenance"]["verifier"]["passed"] is True


def test_run_asks_for_the_measure_when_several_fit():
    df = _frame().assign(cost=1.5)
    result = run(lib.DESCRIBE, Intent("q", "describe", {}, 1.0, "fixture"),
                 engine.connect(df), _sem.infer(df), build_profile(df))
    assert result.verdict is Verdict.ASK and result.ask.parameter == "measure"
    assert set(result.ask.options) == {"revenue", "cost"}


def test_run_refuses_a_measure_that_is_not_one():
    df = _frame()
    result = run(lib.DESCRIBE, Intent("q", "describe", {"measure": "region"}, 1.0, "fixture"),
                 engine.connect(df), _sem.infer(df), build_profile(df))
    assert result.verdict is Verdict.REFUSE and result.reason == "no_measure"


def test_verify_report_lists_a_figure_that_disagrees_with_its_evidence():
    result = load_and_run(lib.PROFILE, Intent("q", "profile", {}, 1.0, "fixture"), _frame())
    report = json.loads(json.dumps(result.report))
    report["figures"][0]["value"] += 1
    with pytest.raises(ReportInvalid, match="/figures/f1"):
        verify_report(report)
    report["headline"]["text"] = "There are 120 rows."
    with pytest.raises(ReportInvalid, match="free numbers"):
        verify_report(report)


def test_notebook_code_cell_tags_are_optional():
    assert nb.code("x = 1")["metadata"] == {}
    assert nb.code("x = 1", tags=["parameters"])["metadata"] == {"tags": ["parameters"]}


def test_notebook_carries_the_same_sql_the_recipe_ran():
    from avaloka.recipes.library import trend
    result = load_and_run(lib.TREND, Intent("q", "trend", {"measure": "revenue", "time": "date", "grain": "month"}, 1.0, "fixture"), _frame())
    assert result.verdict is Verdict.PASS, (result.reason, result.refusal)
    sources = ["".join(c["source"]) for c in result.notebook["cells"]]
    assert any(repr(trend.SQL_PERIODS) in s for s in sources)
    params = [c for c in result.notebook["cells"] if c["metadata"].get("tags") == ["parameters"]]
    assert len(params) == 1
    text = "".join(params[0]["source"])
    for line in ("DATA_SOURCE = 'data.csv'", "measure = 'revenue'", "time = 'date'", "grain = 'month'",
                 "RECIPE = 'trend'", "RECIPE_VERSION = '1.0'"):
        assert line in text, text


def test_trend_reads_an_unambiguous_day_first_column():
    df = _frame(200)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y")
    result = load_and_run(lib.TREND, Intent("q", "trend", {"measure": "revenue", "time": "date", "grain": "month"}, 1.0, "fixture"), df)
    assert result.verdict is Verdict.PASS, (result.reason, result.refusal, result.ask)
    assert result.evidence.tree["periods"]["count"]["value"] == 7
    assert result.evidence.tree["dates"]["unparsed_share"]["value"] == 0.0


def test_a_notebook_failure_keeps_the_verified_report(monkeypatch):
    """The report verified before the notebook is written; a notebook bug must
    not turn a good answer into a refusal."""
    import avaloka.recipes.library  # noqa: F401
    from avaloka.recipes import Intent, Verdict, get
    from avaloka.recipes.run import load_and_run
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(7)
    df = pd.DataFrame({"revenue": rng.normal(100, 10, 200)})
    recipe = get("describe")
    monkeypatch.setattr(type(recipe), "notebook", lambda self, bound, gates=None: 1 / 0)
    out = load_and_run(recipe, Intent("typical revenue?", "describe", {"measure": "revenue"}, 1.0, "fixture"), df)
    assert out.verdict in (Verdict.PASS, Verdict.QUALIFIED)
    assert out.report is not None and out.notebook is None
