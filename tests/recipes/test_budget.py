"""Every run has a budget it cannot exceed (design §3.2 rule 4, §5, §9.3).

Over the row or byte limit the run samples and says so; over the sampled
lane's ceiling it refuses ``too_large``; past its wall clock it refuses
``engine_timeout``. Each breach is a verdict in plain words — no traceback,
no exception text, no silently truncated answer.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

import avaloka.recipes.library as lib
from avaloka import workload
from avaloka.outcomes import semantics as _sem
from avaloka.recipes import Intent, Verdict
from avaloka.recipes import engine
from avaloka.recipes.budget import DEFAULT, ENV_NAMES, Budget, from_env
from avaloka.recipes.report import rendered_slots
from avaloka.recipes.run import REASON_WORDS, build_profile, load_and_run, run

DESCRIBE = Intent("q", "describe", {"measure": "revenue"}, 1.0, "fixture")
TREND = Intent("q", "trend", {"measure": "revenue", "time": "date", "grain": "month"}, 1.0, "fixture")
# hash() per row, so no engine can answer it from the range's bounds: DuckDB 1.5+
# counts range(10^10) in about a second, which raced the one-second budget.
SLOW_SQL = "SELECT sum(hash(range)) FROM range(10000000000)"
LEAKS = ("Traceback", "Exception", "duckdb", "INTERRUPT", "Out of Memory")


def _frame(n, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=730, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"region": rng.choice(["north", "south", "east", "west"], n),
                         "date": rng.choice(days, n),
                         "revenue": np.round(rng.lognormal(4, 0.5, n), 2),
                         "units": rng.integers(1, 50, n)})


def _figures(result):
    return [(f["id"], f["value"]) for f in result.report["figures"]]


def _assert_same_figures(first, second):
    """The same sampled rows give the same figures. Raw values are compared
    to within floating-point summation order (the engine aggregates in
    parallel), rendered sentences exactly."""
    for (fid, a), (gid, b) in zip(_figures(first), _figures(second), strict=True):
        assert fid == gid and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (fid, a, b)
    assert list(rendered_slots(first.report)) == list(rendered_slots(second.report))


# ── the budget itself ──────────────────────────────────────────────────────

def test_default_budget_is_the_online_lane():
    assert DEFAULT.max_rows == workload.ONLINE_MAX_ROWS
    assert DEFAULT.max_bytes == workload.ONLINE_MAX_BYTES
    assert DEFAULT.sample_rows == workload.ONLINE_MAX_ROWS
    assert (DEFAULT.memory_limit, DEFAULT.threads, DEFAULT.wall_clock_s, DEFAULT.seed) == ("512MB", 2, 60.0, 42)


@pytest.mark.parametrize("bad", [
    dict(max_rows=0), dict(max_rows=-1), dict(max_rows=1.5), dict(max_rows=True),
    dict(max_bytes=0), dict(sample_rows=0), dict(threads=0), dict(threads="2"),
    dict(wall_clock_s=0), dict(wall_clock_s=-3.0), dict(wall_clock_s="60"),
    dict(memory_limit="512"), dict(memory_limit="512mb"), dict(memory_limit="1.5GB"), dict(memory_limit="512 MB"),
    dict(memory_limit="512KB"), dict(memory_limit=512), dict(seed=-1), dict(seed=1.0),
])
def test_budget_rejects_bad_values_at_construction(bad):
    with pytest.raises(ValueError, match="Budget\\."):
        Budget(**bad)


def test_budget_is_frozen():
    with pytest.raises(Exception):
        DEFAULT.max_rows = 1  # type: ignore[misc]


def test_from_env_keeps_defaults_when_nothing_is_set():
    assert from_env({}) == DEFAULT
    assert from_env({ENV_NAMES["max_rows"]: "  "}) == DEFAULT


def test_from_env_applies_every_override():
    env = {"AVALOKA_RECIPE_MAX_ROWS": "1000", "AVALOKA_RECIPE_MAX_BYTES": "2048", "AVALOKA_RECIPE_SAMPLE_ROWS": "500",
           "AVALOKA_RECIPE_MEMORY_LIMIT": "2gb", "AVALOKA_RECIPE_THREADS": "4", "AVALOKA_RECIPE_WALL_CLOCK_S": "12.5",
           "AVALOKA_RECIPE_SEED": "7"}
    assert set(env) == set(ENV_NAMES.values())
    assert from_env(env) == Budget(max_rows=1000, max_bytes=2048, sample_rows=500, memory_limit="2GB", threads=4,
                                   wall_clock_s=12.5, seed=7)


@pytest.mark.parametrize("name, value", [
    ("AVALOKA_RECIPE_MAX_ROWS", "lots"), ("AVALOKA_RECIPE_THREADS", "0"), ("AVALOKA_RECIPE_WALL_CLOCK_S", "soon"),
    ("AVALOKA_RECIPE_MEMORY_LIMIT", "half"),
])
def test_from_env_names_the_variable_it_could_not_read(name, value):
    with pytest.raises(ValueError, match="AVALOKA_RECIPE_"):
        from_env({name: value})


def test_from_env_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv("AVALOKA_RECIPE_WALL_CLOCK_S", "5")
    assert from_env().wall_clock_s == 5.0


# ── rows and bytes: sampling ───────────────────────────────────────────────

def test_connect_keeps_the_memory_limit_and_threads_keywords():
    con = engine.connect(_frame(10), memory_limit="256MB", threads=1)
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    con = engine.connect(_frame(10), budget=Budget(threads=1, memory_limit="128MB"))
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    assert int(con.execute("SELECT count(*) FROM t").fetchone()[0]) == 10


def test_a_source_over_max_rows_is_sampled_and_the_report_says_so():
    df = _frame(150_000)
    result = load_and_run(lib.DESCRIBE, DESCRIBE, df)
    # Sampled figures are estimates: the run is qualified even with every gate passed.
    assert result.verdict is Verdict.QUALIFIED, (result.verdict, result.reason, result.refusal)
    assert result.report["method"]["sample"] == {"rows_analyzed": 100_000, "total_rows": 150_000, "is_sampled": True}
    caveats = [c for c in result.report["caveats"] if c["kind"] == "sampling"]
    assert len(caveats) == 1 and "computed on a sample of the rows" in caveats[0]["text"]
    assert not any(ch.isdigit() for ch in caveats[0]["text"])
    spent = result.budget_report
    assert spent is not None and spent.sampled and (spent.total_rows, spent.rows_analyzed) == (150_000, 100_000)
    assert spent.total_bytes == int(df.memory_usage(deep=True).sum()) and not spent.timed_out
    assert (spent.memory_limit, spent.threads) == ("512MB", 2) and 0 < spent.elapsed_s < 60
    # the figures were computed on the sample: the evidence counts 100k rows, the report's sample says 150k
    assert result.evidence.tree["rows"]["raw"]["value"] == 100_000
    assert result.report["evidence"]["rows"]["raw"]["value"] == 100_000


def test_the_same_source_samples_the_same_rows_and_the_same_figures(tmp_path):
    df = _frame(150_000)
    pq = tmp_path / "big.parquet"
    df.to_parquet(pq)

    def sample(source):
        con = engine.connect(source)
        info = engine.register(con, source, DEFAULT)
        assert info == engine.SourceInfo(150_000, info.total_bytes, True, 100_000)
        return engine.query(con, "SELECT * FROM t ORDER BY region, date, revenue, units")

    assert sample(pq).equals(sample(pq))
    assert sample(df).equals(sample(df))
    assert sample(pq).equals(sample(str(pq)))
    first, second = load_and_run(lib.DESCRIBE, DESCRIBE, pq), load_and_run(lib.DESCRIBE, DESCRIBE, pq)
    assert first.verdict in (Verdict.PASS, Verdict.QUALIFIED)
    _assert_same_figures(first, second)
    assert first.report["method"]["sample"] == second.report["method"]["sample"]
    assert first.budget_report.total_bytes == pq.stat().st_size


def test_a_different_seed_samples_different_rows():
    df = _frame(150_000)
    con = engine.connect(df)
    engine.register(con, df, DEFAULT)
    a = engine.query(con, "SELECT sum(revenue) FROM t").iloc[0, 0]
    con = engine.connect(df)
    engine.register(con, df, Budget(seed=43))
    assert a != engine.query(con, "SELECT sum(revenue) FROM t").iloc[0, 0]


def test_a_source_under_the_budget_is_not_sampled():
    df = _frame(50_000)
    result = load_and_run(lib.DESCRIBE, DESCRIBE, df)
    assert result.verdict in (Verdict.PASS, Verdict.QUALIFIED), (result.reason, result.refusal)
    assert result.report["method"]["sample"] == {"rows_analyzed": 50_000, "total_rows": 50_000, "is_sampled": False}
    assert not any(c["kind"] == "sampling" for c in result.report["caveats"])
    assert result.budget_report is not None and not result.budget_report.sampled
    assert (result.budget_report.total_rows, result.budget_report.rows_analyzed) == (50_000, 50_000)
    con = engine.connect(df)
    assert engine.register(con, df, DEFAULT) == engine.SourceInfo(50_000, int(df.memory_usage(deep=True).sum()),
                                                                  False, 50_000)


def test_a_parquet_file_over_max_bytes_is_sampled_by_bytes(tmp_path):
    df = _frame(30_000)
    pq = tmp_path / "wide.parquet"
    df.to_parquet(pq)
    size = pq.stat().st_size
    budget = Budget(max_bytes=size // 2, sample_rows=5_000)
    assert 30_000 < budget.max_rows
    con = engine.connect(pq, budget=budget)
    info = engine.register(con, pq, budget)
    assert info == engine.SourceInfo(30_000, size, True, 5_000)
    assert int(con.execute("SELECT count(*) FROM t").fetchone()[0]) == 5_000
    result = load_and_run(lib.TREND, TREND, pq, budget=budget)
    assert result.verdict is Verdict.QUALIFIED, (result.verdict, result.reason, result.refusal)
    assert result.report["method"]["sample"] == {"rows_analyzed": 5_000, "total_rows": 30_000, "is_sampled": True}
    assert any(c["kind"] == "sampling" for c in result.report["caveats"])


def test_load_and_run_profiles_the_sample_not_the_file(monkeypatch):
    """The whole file is never read into pandas: the profile comes from ``t``."""
    df = _frame(150_000)
    seen = []
    real = engine.query

    def spy(con, sql):
        out = real(con, sql)
        seen.append((sql, len(out)))
        return out

    monkeypatch.setattr("avaloka.recipes.run.query", spy)
    monkeypatch.setattr("avaloka.recipes.run.read_frame", lambda s: pytest.fail("read_frame read the whole source"))
    result = load_and_run(lib.DESCRIBE, DESCRIBE, df)
    assert result.verdict in (Verdict.PASS, Verdict.QUALIFIED)
    assert seen[0] == ("SELECT * FROM t", 100_000)


def test_build_profile_reports_total_rows_and_the_rows_analyzed():
    profile = build_profile(_frame(100), total_rows=5_000)
    assert (profile["rows"], profile["rows_analyzed"]) == (5_000, 100)
    assert build_profile(_frame(100))["rows"] == 100


def test_a_source_over_the_sampled_lane_is_refused_too_large(monkeypatch):
    monkeypatch.setattr(engine, "source_size", lambda con, source: (workload.SAMPLED_MAX_ROWS + 1, 1024))
    result = load_and_run(lib.DESCRIBE, DESCRIBE, _frame(100))
    assert result.verdict is Verdict.REFUSE and result.reason == "too_large"
    assert result.refusal.text == REASON_WORDS["too_large"]
    assert "smaller extract" in result.refusal.text and not any(w in result.refusal.text for w in LEAKS)
    monkeypatch.setattr(engine, "source_size", lambda con, source: (10, workload.SAMPLED_MAX_BYTES + 1))
    assert load_and_run(lib.DESCRIBE, DESCRIBE, _frame(100)).reason == "too_large"


# ── the wall clock ─────────────────────────────────────────────────────────

def test_timed_interrupts_a_slow_query_at_the_limit():
    import duckdb
    con = engine.connect(_frame(10))
    started = time.monotonic()
    with pytest.raises(duckdb.InterruptException) as info:
        with engine.timed(con, 1.0) as clock:
            con.execute(SLOW_SQL).fetchone()
    assert 0.9 < time.monotonic() - started < 2.0
    assert clock.expired and engine.interrupted(info.value)
    assert con.execute("SELECT 1").fetchone() == (1,)  # the connection survives the interrupt


def test_timed_cancels_the_timer_when_the_work_finishes_first():
    con = engine.connect(_frame(10))
    with engine.timed(con, 30.0) as clock:
        con.execute("SELECT 1").fetchone()
    assert not clock.expired and clock.elapsed < 1.0
    assert con.execute("SELECT 2").fetchone() == (2,)


class _Slow:
    """``describe`` with a quality check that runs a query past any budget."""

    def __init__(self, inner, sql=SLOW_SQL):
        self.inner, self.sql = inner, sql

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def check(self, con, bound):
        con.execute(self.sql).fetchone()
        return self.inner.check(con, bound)


class _Sleepy(_Slow):
    """``describe`` with a quality check that spends its time in Python."""

    def check(self, con, bound):
        time.sleep(1.2)
        return self.inner.check(con, bound)


class _Raises(_Slow):
    def __init__(self, inner, exc):
        super().__init__(inner)
        self.exc = exc

    def check(self, con, bound):
        raise self.exc


def _assert_plain_refusal(result, reason):
    assert result.verdict is Verdict.REFUSE and result.reason == reason, (result.verdict, result.reason)
    assert result.report is None and result.refusal.reason == reason
    assert result.refusal.text == REASON_WORDS[reason]
    assert not any(w in result.refusal.text for w in LEAKS)


def test_run_ends_engine_timeout_in_plain_words_when_the_query_is_interrupted():
    df = _frame(1_000)
    budget = Budget(wall_clock_s=1.0)
    started = time.monotonic()
    result = run(_Slow(lib.DESCRIBE), DESCRIBE, engine.connect(df, budget=budget), _sem.infer(df),
                 build_profile(df), budget=budget)
    assert time.monotonic() - started < 3.0
    _assert_plain_refusal(result, "engine_timeout")
    assert result.refusal.text == "The analysis did not finish within its time budget."
    spent = result.budget_report
    assert spent is not None and spent.timed_out and 0.9 < spent.elapsed_s < 2.5
    assert (spent.total_rows, spent.rows_analyzed, spent.sampled) == (1_000, 1_000, False)


def test_load_and_run_ends_engine_timeout_with_the_sources_size():
    df = _frame(150_000)
    result = load_and_run(_Slow(lib.DESCRIBE), DESCRIBE, df, budget=Budget(wall_clock_s=1.0))
    _assert_plain_refusal(result, "engine_timeout")
    spent = result.budget_report
    assert spent.timed_out and spent.sampled and (spent.total_rows, spent.rows_analyzed) == (150_000, 100_000)


def test_a_run_that_overruns_in_python_is_still_over_budget():
    df = _frame(1_000)
    result = load_and_run(_Sleepy(lib.DESCRIBE), DESCRIBE, df, budget=Budget(wall_clock_s=0.5))
    _assert_plain_refusal(result, "engine_timeout")
    assert result.budget_report.timed_out and result.budget_report.elapsed_s >= 1.2


def test_a_run_within_its_clock_passes_with_a_budget_report():
    df = _frame(1_000)
    result = load_and_run(lib.DESCRIBE, DESCRIBE, df, budget=Budget(wall_clock_s=30.0))
    assert result.verdict in (Verdict.PASS, Verdict.QUALIFIED)
    assert result.budget_report is not None and not result.budget_report.timed_out
    assert result.report["method"]["sample"]["is_sampled"] is False


def test_an_out_of_memory_engine_ends_too_large_in_plain_words(caplog):
    import duckdb
    df = _frame(1_000)
    exc = duckdb.OutOfMemoryException("Out of Memory Error: could not allocate block of size 256.0 KiB")
    with caplog.at_level("WARNING", logger="avaloka.recipes.run"):
        result = run(_Raises(lib.DESCRIBE, exc), DESCRIBE, engine.connect(df), _sem.infer(df), build_profile(df))
    _assert_plain_refusal(result, "too_large")
    assert result.budget_report is not None and not result.budget_report.timed_out
    assert "ran out of memory" in caplog.text and "could not allocate" in caplog.text  # the log has the detail


def test_a_timeout_or_too_large_report_never_quotes_the_engine(caplog):
    import duckdb
    df = _frame(1_000)
    for exc, reason in ((duckdb.InterruptException("INTERRUPT Error: Interrupted!"), "engine_timeout"),
                        (duckdb.OutOfMemoryException("Out of Memory Error: 61.0 MiB used"), "too_large")):
        result = run(_Raises(lib.DESCRIBE, exc), DESCRIBE, engine.connect(df), _sem.infer(df), build_profile(df))
        _assert_plain_refusal(result, reason)
        assert str(exc) not in result.refusal.text


def test_too_large_is_the_last_reason_and_has_words():
    from avaloka.recipes import REASONS
    assert REASONS[-1] == "too_large" and "engine_timeout" in REASONS
    assert set(REASON_WORDS) == set(REASONS)


# ── the notebook of a sampled run ──────────────────────────────────────────

def _cell(notebook, tag=None, contains=None):
    for c in notebook["cells"]:
        text = "".join(c["source"])
        if (tag and tag in c["metadata"].get("tags", [])) or (contains and contains in text):
            return text
    raise AssertionError(f"no cell {tag or contains!r}")


def test_every_recipes_notebook_declares_the_sample_parameters_and_the_engines_statement():
    """All ten parameters cells say SAMPLE_ROWS = None / SAMPLE_SEED = 42, and
    every load cell carries ``engine.SAMPLE_SQL`` with the notebook's own
    parameters in the template's slots — the same statement register() runs."""
    from avaloka.recipes import all_recipes
    from tests.recipes._golden import GOLDEN, make_frame
    import json
    notebook_sql = engine.SAMPLE_SQL.replace("{rows}", "{SAMPLE_ROWS}").replace("{seed}", "{SAMPLE_SEED}")
    assert notebook_sql.format(SAMPLE_ROWS=7, SAMPLE_SEED=3) == engine.SAMPLE_SQL.format(rows=7, seed=3)
    seen = 0
    for recipe in all_recipes():
        case = next(c for c in sorted((GOLDEN / recipe.spec.id).iterdir())
                    if (c / "expected.json").is_file()
                    and json.loads((c / "expected.json").read_text())["verdict"] in ("pass", "qualified"))
        intent = json.loads((case / "intent.json").read_text())
        result = load_and_run(recipe, Intent(**intent), make_frame(case))
        assert result.notebook is not None, (recipe.spec.id, result.verdict, result.reason)
        params = _cell(result.notebook, tag="parameters").splitlines()
        assert "SAMPLE_ROWS = None" in params and f"SAMPLE_SEED = {DEFAULT.seed}" in params, (recipe.spec.id, params)
        load = _cell(result.notebook, contains="create_view(\"t\")")
        assert f'con.execute(f"{notebook_sql}")' in load and f'con.execute("{engine.RENAME_SQL}")' in load
        assert "if SAMPLE_ROWS is not None" in load
        seen += 1
    assert seen == 10


def test_a_sampled_runs_notebook_carries_the_sample_it_was_computed_on():
    df = _frame(150_000)
    result = load_and_run(lib.DESCRIBE, DESCRIBE, df)
    assert result.verdict is Verdict.QUALIFIED and result.notebook is not None
    params = _cell(result.notebook, tag="parameters").splitlines()
    assert "SAMPLE_ROWS = 100000" in params and "SAMPLE_SEED = 42" in params, params
    assert len([c for c in result.notebook["cells"] if "parameters" in c["metadata"].get("tags", [])]) == 1
    seeded = load_and_run(lib.DESCRIBE, DESCRIBE, df, budget=Budget(seed=9, sample_rows=20_000))
    assert "SAMPLE_ROWS = 20000" in _cell(seeded.notebook, tag="parameters").splitlines()
    assert "SAMPLE_SEED = 9" in _cell(seeded.notebook, tag="parameters").splitlines()
    plain = load_and_run(lib.DESCRIBE, DESCRIBE, _frame(1_000))
    assert "SAMPLE_ROWS = None" in _cell(plain.notebook, tag="parameters").splitlines()
