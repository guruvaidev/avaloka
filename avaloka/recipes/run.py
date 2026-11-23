"""Run one recipe end to end: bind, check, compute, report, verify, notebook.

The verdict is written by code at each step. A report that fails its own
verification is a recipe bug, so the run ends ``refuse`` with reason
``internal`` and the problem is logged: a wrong number is never shipped.

Every run has a budget (:mod:`avaloka.recipes.budget`). A source over the
row or byte limit is analysed on a sample and the report says so; a run
past its wall clock ends ``refuse`` / ``engine_timeout``; a source the
engine cannot hold ends ``refuse`` / ``too_large``. Each is a verdict in
plain words — nothing from an engine exception reaches the user — and
``RunResult.budget_report`` records what was spent.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from avaloka.outcomes import semantics as _sem
from avaloka.stats import profile_column

from .base import Ask, BudgetReport, Evidence, GateReport, Intent, Recipe, Refusal, RunResult, Verdict
from .budget import DEFAULT, Budget
from .engine import (Clock, RecipesDependencyError, SourceInfo, SourceTooLarge, connect, interrupted, is_parquet,
                     out_of_memory, query, register, timed)
from .library._shared import profile_date_convention
from .notebook_exec import apply_parameters
from .report import SAMPLING_WORDS, ReportInvalid, verify_report

log = logging.getLogger(__name__)

#: Why a failed gate stops the run, in plain words.
REASON_WORDS: dict[str, str] = {
    "insufficient_rows": "There are too few rows for this analysis.",
    "insufficient_periods": "There are too few time periods for this analysis.",
    "no_time_column": "No usable date column was found.",
    "no_measure": "The measure has no values.",
    "ambiguous_measure": "More than one column could be the measure.",
    "ambiguous_dimension": "More than one column could be the dimension.",
    "causal_question": "This data cannot answer a causal question.",
    "leakage_suspected": "A column mirrors the outcome.",
    "fanout_join": "The join multiplied rows.",
    "sample_ratio_mismatch": "The arms are not split as designed.",
    "constant_column": "The measure has a single value, so it has no spread to describe.",
    "engine_timeout": "The analysis did not finish within its time budget.",
    "unsupported_question": "The recipe does not answer this question.",
    "missing_dependency": "A required package is not installed.",
    "internal": "The recipe produced a report that did not verify; nothing is shown.",
    "too_large": "The file is larger than this analysis handles; a smaller extract or a connected source is needed.",
}


def _refused(recipe: Recipe, reason: str, text: str | None = None, *, gates: GateReport | None = None,
             budget_report: BudgetReport | None = None) -> RunResult:
    return RunResult(recipe.spec.id, recipe.spec.version, Verdict.REFUSE, reason=reason, gates=gates,
                     refusal=Refusal(reason, text or REASON_WORDS.get(reason, reason)), budget_report=budget_report)


def _mark_sampled(report: dict[str, Any], info: SourceInfo | None = None) -> None:
    """The report says it was computed on a sample: ``method.sample`` carries
    the real counts (the recipe wrote the sample's own row count into both
    slots, since ``t`` was the sample) and the ``sampling`` caveat leads."""
    sample = report["method"]["sample"]
    sample["is_sampled"] = True
    if info is not None:
        sample["rows_analyzed"] = int(info.rows_analyzed)
        sample["total_rows"] = int(info.total_rows)
    if not any(c["kind"] == "sampling" for c in report["caveats"]):
        report["caveats"].insert(0, {"kind": "sampling", "text": SAMPLING_WORDS})
        report["caveats"] = report["caveats"][:12]


def _budget_report(budget: Budget, info: SourceInfo | None, profile: Mapping[str, Any], clock: Clock,
                   *, timed_out: bool = False) -> BudgetReport:
    if info is None:  # a caller that registered ``t`` itself: the profile's count is all that is known
        rows = int(profile.get("rows") or profile.get("n_rows") or 0)
        info = SourceInfo(rows, 0, False, rows)
    return BudgetReport(total_rows=info.total_rows, rows_analyzed=info.rows_analyzed, total_bytes=info.total_bytes,
                        sampled=info.sampled, elapsed_s=round(clock.elapsed, 3), memory_limit=budget.memory_limit,
                        threads=budget.threads, timed_out=timed_out)


def run(recipe: Recipe, intent: Intent, con: Any, sem: Mapping[str, _sem.ColumnSemantics],
        profile: Mapping[str, Any], *, sampled: bool = False, budget: Budget = DEFAULT,
        source_info: SourceInfo | None = None, clock: Clock | None = None) -> RunResult:
    """bind → (ask | refuse) → check → compute → report → verify → notebook.

    ``source_info`` is what :func:`~avaloka.recipes.engine.register` found;
    when it says the source was sampled the report carries the ``sampling``
    caveat and the real counts. ``clock`` is a wall clock already running
    (:func:`load_and_run` starts it before registering the source); without
    one, the run starts its own over ``budget.wall_clock_s``.
    """
    rid, version = recipe.spec.id, recipe.spec.version
    bound = recipe.bind(intent, sem, profile)
    if isinstance(bound, Ask):
        return RunResult(rid, version, Verdict.ASK, ask=bound)
    if isinstance(bound, Refusal):
        return RunResult(rid, version, Verdict.REFUSE, reason=bound.reason, refusal=bound)
    sampled = sampled or (source_info is not None and source_info.sampled)
    with (nullcontext(clock) if clock is not None else timed(con, budget.wall_clock_s)) as clk:
        try:
            gates = recipe.check(con, bound)
            if gates.verdict is Verdict.REFUSE:
                return _refused(recipe, gates.reason or "internal", gates=gates,
                                budget_report=_budget_report(budget, source_info, profile, clk))
            evidence: Evidence = recipe.compute(con, bound)
            report = recipe.report(evidence, bound, gates)
            if sampled:
                _mark_sampled(report, source_info)
            verify_report(report)
        except RecipesDependencyError as exc:
            return _refused(recipe, "missing_dependency", str(exc))
        except ReportInvalid as exc:
            log.error("recipe %s %s produced a report that did not verify: %s", rid, version, exc.problems)
            return _refused(recipe, "internal", budget_report=_budget_report(budget, source_info, profile, clk))
        except Exception as exc:  # fail closed, say why in the log; the user sees a reason, never the text
            if interrupted(exc):
                log.warning("recipe %s %s ran out of time after %.1fs", rid, version, clk.elapsed)
                return _refused(recipe, "engine_timeout",
                                budget_report=_budget_report(budget, source_info, profile, clk, timed_out=True))
            if out_of_memory(exc):
                log.warning("recipe %s %s ran out of memory (limit %s)", rid, version, budget.memory_limit,
                            exc_info=True)
                return _refused(recipe, "too_large", budget_report=_budget_report(budget, source_info, profile, clk))
            log.exception("recipe %s %s failed", rid, version)
            return _refused(recipe, "internal", budget_report=_budget_report(budget, source_info, profile, clk))
        if clk.expired:  # the clock ran out while the work was in Python: the budget still holds
            log.warning("recipe %s %s overran its wall clock: %.1fs", rid, version, clk.elapsed)
            return _refused(recipe, "engine_timeout",
                            budget_report=_budget_report(budget, source_info, profile, clk, timed_out=True))
        spent = _budget_report(budget, source_info, profile, clk)
    report["provenance"]["verifier"] = {"passed": True, "rejections": []}
    try:
        notebook = recipe.notebook(bound, gates=gates)
        if sampled and source_info is not None:
            # The notebook must reproduce the report: its load cell takes the
            # same sample (engine.SAMPLE_SQL) when these two parameters are set.
            notebook = apply_parameters(notebook, {"SAMPLE_ROWS": int(source_info.rows_analyzed),
                                                   "SAMPLE_SEED": int(budget.seed)})
    except Exception:  # the report verified; a notebook bug costs the notebook, not the answer
        log.exception("recipe %s %s could not write its notebook", rid, version)
        notebook = None
    # A sampled run is qualified even when every gate passed: its figures are
    # estimates from a sample, which changes the reading (design §2.2, §5).
    verdict = Verdict.QUALIFIED if sampled and gates.verdict is Verdict.PASS else gates.verdict
    return RunResult(rid, version, verdict, report=report, notebook=notebook, gates=gates, evidence=evidence,
                     budget_report=spent)


def build_profile(df: pd.DataFrame, *, source: str | None = None, total_rows: int | None = None) -> dict[str, Any]:
    """The profile ``bind`` reads: row and column counts, ``avaloka.stats``
    per-column statistics, the data source, and for text columns that hold
    dates the day/month convention (so ``trend`` can ask about an ambiguous
    column without reading rows itself). When ``df`` is a sample,
    ``total_rows`` is the source's full count: ``rows`` reports it and
    ``rows_analyzed`` the sample."""
    columns = []
    for name in df.columns:
        col = profile_column(df[name], len(df))
        dates = profile_date_convention(df[name])
        if dates:
            col["dates"] = dates
        columns.append(col)
    return {"rows": len(df) if total_rows is None else int(total_rows), "rows_analyzed": len(df),
            "n_cols": len(df.columns), "columns": columns, "source": source or "data.csv"}


def read_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_parquet(source) if is_parquet(source) else pd.read_csv(source)


def load_and_run(recipe: Recipe, intent: Intent, source: str | Path | pd.DataFrame, *,
                 budget: Budget = DEFAULT) -> RunResult:
    """Connect and register ``source`` under ``budget``, infer column semantics
    and a profile from what ``t`` holds (the sample, when the source is over
    the budget — the whole file is never read into pandas), and run."""
    try:
        con = connect(source, budget=budget)
    except RecipesDependencyError as exc:
        return _refused(recipe, "missing_dependency", str(exc))
    name = None if isinstance(source, pd.DataFrame) else str(source)
    with timed(con, budget.wall_clock_s) as clock:
        try:
            info = register(con, source, budget)
            df = query(con, "SELECT * FROM t")
        except SourceTooLarge as exc:
            log.info("recipe %s refused %s: %s", recipe.spec.id, name or "a frame", exc)
            return _refused(recipe, "too_large")
        except Exception as exc:
            if interrupted(exc):
                log.warning("recipe %s ran out of time reading %s after %.1fs", recipe.spec.id, name or "a frame",
                            clock.elapsed)
                return _refused(recipe, "engine_timeout",
                                budget_report=_budget_report(budget, None, {}, clock, timed_out=True))
            if out_of_memory(exc):
                log.warning("recipe %s ran out of memory reading %s (limit %s)", recipe.spec.id, name or "a frame",
                            budget.memory_limit, exc_info=True)
                return _refused(recipe, "too_large", budget_report=_budget_report(budget, None, {}, clock))
            raise
        sem = _sem.infer(df)
        profile = build_profile(df, source=name, total_rows=info.total_rows)
        return run(recipe, intent, con, sem, profile, budget=budget, source_info=info, clock=clock)
