"""Mutation tests: a golden suite that cannot fail proves nothing.

Each test breaks one named thing in one recipe and asserts that at least one
of that recipe's golden cases fails the shared ``assert_golden``. A mutation
that leaves every golden green fails here, which is the review's signal that
the goldens do not pin what they claim to.
"""

from __future__ import annotations

import pytest

import pandas as pd

from avaloka.recipes import notebook_exec
from avaloka.recipes.library import (ab_test, anomalies, cohort_retention, compare_periods, correlate, describe, profile,
                                     regress, segment, trend)
from tests.recipes._golden import assert_golden, assert_notebook_parity, cases


def failing_cases(recipe_id: str) -> list[str]:
    """Golden cases of ``recipe_id`` whose assertions fail under the current mutation."""
    failed = []
    for case in cases():
        if case.parent.name != recipe_id:
            continue
        try:
            assert_golden(case)
        except AssertionError:
            failed.append(case.name)
    return failed


def test_the_goldens_are_green_without_a_mutation():
    for recipe_id in ("profile", "describe", "trend", "compare_periods", "segment", "correlate", "regress", "anomalies",
                      "cohort_retention", "ab_test"):
        assert failing_cases(recipe_id) == []


def test_describe_bites_when_the_median_is_the_mean(monkeypatch):
    mutated = describe.SQL_STATS.replace("median({measure}) AS median", "avg({measure}) AS median")
    assert mutated != describe.SQL_STATS
    monkeypatch.setattr(describe, "SQL_STATS", mutated)
    assert failing_cases("describe")


def test_describe_bites_when_the_dedup_cte_is_dropped(monkeypatch):
    monkeypatch.setattr(describe, "SQL_DEDUP", "SELECT * FROM t")
    assert "duplicates" in failing_cases("describe")


def test_trend_bites_when_the_dedup_cte_is_dropped(monkeypatch):
    monkeypatch.setattr(trend, "SQL_DEDUP", "SELECT * FROM t")
    assert "duplicates" in failing_cases("trend")


def test_trend_bites_when_the_gap_check_is_dropped(monkeypatch):
    monkeypatch.setattr(trend, "_gaps", lambda periods, grain: 0)
    assert "gaps" in failing_cases("trend")


def test_profile_bites_when_duplicates_read_as_zero(monkeypatch):
    monkeypatch.setattr(profile, "SQL_DISTINCT_ROWS", "SELECT count(*) AS rows_distinct FROM t")
    assert "duplicates" in failing_cases("profile")


def test_compare_periods_bites_when_the_windows_are_swapped(monkeypatch):
    original = compare_periods.window_bounds

    def swapped(first, last, window, data_grain):
        current, previous, grain, complete = original(first, last, window, data_grain)
        return previous, current, grain, complete

    monkeypatch.setattr(compare_periods, "window_bounds", swapped)
    assert "clean_monthly" in failing_cases("compare_periods")


def test_segment_bites_when_the_share_is_over_the_top_n(monkeypatch):
    def over_shown(groups: pd.DataFrame) -> pd.DataFrame:
        share = groups["total"].astype(float) / float(groups.head(segment.TOP_N)["total"].sum())
        return groups.assign(share=share, cumulative_share=share.cumsum())

    monkeypatch.setattr(segment, "_concentration", over_shown)
    assert "clean_topn" in failing_cases("segment")


def test_correlate_bites_when_the_within_group_loop_is_skipped(monkeypatch):
    monkeypatch.setattr(correlate, "within_groups", lambda values, a, b, dimension: {})
    assert "simpsons_paradox" in failing_cases("correlate")


def test_regress_bites_when_the_leakage_check_is_dropped(monkeypatch):
    monkeypatch.setattr(regress, "leaking", lambda values, outcome, drivers: [])
    assert "leakage" in failing_cases("regress")


def test_regress_bites_when_the_errors_are_not_robust(monkeypatch):
    monkeypatch.setattr(regress, "COV_TYPE", "nonrobust")
    assert "clean_two_drivers" in failing_cases("regress")


def test_anomalies_bites_when_the_mad_is_the_standard_deviation(monkeypatch):
    def std_z(x):
        mean, std = float(x.mean()), float(x.std())
        return (x - mean) / std if std else x * 0.0, mean, std

    monkeypatch.setattr(anomalies, "robust_z", std_z)
    assert failing_cases("anomalies")


def test_cohort_retention_bites_without_the_cohort_join(monkeypatch):
    """Counting every id active in a period, not only the cohort's own ids,
    is the classic retention mistake: cells exceed the cohort size."""
    joined = "FROM a JOIN c USING (id) WHERE a.period < TIMESTAMP '{cutoff}'"
    unjoined = ("FROM a, (SELECT DISTINCT cohort FROM c) AS c WHERE a.period >= c.cohort "
                "AND a.period < TIMESTAMP '{cutoff}'")
    assert joined in cohort_retention.SQL_MATRIX
    monkeypatch.setattr(cohort_retention, "SQL_MATRIX", cohort_retention.SQL_MATRIX.replace(joined, unjoined))
    assert "clean_monthly" in failing_cases("cohort_retention")


def test_ab_test_bites_when_the_srm_threshold_is_loosened(monkeypatch):
    monkeypatch.setattr(ab_test, "SRM_P", 1e-9)
    assert "srm" in failing_cases("ab_test")


def test_ab_test_bites_when_welch_becomes_the_pooled_t_test(monkeypatch):
    monkeypatch.setattr(ab_test, "EQUAL_VAR", True)
    assert "continuous_welch" in failing_cases("ab_test")


def test_ab_test_bites_when_the_interval_is_one_standard_error(monkeypatch):
    monkeypatch.setattr(ab_test, "ci_multiplier", lambda df: 1.0)
    assert "binary_clear_lift" in failing_cases("ab_test")


@pytest.mark.notebook
@pytest.mark.skipif(notebook_exec.missing_dependency() is not None, reason=notebook_exec.INSTALL_HINT)
def test_notebook_parity_bites_when_the_notebook_median_is_the_mean(monkeypatch):
    """A notebook whose SQL drifts from the recipe's (here only the notebook's
    copy of SQL_STATS; the report is untouched) fails parity on the figure
    it changed, so the parity test cannot pass on a notebook that reproduces
    nothing."""
    original = describe.Describe.notebook

    def drifted(self, bound, gates=None):
        nb = original(self, bound, gates=gates)
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if src.startswith("SQL_STATS = "):
                assert "median({measure}) AS median" in src
                cell["source"] = [src.replace("median({measure}) AS median", "avg({measure}) AS median")]
        return nb

    monkeypatch.setattr(describe.Describe, "notebook", drifted)
    case = next(c for c in cases() if c.parent.name == "describe" and c.name == "clean_by_dimension")
    with pytest.raises(AssertionError, match="/measure/median"):
        assert_notebook_parity(case)


def test_a_wrong_figure_never_ships(monkeypatch):
    """A recipe that mis-reports a figure ends refuse/internal, so the golden fails on
    the verdict rather than on a number a reader could have seen."""
    original = describe.figure

    def wrong(fid, ev, pointer, **kw):
        fig = original(fid, ev, pointer, **kw)
        if pointer == "/measure/median":
            fig["value"] += 1.0
        return fig

    monkeypatch.setattr(describe, "figure", wrong)
    with pytest.raises(AssertionError, match="refuse"):
        assert_golden(next(c for c in cases() if c.parent.name == "describe" and c.name == "clean_by_dimension"))
