"""Every golden case under ``tests/recipes/golden`` runs without a model and
must hold its verdict, reason, figures, caveat kinds and wording."""

from __future__ import annotations

import pytest

from tests.recipes._golden import assert_golden, case_id, cases

REQUIRED = {
    "profile/clean", "profile/duplicates",
    "describe/clean_by_dimension", "describe/constant_column", "describe/small_n",
    "describe/missing_heavy", "describe/duplicates",
    "trend/clean_monthly", "trend/too_few_periods", "trend/gaps", "trend/mixed_dates", "trend/by_dimension",
    "compare_periods/clean_monthly", "compare_periods/incomplete_latest_period",
    "compare_periods/by_dimension_bridge", "compare_periods/one_window_empty",
    "segment/clean_topn", "segment/high_cardinality_ask", "segment/small_groups",
    "correlate/clean_pairs", "correlate/simpsons_paradox", "correlate/small_n", "correlate/id_columns_excluded",
    "regress/clean_two_drivers", "regress/leakage", "regress/collinear", "regress/small_n",
    "anomalies/clean_with_spikes", "anomalies/too_many_flagged", "anomalies/no_time_column",
    "cohort_retention/clean_monthly", "cohort_retention/too_few_cohorts", "cohort_retention/small_cohorts",
    "cohort_retention/incomplete_latest_period", "cohort_retention/with_measure",
    "ab_test/binary_clear_lift", "ab_test/continuous_welch", "ab_test/srm", "ab_test/underpowered",
    "ab_test/three_arms", "ab_test/small_n",
}


def test_the_required_cases_exist():
    assert REQUIRED <= {case_id(c) for c in cases()}


@pytest.mark.parametrize("case", cases(), ids=case_id)
def test_golden(case):
    assert_golden(case)
