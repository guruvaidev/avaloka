"""The empty-source COUNT guard must SKIP fill/impute/set-null operations — their
pseudocode "where …" clause selects rows to MODIFY, not to KEEP, so a 0 count is not
an empty source. Previously it skipped only when the literal words "null" AND "fill"
both appeared, so "fill missing values with the median" (no literal "null") ran a
mis-parsed COUNT and could FALSELY abort the transfer as "empty source".
"""
from unittest import mock

import pytest

import app.agents.data_transfer_agent.daft_coder as dc
from app.agents.data_transfer_agent.daft_coder import (
    _is_impute_or_fill_pseudocode,
    empty_source_guard_node,
)


def test_impute_detector_matches_fill_missing_without_literal_null():
    # The reported case: "fill missing" with no literal "null".
    assert _is_impute_or_fill_pseudocode("Fill missing values in age with the median.")
    for s in (
        "fill null values with 0",
        "Impute the mean for account_balance.",
        "Apply a forward fill on loyalty_score.",
        "Use fillna to replace NaN in age.",
        "Coalesce nulls in status.",
        "Interpolate missing readings.",
        "Set 'age' to NULL where 'status' is 'pending', then fill NULLs.",
    ):
        assert _is_impute_or_fill_pseudocode(s), s


def test_impute_detector_rejects_pure_filters():
    for s in (
        "Keep only rows where age > 100",
        "Filter rows where status = 'active'",
        "Select rows where median_income is greater than 8.0",
        "Rename id to customer_id",
    ):
        assert not _is_impute_or_fill_pseudocode(s), s


_DB_STATE = {
    "data_source_location": "postgresql+psycopg2://u:p@h:5432/db",
    "source_table": "patients",
    "source_type": "postgresql",
    "source_connect_args": {},
}


def test_fill_missing_skips_guard_without_running_count():
    """A fill/impute pseudocode that ALSO contains a where-line must be skipped —
    the guard must not run its COUNT (which could falsely return 0)."""
    state = {
        **_DB_STATE,
        # A fill request whose pseudocode carries a where-line for the fill step.
        "coder_pseudocode": (
            "Fill missing values in 'age' with the median.\n"
            "Filter rows where age is missing."
        ),
    }
    with mock.patch.object(dc, "_run_source_count_query") as count_spy:
        out = empty_source_guard_node(state)

    count_spy.assert_not_called()          # guard skipped — no COUNT
    assert out == {"empty_source": False}  # not a false empty-source abort


def test_real_filter_still_runs_the_guard():
    """A genuine row-reducing filter must still trigger the COUNT guard."""
    state = {
        **_DB_STATE,
        "coder_pseudocode": "Keep only rows where age > 100.",
    }
    with mock.patch.object(dc, "_run_source_count_query", return_value=0) as count_spy:
        out = empty_source_guard_node(state)

    count_spy.assert_called_once()         # guard ran on the real filter
    assert out["empty_source"] is True     # 0 rows → genuine empty-source abort
    assert "0 rows" in out["pipeline_abort_reason"]


# The full end-to-end benchmark (test_daft_coder_pipeline.py) needs a coder LLM key,
# live source/dest Postgres, and a Docker/GKE runner — it can't run in CI. But the
# empty-source guard behaviour for its NULL_HANDLING suite CAN be checked offline:
# every one of those prompts introduces nulls first ("Set X to NULL where …") and
# then fills/imputes/drops. None of them is a pure row-reducing filter on the source
# (which starts with no nulls), so the guard must SKIP them — otherwise it runs a
# `WHERE … IS NULL` COUNT that returns 0 and FALSELY aborts. Two of them
# (null_drop_any, null_drop_specific_col) say "drop … NULL" with no "fill", exactly
# the case the old "null" AND "fill" check missed.
try:
    from tests.test_daft_coder_pipeline import NULL_HANDLING_TEST_SUITE as _NULL_SUITE
except Exception:  # pragma: no cover - heavy benchmark deps may be absent
    _NULL_SUITE = []


@pytest.mark.skipif(not _NULL_SUITE, reason="benchmark suite unavailable")
@pytest.mark.parametrize("case", _NULL_SUITE, ids=[c["name"] for c in _NULL_SUITE])
def test_benchmark_null_handling_prompts_never_false_abort(case):
    # Worst case: the pseudocode also carries a null-related filter line the
    # extractor WILL grab — the guard must still skip on the fill/impute/set intent.
    pseudo = case["prompt"] + "\nFilter rows where the target column is null."
    with mock.patch.object(dc, "_run_source_count_query", return_value=0) as count_spy:
        out = empty_source_guard_node({**_DB_STATE, "coder_pseudocode": pseudo})

    count_spy.assert_not_called()          # never runs the mis-parsed COUNT
    assert out == {"empty_source": False}  # never a false empty-source abort
