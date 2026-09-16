"""
End-to-end render pipeline tests.
==================================
Layer 3 — simulates exactly what execution_agent_node_local does:
  constructs execution_result dicts + real pandas DataFrames, calls
  render_response(), and asserts the final user-facing string.

No subprocess.  No agent imports.  All execution_result dicts are built
by hand to match what execute_code_on_local() returns in production.

This layer validates the full chain:
  execution signals → build_artifact → conclude / render → final string
for every meaningful prompt/result combination that the branch touches.
"""

import json
import pytest
import pandas as pd

from app.agents.result_renderer import render_response, RESULT_START, RESULT_END

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _er(status="success", rc=0, stdout="", stderr="", error=None):
    """Mimic the dict that execute_code_on_local() returns."""
    out = {
        "status": status,
        "returncode": rc,
        "execution_stdout": stdout,
        "execution_stderr": stderr,
    }
    if error:
        out["execution_error"] = error
    return out


def _sentinel(kind, value, columns=None):
    payload = json.dumps({"kind": kind, "value": value, "columns": columns or []})
    return f"{RESULT_START}{payload}{RESULT_END}"


# A realistic schema for tests that need passthrough-df detection
_SCHEMA = {
    "TransactionID": "int64",
    "TransactionAmt": "float64",
    "isFraud": "int64",
}

def _full_df(n=500):
    """Simulate a passthrough (original input) DataFrame."""
    return pd.DataFrame({
        "TransactionID": range(n),
        "TransactionAmt": [float(i) * 1.5 for i in range(n)],
        "isFraud": [i % 2 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Scenario 1: Analytical query — sentinel path (LLM called avaloka_result)
# ---------------------------------------------------------------------------

class TestAnalyticalSentinelPath:

    def test_mean_sentinel_produces_conclusion(self):
        stdout = _sentinel("mean", 149.85, ["TransactionAmt"])
        er = _er(stdout=stdout)
        response = render_response(er, output_df=_full_df(),
                                   user_question="What is the mean of TransactionAmt?",
                                   schema=_SCHEMA)
        assert "mean" in response.lower()
        assert "TransactionAmt" in response
        assert "149.85" in response or "149.8500" in response

    def test_sum_sentinel_produces_conclusion(self):
        stdout = _sentinel("sum", 9_876_543.0, ["TransactionAmt"])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="sum of TransactionAmt", schema=_SCHEMA)
        assert "sum" in response.lower()
        assert "9,876,543" in response

    def test_count_sentinel_produces_conclusion(self):
        stdout = _sentinel("count", 590540.0, [])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="How many rows are there?")
        assert "count" in response.lower() or "590,540" in response

    def test_correlation_strong_positive_sentinel(self):
        stdout = _sentinel("correlation", 0.82, ["length", "weight"])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="correlate length with weight")
        assert "strong" in response
        assert "positive" in response
        assert "r = 0.820" in response

    def test_correlation_no_significant_near_zero(self):
        """r = -0.002: must produce 'no significant', never 'no negative'."""
        stdout = _sentinel("correlation", -0.002, ["P_emaildomain_length", "TransactionAmt"])
        er = _er(stdout=stdout)
        response = render_response(
            er,
            user_question="Does the length of the P_emaildomain string correlate with TransactionAmt?",
            schema=_SCHEMA,
        )
        assert "no significant" in response
        assert "no negative" not in response
        assert "r = -0.002" in response

    def test_sentinel_beats_output_df_for_analytical(self):
        """Sentinel takes priority even when a full output_df is present."""
        stdout = _sentinel("mean", 42.0, ["TransactionAmt"])
        er = _er(stdout=stdout)
        response = render_response(er, output_df=_full_df(),
                                   user_question="calculate the mean of TransactionAmt",
                                   schema=_SCHEMA)
        # Should be a conclusion, NOT a "Transformed data" table message
        assert "Transformed data" not in response
        assert "mean" in response.lower()


# ---------------------------------------------------------------------------
# Scenario 2: Analytical query — 1-row summary df fallback (LLM forgot sentinel)
# ---------------------------------------------------------------------------

class TestAnalyticalOneRowFallback:

    def test_one_row_one_col_df_becomes_scalar(self):
        """Coder returned pd.DataFrame({'TransactionAmt_mean': [149.85]})."""
        er = _er(stdout="")
        output_df = pd.DataFrame({"TransactionAmt_mean": [149.85]})
        response = render_response(er, output_df=output_df,
                                   user_question="What is the mean of TransactionAmt?")
        assert "Transformed data" not in response
        assert "149.85" in response or "149.8500" in response

    def test_one_row_multi_col_df_picks_first_numeric(self):
        er = _er(stdout="")
        output_df = pd.DataFrame({"label": ["total"], "value": [4500.0]})
        response = render_response(er, output_df=output_df,
                                   user_question="total sum of revenue")
        assert "Transformed data" not in response
        assert "4,500" in response or "4500" in response

    def test_one_row_df_with_question_metric_detected(self):
        er = _er(stdout="")
        output_df = pd.DataFrame({"count_result": [1200.0]})
        # Use a pure count question with no transform verbs
        response = render_response(er, output_df=output_df,
                                   user_question="how many rows are there in total?")
        # Should be a scalar conclusion, not a table
        assert "Transformed data" not in response

    def test_wide_one_row_summary_is_narrated(self):
        output_df = pd.DataFrame([{
            "avg_discount_expensive": 12.5,
            "avg_discount_cheaper": 8.0,
            "correlation": 0.62,
            "p_value": 0.01,
        }])
        response = render_response(
            _er(stdout=""),
            output_df=output_df,
            user_question="Compare average discounts and report correlation and significance",
        )
        assert "Transformed data" not in response
        assert "higher" in response.lower()
        assert "correlation" in response.lower()

    def test_wide_one_row_transformation_stays_a_table(self):
        output_df = pd.DataFrame([{"a": 1, "b": 2, "c": 3, "d": 4}])
        response = render_response(
            _er(stdout=""), output_df=output_df,
            user_question="normalize and transform these columns",
        )
        assert "Transformed data" in response


# ---------------------------------------------------------------------------
# Scenario 3: Analytical query — passthrough df + printed scalar (3b path)
# ---------------------------------------------------------------------------

class TestAnalyticalPassthroughFallback:

    def test_passthrough_df_printed_mean(self):
        """Coder returned original df unchanged but did print(f'Mean: {val}')."""
        stdout = "Mean of TransactionAmt: 0.4567"
        er = _er(stdout=stdout)
        response = render_response(er, output_df=_full_df(),
                                   user_question="calculate the mean of TransactionAmt",
                                   schema=_SCHEMA)
        # Should extract the printed number, not show 500-row table
        assert "Transformed data" not in response
        assert "0.4567" in response or "0.4567" in response

    def test_passthrough_df_no_printed_number_falls_to_table(self):
        """No sentinel, no number in stdout, but passthrough df → table (honest output)."""
        stdout = "dtype: object\nprocessing done"
        er = _er(stdout=stdout)
        response = render_response(er, output_df=_full_df(),
                                   user_question="calculate the mean of TransactionAmt",
                                   schema=_SCHEMA)
        assert "Transformed data" in response


# ---------------------------------------------------------------------------
# Scenario 4: Transformation queries — must ALWAYS return table
# ---------------------------------------------------------------------------

class TestTransformationAlwaysTable:

    def test_min_max_with_misfired_sentinel_returns_table(self):
        """Core regression: Min-Max returns 'The maximum of C1 is 2794' was the bug."""
        stdout = _sentinel("max", 2794.0, ["C1"])
        er = _er(stdout=stdout)
        # Coder returned the full transformed df
        transformed_df = pd.DataFrame({
            "C1": [100.0, 500.0, 2794.0],
            "C1_normalized": [0.0, 0.143, 1.0],
        })
        response = render_response(
            er,
            output_df=transformed_df,
            user_question="Apply a Min-Max scaling normalization transform to the C1 column",
        )
        assert "Transformed data" in response
        assert "The maximum" not in response

    def test_min_max_one_row_df_returns_table_not_scalar(self):
        """3a guard: 1-row intermediate df for transform must NOT surface as scalar."""
        er = _er(stdout="")
        # Coder mistakenly returned pd.DataFrame({'C1_max': [2794.0]}) instead of transformed df
        bad_df = pd.DataFrame({"C1_max": [2794.0]})
        response = render_response(
            er,
            output_df=bad_df,
            user_question="Apply Min-Max normalization to C1",
        )
        assert "Transformed data" in response
        assert "2,794" not in response

    def test_normalize_verb_routes_to_table(self):
        er = _er(stdout="")
        df = pd.DataFrame({"X": [0.0, 0.5, 1.0], "X_norm": [0.0, 0.5, 1.0]})
        response = render_response(er, output_df=df,
                                   user_question="normalize the X column")
        assert "Transformed data" in response

    def test_filter_verb_routes_to_table(self):
        er = _er(stdout="")
        df = pd.DataFrame({"amount": [100.0, 200.0, 300.0]})
        response = render_response(er, output_df=df,
                                   user_question="filter rows where amount > 150")
        assert "Transformed data" in response

    def test_zscore_standardize_routes_to_table(self):
        er = _er(stdout="")
        df = pd.DataFrame({"score": [-1.2, 0.0, 1.2]})
        response = render_response(er, output_df=df,
                                   user_question="standardize the score column using z-score")
        assert "Transformed data" in response

    def test_add_column_routes_to_table(self):
        er = _er(stdout="")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "A_plus_B": [4, 6]})
        response = render_response(er, output_df=df,
                                   user_question="add a column A_plus_B that sums A and B")
        assert "Transformed data" in response

    def test_table_response_includes_row_and_col_counts(self):
        er = _er(stdout="")
        df = pd.DataFrame({"x": range(30), "y": range(30)})
        response = render_response(er, output_df=df, user_question="sort by x")
        assert "30" in response
        assert "2" in response  # column count


# ---------------------------------------------------------------------------
# Scenario 5: Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:

    def test_key_error_produces_actionable_message(self):
        er = _er(status="error", rc=1, stderr="KeyError: 'missing_col'")
        response = render_response(er, user_question="mean of missing_col")
        assert "❌" in response
        assert "missing_col" in response

    def test_value_error_numeric_column(self):
        er = _er(status="error", rc=1,
                 stderr="ValueError: could not convert string to float: 'abc'")
        response = render_response(er, user_question="sum of text_col")
        assert "❌" in response
        assert "non-numeric" in response.lower() or "numeric" in response.lower()

    def test_timeout_error_message(self):
        er = _er(status="error", rc=None,
                 error="Script execution timed out after 60 seconds")
        response = render_response(er)
        assert "❌" in response
        assert "timed out" in response.lower() or "time" in response.lower()

    def test_nonzero_returncode_triggers_error(self):
        er = _er(status="success", rc=1, stderr="SyntaxError: invalid syntax")
        response = render_response(er)
        assert "❌" in response

    def test_error_with_output_df_present_still_errors(self):
        """Error must win over any output df — don't show a table for a failed run."""
        er = _er(status="error", rc=1, stderr="MemoryError")
        df = pd.DataFrame({"x": [1, 2, 3]})
        response = render_response(er, output_df=df)
        assert "❌" in response
        assert "Transformed data" not in response


# ---------------------------------------------------------------------------
# Scenario 6: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_df_returns_no_rows_message(self):
        er = _er(stdout="")
        response = render_response(er, output_df=pd.DataFrame())
        assert "no rows" in response.lower()

    def test_no_output_at_all_returns_safe_message(self):
        er = _er(stdout="")
        response = render_response(er)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_render_response_never_raises(self):
        """Even a completely broken input must never propagate an exception."""
        for bad_input in [None, {}, {"status": None}, {"returncode": "oops"}]:
            response = render_response(bad_input)
            assert isinstance(response, str)

    def test_correlation_out_of_range_does_not_crash(self):
        """If coder emits r=2.0 (invalid), renderer must not crash."""
        stdout = _sentinel("correlation", 2.0, ["A", "B"])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="correlate A with B")
        assert isinstance(response, str)

    def test_large_table_shows_truncation_notice(self):
        er = _er(stdout="")
        df = pd.DataFrame({"val": range(500)})
        response = render_response(er, output_df=df,
                                   user_question="sort values",
                                   max_table_rows=100)
        assert "500" in response
        assert "100" in response

    def test_all_transform_verbs_route_to_table(self):
        """Spot-check a sample of _TRANSFORM_VERBS to confirm none slip through."""
        transform_questions = [
            "encode the category column",
            "one-hot encode the status column",
            "merge the two datasets on id",
            "join the tables on customer_id",
            "pivot the dataframe by month",
            "bin the age column into 5 buckets",
            "rename the column price to amount",
            "drop columns with too many null values",
            "impute missing values in score",
            "convert the date column to datetime",
        ]
        er = _er(stdout=_sentinel("max", 999.0))  # a misfired sentinel
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        for question in transform_questions:
            response = render_response(er, output_df=df, user_question=question)
            assert "Transformed data" in response, (
                f"Transform verb not guarded: '{question}' produced: {response!r}"
            )
