"""
Unit tests for app/agents/result_renderer.py
=============================================
Layer 1 — pure Python + pandas, no subprocess, no agent imports.

Covers every branch in:
  clean_stdout          — noise stripping
  extract_result_block  — sentinel parsing
  _extract_last_number  — stdout scalar extraction
  build_artifact        — the type gate (all 6 kinds + every priority path)
  conclude              — metric phrases, correlation grammar
  render_response       — full user-facing strings
"""

import json
import pytest
import pandas as pd

from app.agents.result_renderer import (
    RESULT_START,
    RESULT_END,
    clean_stdout,
    extract_result_block,
    build_artifact,
    conclude,
    render_response,
    _extract_last_number,
    _fmt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentinel(kind="scalar", value=42.0, columns=None):
    """Build a correctly-formatted AVALOKA_RESULT sentinel string."""
    payload = json.dumps({"kind": kind, "value": value, "columns": columns or []})
    return f"{RESULT_START}{payload}{RESULT_END}"


def _er(status="success", rc=0, stdout="", stderr=""):
    """Build a minimal execution_result dict as execution_agent produces it."""
    return {
        "status": status,
        "returncode": rc,
        "execution_stdout": stdout,
        "execution_stderr": stderr,
    }


def _df(data: dict):
    """Shorthand for pd.DataFrame with scalar values → 1-row DataFrame."""
    return pd.DataFrame({k: [v] for k, v in data.items()})


# ---------------------------------------------------------------------------
# 1a. clean_stdout
# ---------------------------------------------------------------------------

class TestCleanStdout:

    def test_strips_ray_start_marker(self):
        assert clean_stdout("INTERACTIVE_SAMPLE_ANALYSIS_START\nreal output") == "real output"

    def test_strips_ray_done_marker(self):
        assert clean_stdout("INTERACTIVE_SAMPLE_ANALYSIS_DONE\nresult") == "result"

    def test_strips_user_code_markers(self):
        raw = "USER_CODE_START\ncomputed\nUSER_CODE_END"
        assert clean_stdout(raw) == "computed"

    def test_strips_groupby_noise(self):
        raw = "Grouping by: ['category']\nAggregating column: price\n99.5"
        assert clean_stdout(raw) == "99.5"

    def test_strips_shape_noise(self):
        raw = "SHAPE (sample): (50000, 394)\nMean: 0.45"
        assert clean_stdout(raw) == "Mean: 0.45"

    def test_strips_metric_noise(self):
        raw = "METRIC throughput_mb_s=12.3\nresult"
        assert clean_stdout(raw) == "result"

    def test_strips_sentinel_block(self):
        sentinel = _sentinel(value=7.0)
        raw = f"preamble\n{sentinel}\npostamble"
        cleaned = clean_stdout(raw)
        assert RESULT_START not in cleaned
        assert RESULT_END not in cleaned
        assert "preamble" in cleaned
        assert "postamble" in cleaned

    def test_keeps_real_printed_output(self):
        raw = "Mean of TransactionAmt: 0.234567"
        assert clean_stdout(raw) == "Mean of TransactionAmt: 0.234567"

    def test_empty_input_returns_empty(self):
        assert clean_stdout("") == ""

    def test_none_input_returns_empty(self):
        assert clean_stdout(None) == ""

    def test_only_noise_returns_empty(self):
        raw = "Grouping by: col\nAggregating column: val\nUSER_CODE_START"
        assert clean_stdout(raw) == ""


# ---------------------------------------------------------------------------
# 1b. extract_result_block
# ---------------------------------------------------------------------------

class TestExtractResultBlock:

    def test_returns_dict_for_valid_block(self):
        stdout = _sentinel(kind="mean", value=123.45, columns=["Price"])
        result = extract_result_block(stdout)
        assert result == {"kind": "mean", "value": 123.45, "columns": ["Price"]}

    def test_returns_none_when_no_block(self):
        assert extract_result_block("just some output") is None

    def test_returns_none_for_empty_string(self):
        assert extract_result_block("") is None

    def test_returns_none_for_none(self):
        assert extract_result_block(None) is None

    def test_returns_none_for_malformed_json(self):
        bad = f"{RESULT_START}{{not valid json}}{RESULT_END}"
        assert extract_result_block(bad) is None

    def test_last_block_wins_when_multiple_present(self):
        first = _sentinel(value=1.0)
        second = _sentinel(value=2.0)
        stdout = f"{first}\nsome noise\n{second}"
        result = extract_result_block(stdout)
        assert result["value"] == 2.0

    def test_ignores_surrounding_noise(self):
        noise = "Grouping by: col\n"
        stdout = noise + _sentinel(value=9.9) + "\nmore noise"
        result = extract_result_block(stdout)
        assert result is not None
        assert result["value"] == 9.9


# ---------------------------------------------------------------------------
# 1c. _extract_last_number
# ---------------------------------------------------------------------------

class TestExtractLastNumber:

    def test_label_colon_float(self):
        assert _extract_last_number("Mean of X: 123.45") == 123.45

    def test_label_equals_float(self):
        assert _extract_last_number("result = 0.99") == 0.99

    def test_bare_integer(self):
        assert _extract_last_number("42") == 42

    def test_bare_negative(self):
        assert _extract_last_number("-7.5") == -7.5

    def test_multiline_takes_last_line_number(self):
        noisy = "Index(['col1','col2'])\ndtype: object\nMean: 0.234567"
        assert _extract_last_number(noisy) == pytest.approx(0.234567)

    def test_dtype_object_line_returns_none(self):
        # "dtype: object" — "object" is not a number
        assert _extract_last_number("dtype: object") is None

    def test_none_returns_none(self):
        assert _extract_last_number(None) is None

    def test_empty_returns_none(self):
        assert _extract_last_number("") is None

    def test_number_with_commas(self):
        # "1,234" should parse as 1234
        result = _extract_last_number("Total: 1,234")
        assert result == 1234

    def test_scientific_notation(self):
        result = _extract_last_number("value: 3.14e-5")
        assert result == pytest.approx(3.14e-5)

    def test_word_after_number_is_not_matched(self):
        # "590540 rows" — number is NOT at end of line
        assert _extract_last_number("Data loaded: 590540 rows") is None


# ---------------------------------------------------------------------------
# 1d. build_artifact  (the type gate)
# ---------------------------------------------------------------------------

class TestBuildArtifact:

    # --- error paths ---

    def test_error_status_returns_error_kind(self):
        er = _er(status="error", rc=1, stderr="KeyError: 'col'")
        result = build_artifact(er)
        assert result["kind"] == "error"
        assert "col" in result["error"]

    def test_nonzero_returncode_returns_error_kind(self):
        er = _er(status="success", rc=1)
        result = build_artifact(er)
        assert result["kind"] == "error"

    def test_timeout_message_in_error(self):
        er = _er(status="error", rc=None)
        er["execution_error"] = "Script execution timed out after 60 seconds"
        result = build_artifact(er)
        assert result["kind"] == "error"
        assert "timed out" in result["error"].lower() or "time" in result["error"].lower()

    # --- sentinel block: analytical query → scalar ---

    def test_sentinel_analytical_returns_scalar(self):
        stdout = _sentinel(kind="mean", value=99.9, columns=["Price"])
        er = _er(stdout=stdout)
        result = build_artifact(er, user_question="What is the mean of Price?")
        assert result["kind"] == "scalar"
        assert result["result"]["value"] == 99.9

    def test_sentinel_correlation_returns_scalar(self):
        stdout = _sentinel(kind="correlation", value=0.75, columns=["A", "B"])
        er = _er(stdout=stdout)
        result = build_artifact(er, user_question="Does A correlate with B?")
        assert result["kind"] == "scalar"
        assert result["result"]["kind"] == "correlation"

    # --- sentinel block: transform query → sentinel skipped ---

    def test_sentinel_transform_query_skips_block(self):
        """Coder misfired avaloka_result() on intermediate max inside Min-Max — must be ignored."""
        stdout = _sentinel(kind="max", value=2794.0, columns=["C1"])
        er = _er(stdout=stdout)
        output_df = pd.DataFrame({"C1": [0.1, 0.5, 0.9], "C1_normalized": [0.0, 0.5, 1.0]})
        result = build_artifact(
            er,
            output_df=output_df,
            user_question="Apply a Min-Max scaling normalization transform to the C1 column",
        )
        assert result["kind"] == "table"

    def test_sentinel_scale_verb_triggers_transform_guard(self):
        stdout = _sentinel(kind="min", value=0.0, columns=["X"])
        er = _er(stdout=stdout)
        df = pd.DataFrame({"X": [0.0, 0.5, 1.0]})
        result = build_artifact(er, output_df=df,
                                user_question="scale the X column using z-score normalization")
        assert result["kind"] == "table"

    # --- 3a: 1-row df, analytical ---

    def test_one_row_df_analytical_returns_scalar(self):
        """Coder returned 1-row summary df without calling avaloka_result()."""
        er = _er(stdout="")
        output_df = _df({"TransactionAmt_mean": 123.45})
        result = build_artifact(er, output_df=output_df,
                                user_question="What is the mean of TransactionAmt?")
        assert result["kind"] == "scalar"
        assert result["result"]["value"] == pytest.approx(123.45)

    def test_one_row_multi_col_df_takes_first_numeric(self):
        er = _er(stdout="")
        output_df = pd.DataFrame({"label": ["mean"], "value": [55.5]})
        result = build_artifact(er, output_df=output_df,
                                user_question="calculate mean of price")
        assert result["kind"] == "scalar"
        assert result["result"]["value"] == pytest.approx(55.5)

    # --- 3a guard: 1-row df, transform → must NOT become scalar ---

    def test_one_row_df_transform_query_returns_table(self):
        """1-row intermediate df (e.g. just the max) during Min-Max must never surface as scalar."""
        er = _er(stdout="")
        output_df = _df({"C1_max": 2794.0})
        result = build_artifact(
            er,
            output_df=output_df,
            user_question="Apply Min-Max normalization to C1",
        )
        assert result["kind"] == "table"

    def test_one_row_df_normalize_verb_returns_table(self):
        er = _er(stdout="")
        output_df = _df({"result": 99.0})
        result = build_artifact(er, output_df=output_df,
                                user_question="normalize the values in column A")
        assert result["kind"] == "table"

    # --- 3b: passthrough df + stdout scalar ---

    def test_passthrough_df_with_printed_scalar(self):
        """Coder returned original df unchanged but printed the mean — extract it from stdout."""
        schema = {"col_A": "float64", "col_B": "int64"}
        stdout = "Mean of col_A: 0.456"
        er = _er(stdout=stdout)
        # Output df has all schema columns → it's a passthrough
        output_df = pd.DataFrame({"col_A": [1.0, 2.0], "col_B": [3, 4]})
        result = build_artifact(er, output_df=output_df,
                                user_question="calculate the mean of col_A",
                                schema=schema)
        assert result["kind"] == "scalar"
        assert result["result"]["value"] == pytest.approx(0.456)

    def test_passthrough_df_no_stdout_number_falls_to_table(self):
        """No sentinel, no printed scalar, passthrough df → table (we can't make up an answer)."""
        schema = {"col_A": "float64", "col_B": "int64"}
        er = _er(stdout="dtype: object")  # no number
        output_df = pd.DataFrame({"col_A": [1.0, 2.0], "col_B": [3, 4]})
        result = build_artifact(er, output_df=output_df,
                                user_question="calculate the mean of col_A",
                                schema=schema)
        assert result["kind"] == "table"

    def test_non_passthrough_df_analytical_returns_table(self):
        """Grouped result (cols don't match schema) → correctly classified as table."""
        schema = {"category": "object", "price": "float64"}
        er = _er(stdout="")
        # Only 2 derived cols, not the full schema
        output_df = pd.DataFrame({"category": ["A", "B"], "price_mean": [10.0, 20.0]})
        result = build_artifact(er, output_df=output_df,
                                user_question="what is the average price by category?",
                                schema=schema)
        assert result["kind"] == "table"

    # --- standard transformation result ---

    def test_multi_row_transform_df_returns_table(self):
        er = _er(stdout="")
        df = pd.DataFrame({"A": range(100), "B": range(100)})
        result = build_artifact(er, output_df=df,
                                user_question="filter rows where A > 50")
        assert result["kind"] == "table"
        assert result["row_count"] == 100

    def test_table_result_includes_columns(self):
        er = _er(stdout="")
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        result = build_artifact(er, output_df=df, user_question="sort by X")
        assert "columns" in result
        assert set(result["columns"]) == {"X", "Y"}

    # --- empty df ---

    def test_empty_df_returns_empty_kind(self):
        er = _er(stdout="")
        result = build_artifact(er, output_df=pd.DataFrame())
        assert result["kind"] == "empty"

    # --- no df, bare stdout scalar ---

    def test_bare_scalar_in_stdout_no_df(self):
        er = _er(stdout="42.0")
        result = build_artifact(er)
        assert result["kind"] == "scalar"

    # --- no df, no stdout ---

    def test_no_output_returns_none_kind(self):
        er = _er(stdout="")
        result = build_artifact(er)
        assert result["kind"] == "none"

    # --- sentinel supplementary_df flag ---

    def test_sentinel_with_df_sets_supplementary_flag(self):
        stdout = _sentinel(kind="sum", value=999.0)
        er = _er(stdout=stdout)
        df = pd.DataFrame({"X": [1, 2, 3]})
        result = build_artifact(er, output_df=df, user_question="sum of X")
        assert result["kind"] == "scalar"
        assert result.get("supplementary_df") is True


# ---------------------------------------------------------------------------
# 1e. conclude
# ---------------------------------------------------------------------------

class TestConclude:

    # --- correlation strength + direction ---

    def test_correlation_near_zero_no_significant(self):
        """r = -0.002: must say 'no significant', NOT 'no negative'."""
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": -0.002, "columns": ["A", "B"]}}
        text = conclude(artifact)
        assert text is not None
        assert "no significant" in text
        assert "negative" not in text

    def test_correlation_small_positive_no_significant(self):
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": 0.05, "columns": ["X", "Y"]}}
        text = conclude(artifact)
        assert "no significant" in text
        assert "positive" not in text

    def test_correlation_weak_negative(self):
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": -0.25, "columns": ["A", "B"]}}
        text = conclude(artifact)
        assert "weak" in text
        assert "negative" in text

    def test_correlation_moderate_positive(self):
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": 0.55, "columns": ["P", "Q"]}}
        text = conclude(artifact)
        assert "moderate" in text
        assert "positive" in text

    def test_correlation_strong_positive_with_column_pair(self):
        artifact = {"kind": "scalar", "result": {
            "kind": "correlation", "value": 0.82,
            "columns": ["length", "weight"]
        }}
        text = conclude(artifact)
        assert "strong" in text
        assert "positive" in text
        assert "`length`" in text
        assert "`weight`" in text

    def test_correlation_strong_negative(self):
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": -0.91, "columns": ["X", "Y"]}}
        text = conclude(artifact)
        assert "strong" in text
        assert "negative" in text

    def test_correlation_includes_r_value(self):
        artifact = {"kind": "scalar", "result": {"kind": "correlation", "value": 0.75, "columns": []}}
        text = conclude(artifact)
        assert "r = 0.750" in text

    # --- metric phrases ---

    def test_mean_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "mean", "value": 150.25, "columns": ["Price"]}}
        text = conclude(artifact, user_question="mean of Price")
        assert text is not None
        assert "mean" in text.lower()
        assert "`Price`" in text

    def test_sum_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "sum", "value": 9999, "columns": ["Revenue"]}}
        text = conclude(artifact)
        assert "sum" in text.lower()
        assert "9,999" in text

    def test_count_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "count", "value": 500, "columns": []}}
        text = conclude(artifact)
        assert "count" in text.lower()
        assert "500" in text

    def test_max_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "max", "value": 2794.0, "columns": ["C1"]}}
        text = conclude(artifact)
        assert "maximum" in text.lower()
        assert "`C1`" in text

    def test_min_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "min", "value": 0.001, "columns": ["score"]}}
        text = conclude(artifact)
        assert "minimum" in text.lower()

    def test_std_phrase(self):
        artifact = {"kind": "scalar", "result": {"kind": "std", "value": 3.14, "columns": ["noise"]}}
        text = conclude(artifact)
        assert "standard deviation" in text.lower()

    # --- gate: non-scalar kinds return None ---

    def test_table_kind_returns_none(self):
        artifact = {"kind": "table", "row_count": 100, "columns": ["A"]}
        assert conclude(artifact) is None

    def test_error_kind_returns_none(self):
        artifact = {"kind": "error", "error": "something broke"}
        assert conclude(artifact) is None

    def test_empty_kind_returns_none(self):
        artifact = {"kind": "empty"}
        assert conclude(artifact) is None

    def test_none_kind_returns_none(self):
        artifact = {"kind": "none"}
        assert conclude(artifact) is None

    def test_none_artifact_returns_none(self):
        assert conclude(None) is None

    # --- number formatting ---

    def test_large_integer_formatted_with_commas(self):
        artifact = {"kind": "scalar", "result": {"kind": "count", "value": 590540, "columns": []}}
        text = conclude(artifact)
        assert "590,540" in text

    def test_float_formatted_to_4_decimal_places(self):
        artifact = {"kind": "scalar", "result": {"kind": "mean", "value": 0.1234567, "columns": ["x"]}}
        text = conclude(artifact)
        assert "0.1235" in text


# ---------------------------------------------------------------------------
# 1f. render_response
# ---------------------------------------------------------------------------

class TestRenderResponse:

    def test_error_starts_with_failure_emoji(self):
        er = _er(status="error", rc=1, stderr="KeyError: 'missing_col'")
        response = render_response(er, user_question="sum of missing_col")
        assert response.startswith("❌")
        assert "Execution failed" in response

    def test_error_includes_clean_cause(self):
        er = _er(status="error", rc=1, stderr="KeyError: 'missing_col'")
        response = render_response(er, user_question="mean")
        assert "missing_col" in response

    def test_scalar_correlation_conclusion(self):
        stdout = _sentinel(kind="correlation", value=0.72, columns=["A", "B"])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="correlate A with B")
        assert "strong" in response
        assert "positive" in response
        assert "r = 0.720" in response

    def test_scalar_mean_conclusion(self):
        stdout = _sentinel(kind="mean", value=99.5, columns=["score"])
        er = _er(stdout=stdout)
        response = render_response(er, user_question="mean of score")
        assert "mean" in response.lower()
        assert "99.5" in response or "99.5000" in response

    def test_table_response_starts_with_transformed_data(self):
        er = _er(stdout="")
        df = pd.DataFrame({"A": range(50), "B": range(50)})
        response = render_response(er, output_df=df,
                                   user_question="filter rows where A > 10")
        assert "Transformed data" in response
        assert "50" in response

    def test_table_truncation_note_when_over_limit(self):
        er = _er(stdout="")
        df = pd.DataFrame({"X": range(200)})
        response = render_response(er, output_df=df,
                                   user_question="sort by X", max_table_rows=100)
        assert "200" in response
        assert "100" in response  # "Showing first 100"

    def test_empty_df_returns_no_rows_message(self):
        er = _er(stdout="")
        response = render_response(er, output_df=pd.DataFrame())
        assert "no rows" in response.lower()

    def test_none_output_returns_executed_message(self):
        er = _er(stdout="")
        response = render_response(er)
        assert "executed" in response.lower() or "no output" in response.lower()

    def test_never_raises_on_internal_error(self):
        """render_response must always return a string, even if internals break."""
        # Pass a broken execution_result that would cause internal errors
        response = render_response({"status": None, "returncode": None,
                                    "execution_stdout": None})
        assert isinstance(response, str)

    def test_no_rows_transform_returns_empty_message(self):
        er = _er(stdout="")
        response = render_response(er, output_df=pd.DataFrame({"col": []}),
                                   user_question="filter rows")
        assert "no rows" in response.lower()


# ---------------------------------------------------------------------------
# 1g. _fmt number formatting helper
# ---------------------------------------------------------------------------

class TestFmt:

    def test_integer_formatted_with_commas(self):
        assert _fmt(590540) == "590,540"

    def test_float_formatted_to_4_decimal_places(self):
        assert _fmt(0.123456) == "0.1235"

    def test_whole_float_formatted_as_integer(self):
        assert _fmt(100.0) == "100"

    def test_none_returns_none_string(self):
        assert _fmt(None) == "None"
