"""Unit tests for the replan loop's diagnosis, policy and wording (app/agents/replan.py).

These cover the pieces in isolation; tests/test_replan_on_execution_error.py
drives the loop through the registered graph.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents import replan as rp
from app.agents import replan_disclosure as words


def test_parse_exception_reads_the_last_traceback_line():
    stderr = ("Traceback (most recent call last):\n  File \"x.py\", line 3, in main\n"
              "pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer\n")
    cls, msg = rp.parse_exception({"execution_stderr": stderr, "execution_error": "Script failed with exit status 1"})
    assert cls == "IntCastingNaNError"
    assert msg.startswith("Cannot convert non-finite values")


def test_code_use_finds_measured_and_key_columns():
    code = ("def main(df):\n"
            "    df['band'] = df['revenue'].astype(int)\n"
            "    t = int(df['cost'].astype(float).sum())\n"
            "    return df.groupby('region', as_index=False).agg(u=('units', 'sum'))\n")
    use = rp.code_use(code, ["region", "units", "revenue", "cost", "unused"])
    assert set(use.used) == {"region", "units", "revenue", "cost"}
    assert set(use.measured) == {"cost", "units"}
    assert use.keys == ["region"]


@pytest.mark.parametrize("policy,expected", [("exclude", rp.EXCLUDE_ROWS), ("impute", rp.IMPUTE)])
def test_measured_column_policy_has_both_options(policy, expected):
    s = pd.Series([1.0, np.nan] + [2.0] * 48)
    d = rp.missing_value_policy(s, column="revenue", measured=True, protected=False,
                                limits=rp.Limits(measured_policy=policy), already_excluded=0, rows_total=50)
    assert d.op == expected and d.ask is None


def test_the_shipped_default_asks_about_the_measured_column():
    assert rp.Limits().measured_policy == "ask"
    s = pd.Series([1.0, np.nan] + [2.0] * 48)
    d = rp.missing_value_policy(s, column="revenue", measured=True, protected=False,
                                limits=rp.Limits(), already_excluded=0, rows_total=50)
    assert d.op is None and d.ask["reason"] == "measured_choice" and d.ask["rows_affected"] == 1


def test_a_given_choice_is_applied_as_answered():
    s = pd.Series([np.nan] * 20 + [2.0] * 30)    # 40% missing: over both limits
    for choice, op in (("median", rp.IMPUTE), ("leave_out", rp.EXCLUDE_ROWS)):
        d = rp.missing_value_policy(s, column="revenue", measured=True, protected=False, limits=rp.Limits(),
                                    already_excluded=0, rows_total=50,
                                    choice={"choice": choice, "chosen_by": "user"})
        assert (d.op, d.chosen_by, d.ask) == (op, "user", None)


def test_keys_are_never_imputed():
    s = pd.Series([1.0, np.nan] + [2.0] * 48)
    d = rp.missing_value_policy(s, column="region_id", measured=False, protected=True,
                                limits=rp.Limits(measured_policy="impute"), already_excluded=0, rows_total=50)
    assert d.op == rp.EXCLUDE_ROWS


def test_limits_ask_instead_of_applying():
    s = pd.Series([np.nan] * 11 + [1.0] * 89)
    d = rp.missing_value_policy(s, column="c", measured=False, protected=False, limits=rp.Limits(),
                                already_excluded=0, rows_total=100)
    assert d.op is None and d.ask["reason"] == "impute_limit" and d.ask["rows_affected"] == 11
    d = rp.missing_value_policy(pd.Series([np.nan] * 3 + [1.0] * 97), column="c", measured=False,
                                protected=True, limits=rp.Limits(), already_excluded=3, rows_total=100)
    assert d.op is None and d.ask["reason"] == "exclude_limit"


def test_limits_are_configurable(monkeypatch):
    monkeypatch.setenv("AVALOKA_REPLAN_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("AVALOKA_REPLAN_MAX_IMPUTE_SHARE", "0.2")
    monkeypatch.setenv("AVALOKA_REPLAN_MAX_EXCLUDE_SHARE", "0.01")
    monkeypatch.setenv("AVALOKA_REPLAN_MAX_COLUMNS", "2")
    monkeypatch.setenv("AVALOKA_REPLAN_MEASURED_POLICY", "nonsense")
    lim = rp.Limits.from_env()
    assert lim.measured_policy == "ask"
    assert lim.max_attempts == rp.HARD_MAX_ATTEMPTS
    assert (lim.max_impute_share, lim.max_exclude_share, lim.max_columns) == (0.2, 0.01, 2)


def test_coerce_counts_each_kind_of_value():
    df = pd.DataFrame({"p": ["1.5", "$1,200.00", "N/A", "(30)", "abc", None]})
    out, rec = rp.coerce_numeric(df, "p")
    assert (rec["values_converted"], rec["null_tokens"], rec["unreadable"]) == (2, 1, 1)
    assert out["p"].tolist()[:2] == [1.5, 1200.0]
    assert rec["example"] == "$1,200.00"


def test_disclosure_lines_carry_the_counts_and_the_method():
    lines = words.disclosure_lines([
        {"op": "impute", "column": "revenue", "rows_total": 200, "rows_affected": 12, "method": "median",
         "value": 41.5},
        {"op": "exclude_rows", "column": "units", "rows_total": 200, "rows_affected": 4, "non_finite": 0},
    ])
    assert "12 of 200 rows had no value in revenue; I filled them with the median (41.50)" in lines[1]
    assert "I left out 4 of 200 rows (2.0%) that had no value in units" in lines[2]
    assert words.disclosure_lines([]) == []


def test_run_truth_keeps_no_cell_values():
    rec = rp._run_truth_transform({"op": "coerce_numeric", "column": "p", "rows_total": 6,
                                   "values_converted": 2, "example": "$1,200.00",
                                   "null_token_examples": ["N/A"]})
    assert "example" not in rec and "null_token_examples" not in rec


def test_telemetry_projection_is_closed_vocabulary_only():
    recovery = {"final": "recovered", "attempts": [{
        "error_class": "ValueError: secret 42", "error_kind": "data", "diagnosis": "dtype_mismatch",
        "diagnosed_by": "rule", "action": "data_step", "outcome": "recovered",
        "transforms": [{"op": "coerce_numeric", "column": "customer_ssn", "value": 99.5}]}]}
    wire = str(rp.recovery_telemetry(recovery))
    assert "customer_ssn" not in wire and "99.5" not in wire and "secret" not in wire
    assert rp.recovery_telemetry(recovery)["attempts"][0]["error_class"] == "other"


def test_run_truth_new_reason_codes_are_additions():
    from app.core import run_truth as rt

    for code, outcome in (("rows_excluded", rt.PARTIAL), ("recovery_exhausted", rt.FAILED),
                          ("needs_decision", rt.FAILED)):
        assert rt.make(code, kind="interactive", detail="x").outcome == outcome


def test_a_failure_the_code_did_not_raise_is_not_retried():
    # "No candidate output path was configured": the harness failed, not the code.
    state = {"execution_result": {"status": "failed", "message": "No candidate output path was configured"},
             "execution_error": "No candidate output path was configured", "execution_stderr": "",
             "coder_definition": {"code": "def main(df):\n    return df\n"}}
    diag = rp.diagnose(state, [])
    assert (diag.kind, diag.diagnosis) == (rp.INFRA, rp.NOT_RECOVERABLE)


def test_stale_prepared_copies_are_swept_when_a_turn_writes_its_first(tmp_path):
    """A data step copies the dataset; the copies of earlier turns do not stay forever."""
    import os
    import time

    base = tmp_path / ".replan"
    base.mkdir()
    stale = base / "revenue_prepared_old.csv"
    stale.write_text("a\n1\n")
    old = time.time() - rp.PREPARED_MAX_AGE_S - 60
    os.utime(stale, (old, old))
    recent = base / "revenue_prepared_recent.csv"      # another turn may still be reading it
    recent.write_text("a\n1\n")
    unrelated = base / "notes.txt"
    unrelated.write_text("keep")

    state = {"output_location": str(tmp_path / "output.csv"), "replan_input_overrides": {}}
    path = rp._prepared_path(state, str(tmp_path / "revenue.csv"))
    assert not stale.exists(), "the stale copy from an earlier turn was kept"
    assert recent.exists() and unrelated.exists()
    assert Path(path).parent == base and Path(path).name.startswith("revenue_prepared_")

    # A later data step of the same turn (overrides already set) sweeps nothing.
    os.utime(recent, (old, old))
    rp._prepared_path({**state, "replan_input_overrides": {"x": path}}, str(tmp_path / "revenue.csv"))
    assert recent.exists()
