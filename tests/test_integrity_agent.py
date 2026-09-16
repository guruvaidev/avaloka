"""Leakage and data-integrity detection.

Each test builds the specific defect the check exists to catch, so a regression
here is legible rather than a number moving.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
import numpy as np  # noqa: E402

from app.agents.integrity_agent import (Severity, check_duplicate_rows,  # noqa: E402
                                        check_identifier_features,
                                        check_near_constant,
                                        check_target_correlation,
                                        check_target_in_features,
                                        check_temporal_leakage,
                                        run_integrity_checks)


def clean_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50_000, 12_000, n),
        "region": rng.choice(["north", "south", "east"], n),
        "churned": rng.integers(0, 2, n),
    })


# --------------------------------------------------------------------------- #
# The check that matters most: target leakage
# --------------------------------------------------------------------------- #

def test_a_feature_derived_from_the_target_is_flagged_as_a_blocker():
    df = clean_frame()
    df["churn_flag_copy"] = df["churned"] * 1.0          # the classic leak
    findings = check_target_correlation(df, "churned", ["age", "income", "churn_flag_copy"])
    assert [f.column for f in findings] == ["churn_flag_copy"]
    assert findings[0].severity is Severity.BLOCKER
    assert findings[0].evidence["correlation"] >= 0.98


def test_an_exact_target_copy_with_a_different_numeric_dtype_is_flagged():
    df = clean_frame()
    df["survived_leak"] = df["churned"].astype(float)
    findings = check_target_correlation(df, "churned", ["age", "survived_leak"])
    assert [finding.column for finding in findings] == ["survived_leak"]
    assert findings[0].evidence["method"] == "exact_copy"


def test_a_genuinely_predictive_feature_is_not_flagged():
    """Signal must survive. A check that flags real features is worse than none."""
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "churned": y,
        # strong but far from perfect — this is what a good feature looks like
        "usage": y * 2.0 + rng.normal(0, 2.0, n),
    })
    assert check_target_correlation(df, "churned", ["usage"]) == []


def test_target_offered_as_its_own_feature_is_blocked():
    findings = check_target_in_features(["a", "churned"], "churned", ["a", "churned"])
    assert findings and findings[0].severity is Severity.BLOCKER


def test_high_cardinality_categorical_is_not_correlation_checked():
    """Encoding 1000 categories to integers yields a meaningless correlation."""
    n = 300
    df = pd.DataFrame({
        "churned": np.arange(n) % 2,
        "uuid": [f"id-{i}" for i in range(n)],
    })
    assert check_target_correlation(df, "churned", ["uuid"]) == []


def test_correlation_check_survives_all_nan_column():
    df = clean_frame(60)
    df["broken"] = np.nan
    check_target_correlation(df, "churned", ["broken"])  # must not raise


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def test_substantial_duplicate_rows_block():
    df = clean_frame(100)
    df = pd.concat([df, df.head(20)], ignore_index=True)   # 20% duplicated
    findings = check_duplicate_rows(df)
    assert findings and findings[0].severity is Severity.BLOCKER
    assert findings[0].evidence["duplicates"] == 20


def test_a_couple_of_duplicates_warn_rather_than_block():
    df = clean_frame(1000)
    df = pd.concat([df, df.head(2)], ignore_index=True)    # 0.2%
    findings = check_duplicate_rows(df)
    assert findings and findings[0].severity is Severity.WARNING


def test_clean_data_produces_no_duplicate_finding():
    assert check_duplicate_rows(clean_frame()) == []


# --------------------------------------------------------------------------- #
# Identifier-like and degenerate features
# --------------------------------------------------------------------------- #

def test_an_id_column_is_flagged():
    df = clean_frame(200)
    df["customer_id"] = [f"c{i}" for i in range(200)]
    flagged = [f.column for f in check_identifier_features(df, ["customer_id", "region"])]
    assert flagged == ["customer_id"]


def test_a_constant_column_is_flagged():
    df = clean_frame(120)
    df["country"] = "IN"
    flagged = [f.column for f in check_near_constant(df, ["country", "age"])]
    assert "country" in flagged


def test_normal_columns_are_left_alone():
    df = clean_frame(200)
    assert check_identifier_features(df, ["region", "age"]) == []


# --------------------------------------------------------------------------- #
# Temporal leakage
# --------------------------------------------------------------------------- #

def test_training_on_the_future_is_blocked():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-03-01", "2026-06-01", "2026-01-01", "2026-02-01"]),
        "v": [1, 2, 3, 4],
    })
    # rows 0-1 are "train" and extend past the start of rows 2-3
    findings = check_temporal_leakage(df, "ts", split_index=2)
    assert findings and findings[0].severity is Severity.BLOCKER


def test_a_correctly_ordered_split_is_clean():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
        "v": [1, 2, 3, 4],
    })
    assert check_temporal_leakage(df, "ts", split_index=2) == []


def test_no_time_column_means_no_temporal_finding():
    assert check_temporal_leakage(clean_frame(), None, 10) == []
    assert check_temporal_leakage(clean_frame(), "ts", None) == []


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def test_clean_dataset_is_safe_to_train():
    report = run_integrity_checks(clean_frame(), target="churned")
    assert report.safe_to_train is True
    assert report.blockers == []


def test_leaking_dataset_is_not_safe_to_train():
    df = clean_frame()
    df["leak"] = df["churned"]
    report = run_integrity_checks(df, target="churned")
    assert report.safe_to_train is False
    assert any(f.check == "target_correlation" for f in report.blockers)


def test_blockers_are_reported_first():
    df = clean_frame(200)
    df["constant"] = 1          # warning
    df["leak"] = df["churned"]  # blocker
    report = run_integrity_checks(df, target="churned")
    assert report.findings[0].severity is Severity.BLOCKER


def test_report_serialises_for_the_api():
    df = clean_frame()
    df["leak"] = df["churned"]
    payload = run_integrity_checks(df, target="churned").as_dict()
    assert payload["safe_to_train"] is False
    assert payload["n_blockers"] >= 1
    assert isinstance(payload["findings"], list)
    assert set(payload["findings"][0]) == {"check", "severity", "column", "detail", "evidence"}


def test_summary_is_human_readable():
    df = clean_frame()
    df["leak"] = df["churned"]
    assert "leak" in run_integrity_checks(df, target="churned").summary()


def test_node_returns_declared_keys_only():
    from app.agents.integrity_agent import integrity_agent_node
    df = clean_frame()
    df["leak"] = df["churned"]
    out = integrity_agent_node({"dataframe": df, "target_column": "churned"})
    assert set(out) == {"integrity_report", "integrity_safe_to_train"}
    assert out["integrity_safe_to_train"] is False


def test_node_is_a_noop_without_a_dataframe():
    from app.agents.integrity_agent import integrity_agent_node
    assert integrity_agent_node({}) == {}
