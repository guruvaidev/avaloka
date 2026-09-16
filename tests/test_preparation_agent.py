"""Cleansing and feature engineering, with leakage as the first-class concern."""

import numpy as np
import pandas as pd
import pytest

from app.agents.preparation_agent import (Action, Objective, PreparationPlan,
                                          _clean_numeric_token, apply_preparation,
                                          choose_imputation, detect_numeric_coercion,
                                          fit_preparation, preparation_agent_node,
                                          suggest_ratio_features)


# --------------------------------------------------------------------------- #
# Currency and numeric-wearing-a-costume
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("$1,234.50", 1234.50),
    ("£99", 99.0),
    ("€1234", 1234.0),
    ("₹2,50,000", 250000.0),      # lakh grouping: separators removed regardless
    ("  $ 42 ", 42.0),
    ("(1,234)", -1234.0),          # accounting negative
    ("$(500)", -500.0),
    ("12%", 0.12),
    ("-7.5", -7.5),
    ("+3", 3.0),
    ("1.5e3", 1500.0),
    ("1.234,56", 1234.56),
    ("€1.234,56", 1234.56),
    ("N/A", None), ("", None), ("--", None), ("unknown", None),
    ("not a number", None),
    ("12/05/2024", None),          # a date must not become 12 divided by anything
])
def test_numeric_token_parsing(raw, expected):
    got = _clean_numeric_token(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_currency_column_is_detected_and_coerced():
    df = pd.DataFrame({"revenue": ["$1,200.00", "$980.50", "$1,000", "$2,340.25"]})
    plan = fit_preparation(df)
    assert "revenue" in plan.numeric_coercions
    assert plan.numeric_coercions["revenue"]["had_currency_marks"] is True
    out = apply_preparation(df, plan)
    assert pd.api.types.is_numeric_dtype(out["revenue"])
    assert out["revenue"].tolist() == [1200.0, 980.5, 1000.0, 2340.25]


def test_a_genuinely_categorical_column_is_left_alone():
    df = pd.DataFrame({"region": ["north", "south", "north", "east"]})
    assert detect_numeric_coercion(df["region"]) is None


def test_european_decimal_commas_are_coerced_when_grouping_is_unambiguous():
    df = pd.DataFrame({"amount": ["1.234,50", "2.345,00", "3.456,75", "4.567,25"]})
    plan = fit_preparation(df)
    out = apply_preparation(df, plan)
    assert out["amount"].tolist() == [1234.5, 2345.0, 3456.75, 4567.25]


# --------------------------------------------------------------------------- #
# Imputation — and the refusal to impute
# --------------------------------------------------------------------------- #

def test_skewed_numeric_uses_median_and_says_why():
    s = pd.Series([1, 1, 2, 2, 3, 3, 4, 500.0])
    strategy, value, reason = choose_imputation(s, missing_rate=0.1)
    assert strategy == "median"
    assert "skew" in reason


def test_symmetric_numeric_uses_mean():
    s = pd.Series([10, 11, 12, 13, 14, 15.0])
    strategy, _, reason = choose_imputation(s, missing_rate=0.1)
    assert strategy == "mean"
    assert "symmetric" in reason


def test_datetime_gaps_are_never_filled():
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01", None]))
    strategy, value, reason = choose_imputation(s, missing_rate=0.33)
    assert strategy == "none" and value is None
    assert "did not happen" in reason


def test_mostly_missing_column_is_flagged_not_filled():
    df = pd.DataFrame({"sparse": [1.0] + [np.nan] * 9})
    plan = fit_preparation(df)
    assert "sparse" in plan.unimputable
    assert "sparse" not in plan.imputations
    assert any(d.action is Action.FLAG_UNIMPUTABLE for d in plan.decisions)


def test_every_imputation_records_a_reason():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": ["x", "x", None, "y"]})
    plan = fit_preparation(df)
    for d in plan.decisions:
        if d.action is Action.IMPUTE:
            assert d.detail and "missing" in d.detail


# --------------------------------------------------------------------------- #
# THE regression that matters: fit on train only
# --------------------------------------------------------------------------- #

def test_imputation_value_comes_from_train_rows_only():
    """The defect this module exists to prevent."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [1000.0, 2000.0, np.nan, 3000.0]})
    plan = fit_preparation(train)
    fitted = plan.imputations["x"]["value"]
    assert fitted == pytest.approx(2.0), "fill must come from train, not the full data"
    out = apply_preparation(test, plan)
    assert out["x"].iloc[2] == pytest.approx(2.0)
    assert out["x"].iloc[2] != pytest.approx(pd.concat([train["x"], test["x"]]).mean())


def test_category_vocabulary_comes_from_train_rows_only():
    train = pd.DataFrame({"plan": ["a"] * 100 + ["b"] * 99 + ["rare"]})
    test = pd.DataFrame({"plan": ["a", "b", "unseen_in_train"]})
    plan = fit_preparation(train)
    assert "rare" in plan.rare_categories.get("plan", [])
    out = apply_preparation(test, plan)
    assert out["plan"].tolist() == ["a", "b", "unseen_in_train"], (
        "applying must not re-derive rarity from the frame it is applied to")


def test_apply_is_pure_with_respect_to_the_frame_it_is_given():
    """Applying to one row must equal applying to many, row for row."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    plan = fit_preparation(train)
    big = pd.DataFrame({"x": [np.nan, 10.0, np.nan, 20.0]})
    whole = apply_preparation(big, plan)["x"].tolist()
    piecemeal = [apply_preparation(big.iloc[[i]], plan)["x"].iloc[0] for i in range(len(big))]
    assert whole == pytest.approx(piecemeal)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def test_missingness_indicator_is_computed_before_imputation():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]})
    plan = fit_preparation(df, objective=Objective.MODELLING)
    assert "x__was_missing" in plan.indicator_columns
    out = apply_preparation(df, plan)
    assert out["x__was_missing"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert out["x"].isna().sum() == 0, "the indicator must survive imputation"


def test_date_parts_and_cyclical_encoding():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15", "2024-06-30", "2024-12-31"])})
    plan = fit_preparation(df, objective=Objective.MODELLING)
    out = apply_preparation(df, plan)
    assert out["ts__month"].tolist() == [1, 6, 12]
    assert out["ts__dayofweek"].notna().all()
    # December and January must be near each other in the encoded space
    dec = np.array([out["ts__month_sin"].iloc[2], out["ts__month_cos"].iloc[2]])
    jan = np.array([out["ts__month_sin"].iloc[0], out["ts__month_cos"].iloc[0]])
    jun = np.array([out["ts__month_sin"].iloc[1], out["ts__month_cos"].iloc[1]])
    assert np.linalg.norm(dec - jan) < np.linalg.norm(dec - jun)


def test_ratios_are_declared_not_guessed():
    df = pd.DataFrame({"rooms": [10.0, 20.0], "households": [5.0, 0.0]})
    plan = fit_preparation(df)
    assert plan.ratio_features == []
    plan = suggest_ratio_features(df, plan, [("rooms", "households", "rooms_per_household")])
    out = apply_preparation(df, plan)
    assert out["rooms_per_household"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(out["rooms_per_household"].iloc[1]), "divide by zero must be NaN, not inf"


# --------------------------------------------------------------------------- #
# Objective awareness
# --------------------------------------------------------------------------- #

def test_analysis_objective_does_not_build_ml_encodings():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-15"] * 10),
        "x": [1.0, np.nan] * 5,
    })
    ml = fit_preparation(df, objective=Objective.MODELLING)
    analysis = fit_preparation(df, objective=Objective.ANALYSIS)
    assert ml.cyclical_columns and not analysis.cyclical_columns
    assert ml.indicator_columns and not analysis.indicator_columns
    assert analysis.date_part_columns, "readable date parts are useful to a human too"


def test_target_column_is_never_prepared():
    df = pd.DataFrame({"y": ["$1", "$2", "$3", "$4"], "x": [1.0, 2.0, 3.0, 4.0]})
    plan = fit_preparation(df, target="y")
    assert "y" not in plan.numeric_coercions, "the target is the model's business, not ours"


# --------------------------------------------------------------------------- #
# Graph node
# --------------------------------------------------------------------------- #

def test_node_returns_plan_and_frame():
    df = pd.DataFrame({"revenue": ["$10", "$20", "$30", None]})
    out = preparation_agent_node({"dataframe": df, "objective": "analysis"})
    assert set(out) == {"preparation_plan", "prepared_dataframe", "preparation_summary"}
    assert out["preparation_plan"]["objective"] == "analysis"
    assert pd.api.types.is_numeric_dtype(out["prepared_dataframe"]["revenue"])


def test_node_fits_on_train_index_when_given():
    df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 999.0, 999.0, np.nan, np.nan]})
    out = preparation_agent_node({"dataframe": df, "objective": "modelling",
                                  "train_index": [0, 1, 2, 3]})
    assert out["preparation_plan"]["fitted_rows"] == 4
    assert out["prepared_dataframe"]["x"].iloc[6] == pytest.approx(1.0)


def test_node_without_a_dataframe_is_a_no_op():
    assert preparation_agent_node({}) == {}


def test_unknown_objective_degrades_rather_than_raising():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = preparation_agent_node({"dataframe": df, "objective": "nonsense"})
    assert out["preparation_plan"]["objective"] == "both"


def test_spec_is_registered_and_declares_its_contract():
    from app.agents.contract import registry
    spec = registry()["preparation"]
    assert spec.stage.value == "prepare"
    assert "dataframe" in spec.reads
    assert set(spec.writes) == {"preparation_plan", "prepared_dataframe", "preparation_summary"}
