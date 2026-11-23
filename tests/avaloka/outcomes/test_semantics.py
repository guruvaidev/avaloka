"""B1: column semantics (§1.2). One 12-column fixture, the expected role per
column; ``zip`` and ``state_fips`` are places, not identifiers.

Bite (run by hand, in the PR): move the geo detector below the
``_ID_NAME_RE`` check in ``classify_column`` and ``zip``/``state_fips`` read
as ``id``, so ``test_twelve_columns_get_their_roles`` goes red.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.outcomes import semantics as S


def _twelve(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    states = ["06", "36", "48", "12", "17", "53"]
    return pd.DataFrame({
        "order_date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "order_id": [f"O{i:05d}" for i in range(n)],
        "zip": rng.choice(["94103", "10001", "73301", "33101", "60601"], n),
        "state_fips": rng.choice(states, n),
        "country": rng.choice(["US", "France", "DEU", "jp"], n),
        "revenue": rng.normal(100, 20, n).round(2),
        "units": rng.integers(0, 40, n),
        "discount_rate": rng.uniform(0, 0.3, n).round(3),
        "share_pct": rng.uniform(0, 100, n).round(1),
        "category": rng.choice(["toys", "books", "garden"], n),
        "fiscal_year": rng.choice([2023, 2024, 2025], n),
        "temperature_delta": rng.normal(0, 1, n),
    })


EXPECTED = {
    "order_date": ("time", None),
    "order_id": ("id", None),
    "zip": ("geo", None),
    "state_fips": ("geo", None),
    "country": ("geo", None),
    "revenue": ("measure", "currency"),
    "units": ("measure", "count"),
    "discount_rate": ("measure", "fraction"),
    "share_pct": ("measure", "percent"),
    "category": ("dimension", None),
    "fiscal_year": ("time", None),
    "temperature_delta": ("measure", "unitless"),
}


def test_twelve_columns_get_their_roles():
    sem = S.infer(_twelve())
    got = {k: (v.role, v.unit) for k, v in sem.items()}
    assert got == EXPECTED
    assert sem["zip"].geo_kind == "us_zip"
    assert sem["state_fips"].geo_kind == "us_state"
    assert sem["country"].geo_kind == "country"
    assert sem["order_date"].grain == "day"
    assert sem["fiscal_year"].grain == "year"


def test_time_grain_follows_the_median_gap():
    months = pd.Series(pd.date_range("2024-01-01", periods=12, freq="MS").astype(str))
    c = S.classify_column("period", months)
    assert (c.role, c.grain) == ("time", "month")


def test_string_dates_need_95_percent():
    vals = ["2024-01-01"] * 94 + ["soon"] * 6
    assert S.classify_column("when", pd.Series(vals)).role != "time"
    vals = ["2024-01-0%d" % (i % 9 + 1) for i in range(96)] + ["soon"] * 4
    assert S.classify_column("when", pd.Series(vals)).role == "time"


def test_finance_hint_makes_amount_currency_and_a_code_column_names_it():
    df = pd.DataFrame({"balance_amount": [1.5, 2.25, 3.0, 9.0], "ccy": ["USD"] * 4,
                       "segment": ["a", "b", "a", "b"]})
    sem = S.infer(df, domain={"hints": {"units": ["one currency per amount (or a currency column)"]}})
    assert sem["balance_amount"].unit == "currency"
    assert S.currency_code(df, sem) == "USD"
    df["ccy"] = ["USD", "EUR", "USD", "USD"]
    assert S.currency_code(df, S.infer(df)) is None


def test_target_outranks_every_other_role():
    df = pd.DataFrame({"order_id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]})
    assert S.infer(df, target="order_id")["order_id"].role == "target"


def test_stage_set_requires_monotone_means():
    df = pd.DataFrame({"impressions": [1000, 900], "clicks": [50, 40], "conversions": [5, 4]})
    sem = S.infer(df)
    assert S.stage_set(df, sem) == ["impressions", "clicks", "conversions"]
    df["conversions"] = [500, 900]
    assert S.stage_set(df, S.infer(df)) == []


@pytest.mark.parametrize("n,role", [(30, "dimension"), (80, "high_card_dimension")])
def test_dimension_cardinality_bands(n, role):
    s = pd.Series([f"c{i % n}" for i in range(400)])
    assert S.classify_column("cat", s).role == role


def test_semantics_never_copy_values():
    df = pd.DataFrame({"secret_city": ["Xyzzyville"] * 3 + ["Plughton"] * 3})
    dump = repr([c.to_dict() for c in S.infer(df).values()])
    assert "Xyzzyville" not in dump and "Plughton" not in dump
