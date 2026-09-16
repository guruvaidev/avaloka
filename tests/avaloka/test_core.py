"""Unit tests for the mission accounting core and data ingestion."""

import pandas as pd
import pytest

from avaloka.io import load_dataset
from avaloka.mission.budget import Budget, BudgetExceeded
from avaloka.mission.economics import compute_economics
from avaloka.mission.ledger import Ledger, WorkUnit


def test_budget_enforces_hard_ceiling():
    b = Budget(limit_usd=1.0)
    assert b.can_afford(0.5)
    b.spend(0.5, "step-1")
    assert b.remaining() == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        b.spend(0.75, "step-2")


def test_unlimited_budget_never_blocks():
    b = Budget(limit_usd=None)
    assert b.unlimited
    b.spend(10_000.0, "anything")
    assert b.remaining() == float("inf")


def test_ledger_aggregates():
    led = Ledger()
    led.record(WorkUnit("a", "Analyst", "x", manual_minutes=60, compute_cost_usd=0.1))
    led.record(WorkUnit("b", "Engineer", "y", manual_minutes=30, model_cost_usd=0.2))
    assert led.manual_minutes == 90
    assert led.compute_cost_usd == pytest.approx(0.1)
    assert led.model_cost_usd == pytest.approx(0.2)


def test_economics_multiplier_is_finite_for_free_local_mission():
    led = Ledger()
    led.record(WorkUnit("a", "Analyst", "x", manual_minutes=600))  # 10 hours
    econ = compute_economics(led, tier="community")  # free tier, zero compute
    # Denominator includes human review value, so the multiplier is defensible.
    assert econ.total_mission_cost_usd == 0.0
    assert econ.human_review_cost_usd > 0
    assert 0 < econ.economic_multiplier < 1000
    assert econ.estimated_labor_value_usd == pytest.approx(10 * 80.0)


def test_loader_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    handle = load_dataset(str(p))
    assert handle.n_rows == 3 and handle.n_cols == 2
    assert handle.fmt == "csv" and len(handle.sha256) == 64


def test_loader_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("/no/such/file.csv")
