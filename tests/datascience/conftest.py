"""Datasets with planted ground truth, for the data-science capability suite.

Every factory here builds a dataset whose *correct answer is known by
construction*, so a test can assert that Avaloka found the planted fact rather
than merely that it produced output. A run that emits a full report and misses
the leak is the failure mode worth guarding, and it looks identical to success
from the outside.

Seeded throughout: a capability test that passes four times in five is not a
capability test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train

SEED = 20260914
N = 400


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _write(frame: pd.DataFrame, tmp_path: Path, name: str = "data.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def write_csv(tmp_path):
    def _inner(frame: pd.DataFrame, name: str = "data.csv") -> Path:
        return _write(frame, tmp_path, name)
    return _inner


@pytest.fixture
def analyze(tmp_path):
    """Run an exploratory mission and hand back the result and its bundle."""
    def _inner(frame: pd.DataFrame, goal: str = "Understand this dataset", **kw):
        out = tmp_path / kw.pop("out", "analysis")
        ctx = MissionContext(
            kind=MissionKind.ANALYZE, goal=goal,
            data_source=str(_write(frame, tmp_path, kw.pop("name", "data.csv"))),
            output_dir=out, budget=Budget(limit_usd=None), **kw,
        )
        return run_analyze(ctx), out
    return _inner


@pytest.fixture
def train(tmp_path):
    """Run a training mission and hand back the result and its bundle."""
    def _inner(frame: pd.DataFrame, target: str, **kw):
        out = tmp_path / kw.pop("out", "model")
        ctx = MissionContext(
            kind=MissionKind.TRAIN, goal=kw.pop("goal", f"Predict {target}"),
            data_source=str(_write(frame, tmp_path, kw.pop("name", "data.csv"))),
            output_dir=out, budget=Budget(limit_usd=None), target=target, **kw,
        )
        return run_train(ctx), out
    return _inner


# ── Datasets ────────────────────────────────────────────────────────────────

@pytest.fixture
def churn_like() -> pd.DataFrame:
    """Binary classification with a genuine, moderate signal.

    tenure and support_tickets really do drive churn, so a model that beats a
    trivial baseline is finding something real rather than memorising noise.
    """
    rng = _rng()
    tenure = rng.integers(1, 60, N)
    tickets = rng.poisson(2, N)
    spend = rng.gamma(2.0, 45.0, N)
    logit = -0.06 * tenure + 0.45 * tickets - 1.0
    churned = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(N)],
        "region": rng.choice(["north", "south", "east", "west"], N),
        "plan": rng.choice(["basic", "pro", "enterprise"], N, p=[.6, .3, .1]),
        "monthly_spend": spend.round(2),
        "tenure_months": tenure,
        "support_tickets": tickets,
        "churned": churned,
    })


@pytest.fixture
def leaky() -> pd.DataFrame:
    """A feature that is the target wearing a hat."""
    rng = _rng()
    frame = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    frame["defaulted"] = (frame["a"] > 0).astype(int)
    frame["collections_flag"] = frame["defaulted"]        # the plant
    return frame


@pytest.fixture
def time_ordered() -> pd.DataFrame:
    """Revenue with a trend, and the date stored as text — as CSVs do.

    Shuffled on disk so nothing can pass by accident of row order.
    """
    rng = _rng()
    dates = pd.date_range("2024-01-01", periods=N, freq="D")
    promo = rng.integers(0, 2, N)
    base = rng.normal(100, 8, N)
    revenue = base * 2 + promo * 12 + np.arange(N) * 0.35 + rng.normal(0, 2.5, N)
    frame = pd.DataFrame({
        "order_date": dates.astype(str),
        "promo": promo,
        "base_demand": base.round(2),
        "revenue": revenue.round(2),
    })
    return frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


@pytest.fixture
def dirty() -> pd.DataFrame:
    """Everything a real export does wrong, at once."""
    rng = _rng()
    frame = pd.DataFrame({
        "row_id": [f"R{i:06d}" for i in range(N)],       # identifier
        "country": ["GB"] * N,                            # constant
        "amount": rng.gamma(2.0, 30.0, N).round(2),
        "score": rng.normal(50, 10, N).round(2),
        "segment": rng.choice(["a", "b", "c"], N),
    })
    # 35% missing, deliberately past HIGH_MISSING_PCT (0.30) so the flag fires.
    frame.loc[rng.choice(N, int(N * 0.35), replace=False), "amount"] = np.nan
    frame.loc[rng.choice(N, 5, replace=False), "score"] = 5000.0     # outliers
    frame["converted"] = (rng.random(N) < 0.3).astype(int)
    # A handful of duplicated rows, not a flood: enough to be caught, few enough
    # that row_id stays above ID_UNIQUE_RATIO (0.98) and is still read as an
    # identifier. Duplicating a third of the file would make it genuinely
    # non-unique, and calling it an id would then be the wrong answer.
    return pd.concat([frame, frame.head(5)], ignore_index=True)


@pytest.fixture
def imbalanced() -> pd.DataFrame:
    """~4% positives — accuracy is a trap here, which is the point."""
    rng = _rng()
    x = rng.normal(size=N)
    fraud = (rng.random(N) < 0.04).astype(int)
    return pd.DataFrame({
        "amount": np.abs(x * 100 + 200).round(2),
        "channel": rng.choice(["web", "app", "phone"], N),
        "risk_score": (x + fraud * 1.5).round(3),
        "is_fraud": fraud,
    })


@pytest.fixture
def pure_noise() -> pd.DataFrame:
    """No relationship at all. A model here must be reported as having found none."""
    rng = _rng()
    return pd.DataFrame({
        "a": rng.normal(size=N),
        "b": rng.normal(size=N),
        "c": rng.choice(["x", "y", "z"], N),
        "outcome": (rng.random(N) < 0.5).astype(int),
    })


@pytest.fixture
def continuous_only() -> pd.DataFrame:
    """Every feature a continuous measurement, all values distinct.

    The shape of most sensor, pricing and latency data, and the shape that
    exposed the identifier rule: it dropped every feature and the fit then
    failed with "Found array with 0 feature(s)".
    """
    rng = _rng()
    latency = rng.gamma(2.0, 30.0, N)
    load = rng.normal(50, 12, N)
    return pd.DataFrame({
        "latency_ms": latency,
        "cpu_load": load,
        "throughput": (1000 / (latency + 1) + load * 0.3 + rng.normal(0, 1, N)),
    })
