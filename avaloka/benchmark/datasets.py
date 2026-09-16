"""Benchmark datasets — deterministic synthetic generators + a Kaggle adapter.

The synthetic generators need *no* network and carry ground truth by
construction (a known dominant segment, a planted leak, known dirty columns, a
known workload lane), so the benchmark always runs and always has something to
grade against. The Kaggle adapter reuses the repo's existing download convention
(``tests/download_kaggle_datasets.py`` writes to ``app/sample_data/<slug__>``),
so the same datasets that exercise the coding agents also exercise the CLI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Same location the existing kaggle test harness downloads into.
KAGGLE_DATA_DIR = Path("app/sample_data")


# --------------------------------------------------------------------------
# Synthetic generators. Each writes a CSV to `dest` and returns the path.
# --------------------------------------------------------------------------
def build_churn(dest: Path, n: int = 1500, seed: int = 7) -> Path:
    """Binary classification with real signal and a dominant categorical segment."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 20, n).clip(15, 150).round(2)
    tickets = rng.poisson(1.5, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.56, 0.28, 0.16])
    logit = (-0.05 * tenure + 0.02 * monthly + 0.5 * tickets
             + np.where(contract == "month-to-month", 1.1, 0.0) - 1.4)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure": tenure, "monthly_charges": monthly, "support_tickets": tickets,
        "contract": contract, "churned": churn,
    })
    df.to_csv(dest, index=False)
    return dest


def build_leaky(dest: Path, n: int = 800, seed: int = 1) -> Path:
    """A dataset where one feature deterministically encodes the target (leakage)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.8 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)) > 0).astype(int)
    df = pd.DataFrame({
        "feature_a": x1, "feature_b": x2, "noise": rng.normal(size=n),
        "leak_score": y.astype(float),          # numeric near-perfect correlation
        "outcome_label": np.where(y == 1, "yes", "no"),  # deterministic categorical map
        "target": y,
    })
    df.to_csv(dest, index=False)
    return dest


def build_housing(dest: Path, n: int = 1200, seed: int = 5) -> Path:
    """Regression with a genuinely predictable target."""
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 400, n).clip(300, 4000)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 80, n)
    location = rng.choice(["urban", "suburban", "rural"], n, p=[0.4, 0.4, 0.2])
    loc_premium = np.select([location == "urban", location == "suburban"], [120_000, 60_000], 0)
    price = (150 * area + 15_000 * bedrooms - 800 * age + loc_premium
             + rng.normal(0, 25_000, n)).round(0)
    df = pd.DataFrame({
        "area_sqft": area.round(1), "bedrooms": bedrooms, "age_years": age,
        "location": location, "price": price,
    })
    df.to_csv(dest, index=False)
    return dest


def build_dirty(dest: Path, n: int = 500, seed: int = 3) -> Path:
    """A deliberately messy table for data-quality detection (DE benchmark)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "record_id": [f"R{i:06d}" for i in range(n)],       # id (near-unique)
        "amount": rng.normal(100, 30, n).round(2),          # numeric
        "region": rng.choice(["north", "south", "east", "west"], n),  # categorical
        "income": rng.normal(50_000, 12_000, n).round(0),   # will be holed out
        "status_flag": np.ones(n, dtype=int),               # constant column
        "free_text_note": [f"note-{rng.integers(0, 10**9)}" for _ in range(n)],  # high-cardinality
    })
    # Punch ~40% missing into `income` (high-missingness).
    holes = rng.random(n) < 0.4
    df.loc[holes, "income"] = np.nan
    # Duplicate ~5% of rows.
    dupes = df.sample(frac=0.05, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df.to_csv(dest, index=False)
    return dest


def build_wide_sampled(dest: Path, n: int = 120_000, seed: int = 11) -> Path:
    """Above the online row threshold (100k) → the router must pick `sampled_online`."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.normal(size=n).round(4),
        "y": rng.integers(0, 2, n),
        "g": rng.choice(["a", "b", "c"], n),
    })
    df.to_csv(dest, index=False)
    return dest


def build_batch_exact(dest: Path, n: int = 20_000, seed: int = 13) -> Path:
    """Mid-size numeric table to prove exact map-reduce convergence."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "value": rng.normal(10, 3, n),
        "weight": rng.gamma(2.0, 2.0, n),
        "bucket": rng.choice(["lo", "hi"], n),
    })
    df.to_csv(dest, index=False)
    return dest


# --------------------------------------------------------------------------
# Kaggle adapter — reuses the repo's existing download convention.
# --------------------------------------------------------------------------
def ensure_kaggle_dataset(slug: str, preferred_file: Optional[str] = None) -> Optional[Path]:
    """Return a local CSV for a Kaggle ``slug``, downloading it if needed.

    Mirrors ``tests/download_kaggle_datasets.py`` / ``test_e2e_kaggle`` exactly
    (same ``app/sample_data/<slug__>`` layout, same ``kaggle`` CLI + ``--unzip``),
    so a dataset already fetched for the coding-agent tests is reused as-is.
    Returns ``None`` (rather than raising) when the dataset is unavailable and
    cannot be downloaded, so the benchmark can skip gracefully.
    """
    target_dir = KAGGLE_DATA_DIR / slug.replace("/", "__")

    if preferred_file:
        for candidate in (KAGGLE_DATA_DIR / preferred_file, target_dir / preferred_file):
            if candidate.exists():
                return candidate

    existing = sorted(target_dir.rglob("*.csv")) if target_dir.exists() else []
    if existing:
        return existing[0]

    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        LOGGER.warning("Kaggle CLI not on PATH; cannot fetch %s. "
                       "Run tests/download_kaggle_datasets.py first.", slug)
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Kaggle dataset %s ...", slug)
    proc = subprocess.run(
        [kaggle_cli, "datasets", "download", "-d", slug, "--unzip", "-p", str(target_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        LOGGER.error("Failed to download %s: %s", slug, proc.stderr.strip())
        return None
    found = sorted(target_dir.rglob("*.csv"))
    return found[0] if found else None


def build_timeseries(dest: Path, n: int = 900, seed: int = 17) -> Path:
    """Revenue with a trend, the date stored as text, rows shuffled on disk.

    The shape a CSV export actually has. A daily date column is near-unique,
    which is how it came to be profiled as an identifier -- denying the planner
    the one fact it uses to pick a temporal split. Shuffled so nothing can pass
    by accident of row order.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    promo = rng.integers(0, 2, n)
    base = rng.normal(100.0, 8.0, n)
    revenue = base * 2.0 + promo * 14.0 + np.arange(n) * 0.4 + rng.normal(0, 2.0, n)
    frame = pd.DataFrame({
        "order_date": dates.astype(str),
        "promo": promo,
        "base_demand": base.round(2),
        "revenue": revenue.round(2),
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame.to_csv(dest, index=False)
    return dest


def build_constant_and_target(dest: Path, n: int = 700, seed: int = 19) -> Path:
    """A single-valued column alongside a real target.

    Ordinary in real exports -- a flag that never varies, a country column in a
    single-country extract -- and it crashed every training mission: the
    engineer drops the column, the validator then indexed the pre-transform
    profile against the post-transform frame.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n)
    tickets = rng.poisson(2.0, n)
    logit = -0.06 * tenure + 0.5 * tickets - 1.0
    frame = pd.DataFrame({
        "row_id": [f"R{i:06d}" for i in range(n)],
        "country": ["GB"] * n,
        "tenure": tenure,
        "support_tickets": tickets,
        "monthly_charges": rng.gamma(2.0, 40.0, n).round(2),
        "churned": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
    })
    frame.to_csv(dest, index=False)
    return dest


def build_continuous_only(dest: Path, n: int = 800, seed: int = 23) -> Path:
    """Every feature a continuous measurement, every value distinct.

    The shape of sensor, pricing and latency data. The identifier guard was
    "all values distinct", so it dropped all of them and the fit then failed
    with "Found array with 0 feature(s)".
    """
    rng = np.random.default_rng(seed)
    latency = rng.gamma(2.0, 30.0, n)
    load = rng.normal(50.0, 12.0, n)
    frame = pd.DataFrame({
        "latency_ms": latency,
        "cpu_load": load,
        "throughput": (1000.0 / (latency + 1.0) + load * 0.3 + rng.normal(0, 1.0, n)),
    })
    frame.to_csv(dest, index=False)
    return dest


def build_no_signal(dest: Path, n: int = 800, seed: int = 29) -> Path:
    """Features and target are independent. Nothing is there to be found.

    A candidate still trains and still reports a score. Without a baseline that
    score reads as a result, which is the failure a baseline exists to catch.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.choice(["x", "y", "z"], n),
        "outcome": (rng.random(n) < 0.5).astype(int),
    })
    frame.to_csv(dest, index=False)
    return dest
