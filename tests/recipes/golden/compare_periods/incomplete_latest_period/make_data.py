"""Seeded generator for the golden case compare_periods/incomplete_latest_period; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2024-06-20", freq="D")
    n = len(dates)
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "date": dates.strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
        "units": rng.integers(1, 50, n),
    })
