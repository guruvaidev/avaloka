"""Seeded generator for the golden case trend/too_few_periods; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(32)
    n = 90
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
        "units": rng.integers(1, 50, n),
        "conversion_rate": np.round(rng.uniform(0.01, 0.4, n), 4),
    })
