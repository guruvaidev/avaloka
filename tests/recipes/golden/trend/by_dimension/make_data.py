"""Seeded generator for the golden case trend/by_dimension; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(35)
    n = 731
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east"], n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
    })
