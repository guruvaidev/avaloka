"""Seeded generator for the golden case trend/gaps; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(33)
    n = 730
    df = pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
    })
    # two whole months have no rows at all
    return df[~df["date"].str.startswith(("2023-06", "2024-03"))].reset_index(drop=True)
