"""Seeded generator for the golden case describe/duplicates; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(25)
    n = 460
    base = pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
        "units": rng.integers(1, 50, n),
    })
    # exact copies of existing rows, so the table is 40 rows longer than its distinct rows
    copies = base.iloc[rng.integers(0, n, 40)]
    return pd.concat([base, copies], ignore_index=True)
