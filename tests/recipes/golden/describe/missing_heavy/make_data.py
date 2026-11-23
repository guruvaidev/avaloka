"""Seeded generator for the golden case describe/missing_heavy; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(24)
    n = 300
    amount = np.round(rng.lognormal(4.0, 0.8, n), 2)
    amount[rng.choice(n, int(n * 0.35), replace=False)] = np.nan
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east"], n),
        "amount": amount,
        "units": rng.integers(1, 50, n),
    })
