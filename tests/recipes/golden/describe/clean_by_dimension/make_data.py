"""Seeded generator for the golden case describe/clean_by_dimension; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 600
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "revenue": np.round(rng.lognormal(5.0, 0.5, n), 2),
        "units": rng.integers(1, 50, n),
    })
