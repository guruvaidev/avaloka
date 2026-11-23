"""Seeded generator for the golden case correlate/clean_pairs; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(61)
    n = 400
    spend = rng.normal(100, 15, n)
    return pd.DataFrame({
        "region": rng.choice(["north", "south", "east"], n),
        "spend": np.round(spend, 2),
        "revenue": np.round(2.0 * spend + rng.normal(0, 25, n), 2),
        "returns": np.round(-0.3 * spend + rng.normal(0, 12, n) + 60, 2),
    })
