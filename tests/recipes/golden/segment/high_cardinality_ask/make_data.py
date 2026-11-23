"""Seeded generator for the golden case segment/high_cardinality_ask; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(52)
    n = 1000
    return pd.DataFrame({
        "customer": [f"c{i:03d}" for i in rng.integers(0, 200, n)],
        "region": rng.choice(["north", "south", "east", "west"], n),
        "channel": rng.choice(["web", "store", "partner"], n),
        "revenue": np.round(rng.lognormal(4.0, 0.5, n), 4),
    })
