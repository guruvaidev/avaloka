"""Seeded generator for the golden case describe/constant_column; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(22)
    n = 200
    return pd.DataFrame({
        "region": rng.choice(["north", "south"], n),
        "fee": np.full(n, 10.0),
        "units": rng.integers(1, 50, n),
    })
