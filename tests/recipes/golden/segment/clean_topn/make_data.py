"""Seeded generator for the golden case segment/clean_topn; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(51)
    n = 2000
    lines = [f"line_{chr(97 + i)}" for i in range(15)]
    weights = np.array([30, 22, 15, 10, 7, 5, 3.5, 3, 2.6, 2.4, 0.9, 0.7, 0.6, 0.6, 0.5])
    return pd.DataFrame({
        "product_line": rng.choice(lines, n, p=weights / weights.sum()),
        "revenue": np.round(rng.lognormal(4.0, 0.5, n), 4),
        "units": rng.integers(1, 20, n),
    })
