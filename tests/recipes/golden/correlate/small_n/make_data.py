"""Seeded generator for the golden case correlate/small_n; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(63)
    n = 18
    spend = rng.normal(100, 15, n)
    return pd.DataFrame({
        "spend": np.round(spend, 2),
        "revenue": np.round(2.0 * spend + rng.normal(0, 25, n), 2),
    })
