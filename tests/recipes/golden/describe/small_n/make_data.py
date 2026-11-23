"""Seeded generator for the golden case describe/small_n; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    n = 18
    return pd.DataFrame({
        "region": rng.choice(["north", "south"], n),
        "score": np.round(rng.lognormal(3.0, 0.7, n), 2),
    })
