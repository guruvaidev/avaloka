"""Seeded generator for the golden case anomalies/too_many_flagged; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(82)
    n = 300
    # a fifth of the points come from a much wider spread: the robust rule flags well over a tenth
    wide = rng.random(n) < 0.2
    latency = 50 + np.where(wide, rng.normal(0, 60, n), rng.normal(0, 5, n))
    return pd.DataFrame({"latency": np.round(latency, 6)})
