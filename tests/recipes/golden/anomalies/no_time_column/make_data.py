"""Seeded generator for the golden case anomalies/no_time_column; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(83)
    n = 200
    weight = 70 + rng.normal(0, 4, n)
    weight[20] += 35.0
    weight[140] -= 30.0
    return pd.DataFrame({"weight": np.round(weight, 4), "group": rng.choice(["a", "b"], n)})
