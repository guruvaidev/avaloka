"""Seeded generator for the golden case segment/small_groups; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(53)
    n = 300
    return pd.DataFrame({
        "channel": rng.choice(["web", "store", "partner", "phone", "kiosk"], n, p=[0.45, 0.3, 0.12, 0.09, 0.04]),
        "revenue": np.round(rng.lognormal(4.0, 0.5, n), 4),
    })
