"""Seeded generator for the golden case regress/small_n; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(74)
    n = 18
    ad_spend = rng.normal(50, 10, n)
    return pd.DataFrame({
        "ad_spend": np.round(ad_spend, 2),
        "sales": np.round(200 + 2.0 * ad_spend + rng.normal(0, 8, n), 2),
    })
