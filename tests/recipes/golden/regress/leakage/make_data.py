"""Seeded generator for the golden case regress/leakage; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(72)
    n = 300
    ad_spend = rng.normal(50, 10, n)
    sales = 200 + 2.0 * ad_spend + rng.normal(0, 8, n)
    return pd.DataFrame({
        "ad_spend": np.round(ad_spend, 2),
        "sales_target": np.round(sales * 1.0 + rng.normal(0, 0.01, n), 4),
        "sales": np.round(sales, 2),
    })
