"""Seeded generator for the golden case regress/collinear; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(73)
    n = 300
    ad_spend = rng.normal(50, 10, n)
    impressions = ad_spend * 20 + rng.normal(0, 15, n)   # nearly the same information as ad_spend
    discount = rng.normal(10, 3, n)
    return pd.DataFrame({
        "ad_spend": np.round(ad_spend, 2),
        "impressions": np.round(impressions, 1),
        "discount": np.round(discount, 2),
        "sales": np.round(200 + 2.0 * ad_spend - 1.5 * discount + rng.normal(0, 8, n), 2),
    })
