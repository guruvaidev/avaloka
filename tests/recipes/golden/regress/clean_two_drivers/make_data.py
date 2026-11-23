"""Seeded generator for the golden case regress/clean_two_drivers; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(71)
    n = 300
    ad_spend = rng.normal(50, 10, n)
    discount = rng.normal(10, 3, n)
    return pd.DataFrame({
        "ad_spend": np.round(ad_spend, 2),
        "discount": np.round(discount, 2),
        "sales": np.round(200 + 2.0 * ad_spend - 1.5 * discount + rng.normal(0, 8, n), 2),
    })
