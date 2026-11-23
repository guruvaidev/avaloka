"""Seeded generator for the golden case correlate/id_columns_excluded; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(64)
    n = 400
    spend = rng.normal(100, 15, n)
    return pd.DataFrame({
        "order_number": np.arange(100000, 100000 + n),
        "customer_id": rng.integers(1, 90, n),
        "product_code_version": rng.integers(1, 6, n),
        "status": "active",
        "spend": np.round(spend, 2),
        "revenue": np.round(2.0 * spend + rng.normal(0, 25, n), 2),
        "units": rng.integers(1, 40, n),
    })
