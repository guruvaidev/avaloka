"""Seeded generator for the golden case anomalies/clean_with_spikes; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(82)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    n = len(dates)
    level = 100 + 30 * np.sin(2 * np.pi * (np.asarray(dates.month) - 1) / 12)
    orders = level + rng.normal(0, 5, n)
    orders[45] += 80.0     # three planted spikes
    orders[150] -= 70.0
    orders[300] += 90.0
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "orders": np.round(orders, 2)})
