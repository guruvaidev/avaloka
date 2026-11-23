"""Seeded generator for the golden case trend/mixed_dates; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(34)
    n = 120
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    # half the rows are written day-first, half month-first: nothing in the data settles which is which
    day_first = days.strftime("%d/%m/%Y")
    month_first = days.strftime("%m/%d/%Y")
    mixed = np.where(np.arange(n) % 2 == 0, day_first, month_first)
    return pd.DataFrame({
        "date": mixed,
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
    })
