"""Seeded generator for the golden case compare_periods/one_window_empty; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(44)
    # January and February, then nothing until a complete May: the period before May is empty.
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="D").append(pd.date_range("2024-05-01", "2024-05-31", freq="D"))
    n = len(dates)
    return pd.DataFrame({
        "region": rng.choice(["north", "south"], n),
        "date": dates.strftime("%Y-%m-%d"),
        "revenue": np.round(rng.lognormal(5.0, 0.6, n), 2),
    })
