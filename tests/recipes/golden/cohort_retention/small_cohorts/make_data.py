"""Seeded generator for the golden case cohort_retention/small_cohorts; never a customer file.

Four cohorts, each smaller than the small_n floor; enough ids in all to run.
"""

import numpy as np
import pandas as pd

RETENTION = [1.0, 0.7, 0.55, 0.4, 0.3, 0.2, 0.15, 0.1]   # share of a cohort active k periods after its first
SIZES = [25, 20, 22, 18]   # cohorts 2024-01 .. 2024-04
FIRST_MONTH = "2024-01-01"
LAST_DAY = "2024-05-31"


def active_ids(size: int, k: int) -> int:
    """How many of a cohort's ids are active k periods on: the first ``size * RETENTION[k]`` of them."""
    return int(size * RETENTION[k] + 1e-9)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(63)
    last = pd.Timestamp(LAST_DAY)
    rows = []
    next_id = 1
    for m, size in enumerate(SIZES):
        start = pd.Timestamp(FIRST_MONTH) + pd.DateOffset(months=m)
        for j in range(size):
            uid = next_id + j
            for k in range(len(RETENTION)):
                period = start + pd.DateOffset(months=k)
                if period > last:
                    break
                active = j < active_ids(size, k)
                if not active:
                    continue
                span = min(period.days_in_month, (last - period).days + 1)
                for d in rng.integers(0, span, rng.integers(1, 4)):
                    day = (period + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                    rows.append((uid, day, round(float(rng.lognormal(3.0, 0.5)), 2)))
        next_id += size
    # One row on the last day pins the span (the id is active in that period anyway).
    rows.append((1, LAST_DAY, 10.0))
    # The same id may draw the same day twice; one row is enough.
    df = pd.DataFrame(rows, columns=["user_id", "date", "amount"])
    return df.drop_duplicates().sample(frac=1, random_state=7).reset_index(drop=True)
