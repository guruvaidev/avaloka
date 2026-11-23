"""Seeded generator for the golden case ab_test/small_n; never a customer file.

Eighteen rows per arm: the recipe refuses (insufficient_rows).
"""

import numpy as np
import pandas as pd

ARMS = {"A": (18, 2), "B": (18, 3)}   # arm -> (rows, rows with the outcome)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(76)
    frames = []
    for arm, (n, ones) in ARMS.items():
        outcome = np.zeros(n, dtype=int)
        outcome[:ones] = 1
        rng.shuffle(outcome)
        frames.append(pd.DataFrame({"variant": arm, "converted": outcome}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
