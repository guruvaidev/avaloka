"""Seeded generator for the golden case ab_test/underpowered; never a customer file.

Sixty rows per arm and a small true lift: the interval on the difference includes zero
and the power at the observed difference is far below the target.
"""

import numpy as np
import pandas as pd

ARMS = {"A": (60, 6), "B": (60, 8)}   # arm -> (rows, rows with the outcome)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(74)
    frames = []
    for arm, (n, ones) in ARMS.items():
        outcome = np.zeros(n, dtype=int)
        outcome[:ones] = 1
        rng.shuffle(outcome)
        frames.append(pd.DataFrame({"variant": arm, "converted": outcome}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
