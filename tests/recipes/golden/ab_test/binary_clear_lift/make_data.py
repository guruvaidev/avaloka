"""Seeded generator for the golden case ab_test/binary_clear_lift; never a customer file.

Two equal arms; the treatment arm converts at thirteen percent against ten, enough rows
that the interval on the difference excludes zero.
"""

import numpy as np
import pandas as pd

ARMS = {"A": (2000, 200), "B": (2000, 260)}   # arm -> (rows, rows with the outcome)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(71)
    frames = []
    for arm, (n, ones) in ARMS.items():
        outcome = np.zeros(n, dtype=int)
        outcome[:ones] = 1
        rng.shuffle(outcome)
        frames.append(pd.DataFrame({"variant": arm, "converted": outcome}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
