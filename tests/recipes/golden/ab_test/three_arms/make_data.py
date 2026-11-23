"""Seeded generator for the golden case ab_test/three_arms; never a customer file.

A control and two treatment arms, each compared with the control; the headline is the
largest difference.
"""

import numpy as np
import pandas as pd

ARMS = {"control": (2000, 200), "variant_b": (2000, 260), "variant_c": (2000, 280)}   # arm -> (rows, rows with the outcome)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(75)
    frames = []
    for arm, (n, ones) in ARMS.items():
        outcome = np.zeros(n, dtype=int)
        outcome[:ones] = 1
        rng.shuffle(outcome)
        frames.append(pd.DataFrame({"arm": arm, "converted": outcome}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
