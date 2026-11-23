"""Seeded generator for the golden case ab_test/srm; never a customer file.

A seventy/thirty split under an equal-allocation design: the sample-ratio check refuses.
The counts put the chi-square p-value between the recipe's threshold and a loosened one.
"""

import numpy as np
import pandas as pd

ARMS = {"A": (112, 11), "B": (48, 5)}   # arm -> (rows, rows with the outcome)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(73)
    frames = []
    for arm, (n, ones) in ARMS.items():
        outcome = np.zeros(n, dtype=int)
        outcome[:ones] = 1
        rng.shuffle(outcome)
        frames.append(pd.DataFrame({"variant": arm, "converted": outcome}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
