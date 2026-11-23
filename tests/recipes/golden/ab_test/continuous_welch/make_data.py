"""Seeded generator for the golden case ab_test/continuous_welch; never a customer file.

Two arms with somewhat unequal rows (the counts still fit an equal split) and
very unequal spread, so Welch's t-test and the pooled t-test disagree at the
reported decimals.
"""

import numpy as np
import pandas as pd

ARMS = {"control": (290, 50.0, 5.0), "treatment": (260, 54.0, 15.0)}   # arm -> (rows, mean, sd)


def make() -> pd.DataFrame:
    rng = np.random.default_rng(72)
    frames = []
    for arm, (n, mean, sd) in ARMS.items():
        frames.append(pd.DataFrame({"arm": arm, "minutes": np.round(rng.normal(mean, sd, n), 3)}))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df.sample(frac=1, random_state=7).reset_index(drop=True)
