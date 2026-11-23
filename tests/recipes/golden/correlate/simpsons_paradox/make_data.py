"""Seeded generator for the golden case correlate/simpsons_paradox; never a customer file."""

import numpy as np
import pandas as pd

def make() -> pd.DataFrame:
    rng = np.random.default_rng(62)
    # Inside each segment the pair slopes down; the segments' levels rise together, so the whole slopes up.
    frames = []
    for level, (x0, y0) in enumerate([(10.0, 20.0), (30.0, 60.0), (50.0, 100.0)]):
        n = 150
        x = x0 + rng.normal(0, 3, n)
        y = y0 - 1.5 * (x - x0) + rng.normal(0, 3, n)
        frames.append(pd.DataFrame({"segment": f"segment_{level}", "spend": np.round(x, 2), "revenue": np.round(y, 2)}))
    return pd.concat(frames, ignore_index=True)
