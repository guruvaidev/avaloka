"""A classification target made of words must train, whatever dtype holds it.

`avaloka train healthcare_dataset.csv --target "Test Results"` -- the example
docs/INSTALL.md gives -- crashed with "Cannot cast array data from dtype('O')
to dtype('int64')". The labels were encoded only when the column's dtype was
`object`; by the time the frame reached training it held them as pandas'
string dtype, so the words went through unencoded and `np.bincount` failed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.modeling import train_candidates


def _frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    label = np.where(x > 0.4, "Abnormal", np.where(x < -0.4, "Normal", "Inconclusive"))
    return pd.DataFrame({"reading": x, "noise": rng.normal(size=n), "result": label})


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_a_text_label_target_trains_in_every_dtype_that_holds_text(dtype: str) -> None:
    frame = _frame()
    frame["result"] = frame["result"].astype(dtype)

    outcome = train_candidates(frame, "result", "multiclass_classification")

    assert outcome.candidates, "no candidate was trained"


def test_a_binary_text_target_trains() -> None:
    frame = _frame()
    frame["result"] = (frame["result"] == "Abnormal").map({True: "yes", False: "no"}).astype("string")

    outcome = train_candidates(frame, "result", "binary_classification")

    assert outcome.candidates
