"""detect_target_leakage must catch noisy categorical leaks, not only exact ones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from avaloka.stats import detect_target_leakage, profile_column


def _columns(frame: pd.DataFrame) -> list[dict]:
    return [profile_column(frame[c], len(frame)) for c in frame.columns]


def _churn_frame(n: int = 5000, noise: float = 0.006, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churned = (rng.random(n) < 0.32).astype(int)
    status = np.where(churned == 1, "closed", "active")
    flip = rng.random(n) < noise
    status[flip] = np.where(status[flip] == "closed", "active", "closed")
    return pd.DataFrame({
        "churned": churned,
        "account_status": status,
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n),
    })


def test_noisy_categorical_leak_is_flagged():
    frame = _churn_frame()
    found = {f["column"] for f in detect_target_leakage(frame, "churned", _columns(frame))}
    assert "account_status" in found


def test_ordinary_categorical_feature_is_not_flagged():
    frame = _churn_frame()
    found = {f["column"] for f in detect_target_leakage(frame, "churned", _columns(frame))}
    assert "contract_type" not in found


def test_imbalanced_target_does_not_flag_ordinary_feature():
    rng = np.random.default_rng(1)
    n = 5000
    frame = pd.DataFrame({
        "fraud": (rng.random(n) < 0.03).astype(int),
        "channel": rng.choice(["web", "app", "store"], n),
    })
    assert detect_target_leakage(frame, "fraud", _columns(frame)) == []


def test_exact_categorical_copy_is_still_flagged():
    rng = np.random.default_rng(2)
    label = rng.choice(["yes", "no"], 200)
    frame = pd.DataFrame({
        "label": label,
        "label_text": np.where(label == "yes", "Y", "N"),
        "region": rng.choice(["a", "b", "c", "d"], 200),
    })
    found = {f["column"] for f in detect_target_leakage(frame, "label", _columns(frame))}
    assert found == {"label_text"}


def _rare_label_frame(rate: float, noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 5000
    fraud = (rng.random(n) < rate).astype(int)
    status = np.where(fraud == 1, "chargeback", "settled")
    flip = rng.random(n) < noise
    status[flip] = np.where(status[flip] == "chargeback", "settled", "chargeback")
    # an honest feature that shifts the rate but never decides the label
    channel = np.where(rng.random(n) < np.where(fraud == 1, 0.6, 0.3), "app", "web")
    return pd.DataFrame({"fraud": fraud, "case_status": status, "channel": channel})


def test_exact_copy_of_a_rare_label_is_still_flagged():
    # 97/3 target: a perfect copy is only 3 points above the majority baseline
    for rate in (0.08, 0.03):
        frame = _rare_label_frame(rate, noise=0.0, seed=4)
        found = {f["column"]: f["kind"] for f in detect_target_leakage(frame, "fraud", _columns(frame))}
        assert found == {"case_status": "deterministic_mapping"}, (rate, found)


def test_noisy_copy_of_a_rare_label_is_flagged():
    frame = _rare_label_frame(0.08, noise=0.004, seed=5)
    found = {f["column"] for f in detect_target_leakage(frame, "fraud", _columns(frame))}
    assert found == {"case_status"}
