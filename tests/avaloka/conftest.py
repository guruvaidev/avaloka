import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def churn_csv(tmp_path):
    """A small, well-behaved binary-classification dataset with signal."""
    rng = np.random.default_rng(7)
    n = 400
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 20, n).clip(15, 150).round(2)
    tickets = rng.poisson(1.5, n)
    contract = rng.choice(["m2m", "1yr", "2yr"], n, p=[0.5, 0.3, 0.2])
    logit = -0.05 * tenure + 0.02 * monthly + 0.4 * tickets + np.where(contract == "m2m", 1.0, 0.0) - 1.2
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "id": [f"C{i:04d}" for i in range(n)],
        "tenure": tenure, "monthly": monthly, "tickets": tickets,
        "contract": contract, "churned": churn,
    })
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def leaky_csv(tmp_path):
    """A dataset where one feature deterministically encodes the target."""
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x, "leak": y, "noise": rng.normal(size=n), "y": y})
    path = tmp_path / "leaky.csv"
    df.to_csv(path, index=False)
    return path
