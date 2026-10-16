import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def no_llm_key(monkeypatch):
    """Safety rail (docs/TEST_PLAN-CLI.md §2): this suite never spends a key.

    ``--llm`` is exercised for its *degradation* contract -- the run must
    complete and produce an identical bundle when no key is configured. A
    developer's exported ANTHROPIC_API_KEY would turn that into a live,
    billable call and make the assertions depend on network and model output.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def write_churn_csv(path, n: int = 400, seed: int = 7):
    """A small, well-behaved binary-classification dataset with signal.

    Shared by the function-scoped ``churn_csv`` fixture and the session-scoped
    ``trained_package`` fixture so both see the same data.
    """
    rng = np.random.default_rng(seed)
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
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def churn_csv(tmp_path):
    """A small, well-behaved binary-classification dataset with signal."""
    return write_churn_csv(tmp_path / "churn.csv")


@pytest.fixture(scope="session")
def trained_package(tmp_path_factory):
    """One deployable train bundle for every deploy/infer test in the session.

    Training takes several seconds; the deploy and infer commands only read
    the bundle, so building it once is the difference between a suite that
    runs in under a minute and one that does not.
    """
    from avaloka.mission.budget import Budget
    from avaloka.mission.context import MissionContext, MissionKind
    from avaloka.missions import run_train

    base = tmp_path_factory.mktemp("trained")
    data = write_churn_csv(base / "churn.csv")
    out = base / "model"
    ctx = MissionContext(kind=MissionKind.TRAIN, goal="Predict churn", data_source=str(data),
                         output_dir=out, budget=Budget(), target="churned", metric="roc_auc",
                         deployable=True)
    run_train(ctx)
    return out


@pytest.fixture
def synthetic_package(tmp_path):
    """A minimal deployable package with a *well-formed* Kubernetes manifest.

    The serving tests in S8 need a manifest the serving layer can parse so the
    build → apply → rollout sequence is observable. The package Avaloka
    generates does not currently qualify (finding F8: kubernetes.yaml is one
    YAML list, not two documents), so until that is fixed the sequence is
    exercised on this hand-written layout and F8 is pinned separately.
    """
    import yaml

    pkg = tmp_path / "pkg"
    for rel in ("model/model.pkl", "service.py", "Dockerfile"):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("x")
    (pkg / "deployment").mkdir()
    (pkg / "deployment" / "kubernetes.yaml").write_text(yaml.safe_dump_all([
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "churn-api"},
         "spec": {"template": {"spec": {"containers": [
             {"name": "inference", "image": "avaloka-inference:abc123"}]}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "churn-svc"}},
    ], sort_keys=False))
    (pkg / "mlflow").mkdir()
    (pkg / "mlflow" / "register.py").write_text("print('registered')")
    (pkg / "tests").mkdir()
    (pkg / "tests" / "example_request.json").write_text('{"instances": [{"tenure": 3}]}\n')
    return pkg


@pytest.fixture
def titanic_like_csv(tmp_path):
    """Text columns with missing values, as every real export has.

    Mirrors the shape of the Kaggle Titanic ``train.csv`` that crashed
    ``avaloka train`` (finding F1 in docs/TEST_PLAN-CLI.md): a numeric column
    with NaN, and two low-cardinality text columns with nulls, plus a target
    with real signal so the run has every reason to succeed once fixed.
    """
    rng = np.random.default_rng(3)
    n = 500
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.2, 0.55])
    sex = rng.choice(["male", "female"], n)
    age = rng.normal(30, 13, n).clip(1, 80).round(1)
    age[rng.random(n) < 0.2] = np.nan
    fare = rng.gamma(2.0, 20.0, n).round(2)
    cabin = rng.choice(["A", "B", "C", "D", "E", "F", "G"], n).astype(object)
    cabin[rng.random(n) < 0.75] = None
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]).astype(object)
    embarked[rng.random(n) < 0.01] = None
    logit = 1.2 * (sex == "female") - 0.8 * (pclass - 1) + 0.01 * fare - 0.5
    survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "passenger_id": range(1, n + 1), "survived": survived, "pclass": pclass,
        "sex": sex, "age": age, "fare": fare, "cabin": cabin, "embarked": embarked,
    })
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def uncorrelated_csv(tmp_path):
    """Independent columns, no quality issues, no target.

    Built so the Data Scout finds nothing above the reporter's correlation
    threshold and the quality report raises no issue: 500 rows (not a small
    sample), floats (never identifier-like), no missing values, no constant
    column, no duplicate rows, and one three-value categorical (never
    high-cardinality). What the reporter says about *this* dataset is therefore
    exactly its fallback behaviour (finding F5).
    """
    rng = np.random.default_rng(11)
    n = 500
    df = pd.DataFrame({
        "alpha": rng.normal(size=n), "beta": rng.normal(size=n),
        "gamma": rng.normal(size=n), "group": rng.choice(["p", "q", "r"], n),
    })
    path = tmp_path / "noise.csv"
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
