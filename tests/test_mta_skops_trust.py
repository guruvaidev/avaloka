"""Tree models must survive MLflow's skops audit, with the audit left on.

mlflow.sklearn saves in skops format and loads the file back through
``skops.io.load(trusted=skops_trusted_types)`` before logging it. From skops
0.16 the audit flags ``TreePredictor`` (HistGradientBoosting) and
``sklearn.tree._tree.Tree`` (random forest), so in a freshly built image every
gradient-boosting and random-forest training failed with
``UntrustedTypesFoundException`` and the user saw "Training stopped because of
an unclassified error". CI had skops 0.15, which does not flag them.

These tests drive ``training_main`` itself -- the path the API ran -- into a
throwaway sqlite MLflow store, then load the model back the two ways the
product does and predict with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
skops_io = pytest.importorskip("skops.io")
pytest.importorskip("torch")

from app.agents.mta_v2 import estimators  # noqa: E402

FEATURES = ["age", "glucose", "bmi"]


def _frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.integers(20, 80, n)
    glucose = rng.normal(110, 25, n)
    bmi = rng.normal(28, 5, n)
    outcome = ((age > 55) & (glucose > 115)).astype(int)
    return pd.DataFrame({"age": age, "glucose": glucose, "bmi": bmi, "Outcome": outcome})


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_BACKEND_STORE_URI",
                "MLFLOW_DEFAULT_ARTIFACT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "skops-trust")
    monkeypatch.chdir(tmp_path)                 # default artifact root lands here
    yield tmp_path
    mlflow.set_tracking_uri(None)


@pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest", "linear"])
@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_everything_the_audit_flags_is_on_the_trusted_list(algorithm, model_type):
    """The list is explicit, not "whatever the model contains": a new family
    that needs another type must fail here, by name, before it fails a user."""
    frame = _frame()
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["Outcome"].to_numpy() if model_type == "classification" else frame["glucose"].to_numpy()
    model = estimators.build_estimator(algorithm, model_type, {"max_iter": 20, "n_estimators": 10})
    model.fit(X, y)
    flagged = set(skops_io.get_untrusted_types(data=skops_io.dumps(model)))
    assert flagged <= set(estimators.SKOPS_TRUSTED_TYPES), sorted(flagged)


@pytest.mark.parametrize("algorithm, needs", [
    ("gradient_boosting", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"),
    ("random_forest", "sklearn.tree._tree.Tree"),
])
def test_training_main_logs_a_tree_model_that_loads_and_predicts(mlflow_store, algorithm, needs):
    from app.agents.mta_v2.local_trainer import setup_mlflow, training_main

    # What LocalTrainer.train does before training_main: point MLflow at the
    # configured store. Without it the run goes wherever an earlier test left
    # MLflow's process-global tracking URI.
    setup_mlflow()
    data = mlflow_store / "outcome.csv"
    _frame().to_csv(data, index=False)

    result = training_main(
        data_uri=str(data),
        model_type="classification",
        target_col="Outcome",
        feature_cols=FEATURES,
        algorithm=algorithm,
        extra_hyperparameters={"max_iter": 20, "n_estimators": 10},
        build_model_card=False,
    )
    run_id = result["mlflow_run_id"]
    assert run_id

    # The audit ran, in skops format, with the list we chose -- not disabled,
    # not switched to pickle.
    client = mlflow.tracking.MlflowClient()
    mlmodel_dir = client.download_artifacts(run_id, "model", str(mlflow_store / "dl"))
    mlmodel = Path(mlmodel_dir, "MLmodel").read_text()
    import yaml
    flavor = yaml.safe_load(mlmodel)["flavors"]["sklearn"]
    assert flavor["serialization_format"] == "skops"
    assert needs in flavor["skops_trusted_types"]

    X = _frame(20)[FEATURES].to_numpy(dtype=float)
    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert sk_model.predict(X).shape == (20,)
    pyfunc = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert len(pyfunc.predict(X)) == 20
    json.dumps(result, default=str)             # what the API stores must serialise


# --------------------------------------------------------------------------- #
# The signature's dtype is the dtype the model was fitted on
# --------------------------------------------------------------------------- #
#
# MLflow enforces a tensor signature's dtype exactly. A tree model logged with
# a float32 signature (it is fitted on float64) refused an ordinary float64
# frame with "dtype of input float64 does not match expected dtype float32".

def _train_tree(mlflow_store, algorithm: str) -> str:
    from app.agents.mta_v2.local_trainer import setup_mlflow, training_main

    setup_mlflow()
    data = mlflow_store / "outcome.csv"
    _frame().to_csv(data, index=False)
    result = training_main(
        data_uri=str(data), model_type="classification", target_col="Outcome",
        feature_cols=FEATURES, algorithm=algorithm,
        extra_hyperparameters={"max_iter": 20, "n_estimators": 10}, build_model_card=False,
    )
    return result["mlflow_run_id"]


def _outcome(call):
    """The call's result, or the exception it raised -- asserted on, not raised."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        return exc


@pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest"])
def test_a_float64_frame_predicts_on_a_logged_tree_model(mlflow_store, algorithm):
    run_id = _train_tree(mlflow_store, algorithm)
    pyfunc = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    frame = _frame(20)[FEATURES].astype("float64")

    out = _outcome(lambda: pyfunc.predict(frame))
    assert not isinstance(out, Exception), f"a float64 frame was refused: {out}"
    assert len(out) == 20
    out = _outcome(lambda: pyfunc.predict(frame.to_numpy()))
    assert not isinstance(out, Exception), f"a float64 array was refused: {out}"
    assert len(out) == 20
    assert pyfunc.metadata.get_input_schema().numpy_types()[0] == np.dtype(estimators.SKLEARN_FEATURE_DTYPE)


@pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest"])
def test_the_model_server_predicts_with_a_logged_tree_model(mlflow_store, algorithm):
    """The 1.7 serving path: app/serve/model_server loads a registry/run model
    directory with no ONNX file through mlflow.pyfunc and casts the request."""
    from app.serve.model_server import load_model

    run_id = _train_tree(mlflow_store, algorithm)
    entry = load_model(run_id, f"runs:/{run_id}/model")
    assert entry.flavor == "pyfunc"

    out = _outcome(lambda: entry.model.predict({"age": 60, "glucose": 130.0, "bmi": 30.0}))
    assert not isinstance(out, Exception), f"the model server refused the request: {out}"
    assert out["prediction"] in (0, 1)


def test_the_model_server_still_sends_float32_to_a_torch_mlp(tmp_path):
    """The torch MLP's signature stays float32 (its trainer's tensors), and the
    server's cast follows the signature, so the torch path keeps working."""
    import torch
    import mlflow.pytorch

    from app.agents.mta_v2.model_signature import build_signature
    from app.serve.model_server import load_model

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 2))
    rows = pd.DataFrame(np.random.default_rng(3).normal(size=(10, 3)), columns=FEATURES)
    signature, example = build_signature(net, rows, FEATURES)
    assert signature is not None
    assert signature.inputs.numpy_types()[0] == np.dtype("float32")
    mlflow.pytorch.save_model(net, str(tmp_path / "mlp"), signature=signature,
                              input_example=example, serialization_format="pickle")

    entry = load_model("run-mlp", str(tmp_path / "mlp"))
    assert entry.flavor == "pyfunc" and entry.model.input_dtype == "float32"
    out = _outcome(lambda: entry.model.predict([1.0, 2.0, 3.0]))
    assert not isinstance(out, Exception), f"the model server refused the request: {out}"
    assert np.asarray(out["scores"]).shape == (1, 2)
