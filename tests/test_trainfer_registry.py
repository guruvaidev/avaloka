"""MLflow model registry: naming, idempotency, aliases, signatures.

Base classes (develop-1.6 @ 9cd9e122): the signature test fails by assertion
(log_model is called without a signature); the registry tests error because
app.core.ml_registry does not exist.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:
    from app.core import ml_store
except ImportError:  # pragma: no cover - base tree only
    ml_store = None


class AlreadyExists(Exception):
    error_code = "RESOURCE_ALREADY_EXISTS"


class FakeMlflowClient:
    """Just enough of MlflowClient's registry surface to observe behaviour."""

    def __init__(self):
        self.models = {}       # name -> {"tags": {...}, "versions": {v: ns}, "aliases": {}}
        self.calls = []

    def create_registered_model(self, name, tags=None, description=None):
        self.calls.append(("create_registered_model", name))
        if name in self.models:
            raise AlreadyExists(f"RESOURCE_ALREADY_EXISTS: {name}")
        self.models[name] = {"tags": dict(tags or {}), "versions": {}, "aliases": {}}
        return SimpleNamespace(name=name)

    def get_registered_model(self, name):
        return SimpleNamespace(name=name, tags=self.models[name]["tags"])

    def create_model_version(self, name, source, run_id, tags=None, description=None):
        self.calls.append(("create_model_version", name, source, run_id))
        versions = self.models[name]["versions"]
        v = len(versions) + 1
        versions[v] = SimpleNamespace(name=name, version=str(v), run_id=run_id, source=source, tags=dict(tags or {}))
        return versions[v]

    def search_model_versions(self, filter_string):
        self.calls.append(("search_model_versions", filter_string))
        out = []
        for name, m in self.models.items():
            for mv in m["versions"].values():
                if filter_string == f"run_id = '{mv.run_id}'" or filter_string == f"name = '{name}'":
                    out.append(mv)
        return out

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append(("set_alias", name, alias, version))
        self.models[name]["aliases"][alias] = str(version)

    def get_model_version_by_alias(self, name, alias):
        v = self.models[name]["aliases"].get(alias)
        if v is None:
            raise Exception("RESOURCE_DOES_NOT_EXIST: no alias")
        return self.models[name]["versions"][int(v)]

    def delete_registered_model_alias(self, name, alias):
        self.calls.append(("delete_alias", name, alias))
        self.models[name]["aliases"].pop(alias, None)

    def delete_model_version(self, name, version):
        self.calls.append(("delete_model_version", name, version))
        self.models[name]["versions"].pop(int(version), None)

    def delete_registered_model(self, name):
        self.calls.append(("delete_registered_model", name))
        self.models.pop(name, None)


@pytest.fixture
def store():
    if ml_store is None:
        yield None
        return
    s = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


RESULT = {"model_name": "Churn Model v2!", "model_type": "classification", "final_accuracy": 0.91,
          "final_f1_score": 0.9, "num_epochs_trained": 7, "session_id": "s-1", "execution_backend": "local"}


# T-09 ------------------------------------------------------------------------

def test_register_run_twice_creates_one_version_with_the_scheme_and_tags(store):
    from app.core import ml_registry as reg
    client = FakeMlflowClient()
    first = reg.register_run("user-1", "run-1", RESULT, client=client, store=store)
    second = reg.register_run("user-1", "run-1", RESULT, client=client, store=store)

    assert first["registered_name"] == "avaloka.user-1.churn-model-v2"
    assert (first["version"], first["created"]) == (1, True)
    assert (second["version"], second["created"]) == (1, False)
    assert sum(1 for c in client.calls if c[0] == "create_model_version") == 1
    mv = client.models["avaloka.user-1.churn-model-v2"]["versions"][1]
    assert mv.source == "runs:/run-1/model"
    assert mv.tags["avaloka.run_id"] == "run-1"
    assert mv.tags["avaloka.user_id"] == "user-1"
    assert mv.tags["avaloka.task_type"] == "classification"
    assert mv.tags["avaloka.requires_gpu"] == "false"
    # no alias until an endpoint is deployed
    assert client.models["avaloka.user-1.churn-model-v2"]["aliases"] == {}
    rec = store.get_model("run-1")
    assert (rec.registered_name, rec.version, rec.alias, rec.task_type) == ("avaloka.user-1.churn-model-v2", 1, None, "classification")
    assert rec.metrics == {"final_accuracy": 0.91, "final_f1_score": 0.9, "num_epochs_trained": 7}


def test_second_run_of_the_same_model_name_is_version_two(store):
    from app.core import ml_registry as reg
    client = FakeMlflowClient()
    reg.register_run("user-1", "run-1", RESULT, client=client, store=store)
    out = reg.register_run("user-1", "run-2", RESULT, client=client, store=store)
    assert (out["registered_name"], out["version"]) == ("avaloka.user-1.churn-model-v2", 2)


def test_slug_collision_appends_the_run_prefix(store):
    from app.core import ml_registry as reg
    client = FakeMlflowClient()
    reg.register_run("user-1", "run-1", {**RESULT, "model_name": "sales/q1"}, client=client, store=store)
    out = reg.register_run("user-1", "run-2abcdef99", {**RESULT, "model_name": "sales q1"}, client=client, store=store)
    assert reg.slug("sales/q1") == reg.slug("sales q1") == "sales-q1"
    assert out["registered_name"] == "avaloka.user-1.sales-q1-run-2abc"


def test_slug_rules():
    from app.core.ml_registry import slug
    assert slug("  Hello World ") == "hello-world"
    assert slug("x" * 80) == "x" * 40
    assert slug("!!!") == "model"
    assert slug("") == "model"


# T-10 ------------------------------------------------------------------------

def test_resource_already_exists_is_tolerated_other_errors_raise(store):
    from app.core import ml_registry as reg

    class Flaky(FakeMlflowClient):
        def create_registered_model(self, name, tags=None, description=None):
            if name.endswith("boom"):
                raise RuntimeError("INTERNAL_ERROR: registry 500")
            self.models.setdefault(name, {"tags": dict(tags or {}), "versions": {}, "aliases": {}})
            raise AlreadyExists("RESOURCE_ALREADY_EXISTS")

    client = Flaky()
    out = reg.register_run("user-1", "run-1", {**RESULT, "model_name": "ok"}, client=client, store=store)
    assert out["version"] == 1
    with pytest.raises(RuntimeError):
        reg.register_run("user-1", "run-2", {**RESULT, "model_name": "boom"}, client=client, store=store)
    assert store.get_model("run-2") is None, "a failed registration leaves no Postgres row"


# aliases ----------------------------------------------------------------------

def test_serving_alias_moves_between_versions_and_clears(store):
    from app.core import ml_registry as reg
    client = FakeMlflowClient()
    reg.register_run("user-1", "run-1", RESULT, client=client, store=store)
    reg.register_run("user-1", "run-2", RESULT, client=client, store=store)
    name = "avaloka.user-1.churn-model-v2"

    reg.set_serving_alias("run-1", client=client, store=store)
    assert client.models[name]["aliases"] == {"serving": "1"}
    reg.set_serving_alias("run-2", client=client, store=store)
    assert client.models[name]["aliases"] == {"serving": "2"}
    assert store.get_model("run-1").alias is None and store.get_model("run-2").alias == "serving"

    reg.clear_serving_alias("run-1", client=client, store=store)   # not the holder: no-op
    assert client.models[name]["aliases"] == {"serving": "2"}
    reg.clear_serving_alias("run-2", client=client, store=store)
    assert client.models[name]["aliases"] == {}
    assert store.get_model("run-2").alias is None


def test_delete_run_version_removes_the_registered_model_when_last(store):
    from app.core import ml_registry as reg
    client = FakeMlflowClient()
    reg.register_run("user-1", "run-1", RESULT, client=client, store=store)
    reg.register_run("user-1", "run-2", RESULT, client=client, store=store)
    reg.delete_run_version("run-1", client=client, store=store)
    assert "avaloka.user-1.churn-model-v2" in client.models
    reg.delete_run_version("run-2", client=client, store=store)
    assert "avaloka.user-1.churn-model-v2" not in client.models
    assert store.get_model("run-1") is None and store.get_model("run-2") is None


# T-11 the trainers log a signature ---------------------------------------------

def test_local_trainer_logs_a_signature_and_a_five_row_example(tmp_path, monkeypatch):
    from app.agents.mta_v2 import local_trainer

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3))
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["target"] = (X[:, 0] > 0).astype(int)
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-trainfer-signature")

    captured = {}
    import mlflow.pytorch as mp
    real_log_model = mp.log_model

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_log_model(*args, **kwargs)

    with patch.object(mp, "log_model", spy):
        out = local_trainer.training_main(
            task_id="sig-test", user_id="u", session_id="s", model_type="classification",
            data_uri=str(csv), target_col="target", feature_cols=["f1", "f2", "f3"],
            model_name="sig", epochs=1, batch_size=16, hidden_sizes=[4], class_names=["0", "1"],
        )
    assert captured.get("signature") is not None, "log_model was called without a signature"
    assert captured["signature"].inputs is not None and captured["signature"].outputs is not None
    assert np.asarray(captured.get("input_example")).shape == (5, 3)
    assert out.get("signature_logged") is True


def test_signature_helper_never_raises():
    from app.agents.mta_v2.model_signature import build_signature
    assert build_signature(object(), [{"a": "not-a-number"}], ["a"]) == (None, None)
    assert build_signature(object(), [], ["a"]) == (None, None)


def test_ray_job_mirrors_the_signature_logging():
    """Static check (ray_job.py needs a Ray Dataset to run): the training image's
    log_model call carries a signature and input example like LocalTrainer's."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "app", "agents", "mta_v2",
                            "training_docker_image", "ray_job.py")).read()
    call = re.search(r"mlflow\.pytorch\.log_model\((.*?)\n\s*\)\n", src, re.S).group(1)
    assert "signature=signature" in call and "input_example=input_example" in call
    assert "infer_signature" in src
