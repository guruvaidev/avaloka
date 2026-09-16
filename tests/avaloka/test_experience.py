"""Tests for workload routing, the Ray batch swarm, persona and serving."""

import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import patch

from avaloka import ray_batch, serving, workload
from avaloka.persona import Avaloka
from avaloka.swarm import clone_of
from avaloka.workload import Lane


# --- workload routing -----------------------------------------------------
def test_classify_lanes():
    assert workload.classify(1_000, 10_000).lane is Lane.ONLINE
    assert workload.classify(500_000, 20 * 1024 * 1024).lane is Lane.SAMPLED_ONLINE
    assert workload.classify(50_000_000, 5 * 1024**3).lane is Lane.BATCH


def test_online_lane_does_not_recommend_batch():
    plan = workload.classify(1_000, 10_000)
    assert plan.recommend_batch is False
    assert "live" in plan.spoken.lower()


def test_peek_counts_rows(tmp_path):
    df = pd.DataFrame({"a": range(1234)})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    n_rows, n_bytes, estimated = workload.peek(str(p))
    assert n_rows == 1234 and not estimated and n_bytes > 0


# --- Ray batch convergence (exactness) ------------------------------------
def test_batch_convergence_is_exact(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(10, 3, 5000), "g": rng.choice(["a", "b"], 5000)})
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)

    res = ray_batch.run_ray_local(str(p), n_partitions=7)
    assert res.converged["n_rows"] == 5000
    xcol = next(c for c in res.converged["columns"] if c["name"] == "x")
    # Mean from combined partial sums must match the exact pandas mean.
    assert xcol["mean"] == pytest.approx(df["x"].mean(), rel=1e-6)
    assert xcol["min"] == pytest.approx(df["x"].min())
    assert xcol["max"] == pytest.approx(df["x"].max())


def test_batch_artifacts_written(tmp_path):
    df = pd.DataFrame({"a": range(100)})
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)
    arts = ray_batch.write_batch_artifacts(tmp_path / "out", str(src),
                                           image="img:1", n_partitions=4)
    assert arts["driver"].exists() and arts["manifest"].exists()
    text = arts["manifest"].read_text()
    assert "RayJob" in text and "workerGroupSpecs" in text


# --- persona --------------------------------------------------------------
def test_persona_observations_are_grounded():
    ava = Avaloka(llm=False)
    profile = {
        "dataset": {"n_rows": 100, "n_cols": 2},
        "quality": {"score": 80, "high_missing_columns": [], "id_columns": ["uid"]},
        "columns": [{"name": "uid", "role": "id"},
                    {"name": "cat", "role": "categorical",
                     "top_values": [{"value": "x", "count": 60}]}],
        "correlations": [],
    }
    lines = ava.observe(profile)
    assert any("cat" in line for line in lines)
    assert any("uid" in line for line in lines)


def test_swarm_clone_names():
    assert clone_of("data_scout")[0] == "Ava-Scout"
    assert clone_of("model_scientist")[0] == "Ava-Scientist"


# --- serving --------------------------------------------------------------
def test_serving_requires_deployable_package(tmp_path):
    (tmp_path / "model").mkdir()
    with pytest.raises(ValueError):
        serving.prepare(str(tmp_path))


def test_serving_prepare_plan(tmp_path):
    # Minimal deployable layout.
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.pkl").write_bytes(b"x")
    (tmp_path / "service.py").write_text("app = 1")
    (tmp_path / "Dockerfile").write_text("FROM python")
    (tmp_path / "mlflow").mkdir()
    (tmp_path / "mlflow" / "register.py").write_text("print('registered')")
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "kubernetes.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: demo\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - image: foo:1\n"
        "---\napiVersion: v1\nkind: Service\nmetadata:\n  name: demo\n")
    plan = serving.prepare(str(tmp_path), mlflow_uri="http://mlflow:5000")
    assert len(plan.steps) == 4
    assert any("docker build" in c for c in plan.commands)
    assert any("kubectl apply" in c for c in plan.commands)
    assert any("register.py" in c for c in plan.commands)  # mlflow stage present


def test_registry_image_is_written_into_applied_manifest(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.pkl").write_bytes(b"x")
    (tmp_path / "service.py").write_text("app = 1")
    (tmp_path / "Dockerfile").write_text("FROM python")
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "kubernetes.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: demo\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - name: inference\n        image: demo:latest\n"
        "---\napiVersion: v1\nkind: Service\nmetadata:\n  name: demo\n"
    )

    plan = serving.prepare(str(tmp_path), registry="gcr.io/acme")
    rendered = serving._manifest_for_image(plan)
    deployment = next(d for d in yaml.safe_load_all(rendered) if d["kind"] == "Deployment")
    assert deployment["spec"]["template"]["spec"]["containers"][0]["image"] == (
        "gcr.io/acme/demo:latest"
    )


def test_apply_registers_mlflow_and_applies_rendered_registry_image(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.pkl").write_bytes(b"x")
    (tmp_path / "service.py").write_text("app = 1")
    (tmp_path / "Dockerfile").write_text("FROM python")
    (tmp_path / "mlflow").mkdir()
    (tmp_path / "mlflow" / "register.py").write_text("print('registered')")
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "kubernetes.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: demo\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - name: inference\n        image: demo:latest\n"
        "---\napiVersion: v1\nkind: Service\nmetadata:\n  name: demo\n"
    )
    plan = serving.prepare(
        str(tmp_path),
        registry="gcr.io/acme",
        mlflow_uri="http://mlflow:5000",
    )
    calls = []

    def fake_run(cmd, timeout=1800, **kwargs):
        calls.append((cmd, kwargs))
        return True, "ok"

    with patch.object(serving, "_have", return_value=True), \
            patch.object(serving, "_run", side_effect=fake_run), \
            patch.object(serving, "_resolve_endpoint", return_value="http://example.test"):
        serving.apply(plan, push=True)

    mlflow_call = calls[0]
    assert mlflow_call[0][-1] == "mlflow/register.py"
    assert mlflow_call[1]["cwd"] == tmp_path
    assert mlflow_call[1]["env"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    kubectl_apply = next(call for call in calls if call[0][:3] == ["kubectl", "apply", "-f"])
    assert "gcr.io/acme/demo:latest" in kubectl_apply[1]["input_text"]
