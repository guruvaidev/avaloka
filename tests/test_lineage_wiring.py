"""The lineage hooks are actually called from the places that produce data.

tests/test_lineage.py proves the hooks and the store are right. Nothing there
notices if a call site stops calling them: delete the hook from an executor and
every test in that file still passes. Each test here drives one real node with
its outside world faked, and then asks the STORE what was recorded.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from app.core import lineage
from app.core.lineage import EdgeKind, LineageStore, NodeKind

USER = "wiring-user"


@pytest.fixture(autouse=True)
def lineage_db(tmp_path, monkeypatch):
    lineage.reset_for_tests()
    monkeypatch.setenv(lineage.ENV_DB_URL, "sqlite:///" + (tmp_path / "lineage.db").as_posix())
    monkeypatch.delenv(lineage.ENV_ENABLED, raising=False)
    yield
    lineage.reset_for_tests()


def _assert_output_recorded(new_state, *, parent, columns):
    output_id = new_state.get("latest_output_dataset_id")
    assert output_id, "the executor did not record its output"
    store = LineageStore(USER)
    node = store.get_node(output_id)
    assert node is not None and node.kind is NodeKind.DATASET
    assert store.ancestors(output_id) == [parent]
    assert store.stats()["node_column"] >= len(columns)
    for name in columns:
        assert store.resolve_column(store.column_id(output_id, name)) == name
    return node


def test_the_remote_executor_records_its_output(monkeypatch):
    import app.agents.execution_agent as exec_mod

    csv_text = pd.DataFrame([{"region": "emea", "revenue": 2}]).to_csv(index=False)
    payload = "data:text/csv;base64," + base64.b64encode(csv_text.encode()).decode()
    monkeypatch.setattr(exec_mod, "execute_code_on_k8s", lambda **kwargs: {
        "status": "success", "output": payload,
        "output_file": {"filename": "by_region.csv", "content": payload, "size": len(csv_text)},
    })
    state = {
        "user_id": USER,
        "messages": [],
        "coder_definition": {"code": "print('x')"},
        "active_dataset_id": "ds:sales",
        "data_source_location": "sales.csv",
        "output_location": "by_region.csv",
        "infrastructure_provisioned": {"status": "provisioned", "details": [{
            "step_name": "Get infra-agent-service IP/Port", "status": "SUCCESS",
            "details": {"ip": "1.2.3.4", "port": "8080"}}]},
    }

    new_state = exec_mod.execution_agent_node(state)

    node = _assert_output_recorded(new_state, parent="ds:sales", columns=["region", "revenue"])
    assert node.label == "by_region.csv"


def test_the_local_executor_records_its_output(monkeypatch, tmp_path):
    import app.agents.execution_agent as exec_mod

    def fake_run(**kwargs):
        pd.DataFrame([{"region": "emea", "revenue": 2}]).to_csv(
            kwargs["output_location"], index=False)
        return {"status": "success"}

    monkeypatch.setattr(exec_mod, "execute_code_on_local", fake_run)
    state = {
        "user_id": USER,
        "messages": [HumanMessage(content="revenue by region")],
        "coder_definition": {"code": "print('x')"},
        "active_dataset_id": "ds:sales",
        "data_source_location": None,
        "output_location": str(tmp_path / "by_region.csv"),
    }

    new_state = exec_mod.execution_agent_node_local(state)

    assert new_state["execution_result"]["status"] == "success", new_state["execution_result"]
    _assert_output_recorded(new_state, parent="ds:sales", columns=["region", "revenue"])


def test_the_ray_executor_records_its_output(monkeypatch, tmp_path):
    import app.agents.execution_agent as exec_mod

    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "output_rows": [["emea", 2]], "output_columns": ["region", "revenue"],
        "output_row_count": 1}))
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")   # no kubectl secret
    monkeypatch.setattr(exec_mod, "get_cloud_connection", lambda connection_id: None)
    monkeypatch.setattr(exec_mod, "_run_coro_sync", lambda coro: {"provider": "s3"})
    monkeypatch.setattr(exec_mod, "render_rayjob_yaml", lambda **kwargs: "kind: RayJob\n")
    monkeypatch.setattr(exec_mod, "run_rayjob_from_yaml", lambda **kwargs: SimpleNamespace(
        status="SUCCEEDED", runtime_s=1.0, logs=f"ARTIFACT_URI={artifact}\n"))
    state = {
        "user_id": USER,
        "messages": [],
        "coder_definition": {"code": "import os\nprint('hello')"},
        "active_dataset_id": "ds:sales",
        "dataset_id": "ds:sales",
        "connection_id": "conn-1",
        "data_source_location_cloud": "s3://bucket/sales.csv",
    }

    new_state = exec_mod.execution_agent_node_ray(state)

    assert new_state["execution_result"]["status"] == "succeeded", new_state["execution_result"]
    _assert_output_recorded(new_state, parent="ds:sales", columns=["region", "revenue"])
    edges = LineageStore(USER).stats()
    assert edges["edge_loaded_from"] == 1, "the source's connection is part of its provenance"


def test_model_training_records_the_model(tmp_path):
    from app.agents.mta_v2.agent import ModelTrainingAgent

    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"f1": [0.1, 0.2], "f2": [1.0, 2.0], "target": [0, 1]}).to_csv(csv_path, index=False)
    plan = {
        "model_architecture": {"model_type": "mlp", "task_type": "classification",
                               "input_size": 2, "output_size": 2, "hidden_layers": [8],
                               "activation": "relu", "dropout_rate": 0.1, "batch_norm": False},
        "ray_config": None,
        "data_config": {"dataset_uri": str(csv_path), "feature_columns": ["f1", "f2"],
                        "target_column": "target"},
    }
    trainer = MagicMock()
    trainer.train.return_value = {"mlflow_run_id": "run-wired", "model_name": "churn",
                                  "final_accuracy": 0.9}
    state = {
        "user_id": USER, "session_id": "s", "messages": [HumanMessage(content="train")],
        "training_plan": plan, "training_result": None, "training_scheduled": False,
        "enable_training": True, "skip_to_training": False, "training_task": None,
        "data_source_location_cloud": None, "data_source_location_local": None,
        "data_source_location": None, "dataset_size_bytes": None,
        "active_dataset_id": "ds:train",
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Training completed.", tool_calls=[])

    with patch("app.agents.mta_v2.agent.llm", llm), \
         patch("app.agents.mta_v2.agent.LocalTrainer", return_value=trainer):
        result = ModelTrainingAgent()._execute_training(state)

    assert result.get("lineage_model_id") == "model:run-wired"
    store = LineageStore(USER)
    assert store.get_node("model:run-wired").kind is NodeKind.MODEL
    stats = store.stats()
    assert stats["edge_trained_on"] == 1
    assert stats["edge_uses_feature"] == 2
    assert store.impact_of_column_change("ds:train", "f1")["models_affected"] == ["model:run-wired"]


def test_the_preparation_pipeline_records_what_its_pii_scan_found():
    from app.agents.preparation_pipeline import preparation_pipeline_node

    frame = pd.DataFrame({
        "email": [f"person{i}@example.com" for i in range(40)],
        "spend": [float(i) for i in range(40)],
    })
    out = preparation_pipeline_node({
        "user_id": USER, "active_dataset_id": "ds:customers", "dataframe": frame,
        "dataset_name": "customers.csv", "objective": "analysis"})

    found = {f["column"] for f in out["pii_report"].get("findings", [])
             if f.get("kind") not in (None, "not_pii")}
    assert "email" in found, "this test needs the scan to flag the email column"
    store = LineageStore(USER)
    assert store.get_node("ds:customers") is not None, "the scan was not recorded"
    assert store.resolve_column(store.column_id("ds:customers", "spend")) == "spend"
    assert store.stats()["edge_classified_as"] >= 1
    store.record_dataset("ds:clean", derived_from="ds:customers", columns=[("spend", "float64")])
    store.record_model("model:m", trained_on="ds:clean", feature_columns=["spend"])
    assert [hit["model"] for hit in store.models_touching_pii()] == ["model:m"]


def test_a_failed_run_records_nothing(monkeypatch, tmp_path):
    import app.agents.execution_agent as exec_mod

    monkeypatch.setattr(exec_mod, "execute_code_on_local", lambda **kwargs: {
        "status": "error", "execution_error": "boom", "execution_stderr": "Traceback"})
    new_state = exec_mod.execution_agent_node_local({
        "user_id": USER, "messages": [], "coder_definition": {"code": "print('x')"},
        "active_dataset_id": "ds:sales", "data_source_location": None,
        "output_location": str(tmp_path / "out.csv")})

    assert not new_state.get("latest_output_dataset_id")
    assert LineageStore(USER).stats() == {"nodes": 0, "edges": 0}
