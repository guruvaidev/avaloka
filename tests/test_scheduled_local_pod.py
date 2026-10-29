import json
import os
from datetime import date, datetime, timezone
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app.agents import execution_agent


def test_scheduled_executor_discovery_uses_helm_deployment(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        stdout = "scheduled-pod" if args[1] == "get" else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(execution_agent.subprocess, "run", fake_run)
    monkeypatch.setenv("AVALOKA_SCHEDULED_EXECUTION_NAMESPACE", "avaloka-test")
    monkeypatch.setattr(
        execution_agent,
        "_SCHEDULED_EXECUTION_DEPLOYMENT",
        "avaloka-celery-executor",
    )
    monkeypatch.setattr(
        execution_agent,
        "_SCHEDULED_EXECUTION_SELECTOR",
        "app.kubernetes.io/component=celery-executor",
    )

    namespace, pod_name = execution_agent._get_scheduled_execution_pod()

    assert (namespace, pod_name) == ("avaloka-test", "scheduled-pod")
    assert calls[0][0][:4] == [
        "kubectl", "rollout", "status", "deployment/avaloka-celery-executor"
    ]
    assert calls[1][0][1:3] == ["get", "pods"]
    assert "app.kubernetes.io/component=celery-executor" in calls[1][0]


def test_scheduled_local_execution_uses_pod_executor(monkeypatch):
    routed = []

    monkeypatch.setattr(
        "app.agents.execution_agent.execution_agent_node_scheduled_pod",
        lambda state: routed.append(state) or state,
    )
    monkeypatch.setattr(
        "app.core.celery_app.convert_message_dicts_to_objects",
        lambda messages: messages,
    )
    monkeypatch.setattr(
        "app.core.celery_app.convert_message_objects_to_dicts",
        lambda messages: messages,
    )
    from app.core import celery_app as celery_module

    state = {
        "task_schedule": {"task_type": "execute"},
        "messages": [AIMessage(content="scheduled")],
        "execution_mode": "local",
    }

    result = celery_module._run_agent_task(state)

    assert len(routed) == 1
    assert routed[0] is not state
    assert "task_schedule" not in routed[0]
    assert routed[0]["execution_mode"] == "local"
    assert routed[0]["messages"] == state["messages"]
    assert state["task_schedule"] == {"task_type": "execute"}
    assert result["task_run_statistics"]["start"]


def test_redbeat_json_round_trips_date_without_changing_type():
    from app.core.celery_app import AvalokaJSONDecoder, AvalokaJSONEncoder

    original_date = date(2025, 4, 3)
    original_datetime = datetime(2025, 4, 3, 12, 30, tzinfo=timezone.utc)

    encoded = json.dumps(
        {"date": original_date, "datetime": original_datetime},
        cls=AvalokaJSONEncoder,
    )
    decoded = json.loads(encoded, cls=AvalokaJSONDecoder)

    assert decoded["date"] == original_date
    assert type(decoded["date"]) is date
    assert decoded["datetime"] == original_datetime


def test_scheduled_pod_stages_cloud_input_from_blob_store(monkeypatch):
    pod_states = []
    copied_paths = []

    class FakeStore:
        def get_file(self, object_key, destination):
            assert object_key == "input.csv"
            destination.write_text("value\n1\n", encoding="utf-8")

    monkeypatch.setattr(
        execution_agent,
        "_get_scheduled_execution_pod",
        lambda: ("avaloka-test", "scheduled-pod"),
    )
    from app.services import storage_service

    monkeypatch.setattr(
        storage_service,
        "_store_and_key_from_uri",
        lambda uri, object_name: (FakeStore(), "input.csv"),
    )

    def fake_command(args, **kwargs):
        if "cp" in args:
            source_path = args[args.index("cp") + 3]
            copied_paths.append(source_path)
            assert os.path.isfile(source_path)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "-i" in args:
            pod_state = json.loads(kwargs["input"])
            pod_states.append(pod_state)
            result = json.dumps(pod_state, default=str)
            return SimpleNamespace(
                stdout=f"AVALOKA_EXECUTION_RESULT:{result}\n",
                stderr="",
                returncode=0,
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(execution_agent, "_run_scheduled_pod_command", fake_command)
    source_uri = "gs://datasets/input.csv"
    state = {
        "messages": [],
        "data_source_location": source_uri,
        "data_source_location_cloud": source_uri,
        "coder_definition": {"code": f"df = read_csv('{source_uri}')"},
    }

    result = execution_agent.execution_agent_node_scheduled_pod(state)

    assert len(pod_states) == 1
    staged_path = pod_states[0]["data_source_location"]
    assert staged_path.startswith("/tmp/avaloka-scheduled-inputs/")
    assert pod_states[0]["coder_definition"]["code"] == f"df = read_csv('{staged_path}')"
    assert copied_paths
    assert result["data_source_location"] == source_uri
    assert result["coder_definition"] == state["coder_definition"]
