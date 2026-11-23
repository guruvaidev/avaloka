"""The batch Ray analysis path says what it analysed.

The generated script counts the file's rows on the workers, then runs the
user's main(df) on the first RAY_ANALYSIS_SAMPLE_ROWS rows only. These tests
render that script through the real node (RAYJOB_DRYRUN), run it against a
local CSV with a stand-in ``ray`` module, and check that

* the analysis saw exactly the sample, and the metrics, the reply and the run
  record say so (rows used and rows in the file);
* no rate is computed over bytes that were not processed;
* a turn meant for Ray that has no connection runs locally only with a label,
  and a scheduled run refuses instead.

No cluster, no network: the data is generated in tmp_path.
"""
import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.execution_agent as ea
from app.infra.ray_job_runner import RayJobOutcome

FAKE_RAY = textwrap.dedent('''\
    """Stand-in for ray: tasks run inline, one worker node."""
    def init(*a, **k): pass
    def nodes(): return [{"Alive": True, "Resources": {"CPU": 2.0}, "NodeManagerAddress": "10.0.0.2"}]
    def cluster_resources(): return {"CPU": 2.0}
    class _Remote:
        def __init__(self, fn): self._fn = fn
        def remote(self, *a, **k): return self._fn(*a, **k)
    def remote(*a, **k):
        if a and callable(a[0]) and not k: return _Remote(a[0])
        return lambda fn: _Remote(fn)
    def get(x): return x
''')
FAKE_RAY_UTIL = "def get_node_ip_address(): return '10.0.0.2'\n"

USER_CODE = textwrap.dedent('''\
    import pandas as pd
    def main(df):
        return pd.DataFrame({"rows_seen": [len(df)]})
''')


def _write_csv(path, rows, pad=0):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id,region,amount,note\n")
        for i in range(rows):
            fh.write(f"{i},r{i % 7},{i * 0.5},{'x' * pad}\n")


def _render_script(data_uri, artifact_uri, metrics_uri):
    state = {
        "ray_namespace": "ray-jobs",
        "ray_workers": 2,
        "data_source_location_cloud": data_uri,
        "connection_id": "conn-abc",
        "dataset_id": "ds-001",
        "coder_definition": {"code": USER_CODE},
        "output_artifact_uri": artifact_uri,
        "output_metrics_uri": metrics_uri,
        "messages": [],
    }
    with patch.dict(os.environ, {"RAYJOB_DRYRUN": "1"}):
        out = ea.execution_agent_node_ray(state)
    docs = [d for d in yaml.safe_load_all(out["execution_result"]["rayjob_yaml"]) if d]
    cm = next(d for d in docs if d.get("kind") == "ConfigMap")
    return cm["data"]["sample_code.py"]


def _run_script(tmp_path, rows, pad=0):
    data = tmp_path / "data.csv"
    _write_csv(data, rows, pad)
    artifact = tmp_path / "artifact.json"
    metrics = tmp_path / "metrics.json"
    script = _render_script(str(data), str(artifact), str(metrics))
    fake = tmp_path / "fake_ray"
    (fake / "ray").mkdir(parents=True)
    (fake / "ray" / "__init__.py").write_text(FAKE_RAY)
    (fake / "ray" / "util.py").write_text(FAKE_RAY_UTIL)
    (tmp_path / "job.py").write_text(script)
    env = {k: v for k, v in os.environ.items() if k != "GCP_SERVICE_ACCOUNT_JSON"}
    env["PYTHONPATH"] = str(fake)
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "job.py")], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return {
        "data": data,
        "logs": proc.stdout,
        "artifact_uri": str(artifact),
        "artifact": json.loads(artifact.read_text()),
        "metrics": json.loads(metrics.read_text()),
    }


@pytest.fixture(autouse=True)
def _no_memory_dispatch(monkeypatch):
    # The local node records the reply to memory through Celery/Redis; keep it offline.
    monkeypatch.setattr(ea, "_dispatch_execution_memory_record", lambda *a, **k: None)


@pytest.fixture(scope="module")
def run_200k(tmp_path_factory):
    return _run_script(tmp_path_factory.mktemp("ray200k"), 200_000)


def test_analysis_sees_only_the_sample_and_metrics_say_so(run_200k):
    # The user's main(df) got the first 50,000 rows, not the file.
    assert run_200k["artifact"]["output_rows"] == [{"rows_seen": 50_000}]
    m = run_200k["metrics"]
    assert m.get("rows_used") == 50_000
    assert m.get("rows_total") == 200_000
    assert m.get("sample") == 1


def test_no_rate_over_bytes_that_were_not_analysed(run_200k):
    m = run_200k["metrics"]
    size = run_200k["data"].stat().st_size
    # The old figure divided the whole file's bytes by the runtime although
    # main(df) saw 50,000 rows. It must not come back under any name.
    assert "throughput_mb_s" not in m
    assert "bytes_processed" not in m
    assert m.get("analysed_rows_per_s") == pytest.approx(
        m["rows_used"] / m["analysis_runtime_s"], rel=0.01
    )
    # The workers really read every byte to count rows; that rate is named
    # for the scan, and over the bytes the scan read.
    assert m.get("scan_bytes") == size


def test_row_total_is_counted_not_estimated(tmp_path):
    # 300-byte rows: the old 130-bytes-per-row estimate would be far off.
    out = _run_script(tmp_path, 60_000, pad=280)
    assert out["metrics"].get("rows_total") == 60_000
    assert out["metrics"].get("rows_used") == 50_000


def test_small_file_is_not_called_a_sample(tmp_path):
    out = _run_script(tmp_path, 1_234)
    m = out["metrics"]
    assert (m.get("rows_used"), m.get("rows_total"), m.get("sample")) == (1_234, 1_234, 0)


def _node_with_outcome(logs, extra_state=None):
    state = {
        "ray_namespace": "ray-jobs", "ray_workers": 2, "ray_timeout_s": 60,
        "data_source_location_cloud": "s3://bucket/data.csv", "connection_id": "conn-abc",
        "dataset_id": "ds-001", "coder_definition": {"code": USER_CODE},
        "messages": [HumanMessage(content="total amount by region")],
        **(extra_state or {}),
    }
    outcome = RayJobOutcome(
        status="SUCCEEDED", rayjob_name="batch-heavy-ds-001", namespace="ray-jobs",
        job_status="SUCCEEDED", deployment_status="Complete", runtime_s=12.0,
        logs=logs, details={},
    )
    with patch.dict(os.environ, {"RAYJOB_DRYRUN": "0", "RAY_DASHBOARD_URL": "http://ray.invalid"}), \
         patch.object(ea, "_run_coro_sync", return_value={"provider": "s3"}), \
         patch.object(ea, "run_rayjob_from_yaml", return_value=outcome), \
         patch.object(ea, "record_analysis_lineage", return_value=None), \
         patch.object(ea, "_dispatch_execution_memory_record", return_value=None):
        return ea.execution_agent_node_ray(state)


def test_reply_and_run_record_state_the_scope(run_200k):
    out = _node_with_outcome(run_200k["logs"])
    reply = out["messages"][-1].content
    assert "computed on a 50,000-row sample of the 200,000-row file" in reply
    assert "not the entire dataset" in reply
    scope = out["execution_result"].get("data_scope") or {}
    assert (scope.get("rows_used"), scope.get("rows_total"), scope.get("sample")) == (50_000, 200_000, True)


def test_scheduled_run_record_carries_the_scope(run_200k):
    import app.core.celery_app as celery_app

    out = _node_with_outcome(run_200k["logs"])
    # As agent_task hands it over: messages serialised to dicts.
    out["messages"] = celery_app.convert_message_objects_to_dicts(out["messages"])
    written = {}
    client = MagicMock()
    client.table.return_value.update.side_effect = lambda row: written.update(row) or MagicMock()
    with patch.object(celery_app, "_supabase", return_value=client):
        celery_app._record_run_finish("sched-1", "run-1", out, 0.0, {"task_schedule": {"task_type": "execute"}})
    scope = (written.get("result") or {}).get("data_scope") or {}
    assert (scope.get("rows_used"), scope.get("rows_total")) == (50_000, 200_000)
    # The run log the UI shows carries the reply, label included.
    assert any("50,000-row sample of the 200,000-row file" in line for line in written["result"]["logs"])


# ── D1.4: no connection_id ───────────────────────────────────────────────────

def _local_state(tmp_path, rows=1_234, **extra):
    data = tmp_path / "local.csv"
    _write_csv(data, rows)
    return {
        "ray_namespace": "ray-jobs", "ray_workers": 2,
        "data_source_location_cloud": "gs://bucket/big.csv",
        "data_source_location_local": str(data),
        "output_location": str(tmp_path / "out" / "output.csv"),
        "connection_id": None, "dataset_id": "ds-001",
        "coder_definition": {"code": USER_CODE},
        "messages": [HumanMessage(content="total amount by region")],
        **extra,
    }


def _fake_local_exec(code, input_data_location, output_location, read_path_map=None):
    os.makedirs(os.path.dirname(output_location), exist_ok=True)
    with open(output_location, "w") as fh:
        fh.write("region,total\nr0,1.0\n")
    return {"status": "success", "execution_stdout": "", "execution_stderr": ""}


def test_interactive_fallback_is_labelled_with_rows_and_reason(tmp_path):
    state = _local_state(tmp_path)
    with patch.object(ea, "execute_code_on_local", side_effect=_fake_local_exec), \
         patch.dict(os.environ, {"RAYJOB_DRYRUN": "0"}):
        out = ea.execution_agent_node_ray(state)
    reply = out["messages"][-1].content
    assert "ran locally on 1,234 rows, not on Ray, because no cloud storage connection" in reply
    fb = out["execution_result"].get("ray_fallback") or {}
    assert fb.get("reason") == "no cloud storage connection is linked to this dataset"
    assert fb.get("rows_used") == 1_234
    assert not out.get("ray_fallback_reason"), "the reason must not carry into a later turn"


def test_server_guard_reason_labels_the_local_run(tmp_path):
    # send_message's guard passes ray_fallback_reason into the graph; the
    # local node is what runs, and it must say so.
    state = _local_state(tmp_path, ray_fallback_reason="the dataset has no cloud copy for Ray to read")
    state["data_source_location"] = state["data_source_location_local"]
    with patch.object(ea, "execute_code_on_local", side_effect=_fake_local_exec):
        out = ea.execution_agent_node_local(state)
    assert "ran locally on 1,234 rows, not on Ray, because the dataset has no cloud copy" in out["messages"][-1].content
    assert (out["execution_result"].get("ray_fallback") or {}).get("rows_used") == 1_234


@pytest.mark.parametrize("how", ["state", "context"])
def test_scheduled_run_refuses_instead_of_running_locally(tmp_path, how):
    from app.core.celery_app import _run_status_from_final
    from app.core.scheduled_run_context import scheduled_run_context

    state = _local_state(tmp_path, **({"schedule_id": "sched-1"} if how == "state" else {}))
    local = MagicMock(side_effect=lambda s: {**s, "execution_result": {"status": "success"}})
    with patch.object(ea, "execution_agent_node_local", local), \
         patch.dict(os.environ, {"RAYJOB_DRYRUN": "0"}):
        if how == "context":
            with scheduled_run_context("sched-1", "run-1"):
                out = ea.execution_agent_node_ray(state)
        else:
            out = ea.execution_agent_node_ray(state)
    assert local.call_count == 0, "a scheduled Ray run must not quietly run locally"
    assert out["execution_result"]["status"] == "error"
    assert "scheduled run was not started" in out["messages"][-1].content
    status, error = _run_status_from_final(out, task_type="execute")
    assert status == "failure" and "no cloud storage connection" in error
