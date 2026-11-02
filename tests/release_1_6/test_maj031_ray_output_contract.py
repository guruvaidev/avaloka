"""MAJ-030 / MAJ-031: a Ray or scheduled result answers the question it was asked.

Owner report (1.6): "list the lowest/max price per <group>" on the Ray path
and in scheduled-task output came back without the price, or with a count
where the price should be.

* MAJ-031: ``_autogenerate_code`` (used when a Ray turn or a scheduled run has
  no coder code) detected an aggregation but always emitted
  ``groupby(col).size()``, so a count was labelled as the measure, and a
  first-row value of other columns was attached to each group.
* MAJ-030 (the Ray path's honesty): whatever the code returned was presented
  as the answer. Nothing checked that the requested measure and group columns
  were present, or that the aggregation was the one asked for.

Each test runs the real path: the Ray script is rendered by
``execution_agent_node_ray``, taken from the RayJob ConfigMap and run as a
subprocess against a generated CSV with a stand-in ``ray`` module (tasks run
inline), and the figures are compared with a pandas reference computed here on
the same CSV. The scheduled tests go through ``celery_app.agent_task`` and read
what ``_record_run_finish`` writes for the run.

Plan case: the 1.6 Master Test Plan has no case for Ray or scheduled output
columns; the nearest are K8-09 (a RayJob runs) and AG-17 (run history keeps
each run's output). No cluster, no network, no committed fixture.
"""
import base64
import io
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from langchain_core.messages import HumanMessage

import app.agents.execution_agent as ea
import app.core.celery_app as celery_app
from app.infra.ray_job_runner import RayJobOutcome

try:  # the server harness's ``client`` fixture, for the task result route
    from tests.server_harness import client  # noqa: F401
except ImportError:  # tests/ on sys.path
    from server_harness import client  # noqa: F401

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

SCHEMA = {"product": "object", "qty": "int64", "category": "object", "price": "float64"}
CATEGORIES = ["books", "garden", "music", "toys"]


def _write_csv(path):
    # 61 rows over 4 categories. Each category's first row is neither its
    # minimum nor its maximum price, so a first-row value cannot pass as either.
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("product,qty,category,price\n")
        for i in range(61):
            price = round(3.5 + ((i * 7 + 5) % 23) * 1.25, 2)
            fh.write(f"p{i},{1 + i % 5},{CATEGORIES[i % 4]},{price}\n")


def _reference(csv_path, agg):
    return pd.read_csv(csv_path).groupby("category")["price"].agg(agg).to_dict()


_AGG_WORDS = {
    "min": ("min", "lowest", "minimum", "smallest"),
    "max": ("max", "highest", "maximum", "largest"),
    "mean": ("mean", "average", "avg"),
}


def _measure_column(frame, agg, n_aggs):
    """The result column holding ``agg`` of price, however the code named it."""
    price_cols = [c for c in frame.columns if "price" in str(c).lower()]
    named = [c for c in price_cols if any(w in str(c).lower() for w in _AGG_WORDS[agg])]
    if len(named) == 1:
        return named[0]
    if n_aggs == 1 and len(price_cols) == 1:
        return price_cols[0]
    raise AssertionError(f"no column for the {agg} of price; result columns are {list(frame.columns)}")


def _assert_measure(frame, csv_path, aggs):
    assert "category" in frame.columns, f"group column missing; result columns are {list(frame.columns)}"
    by_group = frame.set_index("category")
    for agg in aggs:
        col = _measure_column(frame, agg, len(aggs))
        want = _reference(csv_path, agg)
        got = {k: float(v) for k, v in by_group[col].items()}
        assert set(got) == set(want), f"{agg}: groups {sorted(got)} != {sorted(want)}"
        for k in want:
            assert got[k] == pytest.approx(want[k], rel=1e-9), (
                f"{agg} of price for {k!r}: result {got[k]} != pandas {want[k]} (column {col!r})")


def _frame_from_output_file(ofd):
    assert isinstance(ofd, dict) and ofd.get("content", "").startswith("data:text/csv;base64,"), ofd
    return pd.read_csv(io.StringIO(base64.b64decode(ofd["content"].split("base64,", 1)[1]).decode()))


def _fake_rayjob_runner(workdir):
    """Stands in for run_rayjob_from_yaml: runs the ConfigMap's script locally."""
    fake = workdir / "fake_ray"
    (fake / "ray").mkdir(parents=True, exist_ok=True)
    (fake / "ray" / "__init__.py").write_text(FAKE_RAY)
    (fake / "ray" / "util.py").write_text(FAKE_RAY_UTIL)

    def run(yaml_text, namespace, rayjob_name, **_kw):
        docs = [d for d in yaml.safe_load_all(yaml_text) if d]
        cm = next(d for d in docs if d.get("kind") == "ConfigMap")
        script = workdir / "job.py"
        script.write_text(cm["data"]["sample_code.py"])
        env = {k: v for k, v in os.environ.items() if k != "GCP_SERVICE_ACCOUNT_JSON"}
        env["PYTHONPATH"] = str(fake)
        proc = subprocess.run([sys.executable, str(script)], cwd=workdir, env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-3000:]
        return RayJobOutcome(status="SUCCEEDED", rayjob_name=rayjob_name, namespace=namespace,
                             job_status="SUCCEEDED", deployment_status="Complete", runtime_s=1.0,
                             logs=proc.stdout, details={})
    return run


def _ray_state(tmp_path, question, code=None):
    csv_path = tmp_path / "products.csv"
    _write_csv(csv_path)
    return {
        "ray_namespace": "ray-jobs", "ray_workers": 1, "ray_timeout_s": 60,
        "data_source_location_cloud": str(csv_path), "connection_id": "conn-abc",
        "dataset_id": "ds-maj031", "schema": dict(SCHEMA),
        "coder_definition": {"code": code} if code else {},
        "output_artifact_uri": str(tmp_path / "artifact.json"),
        "output_metrics_uri": str(tmp_path / "metrics.json"),
        "messages": [HumanMessage(content=question)],
    }, csv_path


def _ray_patches(tmp_path):
    return [
        patch.dict(os.environ, {"RAYJOB_DRYRUN": "0", "RAY_DASHBOARD_URL": "http://ray.invalid"}),
        patch.object(ea, "_run_coro_sync", return_value={"provider": "s3"}),
        patch.object(ea, "run_rayjob_from_yaml", side_effect=_fake_rayjob_runner(tmp_path)),
        patch.object(ea, "record_analysis_lineage", return_value=None),
        patch.object(ea, "_dispatch_execution_memory_record", return_value=None),
    ]


def _run_ray_node(tmp_path, question, code=None):
    state, csv_path = _ray_state(tmp_path, question, code)
    ps = _ray_patches(tmp_path)
    for p in ps:
        p.start()
    try:
        return ea.execution_agent_node_ray(state), csv_path
    finally:
        for p in reversed(ps):
            p.stop()


def _run_scheduled(tmp_path, question, code=None, mode="k8s-ray", local_result=None,
                   schedule=None, final_out=None):
    """One scheduled run through agent_task; returns what the run record got.
    ``final_out`` (a list) receives agent_task's final state."""
    state, csv_path = _ray_state(tmp_path, question, code)
    # As handle_task_scheduling stores them (convert_message_objects_to_dicts).
    state["messages"] = [{"role": "human", "content": question}]
    state.update({
        "schedule_id": "sched-maj031", "session_id": "sess-1", "user_id": "u-1",
        "execution_mode": mode,
        "task_schedule": schedule or {"task_type": "execute", "schedule_type": "absolute", "max_runs": 3},
    })
    if mode != "k8s-ray":
        state["data_source_location"] = str(csv_path)
    written = {}
    client = MagicMock()
    client.table.return_value.update.side_effect = lambda row: written.update(row) or MagicMock()
    ps = _ray_patches(tmp_path) + [
        patch.object(celery_app, "_supabase", return_value=client),
        patch.object(celery_app, "current_task", SimpleNamespace(request=SimpleNamespace(id="run-maj031"))),
    ]
    if local_result is not None:
        ps.append(patch.object(ea, "execution_agent_node_scheduled_pod",
                               side_effect=lambda s: {**s, **local_result(s, csv_path)}))
    for p in ps:
        p.start()
    try:
        final = celery_app.agent_task(state)
        if final_out is not None:
            final_out.append(final)
    finally:
        for p in reversed(ps):
            p.stop()
    return written, csv_path


# Code an LLM might write that loses the measure: a count where the price should be.
COUNT_AS_MIN_PRICE = textwrap.dedent('''\
    import pandas as pd
    def main(df):
        return df.groupby("category").size().reset_index(name="min_price")
''')
COUNT_ONLY = textwrap.dedent('''\
    import pandas as pd
    def main(df):
        return df.groupby("category").size().reset_index(name="count")
''')
CORRECT_LLM_CODE = textwrap.dedent('''\
    import pandas as pd
    def main(df):
        out = df.groupby("category", as_index=False).agg(lowest_price=("price", "min"),
                                                         highest_price=("price", "max"))
        return out.sort_values("lowest_price")
''')

REFUSAL_MARKERS = ("not shown as an answer",)


# ── (a) the Ray path ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("agg,question", [
    ("min", "What is the lowest price per category?"),
    ("max", "Show the maximum price for each category"),
    ("mean", "average price by category"),
])
def test_maj031_maj030_ray_autogen_grouped_price_matches_pandas(tmp_path, agg, question):
    """MAJ-031/MAJ-030: a Ray turn with no coder code returns the requested
    aggregation of price per category, equal to pandas on the same rows."""
    out, csv_path = _run_ray_node(tmp_path, question)
    assert out["execution_result"]["status"] == "succeeded", out["execution_result"].get("message")
    frame = _frame_from_output_file(out.get("output_file_data"))
    _assert_measure(frame, csv_path, [agg])
    assert (out["execution_result"].get("output_contract") or {}).get("status") == "passed"


def test_maj031_maj030_ray_min_max_mean_price_in_one_question(tmp_path):
    """MAJ-031/MAJ-030: one question asking for min, max and mean price per
    category gets all three measure columns, each equal to pandas."""
    out, csv_path = _run_ray_node(tmp_path, "List the min, max and mean price per category")
    frame = _frame_from_output_file(out.get("output_file_data"))
    _assert_measure(frame, csv_path, ["min", "max", "mean"])
    assert (out["execution_result"].get("output_contract") or {}).get("status") == "passed"


def test_maj031_ray_count_question_is_a_count_without_first_row_values(tmp_path):
    """MAJ-031: a real count question still emits a count, and no other
    column's first-row value is attached to a group as if it were its figure."""
    out, csv_path = _run_ray_node(tmp_path, "How many products are there per category?")
    frame = _frame_from_output_file(out.get("output_file_data"))
    want = pd.read_csv(csv_path).groupby("category").size().to_dict()
    count_cols = [c for c in frame.columns if c != "category"]
    assert len(count_cols) == 1, f"a count per category should be one column; got {list(frame.columns)}"
    assert dict(zip(frame["category"], frame[count_cols[0]])) == want
    assert (out["execution_result"].get("output_contract") or {}).get("status") == "passed"


def test_maj030_ray_refuses_a_count_labelled_as_the_price(tmp_path):
    """MAJ-030: code that returns per-group row counts under a price name is
    refused in plain words; the table is not presented as the answer."""
    out, _ = _run_ray_node(tmp_path, "What is the lowest price per category?", code=COUNT_AS_MIN_PRICE)
    reply = out["messages"][-1].content
    assert any(m in reply for m in REFUSAL_MARKERS), reply
    assert "| books" not in reply and "min_price" not in reply.split("\n")[0]
    assert out.get("output_file_data") is None
    contract = out["execution_result"].get("output_contract") or {}
    assert contract.get("status") == "refused"
    assert any(v["rule"] == "wrong_aggregation" for v in contract.get("violations", [])), contract


def test_maj030_ray_refuses_a_result_without_the_measure(tmp_path):
    """MAJ-030: a result with no price column for a max-price question is refused."""
    out, _ = _run_ray_node(tmp_path, "Show the maximum price for each category", code=COUNT_ONLY)
    reply = out["messages"][-1].content
    assert any(m in reply for m in REFUSAL_MARKERS), reply
    assert out.get("output_file_data") is None
    contract = out["execution_result"].get("output_contract") or {}
    assert [v["rule"] for v in contract.get("violations", [])] == ["missing_measure"], contract


def test_maj030_ray_correct_llm_code_passes_the_contract_unchanged(tmp_path):
    """MAJ-030: code that names its columns its own way (lowest_price,
    highest_price) and is right is checked and presented, not refused."""
    out, csv_path = _run_ray_node(tmp_path, "List the lowest and highest price per category",
                                  code=CORRECT_LLM_CODE)
    frame = _frame_from_output_file(out.get("output_file_data"))
    _assert_measure(frame, csv_path, ["min", "max"])
    contract = out["execution_result"].get("output_contract") or {}
    assert contract.get("status") == "passed" and contract.get("values_checked") is True, contract
    assert not any(m in out["messages"][-1].content for m in REFUSAL_MARKERS)


# ── (b) the scheduled-task output path ───────────────────────────────────────

def test_maj031_maj030_scheduled_ray_run_record_has_min_max_mean_price(tmp_path):
    """MAJ-031/MAJ-030: a scheduled Ray run's recorded output (output_json, what
    the Run History shows) holds min, max and mean price per category = pandas."""
    written, csv_path = _run_scheduled(tmp_path, "List the min, max and mean price per category")
    assert written.get("status") == "success", written.get("error")
    truth = (written.get("result") or {}).get("run_truth") or {}
    assert (truth.get("outcome"), truth.get("reason_code")) == ("succeeded", "ok"), truth
    rows = (written.get("result") or {}).get("output_json")
    assert rows, f"no output rows recorded: {written.get('result')}"
    frame = pd.DataFrame(rows)
    for c in frame.columns:
        if "price" in str(c).lower():   # output_json cells are CSV text
            frame[c] = pd.to_numeric(frame[c])
    _assert_measure(frame, csv_path, ["min", "max", "mean"])


def test_maj030_scheduled_ray_run_flags_a_count_labelled_as_the_price(tmp_path):
    """MAJ-030: a scheduled Ray run whose code returns counts under a price name
    is recorded as not answered, with a plain reason and no output rows."""
    written, _ = _run_scheduled(tmp_path, "What is the lowest price per category?", code=COUNT_AS_MIN_PRICE)
    result = written.get("result") or {}
    # Item 8a: the refusal is run truth failed / output_contract, not a status of its own.
    truth = result.get("run_truth") or {}
    assert (truth.get("outcome"), truth.get("reason_code")) == ("failed", "output_contract"), truth
    assert written.get("status") == "failure", written
    assert result.get("output_json") is None, result.get("output_json")
    assert any(m in str(written.get("error")) for m in REFUSAL_MARKERS), written.get("error")
    assert ((result.get("output_contract") or {}).get("status")) == "refused"


def test_maj031_scheduled_local_run_flags_a_result_without_the_measure(tmp_path):
    """MAJ-031: a scheduled non-Ray run (executor pod) whose output has no price
    column for a min-price question is flagged, not recorded as the answer."""
    def pod_result(_state, _csv):
        csv = "category,count\nbooks,16\ngarden,15\nmusic,15\ntoys,15\n"
        return {
            "execution_result": {"status": "success"},
            "output_file_data": {"filename": "out.csv", "size": len(csv),
                                 "content": "data:text/csv;base64," + base64.b64encode(csv.encode()).decode()},
            "messages": _state["messages"],
        }
    written, _ = _run_scheduled(tmp_path, "What is the lowest price per category?",
                                code=COUNT_ONLY, mode="local", local_result=pod_result)
    result = written.get("result") or {}
    truth = result.get("run_truth") or {}
    assert (truth.get("outcome"), truth.get("reason_code")) == ("failed", "output_contract"), truth
    assert written.get("status") == "failure", written
    assert result.get("output_json") is None
    assert any(m in str(written.get("error")) for m in REFUSAL_MARKERS), written.get("error")


# ── run truth (item 8a, #632): one outcome for a refused contract on every path ──

def _pod_without_measure(state, _csv):
    csv = "category,count\nbooks,16\ngarden,15\nmusic,15\ntoys,15\n"
    return {
        "execution_result": {"status": "success"},
        "output_file_data": {"filename": "out.csv", "size": len(csv),
                             "content": "data:text/csv;base64," + base64.b64encode(csv.encode()).decode()},
        "messages": state["messages"],
    }


def _task_result_route(monkeypatch, final):
    """GET /tasks/t1/result/0 for a finished task whose Celery result is ``final``."""
    from fastapi.testclient import TestClient  # noqa: F401
    import app.api.server as server
    try:
        from tests.server_harness import SESSIONS, make_auth_headers
    except ImportError:  # tests/ on sys.path
        from server_harness import SESSIONS, make_auth_headers
    SESSIONS["s-maj030"] = {"user_id": "u-maj030", "dataset_id": "ds-maj031", "tasks": ["t1"]}
    entry = SimpleNamespace(schedule=SimpleNamespace(
        remaining_estimate=lambda _last: SimpleNamespace(total_seconds=lambda: 0.0)))
    monkeypatch.setattr(server.AvalokaEntry, "generate_key", staticmethod(lambda *a, **k: "key-t1"))
    monkeypatch.setattr(server.AvalokaEntry, "from_key", staticmethod(lambda *a, **k: entry))
    monkeypatch.setattr(server.AvalokaScheduler, "get_task_ids", staticmethod(lambda *a, **k: ["run-maj031"]))
    monkeypatch.setattr(server.AvalokaScheduler, "get_last_run_at", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(server, "_scheduled_task_metadata_from_entry",
                        lambda _e: server.scheduled_task_metadata({"task_type": "execute"}))
    monkeypatch.setattr(server, "AsyncResult", lambda *_a, **_k: SimpleNamespace(
        state="SUCCESS", failed=lambda: False, traceback=None, result=json.loads(json.dumps(final))))

    async def _no_persist(**_kw):
        return None
    monkeypatch.setattr(server, "_persist_assets_background", _no_persist)
    headers = make_auth_headers("u-maj030")
    headers["X-Avaloka-Session"] = "s-maj030"
    return headers


@pytest.mark.parametrize("path", ["ray_run_now", "executor_scheduled"])
def test_maj030_refused_contract_is_run_truth_failed_output_contract(tmp_path, monkeypatch, client, path):
    """MAJ-030 + item 8a: a refused output contract ends the run as run truth
    ``failed`` with reason code ``output_contract``, the same in the run record
    (_record_run_finish) and in GET /tasks/{id}/result/{n}, for a Ray run-now
    job and a scheduled executor-pod run. The reasons stay in
    ``result.output_contract`` and no rows are returned."""
    finals = []
    if path == "ray_run_now":
        # The shape send_message builds for a "Full dataset" question.
        written, _ = _run_scheduled(
            tmp_path, "What is the lowest price per category?", code=COUNT_AS_MIN_PRICE,
            schedule={"task_type": "execute", "schedule_type": "relative", "second": 0, "max_runs": 1},
            final_out=finals)
    else:
        written, _ = _run_scheduled(tmp_path, "What is the lowest price per category?",
                                    code=COUNT_ONLY, mode="local", local_result=_pod_without_measure,
                                    final_out=finals)
    result = written.get("result") or {}
    truth = result.get("run_truth") or {}
    assert (truth.get("outcome"), truth.get("reason_code")) == ("failed", "output_contract"), truth
    assert "did not answer the question" in truth.get("reason", ""), truth
    assert written.get("status") == "failure"
    assert result.get("output_json") is None
    assert (result.get("output_contract") or {}).get("status") == "refused"

    headers = _task_result_route(monkeypatch, finals[0])
    resp = client.get("/tasks/t1/result/0", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    route_truth = body.get("run_truth") or {}
    assert (route_truth.get("outcome"), route_truth.get("reason_code")) == ("failed", "output_contract"), body
    assert body["result"]["output_json"] is None and body["result"]["output_file_data"] is None
    assert (body["result"].get("output_contract") or {}).get("status") == "refused"
