"""No success after failure: a failed or empty step is never reported as worked.

Each test pins one site where the product said "success", "ready", "full
profile", "ran successfully" or "My team is back" after the work under it had
failed or produced nothing. Each fails on base/2.0-stack (ce126765) + #545.
The swarm coordinator's sites are in tests/avaloka/test_no_success_after_failure.py.

No live model, no Anthropic key, Celery in eager mode.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.multi_action as multi
import app.agents.profiling_agent as profiling
import app.agents.sampling_async as sampling_async
import app.agents.scheduler as scheduler
import app.api.server as server
import app.api.workflow as workflow
from app.agents import visualization_agent as va
import app.agents.claim_verifier as claim_verifier
from app.agents.claim_verifier import claim_verifier_node
from app.agents.planner import plan_etl_job
from app.agents.result_narrator import result_narrator_node
from app.api import chat_swarm
from app.api.chat_voice import prompts_after_turn, prompts_after_upload
from app.core import celery_app as celery_module

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import (SESSIONS, FakeGraph, _do_upload, _install_sampler,  # noqa: F401
                                      _quiet_send, client, make_auth_headers)
except ImportError:  # pragma: no cover
    from server_harness import (SESSIONS, FakeGraph, _do_upload, _install_sampler,  # noqa: F401
                                _quiet_send, client, make_auth_headers)

#: Words no failed or empty step may be reported with.
SUCCESS_WORDS = ("success", "complete", "ran successfully", "is back", "ready")


def _says_success(text: str) -> bool:
    low = (text or "").lower()
    return any(word in low for word in SUCCESS_WORDS)


# =========================================================================== #
# Scheduling
# =========================================================================== #

def _make_eager(mp: pytest.MonkeyPatch) -> None:
    # setitem, not setattr: Celery's Settings is a mapping whose keys are not in
    # the instance __dict__. Since pytest 9.2 (#10644) setattr's undo then calls
    # delattr, which Settings rejects -- and the aborted undo leaves eager mode on
    # for every later test in the process.
    mp.setitem(celery_module.celery_app.conf, "task_always_eager", True)
    mp.setitem(celery_module.celery_app.conf, "task_eager_propagates", False)


@pytest.fixture
def eager(monkeypatch):
    _make_eager(monkeypatch)


def test_eager_mode_is_undone_after_the_test():
    """The eager fixture must undo cleanly and leave Celery's config as it found it."""
    conf = celery_module.celery_app.conf
    before = (conf.task_always_eager, conf.task_eager_propagates)
    mp = pytest.MonkeyPatch()
    _make_eager(mp)
    assert (conf.task_always_eager, conf.task_eager_propagates) == (True, False)
    try:
        mp.undo()
        undo_error = None
    except AttributeError as exc:  # pytest >= 9.2 with setattr on Settings
        undo_error = exc
    assert undo_error is None, f"undo failed: {undo_error}"
    assert (conf.task_always_eager, conf.task_eager_propagates) == before


def test_celery_records_a_scheduled_analysis_that_ran_nothing_as_failure(eager, monkeypatch):
    """celery_app.py:306 -- no execution status was "success", Celery SUCCESS."""
    monkeypatch.setattr(celery_module, "_run_agent_task", lambda state: {"messages": []})

    result = celery_module.agent_task.delay({"task_schedule": {"task_type": "execute"}})

    assert result.state == "FAILURE"
    assert isinstance(result.result, celery_module.ScheduledRunFailed)
    assert not _says_success(str(result.result))
    assert celery_module._run_status_from_final({}, task_type="execute")[0] == "failure"
    assert celery_module._run_status_from_final(None, task_type="execute")[0] == "failure"


def test_celery_records_a_failed_scheduled_analysis_as_failure(eager, monkeypatch):
    """celery_app.py:306 -- a failed run returned its state, so Celery said SUCCESS."""
    final = {"messages": [], "execution_result": {"status": "failed", "error": "KeyError: 'region'"}}
    monkeypatch.setattr(celery_module, "_run_agent_task", lambda state: dict(final))

    result = celery_module.agent_task.delay({"task_schedule": {"task_type": "execute"}})

    assert result.state == "FAILURE"
    message = str(result.result)
    assert message == "The scheduled analysis failed: the code asked for a column the data doesn't have."
    assert "KeyError" not in message


def test_background_sampling_failure_is_a_celery_failure(eager):
    """sampling_async.py -- an error was returned, not raised: Celery SUCCESS."""
    result = celery_module.agent_task.delay(
        {"task_schedule": {"task_type": "sample_profile"}, "dataset_id": "ds-1"})

    assert result.state == "FAILURE"
    assert isinstance(result.result, sampling_async.BackgroundSamplingFailed)


class _FakeSession:
    def __init__(self):
        self.saved = None

    async def get(self, session_id):
        return {"sample_status": "quick_sample", "profiling_status": "quick_profile"}

    async def save(self, session_id, sess):
        self.saved = dict(sess)
        return True


class _FakeCache:
    def __init__(self, *_a, **_k):
        pass

    async def close(self):
        pass


def test_a_failed_profile_never_upgrades_the_session_to_full_profile(monkeypatch, tmp_path):
    """sampling_async.py -- the session went to full_profile with no profile."""
    import fsspec

    import app.agents.execution_agent as execution_agent
    import app.agents.sampling_persistence as persistence
    import app.core.cache as cache_mod
    import app.infra.k8s_secrets as k8s_secrets
    import app.infra.ray_job_runner as ray_job_runner
    import app.services.session_service as session_service

    artifact = {
        "portfolio_samples": {"random_baseline": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
        "sample_statistics": {"column_statistics": {"a": {"type": "numeric"}}},
        "schema": {"a": "int", "b": "int"},
        "total_rows": 2,
    }
    yaml_path = tmp_path / "rayjob.yaml"
    yaml_path.write_text("kind: RayJob\n")
    monkeypatch.setattr(sampling_async, "RAYJOB_YAML_PATH", str(yaml_path))
    monkeypatch.setattr(execution_agent, "render_rayjob_yaml", lambda **_k: "yaml")
    monkeypatch.setattr(k8s_secrets, "create_cloud_secret", lambda **_k: "sec-1")
    monkeypatch.setattr(k8s_secrets, "kubectl_delete_secret", lambda *_a, **_k: None)
    monkeypatch.setattr(ray_job_runner, "run_rayjob_from_yaml",
                        lambda **_k: SimpleNamespace(status="SUCCEEDED", runtime_s=1.0, logs=""))
    monkeypatch.setattr(fsspec, "open", lambda *_a, **_k: io.StringIO(json.dumps(artifact)))
    monkeypatch.setattr(fsspec, "filesystem", lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(persistence, "persist_full_profile", lambda **_k: None)
    monkeypatch.setattr(persistence, "upsert_profile", lambda **_k: None)
    # profile_full never raises; a failure comes back in the profile's place.
    monkeypatch.setattr(profiling, "profile_full", lambda **_k: {
        "full_profiling_result": {"status": "error", "reason": "model unavailable"},
        "profiling_status": "error",
    })
    store = _FakeSession()
    monkeypatch.setattr(session_service, "get_session", store.get)
    monkeypatch.setattr(session_service, "save_session", store.save)
    monkeypatch.setattr(cache_mod, "RedisCache", _FakeCache)
    monkeypatch.setattr(sampling_async, "_run_coro_sync", asyncio.run)

    out = sampling_async.run_background_sampling({
        "dataset_id": "ds-1", "data_source_location_cloud": "gs://bkt/data.csv",
        "session_id": "s-1", "profiling_status": "quick_profile",
    })

    assert store.saved is not None, "the session update never ran"
    assert store.saved["profiling_status"] == "quick_profile"
    assert store.saved["sample_status"] == "full_sample"   # the samples are real
    assert out["profiling_status"] != "full_profile"
    assert out["full_profiling_result"] is None


def test_scheduled_completion_message_never_calls_a_failed_run_complete():
    """server.py:4817 -- every non-training failure got the success_message."""
    metadata = {"task_type": "execute", "success_message": "The scheduled analysis is complete."}
    failed = {"execution_result": {"status": "failed", "error": "ParserError: Expected 2 fields"}}

    message = server._scheduled_task_completion_message(failed, [], metadata)

    assert message == "The scheduled analysis failed: I couldn't read the data file."
    assert not _says_success(server._scheduled_task_completion_message({}, [], metadata))


class _FakeSchedule:
    def remaining_estimate(self, last_run_at):
        return datetime.timedelta(seconds=30)


class _FakeEntry:
    args = [{"task_schedule": {"task_type": "execute"}}]
    schedule = _FakeSchedule()
    max_runs = 1
    total_run_count = 1
    last_run_at = datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc)
    due_at = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)

    @staticmethod
    def generate_key(app, task_id):
        return f"redbeat:{task_id}"

    @classmethod
    def from_key(cls, key, app):
        return cls()


class _FakeScheduler:
    @staticmethod
    def get_task_ids(key, generate_key=False):
        return ["exec-1"]

    @staticmethod
    def get_last_run_at(key, generate_key=False):
        return None

    @staticmethod
    def get_last_run_task_id(key, generate_key=False):
        return "exec-1"


_FAILED_FINAL = {
    "messages": [],
    "execution_result": {"status": "failed", "error": "KeyError: 'region'"},
    "output_location": None,
    "output_file_data": None,
    "task_run_statistics": {"start": 0.0, "end": 1.0, "time_spent": 1.0},
}


def _fake_async_result(final, running_polls=0):
    """The first ``running_polls`` lookups report STARTED, later ones SUCCESS."""
    lookups = {"n": 0}

    class _FakeAsyncResult:
        traceback = None

        def __init__(self, task_id, app=None):
            lookups["n"] += 1
            self.state = "STARTED" if lookups["n"] <= running_polls else "SUCCESS"
            self.result = json.loads(json.dumps(final)) if self.state == "SUCCESS" else None

        def failed(self):
            return self.state == "FAILURE"

    return _FakeAsyncResult


def _install_fake_scheduler(monkeypatch, module, final, running_polls=0):
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(module, "AvalokaEntry", _FakeEntry, raising=False)
    monkeypatch.setattr(module, "AvalokaScheduler", _FakeScheduler, raising=False)
    monkeypatch.setattr(module, "AsyncResult", _fake_async_result(final, running_polls), raising=False)
    if module is server:
        monkeypatch.setattr(server, "sleep", _no_sleep, raising=False)


def _sse_events(text):
    return [json.loads(line[len("data: "):]) for block in text.split("\n\n")
            for line in block.splitlines() if line.startswith("data: ")]


def test_task_stream_sends_failed_not_success_for_a_failed_run(client, monkeypatch):
    """server.py:4187 -- the SSE stream sent status SUCCESS for a failed run."""
    upload = _do_upload(client, user_id="w7-sse")
    SESSIONS[upload["session_id"]]["tasks"] = ["sched-1"]
    # Running at request time, so /status streams through task_sse_iter.
    _install_fake_scheduler(monkeypatch, server, _FAILED_FINAL, running_polls=2)
    headers = make_auth_headers("w7-sse")
    headers["X-Avaloka-Session"] = upload["session_id"]

    resp = client.get("/tasks/sched-1/status", headers=headers)

    assert resp.status_code == 200, resp.text
    events = [e for e in _sse_events(resp.text) if e.get("status")]
    assert [e["status"] for e in events] == ["STARTED", "FAILED"], events
    assert not _says_success(events[-1]["message"])

    _install_fake_scheduler(monkeypatch, server, _FAILED_FINAL)

    result = client.get("/tasks/sched-1/result/0", headers=headers).json()
    assert result["status"] == "FAILED"
    assert not _says_success(result["message"])


def test_scheduler_never_says_ran_successfully_for_a_failed_run(monkeypatch):
    """scheduler.py:178-236 -- Celery SUCCESS was read as "ran successfully"."""
    _install_fake_scheduler(monkeypatch, scheduler, _FAILED_FINAL)
    base = {"messages": [], "task_operation": {"task_id": "sched-1", "result_index": 0}}

    status = scheduler.handle_task_status(dict(base, messages=[]))["messages"][-1].content
    result = scheduler.handle_task_result(dict(base, messages=[]))["messages"][-1].content

    for text in (status, result):
        assert "ran successfully" not in text
        assert "failed" in text.lower()


# =========================================================================== #
# Insights
# =========================================================================== #

_ERROR_PROFILE = {
    "full_profiling_result": {"status": "error", "reason": "model unavailable"},
    "profiling_status": "error",
}
_STATS = {"column_statistics": {"a": {"type": "numeric", "missing_ratio": 0.0},
                                "b": {"type": "numeric", "missing_ratio": 0.0}},
          "data_shape": {"rows": 2, "columns": 2}}


def test_register_existing_does_not_keep_an_error_as_the_profile(client, monkeypatch):
    """server.py:3307 -- profile_full's error dict was stored as the profile."""
    _install_sampler(monkeypatch, {"random_baseline": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
                     sample_statistics=_STATS)
    monkeypatch.setattr(server, "profile_full", lambda **_k: dict(_ERROR_PROFILE))

    resp = client.post("/api/register-existing-storage", headers=make_auth_headers("w7-reg"),
                       json={"storage_uri": "gs://external-bucket/some-prefix", "key": "foo.csv",
                             "connection_id": "conn-1"})

    assert resp.status_code == 200, resp.text
    sess = SESSIONS[resp.json()["session_id"]]
    assert not sess.get("full_profiling_result"), sess.get("full_profiling_result")


def test_register_folder_does_not_keep_an_error_as_the_profile(client, monkeypatch):
    """server.py:3680 -- the same error dict, and profiling_status full_profile."""
    async def _conn(connection_id):
        return {"id": connection_id, "user_id": "w7-folder", "provider": "gcp"}

    monkeypatch.setattr(server, "get_cloud_connection", _conn, raising=False)
    monkeypatch.setattr(server, "detect_folder_table_type", lambda keys: {
        "table_type": "parquet_dir", "glob": "**/*.parquet", "hive_partitioning": False,
    }, raising=False)
    _install_sampler(monkeypatch, {"random_baseline": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
                     sample_statistics=_STATS)
    monkeypatch.setattr(server, "profile_full", lambda **_k: dict(_ERROR_PROFILE))

    resp = client.post("/api/register-existing-folder", headers=make_auth_headers("w7-folder"),
                       json={"storage_uri": "gs://bkt", "folder": "tables/sales/", "connection_id": "c-f"})

    assert resp.status_code == 200, resp.text
    sess = SESSIONS[resp.json()["session_id"]]
    assert not sess.get("full_profiling_result"), sess.get("full_profiling_result")
    assert sess.get("profiling_status") != "full_profile"


def test_upload_voice_does_not_call_unmeasured_columns_clean():
    """chat_voice.py:236 -- "Clean and quiet" from a schema with no statistics."""
    prompts = prompts_after_upload(sample_statistics={"data_shape": {"rows": 120, "columns": 3}},
                                   schema=["alpha", "beta", "gamma"])

    message = prompts[0]["message"]
    assert "Clean" not in message
    assert "haven't measured" in message


def test_a_profile_with_every_section_incomplete_is_not_full_profile(monkeypatch):
    """profiling_agent.py:600-628 -- no model call answered, status full_profile."""
    monkeypatch.setattr(profiling, "profiling_llm", object())
    monkeypatch.setattr(profiling, "_resolve_domain",
                        lambda schema, rows: ({"category": "Unknown", "confidence": "low"}, False))
    monkeypatch.setattr(profiling, "_call_all_column_semantics", lambda **k: {
        "column_explanations": {}, "predictive_values": {},
        "columns_without_semantics": list(k["schema"]),
    })
    monkeypatch.setattr(profiling, "_call_insights", lambda *a, **k: None)

    res = profiling.profile_full(schema=["a", "b"], sample_rows=[{"a": 1, "b": 2}],
                                 sample_statistics={"column_statistics": {}})

    assert res["profiling_status"] != "full_profile"
    assert profiling.usable_full_profile(res) is None


def _viz_state(result_csv: str) -> dict:
    import base64

    return {
        "dataset_id": "ds-viz",
        "schema": {"num_countries": "int64"},
        "uploaded_csv_columns": ["num_countries"],
        "uploaded_csv_preview": [["num_countries"], [49]],
        "output_file_data": {"content": "data:text/csv;base64,"
                             + base64.b64encode(result_csv.encode()).decode()},
        "execution_result": {"status": "success"},
    }


def test_visualization_is_not_ready_with_zero_charts():
    """visualization_agent.py:4562 -- "ready" with an empty chart list."""
    out = va.visualization_agent_node(_viz_state("num_countries\n49\n"))

    assert (out["visualization_config"] or {}).get("charts") == [], "precondition: no chart drawn"
    assert out["visualization_status"] != "ready"
    assert not _says_success(out["visualization_status"])


# =========================================================================== #
# Swarm, in the app
# =========================================================================== #

def test_chat_swarm_does_not_say_the_team_is_back_after_a_failed_mission():
    """chat_swarm.py:202 -- "My team is back" for a mission that failed."""
    records = [
        {"type": "swarm.requested", "seq": 1, "goal": "what drives churn", "source": "churn.csv"},
        {"type": "swarm.finished", "seq": 2, "status": "failed", "duration_ms": 1200,
         "economics": None, "error_class": "RuntimeError"},
    ]

    prompt = chat_swarm.summary_prompt(records, edition="free", mission_id="m-1")

    assert "is back" not in prompt["message"]
    assert "failed" in prompt["message"]


# =========================================================================== #
# Execution
# =========================================================================== #

@patch("app.agents.planner.llm")
def test_a_security_refusal_is_not_an_execution_success(mock_llm):
    """planner.py:5135 (and 5155, 5179, 5207) -- a refusal set status "success"."""
    out = plan_etl_job({
        "messages": [HumanMessage(content="please inject code into the training loop")],
        "planner_definition": {}, "memory_hints": [], "session_id": "w7-planner",
    })

    assert "Security Alert" in out["messages"][-1].content, "precondition: the gate refused"
    assert out["execution_result"]["status"] != "success"
    assert not mock_llm.invoke.called


def _action_plan(*actions):
    return multi.ActionPlan(mode="aggregate", actions=[
        multi.MetricAction(id=f"m{i}", operation=op, column=col, group_by=[],
                           output_name=f"{op}_{col}")
        for i, (op, col) in enumerate(actions, 1)
    ])


def test_a_partial_structured_run_is_not_a_success(tmp_path):
    """workflow.py:701 -- some metrics missing, execution_result "success"."""
    source = tmp_path / "sales.csv"
    source.write_text("amount,note\n10,x\n20,y\n")
    state = {
        "messages": [], "data_source_location": str(source), "input_data_type": "csv",
        "structured_action_plan": _action_plan(("sum", "amount"), ("mean", "note")).model_dump(),
    }

    out = workflow.execute_structured_actions_node(state)

    assert out["output_tables"] and out["multi_action_missing"], "precondition: a partial run"
    assert out["execution_result"]["status"] != "success"
    verified = claim_verifier_node({**state, **out, "messages": [AIMessage(content="The total amount is 30.")]})
    assert verified["verification_report"].get("verified") is not True


def test_a_sum_over_a_column_with_no_usable_values_is_not_zero():
    """multi_action.py:283 -- sum over an unusable column returned 0, complete."""
    frame = pd.DataFrame({"amount": [None, None, None]}, dtype=object)

    result = multi.execute_action_plan(frame, _action_plan(("sum", "amount")))

    assert result.status == "incomplete"
    assert not result.tables
    assert result.missing[0]["reason"] == "No usable values were available for this calculation"


@pytest.mark.parametrize("status", ["refused", "timeout", "dryrun", "cancelled"])
def test_the_narrator_does_not_narrate_a_run_that_did_not_succeed(monkeypatch, status):
    """result_narrator.py:345 -- only "error"/"failed" were treated as failure."""
    import app.agents.result_narrator as narrator

    monkeypatch.setattr(narrator, "narrator_llm", None)
    state = {"messages": [HumanMessage(content="total by region")],
             "execution_result": {"status": status},
             "output_json": [{"region": "north", "total": 10}]}

    out = result_narrator_node(state)

    assert out["messages"] == state["messages"]


@patch("app.agents.planner.llm")
def test_chat_voice_does_not_call_a_refusal_an_answer(mock_llm):
    """planner.py refusals, as the chat voice then told them: "The code ran" and
    "That's my answer" for a request that ran nothing. A stale result from the
    previous turn is still on the state."""
    final = plan_etl_job({
        "messages": [HumanMessage(content="please inject code into the training loop")],
        "planner_definition": {}, "memory_hints": [], "session_id": "w7-voice",
        "output_json": [{"total": 10}],
    })

    steps = prompts_after_turn(final, question="inject code")

    for p in steps:
        assert "That's my answer" not in p["message"] and "The code ran" not in p["message"], p
        assert "couldn't finish" not in p["message"], "a refusal is not a failed run"


# =========================================================================== #
# Verifier enforcement
# =========================================================================== #

_ROWS = [{"region": "north", "churn_rate": 0.31}, {"region": "south", "churn_rate": 0.22}]
_UNSAFE_REPLY = ("North has the higher churn rate at 0.31. "
                 "Late deliveries cause churn in the north region.")


def _verified_state():
    return {
        "messages": [AIMessage(content=_UNSAFE_REPLY)],
        "execution_result": {"status": "success"},
        "output_json": list(_ROWS),
    }


def test_the_verifier_flags_the_causal_claim_as_unsafe():
    """Precondition for the enforcement tests: safe_to_present is False here."""
    report = claim_verifier_node(_verified_state())["verification_report"]

    assert report["safe_to_present"] is False
    assert any(f["verdict"] == "overreach" for f in report["findings"])


def test_an_unsafe_claim_is_withheld_from_the_reply():
    report = claim_verifier_node(_verified_state())["verification_report"]

    shown = claim_verifier.presentable_reply(_UNSAFE_REPLY, report)

    assert "cause churn" not in shown
    assert "North has the higher churn rate at 0.31." in shown
    assert "held back 1 statement" in shown


def test_an_unsafe_claim_that_cannot_be_located_withholds_the_whole_reply():
    report = {"safe_to_present": False, "findings": [
        {"claim": "A sentence that is not in the reply at all, and causes things.", "verdict": "overreach"},
    ]}

    assert claim_verifier.presentable_reply(_UNSAFE_REPLY, report) == claim_verifier.WITHHELD_ANSWER


def test_a_safe_report_or_a_failed_step_leaves_the_reply_alone():
    assert claim_verifier.presentable_reply(_UNSAFE_REPLY, {"safe_to_present": True, "findings": []}) == _UNSAFE_REPLY
    # A failed step is unsafe with no findings: its reply is the failure itself.
    assert claim_verifier.presentable_reply("I couldn't read the data file.",
                             {"safe_to_present": False, "findings": []}) == "I couldn't read the data file."


class _UnsafeGraph(FakeGraph):
    """The graph's answer carries a claim the verifier found unsafe."""

    def invoke(self, state_in, config=None):
        final = super().invoke(state_in, config)
        state = _verified_state()
        final.update(state, messages=state_in["messages"] + state["messages"], ready_to_code=False)
        final.update(claim_verifier_node(state))
        return final


def test_the_chat_reply_never_shows_an_unsafe_claim_as_fact(client, monkeypatch):
    """W4: safe_to_present was False for a claim, and the text was still shown."""
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _UnsafeGraph(), raising=False)
    upload = _do_upload(client, user_id="w7-verify")

    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": "which region churns more?",
                             "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers("w7-verify"))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["verification_safe_to_present"] is False
    reply = body["messages"][-1]["content"]
    assert "cause churn" not in reply
    assert "held back" in reply
