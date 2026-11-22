"""The producers' side of the ledger: claims, attachment, the turn settlement
rule in server.py, and a contract that every long-running path records its
terminal state through job_ledger.finish.

    pytest tests/test_job_producers.py -q
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from app.core import job_ledger as jl
from app.core import job_notes as jn
from app.services import notification_service as ns
from tests.job_fakes import FakeDB

REPO = pathlib.Path(__file__).resolve().parents[1]
USER = FakeDB.AUTH_UID
THREAD = FakeDB.THREAD_ID


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(ns, "_client", lambda: fake)
    ns._profile_cache.clear()
    jl.set_store(jl.SupabaseJobs(client=fake))
    jn.set_store(jn.SupabaseNotes(client=fake))
    jn._TURNS_IN_FLIGHT.clear()
    yield fake
    jl.set_store(None)
    jn.set_store(None)


# ---- claim / attach / get -----------------------------------------------------------

def test_claim_sets_kind_resource_and_label_but_not_on_a_terminal_row(db):
    jid = jl.start(USER, "analysis", thread_id=THREAD)
    assert jl.claim(jid, kind="transfer", resource_type="transfer", resource_id="dta-1", label="a ➡️ b") is True
    row = db.jobs[0]
    assert (row["kind"], row["resource_type"], row["resource_id"], row["label"]) == ("transfer", "transfer", "dta-1", "a ➡️ b")
    jl.finish(jid, "succeeded")
    assert jl.claim(jid, kind="analysis") is False and db.jobs[0]["kind"] == "transfer"
    assert jl.claim(None, kind="analysis") is False and jl.claim(jid) is False


def test_claim_rejects_vocabulary_violations(db):
    jid = jl.start(USER, "analysis")
    with pytest.raises(ValueError):
        jl.claim(jid, kind="nope")
    with pytest.raises(ValueError):
        jl.claim(jid, resource_type="file")


def test_claim_scrubs_the_label(db):
    jid = jl.start(USER, "transfer")
    jl.claim(jid, label="copy postgres://etl:" + "FAKE-PW-1" + "@db/x to lake")  # assembled: E11.05
    assert "FAKE-PW-1" not in db.jobs[0]["label"]


def test_attach_marks_an_existing_row_running_and_reports_a_missing_one(db):
    jid = jl.start(USER, "training", status="queued")
    assert jl.attach(jid, celery_task_id="celery-1") is True
    assert db.jobs[0]["status"] == "running" and db.jobs[0]["celery_task_id"] == "celery-1"
    assert jl.attach("00000000-0000-4000-8000-000000000000", celery_task_id="c") is False
    assert jl.attach(None) is False


def test_get_returns_a_job_or_none(db):
    jid = jl.start(USER, "sampling")
    job = jl.get(jid)
    assert job and job.kind == "sampling" and not job.terminal
    assert jl.get("00000000-0000-4000-8000-000000000000") is None and jl.get(None) is None
    db.fail_tables.add("ml_jobs")
    assert jl.get(jid) is None


# ---- the turn settlement rule (server.py) -------------------------------------------------

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    server = None
    _server_skip = str(exc)


def _needs_server():
    if server is None:
        pytest.skip(f"app.api.server could not be imported here: {_server_skip}")


def run(coro):
    return asyncio.run(coro)


def test_settle_hands_the_row_to_the_worker_when_the_turn_scheduled_work(db):
    _needs_server()
    jid = jl.start(USER, "training", thread_id=THREAD, status="running")
    holder = {"ledger_job_id": jid, "long_kind": "training"}
    run(server._settle_turn_job(holder, {"task_info": {"task_id": "beat-entry-1"}}, False, "succeeded", None, None))
    assert holder["ledger_job_id"] is None and db.jobs[0]["status"] == "running" and db.notifications == []


def test_settle_turns_an_unclaimed_long_row_into_a_quiet_analysis(db):
    _needs_server()
    jid = jl.start(USER, "transfer", thread_id=THREAD, status="running")
    holder = {"ledger_job_id": jid, "long_kind": "transfer"}
    run(server._settle_turn_job(holder, {}, False, "succeeded", None, None))
    row = db.jobs[0]
    assert (row["kind"], row["status"]) == ("analysis", "succeeded") and row["notified_at"]
    assert db.notifications == [] and db.notes == []                 # quiet: the user saw the answer


def test_settle_keeps_an_unclaimed_deferred_rows_bell(db):
    _needs_server()
    jid = jl.start(USER, "transfer", thread_id=THREAD, status="running")
    holder = {"ledger_job_id": jid, "long_kind": "transfer"}
    run(server._settle_turn_job(holder, {}, True, "succeeded", None, None))
    assert db.jobs[0]["kind"] == "analysis" and len(db.notifications) == 1
    assert db.notifications[0]["type"] == "analysis.completed"


def test_settle_respects_a_producers_claim(db):
    _needs_server()
    jid = jl.start(USER, "transfer", thread_id=THREAD, status="running")
    jl.claim(jid, resource_type="transfer", resource_id="dta-1", label="a ➡️ b")
    holder = {"ledger_job_id": jid, "long_kind": "transfer"}
    run(server._settle_turn_job(holder, {}, False, "failed", "failed", "boom"))
    row = db.jobs[0]
    assert (row["kind"], row["status"], row["fail_reason"]) == ("transfer", "failed", "failed")
    assert db.notifications[0]["type"] == "transfer.failed" and "boom" not in db.notifications[0]["body"]


def test_settle_is_a_no_op_after_a_producer_finished_the_row(db):
    _needs_server()
    jid = jl.start(USER, "transfer", thread_id=THREAD, status="running")
    jl.claim(jid, resource_type="transfer", resource_id="dta-1")
    jl.finish(jid, "failed", fail_reason="timed_out")
    holder = {"ledger_job_id": jid, "long_kind": "transfer"}
    run(server._settle_turn_job(holder, {}, False, "succeeded", None, None))
    assert db.jobs[0]["status"] == "failed" and len(db.notifications) == 1


def test_settle_without_a_row_does_nothing(db):
    _needs_server()
    run(server._settle_turn_job({}, {}, False, "succeeded", None, None))
    assert db.jobs == []


def test_turn_outcome_maps_run_truth():
    _needs_server()
    from app.api.schemas import ChatResponse
    ok = ChatResponse(messages=[], run_truth={"outcome": "succeeded"})
    assert server._turn_outcome(ok) == ("succeeded", None, None)
    bad = ChatResponse(messages=[], run_truth={"outcome": "failed", "reason": "No column named Age"})
    assert server._turn_outcome(bad) == ("failed", "failed", "No column named Age")
    refused = ChatResponse(messages=[], run_truth={"outcome": "failed", "kind": "refused", "reason": "contract refused the result"})
    assert server._turn_outcome(refused)[1] == "refused"


# ---- the contract: every producer reaches finish() -------------------------------------------

def _calls_in(func_node: ast.AST, attr: str) -> int:
    n = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr:
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ("job_ledger", "jl"):
                n += 1
    return n


def _func(path: str, name: str) -> ast.AST:
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


@pytest.mark.parametrize("path,func,starts,finishes", [
    ("app/agents/planner.py", "_handle_initiate_transfer", 1, 2),          # transfer: start/claim, finish on error and on result
    ("app/agents/execution_agent.py", "execution_agent_node_ray", 1, 2),   # ray job: own row, finish on outcome and on exception
    ("app/core/celery_app.py", "_notify_job_finished", 0, 1),              # celery: scheduled runs, sampling, training, endpoints
    ("app/core/celery_app.py", "_finish_killed_job", 0, 1),                # killed workers
])
def test_producers_record_their_terminal_state_through_the_ledger(path, func, starts, finishes):
    node = _func(path, func)
    assert _calls_in(node, "start") >= starts, f"{func}: start()"
    assert _calls_in(node, "finish") >= finishes, f"{func}: finish()"


def test_mta_training_claims_the_turns_row():
    node = _func("app/agents/mta_v2/agent.py", "_execute_training")
    names = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_claim_training_job" in names, "_execute_training claims nothing"


def test_mta_endpoint_paths_record_their_job_through_training_tasks():
    """Deploy/stop endpoints are Celery jobs with their own ml_jobs row
    (train-infer's enqueue_*_endpoint); the turn claims nothing for them."""
    for func, attr in (("_configure_inference_service", "enqueue_deploy_endpoint"), ("_stop_inference_service", "enqueue_stop_endpoint")):
        node = _func("app/agents/mta_v2/agent.py", func)
        attrs = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert attr in attrs, f"{func} does not enqueue through training_tasks"


def test_agent_task_records_the_start_and_finishes_on_both_paths():
    node = _func("app/core/celery_app.py", "agent_task")
    names = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert names.count("_record_ledger_start") == 1
    assert names.count("_notify_job_finished") == 2


def test_celery_state_carries_the_turns_row_id():
    src = (REPO / "app/graph/etl_state.py").read_text()
    assert "ledger_job_id: Optional[str]" in src and "ledger_job_kind: Optional[str]" in src
    sched = (REPO / "app/agents/scheduler.py").read_text()
    assert "celery_state: ETLState = state.copy()" in sched         # the copy carries every declared channel


def test_ledger_kind_per_task_type():
    from app.core import celery_app as ca
    assert ca._ledger_kind({"task_type": "execute", "schedule_type": "relative", "max_runs": 1}) == "analysis"
    assert ca._ledger_kind({"task_type": "execute", "schedule_type": "crontab", "max_runs": -1}) == "scheduled_run"
    assert ca._ledger_kind({"task_type": "execute", "schedule_type": "absolute"}) == "analysis"
    assert ca._ledger_kind({"task_type": "sample_profile", "schedule_type": "relative", "max_runs": 1}) == "sampling"
    assert ca._ledger_kind({"task_type": "training", "schedule_type": "relative", "max_runs": 1}) == "training"
    assert ca._ledger_kind({"task_type": "start_inference", "schedule_type": "relative", "max_runs": 1}) == "deploy_endpoint"
    assert ca._ledger_kind({"task_type": "stop_inference", "schedule_type": "relative", "max_runs": 1}) == "stop_endpoint"
    for k in {ca._ledger_kind(ts) for ts in ({}, {"task_type": "weird"})}:
        assert k in jl.KINDS
