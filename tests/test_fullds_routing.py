"""scheduler.py / workflow.py changes from 29 Oct.

* Ray execute jobs get a 2 h Celery limit instead of the global 240/300 s, so a
  9 GB scan is not killed mid-run.
* Full-dataset turns go through the scheduler only when a worker is running;
  with no worker they run directly on Ray instead of returning
  "scheduling isn't available".
"""
from __future__ import annotations

import pytest

scheduler = pytest.importorskip("app.agents.scheduler")
workflow = pytest.importorskip("app.api.workflow")

RUN_NOW = {"task_type": "execute", "schedule_type": "relative", "second": 0, "max_runs": 1}
EVERY_MINUTE = {"task_type": "execute", "schedule_type": "crontab", "minute": "*", "max_runs": -1}


# ------------------------------------------------------------ scheduler ---

def test_ray_execute_jobs_get_long_time_limits():
    opts = scheduler._celery_options_for_schedule({"task_type": "execute"}, "k8s-ray")
    assert opts == {"soft_time_limit": 7200, "time_limit": 7500}


def test_local_execute_jobs_keep_short_limits():
    opts = scheduler._celery_options_for_schedule({"task_type": "execute"}, "local") or {}
    assert opts.get("soft_time_limit") != 7200


def test_hard_limit_is_above_soft_limit():
    opts = scheduler._celery_options_for_schedule({"task_type": "execute"}, "k8s-ray")
    assert opts["time_limit"] > opts["soft_time_limit"]


def test_is_run_now():
    assert scheduler.is_run_now(RUN_NOW)
    assert not scheduler.is_run_now(EVERY_MINUTE)


# ------------------------------------------------------------- workflow ---

# The 29 Oct routing change (issue 4) is not in workflow.py yet. These tests
# skip until it lands, then run automatically, with no edit needed here.
needs_workflow_change = pytest.mark.skipif(
    not all(
        hasattr(workflow, name)
        for name in ("_dispatch_full_dataset_run", "_needs_scheduler",
                     "_RUN_NOW_SCHEDULE", "is_celery_worker_running")
    ),
    reason="workflow.py full-dataset routing (issue 4) not applied yet",
)


def _dispatch(updates: dict, state: dict) -> dict:
    out = workflow._dispatch_full_dataset_run(updates, state)
    return out if isinstance(out, dict) else updates


@needs_workflow_change
def test_full_dataset_run_is_scheduled_when_worker_is_running(monkeypatch):
    monkeypatch.setattr(workflow, "is_celery_worker_running", lambda: True)
    result = _dispatch({}, {"execution_mode": "k8s-ray"})
    assert result.get("task_schedule") == workflow._RUN_NOW_SCHEDULE


@needs_workflow_change
def test_full_dataset_run_skips_scheduler_without_worker(monkeypatch):
    monkeypatch.setattr(workflow, "is_celery_worker_running", lambda: False)
    result = _dispatch({"task_schedule": dict(RUN_NOW)}, {"execution_mode": "k8s-ray"})
    assert not result.get("task_schedule"), "with no worker the turn must go straight to Ray"


@needs_workflow_change
def test_run_now_schedule_is_one_shot():
    ts = workflow._RUN_NOW_SCHEDULE
    assert ts["task_type"] == "execute"
    assert ts["max_runs"] == 1


@needs_workflow_change
def test_no_schedule_means_no_scheduler():
    assert not workflow._needs_scheduler(None, "k8s-ray")
    assert not workflow._needs_scheduler(None, "local")