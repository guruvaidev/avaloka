"""server.py changes from 29-30 Oct.

* ``_entire_job_in_flight``: a stale analysis_task_id (purged message stuck in
  PENDING forever) used to lock every later "Full dataset" question onto the
  500-row sample. The lock now expires after ANALYSIS_DEFER_INFLIGHT_TTL_S.
* ``_build_deferred_job_card``: the deterministic reply for full-dataset jobs.
"""
from __future__ import annotations

import time

import pytest

server = pytest.importorskip("app.api.server")


# ------------------------------------------------- _entire_job_in_flight ---

@pytest.fixture
def job_running(monkeypatch):
    monkeypatch.setattr(server, "_scheduled_task_is_in_flight", lambda task_id: True)


@pytest.fixture
def job_finished(monkeypatch):
    monkeypatch.setattr(server, "_scheduled_task_is_in_flight", lambda task_id: False)


def test_no_task_id_is_not_in_flight(job_running):
    assert server._entire_job_in_flight({}) is False
    assert server._entire_job_in_flight({"analysis_task_id": None}) is False


def test_task_without_timestamp_is_treated_as_stale(job_running):
    # The 30 Oct bug: e2cd89e5 had no usable timestamp and reported PENDING forever.
    sess = {"analysis_task_id": "e2cd89e5"}
    assert server._entire_job_in_flight(sess) is False


def test_task_older_than_ttl_is_stale(job_running):
    sess = {"analysis_task_id": "t1", "analysis_task_scheduled_at": time.time() - 2 * 3600}
    assert server._entire_job_in_flight(sess) is False


def test_recent_running_task_is_in_flight(job_running):
    sess = {"analysis_task_id": "t1", "analysis_task_scheduled_at": time.time() - 30}
    assert server._entire_job_in_flight(sess) is True


def test_recent_finished_task_is_not_in_flight(job_finished):
    sess = {"analysis_task_id": "t1", "analysis_task_scheduled_at": time.time() - 30}
    assert server._entire_job_in_flight(sess) is False


def test_ttl_comes_from_env(job_running, monkeypatch):
    monkeypatch.setenv("ANALYSIS_DEFER_INFLIGHT_TTL_S", "10")
    sess = {"analysis_task_id": "t1", "analysis_task_scheduled_at": time.time() - 20}
    assert server._entire_job_in_flight(sess) is False


def test_bad_ttl_env_falls_back_to_default(job_running, monkeypatch):
    monkeypatch.setenv("ANALYSIS_DEFER_INFLIGHT_TTL_S", "not-a-number")
    sess = {"analysis_task_id": "t1", "analysis_task_scheduled_at": time.time() - 60}
    assert server._entire_job_in_flight(sess) is True


# ---------------------------------------------- _build_deferred_job_card ---

def test_job_card_names_the_job_and_dataset():
    card = server._build_deferred_job_card(
        task_id="b7c87680-bcb4-418e-9984-d899555945f5",
        dataset_name="2019-Nov.csv",
        interpreted_question="Rank brands by number of views",
        eta_hint="about 10 minutes",
    )
    assert "Full-dataset analysis scheduled" in card
    assert "b7c87680-bcb4-418e-9984-d899555945f5" in card
    assert "2019-Nov.csv" in card
    assert "Rank brands by number of views" in card
    assert "about 10 minutes" in card


def test_job_card_truncates_long_questions():
    card = server._build_deferred_job_card("t1", "ds", "q" * 1000, "soon")
    assert "q" * 301 not in card
    assert "..." in card


def test_job_card_has_no_numbers_from_a_sample():
    card = server._build_deferred_job_card("t1", "ds", "Rank brands", "soon")
    assert "sample" not in card.lower() or "no partial or sampled results" in card.lower()


def test_job_card_text_fits_full_dataset_jobs():
    """Step 7 (1a): the card is reused for Full-dataset Ray jobs, so it must not
    claim the dataset is over the in-memory budget."""
    card = server._build_deferred_job_card("t1", "ds", "q", "soon")
    assert "in-memory analysis budget" not in card


def test_job_card_does_not_promise_every_row():
    """MAJ-030: the Ray batch job analyses the first RAY_ANALYSIS_SAMPLE_ROWS rows
    and labels the scope in its result, so the card must not claim every row
    (carried over from the declined #617's review)."""
    card = server._build_deferred_job_card("t1", "ds", "q", "soon").lower()
    assert "every row" not in card
    assert "how many of the file's rows it analysed" in card
