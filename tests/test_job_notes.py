"""app/core/job_notes.py: queue once, deliver only at a turn boundary, coalesce,
tag as system context, write the transcript row, never call a model.

    pytest tests/test_job_notes.py -q
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core import job_ledger as jl
from app.core import job_notes as jn
from app.services import notification_service as ns
from tests.job_fakes import FakeDB

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
    jn._TURNS_IN_FLIGHT.clear()


def _done(db, kind="transfer", status="succeeded", reason=None, thread=THREAD, label="orders ➡️ warehouse"):
    jid = jl.start(USER, kind, thread_id=thread, label=label)
    jl.finish(jid, status, fail_reason=reason)
    return jid


# ---- rendering ---------------------------------------------------------------------

def test_note_text_per_status():
    j = jl.Job(id="j", user_id=USER, kind="transfer", status="succeeded", label="a ➡️ b")
    assert jn.render_note(j) == "✓ Data transfer (a ➡️ b) finished."
    j.status, j.fail_reason = "failed", "timed_out"
    assert jn.render_note(j) == "✗ Data transfer (a ➡️ b) did not complete: it timed out."
    j.status = "cancelled"
    assert jn.render_note(j) == "– Data transfer (a ➡️ b) was cancelled."


def test_note_never_contains_the_error_line(db):
    jid = jl.start(USER, "ray_job", thread_id=THREAD)
    jl.finish(jid, "failed", fail_reason="cluster_error", error_line="KeyError: 'patient_ssn' in /tmp/x.py")
    assert "patient_ssn" not in db.notes[0]["text"] and "/tmp" not in db.notes[0]["text"]
    assert db.notes[0]["text"] == "✗ Analysis on the cluster did not complete: it failed on the cluster."


def test_compose_bands():
    one = [{"text": "✓ A finished.", "status": "succeeded"}]
    assert jn.compose(one) == "✓ A finished."
    three = [{"text": f"✓ J{i} finished.", "status": "succeeded"} for i in range(3)]
    out = jn.compose(three)
    assert out.splitlines()[0] == "3 background jobs finished: 3 succeeded, 0 did not."
    assert len(out.splitlines()) == 4 and "more" not in out
    four = three + [{"text": "✗ J3 did not complete: it failed.", "status": "failed"}]
    out = jn.compose(four)
    assert out.splitlines()[0] == "4 background jobs finished: 3 succeeded, 1 did not."
    assert out.splitlines()[-1].startswith("…and 1 more")
    assert jn.compose([]) == ""


# ---- queueing --------------------------------------------------------------------------

def test_enqueue_once_per_job(db):
    jid = _done(db)
    job = jl._job_from_row(db.jobs[0])
    assert len(db.notes) == 1 and db.notes[0]["job_id"] == jid and db.notes[0]["delivered_at"] is None
    assert jn.enqueue(job) is False and len(db.notes) == 1


def test_no_thread_no_note_and_no_note_before_terminal(db):
    assert jn.enqueue(jl.Job(id="x", user_id=USER, kind="transfer", status="succeeded")) is False
    assert jn.enqueue(jl.Job(id="y", user_id=USER, kind="transfer", status="running", thread_id=THREAD)) is False
    assert db.notes == []


# ---- delivery at a turn boundary -----------------------------------------------------

def test_flush_delivers_history_message_transcript_row_and_marks(db):
    jid = _done(db)
    batch = jn.flush_sync(THREAD)
    assert batch and batch["job_ids"] == [jid] and batch["count"] == 1 and batch["transcript_written"]
    assert db.notes[0]["delivered_at"] and db.notes[0]["batch_id"] == batch["batch_id"]
    row = db.messages[0]
    assert (row["analysis_id"], row["role"], row["author_id"]) == (FakeDB.ANALYSIS_ID, "assistant", None)
    assert row["content"] == batch["text"] and row["output"]["job_note"]["batch_id"] == batch["batch_id"]
    assert jn.flush_sync(THREAD) is None                       # nothing left


def test_flush_is_skipped_while_a_turn_is_in_flight_and_resumes_after(db):
    jid = _done(db)
    jn.turn_started(THREAD)
    assert jn.flush_sync(THREAD) is None and db.notes[0]["delivered_at"] is None and db.messages == []
    jn.turn_started(THREAD)                                     # two overlapping turns
    jn.turn_finished(THREAD)
    assert jn.flush_sync(THREAD) is None                        # one still running
    jn.turn_finished(THREAD)
    batch = jn.flush_sync(THREAD)
    assert batch and batch["job_ids"] == [jid]


def test_flush_coalesces_a_burst_into_one_message(db):
    ids = [_done(db) for _ in range(5)]
    batch = jn.flush_sync(THREAD)
    assert batch["count"] == 5 and set(batch["job_ids"]) == set(ids)
    assert len(db.messages) == 1 and db.messages[0]["content"].startswith("5 background jobs finished")
    assert len({n["batch_id"] for n in db.notes}) == 1 and all(n["delivered_at"] for n in db.notes)


def test_flush_caps_a_runaway_producer(db):
    for _ in range(jn.MAX_PENDING + 7):
        _done(db)
    batch = jn.flush_sync(THREAD)
    assert batch["count"] == jn.MAX_PENDING and "More jobs finished than can be listed" in batch["text"]
    assert len(db.messages) == 1
    rest = jn.flush_sync(THREAD)
    assert rest["count"] == 7


def test_flush_without_an_analyses_row_still_marks_delivered(db):
    db.tables["analyses"].clear()
    _done(db)
    batch = jn.flush_sync(THREAD)
    assert batch and not batch["transcript_written"] and db.messages == [] and db.notes[0]["delivered_at"]


def test_flush_scoped_to_the_owner(db):
    _done(db)
    assert jn.flush_sync(THREAD, user_id="someone-else") is None
    assert db.notes[0]["delivered_at"] is None
    assert jn.flush_sync(THREAD, user_id=USER)


def test_transcript_write_failure_does_not_lose_the_note(db):
    _done(db)
    db.fail_tables.add("analysis_messages")
    batch = jn.flush_sync(THREAD)
    assert batch and not batch["transcript_written"] and db.notes[0]["delivered_at"]   # history gets it; UI via bell


def test_flush_never_calls_a_model(db, monkeypatch):
    import app.agents.planner as planner
    for name in dir(planner):
        if name.startswith("get_") and "llm" in name.lower():
            monkeypatch.setattr(planner, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    import langchain_core.language_models.chat_models as cm
    monkeypatch.setattr(cm.BaseChatModel, "invoke", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    monkeypatch.setattr(cm.BaseChatModel, "ainvoke", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    _done(db)
    assert jn.flush_sync(THREAD)


# ---- system-context tagging (ruling 5) ------------------------------------------------

def test_note_message_is_tagged_framed_and_skipped_by_learners():
    note = jn.make_note_message("✓ Data transfer finished.", ["j1"], "b1")
    assert isinstance(note, AIMessage) and jn.is_job_note(note)
    assert note.additional_kwargs[jn.SYSTEM_CONTEXT_FLAG] is True and note.additional_kwargs["job_ids"] == ["j1"]
    assert note.content.startswith(jn.MODEL_FRAME) and note.content.endswith("✓ Data transfer finished.")
    assert not isinstance(note, SystemMessage)                       # a mid-history system message breaks providers
    history = [HumanMessage("copy orders"), AIMessage("Transfer started."), note, HumanMessage("did it finish?")]
    assert jn.without_job_notes(history) == [history[0], history[1], history[3]]
    assert not jn.is_job_note(AIMessage("plain reply"))
