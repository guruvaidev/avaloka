"""Delivery of background-job notes into the chat (server.py + helpers.py +
the planner/coder scans). Hermetic: the outbox runs on tests/job_fakes.FakeDB,
history persistence is captured, no Redis, no Supabase, no model.

    pytest tests/test_server_job_notes.py -q
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"app.api.server could not be imported here: {exc}", allow_module_level=True)

from app.api import helpers
from app.core import job_ledger as jl
from app.core import job_notes as jn
from app.services import notification_service as ns
from tests.job_fakes import FakeDB

USER = FakeDB.AUTH_UID
OTHER = "auth-uid-2222"
THREAD = FakeDB.THREAD_ID


def run(coro):
    return asyncio.run(coro)


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


@pytest.fixture
def history(monkeypatch):
    """A fresh THREAD_META and a recorder for persist_thread_history."""
    meta = {}
    persisted = []
    monkeypatch.setattr(helpers, "THREAD_META", meta)
    monkeypatch.setattr(server, "THREAD_META", meta)

    async def _persist(thread_id):
        persisted.append([m.content for m in meta.get(thread_id, {}).get("lc_msgs", [])])

    monkeypatch.setattr(server, "persist_thread_history", _persist)
    return meta, persisted


@pytest.fixture
def owner_session(monkeypatch):
    async def _get_thread_session(thread_id):
        return "sid-1" if thread_id == THREAD else None

    async def _get_session(sid, **_):
        return {"user_id": USER} if sid == "sid-1" else None

    monkeypatch.setattr(server, "get_thread_session", _get_thread_session)
    monkeypatch.setattr(server, "get_session", _get_session)


def _finish(db, **kw):
    jid = jl.start(USER, kw.pop("kind", "transfer"), thread_id=THREAD, label=kw.pop("label", "orders ➡️ warehouse"))
    jl.finish(jid, kw.pop("status", "succeeded"), **kw)
    return jid


# ---- _deliver_job_notes ----------------------------------------------------------

def test_deliver_appends_one_tagged_note_after_the_reply_and_persists(db, history):
    meta, persisted = history
    meta[THREAD] = {"lc_msgs": [HumanMessage("copy orders"), AIMessage("Transfer started.")], "metadata": {}}
    jid = _finish(db)
    batch = run(server._deliver_job_notes(THREAD, USER))
    assert batch and batch["job_ids"] == [jid]
    msgs = meta[THREAD]["lc_msgs"]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage", "AIMessage"]
    assert jn.is_job_note(msgs[-1]) and not jn.is_job_note(msgs[1])
    assert msgs[-1].content.endswith("✓ Data transfer (orders ➡️ warehouse) finished.")
    assert persisted and persisted[-1][-1] == msgs[-1].content           # written through to Redis
    assert run(server._deliver_job_notes(THREAD, USER)) is None          # delivered once


def test_deliver_is_a_no_op_while_the_turn_is_in_flight(db, history):
    meta, persisted = history
    meta[THREAD] = {"lc_msgs": [HumanMessage("q")], "metadata": {}}
    _finish(db)
    jn.turn_started(THREAD)
    assert run(server._deliver_job_notes(THREAD, USER)) is None
    assert len(meta[THREAD]["lc_msgs"]) == 1 and persisted == [] and db.notes[0]["delivered_at"] is None
    jn.turn_finished(THREAD)
    assert run(server._deliver_job_notes(THREAD, USER))


def test_deliver_never_raises_and_is_bounded(db, history, monkeypatch):
    def slow(*a, **k):
        import time
        time.sleep(0.5)
        return None

    monkeypatch.setattr(server, "JOB_NOTES_FLUSH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(jn, "flush_sync", slow)
    assert run(server._deliver_job_notes(THREAD, USER)) is None
    monkeypatch.setattr(jn, "flush_sync", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert run(server._deliver_job_notes(THREAD, USER)) is None


def test_deliver_calls_no_model(db, history, monkeypatch):
    import langchain_core.language_models.chat_models as cm
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called"))  # noqa: E731
    monkeypatch.setattr(cm.BaseChatModel, "invoke", boom)
    monkeypatch.setattr(cm.BaseChatModel, "ainvoke", boom)
    monkeypatch.setattr(server, "GRAPH", None, raising=False)
    meta, _ = history
    meta[THREAD] = {"lc_msgs": [], "metadata": {}}
    _finish(db)
    assert run(server._deliver_job_notes(THREAD, USER))


# ---- the mid-turn rule, end to end through the turn hooks ------------------------------

def test_a_completion_mid_turn_is_queued_then_flushed_after_the_reply(db, history):
    """Simulates send_message's lifecycle: turn_started → graph runs (the job
    finishes meanwhile) → reply appended → close → deliver. The note must come
    after the reply, once."""
    meta, persisted = history
    meta[THREAD] = {"lc_msgs": [HumanMessage("start the transfer")], "metadata": {}}
    jn.turn_started(THREAD)
    jid = _finish(db)                                                  # Celery finishes during the turn
    assert db.notes[0]["delivered_at"] is None and len(db.notifications) == 1   # bell rang, note queued
    assert run(server._deliver_job_notes(THREAD, USER)) is None         # a flush now does nothing
    meta[THREAD]["lc_msgs"].append(AIMessage("Here is your answer."))    # the turn's own reply
    jn.turn_finished(THREAD)
    batch = run(server._deliver_job_notes(THREAD, USER))
    contents = [m.content for m in meta[THREAD]["lc_msgs"]]
    assert contents[1] == "Here is your answer." and contents[2].endswith("finished.")
    assert batch["job_ids"] == [jid] and len(db.messages) == 1
    assert run(server._deliver_job_notes(THREAD, USER)) is None


def test_n_jobs_finishing_while_turns_keep_coming_deliver_once_coalesced(db, history):
    """Five completions interleaved with three turns: no note lands inside a
    turn, every note is delivered exactly once, one message per boundary."""
    meta, _ = history
    meta[THREAD] = {"lc_msgs": [], "metadata": {}}
    delivered_batches = []
    finished = []
    for turn in range(3):
        jn.turn_started(THREAD)
        meta[THREAD]["lc_msgs"].append(HumanMessage(f"question {turn}"))
        for _ in range(2 if turn < 2 else 1):
            finished.append(_finish(db))
        assert run(server._deliver_job_notes(THREAD, USER)) is None     # mid-turn: nothing
        meta[THREAD]["lc_msgs"].append(AIMessage(f"answer {turn}"))
        jn.turn_finished(THREAD)
        batch = run(server._deliver_job_notes(THREAD, USER))
        assert batch is not None
        delivered_batches.append(batch)
    assert sorted(j for b in delivered_batches for j in b["job_ids"]) == sorted(finished)
    assert [b["count"] for b in delivered_batches] == [2, 2, 1]
    kinds = [type(m).__name__ + ("*" if jn.is_job_note(m) else "") for m in meta[THREAD]["lc_msgs"]]
    assert kinds == ["HumanMessage", "AIMessage", "AIMessage*"] * 3          # note always after the answer
    assert len(db.messages) == 3 and all(n["delivered_at"] for n in db.notes)


# ---- the route ---------------------------------------------------------------------------

def test_job_notes_route_requires_auth(monkeypatch):
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: None)
    with pytest.raises(server.HTTPException) as exc:
        run(server.get_thread_job_notes(object(), THREAD))
    assert exc.value.status_code == 401


def test_job_notes_route_is_owner_only(db, history, owner_session, monkeypatch):
    _finish(db)
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: OTHER)
    with pytest.raises(server.HTTPException) as exc:
        run(server.get_thread_job_notes(object(), THREAD))
    assert exc.value.status_code == 404 and db.notes[0]["delivered_at"] is None
    with pytest.raises(server.HTTPException) as exc:
        monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)
        run(server.get_thread_job_notes(object(), "thread-unknown"))
    assert exc.value.status_code == 404


def test_job_notes_route_delivers_or_defers(db, history, owner_session, monkeypatch):
    meta, _ = history
    meta[THREAD] = {"lc_msgs": [], "metadata": {}}
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)
    _finish(db)
    jn.turn_started(THREAD)
    out = run(server.get_thread_job_notes(object(), THREAD))
    assert out == {"delivered": None, "deferred": True, "pending": 1}
    jn.turn_finished(THREAD)
    out = run(server.get_thread_job_notes(object(), THREAD))
    assert out["deferred"] is False and out["delivered"]["count"] == 1 and out["pending"] == 0


# ---- history round trip and the history API ---------------------------------------------

def test_job_note_tag_survives_the_redis_round_trip():
    note = jn.make_note_message("✓ Data transfer finished.", ["j1", "j2"], "b1")
    items = helpers._lc_msgs_to_jsonable([HumanMessage("q"), AIMessage("a"), note])
    assert items[2]["job_note"] == {"job_ids": ["j1", "j2"], "batch_id": "b1"} and "job_note" not in items[1]
    back = helpers._jsonable_to_lc_msgs(json.loads(json.dumps(items)))
    assert jn.is_job_note(back[2]) and back[2].additional_kwargs["batch_id"] == "b1" and not jn.is_job_note(back[1])
    assert back[2].additional_kwargs[jn.SYSTEM_CONTEXT_FLAG] is True


def test_history_api_marks_the_note(history):
    meta, _ = history
    meta[THREAD] = {"lc_msgs": [AIMessage("a"), jn.make_note_message("✓ x finished.", ["j"], "b")], "metadata": {}}
    out = run(helpers.read_thread_msgs(THREAD))
    assert "job_note" not in out[0] and out[1]["job_note"]["job_ids"] == ["j"] and out[1]["role"] == "assistant"


# ---- the planner's and the coder's scans skip the note (ruling 5) --------------------------

def test_coder_last_instruction_skips_a_trailing_job_note():
    from app.agents import coder
    note = jn.make_note_message("✓ Data transfer finished.", ["j"], "b")
    msgs = [HumanMessage("sum sales by region"), AIMessage("plan"), note]
    kept = coder._without_job_notes(msgs)
    assert kept == msgs[:2]
    assert [m.content for m in reversed(kept)][0] == "plan"


def test_planner_last_ai_turn_skips_a_job_note():
    from app.agents import planner
    note = jn.make_note_message("✓ Data transfer finished.", ["j"], "b")
    assert planner._is_job_note(note) and not planner._is_job_note(AIMessage("1. apples 2. pears"))
    tail = [AIMessage("1. apples 2. pears"), note, HumanMessage("2")]
    last_ai = [m for m in tail if isinstance(m, AIMessage) and not planner._is_job_note(m)][-1:]
    assert last_ai[0].content == "1. apples 2. pears"
