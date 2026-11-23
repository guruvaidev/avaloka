"""A failed transfer must not wedge the thread.

rc3 web re-check (e2ewhgemma): a transfer asked "Which table … should I
transfer into?", the reply "new table patients_copy" failed for want of a Ray
executor, and from then on every question on the thread -- the stroke
question included -- got "⚠️ Table `What` doesn't exist in `e2ewhgemma`".
Two things held it there:

1. The turn that resolved the transfer slot popped ``pending_clarification``
   from the session, but the save merges with ``dict.update``, which never
   deletes, so the slot was read back on the next turn
   (tests/test_server_session_merge.py pins that half).
2. The wizard took any reply as the slot's answer: "What is the stroke rate…"
   became a table named "What", and its "doesn't exist" reply re-armed the
   slot for the next question.
"""
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

import app.agents.planner as planner

_STROKE = "What is the stroke rate for patients with hypertension versus without?"

_SRC = {"db_type": "postgresql", "host": "h", "port": 5432, "database": "d",
        "user": "u", "password": "p", "table": "patients"}
_DST = {"db_type": "postgresql", "host": "h2", "port": 5432, "database": "d2",
        "user": "u", "password": "p"}


@pytest.fixture(autouse=True)
def _no_ray_no_network(monkeypatch):
    monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setattr(planner, "_ray_executor_configured", lambda *h: False)
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type: ["orders", "patients_copy2"])
    monkeypatch.setattr(planner, "_db_api_key_for", lambda state, alias: None)


def _state(pending=None, reply=""):
    return {
        "messages": [HumanMessage(content=reply)] if reply else [],
        "user_prompt": "",
        "user_id": "u1",
        "active_db_customer_id": "e2eclinicgemma",
        "dta_database_registry": {"e2eclinicgemma": dict(_SRC), "e2ewhgemma": dict(_DST)},
        "pending_clarification": pending,
    }


def _slot(awaiting, **collected):
    return {
        "type": "dta_transfer",
        "awaiting": awaiting,
        "collected": {"destination_alias": "e2ewhgemma",
                      "user_prompt": "Transfer the patients table to e2ewhgemma as patients_copy.",
                      **collected},
        "source_agent": "dta",
    }


def _plan(state):
    with mock.patch.object(planner, "_seed_ray_fields"), \
         mock.patch.object(planner, "_seed_infra_request_if_needed"):
        return planner.plan_etl_job(state)


class _ReachedNormalPlanning(BaseException):
    """Raised where the planner starts on an ordinary question.

    A BaseException, because importing app.api.server wraps plan_etl_job in a
    guard that turns any Exception into "the planner failed"."""


def test_failed_transfer_clears_its_slot_and_the_next_question_is_answered(monkeypatch):
    # Turn 1: the table-name slot is answered and the transfer fails (no Ray,
    # and not a plain copy, so the local runner refuses it too).
    state = _state(_slot("table_choice"), "new table patients_copy")
    monkeypatch.setattr(planner, "_transfer_without_ray",
                        lambda *a, **k: "❌ This transfer needs the Ray executor, which isn't configured here.")
    out = _plan(state)
    # Every transfer stops at one confirmation first (J1); the user says yes.
    assert out["pending_clarification"]["awaiting"] == "confirm_transfer"
    out["messages"] = list(out["messages"]) + [HumanMessage(content="yes")]
    out = _plan(out)
    assert "Ray executor" in out["messages"][-1].content
    assert out.get("pending_clarification") is None, "a failed transfer must clear its pending slot"

    # Turn 2: the stroke question reaches ordinary planning, not the wizard.
    calls = []
    monkeypatch.setattr(planner, "_resolve_pending_dta",
                        lambda *a, **k: calls.append(a) or "wizard")
    monkeypatch.setattr(planner, "_resolve_pending_clarification",
                        mock.Mock(side_effect=_ReachedNormalPlanning))
    with pytest.raises(_ReachedNormalPlanning):
        _plan(_state(out.get("pending_clarification"), _STROKE))
    assert not calls


@pytest.mark.parametrize("awaiting", ["table_choice", "confirm_create_table", "new_table_name", "destination"])
def test_a_question_left_in_an_open_slot_is_not_read_as_its_answer(monkeypatch, awaiting):
    """Even if a stale slot is read back, a question is answered, not parsed as a table."""
    calls = []
    monkeypatch.setattr(planner, "_handle_initiate_transfer",
                        lambda *a, **k: calls.append(a) or "⚠️ Table `What` doesn't exist in `e2ewhgemma`.")
    monkeypatch.setattr(planner, "_resolve_pending_clarification",
                        mock.Mock(side_effect=_ReachedNormalPlanning))
    state = _state(_slot(awaiting, dest_table="patients_copy"), _STROKE)
    with pytest.raises(_ReachedNormalPlanning):
        _plan(state)
    assert not calls, "the question was parsed as a transfer"
    assert state.get("pending_clarification") is None


@pytest.mark.parametrize("reply", ["orders", "new table patients_copy", "patients_copy2"])
def test_a_real_slot_answer_still_fills_the_slot(monkeypatch, reply):
    seen = []
    monkeypatch.setattr(planner, "_handle_initiate_transfer",
                        lambda state, params: seen.append(params) or "HANDLED")
    out = _plan(_state(_slot("table_choice"), reply))
    assert seen, f"{reply!r} should have answered the table slot"
    assert out["messages"][-1].content == "HANDLED"


def test_a_bare_create_asks_for_the_name_and_is_never_the_name(monkeypatch):
    """rc8: "create" was taken as a table literally named `create`."""
    seen = []
    monkeypatch.setattr(planner, "_handle_initiate_transfer",
                        lambda state, params: seen.append(params) or "HANDLED")
    out = _plan(_state(_slot("table_choice"), "create"))
    assert not seen
    assert out["messages"][-1].content.endswith("What should the new table be called?")
    assert out["pending_clarification"]["awaiting"] == "new_table_name"


def test_an_unknown_table_name_is_checked_and_does_not_re_arm(monkeypatch):
    """A name not in the catalog gets one specific reply, then the slot closes."""
    monkeypatch.setattr(planner, "_transfer_without_ray", lambda *a, **k: "RAN")
    state = _state(_slot("table_choice"))
    first = planner._resolve_pending_dta(state, state["pending_clarification"], "patients_cpy")
    assert "`patients_cpy` doesn't exist in `e2ewhgemma`" in first
    assert state["pending_clarification"]["awaiting"] == "confirm_create_table"

    second = planner._resolve_pending_dta(state, state["pending_clarification"], "patiens")
    assert "`patiens` isn't a table in `e2ewhgemma` either" in second
    assert state.get("pending_clarification") is None, "a second miss must not re-arm the slot"


def test_a_catalog_table_is_used_without_a_create_offer(monkeypatch):
    ran = []
    monkeypatch.setattr(planner, "_transfer_without_ray",
                        lambda state, params, *a, **k: ran.append(params.dest_table) or "RAN")
    state = _state(_slot("table_choice"))
    msg = planner._resolve_pending_dta(state, state["pending_clarification"], "patients_copy2")
    assert state["pending_clarification"]["awaiting"] == "confirm_transfer", msg
    msg = planner._resolve_pending_dta(state, state["pending_clarification"], "yes")
    assert msg == "RAN" and ran == ["patients_copy2"]
    assert state.get("pending_clarification") is None


# Registry credentials these tests seed live in a stand-in sealed store, not in
# state: the planner opens them by handle after the confirm gate (tests/dta_sealed.py).
try:  # tests/ as a package, or tests/ on sys.path
    from tests import dta_sealed as _dta_sealed
except ImportError:  # pragma: no cover
    import dta_sealed as _dta_sealed


@pytest.fixture(autouse=True)
def _sealed_store(monkeypatch):
    return _dta_sealed.install(monkeypatch)
