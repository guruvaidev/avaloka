"""A mid-wizard reply that is itself a transfer request supersedes the wizard.

The wizard interceptor routed EVERY reply into the awaited slot. Awaiting
new_table_name and answering "transfer to prod instead" therefore created a table
literally named "transfer" in the ORIGINAL destination — silently, reported as
success. A reply that parses as a full transfer phrasing now clears the pending
slot and re-enters normal routing; bare slot answers (a table name, a filename)
still fill the slot as before.
"""
from unittest import mock

from langchain_core.messages import HumanMessage

from app.agents.planner import plan_etl_job


def _state(reply, awaiting="new_table_name"):
    return {
        "messages": [HumanMessage(content=reply)],
        "user_prompt": "",
        "dta_database_registry": {},
        "pending_clarification": {
            "type": "dta_transfer",
            "awaiting": awaiting,
            "collected": {"destination_alias": "old_dest", "user_prompt": "transfer to old_dest"},
            "source_agent": "dta",
        },
    }


def test_new_transfer_request_supersedes_the_pending_slot():
    state = _state("transfer to prod_db into table orders")
    with mock.patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         mock.patch("app.agents.planner._seed_ray_fields"), \
         mock.patch("app.agents.planner._seed_infra_request_if_needed"):
        out = plan_etl_job(state)

    m.assert_called_once()
    _, params = m.call_args[0]
    # Routed as a FRESH transfer to the new destination — not force-fitted into the
    # old wizard's table-name slot.
    assert params.destination_alias == "prod_db"
    assert params.dest_table == "orders"
    assert out.get("pending_clarification") in (None, {}) or \
        out["pending_clarification"].get("collected", {}).get("destination_alias") != "old_dest"


def test_bare_table_name_still_fills_the_awaited_slot():
    state = _state("customer_summary")
    with mock.patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         mock.patch("app.agents.planner._seed_ray_fields"), \
         mock.patch("app.agents.planner._seed_infra_request_if_needed"):
        plan_etl_job(state)

    m.assert_called_once()
    _, params = m.call_args[0]
    # The wizard resumed: the answer became the new table on the ORIGINAL transfer.
    assert params.destination_alias == "old_dest"
    assert params.dest_table == "customer_summary"
    assert params.create_if_missing is True
