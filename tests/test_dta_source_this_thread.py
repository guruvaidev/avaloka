"""J1 follow-up: a transfer's implicit source comes from THIS thread only.

When the thread's session had no database, the send_message bridge used to
scan the user's other sessions and take their most-recent active database as
the transfer source: a silent default on a write path, the same class as the
ghost copy. It must leave the source empty, so the planner asks.
"""
from fastapi.testclient import TestClient

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        FakeGraph, make_auth_headers, client, _do_upload, _quiet_send,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        FakeGraph, make_auth_headers, client, _do_upload, _quiet_send,
    )


def test_b_a_database_from_another_thread_is_never_the_source(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    asked = []

    async def other_threads_db(user_id):
        asked.append(user_id)
        return ("other_thread_db", "patients")

    # The user analysed a database in another thread; this thread holds a CSV.
    monkeypatch.setattr(server, "find_active_db_customer_for_user", other_threads_db, raising=False)
    seen = {}

    class Capture(FakeGraph):
        def invoke(self, state_in, config=None):
            seen.update(state_in)
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", Capture(), raising=False)
    upload = _do_upload(client, user_id="thread-user")
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": "Transfer the patients table to warehouse as ghost_copy.",
              "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers("thread-user"),
    )
    assert resp.status_code == 200, resp.text
    assert seen, "the graph was not invoked"
    assert seen.get("active_db_customer_id") is None
    assert seen.get("active_db_source_table") is None
    assert not asked, "the other sessions were scanned for a source"
