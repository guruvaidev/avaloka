"""The swarm mission API: start a fleet, watch each clone live, read the RL view.

Driven over HTTP through the real server harness. The mission is the real
coordinator on the bundled churn sample with two clones, run on its daemon
thread while the test polls ``/records?after=`` the way the UI does.
"""

from __future__ import annotations

import json
import time

import pytest

from app.services import session_service
from avaloka.missions import analyze_fireflies
from app.trace import swarm_store
from telemetry_service import schema as ingest

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import SESSIONS, _do_upload, client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import SESSIONS, _do_upload, client, make_auth_headers  # noqa: F401

OWNER = make_auth_headers("swarm-owner")
OTHER = make_auth_headers("someone-else")


@pytest.fixture(autouse=True)
def _fresh_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_SWARM_OUTPUT_DIR", str(tmp_path / "swarm-out"))
    monkeypatch.delenv("AVALOKA_SWARM_ALLOW_PATHS", raising=False)
    swarm_store.registry.clear()
    yield
    swarm_store.registry.clear()


def _poll_to_completion(client, mission_id, headers, timeout_s=180.0):
    """Poll the live endpoint until done, the way the UI does, every tick."""
    after, records, deadline = 0, [], time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        page = client.get(f"/api/swarm/missions/{mission_id}/records?after={after}&limit=5",
                          headers=headers).json()
        assert page["mission_id"] == mission_id
        assert all(r["seq"] > after for r in page["records"])
        records.extend(page["records"])
        assert page["cursor"] == (records[-1]["seq"] if records else after)
        after = page["cursor"]
        if page["done"]:
            assert page["status"] in ("done", "failed")
            return page["status"], records
        time.sleep(0.05)
    pytest.fail("mission did not finish in time")


def _start(client, headers=OWNER, **body):
    payload = {"source": {"sample": "churn"}, "goal": "what drives churn", "kind": "analyze",
               "workers": 2, **body}
    return client.post("/api/swarm/missions", json=payload, headers=headers)


# ------------------------------------------------------------------- auth
def test_every_route_needs_a_user(client):
    assert client.post("/api/swarm/missions", json={"source": {"sample": "churn"}, "goal": "g"}).status_code == 401
    assert client.get("/api/swarm/missions").status_code == 401
    for suffix in ("", "/records", "/clones/ava-00", "/episodes", "/telemetry", "/understanding",
                   "/understanding/telemetry"):
        assert client.get(f"/api/swarm/missions/nope{suffix}").status_code == 401
    assert client.delete("/api/swarm/missions/nope").status_code == 401


def test_an_unknown_or_foreign_mission_is_404(client):
    for suffix in ("", "/records", "/clones/ava-00", "/episodes", "/telemetry", "/understanding",
                   "/understanding/telemetry"):
        assert client.get(f"/api/swarm/missions/nope{suffix}", headers=OWNER).status_code == 404
    assert client.delete("/api/swarm/missions/nope", headers=OWNER).status_code == 404
    started = _start(client).json()
    mid = started["mission_id"]
    for suffix in ("", "/records", "/clones/ava-00", "/episodes", "/telemetry", "/understanding",
                   "/understanding/telemetry"):
        assert client.get(f"/api/swarm/missions/{mid}{suffix}", headers=OTHER).status_code == 404
    assert client.delete(f"/api/swarm/missions/{mid}", headers=OTHER).status_code == 404
    assert client.get("/api/swarm/missions", headers=OTHER).json() == {"missions": []}
    _poll_to_completion(client, mid, OWNER)


# -------------------------------------------------------------- the demo
def test_a_two_clone_mission_is_served_live_then_whole(client):
    resp = _start(client)
    assert resp.status_code == 200, resp.text
    started = resp.json()
    mid = started["mission_id"]
    assert started["status"] == "running"
    assert started["grant"]["clones"] == 2 and started["grant"]["parallel"] == 2
    assert started["limits"]["max_clones_per_mission"] == started["grant"]["limits"]["max_clones_per_mission"]
    assert started["source"] == {"kind": "sample", "sample": "churn"}
    assert started["persistence"] == "redis"  # the harness has a (fake) cache

    listed = client.get("/api/swarm/missions", headers=OWNER).json()["missions"]
    assert [m["mission_id"] for m in listed] == [mid] and listed[0]["status"] in ("running", "done")

    status, records = _poll_to_completion(client, mid, OWNER)
    assert status == "done"
    assert [r["seq"] for r in records] == list(range(1, len(records) + 1))
    types = [r["type"] for r in records]
    assert types[:2] == ["swarm.requested", "swarm.sharded"] and types[-2:] == ["swarm.merged", "swarm.finished"]
    assert types.count("clone.dispatched") == types.count("clone.returned") == 2
    steps_per_clone = len(analyze_fireflies())
    assert types.count("clone.step.started") == types.count("clone.step.completed") == 2 * steps_per_clone
    # Polling past the end stays done and empty.
    tail = client.get(f"/api/swarm/missions/{mid}/records?after={records[-1]['seq']}", headers=OWNER).json()
    assert tail == {"mission_id": mid, "status": "done", "records": [], "cursor": records[-1]["seq"], "done": True}

    doc = client.get(f"/api/swarm/missions/{mid}", headers=OWNER).json()
    assert doc["status"] == "done" and doc["kind"] == "analyze" and doc["goal"] == "what drives churn"
    assert "records" not in doc
    assert [c["clone_id"] for c in doc["clones"]] == ["ava-00", "ava-01"]
    assert all(len(c["steps"]) == steps_per_clone and c["status"] == "done" for c in doc["clones"])
    assert doc["merge"]["n_rows"] == 1200 and doc["merge"]["verdict"] in {"pass", "warn", "fail"}
    assert doc["grant"]["granted_clones"] == 2 and doc["persistence"] == "redis"
    with_records = client.get(f"/api/swarm/missions/{mid}?records=1", headers=OWNER).json()
    assert [r["seq"] for r in with_records["records"]] == [r["seq"] for r in records]

    clone = client.get(f"/api/swarm/missions/{mid}/clones/ava-01", headers=OWNER).json()
    assert clone["clone_id"] == "ava-01" and clone["mission_id"] == mid
    assert [s["firefly"] for s in clone["steps"]][0] == "data_scout"
    assert clone["records"] and all(r["clone_id"] == "ava-01" for r in clone["records"])
    assert client.get(f"/api/swarm/missions/{mid}/clones/ava-09", headers=OWNER).status_code == 404

    eps = client.get(f"/api/swarm/missions/{mid}/episodes", headers=OWNER).json()
    assert eps["mission_id"] == mid and [e["episode_id"] for e in eps["episodes"]] == ["ava-00", "ava-01"]
    assert all(e["reward"] is None and len(e["steps"]) == steps_per_clone for e in eps["episodes"])
    assert eps["episodes"][0]["context"]["rows"] == 600

    tele = client.get(f"/api/swarm/missions/{mid}/telemetry", headers=OWNER).json()
    events = tele["events"]
    ingest.validate_batch(events)
    assert events[-1]["event_type"] == "swarm.completed" and events[-1]["clones_granted"] == 2
    assert sum(e["event_type"] == "agent.completed" for e in events) == 2 * steps_per_clone
    wire = json.dumps(events)
    for secret in (mid, "churn", "what drives churn", "Ava-00", "ava-00", "tenure"):
        assert secret not in wire, secret

    listed = client.get("/api/swarm/missions", headers=OWNER).json()["missions"]
    assert listed == [{"mission_id": mid, "status": "done", "kind": "analyze", "goal": "what drives churn",
                       "started_at": doc["started_at"], "ended_at": doc["ended_at"], "clones": 2,
                       "verdict": doc["merge"]["verdict"]}]

    # Finished: persisted to the cache under its key, indexed for its user ...
    assert session_service.cache._data.get(f"swarm:{mid}")
    assert mid in session_service.cache._sets.get("user:swarm-owner:swarms", set())
    # ... and served from there once this process has forgotten it.
    swarm_store.registry.clear()
    again = client.get(f"/api/swarm/missions/{mid}", headers=OWNER).json()
    assert again["status"] == "done" and again["persistence"] == "redis" and "records" not in again
    page = client.get(f"/api/swarm/missions/{mid}/records?after=0&limit=3", headers=OWNER).json()
    assert [r["seq"] for r in page["records"]] == [1, 2, 3] and page["done"] is False
    assert client.get("/api/swarm/missions", headers=OWNER).json()["missions"][0]["mission_id"] == mid

    assert client.delete(f"/api/swarm/missions/{mid}", headers=OWNER).json() == {"mission_id": mid, "deleted": True}
    assert client.get(f"/api/swarm/missions/{mid}", headers=OWNER).status_code == 404
    assert client.get("/api/swarm/missions", headers=OWNER).json() == {"missions": []}


def test_it_works_with_no_redis_at_all(client, monkeypatch):
    monkeypatch.setattr(session_service, "cache", None)
    started = _start(client).json()
    assert started["persistence"] == "memory"
    mid = started["mission_id"]
    status, records = _poll_to_completion(client, mid, OWNER)
    assert status == "done" and records[-1]["type"] == "swarm.finished"
    doc = client.get(f"/api/swarm/missions/{mid}", headers=OWNER).json()
    assert doc["persistence"] == "memory" and "lost when the server restarts" in doc["persistence_note"]
    assert len(doc["clones"]) == 2
    ingest.validate_batch(client.get(f"/api/swarm/missions/{mid}/telemetry", headers=OWNER).json()["events"])
    assert client.delete(f"/api/swarm/missions/{mid}", headers=OWNER).status_code == 200
    assert client.get(f"/api/swarm/missions/{mid}", headers=OWNER).status_code == 404


# ------------------------------------------------------------ the request
def test_a_bad_request_is_400_and_starts_nothing(client):
    assert _start(client, source={"sample": "nope"}).status_code == 400
    assert _start(client, source={"sample": "churn", "path": "/x"}).status_code == 400
    assert _start(client, source={"bucket": "x"}).status_code == 400
    assert _start(client, kind="deploy").status_code == 400
    assert _start(client, kind="train").status_code == 400  # no target
    assert _start(client, goal="  ").status_code == 400
    assert _start(client, workers=0).status_code == 400
    assert _start(client, max_parallel=0).status_code == 400
    assert client.post("/api/swarm/missions", json={"goal": "g"}, headers=OWNER).status_code == 422
    assert client.get("/api/swarm/missions", headers=OWNER).json() == {"missions": []}
    assert swarm_store.registry.for_user("swarm-owner") == []


def test_a_server_path_is_refused_unless_explicitly_allowed(client, monkeypatch, tmp_path):
    resp = _start(client, source={"path": "/etc/passwd"})
    assert resp.status_code == 400 and "AVALOKA_SWARM_ALLOW_PATHS" in resp.json()["detail"]
    monkeypatch.setenv("AVALOKA_SWARM_ALLOW_PATHS", "1")
    resp = _start(client, source={"path": str(tmp_path / "missing.csv")})
    assert resp.status_code == 400 and "not a readable file" in resp.json()["detail"]
    assert swarm_store.registry.for_user("swarm-owner") == []


def test_a_thread_resolves_to_its_dataset_for_its_owner_only(client):
    upload = _do_upload(client, user_id="swarm-owner")
    thread_id = upload["thread_id"]
    sess = next(s for s in SESSIONS.values() if s.get("user_id") == "swarm-owner")
    assert sess.get("work_local_input")

    assert _start(client, OTHER, source={"thread_id": thread_id}).status_code == 400
    resp = _start(client, source={"thread_id": thread_id}, workers=1)
    assert resp.status_code == 200, resp.text
    started = resp.json()
    assert started["source"] == {"kind": "thread", "thread_id": thread_id}
    status, records = _poll_to_completion(client, started["mission_id"], OWNER)
    sharded = next(r for r in records if r["type"] == "swarm.sharded")
    # The upload in the harness is the two-row a,b file: that is what was sharded.
    assert sharded["total_rows"] == 2 and sharded["columns"] == ["a", "b"]


def test_the_edition_cap_is_announced_not_silently_applied(client, monkeypatch):
    """The swarm still runs on oss/free, two clones wide, and says so.

    The HTTP route is the console and is closed on oss (403, see
    tests/test_swarm_console_gate.py), so the open-source proof drives the
    service the chat path uses, with the edition the route would have passed.
    """
    import asyncio

    from app.trace import swarm_service
    from avaloka.swarm_limits import limits_for

    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    assert _start(client, workers=7).status_code == 403

    cap = limits_for("oss").max_clones_per_mission
    assert cap == 2

    async def _no_thread(_tid):
        return None

    started = asyncio.run(swarm_service.start_mission(
        "swarm-owner", source={"sample": "churn"}, goal="what drives churn", kind="analyze",
        workers=cap + 5, edition="oss", thread_resolver=_no_thread))
    assert started["grant"]["limited"] is True and started["grant"]["clones"] == cap
    assert started["grant"]["requested_clones"] == cap + 5 and "limited to" in started["grant"]["message"]
    entry = swarm_store.registry.get(started["mission_id"])
    entry.thread.join(timeout=180)
    assert entry.finished and entry.status == "done"
    records = entry.recorder.records
    assert records[0]["granted_clones"] == cap and records[0]["limited"] is True
    assert [r["type"] for r in records].count("clone.dispatched") == cap


def test_a_mission_without_workers_is_sized_from_the_data(client):
    """No ``workers``: the fleet is derived from the source (2.0) and the
    response says how; the harness has no commercial build, so the edition
    runs its preview of two whatever the data says."""
    resp = client.post("/api/swarm/missions", headers=OWNER,
                       json={"source": {"sample": "churn"}, "goal": "what drives churn"})
    assert resp.status_code == 200, resp.text
    started = resp.json()
    grant = started["grant"]
    assert grant["clones"] == 2 and grant["requested_clones"] >= 2
    assert "rows" in grant["derived_from"] and "ceil(rows/25,000)" in grant["derived_from"]
    assert "churn" not in grant["derived_from"]
    status, records = _poll_to_completion(client, started["mission_id"], OWNER)
    assert status == "done"
    assert records[0]["type"] == "swarm.requested" and records[0]["derived_from"] == grant["derived_from"]
    doc = client.get(f"/api/swarm/missions/{started['mission_id']}", headers=OWNER).json()
    assert doc["grant"]["derived_from"] == grant["derived_from"]
    assert "derived_from" not in json.dumps(
        client.get(f"/api/swarm/missions/{started['mission_id']}/telemetry", headers=OWNER).json())


# ---------------------------------------------------------- understanding
def test_understanding_is_served_live_and_its_projection_is_metadata_only(client):
    """The two-layer routes: ``/understanding`` pages the full-fidelity
    records like ``/records`` does; ``/understanding/telemetry`` is the
    projection the ingest accepts, with nothing of the caller's data in it."""
    from avaloka.understanding_trace import AGENT_RECORD, SWARM_RECORD

    mid = _start(client).json()["mission_id"]
    after, pages, deadline = 0, [], time.monotonic() + 180.0
    while time.monotonic() < deadline:
        page = client.get(f"/api/swarm/missions/{mid}/understanding?after={after}&limit=4", headers=OWNER).json()
        assert page["mission_id"] == mid
        assert all(r["type"] in (AGENT_RECORD, SWARM_RECORD) and r["seq"] > after for r in page["records"])
        pages.extend(page["records"])
        assert page["cursor"] == (pages[-1]["seq"] if pages else after)
        after = page["cursor"]
        if page["done"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("mission did not finish in time")
    assert page["status"] == "done"
    from avaloka.missions import analyze_fireflies

    fireflies = [cls.name for cls in analyze_fireflies()]
    assert [r["type"] for r in pages] == [AGENT_RECORD] * (2 * len(fireflies)) + [SWARM_RECORD]
    # Clones run in parallel, so their steps interleave; each clone's own order is the firefly order.
    for cid in ("ava-00", "ava-01"):
        assert [r["agent"] for r in pages if r.get("clone_id") == cid] == fireflies
    scout = next(r for r in pages if r.get("clone_id") == "ava-00")
    assert scout["data"]["rows"] == 600 and "tenure_months" in scout["data"]["column_names"]  # the caller's own layer
    assert scout["concluded"].startswith("Profiled")
    fleet = pages[-1]
    assert fleet["clones"] == 2 and fleet["steps_total"] == 2 * len(fireflies) and fleet["data"]["rows"] == 1200
    # The same records are in the plain records stream, with the same seqs.
    _, records = _poll_to_completion(client, mid, OWNER)
    assert [r["seq"] for r in pages] == [r["seq"] for r in records if r["type"] in (AGENT_RECORD, SWARM_RECORD)]
    # Polling past the end stays done and empty.
    tail = client.get(f"/api/swarm/missions/{mid}/understanding?after={pages[-1]['seq']}", headers=OWNER).json()
    assert tail == {"mission_id": mid, "status": "done", "records": [], "cursor": pages[-1]["seq"], "done": True}
    # The mission document carries the fleet's record.
    doc = client.get(f"/api/swarm/missions/{mid}", headers=OWNER).json()
    assert doc["understanding"]["steps_total"] == 2 * len(fireflies)
    assert doc["understanding"]["merge"]["verdict"] == doc["merge"]["verdict"]

    tele = client.get(f"/api/swarm/missions/{mid}/understanding/telemetry", headers=OWNER).json()
    events = tele["events"]
    ingest.validate_batch(events)
    assert [e["event_type"] for e in events] == [AGENT_RECORD] * (2 * len(fireflies)) + [SWARM_RECORD]
    assert events[0]["row_count_bucket"] == "0-1e3" and events[-1]["row_count_bucket"] == "1e3-1e4"
    assert events[-1]["clones"] == 2 and events[-1]["steps_per_clone_min"] == len(fireflies)
    wire = json.dumps(events)
    for secret in (mid, "churn", "what drives churn", "Ava-00", "ava-00", "tenure", "Profiled"):
        assert secret not in wire, secret

    # Served from the store once this process has forgotten it, same contract.
    swarm_store.registry.clear()
    page = client.get(f"/api/swarm/missions/{mid}/understanding?after=0&limit=5", headers=OWNER).json()
    assert [r["seq"] for r in page["records"]] == [r["seq"] for r in pages[:5]] and page["done"] is False
    assert len(client.get(f"/api/swarm/missions/{mid}/understanding/telemetry", headers=OWNER).json()["events"]) \
        == 2 * len(fireflies) + 1
    assert client.get(f"/api/swarm/missions/{mid}/understanding", headers=OTHER).status_code == 404
    assert client.get(f"/api/swarm/missions/{mid}/understanding/telemetry", headers=OTHER).status_code == 404


def test_understanding_routes_are_a_commercial_feature(client, monkeypatch):
    """The swarm console is professional/enterprise: in the oss edition the
    routes answer 403 with a reason, not 404 and not a page. The mission still
    runs and its local understanding.jsonl is still written."""
    mid = _start(client).json()["mission_id"]
    _poll_to_completion(client, mid, OWNER)
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    for suffix in ("/understanding", "/understanding/telemetry"):
        resp = client.get(f"/api/swarm/missions/{mid}{suffix}", headers=OWNER)
        assert resp.status_code == 403, resp.text
        assert resp.json()["detail"]["code"] == "swarm_console_unavailable"
        assert resp.json()["detail"]["reason"] == "commercial"
        assert client.get(f"/api/swarm/missions/{mid}{suffix}").status_code == 401  # no user: still 401
    monkeypatch.delenv("AVALOKA_EDITION")
    assert client.get(f"/api/swarm/missions/{mid}/understanding", headers=OWNER).status_code == 200
