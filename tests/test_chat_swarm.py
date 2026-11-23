"""The in-chat swarm: a real mission's records, told back by the narrator.

The mission here is real: avaloka.coordinator runs two clones on the bundled
churn sample through app.trace.swarm_service, exactly as the chat path does,
and every assertion about the summary is against those records. The HTTP
tests drive /threads/{id}/messages through the server harness.
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import chat_swarm
from app.trace import swarm_store

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import SESSIONS, _do_upload, client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import SESSIONS, _do_upload, client, make_auth_headers  # noqa: F401


def _join_all():
    """No mission thread outlives its test: a daemon thread still writing its
    trace races the harness's dicts in the next test's setup."""
    for user in ("user-1", "voice-user", "u"):
        for entry in swarm_store.registry.for_user(user):
            if entry.thread is not None:
                entry.thread.join(timeout=180)


@pytest.fixture(autouse=True)
def _fresh_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_SWARM_OUTPUT_DIR", str(tmp_path / "swarm-out"))
    swarm_store.registry.clear()
    yield
    _join_all()
    swarm_store.registry.clear()


@pytest.fixture(scope="module")
def recorded_mission():
    """One real two-clone mission on churn.csv, recorded once for the module."""
    from app.trace import swarm_service

    swarm_store.registry.clear()
    mission = asyncio.run(chat_swarm.run_mission(
        "voice-user", dataset_path=str(swarm_service.sample_path("churn")),
        goal="what drives churn", edition="free", timeout_s=180))
    assert mission["finished"] and mission["status"] == "done", mission["status"]
    swarm_store.registry.clear()
    return mission


# ---------------------------------------------------------------- the trigger
@pytest.mark.parametrize("text", [
    "swarm this: what drives churn", "Please clone yourself and look at this", "send your team at it",
    "fan out across the data", "Swarm this: What is the average Billing Amount by Admission Type",
])
def test_the_words_that_summon_the_team(text):
    assert chat_swarm.wants_swarm(text)


@pytest.mark.parametrize("text", ["what is the average billing amount", "show me the team column", "hi"])
def test_ordinary_questions_do_not(text):
    assert not chat_swarm.wants_swarm(text)
    assert chat_swarm.wants_swarm(text, {"swarm": True})


@pytest.mark.parametrize("text", ["Check on the swarm", "how's the team doing?", "swarm status"])
def test_asking_after_the_team_is_not_asking_for_another(text):
    assert chat_swarm.wants_swarm(text) and chat_swarm.is_status_ask(text)
    assert not chat_swarm.is_status_ask("Swarm this: what drives churn")


def test_the_goal_is_the_users_words_without_the_request():
    assert chat_swarm.mission_goal("Swarm this: what drives churn?") == "what drives churn"
    assert chat_swarm.mission_goal("send your team at it") == "understand this dataset"
    assert chat_swarm.mission_goal("") == "understand this dataset"


# ---------------------------------------------------------------- the story
def test_the_summary_is_the_narrators_team_story_from_the_real_records(recorded_mission):
    records = recorded_mission["records"]
    types = [r["type"] for r in records]
    assert types[0] == "swarm.requested" and types[-1] == "swarm.finished"
    prompt = chat_swarm.summary_prompt(records, edition="free", mission_id=recorded_mission["mission_id"])
    assert prompt["step"] == "swarm"
    text = prompt["message"]
    # Team size, shards, steps, verdict, merge: each from a record, none canned.
    assert "Team size: 2 clones" in text
    sharded = next(r for r in records if r["type"] == "swarm.sharded")
    assert f"{sharded['total_rows']:,} rows" in text and "2 shards" in text
    returned = [r for r in records if r["type"] == "clone.returned"]
    assert all(f"{r['steps']} steps done" in text for r in returned)
    merged = next(r for r in records if r["type"] == "swarm.merged")
    assert str(merged["verdict"]).upper() in text and "merged the 2 results" in text
    assert "{" not in text and "}" not in text  # no JSON
    assert prompt["question"].endswith("?") and prompt["quick_replies"]
    # The lines are exactly the narrator's -v lines, in order.
    from avaloka.swarm_narration import VERBOSE, narrate
    assert [l.lstrip("- ") for l in text.splitlines()[1:] if l.startswith("- ")] == narrate(records, VERBOSE)


def test_the_brief_names_the_users_file_not_the_servers_working_copy(recorded_mission):
    records = recorded_mission["records"]
    bare = chat_swarm.summary_prompt(records, edition="free", mission_id="m")["message"]
    assert "churn.csv" in bare  # the path the recorder kept, unlabelled
    labelled = chat_swarm.summary_prompt(records, edition="free", mission_id="m",
                                         dataset_label="healthcare_dataset.csv")["message"]
    assert "analyse healthcare_dataset.csv" in labelled and "/" not in labelled.splitlines()[1]
    # Only the brief changed; every other line is still the narrator's.
    assert bare.splitlines()[2:] == labelled.splitlines()[2:]


def test_free_does_not_mention_the_page_and_enterprise_links_to_it(recorded_mission):
    records, mid = recorded_mission["records"], recorded_mission["mission_id"]
    for edition in ("oss", "free"):
        p = chat_swarm.summary_prompt(records, edition=edition, mission_id=mid)
        assert "link" not in p and "/swarm" not in p["message"] and "swarm page" not in p["message"]
    for edition in ("professional", "enterprise"):
        p = chat_swarm.summary_prompt(records, edition=edition, mission_id=mid)
        assert p["link"] == {"label": "Watch it on the swarm page", "href": f"/swarm?mission={mid}"}
        assert "swarm page" in p["message"]


def test_an_unfinished_story_says_so_and_offers_to_check_again(recorded_mission):
    partial = recorded_mission["records"][:4]  # requested, sharded, two dispatches
    p = chat_swarm.summary_prompt(partial, edition="free", mission_id="m", finished=False)
    assert p["message"].startswith("My team is still out")
    assert p["quick_replies"][0]["send"] == "Check on the swarm" and chat_swarm.wants_swarm("Check on the swarm")


def test_the_session_remembers_every_mission_until_its_end_was_told():
    sess = {}
    assert chat_swarm.pending_mission(sess) is None and chat_swarm.pending_missions(sess) == []
    chat_swarm.remember(sess, "m1", told=False)
    chat_swarm.remember(sess, "m2", told=False)
    assert chat_swarm.pending_missions(sess) == ["m1", "m2"] and chat_swarm.pending_mission(sess) == "m2"
    chat_swarm.remember(sess, "m1", told=True)
    assert chat_swarm.pending_missions(sess) == ["m2"]
    chat_swarm.remember(sess, "m2", told=True)
    assert chat_swarm.pending_missions(sess) == [] and sess[chat_swarm.SESSION_KEY]["told"] == ["m1", "m2"]
    # A strange value in the session is read as nothing owed, never as an error.
    assert chat_swarm.pending_missions({chat_swarm.SESSION_KEY: "junk"}) == []


# ----------------------------------------------------------------- over HTTP
def _chat(client, thread_id, text, user="user-1"):
    return client.post(f"/threads/{thread_id}/messages", json={"content": text},
                       headers=make_auth_headers(user))


def test_a_chat_turn_that_asks_for_the_team_runs_a_real_mission_and_tells_it(client, monkeypatch):
    """Free edition: the summary arrives in the chat, two clones wide, no page."""
    import app.api.server as server

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "free")
    up = _do_upload(client)
    resp = _chat(client, up["thread_id"], "Swarm this: what drives b?")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    prompts = body["assistant_prompts"]
    assert len(prompts) == 1 and prompts[0]["step"] == "swarm"
    text = body["messages"][-1]["content"]
    assert "Team size: 2 clones" in text and "the free" in text.lower()  # the edition's cap, announced
    # The brief names the upload (its alias, "test"), never the server's working-copy path.
    assert "analyse test to answer" in text and "/tmp" not in text and "/w/" not in text
    assert "/swarm" not in text and "link" not in prompts[0]
    # The mission is a real one in the store, started for this user.
    missions = swarm_store.registry.for_user("user-1")
    assert len(missions) == 1 and missions[0].finished
    assert [r["type"] for r in missions[0].recorder.records].count("clone.dispatched") == 2
    _drain(client)


def test_the_same_turn_on_enterprise_links_the_live_page(client, monkeypatch):
    import app.api.server as server

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "enterprise")
    up = _do_upload(client)
    body = _chat(client, up["thread_id"], "send your team at it").json()
    prompt = body["assistant_prompts"][0]
    mid = swarm_store.registry.for_user("user-1")[0].mission_id
    assert prompt["link"]["href"] == f"/swarm?mission={mid}"
    assert "swarm page" in body["messages"][-1]["content"]
    _drain(client)


def _wait_finished(client, mission_id, timeout_s=180.0):
    entry = swarm_store.registry.get(mission_id)
    assert entry is not None and entry.thread is not None
    entry.thread.join(timeout=timeout_s)
    assert entry.finished
    _drain(client)


def _drain(client):
    """Let the mission thread's scheduled persist run inside a request, not
    during the harness teardown (where it raced the fake cache's dicts)."""
    client.get("/version")


def test_a_mission_started_from_the_live_page_is_told_on_the_next_chat_turn(client, monkeypatch):
    """Any path: the /swarm page (professional) starts a mission on the thread's
    dataset; the next plain chat turn carries the team's story, with the link."""
    import app.api.server as server

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "professional")
    up = _do_upload(client)
    started = client.post("/api/swarm/missions", json={"source": {"thread_id": up["thread_id"]}, "goal": "g",
                                                       "workers": 2}, headers=make_auth_headers("user-1"))
    assert started.status_code == 200, started.text
    mid = started.json()["mission_id"]
    _wait_finished(client, mid)
    body = _chat(client, up["thread_id"], "hello there").json()
    stories = [p for p in (body.get("assistant_prompts") or []) if p["step"] == "swarm"]
    assert len(stories) == 1 and "Team size: 2 clones" in stories[0]["message"]
    assert stories[0]["link"]["href"] == f"/swarm?mission={mid}"
    # Told once: the next turn does not repeat it.
    again = _chat(client, up["thread_id"], "and again").json()
    assert not [p for p in (again.get("assistant_prompts") or []) if p["step"] == "swarm"]
    _drain(client)


def test_a_mission_noted_by_any_other_path_is_told_on_free_without_a_link(client, monkeypatch):
    """The planner's own fan-out, or anything else, notes the mission through
    server._note_thread_mission and the Free chat tells it, page unmentioned."""
    import asyncio

    import app.api.server as server
    from app.trace import swarm_service

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "free")
    up = _do_upload(client)

    async def _start_and_note():
        started = await swarm_service.start_mission(
            "user-1", source={"sample": "churn"}, goal="what drives churn", edition="free",
            thread_resolver=lambda tid: None)
        assert await server._note_thread_mission(up["thread_id"], started["mission_id"])
        return started["mission_id"]

    mid = asyncio.run(_start_and_note())
    _wait_finished(client, mid)
    body = _chat(client, up["thread_id"], "what is b on average?").json()
    stories = [p for p in (body.get("assistant_prompts") or []) if p["step"] == "swarm"]
    assert len(stories) == 1 and "merged the 2 results" in stories[0]["message"]
    assert "link" not in stories[0] and "/swarm" not in stories[0]["message"]
    # The analysis answer itself is still the assistant message, not the story.
    assert "Team size" not in body["messages"][-1]["content"]
    _drain(client)
