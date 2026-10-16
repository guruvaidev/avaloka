"""Adding datasets to an existing chat ("+ Add" under the chat box).

Covers app/api/server.py:
  * _merge_group_index: adds to a chat's dataset group without dropping the
    datasets already there (older single-dataset groups included).
  * POST /threads/{id}/datasets: attach already-uploaded datasets; owner only.
  * POST /threads/{id}/datasets with analysis_ids: the "+ Add" picker lists
    analyses; every dataset of a picked analysis joins the chat.
  * GET  /threads/{id}/datasets: the chips for a chat.

Sessions live in a dict instead of Redis. server.py is imported once; if it
cannot be imported here the file is skipped with the reason.

    pytest tests/test_thread_datasets.py -q
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"app.api.server could not be imported here: {exc}", allow_module_level=True)

USER = "auth-uid-1111"
OTHER = "auth-uid-2222"


def run(coro):
    return asyncio.run(coro)


class _Req:
    headers: dict = {}
    cookies: dict = {}


@pytest.fixture
def store(monkeypatch):
    """A chat (thread t1) whose group g1 holds two datasets, plus two loose datasets."""
    sessions = {
        "g1": {"user_id": USER, "dataset_id": "dsA", "group_session_id": "g1",
               "dataset_ids": ["dsA", "dsB"], "dataset_session_map": {"dsA": "g1", "dsB": "sB"},
               "filename": "a.csv", "alias": "a"},
        "sB": {"user_id": USER, "dataset_id": "dsB", "group_session_id": "g1",
               "filename": "b.csv", "alias": "b"},
        "sC": {"user_id": USER, "dataset_id": "dsC", "group_session_id": "sC",
               "filename": "c.csv", "alias": "c"},
        "sX": {"user_id": OTHER, "dataset_id": "dsX", "group_session_id": "sX",
               "filename": "x.csv", "alias": "x"},
    }
    threads = {"t1": "g1", "tX": "sX"}
    saved = []

    async def get_session(sid, *a, **k):
        s = sessions.get(sid)
        return dict(s) if s else None

    async def save_session(sid, data):
        sessions[sid] = dict(data)
        saved.append(sid)
        return True

    async def get_thread_session(tid):
        return threads.get(tid)

    async def find_session_by_dataset_for_user(dsid, uid):
        for sid, s in sessions.items():
            if s.get("dataset_id") == dsid and s.get("user_id") == uid:
                return sid
        return None

    async def refresh_session_ttl(sid):
        return True

    for name, fn in (("get_session", get_session), ("save_session", save_session),
                     ("get_thread_session", get_thread_session),
                     ("find_session_by_dataset_for_user", find_session_by_dataset_for_user),
                     ("refresh_session_ttl", refresh_session_ttl)):
        monkeypatch.setattr(server, name, fn)
    analyses = {"an-C": {"owner_id": USER, "session_id": "sC", "thread_id": "tC"},
                "an-X": None}

    async def resolve_aid(aid, uid):
        row = analyses.get(aid)
        return row if row and row["owner_id"] == uid else None

    async def rehydrate(shared, sid):
        return None

    monkeypatch.setattr(server, "_resolve_session_from_aid", resolve_aid)
    monkeypatch.setattr(server, "_rehydrate_session_from_analysis", rehydrate)
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)
    monkeypatch.setattr(server.session_service, "cache", object())
    return {"sessions": sessions, "saved": saved}


# ---------------------------------------------------------------- merge


def test_merge_keeps_existing_datasets_and_appends_new_ones():
    g = {"dataset_id": "dsA", "dataset_ids": ["dsA", "dsB"],
         "dataset_session_map": {"dsA": "g1", "dsB": "sB"}}
    ids = server._merge_group_index(g, "g1", {"dsC": "sC"})
    assert ids == ["dsA", "dsB", "dsC"]
    assert g["dataset_session_map"] == {"dsA": "g1", "dsB": "sB", "dsC": "sC"}


def test_merge_into_a_single_dataset_group_keeps_its_own_dataset_first():
    """A cloud registration / DB sample / old single upload has no dataset_ids."""
    g = {"dataset_id": "dsA"}
    assert server._merge_group_index(g, "g1", {"dsC": "sC"}) == ["dsA", "dsC"]
    assert g["dataset_session_map"] == {"dsA": "g1", "dsC": "sC"}


def test_merge_does_not_duplicate_a_dataset_added_twice():
    g = {"dataset_id": "dsA", "dataset_ids": ["dsA"], "dataset_session_map": {"dsA": "g1"}}
    server._merge_group_index(g, "g1", {"dsC": "sC"})
    assert server._merge_group_index(g, "g1", {"dsC": "sC"}) == ["dsA", "dsC"]


def test_merge_reads_json_encoded_index():
    g = {"dataset_id": "dsA", "dataset_ids": '["dsA"]', "dataset_session_map": '{"dsA": "g1"}'}
    assert server._merge_group_index(g, "g1", {"dsC": "sC"}) == ["dsA", "dsC"]


# ---------------------------------------------------------------- attach endpoint


def test_attach_adds_a_dataset_to_the_chat(store):
    out = run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(dataset_ids=["dsC"])))

    assert out["dataset_ids"] == ["dsA", "dsB", "dsC"]
    assert [d["filename"] for d in out["datasets"]] == ["a.csv", "b.csv", "c.csv"]
    g1 = store["sessions"]["g1"]
    assert g1["dataset_ids"] == ["dsA", "dsB", "dsC"]
    assert g1["dataset_session_map"]["dsC"] == "sC"


def test_attach_is_idempotent(store):
    body = server.AttachDatasetsIn(dataset_ids=["dsC", "dsC"])
    run(server.attach_datasets_to_thread(_Req(), "t1", body))
    out = run(server.attach_datasets_to_thread(_Req(), "t1", body))
    assert out["dataset_ids"] == ["dsA", "dsB", "dsC"]


def test_another_users_dataset_cannot_be_attached(store):
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(dataset_ids=["dsX"])))
    assert exc.value.status_code == 404
    assert store["sessions"]["g1"]["dataset_ids"] == ["dsA", "dsB"]       # unchanged


def test_another_users_chat_cannot_be_changed(store):
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "tX", server.AttachDatasetsIn(dataset_ids=["dsC"])))
    assert exc.value.status_code == 404


def test_unknown_chat_is_404(store):
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "nope", server.AttachDatasetsIn(dataset_ids=["dsC"])))
    assert exc.value.status_code == 404


def test_empty_request_is_422(store):
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn()))
    assert exc.value.status_code == 422


def test_no_auth_is_401(store, monkeypatch):
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: None)
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(dataset_ids=["dsC"])))
    assert exc.value.status_code == 401


# ---------------------------------------------------------------- list endpoint


def test_list_returns_the_chats_datasets(store):
    out = run(server.list_thread_datasets(_Req(), "t1"))
    assert [d["dataset_id"] for d in out["datasets"]] == ["dsA", "dsB"]
    assert all(d["available"] for d in out["datasets"])


def test_list_does_not_write(store):
    run(server.list_thread_datasets(_Req(), "t1"))
    assert store["saved"] == []


def test_list_marks_an_expired_dataset_unavailable(store):
    del store["sessions"]["sB"]
    out = run(server.list_thread_datasets(_Req(), "t1"))
    assert [(d["dataset_id"], d["available"]) for d in out["datasets"]] == [("dsA", True), ("dsB", False)]


# ---------------------------------------------------------------- upload wiring


def test_upload_accepts_attach_to_thread_id():
    import inspect
    assert "attach_to_thread_id" in inspect.signature(server.upload_csv).parameters


# ---------------------------------------------------------------- analyses ("+ Add" picker)


def test_attach_by_analysis_adds_that_analysis_datasets(store):
    out = run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(analysis_ids=["an-C"])))
    assert out["dataset_ids"] == ["dsA", "dsB", "dsC"]


def test_attach_a_multi_sheet_analysis_adds_every_sheet(store):
    store["sessions"]["sC"].update(dataset_ids=["dsC", "dsD"], dataset_session_map={"dsC": "sC", "dsD": "sD"})
    store["sessions"]["sD"] = {"user_id": USER, "dataset_id": "dsD", "group_session_id": "sC",
                               "filename": "d.csv", "alias": "d"}
    out = run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(analysis_ids=["an-C"])))
    assert out["dataset_ids"] == ["dsA", "dsB", "dsC", "dsD"]


def test_attaching_an_analysis_does_not_change_that_analysis(store):
    before = dict(store["sessions"]["sC"])
    run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(analysis_ids=["an-C"])))
    assert store["sessions"]["sC"] == before


def test_an_unknown_or_foreign_analysis_is_404(store):
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(analysis_ids=["an-X"])))
    assert exc.value.status_code == 404
    assert store["sessions"]["g1"]["dataset_ids"] == ["dsA", "dsB"]


def test_an_expired_analysis_that_cannot_be_rebuilt_is_404(store):
    del store["sessions"]["sC"]
    with pytest.raises(server.HTTPException) as exc:
        run(server.attach_datasets_to_thread(_Req(), "t1", server.AttachDatasetsIn(analysis_ids=["an-C"])))
    assert exc.value.status_code == 404