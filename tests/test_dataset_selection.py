"""Which datasets of a chat are in use: all sheets of a workbook by default,
the picked files when several were uploaded (PUT /threads/{id}/selected-datasets)."""
import asyncio
import json

import pytest

from app.api import server


def _json_or_value(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


@pytest.fixture(autouse=True)
def _real_json_helper(monkeypatch):
    # app.api.helpers is stubbed in the isolated test harness; in the repo the
    # real _maybe_json_load is already in place and this is a no-op swap.
    monkeypatch.setattr(server, "_maybe_json_load", _json_or_value)


def _group(**extra):
    g = {"user_id": "u1", "dataset_ids": ["a", "b", "c"],
         "dataset_session_map": {"a": "sa", "b": "sb", "c": "sc"}}
    g.update(extra)
    return g


def test_all_datasets_are_in_use_by_default():
    assert server._group_selected_ids(_group()) == ["a", "b", "c"]


def test_excluded_datasets_are_left_out_and_later_additions_are_in_use():
    g = _group(excluded_dataset_ids=["b"])
    assert server._group_selected_ids(g) == ["a", "c"]
    g["dataset_ids"].append("d")                 # "+ Add" later
    assert server._group_selected_ids(g) == ["a", "c", "d"]


def test_selection_is_never_empty():
    assert server._group_selected_ids(_group(excluded_dataset_ids=["a", "b", "c"])) == ["a", "b", "c"]


def test_json_encoded_fields_from_redis_are_read():
    g = _group(dataset_ids='["a", "b"]', excluded_dataset_ids='["a"]')
    assert server._group_selected_ids(g) == ["b"]


def test_a_single_dataset_session_without_group_index():
    assert server._group_selected_ids({"dataset_id": "x"}) == []


@pytest.fixture
def chat(monkeypatch):
    group = _group()
    members = {
        "sa": {"filename": "plan.xlsx [Agents].csv", "sheet_name": "Agents",
               "source_workbook": "plan.xlsx", "total_rows_exact": 25},
        "sb": {"filename": "plan.xlsx [UI].csv", "sheet_name": "UI",
               "source_workbook": "plan.xlsx", "total_rows_exact": 17},
        "sc": {"filename": "sales.csv", "sheet_name": None,
               "source_workbook": None, "total_rows_exact": 900},
    }
    saved = {}

    async def get_session(sid):
        return members.get(sid)

    async def save_session(sid, data):
        saved[sid] = dict(data)
        return True

    async def resolve_group(thread_id, user_id):
        return ("g1", group) if (thread_id == "t1" and user_id == "u1") else (None, None)

    monkeypatch.setattr(server, "get_session", get_session)
    monkeypatch.setattr(server, "save_session", save_session)
    monkeypatch.setattr(server, "_resolve_thread_group_for_user", resolve_group)
    monkeypatch.setattr(server, "_resolve_user_id", lambda request: "u1")
    monkeypatch.setattr(server.session_service, "cache", object(), raising=False)
    return group, saved


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_chips_carry_workbook_rows_and_selection(chat):
    group, _ = chat
    group["excluded_dataset_ids"] = ["c"]
    chips = _run(server._group_datasets_out(group))
    assert [(c["dataset_id"], c["source_workbook"], c["rows"], c["selected"]) for c in chips] == [
        ("a", "plan.xlsx", 25, True),
        ("b", "plan.xlsx", 17, True),
        ("c", None, 900, False),
    ]


def test_put_selection_stores_the_files_left_out(chat):
    group, saved = chat
    out = _run(server.select_thread_datasets(None, "t1", server.SelectDatasetsIn(dataset_ids=["c"])))
    assert group["excluded_dataset_ids"] == ["a", "b"]
    assert saved["g1"]["excluded_dataset_ids"] == ["a", "b"]
    assert out["selected_dataset_ids"] == ["c"]
    assert [c["selected"] for c in out["datasets"]] == [False, False, True]


def test_put_selection_needs_one_known_dataset(chat):
    with pytest.raises(server.HTTPException) as e:
        _run(server.select_thread_datasets(None, "t1", server.SelectDatasetsIn(dataset_ids=[])))
    assert e.value.status_code == 422
    with pytest.raises(server.HTTPException) as e:
        _run(server.select_thread_datasets(None, "t1", server.SelectDatasetsIn(dataset_ids=["zzz"])))
    assert e.value.status_code == 404


def test_put_selection_of_someone_elses_chat_is_404(chat):
    with pytest.raises(server.HTTPException) as e:
        _run(server.select_thread_datasets(None, "other", server.SelectDatasetsIn(dataset_ids=["a"])))
    assert e.value.status_code == 404


def test_excluded_ids_survive_a_cold_reopen():
    assert "excluded_dataset_ids" in server._SESSION_SNAPSHOT_KEYS