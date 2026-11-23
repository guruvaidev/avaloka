"""T16.7 in chat: ``find_datasets`` returns only the caller's datasets with
code-written summaries; ``use_dataset`` resolves ``my.<name>`` / ``analysis:<id>``
to one (dataset, version) and binds it to the turn like an upload, or says
plainly that Avaloka no longer holds the copy. The planner advertises both
tools, routes a bare "use my.<name>" without a model, and dispatches the
model's tool call. Bites: bind the latest version for an analysis reference;
bind without a location; return another user's dataset; let generate_code
answer "my datasets".
"""

from __future__ import annotations

import pytest

from app.catalog import chat
from app.catalog.store import CatalogStore

from tests.catalog.conftest import ORDERS

LOC = {"dataset_id": "u1", "filename": "orders_2026.csv", "format": "csv"}


@pytest.fixture
def seeded(db_url):
    store = CatalogStore("user-a")
    v1 = store.register("file", LOC, ORDERS, dataset_id="u1", display_name="orders_2026.csv",
                        analysis_id="an:churn", row_count=1200, row_count_kind="exact", tags=["finance"])
    v2 = store.register("file", LOC, ORDERS + [("region", "object")], dataset_id="u1", analysis_id="an:forecast")
    store.register("file", {"dataset_id": "u2", "filename": "hr_roster.xlsx"}, [("name", "object")],
                   dataset_id="u2", display_name="hr_roster.xlsx")
    CatalogStore("user-b").register("file", {"dataset_id": "u9", "filename": "payroll.csv"}, [("x", "int64")],
                                    dataset_id="u9", display_name="payroll.csv")
    return store, v1, v2


SESSIONS = {("u1", "user-a"): {"session_id": "sess-1", "data_source_location": "s3://uploads/u1/orders_2026.csv",
                               "filename": "orders_2026.csv", "uploaded_csv_columns": ["order_id", "customer_email", "amount"],
                               "row_count": 1200}}


def _locate(dataset_id, user_id):
    return SESSIONS.get((dataset_id, user_id))


# --------------------------------------------------------------------------- #
# find_datasets
# --------------------------------------------------------------------------- #

def test_find_lists_only_the_callers_datasets_with_handles_and_summaries(seeded):
    reply = chat.find_datasets({"user_id": "user-a"})
    assert "`my.orders_2026`" in reply and "`my.hr_roster`" in reply and "`u1`" in reply
    assert "payroll" not in reply and "u9" not in reply
    assert "orders_2026.csv: file, 4 columns, row count unknown, version 2" in reply, "version 2 was registered without a count"
    assert "use my.<name>" in reply
    only = chat.find_datasets({"user_id": "user-a"}, "finance")
    assert "orders_2026" in only and "hr_roster" not in only
    assert "No datasets matching `payroll`" in chat.find_datasets({"user_id": "user-a"}, "payroll")
    assert "No datasets yet" in chat.find_datasets({"user_id": "user-c"})
    assert chat.find_datasets({"user_id": "user-a"}, "", kind="spreadsheet").startswith("I don't know the dataset kind")


def test_find_refuses_without_a_user_and_says_so_without_a_catalog(seeded, monkeypatch):
    assert "sign in" in chat.find_datasets({"user_id": None})
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    assert "isn't available" in chat.find_datasets({"user_id": "user-a"})


# --------------------------------------------------------------------------- #
# use_dataset: resolve + bind
# --------------------------------------------------------------------------- #

def test_use_binds_the_latest_version_like_an_upload(seeded):
    _, v1, v2 = seeded
    state = {"user_id": "user-a", "messages": []}
    reply = chat.use_dataset(state, "my.orders_2026", locate=_locate)
    assert reply.startswith("Using `my.orders_2026` (version 2, 4 columns)")
    assert state["active_dataset_id"] == "u1" and state["active_dataset_ids"] == ["u1"]
    assert state["data_source_location"] == "s3://uploads/u1/orders_2026.csv"
    assert state["uploaded_csv_columns"] == ["order_id", "customer_email", "amount", "region"]
    (entry,) = state["datasets_context"]
    assert entry["dataset_id"] == "u1" and entry["filename"] == "orders_2026.csv" and entry["session_id"] == "sess-1"
    assert entry["data_source_location"] == "s3://uploads/u1/orders_2026.csv" and entry["columns"][-1] == "region"
    assert state["multi_dataset_state"] == state["datasets_context"]
    assert state["catalog_binding"] == {"dataset_id": "u1", "version_id": v2.version_id, "version_no": 2,
                                        "kind": "file", "pinned_by": None}


def test_use_pins_the_version_an_analysis_read_not_the_latest(seeded):
    """Bite: bind the latest version for an analysis reference."""
    _, v1, v2 = seeded
    for ref in ("analysis:an:churn", "from analysis an:churn"):
        state = {"user_id": "user-a"}
        reply = chat.use_dataset(state, ref, locate=_locate)
        assert "version 1" in reply and "pinned by analysis an:churn" in reply, reply
        assert state["catalog_binding"]["version_id"] == v1.version_id and state["catalog_binding"]["pinned_by"] == "an:churn"
        assert state["uploaded_csv_columns"] == ["order_id", "customer_email", "amount"], "the schema of that version"
    state = {"user_id": "user-a"}
    chat.use_dataset(state, "my.orders_2026@1", locate=_locate)
    assert state["catalog_binding"]["version_no"] == 1


def test_use_without_a_copy_says_so_and_binds_nothing(seeded):
    """Bite: bind the dataset without a location."""
    state = {"user_id": "user-a"}
    reply = chat.use_dataset(state, "my.hr_roster", locate=_locate)
    assert "no longer holds a copy" in reply and "`my.hr_roster`" in reply and "Upload it again" in reply
    assert "active_dataset_id" not in state and "catalog_binding" not in state and "data_source_location" not in state
    state = {"user_id": "user-a"}
    assert "no longer holds a copy" in chat.use_dataset(state, "my.orders_2026")   # no locator at all
    assert "catalog_binding" not in state


def test_an_analysis_output_binds_from_its_recorded_location(seeded):
    """A derived dataset has no upload session: its locator carries where the
    output landed, and that is what gets bound. Bite: ignore the locator."""
    store, _, _ = seeded
    store.register("derived", {"analysis_id": "an:churn", "parent_dataset_id": "u1",
                               "output_location": "s3://outputs/an-churn/churn_by_region.csv"},
                   [("region", "object"), ("churn_rate", "float64")], dataset_id="ds:output:1",
                   display_name="churn_by_region.csv", row_count=2, row_count_kind="exact")
    state = {"user_id": "user-a"}
    reply = chat.use_dataset(state, "my.churn_by_region", locate=_locate)   # the session store knows nothing of it
    assert reply.startswith("Using `my.churn_by_region` (version 1, 2 columns)")
    assert state["data_source_location"] == "s3://outputs/an-churn/churn_by_region.csv"
    assert state["datasets_context"][0]["filename"] == "churn_by_region.csv"
    assert state["uploaded_csv_columns"] == ["region", "churn_rate"]
    # a derived dataset registered before locations were recorded still says so honestly
    store.register("derived", {"analysis_id": "an:old", "parent_dataset_id": "u1"}, [("x", "int64")],
                   dataset_id="ds:output:0", display_name="old_output.csv")
    state = {"user_id": "user-a"}
    assert "no longer holds a copy" in chat.use_dataset(state, "my.old_output", locate=_locate)
    assert "catalog_binding" not in state


def test_use_is_the_callers_own_and_speaks_in_sentences(seeded):
    state = {"user_id": "user-b"}
    reply = chat.use_dataset(state, "my.orders_2026", locate=_locate)
    assert reply.startswith("I couldn't find that dataset") and "no dataset of yours is called" in reply
    assert "catalog_binding" not in state
    assert "I couldn't find that dataset" in chat.use_dataset({"user_id": "user-a"}, "my.payroll", locate=_locate)
    assert "sign in" in chat.use_dataset({"user_id": None}, "my.orders_2026")
    store = CatalogStore("user-a")
    store.register("file", {"dataset_id": "u3", "filename": "orders_2026.parquet"}, ORDERS, dataset_id="u3",
                   display_name="orders_2026.parquet")
    reply = chat.use_dataset({"user_id": "user-a"}, "my.orders_2026", locate=_locate)
    assert "names 2 of your datasets" in reply and "u1" in reply and "u3" in reply


@pytest.mark.parametrize("text, ref", [
    ("use my.orders_2026", "my.orders_2026"),
    ("Use my.orders_2026@2 please", "my.orders_2026@2"),
    ("please analyse my.orders_2026", "my.orders_2026"),
    ("switch to the dataset my.hr_roster.", "my.hr_roster"),
    ("use analysis:an:churn", "analysis:an:churn"),
    ("use the one from analysis an:churn", "analysis:an:churn"),
    ("load the dataset from analysis an:forecast", "analysis:an:forecast"),
    ("what is my.orders_2026", None),
    ("use the amount column", None),
    ("my datasets", None),
])
def test_the_reference_is_found_in_a_use_sentence_and_nowhere_else(text, ref):
    assert chat.find_reference(text) == ref


def test_find_requests_are_recognised_without_a_model():
    assert chat.is_find_request("Which datasets have I used?")
    assert chat.is_find_request("show my catalog")
    assert not chat.is_find_request("what cloud connections do I have?")
    assert not chat.is_find_request("use my.orders_2026")


# --------------------------------------------------------------------------- #
# The planner: tools advertised, shortcut routed, tool call dispatched
# --------------------------------------------------------------------------- #

planner = pytest.importorskip("app.agents.planner")


def test_the_planner_advertises_both_tools_and_the_prompt_names_them():
    names = {t["function"]["name"] for t in planner.tools}
    assert {"find_datasets", "use_dataset"} <= names
    assert planner.tool_name_to_param["find_datasets"] is planner.FindDatasetsParams
    assert planner.tool_name_to_param["use_dataset"] is planner.UseDatasetParams
    literal = planner.ToolCall.model_fields["name"].annotation
    assert "find_datasets" in getattr(literal, "__args__", ()) and "use_dataset" in getattr(literal, "__args__", ())
    planner.ToolCall(name="use_dataset", parameters=planner.UseDatasetParams(reference="my.orders_2026"))
    import inspect
    source = inspect.getsource(planner)
    assert "18. `find_datasets`" in source and "19. `use_dataset`" in source
    assert "never generate_code, never list_cloud_datasets" in source


def _quiet_planner(monkeypatch):
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(chat, "locate_in_session_store", _locate)


def _turn(monkeypatch, state, text, forced_tool=None, args=None):
    from langchain_core.messages import AIMessage, HumanMessage
    if forced_tool:
        monkeypatch.setattr(planner, "_invoke_planner_with_recovery", lambda msgs, tool_choice, kw: AIMessage(
            content="", tool_calls=[{"name": forced_tool, "id": "t", "args": args or {}}]))
    else:
        def boom(*a, **k):
            raise AssertionError("the model was called for a request the shortcut should have answered")
        monkeypatch.setattr(planner, "_invoke_planner_with_recovery", boom)
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=text)]
    state.update(planner.plan_etl_job(state))
    return [m for m in state["messages"] if isinstance(m, AIMessage)][-1].content


def test_a_bare_use_sentence_binds_without_calling_the_model(seeded, monkeypatch):
    _quiet_planner(monkeypatch)
    state = {"user_id": "user-a", "messages": [], "dta_database_registry": {}}
    reply = _turn(monkeypatch, state, "use my.orders_2026")
    assert reply.startswith("Using `my.orders_2026` (version 2")
    assert state["active_dataset_id"] == "u1" and state["ready_to_code"] is False
    reply = _turn(monkeypatch, state, "which datasets have I used?")
    assert "`my.orders_2026`" in reply and "payroll" not in reply


def test_the_models_tool_calls_are_dispatched(seeded, monkeypatch):
    _quiet_planner(monkeypatch)
    state = {"user_id": "user-a", "messages": [], "dta_database_registry": {}}
    reply = _turn(monkeypatch, state, "find the finance data I used", forced_tool="find_datasets", args={"query": "finance"})
    assert "`my.orders_2026`" in reply and "hr_roster" not in reply
    reply = _turn(monkeypatch, state, "work on the churn analysis's data", forced_tool="use_dataset",
                  args={"reference": "analysis:an:churn"})
    assert "version 1" in reply and state["catalog_binding"]["pinned_by"] == "an:churn"
