"""Registration from the analysis path: ``record_analysis_lineage`` registers
the inputs (pinned to the analysis) and the derived output. Bite: register
nothing from the hook; register the output under the wrong owner."""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.lineage_hooks import record_analysis_lineage
from app.catalog.hooks import parent_registration, record_analysis
from app.catalog.store import CatalogStore
from app.core.lineage import LineageStore


# A connection string with a password, assembled at run time so no tracked
# file carries a credential-shaped literal (tests/contract/test_c1_repo_hygiene.py
# E11.05 and test_c9_public_tree_secrets.py scan the tree for exactly that).
PLANTED_PASSWORD = "hun" + "ter2"


def _dsn_with_password() -> str:
    return "postgresql://" + "u" + ":" + PLANTED_PASSWORD + "@" + "h/db"


def _state(**extra):
    state = {
        "user_id": "user-a",
        "active_dataset_id": "upl-1",
        "uploaded_csv_columns": ["order_id", "customer_email", "amount"],
        "multi_dataset_state": [{"dataset_id": "upl-1", "filename": "orders_2026.csv",
                                 "columns": [{"name": "order_id", "dtype": "int64"},
                                             {"name": "customer_email", "dtype": "object"},
                                             {"name": "amount", "dtype": "float64"}],
                                 "row_count": 1200}],
        "output_file_data": {"filename": "churn_by_region.csv"},
        "output_location": "s3://outputs/an/churn_by_region.csv",
    }
    state.update(extra)
    return state


def test_a_recorded_analysis_registers_its_input_and_output(db_url):
    output = pd.DataFrame({"region": ["n", "s"], "churn_rate": [0.1, 0.2]})
    output_id = record_analysis_lineage(_state(), output)
    assert output_id and output_id.startswith("ds:output:")

    store = CatalogStore("user-a")
    items = {d["dataset_id"]: d for d in store.list()}
    assert set(items) == {"upl-1", output_id}
    assert items["upl-1"]["kind"] == "file" and items["upl-1"]["name"] == "my.orders_2026"
    assert items["upl-1"]["locator"] == {"dataset_id": "upl-1", "filename": "orders_2026.csv", "format": "csv"}
    assert items["upl-1"]["row_count"] == 1200 and items["upl-1"]["columns"] == 3
    assert items[output_id]["kind"] == "derived" and items[output_id]["row_count"] == 2
    assert items[output_id]["display_name"] == "churn_by_region.csv"
    # the analysis that lineage minted pinned the input's version
    (analysis_id,) = store.get("upl-1")["analyses"]
    assert analysis_id.startswith("an:")
    binding = store.resolve(f"analysis:{analysis_id}")
    assert binding.dataset_id == "upl-1" and binding.version_no == 1
    assert items[output_id]["locator"] == {"analysis_id": analysis_id, "parent_dataset_id": "upl-1",
                                           "output_location": "s3://outputs/an/churn_by_region.csv"}
    # the catalog id is the lineage node id
    graph = LineageStore("user-a")
    assert graph.get_node(output_id).kind.value == "dataset"
    assert graph.dataset_lineage(output_id)["parents"][0].id == "upl-1"
    graph.close()
    # a second analysis on the unchanged upload: no new version, one more pin
    record_analysis_lineage(_state(), output)
    assert store.get("upl-1")["version_no"] == 1 and len(store.get("upl-1")["analyses"]) == 2
    assert CatalogStore("user-b").list() == []


def test_a_connection_backed_input_is_a_db_table_whose_locator_names_the_connection():
    spec = parent_registration("ds-9", {"connection_id": "conn-1", "table": "orders", "schema": "sales",
                                        "columns": [("id", "int64")]}, {})
    assert spec["kind"] == "db_table"
    assert spec["locator"] == {"connection_id": "conn-1", "dataset_id": "ds-9", "schema": "sales", "table": "orders"}
    assert spec["display_name"] == "orders" and spec["columns"] == [("id", "int64")]
    upload = parent_registration("upl-2", {"filename": "hr.xlsx", "alias": "HR"}, {}, ["name", "salary"])
    assert upload["kind"] == "file" and upload["locator"]["format"] == "xlsx"
    assert upload["columns"] == [("name", "unknown"), ("salary", "unknown")]


def test_the_hook_never_raises_and_never_fails_the_analysis(db_url, monkeypatch):
    output = pd.DataFrame({"x": [1]})
    # no owner: nothing registered, no exception
    assert record_analysis({"user_id": None}, output, output_id="ds:o", analysis_id="an:1",
                           parents=["p"], metadata_by_id={}) is None
    # no database: nothing registered, no exception
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    assert record_analysis(_state(), output, output_id="ds:o", analysis_id="an:1",
                           parents=["upl-1"], metadata_by_id={}) is None
    # and the lineage hook that calls it still returns None cleanly
    assert record_analysis_lineage(_state(), output) is None


def test_a_credential_in_the_metadata_does_not_reach_the_catalog(db_url):
    """A connection record that leaked a DSN into the dataset metadata must not
    become a locator. The parent is still registered, by id, without it."""
    state = _state(multi_dataset_state=[{"dataset_id": "upl-1", "filename": "orders.csv",
                                         "connection_id": "conn-1", "dsn": _dsn_with_password(),
                                         "columns": [("a", "int64")]}])
    output_id = record_analysis_lineage(state, pd.DataFrame({"a": [1]}))
    store = CatalogStore("user-a")
    detail = store.get("upl-1")
    assert detail is not None and detail["kind"] == "db_table"
    assert PLANTED_PASSWORD not in str(detail) and "dsn" not in detail["locator"]
    assert store.get(output_id) is not None
