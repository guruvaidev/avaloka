"""Regression tests: the coder's preview resolver must handle
uploaded_csv_preview persisted as a JSON string, matching the validator and
_preview_to_csv, so prompt sample rows and stub value matching don't silently
degrade to empty.
"""
import json

from app.agents.coder import _resolve_preview_records

DICT_PREVIEW = [
    {"OrderID": 1, "Product": "Laptop", "Price": 1200},
    {"OrderID": 2, "Product": "Mouse", "Price": 25},
]
LIST_PREVIEW = [
    ["OrderID", "Product", "Price"],
    [1, "Laptop", 1200],
    [2, "Mouse", 25],
]
EXPECTED = [
    {"OrderID": 1, "Product": "Laptop", "Price": 1200},
    {"OrderID": 2, "Product": "Mouse", "Price": 25},
]


def test_list_of_dicts_preview():
    assert _resolve_preview_records({"uploaded_csv_preview": DICT_PREVIEW}) == EXPECTED


def test_list_of_lists_preview():
    assert _resolve_preview_records({"uploaded_csv_preview": LIST_PREVIEW}) == EXPECTED


def test_json_string_of_dicts_preview():
    state = {"uploaded_csv_preview": json.dumps(DICT_PREVIEW)}
    assert _resolve_preview_records(state) == EXPECTED


def test_json_string_of_lists_preview():
    state = {"uploaded_csv_preview": json.dumps(LIST_PREVIEW)}
    assert _resolve_preview_records(state) == EXPECTED


def test_invalid_json_string_falls_back_to_sample_data():
    state = {
        "uploaded_csv_preview": "not json at all",
        "sample_data": "OrderID,Product,Price\n1,Laptop,1200\n2,Mouse,25",
    }
    records = _resolve_preview_records(state)
    assert [r["Product"] for r in records] == ["Laptop", "Mouse"]


def test_empty_inputs_resolve_to_empty():
    assert _resolve_preview_records({}) == []
    assert _resolve_preview_records({"uploaded_csv_preview": ""}) == []
    assert _resolve_preview_records({"uploaded_csv_preview": "   "}) == []
    assert _resolve_preview_records({"uploaded_csv_preview": "null"}) == []
