import io
import json

import pandas as pd

import app.api.workflow as wf


# -------------------------
# _preview_to_csv unit
# -------------------------

DICT_PREVIEW = [
    {"OrderID": 1, "Product": "Laptop", "Price": 1200},
    {"OrderID": 2, "Product": "Mouse", "Price": 25},
]
LIST_PREVIEW = [
    ["OrderID", "Product", "Price"],
    [1, "Laptop", 1200],
    [2, "Mouse", 25],
]


def test_list_of_dicts_emits_header_plus_values():
    csv_text = wf._preview_to_csv(DICT_PREVIEW)
    lines = csv_text.splitlines()
    assert lines[0] == "OrderID,Product,Price"
    # The bug produced N identical header lines; rows must now carry values.
    assert lines[1] != lines[0]
    assert lines[1] != lines[2]
    df = pd.read_csv(io.StringIO(csv_text))
    assert df.iloc[0]["Product"] == "Laptop"
    assert int(df.iloc[1]["Price"]) == 25


def test_list_of_lists_still_supported():
    csv_text = wf._preview_to_csv(LIST_PREVIEW)
    df = pd.read_csv(io.StringIO(csv_text))
    assert list(df.columns) == ["OrderID", "Product", "Price"]
    assert df.iloc[0]["Product"] == "Laptop"


def test_json_string_preview_is_parsed():
    assert wf._preview_to_csv(json.dumps(DICT_PREVIEW)) == wf._preview_to_csv(DICT_PREVIEW)


def test_values_with_commas_are_quoted():
    csv_text = wf._preview_to_csv([{"a": "x,y", "b": 1}])
    df = pd.read_csv(io.StringIO(csv_text))
    assert df.iloc[0]["a"] == "x,y"


def test_ragged_dicts_union_of_keys():
    csv_text = wf._preview_to_csv([{"a": 1}, {"a": 2, "b": 3}])
    lines = csv_text.splitlines()
    assert lines[0] == "a,b"
    df = pd.read_csv(io.StringIO(csv_text))
    assert list(df.columns) == ["a", "b"]


def test_empty_and_invalid_inputs():
    assert wf._preview_to_csv(None) == ""
    assert wf._preview_to_csv([]) == ""
    assert wf._preview_to_csv("not json") == ""
    assert wf._preview_to_csv({"a": 1}) == ""


# -------------------------
# coding_subgraph_node feeds valued sample_data to the coder
# -------------------------

def test_coding_subgraph_passes_valued_sample_data(monkeypatch):
    captured = {}

    class FakeGraph:
        def invoke(self, initial):
            captured["sample_data"] = initial.get("sample_data")
            return {"generated_code": "x = 1", "llm_raw_response": None}

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)

    state = {
        "uploaded_csv_preview": DICT_PREVIEW,
        "user_prompt": "Read the CSV and save it.",
        "messages": [],
        "schema": {},
    }
    wf.coding_subgraph_node(state)

    sample = captured["sample_data"]
    assert "Laptop" in sample and "Mouse" in sample
    lines = sample.splitlines()
    assert lines[0] == "OrderID,Product,Price"
    assert lines[1] != lines[2]


# ---------------------------------------------------------------------------
# Prompt-template integrity
#
# The coder's system prompt is an f-string, so an unescaped `{` in prose is a
# RUNTIME error, not a syntax error -- ast.parse and every offline test still
# pass, and the break only appears when a real prompt reaches the coder. Adding
# a groupby example containing `{'col': ['mean','max']}` did exactly that:
#
#   ValueError: Invalid format specifier ' ['mean','max']' for object of type 'str'
#
# Every prompt in the suite failed with plan=False, code=False. This renders the
# f-string with dummy values so the same mistake costs seconds instead of a
# 25-minute run. No LLM: the coder returns early to the stub path when no API key
# is set, so the prompt is never built that way.
# ---------------------------------------------------------------------------


def test_coder_system_prompt_renders_with_no_unescaped_braces():
    import ast as _ast
    import pathlib as _pathlib

    src = _pathlib.Path("app/agents/coder.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)

    fstrings = [
        node
        for node in _ast.walk(tree)
        if isinstance(node, _ast.JoinedStr) and len(_ast.dump(node)) > 2000
    ]
    assert fstrings, "no large f-string found in coder.py; did the prompt move?"

    for node in fstrings:
        segment = _ast.get_source_segment(src, node)
        if not segment:
            continue
        names = {
            n.id
            for fv in _ast.walk(node)
            if isinstance(fv, _ast.FormattedValue)
            for n in _ast.walk(fv.value)
            if isinstance(n, _ast.Name)
        }
        namespace = {name: "x" for name in names}
        try:
            rendered = eval(segment, {"__builtins__": {}}, namespace)  # noqa: S307
        except ValueError as exc:
            raise AssertionError(
                f"the coder prompt does not render: {exc}. A literal brace in "
                f"prose must be doubled -- write {{{{'col': ['mean']}}}} for "
                f"{{'col': ['mean']}}."
            ) from exc
        assert isinstance(rendered, str) and rendered
