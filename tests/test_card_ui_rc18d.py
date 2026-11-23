"""rc18c (db446704) chart text from the rweb18c regrade.

2. The Notes pie said "with 18 missing" where the bar chart and the null-word
   rule (file_handler/null_policy.py) say 1: the column profile it is built
   from read "none" and "NULL" as missing on top of the rule.
3. "Average avg_fare_amount by payment_type: highest is 1 (21.65, from 1 row)":
   that average is over every payment_type-1 trip; the "1 row" was the result
   table's row. A chart of a result table states no row count.

The fixture (tests/fixtures/card_truth/rweb18c_card_ui.json) holds the shown
sentences and the fare result table recomputed from the regrade's file; the
notes column is the repo's null-words workbook, the file the regrade uploaded.
"""

from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents import visualization_agent as va

HERE = Path(__file__).parent
CASE = json.loads((HERE / "fixtures" / "card_truth" / "rweb18c_card_ui.json").read_text(encoding="utf-8"))
WORKBOOK = HERE / "fixtures" / "null_words" / "three_sheets.xlsx"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _notes_rows():
    raw = pd.read_excel(WORKBOOK, sheet_name="2022", dtype=str, keep_default_na=False)
    return raw.to_dict("records")


# --------------------------------------------------------------------------- #
# 2. The pie and every chart sentence read gaps by the one rule
# --------------------------------------------------------------------------- #
def test_the_notes_pie_counts_the_gaps_the_rule_counts():
    from file_handler.null_policy import gap_cells

    rows = _notes_rows()
    assert "with 18 missing (30.0%)" in CASE["shown"]["notes_pie"]
    gaps = sum(gap_cells([r["Notes"] for r in rows]))
    assert gaps == 1
    notes = next(p for p in va.profile_columns(rows) if p["name"] == "Notes")
    text = va._pie_insight_text("Notes", notes["stats"]["top_k"], len(rows))
    assert "with 1 missing (1.7%)" in text, text
    assert "none (9 rows, 15.0%)" in text and "NULL (8 rows, 13.3%)" in text, text


def test_the_built_profile_agrees_with_itself_and_with_the_rule():
    rows = _notes_rows()
    cfg = va.build_visualization_config_from_sample("ds-notes", rows, use_llm=False)
    notes = next(p for p in cfg["columns"] if p["name"] == "Notes")
    # The pie reads top_k; the missing ratio was already the rule's (1 of 60).
    missing_in_top_k = sum(t["count"] for t in notes["stats"]["top_k"] if t["value"] == "nan")
    assert round(notes["missing_ratio"] * len(rows)) == 1
    assert missing_in_top_k == 1, notes["stats"]["top_k"]
    said = " ".join(str(ch.get("insight") or "") for ch in cfg["charts"])
    assert "18 missing" not in said and "18 rows (30.0%) have no Notes" not in said, said


def test_a_missing_share_by_group_reads_gaps_by_the_rule():
    rows = [{"g": g, "note": n} for g, n in (("a", "none"), ("a", ""), ("b", "NULL"), ("b", "ok"))] * 10
    _p, _m, text = va._ground_row_share_by_group(rows, "g", "note", "share_missing")
    # "none" and "NULL" are labels in a text column; only the blank is missing.
    assert "highest for" not in text and "a (50.0% of 20 rows)" in text, text
    assert "b (0.0% of 20 rows)" in text, text


# --------------------------------------------------------------------------- #
# 3. A result table's rows are not records
# --------------------------------------------------------------------------- #
FARE_CHART = {
    "id": "llm_1", "type": "bar", "intent": "comparison", "title": "Average fare by payment type",
    "encodings": {"x": {"field": "payment_type", "type": "nominal"},
                  "y": {"field": "avg_fare_amount", "type": "quantitative", "aggregate": "mean"}},
    "config": {"top_k": 15},
}


def _build(rows, monkeypatch, **flag):
    monkeypatch.setattr(va, "select_charts", lambda *_a, **_k: [json.loads(json.dumps(FARE_CHART))])
    params = inspect.signature(va.build_visualization_config_from_sample).parameters
    kwargs = flag if ("rows_are_results" in params) else {}
    return va.build_visualization_config_from_sample("ds-taxi", rows, use_llm=False, **kwargs)


def test_a_result_table_average_states_no_row_count(monkeypatch):
    rows = CASE["fare_result_rows"]
    assert CASE["shown"]["fare"].startswith("Average avg_fare_amount by payment_type: highest is 1 (21.65, from 1 row)")
    cfg = _build(rows, monkeypatch, rows_are_results=True)
    text = next(ch["insight"] for ch in cfg["charts"] if ch["id"] == "llm_1")
    assert "highest is 1 (21.65), lowest is 4 (1.51)" in text, text
    assert "from 1 row" not in text, text


def test_the_visualize_node_says_when_it_charts_a_result_table(monkeypatch):
    seen = {}

    def capture(*_a, **kwargs):
        seen.update(kwargs)
        return {"charts": [], "visualization_status": "ready"}

    monkeypatch.setattr(va, "build_visualization_config_from_sample", capture)
    csv = pd.DataFrame(CASE["fare_result_rows"]).to_csv(index=False).encode()
    state = {"dataset_id": "ds-taxi", "messages": [],
             "uploaded_csv_columns": ["VendorID", "payment_type", "fare_amount"],
             "uploaded_csv_preview": [{"VendorID": 1, "payment_type": 1, "fare_amount": 10.0}],
             "output_file_data": {"content": "data:text/csv;base64," + base64.b64encode(csv).decode()}}
    va.visualization_agent_node(state)
    assert seen.get("rows_are_results") is True, seen
    seen.clear()
    va.visualization_agent_node({k: v for k, v in state.items() if k != "output_file_data"})
    assert seen.get("rows_are_results") is False, seen


def test_raw_records_still_say_a_one_row_average(monkeypatch):
    rows = [{"payment_type": p, "avg_fare_amount": v} for p, v, n in ((1, 20.0, 30), (2, 10.0, 25), (3, 5.0, 1))
            for _ in range(n)]
    cfg = _build(rows, monkeypatch, rows_are_results=False)
    text = next(ch["insight"] for ch in cfg["charts"] if ch["id"] == "llm_1")
    assert "lowest is 3 (5, from 1 row)" in text, text
