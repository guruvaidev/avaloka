"""The scope line reaches the user on the state the web chat actually builds.

#586 passed every unit test and never fired in the rc15 web check: asked "How
does the average order amount differ by customer segment?", two of three runs
dropped the returned orders and each answer showed "$545.08 / $182.27 /
$59.77" with no scope line. The tests gave the verifier the bare question. The
graph holds the message the web chat augmented with an "[Analysis context]"
block (each dataset's columns and preview rows, ``"returned": false``), and
read as the user's words that block "asked" for every filter on a column of
the data.

These drive the stored rc15 turns (tests/fixtures/filter_disclosure_live,
taken from rweb15) through claim_verifier_node and presentable_reply, the call
the server makes on the reply (app/api/server.py), to the text the user sees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply

FIXTURES = Path(__file__).parent / "fixtures"
TURNS = json.loads((FIXTURES / "filter_disclosure_live" / "rc15_turns.json").read_text(encoding="utf-8"))["turns"]

#: (segment, orders kept, mean kept, orders returned, all-order mean): the
#: rc15 data's 20,000 orders, 1,522 of them returned.
SEGMENTS = (("enterprise", 1850, 545.08, 150, 545.33), ("smb", 5550, 182.27, 450, 182.17),
            ("consumer", 11078, 59.77, 922, 59.79))


def _related_data(tmp_path):
    customers, orders = [], []
    for i, (segment, kept, mean_kept, returned, mean_all) in enumerate(SEGMENTS):
        customers.append({"customer_id": i + 1, "region": "West", "signup_year": 2022, "segment": segment})
        mean_returned = (mean_all * (kept + returned) - mean_kept * kept) / returned
        orders += [{"customer_id": i + 1, "order_date": "2024-07-31", "amount": mean_kept, "returned": False}] * kept
        orders += [{"customer_id": i + 1, "order_date": "2024-08-04", "amount": mean_returned, "returned": True}] * returned
    frame = pd.DataFrame(orders)
    frame.insert(0, "order_id", range(1, len(frame) + 1))
    paths = {"customers": tmp_path / "customers.csv", "orders": tmp_path / "orders.csv"}
    pd.DataFrame(customers).to_csv(paths["customers"], index=False)
    frame.to_csv(paths["orders"], index=False)
    return paths


def _live_state(turn, paths, human=None):
    """The state at verify time as the server and graph build it: the
    augmented human message, the code that ran, and the reads the executor
    recorded (execution_agent._annotate_reads), main(df)'s unused one first."""
    code = turn["code"]
    reads = []
    for alias, live_path in turn["read_map"].items():
        code = code.replace(live_path, str(paths[alias]))
        reads.append({"path": str(paths[alias]), "sheet_name": None, "via": "code", "alias": alias})
    reads.insert(0, {"path": str(paths["customers"]), "sheet_name": None, "via": "main(df)", "used": False,
                     "alias": "customers"})
    return {
        "messages": [HumanMessage(content=human if human is not None else turn["graph_human_message"]),
                     AIMessage(content=turn["narrative"])],
        "coder_definition": {"code": code},
        "planner_definition": turn["planner_definition"],
        "output_json": turn["output_json"],
        "execution_result": {"status": "success", "fresh_output": True,
                             "output_row_count": len(turn["output_json"]), "reads": reads},
    }


def _shown(state):
    report = claim_verifier_node(state)["verification_report"]
    return presentable_reply(state["messages"][-1].content, report), report


def test_the_fixture_is_the_live_shape():
    for run in ("rw7", "rw7b", "rw7c"):
        turn = TURNS[run]
        assert turn["graph_human_message"].startswith(turn["question"] + "\n")
        assert "[Analysis context]" in turn["graph_human_message"]
        assert '"returned": false' in turn["graph_human_message"]
        assert "Scope:" not in turn["shown_on_rc15"]          # what rc15 showed


def test_rc15_run_a_shows_the_returned_orders_it_left_out(tmp_path):
    """rw7: ``df_orders[df_orders['returned'] == False]``."""
    text, report = _shown(_live_state(TURNS["rw7"], _related_data(tmp_path)))
    line = ("Scope: excludes 1,522 returned orders (returned = True), a filter the question did not ask for; "
            "the figures above are computed on the remaining 18,478 of 20,000 orders.")
    assert line in text
    assert text.index("$545.08") < text.index(line)
    assert report["scoped"] is True and report["verified"] is False


def test_rc15_run_c_shows_the_returned_orders_it_left_out(tmp_path):
    """rw7c: ``returned`` lower-cased with ``astype(str)``, then ``isin(['false', ...])``."""
    text, _report = _shown(_live_state(TURNS["rw7c"], _related_data(tmp_path)))
    assert "Scope: excludes 1,522 orders (returned not in false, 0, no, n)" in text
    assert "remaining 18,478 of 20,000 orders" in text
    assert "['returned']" not in text


def test_rc15_run_b_kept_every_order_and_shows_no_scope_line(tmp_path):
    """rw7b filtered nothing ($545.33 is the all-order mean): no line."""
    text, report = _shown(_live_state(TURNS["rw7b"], _related_data(tmp_path)))
    assert "Scope:" not in text
    assert not report.get("scoped")


def test_a_filter_the_user_typed_is_still_asked_for_under_the_context_block(tmp_path):
    turn = TURNS["rw7"]
    asked = turn["graph_human_message"].replace(turn["question"],
                                                "Excluding returned orders, how does the average order "
                                                "amount differ by customer segment?", 1)
    text, _report = _shown(_live_state(turn, _related_data(tmp_path), human=asked))
    assert "Scope:" not in text


def test_an_uncounted_filter_is_shown_in_words_not_code(tmp_path):
    """A filter after the join cannot be replayed; its line names the column
    and the condition, not ``returned['returned'] == False``."""
    turn = dict(TURNS["rw7"])
    turn["code"] = turn["code"].replace(
        "        df_orders = df_orders[df_orders['returned'] == False]\n", "        pass\n").replace(
        "    result = df_merged.groupby",
        "    df_merged = df_merged[df_merged['returned'] == False]\n    result = df_merged.groupby")
    assert "df_merged[df_merged['returned'] == False]" in turn["code"]
    compile(turn["code"], "<rw7>", "exec")
    text, _report = _shown(_live_state(turn, _related_data(tmp_path)))
    assert "(only where returned = False)" in text
    assert "['returned']" not in text and "==" not in text


def test_a_null_drop_on_a_derived_column_is_disclosed_and_counted(tmp_path):
    """rw7c nullxlsx: ``df.dropna(subset=['Outcome', 'Glucose', 'month'])``
    on two concatenated sheets, ``month`` derived from ``visit_date``. It
    dropped most rows and rc15 said nothing. The two sheets are stacked in the
    replay and ``month`` rebuilt, so it is counted: 100 of 120 rows, which
    leaves the 20 patients (4 + 16) the result shows (rc16 counted what rc15
    called uncountable)."""
    turn = TURNS["rw7c_nullxlsx"]
    code = turn["code"]
    reads = []
    for sheet, live in re.findall(r"path_(\d{4}) = '([^']+)'", code):
        path = tmp_path / f"{sheet}.csv"
        pd.read_excel(FIXTURES / "null_words" / "three_sheets.xlsx", sheet_name=sheet, dtype=str,
                      keep_default_na=False).to_csv(path, index=False)
        code = code.replace(live, str(path))
        reads.append({"path": str(path), "sheet_name": None, "via": "code"})
    assert len(reads) == 2
    state = {"messages": [HumanMessage(content=turn["question"]), AIMessage(content=turn["narrative"])],
             "coder_definition": {"code": code}, "output_json": turn["output_json"],
             "execution_result": {"status": "success", "reads": reads}}
    text, report = _shown(state)
    assert ("Scope: excludes 100 rows with no value in Outcome, Glucose, month (83.3%); the figures above are "
            "computed on the remaining 20 of 120 rows.") in text
    assert sum(int(r["patient_count"]) for r in turn["output_json"]) == 20
    # The dropna on the aggregated table drops groups of the result, not rows of the data.
    assert "patient_count" not in text
    assert [f["readable"] for f in report["scope"]["filters"]] == ["Outcome, Glucose and month have a value"]


@pytest.mark.parametrize("run", ["rw7", "rw7c"])
def test_the_reconstruction_without_the_context_block_agrees(tmp_path, run):
    """The tester's reconstruction (bare question) and the live shape give the same line."""
    paths = _related_data(tmp_path)
    live, _ = _shown(_live_state(TURNS[run], paths))
    bare, _ = _shown(_live_state(TURNS[run], paths, human=TURNS[run]["question"]))
    assert [l for l in live.splitlines() if l.startswith("Scope:")] == \
        [l for l in bare.splitlines() if l.startswith("Scope:")] != []
