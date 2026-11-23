"""A multi-sheet workbook is never double-counted, and the verifier catches a
count that does not match the data.

rc3 web re-check, gemma, on the two-sheet online-retail workbook: "United
Kingdom ... 38,024" and "GLIDERS ... 7,414 units", each exactly twice the
truth. The upload made each sheet its own dataset (input_<id>.csv) and the
prompt gave the coder those csv_paths; the read map knew only aliases and
file names, so both reads missed it and fell back to the primary. One sheet
was read twice and concatenated, the basis said the second sheet was "not
analysed", and the verifier said every claim held, because the doubled
figure was in the result table.

On diabetes, "the readmission rate is consistently 1.0 across all observed
combinations" passed as verified, and "readmitted within 30 days ...
approximately 0.35" was shown although its own check had flagged it.

No live model: every reply here is a stub.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.execution_agent as exec_mod
from app.agents.claim_verifier import (WITHHELD_ANSWER, claim_verifier_node, presentable_reply)
from app.api import chat_voice

SHEETS = ("2010 Q4", "2011 Q1")


@pytest.fixture
def workbook(tmp_path: Path) -> dict:
    """Two sheets with the same columns and different rows, exported the way
    /api/upload exports them: one input_<id>.csv per sheet."""
    q4 = pd.DataFrame({
        "Country": ["United Kingdom"] * 7 + ["Germany"] * 2 + ["France"],
        "Description": ["GLIDERS"] * 4 + ["HEART HOLDER"] * 6,
        "Quantity": [12, 48, 5, 7, 6, 6, 3, 2, 8, 1],
    })
    q1 = pd.DataFrame({
        "Country": ["United Kingdom"] * 5 + ["France"] * 3,
        "Description": ["GLIDERS"] * 2 + ["HEART HOLDER"] * 6,
        "Quantity": [24, 9, 4, 4, 2, 10, 1, 3],
    })
    xlsx = tmp_path / "ecommerce_messy.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        q4.to_excel(writer, sheet_name=SHEETS[0], index=False)
        q1.to_excel(writer, sheet_name=SHEETS[1], index=False)
    metas, paths = [], {}
    for i, (sheet, frame) in enumerate(zip(SHEETS, (q4, q1))):
        dsid = f"a18376{i}1-dca0-4411-b980-d54d9c6f895{i}"
        path = tmp_path / f"s{i}" / f"input_{dsid}.csv"
        path.parent.mkdir()
        frame.to_csv(path, index=False)
        paths[sheet] = str(path)
        metas.append({
            "dataset_id": dsid, "alias": f"ecommerce_messy_{sheet.replace(' ', '_')}",
            "filename": f"ecommerce_messy [{sheet}].csv", "data_source_location": str(path),
            "sheet_name": sheet, "source_workbook": xlsx.name,
        })
    return {"xlsx": xlsx, "frames": {SHEETS[0]: q4, SHEETS[1]: q1}, "metas": metas, "paths": paths}


def _truth(workbook: dict) -> pd.Series:
    both = pd.concat(list(workbook["frames"].values()), ignore_index=True)
    return both.groupby("Country").size()


# What gemma wrote on rc3: both sheets read by the csv_path the prompt gave.
def _gemma_code(path_a: str, path_b: str) -> str:
    return (
        "import pandas as pd\n"
        "def main(df):\n"
        f"    df_2010 = pd.read_csv({path_a!r})\n"
        f"    df_2011 = pd.read_csv({path_b!r})\n"
        "    df_combined = pd.concat([df_2010, df_2011], ignore_index=True)\n"
        "    return df_combined.groupby('Country').size().reset_index(name='Country_Count')\n"
        "if __name__ == '__main__':\n"
        "    df = read_csv_best_effort('input.csv')\n"
        "    main(df).to_csv('output.csv', index=False)\n"
    )


def _run(workbook: dict, code: str, tmp_path: Path) -> dict:
    primary = workbook["metas"][0]
    state = {
        "messages": [HumanMessage(content="Which Country has the most rows?")],
        "coder_definition": {"code": code},
        "data_source_location": primary["data_source_location"],
        "output_location": str(tmp_path / "out" / "output.csv"),
        "multi_dataset_state": workbook["metas"],
    }
    return exec_mod.execution_agent_node_local(state)


# --------------------------------------------------------------------------- #
# (a) the code reads both sheets, and the counts are pandas', not doubled
# --------------------------------------------------------------------------- #

def test_a_two_sheet_read_counts_each_sheet_once(workbook, tmp_path):
    code = _gemma_code(workbook["paths"][SHEETS[0]], workbook["paths"][SHEETS[1]])
    out = _run(workbook, code, tmp_path)
    assert out["execution_result"]["status"] == "success", out["execution_result"]
    got = {r["Country"]: int(r["Country_Count"]) for r in out["output_json"]}
    assert got == _truth(workbook).to_dict()          # UK 12, not 14 (7 read twice)
    read = [(r["sheet_name"], r["via"]) for r in out["execution_result"]["reads"]]
    assert (SHEETS[0], "code") in read and (SHEETS[1], "code") in read, read


def test_a_read_by_alias_or_file_name_gets_its_own_sheet(workbook, tmp_path):
    a, b = workbook["metas"][0], workbook["metas"][1]
    code = _gemma_code(a["alias"], b["filename"])
    out = _run(workbook, code, tmp_path)
    got = {r["Country"]: int(r["Country_Count"]) for r in out["output_json"]}
    assert got == _truth(workbook).to_dict()


def test_one_workbook_file_with_two_sheet_datasets_reads_each_sheet(workbook, tmp_path):
    xlsx = str(workbook["xlsx"])
    metas = [dict(m, data_source_location=xlsx) for m in workbook["metas"]]
    code = _gemma_code(metas[0]["alias"], metas[1]["alias"])
    state = {
        "messages": [HumanMessage(content="Which Country has the most rows?")],
        "coder_definition": {"code": code}, "data_source_location": xlsx,
        "output_location": str(tmp_path / "out" / "output.csv"),
        "multi_dataset_state": metas, "sheet_name": SHEETS[0],
    }
    out = exec_mod.execution_agent_node_local(state)
    assert out["execution_result"]["status"] == "success", out["execution_result"]
    got = {r["Country"]: int(r["Country_Count"]) for r in out["output_json"]}
    assert got == _truth(workbook).to_dict()


def test_the_basis_names_both_sheets_the_code_read(workbook, tmp_path):
    code = _gemma_code(workbook["paths"][SHEETS[0]], workbook["paths"][SHEETS[1]])
    final = _run(workbook, code, tmp_path)
    datasets = [{"dataset_id": m["dataset_id"], "alias": m["alias"], "filename": m["filename"],
                 "sheet_name": m["sheet_name"], "source_workbook": m["source_workbook"],
                 "rows": len(workbook["frames"][m["sheet_name"]])} for m in workbook["metas"]]
    prompts = {p["step"]: p["message"] for p in chat_voice.prompts_after_turn(
        final, question="Which Country has the most rows?",
        execution_context={"rows_analyzed": 10, "total_rows": 10, "is_sampled": False},
        datasets=datasets)}
    assert "both sheets '2010 Q4' and '2011 Q1' (10 + 8 rows)" in prompts["sampling"], prompts["sampling"]
    assert "not analysed" not in prompts["sampling"], prompts["sampling"]


# --------------------------------------------------------------------------- #
# (b) an unmapped path is a plain error, never the primary
# --------------------------------------------------------------------------- #

def test_an_unmapped_path_is_an_error_not_the_primary(workbook, tmp_path):
    # Sheet B by its path, and a path that is no dataset: before, the second
    # silently became the primary (sheet A) and the run "succeeded".
    code = _gemma_code(workbook["paths"][SHEETS[1]], "/var/tmp/avaloka/other/input_unknown.csv")
    out = _run(workbook, code, tmp_path)
    result = out["execution_result"]
    assert result["status"] == "failed", result
    assert "ReadMapError" in result["message"] and "none of this turn's datasets" in result["message"], result
    assert "input_unknown.csv" in result["message"], result
    assert "isn't one of this chat's datasets" in out["messages"][-1].content


def test_one_dataset_read_under_two_names_is_an_error(workbook, tmp_path):
    code = _gemma_code(workbook["paths"][SHEETS[0]], workbook["metas"][0]["alias"])
    out = exec_mod.execute_code_on_local(
        code=code, input_data_location=workbook["paths"][SHEETS[0]],
        output_location=str(tmp_path / "o.csv"), read_path_map=exec_mod._resolve_read_paths(
            {"multi_dataset_state": workbook["metas"]}))
    assert out["status"] == "error"
    assert out["execution_error"].startswith("ReadMapError") and "twice" in out["execution_error"]


# --------------------------------------------------------------------------- #
# (c) a doubled count is caught by the verifier and withheld
# --------------------------------------------------------------------------- #

def _verified(reply: str, rows: list, reads: list) -> dict:
    state = {
        "messages": [HumanMessage(content="q"), AIMessage(content=reply)],
        "execution_result": {"status": "success", "message": "done", "reads": reads},
        "output_json": rows,
    }
    return claim_verifier_node(state)["verification_report"]


def _reads(workbook: dict) -> list:
    return [{"path": workbook["paths"][s], "sheet_name": s, "via": "code", "alias": s} for s in SHEETS]


def test_a_doubled_count_is_contradicted_and_withheld(workbook):
    reply = "The United Kingdom has the most rows, with a count of 24."      # truth: 12
    report = _verified(reply, [{"Country": "United Kingdom", "Country_Count": 24}], _reads(workbook))
    finding = report["findings"][0]
    assert finding["check"] == "group_figure_mismatch" and finding["verdict"] == "contradicted", report
    assert "exactly 2 times" in finding["detail"], finding
    assert report["verified"] is False and report["safe_to_present"] is False
    assert presentable_reply(reply, report) == WITHHELD_ANSWER


def test_a_doubled_sum_is_contradicted(workbook):
    # GLIDERS: Quantity 12+48+5+7 + 24+9 = 105; doubled, 210.
    reply = 'The top seller, "GLIDERS", sold 210 units across both quarters.'
    report = _verified(reply, [{"Description": "GLIDERS", "total_quantity_sold": 210}], _reads(workbook))
    assert report["findings"][0]["check"] == "group_figure_mismatch", report


def test_the_true_count_reconciles(workbook):
    reply = 'The United Kingdom has the most rows, with a count of 12, and "GLIDERS" sold 105 units.'
    report = _verified(reply, [{"Country": "United Kingdom", "Country_Count": 12, "q": 105}],
                       _reads(workbook))
    assert report["findings"] == [] and report["verified"] is True, report


def test_only_one_sheet_said_when_the_code_read_two_is_contradicted(workbook):
    reply = "I analysed only sheet 2010 Q4, where the United Kingdom has 12 rows."
    report = _verified(reply, [{"Country": "United Kingdom", "n": 12}], _reads(workbook))
    assert report["findings"][0]["check"] == "sheet_scope_contradicted", report


# --------------------------------------------------------------------------- #
# (d) a flagged claim never reaches the reply as fact (the 0.35 case)
# --------------------------------------------------------------------------- #

def _turn(question: str, reply: str, csv: str) -> dict:
    return {
        "messages": [HumanMessage(content=question), AIMessage(content=reply)],
        "execution_result": {"status": "success", "message": "done"},
        "output_file_data": {"content": "data:text/csv;base64," + base64.b64encode(csv.encode()).decode()},
    }


def test_an_invented_figure_is_not_shown_as_fact():
    # 0.35 is in neither the question nor the result (0.1116): invented.
    # The question's own 30 is a restatement and is not what flags it.
    question = "What share of encounters were readmitted within 30 days?"
    reply = "The share of encounters readmitted within 30 days was approximately 0.35."
    state = _turn(question, reply, "readmission_share_30_days\n0.1116\n")
    report = claim_verifier_node(state)["verification_report"]
    assert [f["check"] for f in report["findings"]] == ["invented_number"], report
    assert "0.35" in report["findings"][0]["detail"], report
    shown = presentable_reply(reply, report)
    assert "0.35" not in shown and shown == WITHHELD_ANSWER
    voice = {p["step"]: p["message"] for p in chat_voice.prompts_after_turn(
        dict(state, verification_report=report), question=question)}
    assert "held back" in voice["verification"], voice


def test_a_number_restated_from_the_question_is_not_a_finding():
    # 30 is only in the question; 0.11 is the result. An honest answer, shown.
    question = "What was the readmission rate within 30 days?"
    reply = "Within 30 days the rate was 0.11."
    state = _turn(question, reply, "rate\n0.11\n")
    report = claim_verifier_node(state)["verification_report"]
    assert report["findings"] == [] and report["verified"] is True, report
    assert presentable_reply(reply, report) == reply


def test_a_percent_restated_from_the_question_is_not_a_finding():
    question = "How many orders had a discount over 15%?"
    reply = "Orders with a discount over 15 percent number 1,204 in total."
    report = claim_verifier_node(_turn(question, reply, "orders\n1204\n"))["verification_report"]
    assert report["findings"] == [], report


_FLAGGED_035 = "The share of encounters readmitted within 30 days was approximately 0.35."


def _unsupported_report(claim: str, **extra) -> dict:
    """What the verifier writes for an invented number: unsupported, and
    safe_to_present still True. No claims_reconciled: a turn the analysis
    gate does not grade (a reused artifact, a training turn) has none."""
    return {"safe_to_present": True, "claims_checked": 1, "n_problems": 1, **extra,
            "findings": [{"claim": claim, "verdict": "unsupported", "check": "invented_number",
                          "detail": "cites 0.35, which does not appear in the computed evidence"}]}


def test_an_unsupported_number_is_never_shown_as_fact_even_when_safe_to_present():
    shown = presentable_reply(_FLAGGED_035, _unsupported_report(_FLAGGED_035))
    assert "0.35" not in shown and shown == WITHHELD_ANSWER


def test_only_the_unsupported_sentence_goes_and_the_reply_says_so():
    reply = f"I looked at the readmitted column across all encounters. {_FLAGGED_035}"
    shown = presentable_reply(reply, _unsupported_report(_FLAGGED_035))
    assert "0.35" not in shown
    assert shown.startswith("I looked at the readmitted column across all encounters.")
    assert "held back 1 statement" in shown


def test_unsupported_prose_without_a_number_stays():
    reply = "Readmission looks related to the number of prior inpatient visits."
    report = {"safe_to_present": True, "findings": [
        {"claim": reply, "verdict": "unsupported", "check": "unsupported_wording", "detail": "x"}]}
    assert presentable_reply(reply, report) == reply


def test_a_supported_reply_is_shown_unchanged():
    reply = "About 11.2% of encounters were readmitted early, a share of 0.1116."
    csv = "share_lt30\n0.1116\n"
    state = {
        "messages": [HumanMessage(content="q"), AIMessage(content=reply)],
        "execution_result": {"status": "success", "message": "done"},
        "output_file_data": {"content": "data:text/csv;base64," + base64.b64encode(csv.encode()).decode()},
    }
    report = claim_verifier_node(state)["verification_report"]
    assert report["findings"] == [] and report["verified"] is True, report
    assert presentable_reply(reply, report) == reply


# --------------------------------------------------------------------------- #
# (e) a constant-rate claim is checked against the data
# --------------------------------------------------------------------------- #

def test_a_constant_rate_the_data_contradicts(tmp_path):
    data = tmp_path / "diabetic_data.csv"
    pd.DataFrame({"readmitted": ["NO"] * 6 + [">30"] * 3 + ["<30"],
                  "gender": ["Female", "Male"] * 5}).to_csv(data, index=False)
    reply = ("Based on the visible data, the readmission rate is consistently 1.0 across all "
             "observed combinations of factors.")
    report = _verified(reply, [{"gender": "Female", "readmission_rate": "1.0"},
                               {"gender": "Male", "readmission_rate": "1.0"}],
                       [{"path": str(data), "via": "main(df)"}])
    finding = report["findings"][0]
    assert finding["check"] == "constant_rate_contradicted", report
    assert "'readmitted' takes 3 values" in finding["detail"], finding
    assert presentable_reply(reply, report) == WITHHELD_ANSWER


def test_a_constant_rate_the_result_contradicts():
    reply = "The readmission rate is uniformly 1.0 across every specialty in the data."
    report = _verified(reply, [{"s": "a", "readmission_rate": 1.0}, {"s": "b", "readmission_rate": 0.4}], [])
    assert report["findings"][0]["check"] == "constant_rate_contradicted", report


# --------------------------------------------------------------------------- #
# (f) rc5: every frame the run read is in ``reads``, the verifier recomputes on
#     all of them, and the basis is built from them alone
# --------------------------------------------------------------------------- #
# gemma: main(df) used the primary (2010 Q4) and the code read 2011 Q1 by its
# path. The verifier kept only the code's read and withheld a correct 37,881 as
# "contradicted ... 18,869 rows". groq: both sheets read through path
# variables, no read recorded, and the basis said 2011 Q1 was "not analysed".

def _gemma_rc5_code(path_b: str) -> str:
    return (
        "import pandas as pd\n"
        "def main(df):\n"
        f"    df_q1 = pd.read_csv({path_b!r})\n"
        "    uk_2010 = len(df[df['Country'].astype(str).str.strip() == 'United Kingdom'])\n"
        "    uk_2011 = len(df_q1[df_q1['Country'].astype(str).str.strip() == 'United Kingdom'])\n"
        "    return pd.DataFrame({'Total_UK_Count': [uk_2010 + uk_2011]})\n"
    )


def _groq_rc5_code(path_a: str, path_b: str = None) -> str:
    lines = ["import pandas as pd", "def main(df):", f"    path_a = {path_a!r}"]
    if path_b:
        lines += [f"    path_b = {path_b!r}",
                  "    combined = pd.concat([pd.read_csv(path_a, dtype=str), pd.read_csv(path_b, dtype=str)],"
                  " ignore_index=True)"]
    else:
        lines += ["    combined = pd.read_csv(path_a, dtype=str)"]
    lines += ["    uk = int((combined['Country'].str.strip() == 'United Kingdom').sum())",
              "    return pd.DataFrame({'Total_UK_Count': [uk]})",
              "if __name__ == '__main__':",
              "    df = pd.read_csv('input.csv', dtype=str)",
              "    main(df).to_csv('output.csv', index=False)"]
    return "\n".join(lines) + "\n"


def _datasets(workbook: dict) -> list:
    return [{"dataset_id": m["dataset_id"], "alias": m["alias"], "filename": m["filename"],
             "sheet_name": m["sheet_name"], "source_workbook": m["source_workbook"],
             "rows": len(workbook["frames"][m["sheet_name"]])} for m in workbook["metas"]]


def _basis(final: dict, workbook: dict, question: str = "How many rows have Country equal to United Kingdom?") -> str:
    prompts = {p["step"]: p["message"] for p in chat_voice.prompts_after_turn(
        final, question=question,
        execution_context={"rows_analyzed": 10, "total_rows": 10, "is_sampled": False},
        datasets=_datasets(workbook))}
    return prompts["sampling"]


def _verify_turn(final: dict, reply: str) -> dict:
    state = dict(final, messages=[HumanMessage(content="How many rows have Country equal to United Kingdom?"),
                                  AIMessage(content=reply)])
    return claim_verifier_node(state)["verification_report"]


def test_f_a_main_frame_and_a_sheet_read_by_path_are_both_reads_and_verify(workbook, tmp_path):
    final = _run(workbook, _gemma_rc5_code(workbook["paths"][SHEETS[1]]), tmp_path)
    assert final["execution_result"]["status"] == "success", final["execution_result"]
    assert final["output_json"] == [{"Total_UK_Count": 12}]          # 7 + 5, both sheets
    reads = [(r["sheet_name"], r["via"], r.get("used", True)) for r in final["execution_result"]["reads"]]
    assert reads == [(SHEETS[0], "main(df)", True), (SHEETS[1], "code", True)], reads
    report = _verify_turn(final, "The total count for the United Kingdom is 12 rows.")
    assert not [f for f in report["findings"] if f["verdict"] == "contradicted"], report
    assert report["verified"] is True, report


def test_f_sheets_read_through_path_variables_are_recorded_and_named(workbook, tmp_path):
    final = _run(workbook, _groq_rc5_code(workbook["paths"][SHEETS[0]], workbook["paths"][SHEETS[1]]), tmp_path)
    assert final["output_json"] == [{"Total_UK_Count": 12}]
    reads = [(r["sheet_name"], r["via"], r.get("used", True)) for r in final["execution_result"]["reads"]]
    # main(df) is handed the primary but never uses it: not a frame the code read.
    assert reads == [(SHEETS[0], "main(df)", False), (SHEETS[0], "code", True), (SHEETS[1], "code", True)], reads
    msg = _basis(final, workbook)
    assert "both sheets '2010 Q4' and '2011 Q1' (10 + 8 rows) combined" in msg, msg
    assert "not analysed" not in msg, msg
    report = _verify_turn(final, "The total count for the United Kingdom is 12 rows.")
    assert report["verified"] is True, report


def test_f_one_sheet_read_names_that_sheet_and_the_other_as_not_analysed(workbook, tmp_path):
    # Only 2011 Q1, through a path variable; main(df)'s 2010 Q4 is unused.
    final = _run(workbook, _groq_rc5_code(workbook["paths"][SHEETS[1]]), tmp_path)
    assert final["output_json"] == [{"Total_UK_Count": 5}]
    msg = _basis(final, workbook)
    assert "read only sheet '2011 Q1', not the whole workbook" in msg, msg
    assert "all 8 rows of it" in msg, msg
    assert "sheet '2010 Q4' (10 rows) not analysed" in msg, msg
    # A question naming the sheet keeps the plain one-sheet basis.
    msg = _basis(final, workbook, question="How many UK rows are in 2011 Q1?")
    assert msg.startswith("I ran this on all 8 rows of sheet '2011 Q1'"), msg
    assert "sheet '2010 Q4' (10 rows) not analysed" in msg, msg


def test_f_the_same_frame_read_twice_is_still_a_contradicted_double_count(workbook, tmp_path):
    # main(df) is 2010 Q4 and the code reads 2010 Q4 again by its path: UK 14, truth 7.
    code = ("import pandas as pd\n"
            "def main(df):\n"
            f"    again = pd.read_csv({workbook['paths'][SHEETS[0]]!r})\n"
            "    both = pd.concat([df, again], ignore_index=True)\n"
            "    return pd.DataFrame({'Total_UK_Count': [int((both['Country'] == 'United Kingdom').sum())]})\n")
    final = _run(workbook, code, tmp_path)
    assert final["output_json"] == [{"Total_UK_Count": 14}]
    report = _verify_turn(final, "The United Kingdom has a count of 14 rows.")
    finding = report["findings"][0]
    assert finding["verdict"] == "contradicted" and "exactly 2 times" in finding["detail"], report
    assert presentable_reply("The United Kingdom has a count of 14 rows.", report) == WITHHELD_ANSWER


def test_f_glucose_missing_over_both_sheets_is_stated_over_both(tmp_path):
    # null_words_three_sheets.xlsx: Glucose has 2 missing ("N/A", "none") on
    # each of 2022 and 2023, 4 over both. groq said "4 missing" -- right for
    # both sheets -- while the basis said "all 60 rows of sheet '2022'".
    metas, paths, frames = [], {}, {}
    for i, sheet in enumerate(("2022", "2023", "notes")):
        dsid = f"d13201d{i}-074e-43a1-868e-906782717f6{i}"
        frame = (pd.DataFrame({"note": ["a", "b"]}) if sheet == "notes" else
                 pd.DataFrame({"Glucose": ["N/A", "none"] + [str(100 + j) for j in range(58)]}))
        path = tmp_path / f"s{i}" / f"input_{dsid}.csv"
        path.parent.mkdir()
        frame.to_csv(path, index=False)
        paths[sheet], frames[sheet] = str(path), frame
        metas.append({"dataset_id": dsid, "alias": f"null_words_three_sheets_{sheet}",
                      "filename": f"null_words_three_sheets [{sheet}].csv", "data_source_location": str(path),
                      "sheet_name": sheet, "source_workbook": "null_words_three_sheets.xlsx"})
    code = ("import pandas as pd\n"
            "def main(df):\n"
            f"    path_2022 = {paths['2022']!r}\n"
            f"    path_2023 = {paths['2023']!r}\n"
            "    combined = pd.concat([pd.read_csv(path_2022, dtype=str), pd.read_csv(path_2023, dtype=str)])\n"
            "    g = pd.to_numeric(combined['Glucose'].str.strip(), errors='coerce')\n"
            "    return pd.DataFrame({'missing_glucose_count': [int(g.isna().sum())]})\n")
    final = exec_mod.execution_agent_node_local({
        "messages": [HumanMessage(content="how many Glucose values are missing?")],
        "coder_definition": {"code": code}, "data_source_location": paths["2022"],
        "output_location": str(tmp_path / "out" / "output.csv"), "multi_dataset_state": metas})
    assert final["output_json"] == [{"missing_glucose_count": 4}]       # 2 per sheet, 4 over both
    datasets = [dict(m, rows=len(frames[m["sheet_name"]])) for m in metas]
    msg = {p["step"]: p["message"] for p in chat_voice.prompts_after_turn(
        final, question="how many Glucose values are missing?",
        execution_context={"rows_analyzed": 60, "total_rows": 60, "is_sampled": False},
        datasets=datasets)}["sampling"]
    assert "both sheets '2022' and '2023' (60 + 60 rows) combined" in msg, msg
    assert "sheet 'notes' (2 rows) not used" in msg and "not analysed" not in msg, msg
