"""The sampling step's basis says what the code combined.

On the web re-run, a three-sheet workbook (2022, 2023, notes) became three
datasets; the code read 2022 and 2023 and compared 40 readmitted with 80
recovered patients, and the narration said "I ran this on all 60 rows ...
so these numbers are the whole file's": the primary sheet's count. #551's
basis rule, in the CLI, names the sheet analysed and the sheets that were
not; here the same rule names the sheets the code read, their rows together,
and the sheet it left out.
"""

from __future__ import annotations

from app.api import chat_voice

SHEETS = [
    {"alias": "null_words_three_sheets_2022", "filename": "null_words_three_sheets [2022].csv",
     "sheet_name": "2022", "source_workbook": "null_words_three_sheets.xlsx", "rows": 60},
    {"alias": "null_words_three_sheets_2023", "filename": "null_words_three_sheets [2023].csv",
     "sheet_name": "2023", "source_workbook": "null_words_three_sheets.xlsx", "rows": 60},
    {"alias": "null_words_three_sheets_notes", "filename": "null_words_three_sheets [notes].csv",
     "sheet_name": "notes", "source_workbook": "null_words_three_sheets.xlsx", "rows": 6},
]

COMBINED = """
df2022 = pd.read_csv(file_map["null_words_three_sheets_2022"])
df2023 = pd.read_csv(file_map["null_words_three_sheets_2023"])
df = pd.concat([df2022, df2023], ignore_index=True)
result = df.groupby("Outcome")["Glucose"].agg(["mean", "median", "count"])
"""

CONTEXT = {"rows_analyzed": 60, "total_rows": 60, "is_sampled": False, "mode_label": "Portfolio Samples"}


def _final(code: str) -> dict:
    return {"avaloka_intent": "statistical_analysis", "coder_definition": {"code": code},
            "execution_result": {"status": "success", "message": "done", "output_row_count": 2},
            "output_json": [{"Outcome": "Readmitted", "count": 40}, {"Outcome": "Recovered", "count": 80}]}


def _sampling(code: str, datasets=SHEETS, ctx=CONTEXT) -> str:
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        _final(code), question="what drives the Outcome?", execution_context=ctx, datasets=datasets)}
    return prompts["sampling"]["message"]


def test_two_sheets_combined_are_named_with_their_rows_together():
    msg = _sampling(COMBINED)
    assert "all 60 rows" not in msg and "whole file" not in msg, msg
    assert "both sheets '2022' and '2023' (60 + 60 rows) combined" in msg, msg
    assert "sheet 'notes' (6 rows) not used" in msg, msg


def test_a_sheet_that_is_only_mapped_is_not_counted_as_read():
    code = 'df = pd.read_csv(file_map["null_words_three_sheets_2022"])\nresult = df.describe()'
    msg = _sampling(code)
    assert "read only sheet '2022', not the whole workbook" in msg and "all 60 rows of it" in msg, msg
    assert "sheet '2023' (60 rows) and sheet 'notes' (6 rows) not analysed" in msg, msg
    assert "that sheet's, not the whole workbook's" in msg, msg


def test_a_file_that_is_not_a_workbook_keeps_the_whole_file_line():
    single = [{"alias": "stroke", "filename": "stroke.csv", "sheet_name": None, "source_workbook": None,
               "rows": 5110}]
    msg = _sampling("df = pd.read_csv(path)", datasets=single,
                    ctx={**CONTEXT, "rows_analyzed": 5110, "total_rows": 5110})
    assert msg == "I ran this on all 5,110 rows (Portfolio Samples), so these numbers are the whole file's."


def test_an_alias_that_prefixes_another_is_not_read_by_accident():
    datasets = [{"alias": "sales", "filename": "sales.csv", "rows": 10},
                {"alias": "sales_2", "filename": "sales_2.csv", "rows": 10}]
    read, _ = chat_voice.datasets_read(_final('pd.read_csv(file_map["sales"])'), datasets)
    assert [d["alias"] for d in read] == ["sales"]
    read, _ = chat_voice.datasets_read(_final('a = file_map["sales"]; b = file_map["sales_2"]'), datasets)
    assert [d["alias"] for d in read] == ["sales", "sales_2"]


def test_the_fixture_would_catch_a_basis_of_the_primary_only(monkeypatch):
    """Without the read detection the basis is the primary sheet's count
    again, which is the defect: the two-sheet assertion above fails."""
    monkeypatch.setattr(chat_voice, "datasets_read", lambda final, datasets: ([], []))
    msg = _sampling(COMBINED)
    assert msg.startswith("I ran this on all 60 rows") and "combined" not in msg, msg
