"""Write the null-word fixtures: a CSV with stray words in typed columns and a
three-sheet workbook. Fixed seed: the CSV regenerates byte-for-byte; the
workbook differs only in the creation time openpyxl stamps on it.

    python tests/fixtures/null_words/make_fixtures.py

``edge_cases.csv`` is release-test's D3 shape in 200 rows: a numeric column
with exactly three null-like cells (``N/A``, ``none``, ``-``), a date column in
five formats with ``N/A`` gaps, a free-text column where ``n/a``/``none`` are
labels and only empty cells are gaps, and a label column with case variants.
``three_sheets.xlsx`` holds two equal 60-row data sheets and a 5-row notes
sheet, so whichever sheet is analysed, the other data sheet is as big.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_ROWS = 200
SEED = 1104

#: Rows (0-based) whose typed cells carry a stray word, and the word.
GLUCOSE_WORDS = {5: "N/A", 50: "none", 120: "-"}
DATE_GAP_ROWS = (7, 44, 111, 170)          # "N/A" in visit_date
NOTE_EMPTY_ROWS = (3, 60, 150)             # true gaps in the text column
NOTE_LABELS = ("follow-up booked", "n/a", "none", "called back", "NULL", "-", "?")


def rows() -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for index in range(N_ROWS):
        day = index % 28 + 1
        month = index % 12 + 1
        year = 2022 + index % 2
        forms = (
            f"{year:04d}-{month:02d}-{day:02d}",
            f"{day:02d}/{month:02d}/{year:04d}",
            f"{month:02d}-{day:02d}-{year % 100:02d}",
            f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month - 1]} {day}, {year}",
            f"{year:04d}{month:02d}{day:02d}",
        )
        out.append({
            "patient_id": f"P{index:04d}",
            "Glucose": GLUCOSE_WORDS.get(index, str(rng.randint(70, 180))),
            "visit_date": "N/A" if index in DATE_GAP_ROWS else forms[index % 5],
            "Notes": "" if index in NOTE_EMPTY_ROWS else NOTE_LABELS[index % len(NOTE_LABELS)],
            "Outcome": ("Recovered", "RECOVERED", "Readmitted")[index % 3],
        })
    return out


def write_csv(dest: Path) -> Path:
    data = rows()
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return dest


def write_workbook(dest: Path) -> Path:
    import pandas as pd
    data = pd.DataFrame(rows()[:60])
    notes = pd.DataFrame({"sheet": ["notes"] * 5, "text": [f"note {i}" for i in range(5)]})
    with pd.ExcelWriter(dest) as writer:
        data.to_excel(writer, sheet_name="2022", index=False)
        data.to_excel(writer, sheet_name="2023", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)
    return dest


if __name__ == "__main__":
    write_csv(HERE / "edge_cases.csv")
    write_workbook(HERE / "three_sheets.xlsx")
