"""Merge executed results into the Master Test Plan and emit CSV + XLSX.

Every row keeps the plan's original columns and gains: Status, Actual result,
Evidence, Executed by, Executed on. A case with no executed result is left as
NOT RUN with the reason, never silently marked passed.
"""
from __future__ import annotations
import csv, json, os, sys
from datetime import datetime, timezone

S = os.path.dirname(os.path.abspath(__file__))
BRANCH = "temp-develop-1.6"
WHEN = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

cases = list(csv.DictReader(open(os.path.join(S, "cases.csv"))))
DEFECTS = json.load(open(os.path.join(S, "defect_map.json"))) \
    if os.path.exists(os.path.join(S, "defect_map.json")) else {}
FINDINGS = json.load(open(os.path.join(S, "findings.json"))) \
    if os.path.exists(os.path.join(S, "findings.json")) else {}
results = {}
for f in ("pod_results.json", "host_results.json"):
    p = os.path.join(S, f)
    if os.path.exists(p):
        for cid, rec in json.load(open(p)).items():
            # host result wins when both ran the same id (it saw the cluster)
            if cid not in results or f.startswith("host"):
                results[cid] = rec

out_cols = ["Sheet", "ID", "Pri", "Component", "Feature under test",
            "Expected result", "Status", "Actual result", "Evidence",
            "Dataset / fixture", "Source (code · doc · PR)", "Owner",
            "Executed on", "Branch", "Defect ref"]

rows = []
for c in cases:
    r = results.get(c["ID"])
    if r:
        status, actual, evidence = r["status"], r["actual"], r["evidence"]
        when = WHEN
    else:
        status = "NOT RUN"
        actual = "no automated check exists for this case in this run"
        evidence = ""
        when = ""
    rows.append({
        "Sheet": c["Sheet"], "ID": c["ID"], "Pri": c["Pri"], "Component": c["Component"],
        "Feature under test": c["Feature under test"], "Expected result": c["Expected result"],
        "Status": status, "Actual result": actual, "Evidence": evidence,
        "Dataset / fixture": c.get("Dataset / fixture", ""),
        "Source (code · doc · PR)": c.get("Source (code · doc · PR)", ""),
        "Owner": c.get("Owner", ""), "Executed on": when,
        "Branch": BRANCH if r else "", "Defect ref": DEFECTS.get(c["ID"], ""),
    })

csv_path = os.path.join(S, "Avaloka-1.6-Test-Report.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=out_cols)
    w.writeheader(); w.writerows(rows)

from collections import Counter
counts = Counter(r["Status"] for r in rows)
by_sheet = {}
for r in rows:
    by_sheet.setdefault(r["Sheet"], Counter())[r["Status"]] += 1

print(f"  rows: {len(rows)}")
print(f"  overall: {dict(counts)}")
print(f"  wrote {csv_path}")

# ---- XLSX ----------------------------------------------------------------
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    print("  openpyxl unavailable on this interpreter; CSV only")
    sys.exit(0)

FILL = {"PASS": "C6EFCE", "FAIL": "FFC7CE", "BLOCKED": "FFEB9C",
        "MANUAL": "DDEBF7", "N/A": "E7E6E6", "NOT RUN": "F2F2F2"}
FONT = {"PASS": "006100", "FAIL": "9C0006", "BLOCKED": "9C5700",
        "MANUAL": "1F4E78", "N/A": "808080", "NOT RUN": "808080"}

wb = openpyxl.Workbook()
ws = wb.active; ws.title = "Summary"

ws["A1"] = "Avaloka 1.6 — Master Test Plan execution report"
ws["A1"].font = Font(bold=True, size=14)
ws["A2"] = f"Branch: {BRANCH}   ·   Executed: {WHEN}   ·   Cases: {len(rows)}"
ws["A3"] = ("Executed against a live Kubernetes (kind) deployment of the branch. "
            "Nothing is marked PASS that was not actually run.")
ws["A3"].font = Font(italic=True)

ws["A5"] = "Status"; ws["B5"] = "Count"; ws["C5"] = "Meaning"
for c in ("A5", "B5", "C5"): ws[c].font = Font(bold=True)
meaning = {
    "PASS": "executed; the expected result was observed",
    "FAIL": "executed; the expected result was NOT observed",
    "BLOCKED": "could not execute — the reason names the missing dependency",
    "MANUAL": "needs a human (browser UI, visual judgement)",
    "N/A": "not applicable to this branch",
    "NOT RUN": "no automated check exists for this case yet",
}
row = 6
for st in ("PASS", "FAIL", "BLOCKED", "MANUAL", "N/A", "NOT RUN"):
    ws.cell(row=row, column=1, value=st).fill = PatternFill("solid", fgColor=FILL[st])
    ws.cell(row=row, column=1).font = Font(bold=True, color=FONT[st])
    ws.cell(row=row, column=2, value=counts.get(st, 0))
    ws.cell(row=row, column=3, value=meaning[st])
    row += 1

row += 1
ws.cell(row=row, column=1, value="By area").font = Font(bold=True, size=12); row += 1
hdr = ["Area", "PASS", "FAIL", "BLOCKED", "MANUAL", "N/A", "NOT RUN", "Total"]
for i, h in enumerate(hdr, 1):
    ws.cell(row=row, column=i, value=h).font = Font(bold=True)
row += 1
for sheet, c in by_sheet.items():
    ws.cell(row=row, column=1, value=sheet)
    for i, st in enumerate(("PASS", "FAIL", "BLOCKED", "MANUAL", "N/A", "NOT RUN"), 2):
        ws.cell(row=row, column=i, value=c.get(st, 0))
    ws.cell(row=row, column=8, value=sum(c.values()))
    row += 1

for col, width in (("A", 34), ("B", 10), ("C", 62)):
    ws.column_dimensions[col].width = width

# detail sheet
d = wb.create_sheet("All cases")
for i, h in enumerate(out_cols, 1):
    cell = d.cell(row=1, column=i, value=h)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="44546A")
    cell.alignment = Alignment(vertical="center")
d.freeze_panes = "A2"
for ri, r in enumerate(rows, 2):
    for ci, h in enumerate(out_cols, 1):
        cell = d.cell(row=ri, column=ci, value=r[h])
        cell.alignment = Alignment(vertical="top", wrap_text=h in
                                   ("Feature under test", "Expected result", "Actual result", "Evidence"))
        if h == "Status":
            cell.fill = PatternFill("solid", fgColor=FILL.get(r["Status"], "FFFFFF"))
            cell.font = Font(bold=True, color=FONT.get(r["Status"], "000000"))
widths = {"Sheet": 22, "ID": 9, "Pri": 5, "Component": 18, "Feature under test": 42,
          "Expected result": 46, "Status": 11, "Actual result": 58, "Evidence": 44,
          "Dataset / fixture": 26, "Source (code · doc · PR)": 34, "Owner": 12,
          "Executed on": 18, "Branch": 20, "Defect ref": 12}
for i, h in enumerate(out_cols, 1):
    d.column_dimensions[get_column_letter(i)].width = widths.get(h, 18)
d.auto_filter.ref = f"A1:{get_column_letter(len(out_cols))}{len(rows)+1}"

# one sheet per area
for sheet in sorted({r["Sheet"] for r in rows}):
    sub = [r for r in rows if r["Sheet"] == sheet]
    name = sheet[:28].replace("/", "-")
    t = wb.create_sheet(name)
    cols = ["ID", "Pri", "Component", "Feature under test", "Expected result",
            "Status", "Actual result", "Evidence"]
    for i, h in enumerate(cols, 1):
        c = t.cell(row=1, column=i, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="44546A")
    t.freeze_panes = "A2"
    for ri, r in enumerate(sub, 2):
        for ci, h in enumerate(cols, 1):
            c = t.cell(row=ri, column=ci, value=r[h])
            c.alignment = Alignment(vertical="top", wrap_text=h in
                                    ("Feature under test", "Expected result", "Actual result", "Evidence"))
            if h == "Status":
                c.fill = PatternFill("solid", fgColor=FILL.get(r["Status"], "FFFFFF"))
                c.font = Font(bold=True, color=FONT.get(r["Status"], "000000"))
    for i, h in enumerate(cols, 1):
        t.column_dimensions[get_column_letter(i)].width = widths.get(h, 18)

xlsx_path = os.path.join(S, "Avaloka-1.6-Test-Report.xlsx")
wb.save(xlsx_path)
print(f"  wrote {xlsx_path}")
