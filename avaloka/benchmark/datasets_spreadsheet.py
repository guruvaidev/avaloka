"""Spreadsheet benchmark datasets — the shapes that break real readers.

The suite had no non-CSV dataset at all: eight of the nine upload formats the
API declares had zero coverage, so "we support .xlsx" was an unmeasured claim.
A CSV is already a table, which is exactly why a CSV-only benchmark cannot
catch the ingestion defects this project actually ships. A spreadsheet is a
canvas — a title on one row, the header three rows down, the table indented one
column to the right, two tables on one sheet, the units stated in prose — and a
reader that mishandles the canvas returns a frame that is *plausible* rather
than correct. Everything downstream then grades fine on it: that is how insight
text naming ``Unnamed: 13`` was scored 1.00.

Each builder here writes one such shape and declares a :class:`GroundTruth`
beside it, so the comprehension scorers can compare what the reader resolved
against what the file actually contains. The ground truth is declared by the
writer, independently of any reader, because a reader cannot be graded against
its own output.

Shape 1 (:func:`build_owner_file_shape`) is not invented: it reproduces the
geometry of the owner's own segment-review workbook, decoded from that file's
raw sheet XML — ``dimension ref="B2:G142"``, an empty column A, an absent first
row, a one-cell title on row 2, the header on row 3 and 139 data rows under it,
with the pane frozen at B4. It is the file that produced the ``Unnamed:``
insights, so it is the file the benchmark has to hold.

Conventions used throughout:

* Every builder has signature ``(dest: Path) -> Path``: it writes the workbook
  to ``dest`` and returns it. The runner names the file from the spec, so
  ``dest`` already ends ``.xlsx``.
* ``openpyxl`` is imported *inside* each builder. ``avaloka.benchmark`` is
  imported by the CLI, whose runtime dependencies must not grow, so this module
  stays importable without it.
* Nothing is random and nothing reads the clock. Values come from fixed tables
  and index arithmetic, so two runs produce byte-identical column names, row
  counts and values.
* ``GroundTruth.header_row`` is 1-based, as a human counts spreadsheet rows.
  Where a header spans two rows it is the row the header *starts* on, and
  ``shape`` says so.
* Where a shape's answer is a set of row labels rather than a single figure
  (footnote markers, the financial statements), those labels are carried in
  ``planted_answer`` under ``"row_labels"`` / ``"line_items"`` so a scorer can
  check them without parsing prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from avaloka.benchmark.spec import GroundTruth

__all__ = [
    "SPREADSHEET_DATASETS",
    "build_owner_file_shape",
    "build_merged_header_cells",
    "build_stacked_two_row_header",
    "build_multi_sheet_data_on_sheet_two",
    "build_thousands_scaling_in_title",
    "build_parenthesised_negatives",
    "build_footnote_markers_on_labels",
    "build_second_table_below",
    "build_year_columns_header",
    "build_sec_financial_report",
    "build_totals_row_at_bottom",
    "build_subtotals_and_grand_total",
]


# --------------------------------------------------------------------------
# Shared, fixed label tables. No randomness: the index is the generator.
# --------------------------------------------------------------------------
_PLACES: tuple[str, ...] = (
    "Aberdeen", "Bath", "Belfast", "Birmingham", "Brighton", "Bristol",
    "Cambridge", "Cardiff", "Carlisle", "Chester", "Coventry", "Derby",
    "Dundee", "Durham", "Edinburgh", "Exeter", "Glasgow", "Hull",
    "Inverness", "Leeds", "Leicester", "Lincoln", "Liverpool", "Manchester",
    "Newcastle", "Norwich", "Nottingham", "Oxford", "Plymouth", "Portsmouth",
)

_LEDGER_DEPTS: tuple[str, ...] = (
    "Operations", "Engineering", "Sales", "Marketing", "Finance",
    "Facilities", "Support",
)
_LEDGER_KINDS: tuple[str, ...] = ("Payroll", "Travel", "Software", "Consumables")


def _openpyxl() -> Any:
    """The openpyxl module, or an ImportError that says how to get it.

    Deferred to call time on purpose: the CLI imports ``avaloka.benchmark`` and
    must not acquire a spreadsheet dependency to do it.
    """
    try:
        import openpyxl
    except ImportError as exc:                      # pragma: no cover - env-dependent
        raise ImportError(
            "Building the spreadsheet benchmark datasets needs openpyxl, which "
            "is not installed. Install the benchmark extra: "
            "pip install 'avaloka[benchmark]'"
        ) from exc
    return openpyxl


def _write_row(worksheet: Any, row: int, first_column: int,
               values: list[Any]) -> None:
    """Write ``values`` left to right from (``row``, ``first_column``), 1-based.

    ``None`` leaves the cell untouched rather than writing an empty string: a
    cell that exists and holds "" is not blank to a reader, and several of these
    shapes depend on a cell being genuinely absent.
    """
    for offset, value in enumerate(values):
        if value is None:
            continue
        worksheet.cell(row=row, column=first_column + offset, value=value)


def _accounting(value: float) -> Any:
    """A figure as an accounting string: negatives parenthesised, as text.

    ``-1234`` becomes ``"(1,234)"``. Written as text because that is what the
    exports do -- the sign lives in the formatting, and a reader that takes the
    cell at face value reads a cost as a positive number and reports a total
    with the wrong sign.
    """
    if value < 0:
        return f"({abs(value):,.0f})"
    return float(value)


# --------------------------------------------------------------------------
# 1. The owner's own file
# --------------------------------------------------------------------------
_OWNER_SEGMENT_KINDS: tuple[str, ...] = (
    "approach", "grasp", "transfer", "align", "insert", "retract", "inspect",
)
_OWNER_FORESIGHT: tuple[str, ...] = (
    "move to pick pose", "close gripper on part", "lift clear of fixture",
    "rotate to insertion angle", "seat part in socket", "release and withdraw",
    "hold for camera check", "re-approach after slip",
)
_OWNER_HINDSIGHT: tuple[str, ...] = (
    "reached pose, no contact", "part held, slight offset",
    "lifted, fixture cleared", "angle short by a few degrees",
    "seated after second attempt", "withdrew cleanly",
    "check passed", "slip recovered",
)


def build_owner_file_shape(dest: Path) -> Path:
    """The owner's segment-review workbook: table at B2, header on row 3.

    Column A is entirely empty, sheet row 1 does not exist, row 2 holds a
    one-cell title and the header sits on row 3 -- ``dimension ref="B2:G142"``,
    pane frozen at B4, 139 data rows. ``pd.read_excel`` takes row 1 as the
    header regardless, so every column comes back ``Unnamed: N``, the title
    becomes a record and the frame is one row longer than the table. This is
    the file whose profile produced insight sentences about ``Unnamed: 13``,
    and the reason this module exists.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Segments"

    # Row 2, column B: the title, alone in its row.
    sheet.cell(row=2, column=2, value="Hindsight segment review - run 11 (operator notes)")

    header = ["Segment", "Start (s)", "End (s)",
              "Foresight (intended action)", "Hindsight (achieved)",
              "Optimality (1-5)"]
    _write_row(sheet, 3, 2, header)

    for i in range(139):
        start = round(i * 4.25, 2)
        _write_row(sheet, 4 + i, 2, [
            f"{_OWNER_SEGMENT_KINDS[i % len(_OWNER_SEGMENT_KINDS)]} {i + 1:03d}",
            start,
            round(start + 3.75, 2),
            _OWNER_FORESIGHT[(i * 3) % len(_OWNER_FORESIGHT)],
            _OWNER_HINDSIGHT[(i * 5) % len(_OWNER_HINDSIGHT)],
            1 + (i * 3) % 5,
        ])

    sheet.freeze_panes = "B4"
    workbook.save(dest)
    return dest


_OWNER_GROUND_TRUTH = GroundTruth(
    columns=["Segment", "Start (s)", "End (s)",
             "Foresight (intended action)", "Hindsight (achieved)",
             "Optimality (1-5)"],
    n_rows=139,
    header_row=3,
    numeric_columns=["Start (s)", "End (s)", "Optimality (1-5)"],
    text_columns=["Segment", "Foresight (intended action)",
                  "Hindsight (achieved)"],
    shape=("Table occupies B2:G142. Column A is empty and sheet row 1 is "
           "absent; row 2 is a one-cell title; the header is B3:G3; 139 data "
           "rows in B4:G142; pane frozen at B4. Reproduces the geometry of the "
           "owner's own segment-review workbook."),
)


# --------------------------------------------------------------------------
# 2. Merged header cells
# --------------------------------------------------------------------------
def build_merged_header_cells(dest: Path) -> Path:
    """A header whose group labels are merged across columns.

    Row 1 is ``Store | <Transactions spanning B:C> | <Refunds spanning D:E>``
    and row 2 carries the sub-labels ``Count`` / ``Value`` under each span. A
    merged range stores its value only in the top-left cell, so the header row
    reads ``Store | Transactions | None | Refunds | None``: the reader either
    names two real columns after nothing, or -- because row 1 is now half empty
    -- rejects row 1 as a header and starts the table on it. Either way
    ``Transactions Value`` and ``Refunds Value`` lose their identity, and a
    total of refunds is indistinguishable from a total of transactions.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Till summary"

    _write_row(sheet, 1, 1, ["Store", "Transactions", None, "Refunds", None])
    _write_row(sheet, 2, 1, [None, "Count", "Value", "Count", "Value"])
    sheet.merge_cells(start_row=1, start_column=2, end_row=1, end_column=3)
    sheet.merge_cells(start_row=1, start_column=4, end_row=1, end_column=5)

    for i in range(24):
        transactions = 1200 + i * 37
        refunds = 18 + (i * 5) % 31
        _write_row(sheet, 3 + i, 1, [
            _PLACES[i],
            transactions,
            round(transactions * 11.4 + i, 2),
            refunds,
            round(refunds * 12.1 + i, 2),
        ])

    workbook.save(dest)
    return dest


_MERGED_GROUND_TRUTH = GroundTruth(
    columns=["Store", "Transactions Count", "Transactions Value",
             "Refunds Count", "Refunds Value"],
    n_rows=24,
    header_row=1,
    numeric_columns=["Transactions Count", "Transactions Value",
                     "Refunds Count", "Refunds Value"],
    text_columns=["Store"],
    shape=("Two-row header in rows 1-2 with merges across columns: B1:C1 is "
           "one merged cell holding 'Transactions', D1:E1 one merged cell "
           "holding 'Refunds'; row 2 holds 'Count'/'Value' under each span. "
           "The correct read carries the group label down onto each sub-label "
           "('Transactions Count'). header_row is the first of the two."),
)


# --------------------------------------------------------------------------
# 3. Stacked two-row header
# --------------------------------------------------------------------------
def build_stacked_two_row_header(dest: Path) -> Path:
    """A two-row header with the group label repeated rather than merged.

    Row 1 is ``Region | Sales | Sales | Cost | Cost`` and row 2 is
    ``| 2022 | 2023 | 2022 | 2023``. The unmerged twin of the shape above, and
    it fails differently: row 1 is full, so a reader that counts filled cells
    accepts it, and the frame then has two columns called ``Sales`` and two
    called ``Cost`` -- or ``Sales`` and ``Sales_1``, which says nothing about
    which year is which. Row 2 is consumed as the first data record, so the
    row count is one too high and every numeric column is typed from a row of
    years. Header detection that rejects repeated labels instead returns no
    header at all and synthesises ``column_1..column_5``.

    The flattened names declared below ('<measure> <year>') are the choice this
    benchmark grades against; a reader that produces a different but equally
    unambiguous flattening is still wrong against this ground truth, so the
    choice is stated rather than implied.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Sales"

    _write_row(sheet, 1, 1, ["Region", "Sales", "Sales", "Cost", "Cost"])
    _write_row(sheet, 2, 1, [None, "2022", "2023", "2022", "2023"])

    for i in range(24):
        sales_2022 = 480_000 + i * 12_500
        sales_2023 = sales_2022 + 23_000 + (i % 7) * 1_750
        _write_row(sheet, 3 + i, 1, [
            _PLACES[i],
            float(sales_2022),
            float(sales_2023),
            round(sales_2022 * 0.62, 2),
            round(sales_2023 * 0.59, 2),
        ])

    workbook.save(dest)
    return dest


_STACKED_GROUND_TRUTH = GroundTruth(
    columns=["Region", "Sales 2022", "Sales 2023", "Cost 2022", "Cost 2023"],
    n_rows=24,
    header_row=1,
    numeric_columns=["Sales 2022", "Sales 2023", "Cost 2022", "Cost 2023"],
    text_columns=["Region"],
    shape=("Header spans rows 1-2 with no merge: row 1 repeats the measure "
           "('Region | Sales | Sales | Cost | Cost'), row 2 holds the year "
           "under each ('| 2022 | 2023 | 2022 | 2023'). The flattening graded "
           "here is '<measure> <year>'. header_row is the first of the two; 24 "
           "data rows start at sheet row 3."),
)


# --------------------------------------------------------------------------
# 4. Three sheets, data on the second
# --------------------------------------------------------------------------
def build_multi_sheet_data_on_sheet_two(dest: Path) -> Path:
    """The table is on sheet 2; sheet 1 is a cover, sheet 3 a legend.

    Sheet ordering is a reporting convention, not a statement about where the
    data is. A reader that takes ``sheet_name=0`` profiles the cover: five
    key/value cells become a two-column table of one record, and the mission
    then analyses the report's metadata instead of the report. The app path
    does exactly that. Picking "the sheet with the most cells" happens to land
    on the right sheet here, which is worth knowing but is not the same as
    reading the workbook -- a legend longer than the table flips it.
    """
    workbook = _openpyxl().Workbook()

    cover = workbook.active
    cover.title = "Cover"
    for row, (key, value) in enumerate([
        ("Report", "Subscription usage by account"),
        ("Period", "FY2023 Q4"),
        ("Prepared by", "Revenue operations"),
        ("Source system", "billing-prod"),
        ("Version", "3"),
    ], start=1):
        _write_row(cover, row, 1, [key, value])

    data = workbook.create_sheet("Usage")
    _write_row(data, 1, 1,
               ["Account", "Plan", "Seats", "Monthly cost", "Active users"])
    plans = ("starter", "team", "business")
    for i in range(30):
        seats = 5 + (i * 7) % 95
        _write_row(data, 2 + i, 1, [
            f"{_PLACES[i]} Ltd",
            plans[i % 3],
            seats,
            round(seats * (9.0 + 6.0 * (i % 3)), 2),
            max(1, seats - (i % 11)),
        ])

    legend = workbook.create_sheet("Legend")
    _write_row(legend, 1, 1, ["Plan", "Included seats"])
    for row, (plan, included) in enumerate(
            [("starter", 5), ("team", 25), ("business", 100)], start=2):
        _write_row(legend, row, 1, [plan, included])

    workbook.save(dest)
    return dest


_MULTI_SHEET_GROUND_TRUTH = GroundTruth(
    columns=["Account", "Plan", "Seats", "Monthly cost", "Active users"],
    n_rows=30,
    sheet_name="Usage",
    header_row=1,
    numeric_columns=["Seats", "Monthly cost", "Active users"],
    text_columns=["Account", "Plan"],
    shape=("Three sheets: 'Cover' (five key/value cells), 'Usage' (the real "
           "table, header on its row 1, 30 data rows) and 'Legend' (a "
           "three-row plan key). The ground truth describes sheet 2, 'Usage'."),
)


# --------------------------------------------------------------------------
# 5. Scale stated only in the title
# --------------------------------------------------------------------------
def build_thousands_scaling_in_title(dest: Path) -> Path:
    """Figures in thousands, with the scale stated only in prose.

    Row 1 says ``Revenue by region (£ thousands)``; the cells below hold 482.5,
    not 482,500. Nothing in the header, the column names or the cell formatting
    carries the factor, so a reader that discards the title row -- which is the
    correct thing to do structurally -- discards the units with it and reports
    a revenue total three orders of magnitude too small. The planted answer is
    the total at full scale, so a reader that drops the title cannot reach it.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Revenue"

    sheet.cell(row=1, column=1, value="Revenue by region (£ thousands)")
    _write_row(sheet, 2, 1, ["Region", "Revenue", "Direct cost"])

    total_thousands = 0.0
    for i in range(24):
        revenue = round(482.5 + i * 37.25, 2)
        total_thousands += revenue
        _write_row(sheet, 3 + i, 1, [
            _PLACES[i], revenue, round(revenue * 0.63, 2),
        ])

    workbook.save(dest)
    return dest


def _thousands_total() -> float:
    """Total revenue in pounds, i.e. the sheet's figures times 1000."""
    return round(sum(round(482.5 + i * 37.25, 2) for i in range(24)) * 1000.0, 2)


_THOUSANDS_GROUND_TRUTH = GroundTruth(
    columns=["Region", "Revenue", "Direct cost"],
    n_rows=24,
    header_row=2,
    numeric_columns=["Revenue", "Direct cost"],
    text_columns=["Region"],
    planted_answer={
        "label": "Total revenue",
        "value": _thousands_total(),
        "unit": "GBP",
        "note": ("The cells hold thousands. The correct total is the sum of "
                 "the Revenue column multiplied by 1000; a reader that reports "
                 "the raw sum is out by a factor of 1000."),
    },
    shape=("Row 1 is a one-cell title, 'Revenue by region (GBP thousands)', "
           "and is the only place the scale appears -- not in the header, not "
           "in the column names, not in the number formats. Header on row 2, "
           "24 data rows from row 3."),
)


# --------------------------------------------------------------------------
# 6. Parenthesised negatives
# --------------------------------------------------------------------------
def build_parenthesised_negatives(dest: Path) -> Path:
    """Accounting negatives written ``(1,234)``, mixed with plain positives.

    The sign is in the punctuation. ``(1,234)`` is text, so the column arrives
    as ``object`` and is either skipped by every numeric statistic or parsed as
    +1234 -- and a movement column read with the wrong sign makes the closing
    balance disagree with opening + movement while every individual number
    looks right. The planted answer is the net movement, which is only
    reachable if the brackets were read as a minus sign. Positives are left as
    real numbers in the same column, so a reader cannot get this right by
    treating the whole column as text either.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Ledger"

    _write_row(sheet, 1, 1,
               ["Account", "Opening balance", "Movement", "Closing balance"])

    for i, (dept, kind) in enumerate(
            (d, k) for d in _LEDGER_DEPTS for k in _LEDGER_KINDS):
        opening = 120_000 + i * 4_300
        movement = _ledger_movement(i)
        _write_row(sheet, 2 + i, 1, [
            f"{dept} - {kind}",
            float(opening),
            _accounting(movement),
            _accounting(opening + movement),
        ])

    workbook.save(dest)
    return dest


def _ledger_movement(i: int) -> int:
    """The signed movement for ledger row ``i``. Every third row is a credit."""
    magnitude = 1_250 + (i * 917) % 48_000
    return -magnitude if i % 3 != 0 else magnitude


def _ledger_net_movement() -> float:
    return float(sum(_ledger_movement(i) for i in range(28)))


_NEGATIVES_GROUND_TRUTH = GroundTruth(
    columns=["Account", "Opening balance", "Movement", "Closing balance"],
    n_rows=28,
    header_row=1,
    numeric_columns=["Opening balance", "Movement", "Closing balance"],
    text_columns=["Account"],
    planted_answer={
        "label": "Net movement",
        "column": "Movement",
        "value": _ledger_net_movement(),
        "note": ("Sum of the Movement column. Only reachable if '(1,234)' was "
                 "read as -1234; reading the brackets as decoration gives a "
                 "positive total of the absolute values."),
    },
    shape=("Header on row 1, 28 data rows. Movement and Closing balance mix "
           "plain numeric positives with accounting negatives written as text "
           "'(1,234)' meaning -1234. Closing balance = Opening balance + "
           "Movement for every row, so a mis-signed read is internally "
           "inconsistent."),
)


# --------------------------------------------------------------------------
# 7. Footnote markers glued to labels
# --------------------------------------------------------------------------
_FOOTNOTE_MARKERS: tuple[str, ...] = ("*", " (a)", "†", "")


def build_footnote_markers_on_labels(dest: Path) -> Path:
    """Footnote markers on header and row labels, with a footnote block below.

    ``Headcount (a)`` and ``Attrition rate†`` are the same columns as
    ``Headcount`` and ``Attrition rate``, and ``Bath*`` is the same branch as
    ``Bath`` -- but not to anything matching on names, so a question about
    attrition finds no such column and a join against clean reference data
    drops the marked rows. The three-line footnote block at the bottom is worse
    than cosmetic: separated by a single blank row, it is pulled in as three
    more data records, so the row count is 25 instead of 22 and the text column
    acquires three sentences as values.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Headcount"

    _write_row(sheet, 1, 1,
               ["Segment*", "Headcount (a)", "Attrition rate†", "Notes"])

    for i in range(22):
        _write_row(sheet, 2 + i, 1, [
            f"{_PLACES[i]}{_FOOTNOTE_MARKERS[i % len(_FOOTNOTE_MARKERS)]}",
            _footnote_headcount(i),
            round(0.04 + (i % 9) * 0.011, 3),
            "reviewed" if i % 2 == 0 else "pending review",
        ])

    # One blank row, then the footnotes. A single blank row is a spacer to
    # every block-splitter in the codebase, which is why this gets absorbed.
    for offset, note in enumerate([
        "* excludes contractors and agency staff",
        "(a) headcount as at the last day of the period",
        "† attrition rate is trailing twelve months",
    ]):
        sheet.cell(row=25 + offset, column=1, value=note)

    workbook.save(dest)
    return dest


def _footnote_headcount(i: int) -> int:
    return 48 + (i * 13) % 240


_FOOTNOTES_GROUND_TRUTH = GroundTruth(
    columns=["Segment", "Headcount", "Attrition rate", "Notes"],
    n_rows=22,
    header_row=1,
    numeric_columns=["Headcount", "Attrition rate"],
    text_columns=["Segment", "Notes"],
    planted_answer={
        "label": "Bath",
        "column": "Headcount",
        "value": float(_footnote_headcount(1)),
        "row_labels": list(_PLACES[:22]),
        "note": ("The second row's label is written 'Bath*'. A correct reader "
                 "resolves it to 'Bath'; row_labels is the full clean set."),
    },
    shape=("Header on row 1 with markers glued on: 'Segment*', "
           "'Headcount (a)', 'Attrition rate<dagger>'. 22 data rows whose "
           "labels carry the same markers cyclically. Sheet row 24 is blank "
           "and rows 25-27 are a three-line footnote block in column A, which "
           "is not data. The declared columns and row_labels are the clean "
           "names, markers stripped."),
)


# --------------------------------------------------------------------------
# 8. A second table further down the sheet
# --------------------------------------------------------------------------
def build_second_table_below(dest: Path) -> Path:
    """Two tables stacked on one sheet, separated by two blank rows.

    Read as one frame the sheet becomes a 20-row table with two empty records
    in the middle followed by eight records of a different shape, whose values
    land under column names that do not describe them: the second table's
    ``Owner`` strings arrive in ``Hours open``, so a numeric column becomes
    object and its statistics vanish. The ground truth describes the first
    table only; a reader returning 30 rows has returned two tables glued
    together, not this one.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Tickets"

    _write_row(sheet, 1, 1, ["Ticket", "Queue", "Hours open", "Reopened"])
    queues = ("billing", "platform", "onboarding", "security")
    for i in range(20):
        _write_row(sheet, 2 + i, 1, [
            f"TK-{4100 + i * 7}",
            queues[i % 4],
            round(1.5 + (i * 3.25) % 71, 2),
            int(i % 5 == 0),
        ])

    # Rows 22 and 23 left entirely blank, then a differently-shaped table.
    _write_row(sheet, 24, 1, ["Queue", "Target hours", "Owner"])
    targets = ((("billing", 8), ("platform", 4), ("onboarding", 24),
                ("security", 2), ("billing-eu", 8), ("platform-eu", 4),
                ("onboarding-eu", 24), ("security-eu", 2)))
    for offset, (queue, target) in enumerate(targets):
        _write_row(sheet, 25 + offset, 1, [
            queue, target, f"{_PLACES[offset]} team",
        ])

    workbook.save(dest)
    return dest


_SECOND_TABLE_GROUND_TRUTH = GroundTruth(
    columns=["Ticket", "Queue", "Hours open", "Reopened"],
    n_rows=20,
    header_row=1,
    numeric_columns=["Hours open", "Reopened"],
    text_columns=["Ticket", "Queue"],
    shape=("First table: header on row 1, 20 data rows (sheet rows 2-21). "
           "Sheet rows 22-23 are entirely blank. A second, unrelated table "
           "starts at sheet row 24 with its own header "
           "'Queue | Target hours | Owner' and 8 data rows (sheet rows "
           "25-32). The ground truth describes the first table only."),
)


# --------------------------------------------------------------------------
# 9. Year columns in the header
# --------------------------------------------------------------------------
def build_year_columns_header(dest: Path) -> Path:
    """A header of years: ``Region | 2021 | 2022 | 2023`` on row 1.

    The most ordinary report layout there is, and it defeats header detection
    outright. ``table_shape.detect_header_row`` asks ``_looks_like_a_label`` of
    each header cell, and that returns False for anything that parses as a
    number -- "2021" does. One label in four cells fails the
    ``len(labels) < len(cells) * 0.7`` test, so the function concludes the data
    starts on row 1 and returns None; ``reshape`` then synthesises
    ``column_1..column_4`` and the years are gone from the schema entirely. The
    ground truth pins the four real names so the check fails on ``column_N``
    rather than passing on a plausible-looking frame.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "By year"

    # Written as strings, which is what a year header is in these exports: the
    # cell says "2021", it is not the number 2021.
    _write_row(sheet, 1, 1, ["Region", "2021", "2022", "2023"])

    for i in range(20):
        base = 310_000 + i * 18_400
        _write_row(sheet, 2 + i, 1, [
            _PLACES[i],
            float(base),
            float(base + 21_500 + (i % 5) * 1_900),
            float(base + 44_700 + (i % 7) * 2_300),
        ])

    workbook.save(dest)
    return dest


_YEAR_COLUMNS_GROUND_TRUTH = GroundTruth(
    columns=["Region", "2021", "2022", "2023"],
    n_rows=20,
    header_row=1,
    numeric_columns=["2021", "2022", "2023"],
    text_columns=["Region"],
    shape=("Header on row 1: 'Region | 2021 | 2022 | 2023', the year cells "
           "stored as text. 20 data rows. Three of the four header cells "
           "parse as numbers, which is what makes "
           "table_shape.detect_header_row return None and reshape synthesise "
           "column_1..column_4."),
)


# --------------------------------------------------------------------------
# 10. A synthetic SEC Financial_Report.xlsx
# --------------------------------------------------------------------------
_SEC_PERIODS: tuple[str, ...] = ("Dec. 31, 2023", "Dec. 31, 2022", "Dec. 31, 2021")

#: The independent figures per period, in thousands. Everything else on the
#: statement is derived from these, so the sheet is internally consistent and a
#: reader that understood it can be checked line against line.
_SEC_INPUTS: dict[str, dict[str, int]] = {
    "Dec. 31, 2023": {"revenue": 4_812_000, "cost_of_revenue": 1_684_200,
                      "research": 962_400, "selling": 721_800,
                      "general": 384_960, "interest": 48_120,
                      "other_income": 21_400, "tax": 216_703},
    "Dec. 31, 2022": {"revenue": 4_105_000, "cost_of_revenue": 1_478_800,
                      "research": 861_050, "selling": 657_500,
                      "general": 352_030, "interest": 52_400,
                      "other_income": 15_200, "tax": 147_050},
    "Dec. 31, 2021": {"revenue": 3_402_000, "cost_of_revenue": 1_292_760,
                      "research": 748_440, "selling": 578_340,
                      "general": 306_180, "interest": 59_800,
                      "other_income": 9_600, "tax": 89_600},
}

#: Balance-sheet independents, in thousands, for the two periods it presents.
_SEC_BALANCE_PERIODS: tuple[str, ...] = ("Dec. 31, 2023", "Dec. 31, 2022")
_SEC_BALANCE: dict[str, dict[str, int]] = {
    "Dec. 31, 2023": {"cash": 1_240_500, "receivables": 684_300,
                      "inventory": 212_800, "ppe": 1_480_200,
                      "goodwill": 962_000, "payables": 318_400,
                      "accrued": 264_700, "debt": 1_100_000,
                      "common_stock": 480_000, "aoci_loss": 37_200},
    "Dec. 31, 2022": {"cash": 980_400, "receivables": 601_250,
                      "inventory": 198_300, "ppe": 1_352_700,
                      "goodwill": 962_000, "payables": 291_600,
                      "accrued": 243_900, "debt": 1_250_000,
                      "common_stock": 465_000, "aoci_loss": 41_900},
}

_SEC_OPERATIONS_LABELS: tuple[str, ...] = (
    "Revenue",
    "Cost of revenue (a)",
    "Gross profit",
    "Research and development",
    "Sales and marketing",
    "General and administrative",
    "Total operating expenses",
    "Operating income",
    "Interest expense",
    "Other income, net",
    "Income before income taxes",
    "Provision for income taxes",
    "Net income",
)

#: What the labels above mean once the footnote marker is resolved.
_SEC_OPERATIONS_CLEAN_LABELS: tuple[str, ...] = tuple(
    "Cost of revenue" if label == "Cost of revenue (a)" else label
    for label in _SEC_OPERATIONS_LABELS
)


def _sec_operations(period: str) -> dict[str, float]:
    """One period of the income statement, derived from the independents."""
    v = _SEC_INPUTS[period]
    gross = v["revenue"] - v["cost_of_revenue"]
    opex = v["research"] + v["selling"] + v["general"]
    operating = gross - opex
    pretax = operating - v["interest"] + v["other_income"]
    return {
        "Revenue": float(v["revenue"]),
        "Cost of revenue (a)": float(-v["cost_of_revenue"]),
        "Gross profit": float(gross),
        "Research and development": float(-v["research"]),
        "Sales and marketing": float(-v["selling"]),
        "General and administrative": float(-v["general"]),
        "Total operating expenses": float(-opex),
        "Operating income": float(operating),
        "Interest expense": float(-v["interest"]),
        "Other income, net": float(v["other_income"]),
        "Income before income taxes": float(pretax),
        "Provision for income taxes": float(-v["tax"]),
        "Net income": float(pretax - v["tax"]),
    }


_SEC_BALANCE_LABELS: tuple[str, ...] = (
    "Cash and cash equivalents",
    "Accounts receivable",
    "Inventory",
    "Total current assets",
    "Property and equipment, net",
    "Goodwill",
    "Total assets",
    "Accounts payable",
    "Accrued liabilities",
    "Total current liabilities",
    "Long-term debt",
    "Total liabilities",
    "Common stock",
    "Retained earnings",
    "Accumulated other comprehensive loss",
    "Total stockholders' equity",
    "Total liabilities and stockholders' equity",
)


def _sec_balance(period: str) -> dict[str, float]:
    """One period of the balance sheet, derived so that it balances.

    Liabilities are presented positive, as EDGAR presents them; the one
    negative line is the accumulated other comprehensive loss, which is where a
    real balance sheet puts its brackets.
    """
    v = _SEC_BALANCE[period]
    current_assets = v["cash"] + v["receivables"] + v["inventory"]
    total_assets = current_assets + v["ppe"] + v["goodwill"]
    current_liabilities = v["payables"] + v["accrued"]
    total_liabilities = current_liabilities + v["debt"]
    equity = total_assets - total_liabilities
    return {
        "Cash and cash equivalents": float(v["cash"]),
        "Accounts receivable": float(v["receivables"]),
        "Inventory": float(v["inventory"]),
        "Total current assets": float(current_assets),
        "Property and equipment, net": float(v["ppe"]),
        "Goodwill": float(v["goodwill"]),
        "Total assets": float(total_assets),
        "Accounts payable": float(v["payables"]),
        "Accrued liabilities": float(v["accrued"]),
        "Total current liabilities": float(current_liabilities),
        "Long-term debt": float(v["debt"]),
        "Total liabilities": float(total_liabilities),
        "Common stock": float(v["common_stock"]),
        "Retained earnings": float(equity - v["common_stock"] + v["aoci_loss"]),
        "Accumulated other comprehensive loss": float(-v["aoci_loss"]),
        "Total stockholders' equity": float(equity),
        "Total liabilities and stockholders' equity": float(total_liabilities + equity),
    }


def build_sec_financial_report(dest: Path) -> Path:
    """A workbook shaped like EDGAR's ``Financial_Report.xlsx``.

    Every defect above at once, which is what makes it the shape a 10-K
    actually has: a cover sheet first so ``sheet_name=0`` reads metadata, a
    title row carrying the scale ("$ in Thousands") that no header or format
    repeats, period labels ("Dec. 31, 2023") as column names, line-item labels
    down the first column rather than values, costs written ``(1,684,200)``,
    and a footnote marker on 'Cost of revenue (a)'. The frame a naive read
    returns is the transpose of the table a question is asked about: "what was
    net income in 2023" is a cell lookup by row label, and a reader that lost
    the row labels or the sign cannot answer it even though it returned a
    frame.

    The arithmetic is derived, not typed: gross profit is revenue minus cost of
    revenue, operating income is gross profit minus total operating expenses,
    net income is pre-tax income minus the tax provision, and total assets
    equals total liabilities plus equity. A reader that understood the
    statement can therefore be checked against it line by line.
    """
    workbook = _openpyxl().Workbook()

    cover = workbook.active
    cover.title = "Cover"
    for row, (key, value) in enumerate([
        ("Document Type", "10-K"),
        ("Document Fiscal Year Focus", "2023"),
        ("Document Period End Date", "Dec. 31, 2023"),
        ("Entity Registrant Name", "NORTHWIND INSTRUMENTS INC"),
        ("Entity Central Index Key", "0000000000"),
        ("Amounts in", "thousands, except share data"),
    ], start=1):
        _write_row(cover, row, 1, [key, value])

    operations = workbook.create_sheet("CONSOLIDATED STATEMENTS OF OPER")
    operations.cell(row=1, column=1,
                    value="CONSOLIDATED STATEMENTS OF OPER - USD ($) "
                          "$ in Thousands")
    _write_row(operations, 2, 1, ["Line item", *_SEC_PERIODS])
    for offset, label in enumerate(_SEC_OPERATIONS_LABELS):
        _write_row(operations, 3 + offset, 1, [
            label,
            *[_accounting(_sec_operations(period)[label])
              for period in _SEC_PERIODS],
        ])
    operations.cell(row=3 + len(_SEC_OPERATIONS_LABELS) + 1, column=1,
                    value="(a) Cost of revenue excludes amortisation of "
                          "acquired intangibles.")

    balance = workbook.create_sheet("Consolidated Balance Sheets")
    balance.cell(row=1, column=1,
                 value="Consolidated Balance Sheets - USD ($) $ in Thousands")
    _write_row(balance, 2, 1, ["Line item", *_SEC_BALANCE_PERIODS])
    for offset, label in enumerate(_SEC_BALANCE_LABELS):
        _write_row(balance, 3 + offset, 1, [
            label,
            *[_accounting(_sec_balance(period)[label])
              for period in _SEC_BALANCE_PERIODS],
        ])

    workbook.save(dest)
    return dest


_SEC_GROUND_TRUTH = GroundTruth(
    columns=["Line item", *_SEC_PERIODS],
    n_rows=len(_SEC_OPERATIONS_LABELS),
    header_row=2,
    sheet_name="CONSOLIDATED STATEMENTS OF OPER",
    numeric_columns=list(_SEC_PERIODS),
    text_columns=["Line item"],
    planted_answer={
        "label": "Net income",
        "period": "Dec. 31, 2023",
        "value": _sec_operations("Dec. 31, 2023")["Net income"],
        "unit": "thousands",
        "line_items": list(_SEC_OPERATIONS_CLEAN_LABELS),
        "note": ("Net income for the latest period, in thousands, as presented "
                 "on 'CONSOLIDATED STATEMENTS OF OPER'. Reachable only "
                 "if the period columns were read as columns, the line-item "
                 "labels as row labels, and '(216,703)' as a negative. "
                 "line_items is the full statement with the footnote marker "
                 "resolved off 'Cost of revenue (a)'."),
    },
    shape=("Three sheets: 'Cover' (six key/value cells of filing metadata), "
           "'CONSOLIDATED STATEMENTS OF OPER' and 'Consolidated Balance "
           "Sheets'. The ground truth describes the operations statement: "
           "sheet row 1 is the title and the only statement of scale ('$ in "
           "Thousands'), row 2 is the header 'Line item | Dec. 31, 2023 | Dec. "
           "31, 2022 | Dec. 31, 2021', rows 3-15 are the 13 line items, and "
           "sheet row 17 is a footnote resolving the '(a)' on 'Cost of "
           "revenue'. Costs are written as accounting negatives, e.g. "
           "'(1,684,200)'. Arithmetic is derived: gross profit = revenue - "
           "cost of revenue, operating income = gross profit - total operating "
           "expenses, net income = income before income taxes - provision for "
           "income taxes; on the balance sheet total assets = total "
           "liabilities + total stockholders' equity."),
)


# --------------------------------------------------------------------------
# The registry the suite binds tasks from.
# --------------------------------------------------------------------------
#: name -> (builder, declared ground truth). Keyed by a short stable name so
#: suite tasks can be built from it without hand-copying ground truth: a
#: builder and the truth about what it wrote cannot drift apart if they are
#: only ever read from the same entry.
# --------------------------------------------------------------------------
# Sheet choice: the decoy is bigger than the answer
# --------------------------------------------------------------------------
def build_decoy_sheet_outweighs_the_answer(dest: Path) -> Path:
    """The sheet holding the answer is the smaller sheet in the workbook.

    Every multi-sheet workbook in this module so far is only *incidentally*
    adversarial: in ``sec_financial_report`` the balance sheet has 19 rows
    against the income statement's 17, which fools the upload path (it ranks by
    row count) but not the CLI (which ranks by non-empty cells, and the income
    statement has 58 against 55). A defect that one reader happens to dodge is
    not pinned.

    So this file is built to defeat **both** heuristics at once. ``Reference
    data`` is longer *and* denser than ``Quarterly summary``, by a wide margin
    in both, while the question -- and the planted figure -- is on
    ``Quarterly summary``. There is no size heuristic that lands on the right
    sheet here, which is the point: the choice has to be made on something
    other than size, and neither reader has anything else to make it on.

    Nothing about the shape is unusual. A reference or lookup tab dwarfing the
    summary someone actually reads is the ordinary condition of a working
    spreadsheet.
    """
    openpyxl = _openpyxl()
    workbook = openpyxl.Workbook()

    answer = workbook.active
    answer.title = ANSWER_SHEET
    answer["A1"] = "Quarterly summary (GBP)"
    for column, label in enumerate(["Quarter", "Revenue", "Headcount"], start=1):
        answer.cell(row=2, column=column, value=label)
    quarters = ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4"]
    revenues = [412_500, 438_200, 455_900, 470_400]
    headcounts = [96, 101, 107, 112]
    for offset, (quarter, revenue, headcount) in enumerate(
            zip(quarters, revenues, headcounts)):
        answer.cell(row=3 + offset, column=1, value=quarter)
        answer.cell(row=3 + offset, column=2, value=revenue)
        answer.cell(row=3 + offset, column=3, value=headcount)

    # The decoy: a currency/region lookup table, of the kind that sits behind a
    # summary and is never the thing being asked about.
    decoy = workbook.create_sheet(DECOY_SHEET)
    decoy["A1"] = "Reference data - do not edit"
    for column, label in enumerate(
            ["Code", "Region", "Currency", "FX rate", "Active"], start=1):
        decoy.cell(row=2, column=column, value=label)
    regions = ["EMEA", "APAC", "LATAM", "NA"]
    currencies = ["EUR", "JPY", "BRL", "USD"]
    for index in range(40):
        decoy.cell(row=3 + index, column=1, value=f"R{index + 100:03d}")
        decoy.cell(row=3 + index, column=2, value=regions[index % 4])
        decoy.cell(row=3 + index, column=3, value=currencies[index % 4])
        decoy.cell(row=3 + index, column=4, value=round(1.0 + index / 50.0, 4))
        decoy.cell(row=3 + index, column=5, value="yes" if index % 3 else "no")

    workbook.save(dest)
    return dest


#: Named so the ground truth and the builder cannot disagree about which sheet
#: is which -- the whole check is an equality on this string.
ANSWER_SHEET = "Quarterly summary"
DECOY_SHEET = "Reference data"

DECOY_SHEET_TRUTH = GroundTruth(
    columns=["Quarter", "Revenue", "Headcount"],
    n_rows=4,
    header_row=2,
    sheet_name=ANSWER_SHEET,
    numeric_columns=["Revenue", "Headcount"],
    text_columns=["Quarter"],
    planted_answer={
        "label": "Total revenue",
        "value": 1_777_000.0,
        "row_labels": ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4"],
    },
    shape=("two sheets: the answer is on the smaller one ('Quarterly summary', "
           "4 data rows), while the decoy ('Reference data', 40 rows) is longer "
           "and denser, so both size heuristics pick the wrong sheet"),
)


# --------------------------------------------------------------------------
# 12. A totals row under the data
# --------------------------------------------------------------------------
def _totals_units(i: int) -> int:
    return 120 + (i * 37) % 260


def _totals_revenue(i: int) -> float:
    return round(_totals_units(i) * (9.5 + (i % 4) * 1.25), 2)


_TOTALS_N = 12


def build_totals_row_at_bottom(dest: Path) -> Path:
    """``Region | Units | Revenue``, twelve records, then ``Total | ... | ...``.

    The worst kind of wrong, because nothing looks wrong. The header is on row
    1, the names are right, the types are right and the frame prints cleanly --
    there is no ``Unnamed: 3`` to notice. But the totals row is read as a
    thirteenth region, so every sum is exactly double, the maximum of each
    column *is* the total, and the mean is inflated. A doubled revenue figure
    is the kind of number that goes into a board pack.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "Sales"

    _write_row(sheet, 1, 1, ["Region", "Units", "Revenue"])
    for i in range(_TOTALS_N):
        _write_row(sheet, 2 + i, 1,
                   [_PLACES[i], _totals_units(i), _totals_revenue(i)])
    _write_row(sheet, 2 + _TOTALS_N, 1, [
        "Total",
        sum(_totals_units(i) for i in range(_TOTALS_N)),
        round(sum(_totals_revenue(i) for i in range(_TOTALS_N)), 2),
    ])

    workbook.save(dest)
    return dest


_TOTALS_GROUND_TRUTH = GroundTruth(
    columns=["Region", "Units", "Revenue"],
    n_rows=_TOTALS_N,
    header_row=1,
    numeric_columns=["Units", "Revenue"],
    text_columns=["Region"],
    planted_answer={
        "column": "Revenue",
        "sum": round(sum(_totals_revenue(i) for i in range(_TOTALS_N)), 2),
        "max": max(_totals_revenue(i) for i in range(_TOTALS_N)),
        "units_sum": sum(_totals_units(i) for i in range(_TOTALS_N)),
        "row_labels": list(_PLACES[:_TOTALS_N]),
        "note": ("The sheet's last row is 'Total'. It is not a region: a "
                 "correct read has 12 rows and sums to the figure the totals "
                 "row states, not to twice it."),
    },
    shape=("Header on row 1, 12 data rows, then a 'Total' row on sheet row 14 "
           "holding the column sums. Read as data it doubles every sum and "
           "becomes the maximum of every numeric column. The ground truth is "
           "the 12 data rows only."),
)


# --------------------------------------------------------------------------
# 13. Subtotals per group and a grand total
# --------------------------------------------------------------------------
_SUBTOTAL_GROUPS: tuple[tuple[str, int], ...] = (("North", 5), ("South", 4), ("West", 3))


def _subtotal_records() -> list[tuple[str, str, int, float]]:
    records, i = [], 0
    for area, n_stores in _SUBTOTAL_GROUPS:
        for _ in range(n_stores):
            records.append((area, _PLACES[i], _totals_units(i + 3), _totals_revenue(i + 3)))
            i += 1
    return records


def build_subtotals_and_grand_total(dest: Path) -> Path:
    """How an exported report or a pivot actually looks.

    ``Area | Store | Units | Revenue`` with a ``North subtotal`` row closing
    each group and a ``Grand Total`` at the bottom. Four of the sixteen rows
    under the header are not records. Read as data the column sums come out at
    exactly three times the truth -- the records, the subtotals of the records,
    and the total of the subtotals -- and a group-by on ``Area`` grows three
    extra areas named after their own subtotals.
    """
    workbook = _openpyxl().Workbook()
    sheet = workbook.active
    sheet.title = "By area"

    _write_row(sheet, 1, 1, ["Area", "Store", "Units", "Revenue"])
    row = 2
    records = _subtotal_records()
    for area, _n in _SUBTOTAL_GROUPS:
        group = [r for r in records if r[0] == area]
        for record in group:
            _write_row(sheet, row, 1, list(record))
            row += 1
        _write_row(sheet, row, 1, [
            f"{area} subtotal", None,
            sum(r[2] for r in group), round(sum(r[3] for r in group), 2)])
        row += 1
    _write_row(sheet, row, 1, [
        "Grand Total", None,
        sum(r[2] for r in records), round(sum(r[3] for r in records), 2)])

    workbook.save(dest)
    return dest


_SUBTOTALS_GROUND_TRUTH = GroundTruth(
    columns=["Area", "Store", "Units", "Revenue"],
    n_rows=len(_subtotal_records()),
    header_row=1,
    numeric_columns=["Units", "Revenue"],
    text_columns=["Area", "Store"],
    planted_answer={
        "column": "Revenue",
        "sum": round(sum(r[3] for r in _subtotal_records()), 2),
        "max": max(r[3] for r in _subtotal_records()),
        "units_sum": sum(r[2] for r in _subtotal_records()),
        "groups": [area for area, _n in _SUBTOTAL_GROUPS],
        "note": ("'North subtotal', 'South subtotal', 'West subtotal' and "
                 "'Grand Total' are not records. A correct read has 12 rows "
                 "and exactly three distinct areas."),
    },
    shape=("Header on row 1; 12 store records in three groups, each closed by "
           "an '<Area> subtotal' row, and a 'Grand Total' row last (16 rows "
           "under the header, 12 of them data). Read as data every sum is "
           "three times the truth. The ground truth is the 12 records only."),
)


SPREADSHEET_DATASETS: dict[str, tuple[Callable[[Path], Path], GroundTruth]] = {
    "decoy_sheet_outweighs_the_answer": (build_decoy_sheet_outweighs_the_answer,
                                         DECOY_SHEET_TRUTH),
    "owner_file_shape": (build_owner_file_shape, _OWNER_GROUND_TRUTH),
    "merged_header_cells": (build_merged_header_cells, _MERGED_GROUND_TRUTH),
    "stacked_two_row_header": (build_stacked_two_row_header,
                               _STACKED_GROUND_TRUTH),
    "multi_sheet_data_on_sheet_two": (build_multi_sheet_data_on_sheet_two,
                                      _MULTI_SHEET_GROUND_TRUTH),
    "thousands_scaling_in_title": (build_thousands_scaling_in_title,
                                   _THOUSANDS_GROUND_TRUTH),
    "parenthesised_negatives": (build_parenthesised_negatives,
                                _NEGATIVES_GROUND_TRUTH),
    "footnote_markers_on_labels": (build_footnote_markers_on_labels,
                                   _FOOTNOTES_GROUND_TRUTH),
    "second_table_below": (build_second_table_below,
                           _SECOND_TABLE_GROUND_TRUTH),
    "year_columns_header": (build_year_columns_header,
                            _YEAR_COLUMNS_GROUND_TRUTH),
    "sec_financial_report": (build_sec_financial_report, _SEC_GROUND_TRUTH),
    "totals_row_at_bottom": (build_totals_row_at_bottom, _TOTALS_GROUND_TRUTH),
    "subtotals_and_grand_total": (build_subtotals_and_grand_total,
                                  _SUBTOTALS_GROUND_TRUTH),
}


def ground_truth_for(name: str) -> Optional[GroundTruth]:
    """The declared ground truth for one shape, or None if there is no such shape."""
    entry = SPREADSHEET_DATASETS.get(name)
    return entry[1] if entry else None
