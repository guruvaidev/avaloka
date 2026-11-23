"""Data Scout — replaces the data analyst.

Economic output: a schema, distribution and data-quality report. The Scout
loads the dataset, profiles every column, scores quality and records concrete
issues. Its output (``data_quality.json`` + the in-memory profile) is the
factual foundation every downstream firefly reasons about.
"""

from __future__ import annotations

from typing import Any

from avaloka.domains import classify as classify_domain
from avaloka.fireflies.base import Firefly
from avaloka.io import load_dataset
from avaloka.stats import correlation_pairs, profile_column, quality_report
from avaloka.util import write_json


def _loader_issues(handle) -> list[dict[str, Any]]:
    """Issues the loader found while reading, said to the user as quality.

    A numeric column whose stray "none"/"-" cells were read as missing, a date
    column whose gap words were, a mostly-numeric column kept as text because
    it holds real values that are not numbers, and a workbook whose other
    sheets were not analysed: each used to happen silently, and silence is how
    a 20,000-row answer to a 40,000-row workbook passed.
    """
    issues: list[dict[str, Any]] = []
    for column, info in (handle.structure.get("coerced_columns") or {}).items():
        words = ", ".join(repr(w) for w in info.get("words", [])[:5])
        if info["kind"] == "mixed":
            # Mostly numbers, but some cells hold real values that are not
            # ("V45" among ICD-9 codes). Those are never read as missing: the
            # column stays text, and the user hears why it is not numeric.
            examples = ", ".join(repr(e) for e in info.get("examples", [])[:5])
            gaps = (f"; only its {info['cells']} gap cell(s) ({words}) were read as missing"
                    if info["cells"] else "")
            issues.append({
                "severity": "low", "column": column, "kind": "real_values_kept_as_text",
                "detail": (f"This column is mostly numbers, but {info.get('kept', 0)} cell(s) hold "
                           f"values that are not numbers ({examples}). They were kept as real "
                           f"values, so the column is read as text, not numeric{gaps}."),
            })
            continue
        if info.get("accounting"):
            # A workbook's "(1,234)" is -1234. Said, because a column that
            # used to read as text now reads as numbers and its sign matters.
            issues.append({
                "severity": "low", "column": column, "kind": "accounting_figures_read_as_numbers",
                "detail": (f"{info['accounting']} cell(s) written as accounting figures, such as "
                           "(1,234) or 1,234, were read as numbers: a figure in brackets is "
                           "negative."),
            })
            if not info["cells"]:
                continue
        issues.append({
            "severity": "low", "column": column,
            "kind": ("null_words_read_as_missing" if info["kind"] == "numeric"
                     else "unparseable_dates_read_as_missing"),
            "detail": (f"{info['cells']} cell(s) ({words}) in a {info['kind']} column "
                       "were read as missing values, not as categories."),
        })
    issues += _shape_issues(handle.structure.get("shape") or {})
    issues += _unit_issues(handle.structure.get("units") or {})
    sheets = handle.structure.get("sheets") or {}
    if sheets.get("others"):
        others = sheets["others"]
        held = sum(int(s.get("rows", 0)) for s in others)
        names = ", ".join(f"{s['name']} ({s['rows']} rows)" for s in others)
        # Why this sheet: a workbook read off the wrong sheet reads perfectly,
        # so the rule that chose it is said, not left to be guessed.
        choice = sheets.get("chosen_by") or {}
        why = {
            "question": (" It was chosen because the question names "
                         + ", ".join(choice.get("matched") or []) + ", which only it has."),
            "named": " It was the sheet asked for.",
            "size": (" It was chosen as the sheet with the most filled cells; nothing in "
                     "the question pointed to one sheet over the others."),
        }.get(choice.get("by"), "")
        issues.append({
            "severity": "medium", "column": None, "kind": "other_sheets_not_analysed",
            "detail": (f"Analysed sheet {sheets['analysed']!r} only; {len(others)} other "
                       f"sheet(s) hold {held} rows and were not read: {names}.{why}"),
        })
    return issues


def _rows_named(entries: list[dict[str, Any]], limit: int = 4) -> str:
    """"row 14 (Total), row 9 (North subtotal)" -- labels unquoted, because a
    quoted word in a report reads as a column name."""
    shown = []
    for entry in entries[:limit]:
        where = f"row {entry['row']}" if entry.get("row") else "a row"
        shown.append(f"{where} ({entry['label']})" if entry.get("label") else where)
    more = len(entries) - limit
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def _shape_issues(shape: dict[str, Any]) -> list[dict[str, Any]]:
    """What the workbook read left out under the table, said rather than done
    silently: each of these changes the row count and every total."""
    issues: list[dict[str, Any]] = []
    aggregates = shape.get("aggregate_rows") or []
    if aggregates:
        issues.append({
            "severity": "medium", "column": None, "kind": "aggregate_rows_left_out",
            "detail": (f"Left out {len(aggregates)} row(s) whose numbers add up to the rows "
                       f"they close: {_rows_named(aggregates)}. Read as records they would "
                       "count every figure twice."),
        })
    kept = shape.get("labelled_total_rows_kept") or []
    if kept:
        issues.append({
            "severity": "low", "column": None, "kind": "total_rows_kept_as_records",
            "detail": (f"Kept {len(kept)} row(s) labelled as totals as records, because "
                       "nothing in the sheet shows they only sum other rows: "
                       f"{_rows_named(kept)}. A column sum counts them."),
        })
    notes = shape.get("trailing_notes") or []
    if notes:
        markers = shape.get("footnote_markers") or []
        stripped = (f" The footnote marks they define ({' '.join(markers)}) were taken off "
                    "column names and labels." if markers else "")
        issues.append({
            "severity": "low", "column": None, "kind": "notes_below_table_left_out",
            "detail": (f"Left out {len(notes)} line(s) of notes under the table; they are not "
                       f"records.{stripped}"),
        })
    if shape.get("rows_below_table"):
        where = f" starting at row {shape['next_table_row']}" if shape.get("next_table_row") else ""
        issues.append({
            "severity": "medium", "column": None, "kind": "second_table_not_read",
            "detail": (f"A second table sits below this one{where}; its "
                       f"{shape['rows_below_table']} row(s), header included, were not read "
                       "as part of it."),
        })
    return issues


def _unit_issues(units: dict[str, Any]) -> list[dict[str, Any]]:
    """Say the unit the file states, where it states it, and that the figures
    are shown as written -- a total read off a "£ thousands" sheet is in
    thousands until someone multiplies it."""
    from avaloka.units import unit_words

    stated = units.get("stated")
    if not stated:
        return []
    source = stated.get("source") or {}
    where = f"the title on row {source['sheet_row']}" if source.get("sheet_row") else "the sheet's title"
    applied = sorted(name for name, unit in (units.get("columns") or {}).items()
                     if (unit.get("source") or {}).get("where") == "title")
    exempt = units.get("exempt") or []
    detail = (f"The sheet states its figures are in {unit_words(stated)} ({where}). "
              f"Values are shown as written, not multiplied out; this applies to "
              f"{', '.join(applied) if applied else 'no column'}.")
    if exempt:
        detail += f" Not applied to {', '.join(exempt)}, which count things or are rates."
    return [{"severity": "low", "column": None, "kind": "unit_stated_in_title", "detail": detail}]


class DataScout(Firefly):
    name = "data_scout"
    human_role = "Data analyst"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        # The goal goes to the reader: a workbook's sheet is chosen by what
        # the question is about when that settles it (loader._choose_sheet).
        handle = load_dataset(self.ctx.data_source, question=self.ctx.goal or None)
        frame = handle.frame
        n_rows = handle.n_rows

        columns = [profile_column(frame[c], n_rows) for c in frame.columns]
        # The unit the file states for a column rides on its profile entry,
        # with where the file says it; a column the file says nothing about
        # gets no unit key at all (avaloka/units.py).
        stated = (handle.structure.get("units") or {}).get("columns") or {}
        for column in columns:
            if column["name"] in stated:
                column["unit"] = stated[column["name"]]
        quality = quality_report(frame, columns, extra_issues=_loader_issues(handle))
        correlations = correlation_pairs(frame)
        # Names decide the domain; the three rows only confirm value types
        # (names grouped by type), so no cell reaches the profile through here.
        domain = classify_domain([c["name"] for c in columns],
                                 sample_rows=frame.head(3).to_dict("records"))

        profile = {
            "dataset": {
                "source": handle.source,
                "format": handle.fmt,
                "sha256": handle.sha256,
                "n_rows": handle.n_rows,
                "n_cols": handle.n_cols,
                "delimiter": handle.delimiter,
                "structure": handle.structure,
            },
            "columns": columns,
            "quality": quality,
            "correlations": correlations,
            "domain": domain,
        }

        # Hand the loaded frame + profile to downstream fireflies.
        self.ctx.blackboard["handle"] = handle
        self.ctx.blackboard["frame"] = frame
        self.ctx.blackboard["profile"] = profile

        write_json(self.ctx.path("data_quality.json"), profile)

        manual_minutes = 45 + 6 * handle.n_cols
        read = (f"; reads as {domain['label'].lower()} data (confidence {domain['confidence']:.2f})"
                if domain["domain"] != "generic" else "")
        return (
            f"Profiled {handle.n_cols} columns x {handle.n_rows} rows; quality score "
            f"{quality['score']}/100 with {len(quality['issues'])} issues flagged{read}.",
            manual_minutes,
            {"quality_score": quality["score"], "n_issues": len(quality["issues"]),
             "domain": domain["domain"]},
        )
