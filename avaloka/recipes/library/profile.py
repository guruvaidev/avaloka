"""``profile``: what is in this table and can it be trusted?

Row and column counts, per-column role, missing share and distinct count,
exact-duplicate rows, constant columns and the ``avaloka.stats`` quality
score. It is the gate the other recipes rest on, so it refuses nothing but an
empty table; duplicates and heavy missingness are named, not hidden.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics
from avaloka.stats import profile_column, quality_report

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import GATE_WORDS, build_report, figure
from . import _shared as sh

VERSION = "1.0"
MISSING_SHARE_LIMIT = 0.20   # a column missing more than this is a data_quality caveat
MISSING_CHART_TOP = 12

SQL_ROWS = "SELECT count(*) AS rows_raw FROM t"
SQL_DISTINCT_ROWS = "SELECT count(*) AS rows_distinct FROM (SELECT DISTINCT * FROM t)"
SQL_COLUMN = "SELECT count(*) - count({col}) AS missing, count(DISTINCT {col}) AS distinct_count FROM t"
SQL_COLUMNS = "SELECT column_name FROM (DESCRIBE t)"
SQL_ALL_ROWS = "SELECT * FROM t"

EVIDENCE_CELL = """\
evidence = {
    "rows": leaf(rows_raw, "count"), "columns": leaf(len(columns), "count"),
    "duplicates": {"rows": leaf(duplicates, "count"),
                   "share": leaf(duplicates / rows_raw if rows_raw else 0.0, "fraction")},
    "constant_columns": leaf(sum(p["role"] == "constant" for p in profiles), "count"),
    "worst_missing_share": leaf(per_column["missing_share"].max() if len(per_column) else 0.0, "fraction"),
    "quality": {"score": leaf(quality["score"], "unitless")},
    "per_column": {c: {"missing_share": leaf(r["missing_share"], "fraction"), "distinct": leaf(r["distinct_count"], "count")}
                   for c, r in per_column.set_index("column").iterrows()},
}
"""

SPEC = RecipeSpec(
    id="profile", version=VERSION, title="Data profile", params=(),
    preconditions=(), quality_gates=("duplicates", "missing_measure"),
    evidence_kind="associational", explain_slots=("headline",),
)


def _counts(con: Any) -> tuple[int, int]:
    rows = int(query(con, SQL_ROWS).iloc[0, 0])
    distinct = int(query(con, SQL_DISTINCT_ROWS).iloc[0, 0])
    return rows, distinct


def _column_rows(con: Any, rows: int) -> list[dict[str, Any]]:
    """Per-column missing share and distinct count, from SQL, plus the role
    ``avaloka.stats`` gives the column (the quality score needs the same)."""
    frame = query(con, SQL_ALL_ROWS)
    out = []
    for name in query(con, SQL_COLUMNS)["column_name"]:
        stat = query(con, SQL_COLUMN.format(col=ident(name))).iloc[0]
        prof = profile_column(frame[name], rows)
        out.append({"name": str(name), "role": prof["role"], "missing": int(stat["missing"]),
                    "missing_share": (int(stat["missing"]) / rows) if rows else 0.0,
                    "distinct": int(stat["distinct_count"]), "profile": prof})
    return out


class Profile:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        if sh.rows_of(profile) <= 0:
            return Refusal("insufficient_rows", "The table has no rows to profile.")
        return sh.RecipeBound("profile", VERSION, {}, source=sh.source_of(profile))

    def check(self, con: Any, bound: Bound) -> GateReport:
        rows, distinct = _counts(con)
        duplicates = rows - distinct
        worst = max((c["missing_share"] for c in _column_rows(con, rows)), default=0.0)
        return GateReport((
            Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
            if duplicates else Gate("duplicates", "passed"),
            Gate("missing_measure", "qualified", caveat_kind="data_quality", value=worst, unit="fraction")
            if worst > MISSING_SHARE_LIMIT else Gate("missing_measure", "passed"),
        ))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        rows, distinct = _counts(con)
        columns = _column_rows(con, rows)
        quality = quality_report(query(con, SQL_ALL_ROWS), [c["profile"] for c in columns])
        constant = [c["name"] for c in columns if c["role"] == "constant"]
        worst = max(columns, key=lambda c: c["missing_share"]) if columns else None
        tree = {
            "rows": sh.leaf(rows, "count"),
            "columns": sh.leaf(len(columns), "count"),
            "duplicates": {"rows": sh.leaf(rows - distinct, "count"),
                           "share": sh.leaf((rows - distinct) / rows if rows else 0.0, "fraction")},
            "constant_columns": sh.leaf(len(constant), "count"),
            "worst_missing_share": sh.leaf(worst["missing_share"] if worst else 0.0, "fraction"),
            "quality": {"score": sh.leaf(quality["score"], "unitless")},
            "per_column": {c["name"]: {"missing_share": sh.leaf(c["missing_share"], "fraction"),
                                       "distinct": sh.leaf(c["distinct"], "count")} for c in columns},
        }
        table = sh.table(["column", "role", "missing_share", "distinct", "constant"],
                         [[c["name"], c["role"], round(c["missing_share"], 6), c["distinct"],
                           c["role"] == "constant"] for c in columns])
        return Evidence(tree, {"columns": table})

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        ev = evidence.tree
        figs = [
            figure("f1", ev, "/rows", unit="count", decimals=0),
            figure("f2", ev, "/columns", unit="count", decimals=0),
            figure("f3", ev, "/duplicates/rows", unit="count", decimals=0),
            figure("f4", ev, "/duplicates/share", unit="percent", decimals=1),
            figure("f5", ev, "/quality/score", unit="unitless", decimals=0),
            figure("f6", ev, "/constant_columns", unit="count", decimals=0),
            figure("f7", ev, "/worst_missing_share", unit="percent", decimals=1),
        ]
        duplicates = ev["duplicates"]["rows"]["value"] > 0
        bullets = [
            {"so_what": "Quality score {f5} of a hundred; {f6} columns hold a single value.",
             "now_what": ("Remove the duplicate rows before any total is computed." if duplicates
                          else "Nothing in the shape of the table blocks a further analysis.")},
            {"so_what": "The most incomplete column is missing {f7} of its values.",
             "now_what": "Check how those gaps arose before relying on that column."},
        ]
        kpis = [
            {"id": "rows", "label": "Rows", "value": "f1", "direction_good": "neutral"},
            {"id": "columns", "label": "Columns", "value": "f2", "direction_good": "neutral"},
            {"id": "duplicate_share", "label": "Duplicate rows", "value": "f4", "direction_good": "down"},
            {"id": "quality", "label": "Quality score", "value": "f5", "direction_good": "up"},
        ]
        cols = evidence.tables["columns"]["rows"]
        top = sorted(cols, key=lambda r: -r[2])[:MISSING_CHART_TOP]
        chart = sh.bar_chart("missing", "Missing share by column", [r[0] for r in top],
                             [round(r[2] * 100, 2) for r in top], x_name="column", y_name="missing_percent",
                             alt_text=f"Bar chart of the share of missing values for the {len(top)} most "
                                      "incomplete columns, highest first.")
        rows = int(ev["rows"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=bound, evidence=evidence, gates=gates,
            headline_template="The table has {f1} rows and {f2} columns; {f3} rows are exact duplicates ({f4} of the table).",
            bullets=bullets, figures=figs, charts=[chart], kpis=kpis,
            method_summary="Row, column, duplicate and per-column missing and distinct counts in DuckDB; "
                           "roles and the quality score from avaloka.stats.",
            rows_analyzed=rows, total_rows=rows,
            gate_refs={"duplicates": "/duplicates/rows", "missing_measure": "/worst_missing_share"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = bound if isinstance(bound, sh.RecipeBound) else sh.RecipeBound(bound.recipe_id, bound.version, bound.params)
        cells = [
            sh.header_cell("profile", VERSION, SPEC.title),
            sh.parameters_cell(b),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_ROWS", SQL_ROWS, "rows_raw = int(con.sql(SQL_ROWS).df().iloc[0, 0])\nrows_raw"),
            sh.sql_cell("SQL_DISTINCT_ROWS", SQL_DISTINCT_ROWS,
                        "rows_distinct = int(con.sql(SQL_DISTINCT_ROWS).df().iloc[0, 0])\n"
                        "duplicates = rows_raw - rows_distinct\nduplicates"),
            sh.sql_cell("SQL_COLUMNS", SQL_COLUMNS, "columns = con.sql(SQL_COLUMNS).df()['column_name'].tolist()"),
            sh.sql_cell("SQL_COLUMN", SQL_COLUMN,
                        "per_column = pd.DataFrame([\n"
                        "    dict(column=c, **con.sql(SQL_COLUMN.format(col=ident(c))).df().iloc[0])\n"
                        "    for c in columns])\n"
                        "per_column['missing_share'] = per_column['missing'] / rows_raw\nper_column"),
            sh.sql_cell("SQL_ALL_ROWS", SQL_ALL_ROWS,
                        "from avaloka.stats import profile_column, quality_report\n"
                        "df = con.sql(SQL_ALL_ROWS).df()\n"
                        "profiles = [profile_column(df[c], rows_raw) for c in columns]\n"
                        "quality = quality_report(df, profiles)\nquality['score']"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     f"(per_column.sort_values('missing_share', ascending=False).head({MISSING_CHART_TOP})\n"
                     "    .plot.bar(x='column', y='missing_share', title='Missing share by column'))"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
