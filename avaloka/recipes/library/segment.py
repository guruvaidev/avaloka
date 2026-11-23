"""``segment``: which groups of a dimension matter — share of the total,
concentration and the top groups.

Per-group count, sum and mean are DuckDB SQL over distinct rows; share of
total, cumulative share and the Herfindahl index come from pandas over every
group, so a top-N view cannot shrink the denominator. The largest groups by
total are shown and the rest are ``other``.

``high_cardinality`` is raised at bind as an Ask when a lower-cardinality
dimension exists, and as a qualified gate when none does; ``small_groups``
is a ``small_n`` caveat; ``missing_measure`` fails when no group has a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import build_report, figure, merged_gate_words
from . import _shared as sh

VERSION = "1.0"
TOP_N = 10                   # groups shown by total; the rest are "other"
HIGH_CARDINALITY = 50        # more distinct values than this asks for another dimension
SMALL_N = 30                 # a kept group with fewer rows is a small_n caveat
ASK_OPTIONS = 6              # lower-cardinality dimensions offered in the Ask

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_GROUPS = ("WITH d AS ({dedup}) SELECT CAST({dimension} AS VARCHAR) AS grp, count({measure}) AS n, "
              "sum({measure}) AS total, avg({measure}) AS mean FROM d "
              "WHERE {dimension} IS NOT NULL AND {measure} IS NOT NULL GROUP BY 1 ORDER BY total DESC, grp")

EVIDENCE_CELL = """\
unit = evidence_unit(MEASURE_UNIT)
evidence = {
    "rows": rows_leaves(counts),
    "groups": {"count": leaf(len(groups), "count"), "shown": leaf(len(shown), "count"),
               "smallest_n": leaf(shown["n"].min(), "count"), "total": leaf(num(groups["total"].sum()), unit),
               "largest_total": leaf(num(groups["total"].iloc[0]), unit),
               "largest_share": leaf(num(groups["share"].iloc[0]), "fraction"),
               "shown_share": leaf(num(shown["share"].sum()), "fraction"),
               "herfindahl": leaf(herfindahl, "unitless")},
}
"""

SPEC = RecipeSpec(
    id="segment", version=VERSION, title="Segment a measure by a dimension",
    params=(ParamSpec("measure", "measure"), ParamSpec("dimension", "dimension")),
    preconditions=("has_measure", "has_dimension"),
    quality_gates=("missing_measure", "high_cardinality", "small_groups", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "high_cardinality": "The dimension has many values; the largest groups are shown and the rest are grouped as other.",
})


@dataclass(frozen=True)
class SegmentBound(sh.RecipeBound):
    unit: str = "unitless"       # the measure's semantic unit, fixed at bind time


def _sql(bound: Bound) -> dict[str, str]:
    p = bound.params
    return {"dedup": SQL_DEDUP, "measure": ident(p["measure"]), "dimension": ident(p["dimension"])}


def _groups(con: Any, bound: Bound) -> pd.DataFrame:
    return query(con, SQL_GROUPS.format(**_sql(bound)))


def _concentration(groups: pd.DataFrame) -> pd.DataFrame:
    """Share of the total and cumulative share per group, over every group."""
    total = float(groups["total"].sum())
    share = groups["total"].astype(float) / total if total else groups["total"].astype(float) * 0.0
    return groups.assign(share=share, cumulative_share=share.cumsum())


def _herfindahl(shares: pd.Series) -> float:
    return float((shares.astype(float) ** 2).sum())


def _kept(groups: pd.DataFrame) -> pd.DataFrame:
    """The top groups and one ``other`` row for the rest (share and cumulative share included)."""
    top = groups.head(TOP_N)
    rest = groups.iloc[TOP_N:]
    if len(rest):
        n, total = int(rest["n"].sum()), float(rest["total"].sum())
        other = pd.DataFrame([{"grp": sh.OTHER, "n": n, "total": total, "mean": total / n if n else 0.0,
                               "share": float(rest["share"].sum()), "cumulative_share": float(groups["cumulative_share"].iloc[-1])}])
        top = pd.concat([top, other], ignore_index=True)
    return top


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class Segment:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=True, missing_reason="no_measure",
                                 question="Which column holds the measure to split by group?")
        if not isinstance(measure, str):
            return measure
        dimension = sh.bind_column(intent, sem, "dimension", ("dimension", "high_card_dimension"), required=True,
                                   missing_reason="ambiguous_dimension", question="Which column groups the rows?")
        if not isinstance(dimension, str):
            return dimension
        if sem[dimension].distinct > HIGH_CARDINALITY:
            lower = [n for n, c in sem.items() if c.role == "dimension" and c.distinct <= HIGH_CARDINALITY and n != dimension]
            if lower:
                return Ask("dimension", "The dimension named has many distinct values; which column should group the rows instead?",
                           tuple(lower[:ASK_OPTIONS]))
        return SegmentBound("segment", VERSION, {"measure": measure, "dimension": dimension},
                            source=sh.source_of(profile), unit=sem[measure].unit or "unitless")

    def check(self, con: Any, bound: Bound) -> GateReport:
        groups = _groups(con, bound)
        if not len(groups):
            return GateReport((Gate("missing_measure", "failed", reason="no_measure", value=0, unit="count"),))
        gates = [Gate("high_cardinality", "qualified", caveat_kind="data_quality", value=len(groups), unit="count")
                 if len(groups) > HIGH_CARDINALITY else Gate("high_cardinality", "passed")]
        kept = _kept(_concentration(groups))
        smallest = int(kept[kept["grp"] != sh.OTHER]["n"].min())
        gates.append(Gate("small_groups", "qualified", caveat_kind="small_n", value=smallest, unit="count")
                     if smallest < SMALL_N else Gate("small_groups", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(bound))).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        unit = sh.evidence_unit(getattr(bound, "unit", None))
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(bound))).iloc[0]
        rows_raw, rows_distinct = int(counts["rows_raw"]), int(counts["rows_distinct"])
        groups = _concentration(_groups(con, bound))
        kept = _kept(groups)
        shown = kept[kept["grp"] != sh.OTHER]
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(rows_raw, "count"), "distinct": sh.leaf(rows_distinct, "count"),
                     "duplicates": sh.leaf(rows_raw - rows_distinct, "count")},
            "groups": {
                "count": sh.leaf(len(groups), "count"), "shown": sh.leaf(len(shown), "count"),
                "smallest_n": sh.leaf(int(shown["n"].min()), "count"),
                "total": sh.leaf(_num(groups["total"].sum()), unit),
                "largest_total": sh.leaf(_num(groups["total"].iloc[0]), unit),
                "largest_share": sh.leaf(_num(groups["share"].iloc[0]), "fraction"),
                "shown_share": sh.leaf(_num(shown["share"].sum()), "fraction"),
                "herfindahl": sh.leaf(_herfindahl(groups["share"]), "unitless"),
                "by": {str(r["grp"]): {"n": sh.leaf(r["n"], "count"), "total": sh.leaf(_num(r["total"]), unit),
                                       "mean": sh.leaf(_num(r["mean"]), unit), "share": sh.leaf(_num(r["share"]), "fraction"),
                                       "cumulative_share": sh.leaf(_num(r["cumulative_share"]), "fraction")}
                       for _, r in kept.iterrows()},
            },
        }
        tables = {"groups": sh.table(["group", "n", "total", "mean", "share", "cumulative_share"],
                                     [[str(r["grp"]), int(r["n"]), _num(r["total"]), _num(r["mean"]), _num(r["share"]),
                                       _num(r["cumulative_share"])] for _, r in kept.iterrows()])}
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        ev, p = evidence.tree, bound.params
        unit = sh.figure_unit(getattr(bound, "unit", None))
        d = 0 if unit == "count" else 2
        figs = [figure("f1", ev, "/groups/count", unit="count", decimals=0),
                figure("f2", ev, "/groups/largest_share", unit="percent", decimals=1),
                figure("f3", ev, "/groups/shown_share", unit="percent", decimals=1),
                figure("f4", ev, "/groups/herfindahl", unit="unitless", decimals=3),
                figure("f5", ev, "/groups/smallest_n", unit="count", decimals=0),
                figure("f6", ev, "/rows/duplicates", unit="count", decimals=0),
                figure("f7", ev, "/groups/largest_total", unit=unit, decimals=d),
                figure("f8", ev, "/groups/total", unit=unit, decimals=d),
                figure("f9", ev, "/groups/shown", unit="count", decimals=0)]
        bullets = [{"so_what": "Concentration (the Herfindahl index of the shares) is {f4}; a value near one means a single group holds nearly everything.",
                    "now_what": "Weigh the largest groups first; the rest move the total little."}]
        if ev["groups"]["smallest_n"]["value"] < SMALL_N:
            bullets.append({"so_what": "The smallest group shown holds {f5} rows.",
                            "now_what": "Read a small group's share as rough."})
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f6} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so totals elsewhere are not inflated."})
        if len(bullets) < 3:
            bullets.append({"so_what": "The largest group totals {f7} of {f8} across all groups.",
                            "now_what": "Compare a group's mean, not only its total, before calling it strong."})
        kpis = [{"id": "largest_share", "label": "Largest group's share", "value": "f2", "direction_good": "neutral"},
                {"id": "concentration", "label": "Herfindahl index", "value": "f4", "direction_good": "neutral"},
                {"id": "groups", "label": "Groups", "value": "f1", "direction_good": "neutral"}]
        rows = evidence.tables["groups"]["rows"]
        label_m, label_d = sh.label(p["measure"]), sh.label(p["dimension"])
        charts = [sh.bar_chart("ranked", f"{label_m} by {label_d}", [r[0] for r in rows], [r[2] for r in rows],
                               x_name=label_d, y_name=f"total {label_m}",
                               alt_text=f"Ranked bar chart of total {label_m} for the largest {label_d} groups; "
                                        "the largest group's share is given in the headline.")]
        n_rows = int(ev["rows"]["raw"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=bound, evidence=evidence, gates=gates,
            headline_template="The largest group holds {f2} of the total across {f1} groups; the {f9} groups shown together hold {f3}.",
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Count, sum and mean of {p['measure']} per {p['dimension']} over distinct rows in DuckDB; "
                           "share of the total, cumulative share and the Herfindahl index over every group in pandas; "
                           "the largest groups by total are shown and the rest grouped as other.",
            rows_analyzed=n_rows, total_rows=n_rows, gate_words=GATE_WORDS,
            gate_refs={"high_cardinality": "/groups/count", "small_groups": "/groups/smallest_n",
                       "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = bound if isinstance(bound, SegmentBound) else SegmentBound(bound.recipe_id, bound.version, bound.params)
        args = "dedup=SQL_DEDUP, measure=ident(measure), dimension=ident(dimension)"
        cells = [
            sh.header_cell("segment", VERSION, SPEC.title),
            sh.parameters_cell(b, {"MEASURE_UNIT": b.unit, "TOP_N": TOP_N}),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# every group below is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, f"counts = con.sql(SQL_ROW_COUNTS.format({args})).df()\ncounts"),
            sh.sql_cell("SQL_GROUPS", SQL_GROUPS, f"groups = con.sql(SQL_GROUPS.format({args})).df()\ngroups"),
            _nb.code("total = float(groups['total'].sum())\n"
                     "groups['share'] = groups['total'].astype(float) / total if total else groups['total'].astype(float) * 0.0\n"
                     "groups['cumulative_share'] = groups['share'].cumsum()\n"
                     "herfindahl = float((groups['share'] ** 2).sum())\n"
                     "shown = groups.head(TOP_N)\n"
                     "herfindahl, shown['share'].sum()"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     "shown.plot.bar(x='grp', y='total', title=f'{measure} by {dimension}')"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
