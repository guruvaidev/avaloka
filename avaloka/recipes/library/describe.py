"""``describe``: the typical value, spread and shape of one measure, overall
and by a dimension.

Quantiles come from DuckDB (``quantile_cont``), skew and kurtosis from pandas;
statistics are computed on distinct rows, with the raw row count kept in the
evidence, so a duplicated export cannot shift the median unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import GATE_WORDS, build_report, figure
from . import _shared as sh

VERSION = "1.0"
MISSING_SHARE_LIMIT = 0.20   # more missing than this is a data_quality caveat
SMALL_N = 30                 # fewer rows (overall or in a group) is a small_n caveat
STATS = ("n", "mean", "median", "p10", "p25", "p75", "p90", "std", "min", "max")

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = ("SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct, "
                  "count(*) - count({measure}) AS measure_missing FROM t")
SQL_STATS = ("WITH d AS ({dedup}) SELECT count({measure}) AS n, avg({measure}) AS mean, "
             "median({measure}) AS median, quantile_cont({measure}, 0.10) AS p10, "
             "quantile_cont({measure}, 0.25) AS p25, quantile_cont({measure}, 0.75) AS p75, "
             "quantile_cont({measure}, 0.90) AS p90, stddev_samp({measure}) AS std, "
             "min({measure}) AS min, max({measure}) AS max FROM d")
SQL_STATS_BY = ("WITH d AS ({dedup}), top AS (SELECT {dimension} AS g FROM d WHERE {dimension} IS NOT NULL "
                "GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {top}), "
                "b AS (SELECT CASE WHEN {dimension} IN (SELECT g FROM top) THEN CAST({dimension} AS VARCHAR) "
                "ELSE '{other}' END AS grp, {measure} AS v FROM d) "
                "SELECT grp, count(v) AS n, avg(v) AS mean, median(v) AS median, "
                "quantile_cont(v, 0.10) AS p10, quantile_cont(v, 0.25) AS p25, quantile_cont(v, 0.75) AS p75, "
                "quantile_cont(v, 0.90) AS p90, stddev_samp(v) AS std, min(v) AS min, max(v) AS max "
                "FROM b GROUP BY grp ORDER BY n DESC, grp")
SQL_VALUES = "WITH d AS ({dedup}) SELECT {measure} AS v FROM d WHERE {measure} IS NOT NULL"

EVIDENCE_CELL = """\
unit = evidence_unit(MEASURE_UNIT)
c, s = counts.iloc[0], stats.iloc[0]
evidence = {
    "rows": rows_leaves(counts),
    "measure": {"missing_share": leaf(int(c["measure_missing"]) / int(c["rows_raw"]) if int(c["rows_raw"]) else 1.0, "fraction"),
                "n": leaf(s["n"], "count"),
                **{k: leaf(num(s[k]), unit) for k in ("mean", "median", "p10", "p25", "p75", "p90", "std", "min", "max")},
                "skew": leaf(num(skew) if len(values) > 2 else 0.0, "unitless"),
                "kurtosis": leaf(num(kurtosis) if len(values) > 3 else 0.0, "unitless")},
}
if dimension:
    evidence["groups"] = {"count": leaf(len(stats_by), "count"),
                          "smallest_n": leaf(stats_by["n"].min() if len(stats_by) else 0, "count")}
"""

SPEC = RecipeSpec(
    id="describe", version=VERSION, title="Describe a measure",
    params=(ParamSpec("measure", "measure"), ParamSpec("dimension", "dimension", required=False)),
    preconditions=("has_measure",),
    quality_gates=("missing_measure", "constant", "duplicates", "small_groups"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)


@dataclass(frozen=True)
class DescribeBound(sh.RecipeBound):
    unit: str = "unitless"       # the measure's semantic unit, fixed at bind time


def _sql(bound: Bound) -> dict[str, str]:
    p = bound.params
    args = {"dedup": SQL_DEDUP, "measure": ident(p["measure"])}
    if p.get("dimension"):
        args.update(dimension=ident(p["dimension"]), top=sh.TOP_GROUPS, other=sh.OTHER)
    return args


def _counts(con: Any, bound: Bound) -> pd.Series:
    return query(con, SQL_ROW_COUNTS.format(**_sql(bound))).iloc[0]


def _stats(con: Any, bound: Bound) -> pd.Series:
    return query(con, SQL_STATS.format(**_sql(bound))).iloc[0]


def _stats_by(con: Any, bound: Bound) -> pd.DataFrame | None:
    if not bound.params.get("dimension"):
        return None
    return query(con, SQL_STATS_BY.format(**_sql(bound)))


def _values(con: Any, bound: Bound) -> pd.Series:
    return query(con, SQL_VALUES.format(**_sql(bound)))["v"].astype(float)


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


def _histogram(values: pd.Series) -> dict[str, Any]:
    arr = values.to_numpy(dtype=float)
    bins = min(30, max(5, int(np.sqrt(len(arr))))) if len(arr) else 5
    counts, edges = np.histogram(arr, bins=bins) if len(arr) else (np.zeros(bins, int), np.arange(bins + 1))
    labels = [f"{edges[i]:.4g} to {edges[i + 1]:.4g}" for i in range(len(counts))]
    return sh.table(["bin", "rows"], zip(labels, [int(c) for c in counts]))


class Describe:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=True, missing_reason="no_measure",
                                 question="Which column holds the measure to describe?")
        if not isinstance(measure, str):
            return measure
        dimension = sh.bind_column(intent, sem, "dimension", ("dimension", "high_card_dimension"), required=False,
                                   missing_reason="ambiguous_dimension", question="Which column groups the rows?")
        if isinstance(dimension, (Ask, Refusal)):
            return dimension
        params = {"measure": measure, **({"dimension": dimension} if dimension else {})}
        return DescribeBound("describe", VERSION, params, source=sh.source_of(profile),
                             unit=sem[measure].unit or "unitless")

    def check(self, con: Any, bound: Bound) -> GateReport:
        counts = _counts(con, bound)
        rows_raw, rows_distinct = int(counts["rows_raw"]), int(counts["rows_distinct"])
        missing_share = int(counts["measure_missing"]) / rows_raw if rows_raw else 1.0
        gates: list[Gate] = []
        if missing_share >= 1.0:
            gates.append(Gate("missing_measure", "failed", reason="no_measure", value=missing_share, unit="fraction"))
            return GateReport(tuple(gates))
        gates.append(Gate("missing_measure", "qualified", caveat_kind="data_quality", value=missing_share, unit="fraction")
                     if missing_share > MISSING_SHARE_LIMIT else Gate("missing_measure", "passed"))
        stats = _stats(con, bound)
        std = stats["std"]
        if std is None or pd.isna(std) or float(std) == 0.0:
            gates.append(Gate("constant", "failed", reason="constant_column", value=0.0, unit="unitless"))
        else:
            gates.append(Gate("constant", "passed"))
        duplicates = rows_raw - rows_distinct
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        by = _stats_by(con, bound)
        smallest = int(by["n"].min()) if by is not None and len(by) else rows_distinct
        small = min(rows_distinct, smallest)
        gates.append(Gate("small_groups", "qualified", caveat_kind="small_n", value=small, unit="count")
                     if small < SMALL_N else Gate("small_groups", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        unit = sh.evidence_unit(getattr(bound, "unit", None))
        counts, stats, by = _counts(con, bound), _stats(con, bound), _stats_by(con, bound)
        values = _values(con, bound)
        rows_raw, rows_distinct = int(counts["rows_raw"]), int(counts["rows_distinct"])
        measure = {"missing_share": sh.leaf(int(counts["measure_missing"]) / rows_raw if rows_raw else 1.0, "fraction"),
                   "n": sh.leaf(stats["n"], "count")}
        for k in STATS[1:]:
            measure[k] = sh.leaf(_num(stats[k]), unit)
        # pandas: bias-corrected sample skew and excess kurtosis, as pandas defines them.
        measure["skew"] = sh.leaf(_num(values.skew()) if len(values) > 2 else 0.0, "unitless")
        measure["kurtosis"] = sh.leaf(_num(values.kurt()) if len(values) > 3 else 0.0, "unitless")
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(rows_raw, "count"), "distinct": sh.leaf(rows_distinct, "count"),
                     "duplicates": sh.leaf(rows_raw - rows_distinct, "count")},
            "measure": measure,
        }
        tables = {"stats": sh.table(list(STATS), [[_num(stats[k]) for k in STATS]]),
                  "histogram": _histogram(values)}
        if by is not None:
            tree["groups"] = {
                "count": sh.leaf(len(by), "count"),
                "smallest_n": sh.leaf(int(by["n"].min()) if len(by) else 0, "count"),
                "by": {str(r["grp"]): {k: sh.leaf(_num(r[k]), "count" if k == "n" else unit) for k in STATS}
                       for _, r in by.iterrows()},
            }
            tables["stats_by"] = sh.table(["group", *STATS], [[str(r["grp"]), *[_num(r[k]) for k in STATS]]
                                                              for _, r in by.iterrows()])
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        ev, p = evidence.tree, bound.params
        unit = sh.figure_unit(getattr(bound, "unit", None))
        d = 0 if unit == "count" else 2
        figs = [figure("f1", ev, "/measure/n", unit="count", decimals=0)]
        for i, k in enumerate(("median", "mean", "std", "min", "max", "p10", "p90"), start=2):
            figs.append(figure(f"f{i}", ev, f"/measure/{k}", unit=unit, decimals=d))
        figs.append(figure("f9", ev, "/measure/skew", unit="unitless", decimals=2))
        figs.append(figure("f10", ev, "/rows/duplicates", unit="count", decimals=0))
        bullets = [{"so_what": "The spread (standard deviation) is {f4}; values run from {f5} to {f6}, with skew {f9}.",
                    "now_what": "Compare the median rather than the mean when the shape is skewed."}]
        charts = [sh.histogram_chart("distribution", f"Distribution of {sh.label(p['measure'])}",
                                     [r[0] for r in evidence.tables["histogram"]["rows"]],
                                     [r[1] for r in evidence.tables["histogram"]["rows"]],
                                     x_name=sh.label(p["measure"]),
                                     alt_text=f"Histogram of {sh.label(p['measure'])} over the distinct rows; "
                                              "the median and mean are given in the headline.")]
        if "groups" in ev:
            figs.append(figure("f11", ev, "/groups/count", unit="count", decimals=0))
            figs.append(figure("f12", ev, "/groups/smallest_n", unit="count", decimals=0))
            bullets.append({"so_what": "The rows split into {f11} groups; the smallest holds {f12} rows.",
                            "now_what": "Read a small group's statistics as rough."})
            by = evidence.tables["stats_by"]["rows"]
            charts.append(sh.boxplot_chart("by_group", f"{sh.label(p['measure'])} by {sh.label(p['dimension'])}",
                                           [r[0] for r in by], [[r[9], r[5], r[3], r[6], r[10]] for r in by],
                                           group_name=sh.label(p["dimension"]), y_name=sh.label(p["measure"]),
                                           alt_text=f"Box plot of {sh.label(p['measure'])} for each "
                                                    f"{sh.label(p['dimension'])} value; each box spans the middle half."))
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f10} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so totals elsewhere are not inflated."})
        kpis = [{"id": "median", "label": f"Median {sh.label(p['measure'])}"[:40], "value": "f2", "direction_good": "neutral"},
                {"id": "mean", "label": f"Mean {sh.label(p['measure'])}"[:40], "value": "f3", "direction_good": "neutral"},
                {"id": "spread", "label": "Standard deviation", "value": "f4", "direction_good": "neutral"},
                {"id": "rows", "label": "Rows with a value", "value": "f1", "direction_good": "neutral"}]
        rows = int(ev["rows"]["raw"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=bound, evidence=evidence, gates=gates,
            headline_template="Across {f1} rows the median is {f2} and the mean {f3}; most values sit between {f7} and {f8}.",
            bullets=bullets, figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Count, mean, median, deciles, quartiles, standard deviation and range of "
                           f"{p['measure']} over distinct rows in DuckDB"
                           + (f", overall and by {p['dimension']} (largest groups; the rest as other)" if p.get("dimension") else "")
                           + "; skew and kurtosis from pandas.",
            rows_analyzed=rows, total_rows=rows,
            # small_groups decided on the smallest group when there is a dimension, else on the distinct rows
            gate_refs={"missing_measure": "/measure/missing_share", "duplicates": "/rows/duplicates",
                       "small_groups": "/groups/smallest_n" if "groups" in ev else "/rows/distinct"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = bound if isinstance(bound, DescribeBound) else DescribeBound(bound.recipe_id, bound.version, bound.params)
        p = b.params
        args = "dedup=SQL_DEDUP, measure=ident(measure)"
        cells = [
            sh.header_cell("describe", VERSION, SPEC.title),
            sh.parameters_cell(b, {"MEASURE_UNIT": b.unit, **({} if p.get("dimension") else {"dimension": None})}),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# every statistic below is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, f"counts = con.sql(SQL_ROW_COUNTS.format({args})).df()\ncounts"),
            sh.sql_cell("SQL_STATS", SQL_STATS, f"stats = con.sql(SQL_STATS.format({args})).df()\nstats"),
        ]
        if p.get("dimension"):
            cells.append(sh.sql_cell("SQL_STATS_BY", SQL_STATS_BY,
                                     f"stats_by = con.sql(SQL_STATS_BY.format({args}, dimension=ident(dimension), "
                                     f"top={sh.TOP_GROUPS}, other={sh.OTHER!r})).df()\nstats_by"))
        cells += [
            sh.sql_cell("SQL_VALUES", SQL_VALUES, f"values = con.sql(SQL_VALUES.format({args})).df()['v'].astype(float)"),
            _nb.code("skew = values.skew()\nkurtosis = values.kurt()\nskew, kurtosis"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     "values.plot.hist(bins=min(30, max(5, int(len(values) ** 0.5))), title=f'Distribution of {measure}')"
                     + ("\nstats_by.plot.bar(x='grp', y='median', title=f'Median {measure} by {dimension}')"
                        if p.get("dimension") else "")),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
