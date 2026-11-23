"""``anomalies``: which points of a measure are unusual.

A robust z-score from the median and the median absolute deviation (MAD),
and the fences of the interquartile rule, over every point. With a time
column the score is taken on the seasonal residual: the measure less the
median of its period at a grain one step coarser than the data's own, so a
level that moves with the season is not read as a run of outliers. The
flagged points, largest first, are listed in a data table.

Each row is a point, so the statistics run over the rows as they are; the
``duplicates`` gate counts repeated rows and says a duplicated outlier is
flagged more than once. ``min_rows`` fails on too few points, and
``flag_share`` qualifies when more than a tenth of the points are flagged
(the rule, not the data, is suspect).
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
from ..report import build_report, figure, merged_gate_words
from . import _shared as sh
from . import trend as _trend

VERSION = "1.0"
MIN_ROWS = 30                # fewer points than this is a refusal
Z_LIMIT = 3.5                # |robust z| above this is flagged (Iglewicz and Hoaglin)
MAD_SCALE = 0.6745           # MAD to a normal standard deviation
IQR_K = 1.5                  # fences at the quartiles plus or minus this many interquartile ranges
FLAG_SHARE_LIMIT = 0.10      # more of the points flagged than this is a data_quality caveat
TOP_POINTS = 20              # flagged points listed, largest |z| first
UNPARSED_LIMIT = 0.05
#: The grain of the seasonal median: one step coarser than the data's own grain.
SEASON_GRAIN = {"day": "month", "week": "month", "month": "quarter", "quarter": "year", "year": "year"}
TIME_EXPR = _trend.TIME_EXPR

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_PARSE = "SELECT count({time}) AS non_null, count({ts}) AS parsed FROM t"
SQL_VALUES = "SELECT row_number() OVER () AS idx, {measure} AS v FROM t WHERE {measure} IS NOT NULL"
SQL_RESIDUALS = ("WITH p AS (SELECT {ts} AS ts, {measure} AS v, date_trunc('{grain}', {ts}) AS period FROM t "
                 "WHERE {ts} IS NOT NULL AND {measure} IS NOT NULL), "
                 "m AS (SELECT period, median(v) AS med FROM p GROUP BY 1) "
                 "SELECT p.ts, p.period, p.v, p.v - m.med AS resid FROM p JOIN m USING (period) ORDER BY p.ts, p.v")

EVIDENCE_CELL = """\
unit = evidence_unit(MEASURE_UNIT)
flagged = points[points["flagged"]]
evidence = {
    "rows": rows_leaves(counts), "n": leaf(len(points), "count"),
    "robust": {"median": leaf(med, unit), "mad": leaf(mad, unit),
               "value_median": leaf(points["v"].astype(float).median(), unit),
               "max_abs_z": leaf(points["z"].abs().max(), "unitless")},
    "flagged": {"count": leaf(len(flagged), "count"), "share": leaf(len(flagged) / len(points), "fraction")},
    "iqr": {"low": leaf(fences[0], unit), "high": leaf(fences[1], unit),
            "count": leaf(int(((x < fences[0]) | (x > fences[1])).sum()), "count")},
}
if time:
    evidence["dates"] = unparsed_leaves(parse)
"""

SPEC = RecipeSpec(
    id="anomalies", version=VERSION, title="Unusual points of a measure",
    params=(ParamSpec("measure", "measure"), ParamSpec("time", "time", required=False)),
    preconditions=("has_measure", "min_rows_30"),
    quality_gates=("min_rows", "date_parse", "flag_share", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "min_rows": "Too few points hold a value.",
    "flag_share": "More than a tenth of the points are flagged, which says the rule is mis-sized for this data "
                    "rather than that so many points are unusual.",
    "duplicates": "Some rows are exact duplicates; each row is a point here, so a duplicated outlier is flagged "
                  "more than once.",
})


@dataclass(frozen=True)
class AnomaliesBound(sh.RecipeBound):
    unit: str = "unitless"   # the measure's semantic unit
    parse: str = "cast"      # "cast", "year", or a strptime format for a slash-date text column
    grain: str | None = None  # the seasonal median's grain; None without a time column

    @property
    def time_expr(self) -> str:
        time = ident(self.params["time"])
        if self.parse in ("cast", "year"):
            return TIME_EXPR[self.parse].format(time=time)
        return TIME_EXPR["format"].format(time=time, fmt=self.parse)


def _sql(b: AnomaliesBound) -> dict[str, Any]:
    p = b.params
    args: dict[str, Any] = {"dedup": SQL_DEDUP, "measure": ident(p["measure"])}
    if p.get("time"):
        args.update(time=ident(p["time"]), ts=b.time_expr, grain=b.grain)
    return args


def _as_bound(bound: Bound) -> AnomaliesBound:
    return bound if isinstance(bound, AnomaliesBound) else AnomaliesBound(bound.recipe_id, bound.version, bound.params)


def robust_z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``(z, median, mad)``: the modified z-score ``0.6745 (x - median) / MAD``;
    all zeros when the MAD is zero (more than half the points are equal)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0.0:
        return np.zeros(len(x)), med, mad
    return MAD_SCALE * (x - med) / mad, med, mad


def iqr_fences(x: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(x, [25, 75])
    return float(q1 - IQR_K * (q3 - q1)), float(q3 + IQR_K * (q3 - q1))


def _points(con: Any, b: AnomaliesBound) -> pd.DataFrame:
    """``label``, ``value`` and the scored series ``x`` (the residual with a time column)."""
    if b.params.get("time"):
        rows = query(con, SQL_RESIDUALS.format(**_sql(b)))
        return pd.DataFrame({"label": [pd.Timestamp(t).strftime("%Y-%m-%d") for t in rows["ts"]],
                             "value": rows["v"].astype(float), "x": rows["resid"].astype(float)})
    rows = query(con, SQL_VALUES.format(**_sql(b)))
    return pd.DataFrame({"label": rows["idx"].astype(int).astype(str), "value": rows["v"].astype(float),
                         "x": rows["v"].astype(float)})


def _scored(points: pd.DataFrame) -> tuple[pd.DataFrame, float, float, tuple[float, float]]:
    x = points["x"].to_numpy(dtype=float)
    z, med, mad = robust_z(x)
    fences = iqr_fences(x)
    scored = points.assign(z=z, flagged=np.abs(z) > Z_LIMIT, outside=(x < fences[0]) | (x > fences[1]))
    return scored, med, mad, fences


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class Anomalies:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=True, missing_reason="no_measure",
                                 question="Which column holds the measure to scan for unusual points?")
        if not isinstance(measure, str):
            return measure
        time = intent.params.get("time")
        if not isinstance(time, str):
            candidates = _trend._time_candidates(sem, profile)
            time = candidates[0] if len(candidates) == 1 else None
        params: dict[str, Any] = {"measure": measure}
        parse, grain = "cast", None
        if time:
            rule = _trend._parse_rule(time, sem, profile)
            if not isinstance(rule, str):
                return rule
            parse = rule
            grain = SEASON_GRAIN[(sem[time].grain if time in sem else None) or "day"]
            params["time"] = time
        return AnomaliesBound("anomalies", VERSION, params, source=sh.source_of(profile),
                              unit=sem[measure].unit or "unitless", parse=parse, grain=grain)

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        args = _sql(b)
        gates: list[Gate] = []
        if b.params.get("time"):
            parse = query(con, SQL_PARSE.format(**args)).iloc[0]
            non_null, parsed = int(parse["non_null"]), int(parse["parsed"])
            unparsed = (non_null - parsed) / non_null if non_null else 1.0
            gates.append(Gate("date_parse", "qualified", caveat_kind="data_quality", value=unparsed, unit="fraction")
                         if unparsed > UNPARSED_LIMIT else Gate("date_parse", "passed"))
        points = _points(con, b)
        if len(points) < MIN_ROWS:
            gates.append(Gate("min_rows", "failed", reason="insufficient_rows", value=len(points), unit="count"))
            return GateReport(tuple(gates))
        gates.append(Gate("min_rows", "passed"))
        scored, _, _, _ = _scored(points)
        share = float(scored["flagged"].mean())
        gates.append(Gate("flag_share", "qualified", caveat_kind="data_quality", value=share, unit="fraction")
                     if share > FLAG_SHARE_LIMIT else Gate("flag_share", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        args = _sql(b)
        unit = sh.evidence_unit(b.unit)
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        scored, med, mad, fences = _scored(_points(con, b))
        flagged = scored[scored["flagged"]].assign(size=lambda d: d["z"].abs()).sort_values(["size", "label"], ascending=[False, True])
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "n": sh.leaf(len(scored), "count"),
            "robust": {"median": sh.leaf(med, unit), "mad": sh.leaf(mad, unit),
                       "value_median": sh.leaf(float(scored["value"].median()), unit),
                       "max_abs_z": sh.leaf(float(scored["z"].abs().max()), "unitless")},
            "flagged": {"count": sh.leaf(len(flagged), "count"), "share": sh.leaf(len(flagged) / len(scored), "fraction")},
            "iqr": {"low": sh.leaf(fences[0], unit), "high": sh.leaf(fences[1], unit),
                    "count": sh.leaf(int(scored["outside"].sum()), "count")},
        }
        if b.params.get("time"):
            parse = query(con, SQL_PARSE.format(**args)).iloc[0]
            tree["dates"] = {"unparsed_share": sh.leaf((int(parse["non_null"]) - int(parse["parsed"])) / int(parse["non_null"])
                                                       if int(parse["non_null"]) else 1.0, "fraction")}
        tables = {"points": sh.table(["point", "value", "scored", "z", "flagged"],
                                     [[r["label"], r["value"], r["x"], r["z"], int(r["flagged"])] for _, r in scored.iterrows()]),
                  "flagged": sh.table(["point", "value", "z"],
                                      [[r["label"], r["value"], r["z"]] for _, r in flagged.head(TOP_POINTS).iterrows()])}
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        unit = sh.figure_unit(b.unit)
        figs = [figure("f1", ev, "/n", unit="count", decimals=0),
                figure("f2", ev, "/flagged/count", unit="count", decimals=0),
                figure("f3", ev, "/flagged/share", unit="percent", decimals=1),
                figure("f4", ev, "/robust/max_abs_z", unit="unitless", decimals=1),
                figure("f5", ev, "/iqr/count", unit="count", decimals=0),
                figure("f6", ev, "/rows/duplicates", unit="count", decimals=0),
                figure("f7", ev, "/robust/value_median", unit=unit, decimals=2)]
        bullets = [{"so_what": "The fences of the interquartile rule mark {f5} points, against {f2} by the robust z rule.",
                    "now_what": "Check first the points both rules agree on."}]
        if ev["flagged"]["share"]["value"] > FLAG_SHARE_LIMIT:
            bullets.append({"so_what": "More than a tenth of the points are flagged, so the rule is reading ordinary spread as unusual.",
                            "now_what": "Look for a level shift or a change in spread before reading each point as unusual."})
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f6} rows are exact duplicates; a duplicated outlier is flagged more than once.",
                            "now_what": "Remove them at the source and re-run."})
        if len(bullets) < 3:
            bullets.append({"so_what": "The typical value (median) is {f7}" + ("; each point is scored against its own period's level."
                                                                              if p.get("time") else "."),
                            "now_what": "Read a flagged point against that level, not against the mean."})
        kpis = [{"id": "flagged", "label": "Points flagged", "value": "f2", "direction_good": "neutral"},
                {"id": "share", "label": "Share flagged", "value": "f3", "direction_good": "down"},
                {"id": "points", "label": "Points", "value": "f1", "direction_good": "neutral"}]
        rows = evidence.tables["points"]["rows"]
        label_m = sh.label(p["measure"])
        charts = [sh.flagged_line_chart("points", f"{label_m} with flagged points", [r[0] for r in rows], [r[1] for r in rows],
                                        [bool(r[4]) for r in rows], x_name=sh.label(p["time"]) if p.get("time") else "row",
                                        y_name=label_m, kind="line" if p.get("time") else "scatter",
                                        alt_text=f"{'Line' if p.get('time') else 'Scatter'} chart of {label_m} over "
                                                 f"{'time' if p.get('time') else 'the row order'} with the flagged points "
                                                 "marked; their count is given in the headline.")]
        n_rows = int(ev["rows"]["raw"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates,
            headline_template="{f2} of {f1} points stand out from the rest ({f3} of all points), the most extreme at a robust z of {f4}.",
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Modified z-score of {p['measure']} from the median and median absolute deviation, flagged beyond "
                           f"{Z_LIMIT}; fences at the quartiles plus or minus {IQR_K} interquartile ranges"
                           + (f"; scored on the residual from the median of each {b.grain} of {p['time']} (DuckDB)"
                              if p.get("time") else "; scored on the values in row order") + ".",
            rows_analyzed=n_rows, total_rows=n_rows, gate_words=GATE_WORDS,
            gate_refs={"date_parse": "/dates/unparsed_share", "flag_share": "/flagged/share",
                       "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        p = b.params
        extra = {"MEASURE_UNIT": b.unit, "TIME_PARSE": b.parse, "SEASON": b.grain, "Z_LIMIT": Z_LIMIT, "IQR_K": IQR_K,
                 **({} if p.get("time") else {"time": None})}
        cells = [
            sh.header_cell("anomalies", VERSION, SPEC.title),
            sh.parameters_cell(b, extra),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# duplicates are counted; each row stays a point below"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
        ]
        if p.get("time"):
            cells += [
                _nb.code(f"TIME_EXPR = {TIME_EXPR!r}\n"
                         "ts = (TIME_EXPR[TIME_PARSE].format(time=ident(time)) if TIME_PARSE in TIME_EXPR\n"
                         "      else TIME_EXPR['format'].format(time=ident(time), fmt=TIME_PARSE))\nts"),
                sh.sql_cell("SQL_PARSE", SQL_PARSE, "parse = con.sql(SQL_PARSE.format(time=ident(time), ts=ts)).df()\nparse"),
                sh.sql_cell("SQL_RESIDUALS", SQL_RESIDUALS,
                            "points = con.sql(SQL_RESIDUALS.format(ts=ts, measure=ident(measure), grain=SEASON)).df()\n"
                            "x = points['resid'].astype(float).to_numpy()"),
            ]
        else:
            cells.append(sh.sql_cell("SQL_VALUES", SQL_VALUES,
                                     "points = con.sql(SQL_VALUES.format(measure=ident(measure))).df()\n"
                                     "x = points['v'].astype(float).to_numpy()"))
        cells += [
            _nb.code("import numpy as np\n"
                     f"MAD_SCALE = {MAD_SCALE!r}\n"
                     "med = np.median(x)\nmad = np.median(np.abs(x - med))\n"
                     "points['z'] = MAD_SCALE * (x - med) / mad if mad else 0.0\n"
                     "points['flagged'] = points['z'].abs() > Z_LIMIT\n"
                     "q1, q3 = np.percentile(x, [25, 75])\n"
                     "fences = (q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1))\n"
                     "dict(n=len(points), flagged=int(points['flagged'].sum()), share=points['flagged'].mean(),\n"
                     "     outside_fences=int(((x < fences[0]) | (x > fences[1])).sum()), median=med, mad=mad)"),
            _nb.code("points[points['flagged']].reindex(points['z'].abs().sort_values(ascending=False).index).head(20)"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     + ("ax = points.plot(x='ts', y='v', title=f'{measure} with flagged points')\n"
                        "points[points['flagged']].plot.scatter(x='ts', y='v', ax=ax, color='red')"
                        if p.get("time") else
                        "ax = points.plot.scatter(x='idx', y='v', title=f'{measure} with flagged points')\n"
                        "points[points['flagged']].plot.scatter(x='idx', y='v', ax=ax, color='red')")),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
