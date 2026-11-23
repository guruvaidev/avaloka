"""``trend``: how a measure moved over time, at which grain, and how volatile
it is.

Periods are ``date_trunc`` buckets in DuckDB over distinct rows; the measure
is summed per period (averaged when it is a rate). Period-over-period changes,
the first-to-last change and the gaps in the regular period sequence come
from pandas. A text date column whose day/month order the data cannot settle
is asked about, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import GRAINS, ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import GATE_WORDS, build_report, figure
from . import _shared as sh

VERSION = "1.0"
MIN_PERIODS = 4              # fewer periods is not a trend (recommend.TREND_MIN_PERIODS)
UNPARSED_LIMIT = 0.05        # more unparseable dates than this is a data_quality caveat
TOP_SERIES = 5               # series drawn by dimension; the rest are "other"
DEFAULT_GRAIN = "month"
RATE_UNITS = ("fraction", "percent")

#: How the time column becomes a TIMESTAMP, by the parse rule bound at bind time.
TIME_EXPR = {
    "cast": "TRY_CAST({time} AS TIMESTAMP)",
    "year": "TRY_STRPTIME(CAST({time} AS VARCHAR), '%Y')",
    "format": "COALESCE(TRY_CAST({time} AS TIMESTAMP), TRY_STRPTIME({time}, '{fmt}'))",
}
PERIOD_FREQ = {"day": "D", "week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS"}
PERIOD_LABEL = {"day": "%Y-%m-%d", "week": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_PARSE = "SELECT count({time}) AS non_null, count({ts}) AS parsed FROM t"
SQL_PERIODS = ("WITH d AS ({dedup}) SELECT date_trunc('{grain}', {ts}) AS period, {agg}({measure}) AS value, "
               "count({measure}) AS n FROM d WHERE {ts} IS NOT NULL AND {measure} IS NOT NULL "
               "GROUP BY 1 ORDER BY 1")
SQL_PERIODS_BY = ("WITH d AS ({dedup}), top AS (SELECT {dimension} AS g FROM d WHERE {dimension} IS NOT NULL "
                  "GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {top}), "
                  "b AS (SELECT date_trunc('{grain}', {ts}) AS period, CASE WHEN {dimension} IN (SELECT g FROM top) "
                  "THEN CAST({dimension} AS VARCHAR) ELSE '{other}' END AS grp, {measure} AS v FROM d "
                  "WHERE {ts} IS NOT NULL AND {measure} IS NOT NULL) "
                  "SELECT period, grp, {agg}(v) AS value, count(v) AS n FROM b GROUP BY 1, 2 ORDER BY 1, 2")
SQL_TOTAL_RAW = "SELECT {agg}({measure}) AS total_raw FROM t"

EVIDENCE_CELL = """\
unit = evidence_unit(MEASURE_UNIT)
first, last = num(values.iloc[0]), num(values.iloc[-1])
evidence = {
    "rows": rows_leaves(counts), "dates": unparsed_leaves(parse),
    "periods": {"count": leaf(len(periods), "count"), "gaps": leaf(gaps, "count"),
                "first": leaf(first, unit), "last": leaf(last, unit), "change": leaf(last - first, unit),
                "change_mean": leaf(num(change.mean()) if len(change) else 0.0, unit),
                "change_std": leaf(num(change.std()) if len(change) > 1 else 0.0, unit),
                "peak": leaf(num(values.max()), unit), "low": leaf(num(values.min()), unit)},
    "measure": {"total": leaf(num(values.sum()) if AGG == "sum" else num(values.mean()), unit),
                "total_raw": leaf(num(total_raw), unit)},
}
if first != 0.0:
    evidence["periods"]["change_share"] = leaf((last - first) / abs(first), "fraction")
if dimension:
    evidence["series"] = {"count": leaf(periods_by["grp"].nunique(), "count")}
"""

SPEC = RecipeSpec(
    id="trend", version=VERSION, title="Trend of a measure over time",
    params=(ParamSpec("measure", "measure"), ParamSpec("time", "time"),
            ParamSpec("grain", "grain", required=False, choices=tuple(GRAINS)),
            ParamSpec("dimension", "dimension", required=False)),
    preconditions=("has_measure", "has_time", "min_periods_4"),
    quality_gates=("date_parse", "period_gaps", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)


@dataclass(frozen=True)
class TrendBound(sh.RecipeBound):
    unit: str = "unitless"   # the measure's semantic unit: a rate is averaged, anything else summed
    parse: str = "cast"      # "cast", "year", or a strptime format for a slash-date text column

    @property
    def agg(self) -> str:
        return "mean" if self.unit in RATE_UNITS else "sum"

    @property
    def time_expr(self) -> str:
        time = ident(self.params["time"])
        if self.parse in ("cast", "year"):
            return TIME_EXPR[self.parse].format(time=time)
        return TIME_EXPR["format"].format(time=time, fmt=self.parse)


def _column_dates(profile: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """The ``dates`` entry :func:`avaloka.recipes.run.build_profile` records for a column."""
    for col in profile.get("columns") or []:
        if isinstance(col, Mapping) and col.get("name") == name:
            return col.get("dates")
    return None


def _time_candidates(sem: Mapping[str, ColumnSemantics], profile: Mapping[str, Any]) -> list[str]:
    out = sh.columns_of_role(sem, "time")
    for col in profile.get("columns") or []:
        if isinstance(col, Mapping) and col.get("dates") and col["name"] not in out:
            out.append(str(col["name"]))
    return out


def _parse_rule(time: str, sem: Mapping[str, ColumnSemantics], profile: Mapping[str, Any]) -> str | Ask | Refusal:
    dates = _column_dates(profile, time)
    if dates and dates.get("convention") == "ambiguous":
        return Ask("time", "Dates in the time column could be read day-first or month-first; "
                          "which convention does it use?", ("day-first", "month-first"))
    if dates and dates.get("format"):
        return str(dates["format"])
    col = sem.get(time)
    if col is None or col.role != "time":
        return Refusal("no_time_column", "The time column named in the question does not hold dates.")
    if col.grain == "year" and dates is None and not _is_text(profile, time):
        return "year"
    return "cast"


def _is_text(profile: Mapping[str, Any], name: str) -> bool:
    for col in profile.get("columns") or []:
        if isinstance(col, Mapping) and col.get("name") == name:
            return str(col.get("dtype", "")) in ("object", "string", "str")
    return False


def _sql(b: TrendBound) -> dict[str, Any]:
    p = b.params
    args: dict[str, Any] = {"dedup": SQL_DEDUP, "measure": ident(p["measure"]), "time": ident(p["time"]),
                            "ts": b.time_expr, "grain": p["grain"], "agg": b.agg}
    if p.get("dimension"):
        args.update(dimension=ident(p["dimension"]), top=TOP_SERIES, other=sh.OTHER)
    return args


def _as_bound(bound: Bound) -> TrendBound:
    return bound if isinstance(bound, TrendBound) else TrendBound(bound.recipe_id, bound.version, bound.params)


def _periods(con: Any, b: TrendBound) -> pd.DataFrame:
    return query(con, SQL_PERIODS.format(**_sql(b)))


def _gaps(periods: pd.Series, grain: str) -> int:
    """Periods of the regular sequence from first to last that have no rows."""
    if len(periods) < 2:
        return 0
    expected = pd.date_range(periods.iloc[0], periods.iloc[-1], freq=PERIOD_FREQ[grain])
    return int(len(expected.difference(pd.DatetimeIndex(periods))))


def _label(ts: pd.Timestamp, grain: str) -> str:
    if grain == "quarter":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime(PERIOD_LABEL[grain])


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class Trend:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=True, missing_reason="no_measure",
                                 question="Which column holds the measure to follow over time?")
        if not isinstance(measure, str):
            return measure
        time = intent.params.get("time")
        if not isinstance(time, str):
            candidates = _time_candidates(sem, profile)
            if not candidates:
                return Refusal("no_time_column", "This data has no date column to follow over time.")
            if len(candidates) > 1:
                return Ask("time", "Which column holds the date?", tuple(candidates[:8]))
            time = candidates[0]
        parse = _parse_rule(time, sem, profile)
        if not isinstance(parse, str):
            return parse
        grain = intent.params.get("grain") or (sem[time].grain if time in sem else None) or DEFAULT_GRAIN
        if grain not in GRAINS:
            return Refusal("unsupported_question", "The time grain asked for is not one the recipe knows.")
        dimension = sh.bind_column(intent, sem, "dimension", ("dimension", "high_card_dimension"), required=False,
                                   missing_reason="ambiguous_dimension", question="Which column splits the series?")
        if isinstance(dimension, (Ask, Refusal)):
            return dimension
        params = {"measure": measure, "time": time, "grain": str(grain),
                  **({"dimension": dimension} if dimension else {})}
        return TrendBound("trend", VERSION, params, source=sh.source_of(profile),
                          unit=sem[measure].unit or "unitless", parse=parse)

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        args = _sql(b)
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        non_null, parsed = int(parse["non_null"]), int(parse["parsed"])
        unparsed = (non_null - parsed) / non_null if non_null else 1.0
        gates = [Gate("date_parse", "qualified", caveat_kind="data_quality", value=unparsed, unit="fraction")
                 if unparsed > UNPARSED_LIMIT else Gate("date_parse", "passed")]
        periods = _periods(con, b)
        if len(periods) < MIN_PERIODS:
            gates.append(Gate("period_gaps", "failed", reason="insufficient_periods", value=len(periods), unit="count"))
            return GateReport(tuple(gates))
        gaps = _gaps(periods["period"], b.params["grain"])
        gates.append(Gate("period_gaps", "qualified", caveat_kind="data_quality", value=gaps, unit="count")
                     if gaps else Gate("period_gaps", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        args, grain = _sql(b), b.params["grain"]
        unit = sh.evidence_unit(b.unit)
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        periods = _periods(con, b)
        values = periods["value"].astype(float)
        change = values.diff().dropna()
        first, last = _num(values.iloc[0]), _num(values.iloc[-1])
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "dates": {"unparsed_share": sh.leaf((int(parse["non_null"]) - int(parse["parsed"])) / int(parse["non_null"])
                                                if int(parse["non_null"]) else 1.0, "fraction")},
            "periods": {"count": sh.leaf(len(periods), "count"),
                        "gaps": sh.leaf(_gaps(periods["period"], grain), "count"),
                        "first": sh.leaf(first, unit), "last": sh.leaf(last, unit),
                        "change": sh.leaf(last - first, unit),
                        "change_mean": sh.leaf(_num(change.mean()) if len(change) else 0.0, unit),
                        "change_std": sh.leaf(_num(change.std()) if len(change) > 1 else 0.0, unit),
                        "peak": sh.leaf(_num(values.max()), unit), "low": sh.leaf(_num(values.min()), unit)},
            "measure": {"total": sh.leaf(_num(values.sum()) if b.agg == "sum" else _num(values.mean()), unit),
                        "total_raw": sh.leaf(_num(query(con, SQL_TOTAL_RAW.format(**args)).iloc[0, 0]), unit)},
        }
        if first != 0.0:
            tree["periods"]["change_share"] = sh.leaf((last - first) / abs(first), "fraction")
        labels = [_label(pd.Timestamp(t), grain) for t in periods["period"]]
        tables = {"periods": sh.table(["period", "value", "n"],
                                      [[lab, _num(v), int(n)] for lab, v, n in zip(labels, values, periods["n"])])}
        if b.params.get("dimension"):
            by = query(con, SQL_PERIODS_BY.format(**args))
            tree["series"] = {"count": sh.leaf(by["grp"].nunique(), "count")}
            tables["periods_by"] = sh.table(["period", "group", "value", "n"],
                                            [[_label(pd.Timestamp(r["period"]), grain), str(r["grp"]), _num(r["value"]), int(r["n"])]
                                             for _, r in by.iterrows()])
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        unit = sh.figure_unit(b.unit)
        d = 2
        figs = [figure("f1", ev, "/periods/count", unit="count", decimals=0),
                figure("f2", ev, "/periods/first", unit=unit, decimals=d),
                figure("f3", ev, "/periods/last", unit=unit, decimals=d),
                figure("f4", ev, "/periods/change", unit=unit, decimals=d),
                figure("f5", ev, "/periods/change_std", unit=unit, decimals=d),
                figure("f6", ev, "/periods/gaps", unit="count", decimals=0),
                figure("f7", ev, "/periods/peak", unit=unit, decimals=d),
                figure("f8", ev, "/periods/low", unit=unit, decimals=d),
                figure("f9", ev, "/rows/duplicates", unit="count", decimals=0)]
        headline = "Over {f1} periods the measure moved from {f2} to {f3}, a change of {f4}."
        kpi_last: dict[str, Any] = {"id": "last", "label": "Latest period", "value": "f3", "baseline": "f2",
                                    "baseline_label": "first period", "direction_good": "neutral"}
        if "change_share" in ev["periods"]:
            figs.append(figure("f10", ev, "/periods/change_share", unit="percent", decimals=1))
            headline = "Over {f1} periods the measure moved from {f2} to {f3}, a change of {f10}."
            kpi_last.update(delta="f10", delta_kind="pct")
        bullets = [{"so_what": "Period-to-period moves have a typical size of {f5}; the peak period reached {f7}, the lowest {f8}.",
                    "now_what": "Read the overall change against that typical move before calling it a trend."}]
        if ev["periods"]["gaps"]["value"] > 0:
            bullets.append({"so_what": "{f6} periods in the sequence have no rows at all.",
                            "now_what": "Check whether the gaps are missing data or genuine pauses."})
        if ev["rows"]["duplicates"]["value"] > 0:
            bullets.append({"so_what": "{f9} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so totals elsewhere are not inflated."})
        if "series" in ev and len(bullets) < 3:
            figs.append(figure("f11", ev, "/series/count", unit="count", decimals=0))
            bullets.append({"so_what": "The chart splits the measure into {f11} series.",
                            "now_what": "Compare the series' shapes, not only their levels."})
        kpis = [kpi_last,
                {"id": "volatility", "label": "Typical period move", "value": "f5", "direction_good": "down"},
                {"id": "periods", "label": "Periods", "value": "f1", "direction_good": "neutral"}]
        charts = [self._chart(evidence, b)]
        rows = int(ev["rows"]["raw"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates, headline_template=headline,
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"{b.agg.capitalize()} of {p['measure']} per {p['grain']} of {p['time']} over distinct rows "
                           f"in DuckDB" + (f", split by {p['dimension']}" if p.get("dimension") else "")
                           + "; period-over-period changes, their mean and standard deviation, and the gaps in the "
                             "regular period sequence from pandas.",
            rows_analyzed=rows, total_rows=rows,
            gate_refs={"date_parse": "/dates/unparsed_share", "period_gaps": "/periods/gaps",
                       "duplicates": "/rows/duplicates"})

    @staticmethod
    def _chart(evidence: Evidence, b: TrendBound) -> dict[str, Any]:
        p = b.params
        label_m, label_t = sh.label(p["measure"]), sh.label(p["time"])
        periods = [r[0] for r in evidence.tables["periods"]["rows"]]
        if "periods_by" in evidence.tables:
            rows = evidence.tables["periods_by"]["rows"]
            series: dict[str, list[Any]] = {}
            for g in dict.fromkeys(r[1] for r in rows):
                lookup = {r[0]: r[2] for r in rows if r[1] == g}
                series[g] = [lookup.get(per) for per in periods]
            return sh.line_chart("trend", f"{label_m} by {p['grain']} and {sh.label(p['dimension'])}", periods, series,
                                 x_name=label_t, y_name=label_m, series_name=sh.label(p["dimension"]),
                                 alt_text=f"Line chart of {b.agg} of {label_m} per {p['grain']} over {len(periods)} "
                                          f"periods, one line per {sh.label(p['dimension'])}.")
        values = [r[1] for r in evidence.tables["periods"]["rows"]]
        return sh.line_chart("trend", f"{label_m} by {p['grain']}", periods, {label_m: values},
                             x_name=label_t, y_name=label_m,
                             alt_text=f"Line chart of {b.agg} of {label_m} per {p['grain']} over {len(periods)} periods; "
                                      "the first and last values are given in the headline.")

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        p = b.params
        extra = {"MEASURE_UNIT": b.unit, "TIME_PARSE": b.parse, "AGG": b.agg,
                 **({} if p.get("dimension") else {"dimension": None})}
        args = "dedup=SQL_DEDUP, measure=ident(measure), time=ident(time), ts=ts, grain=grain, agg=AGG"
        cells = [
            sh.header_cell("trend", VERSION, SPEC.title),
            sh.parameters_cell(b, extra),
            sh.load_cell(b.source),
            _nb.code(f"TIME_EXPR = {TIME_EXPR!r}\n"
                     "ts = (TIME_EXPR[TIME_PARSE].format(time=ident(time)) if TIME_PARSE in TIME_EXPR\n"
                     "      else TIME_EXPR['format'].format(time=ident(time), fmt=TIME_PARSE))\nts"),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# every period below is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
            sh.sql_cell("SQL_PARSE", SQL_PARSE, "parse = con.sql(SQL_PARSE.format(time=ident(time), ts=ts)).df()\nparse"),
            sh.sql_cell("SQL_PERIODS", SQL_PERIODS, f"periods = con.sql(SQL_PERIODS.format({args})).df()\nperiods"),
        ]
        if p.get("dimension"):
            cells.append(sh.sql_cell("SQL_PERIODS_BY", SQL_PERIODS_BY,
                                     f"periods_by = con.sql(SQL_PERIODS_BY.format({args}, dimension=ident(dimension), "
                                     f"top={TOP_SERIES}, other={sh.OTHER!r})).df()\nperiods_by"))
        cells += [
            sh.sql_cell("SQL_TOTAL_RAW", SQL_TOTAL_RAW, "total_raw = con.sql(SQL_TOTAL_RAW.format(measure=ident(measure), agg=AGG)).df().iloc[0, 0]"),
            _nb.code(f"PERIOD_FREQ = {PERIOD_FREQ!r}\n"
                     "values = periods['value'].astype(float)\n"
                     "change = values.diff().dropna()\n"
                     "expected = pd.date_range(periods['period'].iloc[0], periods['period'].iloc[-1], freq=PERIOD_FREQ[grain])\n"
                     "gaps = len(expected.difference(pd.DatetimeIndex(periods['period'])))\n"
                     "summary = dict(periods=len(periods), first=values.iloc[0], last=values.iloc[-1],\n"
                     "               change=values.iloc[-1] - values.iloc[0], change_mean=change.mean(),\n"
                     "               change_std=change.std(), gaps=gaps)\nsummary"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     + ("periods_by.pivot(index='period', columns='grp', values='value').plot(title=f'{measure} by {grain} and {dimension}')"
                        if p.get("dimension") else "periods.plot(x='period', y='value', title=f'{measure} by {grain}')")),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
