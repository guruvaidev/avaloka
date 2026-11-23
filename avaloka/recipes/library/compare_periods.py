"""``compare_periods``: what changed between two windows of time, and which
groups of a dimension moved it.

The windows are decided from the latest date in the data. ``last_period`` is
the most recent complete period at the comparison grain (the time column's
own grain, no finer than a month) against the one before it; the other
windows are the closed vocabulary ``WINDOWS``. Both windows' aggregates and
the bridge by dimension are DuckDB SQL over distinct rows. A latest period
the data has not finished (fewer days than the grain implies, judged from the
latest date) is left out, the windows shift back one period, and the report
says so.

The ``incomplete_period`` gate is qualified when the latest period was
dropped and failed (``insufficient_periods``) when a window holds no rows.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import build_report, figure, merged_gate_words
from . import _shared as sh
from . import trend as _trend

VERSION = "1.0"
#: The ``window`` parameter's closed vocabulary (PR-4's ``specs.WINDOWS`` is this tuple).
WINDOWS: tuple[str, ...] = ("last_period", "last_month", "last_quarter", "last_year", "ytd", "all")
DEFAULT_WINDOW = "last_period"
TOP_CONTRIBUTORS = 8         # bridge steps shown by size of change; the rest are "other"
UNPARSED_LIMIT = 0.05        # more unparseable dates than this is a data_quality caveat
RATE_UNITS = ("fraction", "percent")
GRAIN_ORDER = ("day", "week", "month", "quarter", "year")
#: The comparison grain of ``last_period``: the time column's grain, no finer than a month.
PERIOD_GRAIN = {"day": "month", "week": "month", "month": "month", "quarter": "quarter", "year": "year"}
WINDOW_GRAIN = {"last_month": "month", "last_quarter": "quarter", "last_year": "year"}
GRAIN_OFFSET = {"month": pd.DateOffset(months=1), "quarter": pd.DateOffset(months=3), "year": pd.DateOffset(years=1)}
TIME_EXPR = _trend.TIME_EXPR
OTHER = sh.OTHER

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_PARSE = "SELECT count({time}) AS non_null, count({ts}) AS parsed FROM t"
SQL_SPAN = "SELECT min({ts}) AS first_ts, max({ts}) AS last_ts FROM t WHERE {measure} IS NOT NULL"
SQL_DAYS = ("SELECT count(DISTINCT CAST({ts} AS DATE)) AS days FROM t "
            "WHERE {ts} >= TIMESTAMP '{start}' AND {ts} < TIMESTAMP '{end}'")
SQL_WINDOW_CASE = ("CASE WHEN {ts} >= TIMESTAMP '{cur_start}' AND {ts} < TIMESTAMP '{cur_end}' THEN 'current' "
                   "WHEN {ts} >= TIMESTAMP '{prev_start}' AND {ts} < TIMESTAMP '{prev_end}' THEN 'previous' END")
SQL_WINDOWS = ("WITH d AS ({dedup}), w AS (SELECT " + SQL_WINDOW_CASE + " AS win, {measure} AS v FROM d "
               "WHERE {ts} IS NOT NULL AND {measure} IS NOT NULL) "
               "SELECT win, {agg}(v) AS value, count(v) AS n FROM w WHERE win IS NOT NULL GROUP BY 1 ORDER BY 1")
SQL_BRIDGE = ("WITH d AS ({dedup}), w AS (SELECT " + SQL_WINDOW_CASE + " AS win, {measure} AS v, "
              "CAST({dimension} AS VARCHAR) AS grp FROM d WHERE {ts} IS NOT NULL AND {measure} IS NOT NULL) "
              "SELECT grp, {agg}(CASE WHEN win = 'current' THEN v END) AS current, "
              "{agg}(CASE WHEN win = 'previous' THEN v END) AS previous, count(v) AS n "
              "FROM w WHERE win IS NOT NULL GROUP BY 1 ORDER BY 1")

EVIDENCE_CELL = """\
unit = evidence_unit(MEASURE_UNIT)
by_n = windows.set_index("win")["n"]
evidence = {
    "rows": rows_leaves(counts), "dates": unparsed_leaves(parse),
    "windows": {"current": {"value": leaf(num(by_window["current"]), unit), "n": leaf(by_n["current"], "count")},
                "previous": {"value": leaf(num(by_window["previous"]), unit), "n": leaf(by_n["previous"], "count")}},
    "change": {"absolute": leaf(change, unit)},
}
if relative is not None:
    evidence["change"]["relative"] = leaf(relative, "fraction")
if not complete:
    evidence["latest_period"] = {"days_seen": leaf(days_seen, "count"), "days_expected": leaf(days_expected, "count")}
if dimension:
    largest = max(steps, key=lambda s: abs(s[1])) if steps else (OTHER, 0.0)
    evidence["bridge"] = {"count": leaf(len(steps), "count"), "largest": leaf(largest[1], unit),
                          "by": {g: leaf(v, unit) for g, v in steps}}
"""

SPEC = RecipeSpec(
    id="compare_periods", version=VERSION, title="Compare two periods",
    params=(ParamSpec("measure", "measure"), ParamSpec("time", "time"),
            ParamSpec("window", "window", required=False, choices=WINDOWS),
            ParamSpec("dimension", "dimension", required=False)),
    preconditions=("has_measure", "has_time"),
    quality_gates=("date_parse", "incomplete_period", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

#: What the two windows are, for the reader, by window; no digits.
WINDOW_WORDS = {
    "last_period": "Both windows are whole periods at the same grain, so the two figures compare like for like.",
    "ytd": "Both windows run from the start of their year to the same date, so the two figures compare like for like.",
    "all": "The windows are the earlier and the later half of the dates in the data.",
}
WINDOW_WORDS.update({w: WINDOW_WORDS["last_period"] for w in ("last_month", "last_quarter", "last_year")})

GATE_WORDS = merged_gate_words({})


@dataclass(frozen=True)
class CompareBound(sh.RecipeBound):
    unit: str = "unitless"   # the measure's semantic unit: a rate is averaged, anything else summed
    parse: str = "cast"      # "cast", "year", or a strptime format for a slash-date text column
    data_grain: str = "day"  # the time column's own grain (semantics), which sets the comparison grain

    @property
    def agg(self) -> str:
        return "mean" if self.unit in RATE_UNITS else "sum"

    @property
    def time_expr(self) -> str:
        time = ident(self.params["time"])
        if self.parse in ("cast", "year"):
            return TIME_EXPR[self.parse].format(time=time)
        return TIME_EXPR["format"].format(time=time, fmt=self.parse)


# ── the windows (pure pandas; the notebook embeds this source) ─────────────

def period_start(ts: pd.Timestamp, grain: str) -> pd.Timestamp:
    if grain == "year":
        return pd.Timestamp(year=ts.year, month=1, day=1)
    month = ts.month if grain == "month" else (ts.month - 1) // 3 * 3 + 1
    return pd.Timestamp(year=ts.year, month=month, day=1)


def window_bounds(first: pd.Timestamp, last: pd.Timestamp, window: str, data_grain: str):
    """``(current, previous, grain, complete)``: each window as ``[start, end)``
    timestamps, the comparison grain (None for ``ytd`` and ``all``) and whether
    the latest period was complete. Only the first and last dates decide."""
    day = pd.Timedelta(days=1)
    if window == "ytd":
        start = pd.Timestamp(year=last.year, month=1, day=1)
        prev_start = pd.Timestamp(year=last.year - 1, month=1, day=1)
        prev_end = (last - pd.DateOffset(years=1)).normalize() + day
        return (start, last.normalize() + day), (prev_start, prev_end), None, True
    if window == "all":
        lo, hi = first.normalize(), last.normalize() + day
        mid = (lo + (hi - lo) / 2).normalize()
        return (mid, hi), (lo, mid), None, True
    grain = PERIOD_GRAIN[data_grain] if window == "last_period" else WINDOW_GRAIN[window]
    step = GRAIN_OFFSET[grain]
    latest = period_start(last, grain)
    complete = (GRAIN_ORDER.index(data_grain) >= GRAIN_ORDER.index(grain)
                or last.normalize() + day >= latest + step)
    cur_start = latest if complete else latest - step
    return (cur_start, cur_start + step), (cur_start - step, cur_start), grain, complete


@dataclass(frozen=True)
class Windows:
    current: tuple[pd.Timestamp, pd.Timestamp]
    previous: tuple[pd.Timestamp, pd.Timestamp]
    grain: str | None
    complete: bool
    window: str

    def labels(self) -> tuple[str, str]:
        """``(previous, current)`` labels for tables and charts."""
        if self.window == "ytd":
            y = self.current[0].year
            return f"{y - 1} to the same date", f"{y} to date"
        if self.window == "all":
            return "earlier half", "later half"
        return _trend._label(self.previous[0], self.grain), _trend._label(self.current[0], self.grain)

    def sql(self) -> dict[str, str]:
        fmt = "%Y-%m-%d %H:%M:%S"
        return {"cur_start": self.current[0].strftime(fmt), "cur_end": self.current[1].strftime(fmt),
                "prev_start": self.previous[0].strftime(fmt), "prev_end": self.previous[1].strftime(fmt)}


def _sql(b: CompareBound) -> dict[str, Any]:
    p = b.params
    args: dict[str, Any] = {"dedup": SQL_DEDUP, "measure": ident(p["measure"]), "time": ident(p["time"]),
                            "ts": b.time_expr, "agg": b.agg}
    if p.get("dimension"):
        args["dimension"] = ident(p["dimension"])
    return args


def _as_bound(bound: Bound) -> CompareBound:
    return bound if isinstance(bound, CompareBound) else CompareBound(bound.recipe_id, bound.version, bound.params)


def _windows(con: Any, b: CompareBound) -> Windows | None:
    span = query(con, SQL_SPAN.format(**_sql(b))).iloc[0]
    if pd.isna(span["first_ts"]) or pd.isna(span["last_ts"]):
        return None
    cur, prev, grain, complete = window_bounds(pd.Timestamp(span["first_ts"]), pd.Timestamp(span["last_ts"]),
                                               b.params["window"], b.data_grain)
    return Windows(cur, prev, grain, complete, b.params["window"])


def _window_values(con: Any, b: CompareBound, w: Windows) -> dict[str, tuple[float, int]]:
    rows = query(con, SQL_WINDOWS.format(**_sql(b), **w.sql()))
    return {str(r["win"]): (_num(r["value"]), int(r["n"])) for _, r in rows.iterrows()}


def _dropped_days(con: Any, b: CompareBound, w: Windows) -> tuple[int, int]:
    """``(days seen, days the period spans)`` of the excluded latest period."""
    start, end = w.current[1], w.current[1] + GRAIN_OFFSET[w.grain]
    fmt = "%Y-%m-%d %H:%M:%S"
    seen = query(con, SQL_DAYS.format(ts=b.time_expr, start=start.strftime(fmt), end=end.strftime(fmt))).iloc[0, 0]
    return int(seen), int((end - start).days)


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


def bridge_steps(rows: pd.DataFrame, agg: str, total_change: float) -> list[tuple[str, float]]:
    """``(group, change)`` steps from the bridge query's rows: the largest
    movers by size, then ``other`` holding the rest so a summed bridge
    reconciles exactly from the previous window to the current one. A rate
    (mean) has no ``other``. Pure pandas; the notebook embeds this source."""
    if agg == "sum":
        rows = rows.fillna({"current": 0.0, "previous": 0.0})
    else:
        rows = rows.dropna(subset=["current", "previous"])
    rows = rows.assign(change=rows["current"].astype(float) - rows["previous"].astype(float))
    rows = rows.assign(size=rows["change"].abs()).sort_values(["size", "grp"], ascending=[False, True])
    steps = [(str(r["grp"]), float(r["change"])) for _, r in rows.head(TOP_CONTRIBUTORS).iterrows()]
    if agg == "sum":
        other = total_change - sum(v for _, v in steps)
        if len(rows) > TOP_CONTRIBUTORS or abs(other) > 1e-9 * max(1.0, abs(total_change)):
            steps.append((OTHER, other))
    return steps


def _bridge(con: Any, b: CompareBound, w: Windows, total_change: float) -> list[tuple[str, float]]:
    return bridge_steps(query(con, SQL_BRIDGE.format(**_sql(b), **w.sql())), b.agg, total_change)


class ComparePeriods:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=True, missing_reason="no_measure",
                                 question="Which column holds the measure to compare across periods?")
        if not isinstance(measure, str):
            return measure
        time = intent.params.get("time")
        if not isinstance(time, str):
            candidates = _trend._time_candidates(sem, profile)
            if not candidates:
                return Refusal("no_time_column", "This data has no date column to split into periods.")
            if len(candidates) > 1:
                return Ask("time", "Which column holds the date?", tuple(candidates[:8]))
            time = candidates[0]
        parse = _trend._parse_rule(time, sem, profile)
        if not isinstance(parse, str):
            return parse
        window = intent.params.get("window") or DEFAULT_WINDOW
        if window not in WINDOWS:
            return Refusal("unsupported_question", "The period asked for is not one the recipe knows.")
        dimension = sh.bind_column(intent, sem, "dimension", ("dimension", "high_card_dimension"), required=False,
                                   missing_reason="ambiguous_dimension", question="Which column splits the change?")
        if isinstance(dimension, (Ask, Refusal)):
            return dimension
        data_grain = (sem[time].grain if time in sem else None) or "day"
        params = {"measure": measure, "time": time, "window": str(window),
                  **({"dimension": dimension} if dimension else {})}
        return CompareBound("compare_periods", VERSION, params, source=sh.source_of(profile),
                            unit=sem[measure].unit or "unitless", parse=parse, data_grain=data_grain)

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        args = _sql(b)
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        non_null, parsed = int(parse["non_null"]), int(parse["parsed"])
        unparsed = (non_null - parsed) / non_null if non_null else 1.0
        gates = [Gate("date_parse", "qualified", caveat_kind="data_quality", value=unparsed, unit="fraction")
                 if unparsed > UNPARSED_LIMIT else Gate("date_parse", "passed")]
        w = _windows(con, b)
        if w is None:
            gates.append(Gate("incomplete_period", "failed", reason="insufficient_periods", value=0, unit="count"))
            return GateReport(tuple(gates))
        wins = _window_values(con, b, w)
        if "current" not in wins or "previous" not in wins:
            gates.append(Gate("incomplete_period", "failed", reason="insufficient_periods", value=len(wins), unit="count"))
            return GateReport(tuple(gates))
        if w.complete:
            gates.append(Gate("incomplete_period", "passed"))
        else:
            seen, _ = _dropped_days(con, b, w)
            gates.append(Gate("incomplete_period", "qualified", caveat_kind="data_quality", value=seen, unit="count"))
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
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        w = _windows(con, b)
        assert w is not None  # check() refused an empty span
        wins = _window_values(con, b, w)
        (cur, cur_n), (prev, prev_n) = wins["current"], wins["previous"]
        change = cur - prev
        prev_label, cur_label = w.labels()
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "dates": {"unparsed_share": sh.leaf((int(parse["non_null"]) - int(parse["parsed"])) / int(parse["non_null"])
                                                if int(parse["non_null"]) else 1.0, "fraction")},
            "windows": {"current": {"value": sh.leaf(cur, unit), "n": sh.leaf(cur_n, "count")},
                        "previous": {"value": sh.leaf(prev, unit), "n": sh.leaf(prev_n, "count")}},
            "change": {"absolute": sh.leaf(change, unit)},
        }
        if prev != 0.0:
            tree["change"]["relative"] = sh.leaf(change / abs(prev), "fraction")
        if not w.complete:
            seen, expected = _dropped_days(con, b, w)
            tree["latest_period"] = {"days_seen": sh.leaf(seen, "count"), "days_expected": sh.leaf(expected, "count")}
        tables = {"windows": sh.table(["window", "label", "value", "n"],
                                      [["previous", prev_label, prev, prev_n], ["current", cur_label, cur, cur_n]])}
        if b.params.get("dimension"):
            steps = _bridge(con, b, w, change)
            largest = max(steps, key=lambda s: abs(s[1])) if steps else (sh.OTHER, 0.0)
            tree["bridge"] = {"count": sh.leaf(len(steps), "count"), "largest": sh.leaf(largest[1], unit),
                              "by": {g: sh.leaf(v, unit) for g, v in steps}}
            tables["bridge"] = sh.table(["group", "change"], [[g, v] for g, v in steps])
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        unit = sh.figure_unit(b.unit)
        d = 2
        figs = [figure("f1", ev, "/windows/current/value", unit=unit, decimals=d),
                figure("f2", ev, "/windows/previous/value", unit=unit, decimals=d),
                figure("f3", ev, "/change/absolute", unit=unit, decimals=d),
                figure("f5", ev, "/rows/duplicates", unit="count", decimals=0)]
        headline = "The measure moved from {f2} in the previous period to {f1} in the latest, a change of {f3}."
        kpi: dict[str, Any] = {"id": "latest", "label": "Latest period", "value": "f1", "baseline": "f2",
                               "baseline_label": "previous period", "direction_good": "neutral"}
        if "relative" in ev["change"]:
            figs.append(figure("f4", ev, "/change/relative", unit="percent", decimals=1))
            headline = "The measure moved from {f2} in the previous period to {f1} in the latest, a change of {f3} ({f4})."
            kpi.update(delta="f4", delta_kind="pct")
        bullets: list[dict[str, str]] = []
        if "bridge" in ev:
            figs.append(figure("f6", ev, "/bridge/count", unit="count", decimals=0))
            figs.append(figure("f7", ev, "/bridge/largest", unit=unit, decimals=d))
            bullets.append({"so_what": "The largest single mover shifted the measure by {f7}; the bridge spans {f6} groups.",
                            "now_what": "Explain the largest mover before explaining the total."})
        if "latest_period" in ev:
            figs.append(figure("f8", ev, "/latest_period/days_seen", unit="count", decimals=0))
            bullets.append({"so_what": "The latest period holds {f8} days of data and is not finished, so it was left out "
                                       "and the windows moved back one period.",
                            "now_what": "Re-run once the period closes to compare it."})
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f5} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so totals elsewhere are not inflated."})
        if len(bullets) < 3:
            bullets.append({"so_what": WINDOW_WORDS.get(p["window"], WINDOW_WORDS["last_period"]),
                            "now_what": "Check the window boundaries in the data table before reading the change."})
        kpis = [kpi, {"id": "change", "label": "Change", "value": "f3", "direction_good": "neutral"}]
        prev_label, cur_label = evidence.tables["windows"]["rows"][0][1], evidence.tables["windows"]["rows"][1][1]
        label_m = sh.label(p["measure"])
        if "bridge" in evidence.tables and b.agg == "sum":
            steps = [(r[0], r[1]) for r in evidence.tables["bridge"]["rows"]]
            charts = [sh.waterfall_chart("bridge", f"{label_m}: {prev_label} to {cur_label} by {sh.label(p['dimension'])}",
                                         (prev_label, ev["windows"]["previous"]["value"]["value"]), steps,
                                         measure_label=label_m)]
        elif "bridge" in evidence.tables:
            rows = evidence.tables["bridge"]["rows"]
            charts = [sh.bar_chart("bridge", f"Change in {label_m} by {sh.label(p['dimension'])}",
                                   [r[0] for r in rows], [r[1] for r in rows],
                                   x_name=sh.label(p["dimension"]), y_name=f"change in {label_m}",
                                   alt_text=f"Bar chart of the change in mean {label_m} from {prev_label} to {cur_label} "
                                            f"for the largest movers by {sh.label(p['dimension'])}.")]
        else:
            charts = [sh.bar_chart("windows", f"{label_m}: {prev_label} against {cur_label}", [prev_label, cur_label],
                                   [ev["windows"]["previous"]["value"]["value"], ev["windows"]["current"]["value"]["value"]],
                                   x_name="window", y_name=label_m,
                                   alt_text=f"Two bars: {b.agg} of {label_m} in {prev_label} and in {cur_label}; "
                                            "the change is given in the headline.")]
        rows = int(ev["rows"]["raw"]["value"])
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates, headline_template=headline,
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"{b.agg.capitalize()} of {p['measure']} in two windows of {p['time']} ({p['window']}: "
                           f"{prev_label} against {cur_label}) over distinct rows in DuckDB; absolute and relative change"
                           + (f"; bridge of the change by {p['dimension']}, largest movers first, the rest as other"
                              if p.get("dimension") else "") + ".",
            rows_analyzed=rows, total_rows=rows, gate_words=GATE_WORDS,
            gate_refs={"date_parse": "/dates/unparsed_share", "incomplete_period": "/latest_period/days_seen",
                       "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        p = b.params
        extra = {"MEASURE_UNIT": b.unit, "TIME_PARSE": b.parse, "AGG": b.agg, "DATA_GRAIN": b.data_grain,
                 "TOP_CONTRIBUTORS": TOP_CONTRIBUTORS, **({} if p.get("dimension") else {"dimension": None})}
        args = "dedup=SQL_DEDUP, measure=ident(measure), time=ident(time), ts=ts, agg=AGG"
        cells = [
            sh.header_cell("compare_periods", VERSION, SPEC.title),
            sh.parameters_cell(b, extra),
            sh.load_cell(b.source),
            _nb.code(f"TIME_EXPR = {TIME_EXPR!r}\n"
                     "ts = (TIME_EXPR[TIME_PARSE].format(time=ident(time)) if TIME_PARSE in TIME_EXPR\n"
                     "      else TIME_EXPR['format'].format(time=ident(time), fmt=TIME_PARSE))\nts"),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# every window below is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
            sh.sql_cell("SQL_PARSE", SQL_PARSE, "parse = con.sql(SQL_PARSE.format(time=ident(time), ts=ts)).df()\nparse"),
            sh.sql_cell("SQL_SPAN", SQL_SPAN, "span = con.sql(SQL_SPAN.format(ts=ts, measure=ident(measure))).df().iloc[0]\n"
                                               "first, last = pd.Timestamp(span['first_ts']), pd.Timestamp(span['last_ts'])"),
            _nb.code(f"GRAIN_ORDER = {GRAIN_ORDER!r}\nPERIOD_GRAIN = {PERIOD_GRAIN!r}\nWINDOW_GRAIN = {WINDOW_GRAIN!r}\n"
                     "GRAIN_OFFSET = {'month': pd.DateOffset(months=1), 'quarter': pd.DateOffset(months=3), "
                     "'year': pd.DateOffset(years=1)}\n\n"
                     + inspect.getsource(period_start) + "\n\n" + inspect.getsource(window_bounds) + "\n"
                     "current, previous, grain, complete = window_bounds(first, last, window, DATA_GRAIN)\n"
                     "fmt = '%Y-%m-%d %H:%M:%S'\n"
                     "bounds = dict(cur_start=current[0].strftime(fmt), cur_end=current[1].strftime(fmt),\n"
                     "              prev_start=previous[0].strftime(fmt), prev_end=previous[1].strftime(fmt))\nbounds, complete"),
            sh.sql_cell("SQL_WINDOWS", SQL_WINDOWS, f"windows = con.sql(SQL_WINDOWS.format({args}, **bounds)).df()\nwindows"),
            _nb.code("by_window = windows.set_index('win')['value']\n"
                     "change = by_window['current'] - by_window['previous']\n"
                     "relative = change / abs(by_window['previous']) if by_window['previous'] else None\n"
                     "change, relative"),
            sh.sql_cell("SQL_DAYS", SQL_DAYS,
                        "days_seen = days_expected = None  # the latest period, when it was left out as unfinished\n"
                        "if not complete:\n"
                        "    start, end = current[1], current[1] + GRAIN_OFFSET[grain]\n"
                        "    days_seen = int(con.sql(SQL_DAYS.format(ts=ts, start=start.strftime(fmt), end=end.strftime(fmt))).df().iloc[0, 0])\n"
                        "    days_expected = int((end - start).days)\n"
                        "days_seen, days_expected"),
        ]
        if p.get("dimension"):
            cells.append(sh.sql_cell("SQL_BRIDGE", SQL_BRIDGE,
                                     f"OTHER = {OTHER!r}\n\n" + inspect.getsource(bridge_steps) + "\n"
                                     f"bridge = con.sql(SQL_BRIDGE.format({args}, dimension=ident(dimension), **bounds)).df()\n"
                                     "steps = bridge_steps(bridge, AGG, change)\n"
                                     "pd.DataFrame(steps, columns=['group', 'change'])"))
        cells += [
            _nb.code("import matplotlib.pyplot as plt\n"
                     + ("pd.DataFrame(steps, columns=['group', 'change']).plot.bar(x='group', y='change', title=f'Change in {measure} by {dimension}')"
                        if p.get("dimension") else "windows.plot.bar(x='win', y='value', title=f'{measure} by window')")),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
