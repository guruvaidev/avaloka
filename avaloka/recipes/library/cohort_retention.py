"""``cohort_retention``: how cohorts of ids keep coming back, period by period.

Each id's first active period is its cohort; a period is a month, or a week
when the data spans less than three months. The cohort × period-number
matrix of active ids is DuckDB SQL over distinct rows (an id is active in a
period when it has any row there, or any row with the measure above zero
when a measure is bound); retention per cell, cohort sizes and the
size-weighted average curve come from pandas. A latest period the data has
not finished is left out of the matrix and the report says so.

``min_rows`` fails below the id floor, ``min_cohorts`` fails
(``insufficient_periods``) below three cohorts, ``incomplete_period``
qualifies when the latest period was dropped, ``small_groups`` qualifies as
``small_n`` when a cohort is small. The wording is associational: a cohort's
curve is what happened to it, never why.
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
MIN_IDS = 30                 # fewer distinct active ids is a refusal
MIN_COHORTS = 3              # fewer cohorts is a refusal (the design's COHORT_MIN)
SMALL_COHORT = 30            # a cohort with fewer ids is a small_n caveat
UNPARSED_LIMIT = 0.05        # more unparseable dates than this is a data_quality caveat
SHORT_SPAN_DAYS = 90         # a span shorter than this is read at week grain, else month
MAX_COHORTS_SHOWN = 24       # cohort rows drawn in the heatmap; the matrix table holds them all
GRAIN_OFFSET = {"week": pd.DateOffset(weeks=1), "month": pd.DateOffset(months=1)}
TIME_EXPR = _trend.TIME_EXPR
ANY_ROW = "TRUE"             # the activity clause when no measure is bound

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_PARSE = "SELECT count({time}) AS non_null, count({ts}) AS parsed FROM t"
SQL_SPAN = ("SELECT min({ts}) AS first_ts, max({ts}) AS last_ts, count(DISTINCT {id}) AS ids FROM t "
            "WHERE {id} IS NOT NULL AND {ts} IS NOT NULL AND {active}")
SQL_DAYS = ("SELECT count(DISTINCT CAST({ts} AS DATE)) AS days FROM t "
            "WHERE {ts} >= TIMESTAMP '{start}' AND {ts} < TIMESTAMP '{end}'")
SQL_MATRIX = ("WITH d AS ({dedup}), a AS (SELECT {id} AS id, date_trunc('{grain}', {ts}) AS period FROM d "
              "WHERE {id} IS NOT NULL AND {ts} IS NOT NULL AND {active} GROUP BY 1, 2), "
              "c AS (SELECT id, min(period) AS cohort FROM a GROUP BY 1) "
              "SELECT c.cohort, date_diff('{grain}', c.cohort, a.period) AS period_number, count(DISTINCT a.id) AS active "
              "FROM a JOIN c USING (id) WHERE a.period < TIMESTAMP '{cutoff}' GROUP BY 1, 2 ORDER BY 1, 2")

EVIDENCE_CELL = """\
evidence = {
    "rows": rows_leaves(counts), "dates": unparsed_leaves(parse),
    "ids": {"count": leaf(int(sizes.sum()), "count")},
    "cohorts": {"count": leaf(len(sizes), "count"), "smallest": leaf(int(sizes.min()), "count"),
                "largest": leaf(int(sizes.max()), "count")},
    "periods": {"observed": leaf(int(active.columns.max()), "count")},
    "retention": {"by": {f"p{int(k)}": leaf(v, "fraction") for k, v in curve.items() if not pd.isna(v)}},
}
for k in (1, 3):
    if k in curve.index and not pd.isna(curve[k]):
        evidence["retention"][f"period_{k}"] = leaf(curve[k], "fraction")
if not complete:
    evidence["latest_period"] = {"days_seen": leaf(days_seen, "count"), "days_expected": leaf(days_expected, "count")}
"""

SPEC = RecipeSpec(
    id="cohort_retention", version=VERSION, title="Cohort retention",
    params=(ParamSpec("id", "id"), ParamSpec("time", "time"), ParamSpec("measure", "measure", required=False)),
    preconditions=("has_id", "has_time"),
    quality_gates=("min_rows", "date_parse", "min_cohorts", "incomplete_period", "small_groups", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "min_rows": "There are too few ids for a cohort analysis.",
    "incomplete_period": "The latest period is incomplete and was left out of the matrix.",
    "small_groups": "Some cohorts hold few ids, so their retention figures are rough.",
})


@dataclass(frozen=True)
class CohortBound(sh.RecipeBound):
    parse: str = "cast"      # "cast", "year", or a strptime format for a slash-date text column

    @property
    def time_expr(self) -> str:
        time = ident(self.params["time"])
        if self.parse in ("cast", "year"):
            return TIME_EXPR[self.parse].format(time=time)
        return TIME_EXPR["format"].format(time=time, fmt=self.parse)

    @property
    def active(self) -> str:
        measure = self.params.get("measure")
        return f"{ident(measure)} > 0" if measure else ANY_ROW


# ── the grain and the cutoff (pure pandas; the notebook embeds this source) ─

def cohort_grain(first: pd.Timestamp, last: pd.Timestamp) -> str:
    """Month, or week when the data spans less than SHORT_SPAN_DAYS."""
    return "week" if (last - first).days < SHORT_SPAN_DAYS else "month"


def period_start(ts: pd.Timestamp, grain: str) -> pd.Timestamp:
    if grain == "month":
        return pd.Timestamp(year=ts.year, month=ts.month, day=1)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def periods_between(start: pd.Timestamp, end: pd.Timestamp, grain: str) -> int:
    """Whole periods from ``start`` (a period start) up to ``end``."""
    if grain == "month":
        return (end.year - start.year) * 12 + end.month - start.month
    return (end - start).days // 7


def matrix_cutoff(last: pd.Timestamp, grain: str) -> tuple[pd.Timestamp, bool]:
    """``(cutoff, complete)``: periods starting before ``cutoff`` are in the matrix.
    The latest period is complete when the last date reaches its final day."""
    latest = period_start(last, grain)
    step = GRAIN_OFFSET[grain]
    complete = last.normalize() + pd.Timedelta(days=1) >= latest + step
    return (latest + step if complete else latest), complete


def retention(matrix: pd.DataFrame, cutoff: pd.Timestamp, grain: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """``(active, rate, sizes)``: the cohort × period-number matrix of active
    ids (zero where a cohort had none in an observable period, NaN beyond
    the periods the cohort could have reached), the same matrix as a share
    of the cohort, and each cohort's size (its ids at period zero)."""
    active = matrix.pivot(index="cohort", columns="period_number", values="active").astype(float)
    active = active.reindex(columns=range(int(active.columns.max()) + 1)).fillna(0.0)
    for cohort in active.index:
        observable = periods_between(pd.Timestamp(cohort), cutoff, grain)
        active.loc[cohort, [k for k in active.columns if k >= observable]] = float("nan")
    sizes = active[0]
    return active, active.div(sizes, axis=0), sizes


def average_curve(active: pd.DataFrame, sizes: pd.Series) -> pd.Series:
    """Retention per period number over the cohorts that reached it, weighted by cohort size."""
    out = {}
    for k in active.columns:
        reached = active[k].notna()
        out[int(k)] = float(active.loc[reached, k].sum() / sizes[reached].sum()) if reached.any() else float("nan")
    return pd.Series(out)


# ── plumbing ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Frame:
    grain: str
    cutoff: pd.Timestamp
    complete: bool
    first: pd.Timestamp
    last: pd.Timestamp
    ids: int


def _sql(b: CohortBound) -> dict[str, Any]:
    p = b.params
    return {"dedup": SQL_DEDUP, "id": ident(p["id"]), "time": ident(p["time"]), "ts": b.time_expr, "active": b.active}


def _as_bound(bound: Bound) -> CohortBound:
    return bound if isinstance(bound, CohortBound) else CohortBound(bound.recipe_id, bound.version, bound.params)


def _frame(con: Any, b: CohortBound) -> Frame | None:
    span = query(con, SQL_SPAN.format(**_sql(b))).iloc[0]
    if pd.isna(span["first_ts"]) or pd.isna(span["last_ts"]):
        return None
    first, last = pd.Timestamp(span["first_ts"]), pd.Timestamp(span["last_ts"])
    grain = cohort_grain(first, last)
    cutoff, complete = matrix_cutoff(last, grain)
    return Frame(grain, cutoff, complete, first, last, int(span["ids"]))


def _matrix(con: Any, b: CohortBound, f: Frame) -> pd.DataFrame:
    sql = SQL_MATRIX.format(**_sql(b), grain=f.grain, cutoff=f.cutoff.strftime("%Y-%m-%d %H:%M:%S"))
    out = query(con, sql)
    out["cohort"] = pd.to_datetime(out["cohort"])
    return out


def _dropped_days(con: Any, b: CohortBound, f: Frame) -> tuple[int, int]:
    """``(days seen, days the period spans)`` of the excluded latest period."""
    start, end = f.cutoff, f.cutoff + GRAIN_OFFSET[f.grain]
    fmt = "%Y-%m-%d %H:%M:%S"
    seen = query(con, SQL_DAYS.format(ts=b.time_expr, start=start.strftime(fmt), end=end.strftime(fmt))).iloc[0, 0]
    return int(seen), int((end - start).days)


def _label(ts: pd.Timestamp, grain: str) -> str:
    return _trend._label(ts, grain)


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class CohortRetention:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        entity = sh.bind_column(intent, sem, "id", ("id",), required=True, missing_reason="unsupported_question",
                                question="Which column identifies the entity whose cohort is followed?")
        if isinstance(entity, Refusal):
            return Refusal("unsupported_question", "This data has no id column, so there is nothing to follow across periods.")
        if not isinstance(entity, str):
            return entity
        time = intent.params.get("time")
        if not isinstance(time, str):
            candidates = _trend._time_candidates(sem, profile)
            if not candidates:
                return Refusal("no_time_column", "This data has no date column to form cohorts from.")
            if len(candidates) > 1:
                return Ask("time", "Which column holds the date?", tuple(candidates[:8]))
            time = candidates[0]
        parse = _trend._parse_rule(time, sem, profile)
        if not isinstance(parse, str):
            return parse
        measure = sh.bind_column(intent, sem, "measure", ("measure",), required=False, missing_reason="no_measure",
                                 question="Which column says whether a row counts as activity?")
        if isinstance(measure, (Ask, Refusal)):
            return measure
        params = {"id": entity, "time": time, **({"measure": measure} if measure else {})}
        return CohortBound("cohort_retention", VERSION, params, source=sh.source_of(profile), parse=parse)

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        args = _sql(b)
        f = _frame(con, b)
        if f is None or f.ids < MIN_IDS:
            return GateReport((Gate("min_rows", "failed", reason="insufficient_rows", value=0 if f is None else f.ids, unit="count"),))
        gates = [Gate("min_rows", "passed")]
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        non_null, parsed = int(parse["non_null"]), int(parse["parsed"])
        unparsed = (non_null - parsed) / non_null if non_null else 1.0
        gates.append(Gate("date_parse", "qualified", caveat_kind="data_quality", value=unparsed, unit="fraction")
                     if unparsed > UNPARSED_LIMIT else Gate("date_parse", "passed"))
        matrix = _matrix(con, b, f)
        cohorts = int(matrix["cohort"].nunique()) if len(matrix) else 0
        if cohorts < MIN_COHORTS:
            gates.append(Gate("min_cohorts", "failed", reason="insufficient_periods", value=cohorts, unit="count"))
            return GateReport(tuple(gates))
        gates.append(Gate("min_cohorts", "passed"))
        if f.complete:
            gates.append(Gate("incomplete_period", "passed"))
        else:
            seen, _ = _dropped_days(con, b, f)
            gates.append(Gate("incomplete_period", "qualified", caveat_kind="data_quality", value=seen, unit="count"))
        _, _, sizes = retention(matrix, f.cutoff, f.grain)
        smallest = int(sizes.min())
        gates.append(Gate("small_groups", "qualified", caveat_kind="small_n", value=smallest, unit="count")
                     if smallest < SMALL_COHORT else Gate("small_groups", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        args = _sql(b)
        counts = query(con, SQL_ROW_COUNTS.format(**args)).iloc[0]
        parse = query(con, SQL_PARSE.format(**args)).iloc[0]
        f = _frame(con, b)
        assert f is not None  # check() refused an empty span
        matrix = _matrix(con, b, f)
        active, rate, sizes = retention(matrix, f.cutoff, f.grain)
        curve = average_curve(active, sizes)
        observed = int(active.columns.max())
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "dates": {"unparsed_share": sh.leaf((int(parse["non_null"]) - int(parse["parsed"])) / int(parse["non_null"])
                                                if int(parse["non_null"]) else 1.0, "fraction")},
            "ids": {"count": sh.leaf(int(sizes.sum()), "count")},
            "cohorts": {"count": sh.leaf(len(sizes), "count"), "smallest": sh.leaf(int(sizes.min()), "count"),
                        "largest": sh.leaf(int(sizes.max()), "count"),
                        "by": {_label(pd.Timestamp(c), f.grain): {"size": sh.leaf(int(s), "count")} for c, s in sizes.items()}},
            "periods": {"observed": sh.leaf(observed, "count")},
            "retention": {"by": {f"p{int(k)}": sh.leaf(v, "fraction") for k, v in curve.items() if not pd.isna(v)}},
        }
        for k in (1, 3):
            if k in curve.index and not pd.isna(curve[k]):
                tree["retention"][f"period_{k}"] = sh.leaf(curve[k], "fraction")
        if not f.complete:
            seen, expected = _dropped_days(con, b, f)
            tree["latest_period"] = {"days_seen": sh.leaf(seen, "count"), "days_expected": sh.leaf(expected, "count")}
        matrix_rows = [[_label(pd.Timestamp(c), f.grain), int(k), int(active.loc[c, k]), int(sizes[c]), _num(rate.loc[c, k])]
                       for c in active.index for k in active.columns if not pd.isna(active.loc[c, k])]
        curve_rows = [[int(k), _num(v), int(active[k].notna().sum()), int(sizes[active[k].notna()].sum())]
                      for k, v in curve.items() if not pd.isna(v)]
        tables = {"frame": sh.table(["grain", "cutoff", "complete"], [[f.grain, f.cutoff.strftime("%Y-%m-%d"), int(f.complete)]]),
                  "matrix": sh.table(["cohort", "period_number", "active", "size", "retention"], matrix_rows),
                  "curve": sh.table(["period_number", "retention", "cohorts", "ids"], curve_rows)}
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        figs = [figure("f1", ev, "/cohorts/count", unit="count", decimals=0),
                figure("f2", ev, "/ids/count", unit="count", decimals=0),
                figure("f5", ev, "/cohorts/smallest", unit="count", decimals=0),
                figure("f6", ev, "/rows/duplicates", unit="count", decimals=0),
                figure("f8", ev, "/periods/observed", unit="count", decimals=0),
                figure("f9", ev, "/cohorts/largest", unit="count", decimals=0)]
        headline = "Across {f1} cohorts holding {f2} ids, the matrix reaches {f8} periods past the first."
        kpis = [{"id": "cohorts", "label": "Cohorts", "value": "f1", "direction_good": "neutral"},
                {"id": "ids", "label": "Ids", "value": "f2", "direction_good": "neutral"}]
        if "period_1" in ev["retention"]:
            figs.append(figure("f3", ev, "/retention/period_1", unit="percent", decimals=1))
            headline = "Across {f1} cohorts holding {f2} ids, {f3} of a cohort is still active one period after its first."
            kpis.insert(0, {"id": "period_1", "label": "Retention one period on", "value": "f3", "direction_good": "up"})
        if "period_3" in ev["retention"]:
            figs.append(figure("f4", ev, "/retention/period_3", unit="percent", decimals=1))
            headline = ("Across {f1} cohorts holding {f2} ids, {f3} of a cohort is still active one period after its "
                        "first and {f4} three periods after.")
        bullets = [{"so_what": "The smallest cohort holds {f5} ids and the largest {f9}; the average curve weights each "
                               "cohort by its size.",
                    "now_what": "Read a small cohort's row of the matrix as rough; compare cohorts at the same period "
                                "number, not the same calendar period."}]
        if "latest_period" in ev:
            figs.append(figure("f7", ev, "/latest_period/days_seen", unit="count", decimals=0))
            bullets.append({"so_what": "The latest period holds {f7} days of data and is not finished, so it was left out "
                                       "of the matrix.",
                            "now_what": "Re-run once the period closes to include it."})
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f6} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so counts elsewhere are not inflated."})
        if len(bullets) < 3:
            bullets.append({"so_what": "Each id's first active period is its cohort; a later cell is the share of that "
                                       "cohort active in that period.",
                            "now_what": "Look for a cohort whose row sits above or below the others before reading the average."})
        charts = self._charts(evidence, b)
        n_rows = int(ev["rows"]["raw"]["value"])
        activity = f"any row with {p['measure']} above zero" if p.get("measure") else "any row"
        grain = str(evidence.tables["frame"]["rows"][0][0])
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates, headline_template=headline,
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Each {p['id']}'s first active period of {p['time']} at {grain} grain is its cohort "
                           f"(activity: {activity}); the cohort by period-number matrix of active ids over distinct rows "
                           "in DuckDB; retention per cell and the size-weighted average curve in pandas; an unfinished "
                           "latest period is left out.",
            rows_analyzed=n_rows, total_rows=n_rows, gate_words=GATE_WORDS,
            gate_refs={"date_parse": "/dates/unparsed_share", "incomplete_period": "/latest_period/days_seen",
                       "small_groups": "/cohorts/smallest", "duplicates": "/rows/duplicates"})

    @staticmethod
    def _charts(evidence: Evidence, b: CohortBound) -> list[dict[str, Any]]:
        rows = evidence.tables["matrix"]["rows"]
        cohorts = list(dict.fromkeys(r[0] for r in rows))[-MAX_COHORTS_SHOWN:]
        periods = sorted({int(r[1]) for r in rows})
        cells = {(r[0], int(r[1])): r[4] for r in rows}
        matrix = [[cells.get((c, k)) for k in periods] for c in cohorts]
        curve = evidence.tables["curve"]["rows"]
        label_id = sh.label(b.params["id"])
        return [sh.cohort_heatmap_chart("cohorts", f"Retention of {label_id} cohorts by period", cohorts,
                                        [str(k) for k in periods], matrix, value_name="retention",
                                        alt_text=f"Heatmap of {len(cohorts)} cohorts of {label_id} by period number; each "
                                                 "cell is the share of the cohort active in that period."),
                sh.line_chart("curve", "Average retention by period number", [str(r[0]) for r in curve],
                              {"retention": [r[1] for r in curve]}, x_name="period number", y_name="retention",
                              alt_text=f"Line of the size-weighted average retention over {len(curve)} period numbers; "
                                       "the first-period figure is given in the headline.")]

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        p = b.params
        extra = {"TIME_PARSE": b.parse, **({} if p.get("measure") else {"measure": None})}
        cells = [
            sh.header_cell("cohort_retention", VERSION, SPEC.title),
            sh.parameters_cell(b, extra),
            sh.load_cell(b.source),
            _nb.code(f"TIME_EXPR = {TIME_EXPR!r}\n"
                     "ts = (TIME_EXPR[TIME_PARSE].format(time=ident(time)) if TIME_PARSE in TIME_EXPR\n"
                     "      else TIME_EXPR['format'].format(time=ident(time), fmt=TIME_PARSE))\n"
                     f"active = f'{{ident(measure)}} > 0' if measure else {ANY_ROW!r}\n"
                     "args = dict(dedup=None, id=ident(id), time=ident(time), ts=ts, active=active)\nts, active"),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "args['dedup'] = SQL_DEDUP  # the matrix is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(**args)).df()\ncounts"),
            sh.sql_cell("SQL_PARSE", SQL_PARSE, "parse = con.sql(SQL_PARSE.format(**args)).df()\nparse"),
            sh.sql_cell("SQL_SPAN", SQL_SPAN, "span = con.sql(SQL_SPAN.format(**args)).df().iloc[0]\n"
                                               "first, last = pd.Timestamp(span['first_ts']), pd.Timestamp(span['last_ts'])\nspan"),
            _nb.code(f"SHORT_SPAN_DAYS = {SHORT_SPAN_DAYS}\n"
                     "GRAIN_OFFSET = {'week': pd.DateOffset(weeks=1), 'month': pd.DateOffset(months=1)}\n\n"
                     + inspect.getsource(cohort_grain) + "\n\n" + inspect.getsource(period_start) + "\n\n"
                     + inspect.getsource(periods_between) + "\n\n" + inspect.getsource(matrix_cutoff) + "\n"
                     "grain = cohort_grain(first, last)\n"
                     "cutoff, complete = matrix_cutoff(last, grain)\n"
                     "grain, cutoff, complete"),
            sh.sql_cell("SQL_MATRIX", SQL_MATRIX,
                        "matrix = con.sql(SQL_MATRIX.format(**args, grain=grain, cutoff=cutoff.strftime('%Y-%m-%d %H:%M:%S'))).df()\n"
                        "matrix['cohort'] = pd.to_datetime(matrix['cohort'])\nmatrix"),
            _nb.code(inspect.getsource(retention) + "\n\n" + inspect.getsource(average_curve) + "\n"
                     "active, rate, sizes = retention(matrix, cutoff, grain)\n"
                     "curve = average_curve(active, sizes)\n"
                     "dict(cohorts=len(sizes), ids=int(sizes.sum()), period_1=curve.get(1), period_3=curve.get(3))"),
            sh.sql_cell("SQL_DAYS", SQL_DAYS,
                        "days_seen = days_expected = None  # the latest period, when it was left out as unfinished\n"
                        "if not complete:\n"
                        "    start, end = cutoff, cutoff + GRAIN_OFFSET[grain]\n"
                        "    fmt = '%Y-%m-%d %H:%M:%S'\n"
                        "    days_seen = int(con.sql(SQL_DAYS.format(ts=ts, start=start.strftime(fmt), end=end.strftime(fmt))).df().iloc[0, 0])\n"
                        "    days_expected = int((end - start).days)\n"
                        "days_seen, days_expected"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
                     "ax1.imshow(rate.values, aspect='auto', vmin=0, vmax=1)\n"
                     "ax1.set_yticks(range(len(rate.index)))\n"
                     "ax1.set_yticklabels([str(c.date()) for c in rate.index])\n"
                     "ax1.set_title(f'Retention of {id} cohorts by {grain}')\n"
                     "curve.plot(ax=ax2, title='Average retention by period number')"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
