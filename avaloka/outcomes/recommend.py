"""Deterministic chart recommender (§4.2 of docs/design/2.0-OUTCOMES.md).

The question type comes from the goal's wording (the cue table below, in the
design's row order) and the column roles of :mod:`avaloka.outcomes.semantics`.
The first cue-matching row whose roles are present wins; a goal with no cue
falls back on the roles alone. The result is an ordered list of
``(chart_kind, encoding)`` candidates; a model may pick an index, never invent
a kind. Each "Refused when" rule of §4.2 removes a candidate or the whole
question and says why, so a refusal is visible rather than a silent swap.

No model call, no randomness: the same table and goal give the same list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from avaloka.domains import normalise
from avaloka.outcomes import geo as _geo
from avaloka.outcomes.semantics import ColumnSemantics, by_role, funnel_candidates, stage_means_monotone

QUESTION_TYPES = ("trend", "composition", "distribution", "relationship", "funnel", "cohort",
                  "geo", "bridge", "flow", "ranking")

# §4.2 "Goal cues", in the table's row order: ties go to the first matching row.
GOAL_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trend", (r"over time", r"\btrends?\b", r"\bmonthly\b", r"\bweekly\b", r"\bgrowth\b", r"\bsince\b")),
    ("composition", (r"\bshare\b", r"\bmix\b", r"\bbreakdown\b", r"\bsplit\b", r"\bcontribution\b")),
    ("distribution", (r"\bdistribution\b", r"\bspread\b", r"\brange\b", r"\boutliers?\b")),
    ("relationship", (r"\bvs\.?\b", r"\bversus\b", r"\brelationship\b", r"\bcorrelat", r"\bdrivers?\b(?! of change)")),
    ("funnel", (r"\bfunnel\b", r"\bstages?\b", r"\bdrop-?off\b")),
    ("cohort", (r"\bcohorts?\b", r"\bvintages?\b", r"\bretention\b", r"months on book")),
    ("geo", (r"\bmap\b", r"\bregions?\b", r"\bcountry\b", r"\bcountries\b", r"\bstates?\b", r"\bwhere\b")),
    ("bridge", (r"\bwhy did\b", r"\bwhat changed\b", r"drivers of change", r"\byoy\b")),
    ("flow", (r"\bpaths?\b", r"\battribution\b", r"\bjourneys?\b", r"\bfrom\b.+\bto\b")),
    ("ranking", (r"\btop\b", r"\bbest\b", r"\bworst\b", r"\bleading\b")),
)
_CUE_RE = {q: [re.compile(c) for c in cues] for q, cues in GOAL_CUES}

# Limits named in §4.2.
TREND_MIN_PERIODS = 4
LINE_MAX_SERIES = 5
BAR_MAX_PERIODS = 12
PIE_MAX_PARTS = 5
DIST_MIN_ROWS = 20
BOX_MAX_GROUPS = 12
SCATTER_MAX_POINTS = 2000
COHORT_MIN = 3
SANKEY_MAX_NODES = 30
RANK_TOP = 10

_AGE_RE = re.compile(r"\b(mob|months? on book|age|tenure|months? since|period number|months?)\b")
_RATE_RE = re.compile(r"\b(rate|share|pct|percent|percentage|ratio|ctr|cvr|roas|apr|avg|average|mean)\b")


@dataclass(frozen=True)
class Candidate:
    chart_kind: str
    encoding: dict[str, Any]
    note: str = ""


@dataclass
class Recommendation:
    question_type: str | None
    candidates: list[Candidate] = field(default_factory=list)
    refused: list[dict[str, str]] = field(default_factory=list)   # {"chart_kind" | "question_type", "reason"}
    matched_cue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_rate(name: str, sem: ColumnSemantics) -> bool:
    """Rates and averages are not additive (#532 marketing sanity check)."""
    return sem.unit in ("fraction", "percent") or bool(_RATE_RE.search(normalise(name)))


def cue_matches(goal: str) -> list[tuple[str, str]]:
    text = (goal or "").lower()
    hits = []
    for q, pats in _CUE_RE.items():
        for p in pats:
            if p.search(text):
                hits.append((q, p.pattern))
                break
    return hits


def _roles_present(q: str, sem: Mapping[str, ColumnSemantics], df: pd.DataFrame) -> bool:
    has = lambda *r: bool(by_role(sem, *r))  # noqa: E731
    measures = by_role(sem, "measure", "target")
    if q == "trend":
        return has("time") and bool(measures)
    if q in ("composition", "ranking"):
        return has("dimension", "high_card_dimension") and bool(measures)
    if q == "distribution":
        return bool(measures)
    if q == "relationship":
        return len(measures) >= 2
    if q == "funnel":
        return len(funnel_candidates(sem)) >= 3 or _funnel_dimension(sem, df) is not None
    if q == "cohort":
        return has("time") and _age_column(sem) is not None and bool(measures)
    if q == "geo":
        return has("geo")
    if q == "bridge":
        return has("time") and bool(measures) and has("dimension")
    if q == "flow":
        return len(by_role(sem, "dimension")) >= 2
    return False


def question_type(goal: str, sem: Mapping[str, ColumnSemantics], df: pd.DataFrame) -> tuple[str | None, str | None]:
    """(question type, the cue that chose it). Roles decide when no cue does."""
    hits = cue_matches(goal)
    for q, cue in hits:
        if _roles_present(q, sem, df):
            return q, cue
    if hits:
        return hits[0]
    for q in ("trend", "geo", "ranking", "distribution"):
        if _roles_present(q, sem, df):
            return q, None
    return None, None


def _first_measure(sem, prefer_additive: bool = False) -> str | None:
    ms = by_role(sem, "target", "measure")
    if prefer_additive:
        add = [m for m in ms if not is_rate(m, sem[m])]
        if add:
            return add[0]
    return ms[0] if ms else None


def _small_dimension(sem, df, limit: int, exclude: Sequence[str] = ()) -> str | None:
    for d in by_role(sem, "dimension"):
        if d not in exclude and df[d].nunique() <= limit:
            return d
    return None


def _age_column(sem) -> str | None:
    for n, c in sem.items():
        if c.role in ("measure", "dimension") and _AGE_RE.search(normalise(n)):
            return n
    return None


def _funnel_dimension(sem, df) -> str | None:
    from avaloka.outcomes.semantics import _FUNNEL_WORDS
    for d in by_role(sem, "dimension"):
        vals = [normalise(v) for v in df[d].dropna().unique()]
        if len(vals) >= 3 and sum(any(w in v for w in _FUNNEL_WORDS) for v in vals) >= 0.8 * len(vals):
            return d
    return None


def _has_cycle(df: pd.DataFrame, src: str, dst: str) -> bool:
    edges: dict[str, set[str]] = {}
    for a, b in df[[src, dst]].dropna().drop_duplicates().itertuples(index=False):
        edges.setdefault(str(a), set()).add(str(b))
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        state[n] = 1
        for m in edges.get(n, ()):
            s = state.get(m, 0)
            if s == 1 or (s == 0 and visit(m)):
                return True
        state[n] = 2
        return False
    return any(state.get(n, 0) == 0 and visit(n) for n in list(edges))


def recommend(goal: str, df: pd.DataFrame, sem: Mapping[str, ColumnSemantics], *,
              question: str | None = None, k_floor: int = _geo.K_FLOOR,
              pii_columns: Sequence[str] = ()) -> Recommendation:
    q, cue = (question, None) if question else question_type(goal, sem, df)
    rec = Recommendation(question_type=q, matched_cue=cue)
    if q is None:
        rec.refused.append({"question_type": "none", "reason": "no column has a role any chart can use"})
        return rec
    if not _roles_present(q, sem, df):
        rec.refused.append({"question_type": q, "reason": f"{q} needs columns this table does not have"})
        return rec
    getattr(_Rules, q)(rec, df, sem, k_floor=k_floor, pii_columns=pii_columns)
    return rec


class _Rules:
    """One method per §4.2 row; each appends candidates in the row's order."""

    @staticmethod
    def trend(rec, df, sem, **_):
        t = by_role(sem, "time")[0]
        m = _first_measure(sem)
        periods = int(df[t].nunique())
        if periods < TREND_MIN_PERIODS:
            rec.refused.append({"question_type": "trend",
                                "reason": f"{periods} periods; a trend needs at least {TREND_MIN_PERIODS}"})
            return
        series = _small_dimension(sem, df, LINE_MAX_SERIES)
        enc = {"x": t, "y": m, "series": series}
        rec.candidates.append(Candidate("line", enc))
        if series and not is_rate(m, sem[m]) and (df[m].dropna() >= 0).all():
            rec.candidates.append(Candidate("area", {**enc, "stack": True}, "parts sum to the whole"))
        elif series:
            rec.refused.append({"chart_kind": "area", "reason": "a stacked area needs additive, non-negative parts"})
        if periods <= BAR_MAX_PERIODS:
            rec.candidates.append(Candidate("bar", {"x": t, "y": m, "series": series}))
        else:
            rec.refused.append({"chart_kind": "bar", "reason": f"{periods} periods; bars stop at {BAR_MAX_PERIODS}"})

    @staticmethod
    def composition(rec, df, sem, **_):
        dims = by_role(sem, "dimension", "high_card_dimension")
        m = _first_measure(sem, prefer_additive=True)
        if is_rate(m, sem[m]):
            rec.refused.append({"question_type": "composition",
                                "reason": f"{m} is a rate; rates are not additive, so it has no share of a whole"})
            return
        d = dims[0]
        rec.candidates.append(Candidate("bar", {"x": d, "y": m, "sort": "desc"}))
        d2 = _small_dimension(sem, df, LINE_MAX_SERIES, exclude=[d]) or (by_role(sem, "time") or [None])[0]
        if d2:
            rec.candidates.append(Candidate("stacked_bar", {"x": d, "y": m, "series": d2}))
        parts = int(df[d].nunique())
        if parts <= PIE_MAX_PARTS and (df[m].dropna() >= 0).all():
            rec.candidates.append(Candidate("pie", {"category": d, "value": m}))
        else:
            rec.refused.append({"chart_kind": "pie",
                                "reason": f"{parts} parts; a pie shows at most {PIE_MAX_PARTS} non-negative parts"})

    @staticmethod
    def distribution(rec, df, sem, **_):
        m = _first_measure(sem)
        n = int(df[m].notna().sum())
        if n < DIST_MIN_ROWS:
            rec.refused.append({"question_type": "distribution",
                                "reason": f"{n} rows; a distribution needs at least {DIST_MIN_ROWS}"})
            return
        rec.candidates.append(Candidate("histogram", {"x": m}))
        d = _small_dimension(sem, df, BOX_MAX_GROUPS)
        if d:
            rec.candidates.append(Candidate("boxplot", {"y": m, "by": d}))

    @staticmethod
    def relationship(rec, df, sem, **_):
        x, y = by_role(sem, "measure", "target")[:2]
        n = int((df[x].notna() & df[y].notna()).sum())
        rec.candidates.append(Candidate("scatter", {"x": x, "y": y, "sample": min(n, SCATTER_MAX_POINTS)},
                                        "sampled" if n > SCATTER_MAX_POINTS else ""))
        rec.candidates.append(Candidate("heatmap", {"x": x, "y": y, "binned": True}))

    @staticmethod
    def funnel(rec, df, sem, **_):
        cols = funnel_candidates(sem)
        if len(cols) >= 3:
            if not stage_means_monotone(df, cols):
                rec.refused.append({"question_type": "funnel",
                                    "reason": "stage means do not fall from one stage to the next"})
                return
            rec.candidates.append(Candidate("funnel", {"stages": cols}))
            rec.candidates.append(Candidate("bar", {"stages": cols, "step_conversion": True}))
            return
        d = _funnel_dimension(sem, df)
        m = _first_measure(sem, prefer_additive=True)
        rec.candidates.append(Candidate("funnel", {"stage": d, "value": m}))
        rec.candidates.append(Candidate("bar", {"stage": d, "value": m, "step_conversion": True}))

    @staticmethod
    def cohort(rec, df, sem, **_):
        t = by_role(sem, "time")[0]
        age = _age_column(sem)
        m = next(x for x in by_role(sem, "target", "measure") if x != age)
        cohorts = int(df[t].nunique())
        if cohorts < COHORT_MIN:
            rec.refused.append({"question_type": "cohort",
                                "reason": f"{cohorts} cohorts; a cohort view needs at least {COHORT_MIN}"})
            return
        enc = {"cohort": t, "age": age, "value": m}
        rec.candidates.append(Candidate("cohort_heatmap", enc))
        rec.candidates.append(Candidate("vintage_lines", enc))

    @staticmethod
    def geo(rec, df, sem, *, k_floor, pii_columns, **_):
        gcols = by_role(sem, "geo")
        m = _first_measure(sem, prefer_additive=True)
        kind = sem[gcols[0]].geo_kind
        if kind == "latlon":
            lat, lon = _geo.detect_latlon(df) or (gcols[0], gcols[1])
            pii = bool(set(pii_columns) & {lat, lon})
            pts = _geo.point_table(df, lat, lon, m, pii=pii, k=k_floor)
            if not pts.points:
                rec.refused.append({"question_type": "geo", "reason": f"every cell has fewer than {k_floor} rows"})
                return
            rec.candidates.append(Candidate("point_map", {"lat": lat, "lon": lon, "value": m, "binned": pii},
                                            "binned to geohash 5 for privacy" if pii else ""))
            return
        if kind not in ("country", "us_state"):
            rec.refused.append({"question_type": "geo", "reason": f"{kind} maps are not in this release"})
            return
        g = gcols[0]
        table = _geo.region_table(df, g, kind, m, k=k_floor)
        if not table.drawable:
            rec.refused.append({"question_type": "geo", "reason": f"every region has fewer than {k_floor} rows"})
            return
        rec.candidates.append(Candidate("choropleth", {"region": g, "value": m, "geo_kind": kind},
                                        f"{table.suppressed} regions suppressed" if table.suppressed else ""))

    @staticmethod
    def bridge(rec, df, sem, **_):
        t = by_role(sem, "time")[0]
        m = _first_measure(sem, prefer_additive=True)
        d = by_role(sem, "dimension")[0]
        rec.candidates.append(Candidate("waterfall", {"period": t, "value": m, "component": d}))

    @staticmethod
    def flow(rec, df, sem, **_):
        a, b = by_role(sem, "dimension")[:2]
        m = _first_measure(sem, prefer_additive=True)
        nodes = int(pd.concat([df[a], df[b]]).nunique())
        enc = {"source": a, "target": b, "value": m}
        if _has_cycle(df, a, b):
            rec.refused.append({"chart_kind": "sankey", "reason": "the paths contain a cycle; a sankey cannot draw one"})
        elif nodes > SANKEY_MAX_NODES:
            rec.refused.append({"chart_kind": "sankey", "reason": f"{nodes} nodes; a sankey stops at {SANKEY_MAX_NODES}"})
        else:
            rec.candidates.append(Candidate("sankey", enc))
        rec.candidates.append(Candidate("heatmap", {**enc, "matrix": True}))

    @staticmethod
    def ranking(rec, df, sem, **_):
        d = by_role(sem, "dimension", "high_card_dimension")[0]
        m = _first_measure(sem)
        rec.candidates.append(Candidate("bar", {"x": d, "y": m, "sort": "desc", "top": RANK_TOP, "other": True}))
