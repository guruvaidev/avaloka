"""``correlate``: which measures move together, and whether the overall
relationship holds inside the groups of a dimension.

Pearson and Spearman correlations with a 95 % interval (Fisher z) for every
pair of measures over distinct, complete rows; a within-group Pearson r per
level of a small dimension, so a pair whose overall sign differs from the
sign in every large group is named as a reversal (Simpson's paradox) and the
report is qualified. Identifier-like columns are left out of the default
measure set and said so in ``method``; constant columns are dropped the same
way.

``min_rows`` fails when too few rows hold every measure, ``constant`` when
fewer than two measures vary, and ``simpson`` qualifies as ``associational``
on a reversal.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import build_report, figure, merged_gate_words
from . import _shared as sh

VERSION = "1.0"
MIN_ROWS = 30                # fewer complete rows than this is a refusal
MAX_MEASURES = 8             # measures taken by default, and the heatmap's limit
GROUP_MIN = 30               # a group with fewer rows does not take part in the within-group check
SIMPSON_LEVELS = (2, 6)      # a dimension with this many distinct values is the default for the check
Z95 = 1.959963984540054
POINTS = 500                 # points drawn in the scatter

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_VALUES = "WITH d AS ({dedup}) SELECT {columns} FROM d WHERE {complete} ORDER BY {columns}"

EVIDENCE_CELL = """\
strongest = pairs.loc[pairs["pearson_r"].abs().idxmax()]
stat_keys = ("n", "pearson_r", "pearson_p", "ci_low", "ci_high", "spearman_r", "spearman_p")
evidence = {
    "rows": rows_leaves(counts), "n": leaf(len(values), "count"),
    "measures": {"count": leaf(len(kept), "count"), "constant": leaf(len(measures) - len(kept), "count"),
                 "excluded_ids": leaf(len(EXCLUDED_IDS), "count")},
    "pairs": {"count": leaf(len(pairs), "count"),
              "by": {f"p{i}": {k: leaf(r[k], "count" if k == "n" else "ratio") for k in stat_keys}
                     for i, (_, r) in enumerate(pairs.iterrows(), start=1)}},
    "strongest": {k: leaf(strongest[k], "count" if k == "n" else "ratio") for k in stat_keys},
}
if dimension:
    evidence["simpson"] = {"reversed_pairs": leaf(len(reversed_pairs), "count")}
    if reversed_pairs:
        a, b, overall, groups = max(reversed_pairs, key=lambda t: abs(t[2]))
        rs = [r for _, r in groups.values()]
        evidence["simpson"].update(r_overall=leaf(overall, "ratio"), r_within_min=leaf(min(rs), "ratio"),
                                   r_within_max=leaf(max(rs), "ratio"), groups=leaf(len(groups), "count"))
"""

SPEC = RecipeSpec(
    id="correlate", version=VERSION, title="Correlate measures",
    params=(ParamSpec("measures", "measure", many=True), ParamSpec("dimension", "dimension", required=False)),
    preconditions=("has_two_measures", "min_rows_30"),
    quality_gates=("min_rows", "constant", "simpson", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "min_rows": "Too few rows hold every measure.",
    "simpson": "Within each group of the dimension the relationship runs the other way from the overall "
                    "figure; the overall figure mixes differences between groups with the pattern inside them.",
})


@dataclass(frozen=True)
class CorrelateBound(sh.RecipeBound):
    excluded_ids: tuple[str, ...] = ()   # measure columns left out as identifiers, named in method


def _sql(bound: Bound) -> dict[str, str]:
    p = bound.params
    cols = list(p["measures"]) + ([p["dimension"]] if p.get("dimension") else [])
    return {"dedup": SQL_DEDUP, "columns": ", ".join(ident(c) for c in cols),
            "complete": " AND ".join(f"{ident(c)} IS NOT NULL" for c in p["measures"])}


def _as_bound(bound: Bound) -> CorrelateBound:
    return bound if isinstance(bound, CorrelateBound) else CorrelateBound(bound.recipe_id, bound.version, bound.params)


def _values(con: Any, bound: Bound) -> pd.DataFrame:
    return query(con, SQL_VALUES.format(**_sql(bound)))


def _varying(values: pd.DataFrame, measures: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(kept, constant)`` measures by sample standard deviation."""
    kept = [m for m in measures if len(values) > 1 and float(values[m].astype(float).std()) > 0.0]
    return kept, [m for m in measures if m not in kept]


def fisher_interval(r: float, n: int) -> tuple[float, float]:
    """95 % interval of a correlation by the Fisher z transform."""
    if n <= 3 or abs(r) >= 1.0:
        return r, r
    z, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    return math.tanh(z - Z95 * se), math.tanh(z + Z95 * se)


def _pair(x: pd.Series, y: pd.Series) -> dict[str, float]:
    from scipy import stats

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    r = float(pearson[0])
    lo, hi = fisher_interval(r, len(x))
    return {"n": float(len(x)), "pearson_r": r, "pearson_p": float(pearson[1]), "ci_low": lo, "ci_high": hi,
            "spearman_r": float(spearman[0]), "spearman_p": float(spearman[1])}


def within_groups(values: pd.DataFrame, a: str, b: str, dimension: str) -> dict[str, tuple[int, float]]:
    """Pearson r of ``a`` against ``b`` inside each level of ``dimension`` with at least ``GROUP_MIN`` rows."""
    out: dict[str, tuple[int, float]] = {}
    for level, g in values.dropna(subset=[dimension]).groupby(dimension, sort=True):
        if len(g) < GROUP_MIN or g[a].std() == 0 or g[b].std() == 0:
            continue
        out[str(level)] = (len(g), float(g[a].astype(float).corr(g[b].astype(float))))
    return out


def reverses(overall: float, groups: Mapping[str, tuple[int, float]]) -> bool:
    """The overall sign differs from the sign in every large group (at least two of them)."""
    if len(groups) < 2 or overall == 0.0:
        return False
    return all(r != 0.0 and (r > 0) != (overall > 0) for _, r in groups.values())


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class Correlate:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        named = intent.params.get("measures")
        named = (named,) if isinstance(named, str) else tuple(named or ())
        for m in named:
            if m not in sem or sem[m].role not in ("measure", "target"):
                return Refusal("no_measure", "A column named in the question is not a measure column.")
        excluded: tuple[str, ...] = ()
        measures = list(named)
        if len(measures) < 2:
            pool = sh.columns_of_role(sem, "measure", "target")
            excluded = tuple(m for m in pool if sh.is_id_like(m, sem) and m not in measures)
            for m in pool:
                if m not in measures and m not in excluded and len(measures) < MAX_MEASURES:
                    measures.append(m)
        if len(measures) < 2:
            return Refusal("no_measure", "This data has fewer than two measure columns to relate.")
        dimension = sh.bind_column(intent, sem, "dimension", ("dimension",), required=False,
                                   missing_reason="ambiguous_dimension", question="Which column groups the rows?")
        if isinstance(dimension, (Ask, Refusal)):
            return dimension
        if dimension is None:
            small = [n for n, c in sem.items() if c.role == "dimension" and SIMPSON_LEVELS[0] <= c.distinct <= SIMPSON_LEVELS[1]]
            dimension = small[0] if len(small) == 1 else None
        params = {"measures": tuple(measures), **({"dimension": dimension} if dimension else {})}
        return CorrelateBound("correlate", VERSION, params, source=sh.source_of(profile), excluded_ids=excluded)

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        values = _values(con, b)
        if len(values) < MIN_ROWS:
            return GateReport((Gate("min_rows", "failed", reason="insufficient_rows", value=len(values), unit="count"),))
        gates = [Gate("min_rows", "passed")]
        kept, _ = _varying(values, b.params["measures"])
        if len(kept) < 2:
            gates.append(Gate("constant", "failed", reason="constant_column", value=len(kept), unit="count"))
            return GateReport(tuple(gates))
        gates.append(Gate("constant", "passed"))
        reversed_min: float | None = None
        if b.params.get("dimension"):
            # The deciding figure is the evidence's /simpson/r_within_min: the lowest
            # within-group r of the reversed pair with the strongest overall r.
            reversed_pairs: list[tuple[float, float]] = []
            for a, c in combinations(kept, 2):
                overall = float(values[a].astype(float).corr(values[c].astype(float)))
                groups = within_groups(values, a, c, b.params["dimension"])
                if reverses(overall, groups):
                    reversed_pairs.append((overall, min(r for _, r in groups.values())))
            if reversed_pairs:
                reversed_min = max(reversed_pairs, key=lambda t: abs(t[0]))[1]
        gates.append(Gate("simpson", "qualified", caveat_kind="associational", value=reversed_min, unit="ratio")
                     if reversed_min is not None else Gate("simpson", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        values = _values(con, b)
        kept, constant = _varying(values, b.params["measures"])
        dimension = b.params.get("dimension")
        pairs: list[tuple[str, str, dict[str, float]]] = []
        for a, c in combinations(kept, 2):
            pairs.append((a, c, _pair(values[a].astype(float), values[c].astype(float))))
        strongest = max(pairs, key=lambda t: abs(t[2]["pearson_r"]))
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "n": sh.leaf(len(values), "count"),
            "measures": {"count": sh.leaf(len(kept), "count"), "constant": sh.leaf(len(constant), "count"),
                         "excluded_ids": sh.leaf(len(b.excluded_ids), "count")},
            "pairs": {"count": sh.leaf(len(pairs), "count"),
                      "by": {f"p{i}": {k: sh.leaf(v, "count" if k == "n" else "ratio") for k, v in stats.items()}
                             for i, (_, _, stats) in enumerate(pairs, start=1)}},
            "strongest": {k: sh.leaf(v, "count" if k == "n" else "ratio") for k, v in strongest[2].items()},
        }
        tables = {"pairs": sh.table(["measure_a", "measure_b", "n", "pearson_r", "ci_low", "ci_high", "pearson_p", "spearman_r", "spearman_p"],
                                    [[a, c, int(s["n"]), s["pearson_r"], s["ci_low"], s["ci_high"], s["pearson_p"],
                                      s["spearman_r"], s["spearman_p"]] for a, c, s in pairs]),
                  "points": sh.table([strongest[0], strongest[1]],
                                     values[[strongest[0], strongest[1]]].head(POINTS).astype(float).values.tolist())}
        if dimension:
            within_rows: list[list[Any]] = []
            reversed_pairs: list[tuple[str, str, float, dict[str, tuple[int, float]]]] = []
            for a, c, stats in pairs:
                groups = within_groups(values, a, c, dimension)
                within_rows += [[a, c, level, n, r] for level, (n, r) in groups.items()]
                if reverses(stats["pearson_r"], groups):
                    reversed_pairs.append((a, c, stats["pearson_r"], groups))
            tables["within"] = sh.table(["measure_a", "measure_b", "group", "n", "pearson_r"], within_rows)
            tree["simpson"] = {"reversed_pairs": sh.leaf(len(reversed_pairs), "count")}
            if reversed_pairs:
                a, c, overall, groups = max(reversed_pairs, key=lambda t: abs(t[2]))
                rs = [r for _, r in groups.values()]
                tree["simpson"].update({"r_overall": sh.leaf(overall, "ratio"), "r_within_min": sh.leaf(min(rs), "ratio"),
                                        "r_within_max": sh.leaf(max(rs), "ratio"), "groups": sh.leaf(len(groups), "count")})
                tables["reversal"] = sh.table(["measure_a", "measure_b", "group", "n", "pearson_r"],
                                              [[a, c, "overall", len(values), overall]] + [[a, c, g, n, r] for g, (n, r) in groups.items()])
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev = evidence.tree
        figs = [figure("f1", ev, "/n", unit="count", decimals=0),
                figure("f2", ev, "/strongest/pearson_r", unit="ratio", decimals=2),
                figure("f3", ev, "/strongest/ci_low", unit="ratio", decimals=2),
                figure("f4", ev, "/strongest/ci_high", unit="ratio", decimals=2),
                figure("f5", ev, "/strongest/spearman_r", unit="ratio", decimals=2),
                figure("f6", ev, "/pairs/count", unit="count", decimals=0),
                figure("f7", ev, "/rows/duplicates", unit="count", decimals=0),
                figure("f12", ev, "/measures/count", unit="count", decimals=0),
                figure("f13", ev, "/measures/excluded_ids", unit="count", decimals=0)]
        headline = ("Across {f1} rows the strongest of {f6} pairs has a correlation of {f2}, "
                    "with a plausible range of {f3} to {f4}.")
        bullets = [{"so_what": "The rank correlation of the strongest pair is {f5}; a gap from the straight-line figure "
                               "points to a curved or outlier-led relationship.",
                    "now_what": "Look at the scatter before reading the figure as a straight-line link."}]
        if "r_overall" in ev.get("simpson", {}):
            figs += [figure("f8", ev, "/simpson/r_overall", unit="ratio", decimals=2),
                     figure("f9", ev, "/simpson/r_within_min", unit="ratio", decimals=2),
                     figure("f10", ev, "/simpson/r_within_max", unit="ratio", decimals=2),
                     figure("f11", ev, "/simpson/groups", unit="count", decimals=0)]
            headline = ("The relationship between the measures reverses within the groups of the dimension: "
                        "{f8} overall, but between {f9} and {f10} inside every group.")
            bullets.insert(0, {"so_what": "In each of the {f11} groups the pair moves the other way from the overall figure, "
                                          "which reflects differences between the groups.",
                               "now_what": "Read the within-group figures; compare like with like."})
        if ev["measures"]["excluded_ids"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f12} measures were related; {f13} numeric columns were left out as identifiers.",
                            "now_what": "Name a left-out column in the question if it is a quantity after all."})
        if ev["rows"]["duplicates"]["value"] > 0 and len(bullets) < 3:
            bullets.append({"so_what": "{f7} rows are exact duplicates and were counted once.",
                            "now_what": "Remove them at the source so totals elsewhere are not inflated."})
        kpis = [{"id": "strongest", "label": "Strongest correlation", "value": "f2", "direction_good": "neutral"},
                {"id": "rows", "label": "Complete rows", "value": "f1", "direction_good": "neutral"}]
        a, c = evidence.tables["points"]["columns"]
        pairs = evidence.tables["pairs"]["rows"]
        labels = list(dict.fromkeys([r[0] for r in pairs] + [r[1] for r in pairs]))
        matrix = [[1.0 if i == j else 0.0 for j in labels] for i in labels]
        for r in pairs:
            i, j = labels.index(r[0]), labels.index(r[1])
            matrix[i][j] = matrix[j][i] = r[3]
        charts = [sh.scatter_chart("strongest", f"{sh.label(c)} against {sh.label(a)}", evidence.tables["points"]["rows"],
                                   x_name=sh.label(a), y_name=sh.label(c),
                                   alt_text=f"Scatter plot of {sh.label(c)} against {sh.label(a)}, the most correlated pair; "
                                            "the correlation is given in the headline."),
                  sh.heatmap_chart("matrix", "Correlation matrix", labels, matrix, value_name="pearson_r",
                                   alt_text=f"Heatmap of Pearson correlations among {len(labels)} measures; "
                                            "the data table lists each pair.")]
        rows = int(ev["rows"]["raw"]["value"])
        measures = ", ".join(b.params["measures"])
        notes = []
        if b.excluded_ids:
            notes.append("excluded as identifiers: " + ", ".join(b.excluded_ids))
        if ev["measures"]["constant"]["value"]:
            kept = labels
            notes.append("dropped as constant: " + ", ".join(m for m in b.params["measures"] if m not in kept))
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates, headline_template=headline,
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Pearson and Spearman correlation (scipy.stats) for every pair of {measures} over distinct rows "
                           f"holding every measure; the interval by the Fisher z transform"
                           + (f"; Pearson r inside each {b.params['dimension']} group of at least {GROUP_MIN} rows "
                              "to check the overall sign" if b.params.get("dimension") else "")
                           + ("; " + "; ".join(notes) if notes else "") + ".",
            rows_analyzed=rows, total_rows=rows, gate_words=GATE_WORDS,
            gate_refs={"simpson": "/simpson/r_within_min", "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        p = b.params
        extra = {"GROUP_MIN": GROUP_MIN, "EXCLUDED_IDS": tuple(b.excluded_ids),
                 **({} if p.get("dimension") else {"dimension": None})}
        cells = [
            sh.header_cell("correlate", VERSION, SPEC.title),
            sh.parameters_cell(b, extra),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# every statistic below is computed over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
            sh.sql_cell("SQL_VALUES", SQL_VALUES,
                        "columns = ', '.join(ident(c) for c in list(measures) + ([dimension] if dimension else []))\n"
                        "complete = ' AND '.join(f'{ident(c)} IS NOT NULL' for c in measures)\n"
                        "values = con.sql(SQL_VALUES.format(dedup=SQL_DEDUP, columns=columns, complete=complete)).df()"),
            _nb.code("from itertools import combinations\nimport math\nfrom typing import Mapping\nfrom scipy import stats\n\n"
                     f"Z95 = {Z95!r}\n\n" + inspect.getsource(fisher_interval) + "\n"
                     "kept = [m for m in measures if values[m].astype(float).std() > 0]\n"
                     "rows = []\n"
                     "for a, b in combinations(kept, 2):\n"
                     "    x, y = values[a].astype(float), values[b].astype(float)\n"
                     "    r, p = stats.pearsonr(x, y)\n"
                     "    rho, p_rho = stats.spearmanr(x, y)\n"
                     "    lo, hi = fisher_interval(float(r), len(x))\n"
                     "    rows.append(dict(a=a, b=b, n=len(x), pearson_r=float(r), ci_low=lo, ci_high=hi, pearson_p=float(p),\n"
                     "                     spearman_r=float(rho), spearman_p=float(p_rho)))\n"
                     "pairs = pd.DataFrame(rows)\npairs"),
        ]
        if p.get("dimension"):
            cells.append(_nb.code(inspect.getsource(within_groups) + "\n\n" + inspect.getsource(reverses) + "\n"
                                  "within, reversed_pairs = [], []\n"
                                  "for _, pair in pairs.iterrows():\n"
                                  "    groups = within_groups(values, pair['a'], pair['b'], dimension)\n"
                                  "    within += [dict(a=pair['a'], b=pair['b'], group=level, n=n, pearson_r=r, overall=pair['pearson_r'])\n"
                                  "               for level, (n, r) in groups.items()]\n"
                                  "    if reverses(pair['pearson_r'], groups):\n"
                                  "        reversed_pairs.append((pair['a'], pair['b'], pair['pearson_r'], groups))\n"
                                  "within = pd.DataFrame(within)\n"
                                  "# a pair reverses when every large group's sign differs from the overall sign\n"
                                  "within.assign(reversed=(within['pearson_r'] > 0) != (within['overall'] > 0)) if len(within) else within"))
        cells += [
            _nb.code("import matplotlib.pyplot as plt\n"
                     "top = pairs.iloc[pairs['pearson_r'].abs().idxmax()]\n"
                     "values.plot.scatter(x=top['a'], y=top['b'], title=f\"{top['b']} against {top['a']}\")\n"
                     "plt.matshow(values[kept].astype(float).corr()); plt.colorbar()"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
