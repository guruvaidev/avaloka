"""``ab_test``: did a treatment arm differ from the control arm of an experiment?

Per-arm count, mean and variance are DuckDB SQL; the tests are scipy and
statsmodels over the rows of the two arms. A binary outcome (every value is
zero or one) gets the two-proportion z-test, a Wilson interval per arm and
an interval on the difference; any other outcome gets Welch's t-test with
the interval on the mean difference. Each treatment arm is compared with
the control arm (the arm called control, baseline or a; else the first
alphabetically); the headline is the comparison with the largest
difference. Power at the observed sample size for the observed difference
and the smallest difference detectable at the target power come from
``statsmodels.stats.power``.

This is the one recipe whose wording may say "effect", and only because the
sample-ratio check passed: ``sample_ratio`` fails (``sample_ratio_mismatch``)
when the arm counts do not fit an equal split, and no report is produced.
``min_rows`` fails when an arm is small; ``power`` qualifies as ``small_n``
when the headline comparison is under-powered. Rows are taken as they are:
without a unit id the test cannot tell repeated units from identical
observations, so ``duplicates`` reports exact duplicates and keeps them.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from avaloka import notebook as _nb
from avaloka.outcomes.semantics import ColumnSemantics

from ..base import Ask, Bound, Evidence, Gate, GateReport, Intent, ParamSpec, RecipeSpec, Refusal
from ..engine import ident, query
from ..report import build_report, figure, merged_gate_words
from . import _shared as sh

VERSION = "1.0"
MIN_ARM_N = 30               # an arm with fewer rows is a refusal
MIN_ARMS, MAX_ARMS = 2, 4    # arms the recipe compares
SRM_P = 0.001                # a smaller chi-square p-value on the arm counts is a sample-ratio mismatch
ALPHA = 0.05                 # two-sided test level; the intervals are at one minus this
POWER_TARGET = 0.8           # power below this is a small_n caveat; the detectable difference is at this power
EQUAL_VAR = False            # Welch: the arms' variances are not assumed equal
POWER_CURVE_POINTS = 12      # rows per arm at which the power curve is evaluated
#: Arm values that name the control (the vocabulary ``specialist_plans`` uses).
CONTROL_LIKE = re.compile(r"^(control|baseline|a)$", re.IGNORECASE)
#: Column names that describe an assignment to an arm (the same vocabulary).
ARM_NAME = re.compile(r"(variant|arm|treatment|group|bucket|experiment|condition|assignment)", re.IGNORECASE)
NO_ARM_WORDS = "No two-valued column names the arms; name the assignment column."

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_ARMS = ("SELECT CAST({arm} AS VARCHAR) AS arm, count({outcome}) AS n, avg({outcome}) AS mean, "
            "var_samp({outcome}) AS variance, sum({outcome}) AS total, "
            "count(*) FILTER (WHERE {outcome} IN (0, 1)) AS n01 FROM t "
            "WHERE {arm} IS NOT NULL AND {outcome} IS NOT NULL GROUP BY 1 ORDER BY 1")
SQL_VALUES = ("SELECT CAST({arm} AS VARCHAR) AS arm, {outcome} AS v FROM t "
              "WHERE {arm} IS NOT NULL AND {outcome} IS NOT NULL ORDER BY 1, 2")

EVIDENCE_CELL = """\
mean_unit, _, diff_unit, scale = outcome_units(binary, OUTCOME_UNIT)
lead = max(comparisons, key=lambda t: abs(comparisons[t]["difference"]))

def comparison_leaves(treatment):
    c, rt, rc = comparisons[treatment], by[treatment], by[control]
    mean_c = float(rc["mean"])
    leaves = {"treatment_n": leaf(rt["n"], "count"), "control_n": leaf(rc["n"], "count"),
              "treatment_mean": leaf(rt["mean"], mean_unit), "control_mean": leaf(mean_c, mean_unit),
              "difference": leaf(c["difference"] * scale, diff_unit),
              "ci_low": leaf(c["ci_low"] * scale, diff_unit), "ci_high": leaf(c["ci_high"] * scale, diff_unit),
              "se": leaf(c["se"] * scale, diff_unit), "p": leaf(c["p"], "unitless"),
              "power": leaf(c["power"], "fraction"), "mde": leaf(c["mde"] * scale, diff_unit)}
    if mean_c:
        leaves["relative"] = leaf(c["difference"] / abs(mean_c), "fraction")
    return leaves

evidence = {
    "rows": rows_leaves(counts),
    "arms": {"count": leaf(len(arms), "count"), "smallest_n": leaf(arms["n"].min(), "count"),
             "srm_p": leaf(srm_p, "unitless")},
    "comparisons": {"count": leaf(len(comparisons), "count"),
                    "by": {f"c{i}": comparison_leaves(t) for i, t in enumerate(comparisons, start=1)}},
    "lead": comparison_leaves(lead),
}
"""

SPEC = RecipeSpec(
    id="ab_test", version=VERSION, title="Compare the arms of an experiment",
    params=(ParamSpec("arm", "dimension"), ParamSpec("outcome", "measure")),
    preconditions=("has_arm", "has_measure"),
    quality_gates=("min_rows", "sample_ratio", "power", "duplicates"),
    evidence_kind="experimental", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "min_rows": "An arm holds too few rows for the test.",
    "duplicates": "Some rows are exact duplicates; without a unit id the test cannot tell repeated units from "
                  "identical observations, so every row was kept.",
})


@dataclass(frozen=True)
class ABBound(sh.RecipeBound):
    unit: str = "unitless"       # the outcome's semantic unit, fixed at bind time


# ── the tests (pure scipy/statsmodels; the notebook embeds this source) ────

def ci_multiplier(df: float | None) -> float:
    """The two-sided quantile an interval half-width is built from: normal, or Student's t at ``df``."""
    from scipy import stats
    return float(stats.norm.ppf(1 - ALPHA / 2) if df is None else stats.t.ppf(1 - ALPHA / 2, df))


def wilson(x: int, n: int) -> tuple[float, float]:
    """The Wilson score interval of a proportion ``x / n``."""
    z = ci_multiplier(None)
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def compare_binary(x_t: int, n_t: int, x_c: int, n_c: int) -> dict[str, float]:
    """Two-proportion z-test (pooled) with an unpooled interval on the
    difference, power at the observed counts and the detectable difference."""
    from scipy import stats
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    p_t, p_c = x_t / n_t, x_c / n_c
    pooled = (x_t + x_c) / (n_t + n_c)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se_pooled if se_pooled else 0.0
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    half = ci_multiplier(None) * se
    power_model = NormalIndPower()
    power = float(power_model.power(effect_size=abs(proportion_effectsize(p_t, p_c)), nobs1=n_t, alpha=ALPHA, ratio=n_c / n_t))
    h = float(power_model.solve_power(effect_size=None, nobs1=n_t, alpha=ALPHA, power=POWER_TARGET, ratio=n_c / n_t))
    detectable = min(1.0, math.sin(math.asin(math.sqrt(p_c)) + h / 2) ** 2) - p_c
    return {"difference": p_t - p_c, "se": se, "ci_low": p_t - p_c - half, "ci_high": p_t - p_c + half,
            "p": float(2 * stats.norm.sf(abs(z))), "power": power, "mde": detectable, "df": float("nan")}


def compare_continuous(t: Sequence[float], c: Sequence[float]) -> dict[str, float]:
    """Welch's t-test (pooled when EQUAL_VAR) with the interval on the mean
    difference, power at the observed rows and the detectable difference."""
    from scipy import stats
    from statsmodels.stats.power import TTestIndPower

    t_arr, c_arr = pd.Series(t, dtype=float), pd.Series(c, dtype=float)
    n_t, n_c = len(t_arr), len(c_arr)
    v_t, v_c = float(t_arr.var(ddof=1)), float(c_arr.var(ddof=1))
    result = stats.ttest_ind(t_arr, c_arr, equal_var=EQUAL_VAR)
    if EQUAL_VAR:
        sp2 = ((n_t - 1) * v_t + (n_c - 1) * v_c) / (n_t + n_c - 2)
        se, df = math.sqrt(sp2 * (1 / n_t + 1 / n_c)), float(n_t + n_c - 2)
    else:
        a, b = v_t / n_t, v_c / n_c
        se = math.sqrt(a + b)
        df = (a + b) ** 2 / (a * a / (n_t - 1) + b * b / (n_c - 1)) if a + b else float(n_t + n_c - 2)
    diff = float(t_arr.mean() - c_arr.mean())
    half = ci_multiplier(df) * se
    sd_pooled = math.sqrt(((n_t - 1) * v_t + (n_c - 1) * v_c) / (n_t + n_c - 2))
    power_model = TTestIndPower()
    power = float(power_model.power(effect_size=abs(diff) / sd_pooled if sd_pooled else 0.0, nobs1=n_t, alpha=ALPHA, ratio=n_c / n_t))
    d = float(power_model.solve_power(effect_size=None, nobs1=n_t, alpha=ALPHA, power=POWER_TARGET, ratio=n_c / n_t))
    return {"difference": diff, "se": se, "ci_low": diff - half, "ci_high": diff + half,
            "p": float(result.pvalue), "power": power, "mde": d * sd_pooled, "df": df}


def power_curve(binary: bool, p_c: float, sd: float, difference: float, ratio: float, n_max: int) -> list[tuple[int, float]]:
    """Power at the observed difference for a range of rows in the treatment arm."""
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    if binary:
        effect = abs(proportion_effectsize(min(1.0, max(0.0, p_c + difference)), p_c))
        model: Any = NormalIndPower()
    else:
        effect = abs(difference) / sd if sd else 0.0
        model = TTestIndPower()
    step = max(1, n_max // POWER_CURVE_POINTS)
    out = []
    for n in range(step, n_max + 1, step):
        out.append((n, float(model.power(effect_size=effect, nobs1=n, alpha=ALPHA, ratio=ratio))))
    return out


def control_arm(arms: Sequence[str]) -> str:
    ordered = sorted(arms, key=str)
    return next((a for a in ordered if CONTROL_LIKE.match(a)), ordered[0])


# ── plumbing ──────────────────────────────────────────────────────────────

def _sql(bound: Bound) -> dict[str, str]:
    p = bound.params
    return {"dedup": SQL_DEDUP, "arm": ident(p["arm"]), "outcome": ident(p["outcome"])}


def _as_bound(bound: Bound) -> ABBound:
    return bound if isinstance(bound, ABBound) else ABBound(bound.recipe_id, bound.version, bound.params)


def _arms(con: Any, bound: Bound) -> pd.DataFrame:
    return query(con, SQL_ARMS.format(**_sql(bound)))


def _is_binary(arms: pd.DataFrame) -> bool:
    return int(arms["n"].sum()) == int(arms["n01"].sum())


def outcome_units(binary: bool, unit: str) -> tuple[str, str, str, float]:
    """``(mean evidence unit, mean figure unit, difference unit, difference scale)``."""
    if binary or unit == "fraction":
        return "fraction", "percent", "pp", 100.0
    if unit == "percent":
        return "percent", "percent", "pp", 1.0
    if unit == "currency":
        return "currency", "currency", "currency", 1.0
    return "unitless", "unitless", "unitless", 1.0


def _comparisons(con: Any, bound: Bound, arms: pd.DataFrame) -> tuple[str, bool, list[dict[str, Any]]]:
    """``(control, binary, comparisons)``: every treatment arm against the control."""
    binary = _is_binary(arms)
    labels = [str(a) for a in arms["arm"]]
    control = control_arm(labels)
    by = {str(r["arm"]): r for _, r in arms.iterrows()}
    values = query(con, SQL_VALUES.format(**_sql(bound))) if not binary else None
    out = []
    for treatment in labels:
        if treatment == control:
            continue
        rt, rc = by[treatment], by[control]
        if binary:
            stats = compare_binary(int(rt["total"]), int(rt["n"]), int(rc["total"]), int(rc["n"]))
        else:
            stats = compare_continuous(values.loc[values["arm"] == treatment, "v"].astype(float).tolist(),
                                       values.loc[values["arm"] == control, "v"].astype(float).tolist())
        mean_c = float(rc["mean"])
        out.append({"treatment": treatment, "control": control, "n_t": int(rt["n"]), "n_c": int(rc["n"]),
                    "mean_t": float(rt["mean"]), "mean_c": mean_c,
                    "relative": stats["difference"] / abs(mean_c) if mean_c else None, **stats})
    return control, binary, out


def _lead(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return max(comparisons, key=lambda c: abs(c["difference"]))


def _srm_p(arms: pd.DataFrame) -> float:
    from scipy import stats
    return float(stats.chisquare(arms["n"].astype(float).to_numpy()).pvalue)


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class ABTest:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        arm = intent.params.get("arm")
        if isinstance(arm, str):
            if arm not in sem or sem[arm].role != "dimension":
                return Refusal("unsupported_question", NO_ARM_WORDS)
        else:
            candidates = sh.columns_of_role(sem, "dimension")
            named = [c for c in candidates if ARM_NAME.search(c)]
            candidates = named or candidates
            if not candidates:
                return Refusal("unsupported_question", NO_ARM_WORDS)
            if len(candidates) > 1:
                return Ask("arm", "Which column holds the arm each row was assigned to?", tuple(candidates[:8]))
            arm = candidates[0]
        if sem[arm].distinct < MIN_ARMS:
            return Refusal("unsupported_question", "The assignment column holds a single arm, so there is nothing to compare.")
        if sem[arm].distinct > MAX_ARMS:
            return Refusal("unsupported_question", "The assignment column holds more arms than the recipe compares; "
                                                   "name a column with two to four arms.")
        outcome = sh.bind_column(intent, sem, "outcome", ("measure", "target"), required=True, missing_reason="no_measure",
                                 question="Which column holds the outcome of the test?")
        if not isinstance(outcome, str):
            return outcome
        return ABBound("ab_test", VERSION, {"arm": arm, "outcome": outcome},
                       source=sh.source_of(profile), unit=sem[outcome].unit or "unitless")

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        arms = _arms(con, b)
        smallest = int(arms["n"].min()) if len(arms) >= MIN_ARMS else 0
        if smallest < MIN_ARM_N:
            return GateReport((Gate("min_rows", "failed", reason="insufficient_rows", value=smallest, unit="count"),))
        gates = [Gate("min_rows", "passed")]
        srm_p = _srm_p(arms)
        if srm_p < SRM_P:
            gates.append(Gate("sample_ratio", "failed", reason="sample_ratio_mismatch", value=srm_p, unit="unitless"))
            return GateReport(tuple(gates))
        gates.append(Gate("sample_ratio", "passed"))
        _, _, comparisons = _comparisons(con, b, arms)
        power = _lead(comparisons)["power"]
        gates.append(Gate("power", "qualified", caveat_kind="small_n", value=power, unit="fraction")
                     if power < POWER_TARGET else Gate("power", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        arms = _arms(con, b)
        control, binary, comparisons = _comparisons(con, b, arms)
        lead = _lead(comparisons)
        mean_unit, _, diff_unit, scale = outcome_units(binary, b.unit)
        arm_rows = []
        for _, r in arms.iterrows():
            n, mean = int(r["n"]), float(r["mean"])
            if binary:
                low, high = wilson(int(r["total"]), n)
            else:
                half = ci_multiplier(n - 1) * math.sqrt(_num(r["variance"]) / n)
                low, high = mean - half, mean + half
            arm_rows.append([str(r["arm"]), n, mean, low, high])

        def comparison_leaves(c: dict[str, Any]) -> dict[str, Any]:
            leaves = {"treatment_n": sh.leaf(c["n_t"], "count"), "control_n": sh.leaf(c["n_c"], "count"),
                      "treatment_mean": sh.leaf(c["mean_t"], mean_unit), "control_mean": sh.leaf(c["mean_c"], mean_unit),
                      "difference": sh.leaf(c["difference"] * scale, diff_unit),
                      "ci_low": sh.leaf(c["ci_low"] * scale, diff_unit), "ci_high": sh.leaf(c["ci_high"] * scale, diff_unit),
                      "se": sh.leaf(c["se"] * scale, diff_unit), "p": sh.leaf(c["p"], "unitless"),
                      "power": sh.leaf(c["power"], "fraction"), "mde": sh.leaf(c["mde"] * scale, diff_unit)}
            if c["relative"] is not None:
                leaves["relative"] = sh.leaf(c["relative"], "fraction")
            return leaves

        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "arms": {"count": sh.leaf(len(arms), "count"), "smallest_n": sh.leaf(int(arms["n"].min()), "count"),
                     "srm_p": sh.leaf(_srm_p(arms), "unitless"),
                     "by": {row[0]: {"n": sh.leaf(row[1], "count"), "mean": sh.leaf(row[2], mean_unit),
                                     "ci_low": sh.leaf(row[3], mean_unit), "ci_high": sh.leaf(row[4], mean_unit)}
                            for row in arm_rows}},
            "comparisons": {"count": sh.leaf(len(comparisons), "count"),
                            "by": {f"c{i}": comparison_leaves(c) for i, c in enumerate(comparisons, start=1)}},
            "lead": comparison_leaves(lead),
        }
        sd_c = math.sqrt(_num(arms.loc[arms["arm"] == control, "variance"].iloc[0]))
        curve = power_curve(binary, lead["mean_c"], sd_c, lead["difference"], lead["n_c"] / lead["n_t"], 2 * lead["n_t"])
        tables = {"arms": sh.table(["arm", "n", "mean", "ci_low", "ci_high"], arm_rows),
                  "comparisons": sh.table(["treatment", "control", "difference", "ci_low", "ci_high", "p", "power", "mde"],
                                          [[c["treatment"], c["control"], c["difference"] * scale, c["ci_low"] * scale,
                                            c["ci_high"] * scale, c["p"], c["power"], c["mde"] * scale] for c in comparisons]),
                  "power_curve": sh.table(["treatment_rows", "power"], [[n, pw] for n, pw in curve]),
                  "design": sh.table(["binary", "control"], [[int(binary), control]])}
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        binary = bool(evidence.tables["design"]["rows"][0][0])
        _, mean_fig, diff_unit, _ = outcome_units(binary, b.unit)
        d_mean = 1 if mean_fig == "percent" else 2
        d_diff = 1 if diff_unit == "pp" else 2
        figs = [figure("f1", ev, "/arms/smallest_n", unit="count", decimals=0),
                figure("f2", ev, "/lead/treatment_mean", unit=mean_fig, decimals=d_mean),
                figure("f3", ev, "/lead/control_mean", unit=mean_fig, decimals=d_mean),
                figure("f4", ev, "/lead/difference", unit=diff_unit, decimals=d_diff),
                figure("f6", ev, "/lead/ci_low", unit=diff_unit, decimals=d_diff),
                figure("f7", ev, "/lead/ci_high", unit=diff_unit, decimals=d_diff),
                figure("f8", ev, "/lead/p", unit="unitless", decimals=3),
                figure("f9", ev, "/lead/power", unit="percent", decimals=0),
                figure("f10", ev, "/lead/mde", unit=diff_unit, decimals=d_diff),
                figure("f11", ev, "/arms/srm_p", unit="unitless", decimals=3),
                figure("f12", ev, "/comparisons/count", unit="count", decimals=0),
                figure("f13", ev, "/rows/duplicates", unit="count", decimals=0)]
        lead = ev["lead"]
        excludes_zero = lead["ci_low"]["value"] > 0 or lead["ci_high"]["value"] < 0
        relative = "relative" in lead
        if relative:
            figs.append(figure("f5", ev, "/lead/relative", unit="percent", decimals=1))
        if excludes_zero:
            headline = ("The treatment arm's outcome is {f2} against {f3} in the control arm: an effect of {f4}"
                        + (" ({f5})" if relative else "") + ", with an interval of {f6} to {f7} that excludes zero.")
        else:
            headline = ("The treatment arm's outcome is {f2} against {f3} in the control arm; a difference of {f4} "
                        "({f6} to {f7}) not distinguishable from zero at this sample size.")
        under_powered = lead["power"]["value"] < POWER_TARGET
        bullets = [{"so_what": "The p-value is {f8}; power at the observed difference is {f9}; the smallest difference "
                               "detectable at this sample size is {f10}.",
                    "now_what": ("Read the interval, not the point estimate; a larger sample is needed to settle a difference "
                                 "of this size." if under_powered else
                                 "Weigh the size of the effect against what acting on it costs.")},
                   {"so_what": "The smallest arm holds {f1} rows, and the arm counts fit an equal split (sample-ratio "
                               "test p-value {f11}).",
                    "now_what": "Confirm that rows were assigned to arms at random before reading this as a test."}]
        if ev["comparisons"]["count"]["value"] > 1:
            bullets.append({"so_what": "{f12} treatment arms were each compared with the control arm; the headline is "
                                       "the largest difference.",
                            "now_what": "Read the other comparisons in the table; several tests on one dataset raise "
                                        "the chance of a false finding."})
        elif ev["rows"]["duplicates"]["value"] > 0:
            bullets.append({"so_what": "{f13} rows are exact duplicates and were kept, since no unit id separates "
                                       "repeated units from identical observations.",
                            "now_what": "Add a unit id and re-run if the same unit can appear more than once."})
        kpis = [{"id": "treatment", "label": "Treatment arm", "value": "f2", "baseline": "f3",
                 "baseline_label": "control arm", "direction_good": "neutral",
                 **({"delta": "f5", "delta_kind": "pct"} if relative else {})},
                {"id": "difference", "label": "Difference", "value": "f4", "direction_good": "neutral"},
                {"id": "power", "label": "Power at observed difference", "value": "f9", "direction_good": "up"}]
        arms = evidence.tables["arms"]["rows"]
        curve = evidence.tables["power_curve"]["rows"]
        label_o, label_a = sh.label(p["outcome"]), sh.label(p["arm"])
        charts = [sh.hbar_interval_chart("arms", f"{label_o} by {label_a}", [r[0] for r in arms], [r[2] for r in arms],
                                         [r[3] for r in arms], [r[4] for r in arms], x_name=label_o, y_name=label_a,
                                         alt_text=f"Horizontal bars of mean {label_o} per {label_a} with each arm's "
                                                  "interval; the difference is given in the headline."),
                  sh.line_chart("power", "Power by rows in the treatment arm", [str(r[0]) for r in curve],
                                {"power": [r[1] for r in curve]}, x_name="rows in the treatment arm", y_name="power",
                                alt_text=f"Line of the test's power at the observed difference over {len(curve)} sample "
                                         "sizes; the power at the observed rows is given in the first bullet.")]
        n_rows = int(ev["rows"]["raw"]["value"])
        test = ("two-proportion z-test with Wilson intervals per arm and an unpooled interval on the difference"
                if binary else "Welch's t-test with the interval on the mean difference")
        control = str(evidence.tables["design"]["rows"][0][1])
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates, headline_template=headline,
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Each arm of {p['arm']} against the control arm ({control}) on {p['outcome']}: {test} "
                           "(scipy); chi-square test of the arm counts against an equal split; power at the observed "
                           "rows for the observed difference and the smallest difference detectable at the target "
                           "power (statsmodels). Rows are taken as they are.",
            rows_analyzed=n_rows, total_rows=n_rows, gate_words=GATE_WORDS,
            gate_refs={"power": "/lead/power", "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        cells = [
            sh.header_cell("ab_test", VERSION, SPEC.title),
            sh.parameters_cell(b, {"OUTCOME_UNIT": b.unit}),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# duplicates are counted, not removed: rows are taken as they are"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
            sh.sql_cell("SQL_ARMS", SQL_ARMS, "arms = con.sql(SQL_ARMS.format(arm=ident(arm), outcome=ident(outcome))).df()\narms"),
            sh.sql_cell("SQL_VALUES", SQL_VALUES, "values = con.sql(SQL_VALUES.format(arm=ident(arm), outcome=ident(outcome))).df()"),
            _nb.code("import math, re\nfrom typing import Any, Sequence\n"
                     f"ALPHA = {ALPHA}\nPOWER_TARGET = {POWER_TARGET}\nEQUAL_VAR = {EQUAL_VAR}\nSRM_P = {SRM_P}\n"
                     f"CONTROL_LIKE = re.compile({CONTROL_LIKE.pattern!r}, re.IGNORECASE)\n\n"
                     + inspect.getsource(ci_multiplier) + "\n\n" + inspect.getsource(wilson) + "\n\n"
                     + inspect.getsource(compare_binary) + "\n\n" + inspect.getsource(compare_continuous) + "\n\n"
                     + inspect.getsource(control_arm) + "\n\n" + inspect.getsource(outcome_units) + "\n"
                     "from scipy import stats\n"
                     "binary = int(arms['n'].sum()) == int(arms['n01'].sum())\n"
                     "control = control_arm([str(a) for a in arms['arm']])\n"
                     "srm_p = stats.chisquare(arms['n'].astype(float).to_numpy()).pvalue\n"
                     "by = {str(r['arm']): r for _, r in arms.iterrows()}\n"
                     "comparisons = {}\n"
                     "for treatment in by:\n"
                     "    if treatment == control:\n"
                     "        continue\n"
                     "    if binary:\n"
                     "        comparisons[treatment] = compare_binary(int(by[treatment]['total']), int(by[treatment]['n']),\n"
                     "                                                int(by[control]['total']), int(by[control]['n']))\n"
                     "    else:\n"
                     "        comparisons[treatment] = compare_continuous(values.loc[values['arm'] == treatment, 'v'].astype(float),\n"
                     "                                                    values.loc[values['arm'] == control, 'v'].astype(float))\n"
                     "dict(binary=binary, control=control, srm_p=srm_p, srm_mismatch=srm_p < SRM_P), pd.DataFrame(comparisons).T"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     "arms.plot.barh(x='arm', y='mean', xerr=arms['variance'].pow(0.5) / arms['n'].pow(0.5), "
                     "title=f'{outcome} by {arm}')"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
