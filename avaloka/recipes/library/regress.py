"""``regress``: how much of an outcome a few drivers account for, and which.

An ordinary least squares fit with HC3 robust standard errors (statsmodels)
over distinct, complete rows: coefficients, standardized betas, p-values, R²
and adjusted R²; a variance inflation factor per driver; Breusch-Pagan and
Jarque-Bera residual tests; the residual standard deviation. The wording is
associational throughout: a coefficient is a pattern in this data with the
other drivers held fixed, never what a change would do.

``min_rows`` fails when too few rows hold every column, ``leakage`` when a
driver mirrors the outcome, ``constant`` when no driver varies (constant
drivers are dropped and named in ``method``); ``collinearity`` qualifies as
``independence`` when drivers mirror each other (VIF above the limit) and
``heteroscedasticity`` as ``additive_model`` when the residual spread is not
constant.
"""

from __future__ import annotations

import inspect
import math
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
MIN_ROWS = 30                # fewer complete rows than this is a refusal
ROWS_PER_DRIVER = 10         # and fewer than this many rows per driver
MAX_DRIVERS = 8              # drivers taken by default
LEAK_R = 0.98                # a driver this correlated with the outcome is taken to be computed from it
VIF_LIMIT = 10.0             # a larger variance inflation factor is an independence caveat
HETERO_P = 0.05              # a smaller Breusch-Pagan p-value is an additive_model caveat
Z95 = 1.959963984540054
COV_TYPE = "HC3"

SQL_DEDUP = "SELECT DISTINCT * FROM t"
SQL_ROW_COUNTS = "SELECT count(*) AS rows_raw, (SELECT count(*) FROM ({dedup})) AS rows_distinct FROM t"
SQL_VALUES = "WITH d AS ({dedup}) SELECT {columns} FROM d WHERE {complete} ORDER BY {columns}"

EVIDENCE_CELL = """\
unit = evidence_unit(OUTCOME_UNIT)
strongest = max(result["drivers"], key=lambda d: abs(d["std_beta"]))
evidence = {
    "rows": rows_leaves(counts), "n": leaf(result["n"], "count"),
    "drivers": {"count": leaf(len(drivers), "count"), "constant": leaf(len(measures) - len(drivers), "count"),
                "by": {f"d{i}": {"coef": leaf(d["coef"], "unitless"), "std_beta": leaf(d["std_beta"], "unitless"),
                                 "p": leaf(d["p"], "unitless"), "vif": leaf(d["vif"], "unitless")}
                       for i, d in enumerate(result["drivers"], start=1)}},
    "fit": {"r2": leaf(result["r2"], "fraction"), "adj_r2": leaf(result["adj_r2"], "fraction"),
            "residual_std": leaf(result["residual_std"], unit), "f_p": leaf(result["f_p"], "unitless")},
    "tests": {"breusch_pagan_p": leaf(result["breusch_pagan_p"], "unitless"),
              "jarque_bera_p": leaf(result["jarque_bera_p"], "unitless"),
              "vif_max": leaf(max(d["vif"] for d in result["drivers"]), "unitless")},
    "strongest": {"std_beta": leaf(strongest["std_beta"], "unitless"), "p": leaf(strongest["p"], "unitless"),
                  "se": leaf(strongest["se"], "unitless")},
}
"""

SPEC = RecipeSpec(
    id="regress", version=VERSION, title="Regress an outcome on drivers",
    params=(ParamSpec("outcome", "measure"), ParamSpec("measures", "measure", required=False, many=True)),
    preconditions=("has_two_measures", "min_rows_30"),
    quality_gates=("min_rows", "leakage", "constant", "collinearity", "heteroscedasticity", "duplicates"),
    evidence_kind="associational", explain_slots=("headline", "bullet_1"),
)

GATE_WORDS = merged_gate_words({
    "min_rows": "Too few rows hold the outcome and every driver.",
    "collinearity": "Some drivers move together (variance inflation above the limit), so their separate coefficients "
                    "are not well separated.",
    "heteroscedasticity": "The residual spread is not constant across the fitted range, so a straight additive fit is an "
                          "approximation; the standard errors are robust to it.",
})


@dataclass(frozen=True)
class RegressBound(sh.RecipeBound):
    unit: str = "unitless"       # the outcome's semantic unit, for the residual spread


def _sql(bound: Bound) -> dict[str, str]:
    p = bound.params
    cols = [p["outcome"]] + list(p["measures"])
    return {"dedup": SQL_DEDUP, "columns": ", ".join(ident(c) for c in cols),
            "complete": " AND ".join(f"{ident(c)} IS NOT NULL" for c in cols)}


def _as_bound(bound: Bound) -> RegressBound:
    return bound if isinstance(bound, RegressBound) else RegressBound(bound.recipe_id, bound.version, bound.params)


def _values(con: Any, bound: Bound) -> pd.DataFrame:
    return query(con, SQL_VALUES.format(**_sql(bound))).astype(float)


def _varying(values: pd.DataFrame, drivers: Sequence[str]) -> tuple[list[str], list[str]]:
    kept = [d for d in drivers if len(values) > 1 and float(values[d].std()) > 0.0]
    return kept, [d for d in drivers if d not in kept]


def named_like(driver: str, outcome: str) -> bool:
    """The driver's name is the outcome's with a suffix (``revenue_target`` for ``revenue``)."""
    d, o = driver.strip().lower(), outcome.strip().lower()
    return len(d) > len(o) and d.startswith(o)


def leaking(values: pd.DataFrame, outcome: str, drivers: Sequence[str]) -> list[tuple[str, float]]:
    """Drivers that mirror the outcome: |r| above ``LEAK_R`` or named like it."""
    out = []
    for d in drivers:
        r = abs(float(values[d].corr(values[outcome])))
        if (not math.isnan(r) and r > LEAK_R) or named_like(d, outcome):
            out.append((d, 0.0 if math.isnan(r) else r))
    return out


def fit(values: pd.DataFrame, outcome: str, drivers: Sequence[str]) -> dict[str, Any]:
    """OLS with HC3 errors, VIFs and residual tests, as plain floats."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import jarque_bera

    X = sm.add_constant(values[list(drivers)], has_constant="add")
    y = values[outcome]
    model = sm.OLS(y, X).fit(cov_type=COV_TYPE)
    sd_y = float(y.std())
    rows = []
    for i, d in enumerate(drivers, start=1):
        scale = float(values[d].std()) / sd_y if sd_y else 0.0
        coef, se = float(model.params[d]), float(model.bse[d])
        rows.append({"driver": d, "coef": coef, "se": se, "std_beta": coef * scale, "p": float(model.pvalues[d]),
                     "ci_low": coef - Z95 * se, "ci_high": coef + Z95 * se,
                     "std_low": (coef - Z95 * se) * scale, "std_high": (coef + Z95 * se) * scale,
                     "vif": float(variance_inflation_factor(X.values, i)) if len(drivers) > 1 else 1.0})
    bp = het_breuschpagan(model.resid, model.model.exog)
    jb = jarque_bera(model.resid)
    return {"n": int(model.nobs), "r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj),
            "f_p": float(model.f_pvalue) if not math.isnan(model.f_pvalue) else 1.0,
            "residual_std": float(math.sqrt(model.mse_resid)), "breusch_pagan_p": float(bp[1]),
            "jarque_bera_p": float(jb[1]), "drivers": rows}


def _num(v: Any) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


class Regress:
    spec = SPEC

    def bind(self, intent: Intent, sem: Mapping[str, ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
        outcome = sh.bind_column(intent, sem, "outcome", ("measure", "target"), required=True, missing_reason="no_measure",
                                 question="Which column is the outcome to account for?")
        if not isinstance(outcome, str):
            return outcome
        named = intent.params.get("measures")
        named = (named,) if isinstance(named, str) else tuple(named or ())
        for m in named:
            if m not in sem or sem[m].role not in ("measure", "target") or m == outcome:
                return Refusal("no_measure", "A driver named in the question is not a measure column other than the outcome.")
        drivers = list(named)
        if not drivers:
            drivers = [m for m in sh.columns_of_role(sem, "measure", "target")
                       if m != outcome and not sh.is_id_like(m, sem)][:MAX_DRIVERS]
        if not drivers:
            return Refusal("no_measure", "This data has no measure columns besides the outcome to use as drivers.")
        return RegressBound("regress", VERSION, {"outcome": outcome, "measures": tuple(drivers)},
                            source=sh.source_of(profile), unit=sem[outcome].unit or "unitless")

    def check(self, con: Any, bound: Bound) -> GateReport:
        b = _as_bound(bound)
        outcome, drivers = b.params["outcome"], b.params["measures"]
        values = _values(con, b)
        kept, _ = _varying(values, drivers)
        n = len(values)
        if n < MIN_ROWS or n < ROWS_PER_DRIVER * max(len(kept), 1):
            return GateReport((Gate("min_rows", "failed", reason="insufficient_rows", value=n, unit="count"),))
        gates = [Gate("min_rows", "passed")]
        if not kept or float(values[outcome].std()) == 0.0:
            gates.append(Gate("constant", "failed", reason="constant_column", value=len(kept), unit="count"))
            return GateReport(tuple(gates))
        leaks = leaking(values, outcome, kept)
        if leaks:
            gates.append(Gate("leakage", "failed", reason="leakage_suspected", value=max(r for _, r in leaks), unit="ratio"))
            return GateReport(tuple(gates))
        result = fit(values, outcome, kept)
        vif_max = max(d["vif"] for d in result["drivers"])
        gates += [Gate("leakage", "passed"), Gate("constant", "passed")]
        gates.append(Gate("collinearity", "qualified", caveat_kind="independence", value=vif_max, unit="unitless")
                     if vif_max > VIF_LIMIT else Gate("collinearity", "passed"))
        gates.append(Gate("heteroscedasticity", "qualified", caveat_kind="additive_model", value=result["breusch_pagan_p"], unit="unitless")
                     if result["breusch_pagan_p"] < HETERO_P else Gate("heteroscedasticity", "passed"))
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        duplicates = int(counts["rows_raw"]) - int(counts["rows_distinct"])
        gates.append(Gate("duplicates", "qualified", caveat_kind="data_quality", value=duplicates, unit="count")
                     if duplicates else Gate("duplicates", "passed"))
        return GateReport(tuple(gates))

    def compute(self, con: Any, bound: Bound) -> Evidence:
        b = _as_bound(bound)
        outcome, drivers = b.params["outcome"], b.params["measures"]
        unit = sh.evidence_unit(b.unit)
        counts = query(con, SQL_ROW_COUNTS.format(**_sql(b))).iloc[0]
        values = _values(con, b)
        kept, constant = _varying(values, drivers)
        result = fit(values, outcome, kept)
        strongest = max(result["drivers"], key=lambda d: abs(d["std_beta"]))
        tree: dict[str, Any] = {
            "rows": {"raw": sh.leaf(counts["rows_raw"], "count"), "distinct": sh.leaf(counts["rows_distinct"], "count"),
                     "duplicates": sh.leaf(int(counts["rows_raw"]) - int(counts["rows_distinct"]), "count")},
            "n": sh.leaf(result["n"], "count"),
            "drivers": {"count": sh.leaf(len(kept), "count"), "constant": sh.leaf(len(constant), "count"),
                        "by": {f"d{i}": {"coef": sh.leaf(d["coef"], "unitless"), "std_beta": sh.leaf(d["std_beta"], "unitless"),
                                         "p": sh.leaf(d["p"], "unitless"), "vif": sh.leaf(d["vif"], "unitless")}
                               for i, d in enumerate(result["drivers"], start=1)}},
            "fit": {"r2": sh.leaf(result["r2"], "fraction"), "adj_r2": sh.leaf(result["adj_r2"], "fraction"),
                    "residual_std": sh.leaf(result["residual_std"], unit), "f_p": sh.leaf(result["f_p"], "unitless")},
            "tests": {"breusch_pagan_p": sh.leaf(result["breusch_pagan_p"], "unitless"),
                      "jarque_bera_p": sh.leaf(result["jarque_bera_p"], "unitless"),
                      "vif_max": sh.leaf(max(d["vif"] for d in result["drivers"]), "unitless")},
            "strongest": {"std_beta": sh.leaf(strongest["std_beta"], "unitless"), "p": sh.leaf(strongest["p"], "unitless"),
                          "se": sh.leaf(strongest["se"], "unitless")},
        }
        tables = {"drivers": sh.table(["driver", "coef", "se", "std_beta", "std_low", "std_high", "p", "ci_low", "ci_high", "vif"],
                                      [[d["driver"], d["coef"], d["se"], d["std_beta"], d["std_low"], d["std_high"], d["p"],
                                        d["ci_low"], d["ci_high"], d["vif"]] for d in result["drivers"]])}
        return Evidence(tree, tables)

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]:
        b = _as_bound(bound)
        ev, p = evidence.tree, b.params
        unit = sh.figure_unit(b.unit)
        figs = [figure("f1", ev, "/fit/r2", unit="percent", decimals=1),
                figure("f2", ev, "/n", unit="count", decimals=0),
                figure("f3", ev, "/drivers/count", unit="count", decimals=0),
                figure("f4", ev, "/strongest/std_beta", unit="unitless", decimals=2),
                figure("f5", ev, "/strongest/p", unit="unitless", decimals=3),
                figure("f6", ev, "/fit/adj_r2", unit="percent", decimals=1),
                figure("f7", ev, "/tests/vif_max", unit="unitless", decimals=1),
                figure("f8", ev, "/tests/breusch_pagan_p", unit="unitless", decimals=3),
                figure("f9", ev, "/fit/residual_std", unit=unit, decimals=2),
                figure("f10", ev, "/rows/duplicates", unit="count", decimals=0),
                figure("f11", ev, "/tests/jarque_bera_p", unit="unitless", decimals=3),
                figure("f12", ev, "/strongest/se", unit="unitless", decimals=3)]
        bullets = [{"so_what": "The strongest of {f3} drivers has standardized coefficient {f4} "
                               "(robust standard error {f12}, p-value {f5}); the adjusted fit is {f6}.",
                    "now_what": "Read each coefficient as an association in this data with the other drivers held fixed, "
                                "not as what a change would do."}]
        if ev["tests"]["vif_max"]["value"] > VIF_LIMIT:
            bullets.append({"so_what": "The largest variance inflation factor is {f7}, so some drivers overlap.",
                            "now_what": "Drop or combine overlapping drivers before weighing them separately."})
        if ev["tests"]["breusch_pagan_p"]["value"] < HETERO_P:
            bullets.append({"so_what": "The residual spread varies across the fitted range (test p-value {f8}); "
                                       "the robust standard errors allow for it.",
                            "now_what": "Try a transformed outcome or added terms if the spread pattern matters."})
        if len(bullets) < 3:
            bullets.append({"so_what": "The typical miss of the fit (residual standard deviation) is {f9}; "
                                       "the residual normality test has p-value {f11}.",
                            "now_what": "Compare that miss with the spread of the outcome before relying on the fit."})
        kpis = [{"id": "r2", "label": "Variation accounted for", "value": "f1", "direction_good": "neutral"},
                {"id": "strongest", "label": "Strongest standardized beta", "value": "f4", "direction_good": "neutral"},
                {"id": "rows", "label": "Complete rows", "value": "f2", "direction_good": "neutral"}]
        rows = evidence.tables["drivers"]["rows"]
        charts = [sh.hbar_interval_chart("betas", f"Standardized drivers of {sh.label(p['outcome'])}",
                                         [r[0] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
                                         x_name="standardized coefficient", y_name="driver",
                                         alt_text=f"Horizontal bars of the standardized coefficient of each driver of "
                                                  f"{sh.label(p['outcome'])} with its interval; the share of variation "
                                                  "accounted for is given in the headline.")]
        n_rows = int(ev["rows"]["raw"]["value"])
        kept = [r[0] for r in rows]
        dropped = [d for d in p["measures"] if d not in kept]
        return build_report(
            recipe_spec=SPEC, bound=b, evidence=evidence, gates=gates,
            headline_template="Together the drivers account for {f1} of the variation in the outcome.",
            bullets=bullets[:3], figures=figs, charts=charts, kpis=kpis,
            method_summary=f"Ordinary least squares of {p['outcome']} on {', '.join(kept)} with {COV_TYPE} robust standard "
                           "errors (statsmodels) over distinct rows holding every column; standardized betas, variance "
                           "inflation factors, Breusch-Pagan and Jarque-Bera residual tests"
                           + (f"; dropped as constant: {', '.join(dropped)}" if dropped else "") + ".",
            rows_analyzed=n_rows, total_rows=n_rows, gate_words=GATE_WORDS,
            gate_refs={"collinearity": "/tests/vif_max", "heteroscedasticity": "/tests/breusch_pagan_p",
                       "duplicates": "/rows/duplicates"})

    def notebook(self, bound: Bound, gates: GateReport | None = None) -> dict[str, Any]:
        b = _as_bound(bound)
        cells = [
            sh.header_cell("regress", VERSION, SPEC.title),
            sh.parameters_cell(b, {"OUTCOME_UNIT": b.unit, "COV_TYPE": COV_TYPE}),
            sh.load_cell(b.source),
            sh.sql_cell("SQL_DEDUP", SQL_DEDUP, "# the fit is over distinct rows"),
            sh.sql_cell("SQL_ROW_COUNTS", SQL_ROW_COUNTS, "counts = con.sql(SQL_ROW_COUNTS.format(dedup=SQL_DEDUP)).df()\ncounts"),
            sh.sql_cell("SQL_VALUES", SQL_VALUES,
                        "cols = [outcome] + list(measures)\n"
                        "columns = ', '.join(ident(c) for c in cols)\n"
                        "complete = ' AND '.join(f'{ident(c)} IS NOT NULL' for c in cols)\n"
                        "values = con.sql(SQL_VALUES.format(dedup=SQL_DEDUP, columns=columns, complete=complete)).df().astype(float)"),
            _nb.code("import math\nfrom typing import Any, Sequence\nimport statsmodels.api as sm\n\n"
                     f"Z95 = {Z95!r}\n\n" + inspect.getsource(fit) + "\n"
                     "drivers = [d for d in measures if values[d].std() > 0]\n"
                     "result = fit(values, outcome, drivers)\n"
                     "pd.DataFrame(result['drivers'])"),
            _nb.code("model = sm.OLS(values[outcome], sm.add_constant(values[drivers], has_constant='add')).fit(cov_type=COV_TYPE)\n"
                     "print(model.summary())\n"
                     "{k: v for k, v in result.items() if k != 'drivers'}"),
            _nb.code("import matplotlib.pyplot as plt\n"
                     "pd.Series({d['driver']: d['std_beta'] for d in result['drivers']}).plot.barh(title=f'Standardized drivers of {outcome}')"),
            sh.evidence_cell(EVIDENCE_CELL),
            sh.limitations_cell(gates, GATE_WORDS),
        ]
        return _nb.notebook(cells)
