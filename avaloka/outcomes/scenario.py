"""The what-if scenario model (docs/design/2.0-OUTCOMES.md §2.2).

Bounded, clamped, additive, with an uncertainty band:

- each requested driver value is clamped to its observed [lo, hi] (P5-P95),
  and every clamp is recorded, because a scenario saved from the SDK must say
  that it was held at the edge of the data;
- the KPI change is the sum of the drivers' changes, with no interactions;
- the band is ``y ± z·σ`` with σ the root-sum-square of the drivers' errors
  (independent errors assumed), widened to at least ``± conformal_q`` when a
  split-conformal quantile exists, since a scenario cannot be more certain
  than one prediction.

``assets/outcome-runtime.js`` implements the same model in the same operation
order, so the slider in the HTML export and a scenario saved through the API
give the same floats; ``golden/scenario.json`` holds the two equal.

Forms in 2.0: ``linear`` (Δ = β·(x − x⁰), σ = |x − x⁰|·SE) and ``step`` (an
experiment arm: x is 0 for control or 1 for treatment, ``beta`` holds the
measured effect and ``se`` the CI half-width over 1.96, so the linear formulas
give the effect and its error). ``piecewise`` and ``elasticity`` are in the
schema but are second cut, and are refused here rather than approximated.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

Z_DEFAULT = 1.96
SUPPORTED_FORMS = ("linear", "step")
SURE_LEVELS = ("fairly_sure", "direction_only", "either_way")
SURE_TEXT = {
    "fairly_sure": "Fairly sure of the direction and size.",
    "direction_only": "Sure of the direction, not the size.",
    "either_way": "Could go either way at this setting.",
}
_FAIRLY_SURE_R = 0.5


class ScenarioError(ValueError):
    """An input the model refuses; ``code`` is shared with the JS runtime."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ScenarioError("value", f"{what} must be a finite number, got {value!r}")
    return value


def _check_driver(driver: Mapping[str, Any]) -> None:
    did = driver.get("id")
    form = driver.get("form")
    if form not in SUPPORTED_FORMS:
        raise ScenarioError("unsupported_form",
                            f"driver {did!r}: form {form!r} is not computed in this version "
                            f"(supported: {', '.join(SUPPORTED_FORMS)})")
    lo = _number(driver.get("lo"), f"driver {did!r} lo")
    hi = _number(driver.get("hi"), f"driver {did!r} hi")
    baseline = _number(driver.get("baseline"), f"driver {did!r} baseline")
    _number(driver.get("beta"), f"driver {did!r} beta")
    se = _number(driver.get("se"), f"driver {did!r} se")
    if not lo < hi:
        raise ScenarioError("driver", f"driver {did!r}: lo must be below hi")
    if not lo <= baseline <= hi:
        raise ScenarioError("driver", f"driver {did!r}: baseline is outside [lo, hi]")
    if se < 0:
        raise ScenarioError("driver", f"driver {did!r}: se must not be negative")
    if form == "step" and (lo != 0 or hi != 1 or baseline not in (0, 1)):
        raise ScenarioError("driver", f"driver {did!r}: a step driver has lo 0, hi 1 and "
                                      "a baseline of 0 (control) or 1 (treatment)")


def evaluate(whatif: Mapping[str, Any], values: Mapping[str, Any] | None = None,
             *, conformal_q: float | None = None) -> dict[str, Any]:
    """Evaluate one what-if panel at the requested driver values.

    ``values`` maps driver id to the requested value; a driver not named stays
    at its baseline. ``conformal_q`` defaults to the panel's own. Returns
    ``{"applied", "clamped", "sigma", "result": {"y", "lo", "hi", "sure"}}``.
    """
    drivers = list(whatif.get("drivers") or [])
    if not drivers:
        raise ScenarioError("driver", "a what-if panel needs at least one driver")
    y0 = _number(whatif.get("y0"), "y0")
    z = _number(whatif.get("z", Z_DEFAULT), "z")
    if conformal_q is None:
        conformal_q = whatif.get("conformal_q")
    if conformal_q is not None:
        conformal_q = _number(conformal_q, "conformal_q")
    values = dict(values or {})
    known = {d.get("id") for d in drivers}
    for name in values:
        if name not in known:
            raise ScenarioError("unknown_driver", f"{name!r} is not a driver of this panel")

    applied: dict[str, float] = {}
    clamped: list[dict[str, Any]] = []
    y = y0
    variance = 0.0
    for driver in drivers:
        _check_driver(driver)
        did = driver["id"]
        x0, lo, hi = driver["baseline"], driver["lo"], driver["hi"]
        requested = _number(values.get(did, x0), f"value for {did!r}")
        if driver["form"] == "step" and requested not in (0, 1):
            raise ScenarioError("step_level", f"{did!r} takes 0 (control) or 1 (treatment)")
        x = min(max(requested, lo), hi)
        if x != requested:
            clamped.append({"driver": did, "requested": requested, "applied": x,
                            "bound": "lo" if x == lo else "hi"})
        applied[did] = x
        y += driver["beta"] * (x - x0)
        s = abs(x - x0) * driver["se"]
        variance += s * s

    sigma = math.sqrt(variance)
    lo_y = y - z * sigma
    hi_y = y + z * sigma
    if conformal_q is not None and conformal_q > 0:
        lo_y = min(lo_y, y - conformal_q)
        hi_y = max(hi_y, y + conformal_q)
    return {"applied": applied, "clamped": clamped, "sigma": sigma,
            "result": {"y": y, "lo": lo_y, "hi": hi_y,
                       "sure": how_sure(y0, y, lo_y, hi_y, clamped=bool(clamped))}}


def how_sure(y0: float, y: float, lo: float, hi: float, *, clamped: bool = False) -> str:
    """The §2.2 "how sure are we" level, one level lower when a driver was clamped."""
    excludes_no_change = lo > y0 or hi < y0
    if not excludes_no_change:
        level = 2
    elif (hi - lo) / (2 * abs(y - y0)) < _FAIRLY_SURE_R:
        level = 0
    else:
        level = 1
    if clamped:
        level = min(level + 1, 2)
    return SURE_LEVELS[level]


def scenario_block(whatif: Mapping[str, Any], values: Mapping[str, Any] | None,
                   *, base_version: int, name: str | None = None,
                   conformal_q: float | None = None) -> dict[str, Any]:
    """The report's ``scenario`` object (§4.1 ``$defs/scenario``) for a saved scenario."""
    out = evaluate(whatif, values, conformal_q=conformal_q)
    block: dict[str, Any] = {
        "whatif": whatif["id"], "base_version": base_version,
        "values": {d["id"]: (values or {}).get(d["id"], d["baseline"])
                   for d in whatif["drivers"]},
        "applied": out["applied"], "clamped": out["clamped"],
        "evidence_kind": whatif["evidence_kind"], "result": out["result"],
    }
    if name is not None:
        block = {"name": name, **block}
    return block
