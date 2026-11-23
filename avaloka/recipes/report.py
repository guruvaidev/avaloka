"""Assemble and verify an Outcome Report (schema 1.1) from a recipe run.

A recipe hands over an evidence tree, template sentences with ``{fN}``
placeholders and the figures that fill them; this module writes the rest of
the document (analysis block, steps, completion, the ``data_checks`` of the
GateReport, provenance with the recipe and engine versions) and runs every
check the Outcome layer already has — the schema, unresolved references,
figure-to-evidence matching and the free-number rule — before anything is
shown. A figure that fails here is a bug in the recipe, so the caller fails
closed rather than shipping a wrong number. A gate's deciding value is shown
the same way: as a figure read from the evidence leaf the recipe names for
it (``gate_refs``), never as a number copied from the gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from avaloka.outcomes import format as _format
from avaloka.outcomes import spec as _spec

from .base import Bound, Evidence, GateReport, RecipeSpec, Verdict

#: What a qualified gate means for the reader, in plain words and no digits:
#: the deciding figure is in the evidence tree, never in this text.
GATE_WORDS: dict[str, str] = {
    "duplicates": "Some rows are exact duplicates; statistics use each distinct row once.",
    "missing_measure": "A large share of the measure is missing; statistics cover the rows that have it.",
    "date_parse": "Some dates could not be read and are left out of the periods.",
    "constant": "The measure has a single value.",
    "fanout_join": "The join multiplied rows.",
    "small_groups": "Some groups hold few rows, so their statistics are rough.",
    "high_cardinality": "The dimension has many values; only the largest are shown.",
    "sample_ratio": "The arms are not split as designed.",
    "leakage": "A column mirrors the outcome.",
    "period_gaps": "The period sequence has gaps; changes across a gap span more than one period.",
    "min_rows": "There are too few rows for this analysis.",
    "incomplete_period": "The latest period is incomplete and was left out; the windows are the last two complete periods.",
    "simpson": "Within each group of the dimension the relationship runs the other way from the overall figure.",
    "collinearity": "Some drivers move together, so their separate coefficients are not well separated.",
    "heteroscedasticity": "The residual spread is not constant across the fitted range; the standard errors are robust to it.",
    "flag_share": "More than a tenth of the points are flagged, which says the rule, not the data, is suspect.",
    "min_cohorts": "There are too few cohorts to read retention across them.",
    "power": "At this sample size the test would often miss a difference of the observed size; read the interval, "
             "not the point estimate.",
}
SAMPLING_WORDS = "Figures are computed on a sample of the rows."

#: Text-slot limits of the schema, by the name a recipe knows the slot by.
SLOT_LIMITS = {"headline": 160, "so_what": 140, "now_what": 140, "method_summary": 400}

#: How a gate's value is shown, by the evidence unit the gate declared:
#: (figure unit, decimals). A fraction reads as a percent, like every other.
_GATE_FIGURE_UNITS = {"fraction": ("percent", 1), "percent": ("percent", 1), "count": ("count", 0),
                      "currency": ("currency", 2), "pp": ("pp", 1), "ratio": ("ratio", 2),
                      "days": ("days", 0), "unitless": ("unitless", 2)}

#: Packages whose versions ``provenance.engine`` records when they were used
#: (imported) by the run; duckdb is always there, the rest only if loaded.
_ENGINES = ("duckdb", "scipy", "statsmodels", "pandas")


class ReportInvalid(ValueError):
    """The assembled report failed a check; ``problems`` lists each one."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("report failed verification: " + "; ".join(self.problems))


# ── figures ────────────────────────────────────────────────────────────────

def figure(fid: str, evidence: Mapping[str, Any], pointer: str, *, unit: str, decimals: int,
           currency: str | None = None) -> dict[str, Any]:
    """A ``figures[]`` entry read from the evidence leaf at ``pointer``.

    The value is taken from the leaf, never typed by hand, so a figure cannot
    disagree with its evidence; a fraction leaf shown as a percent is scaled.
    """
    value, leaf_unit, leaf_currency = _spec.evidence_leaf(evidence, pointer)
    shift = _format.evidence_shift(unit, leaf_unit)
    if shift is None:
        raise ReportInvalid([f"{pointer}: evidence unit {leaf_unit!r} cannot be shown as {unit!r}"])
    fig: dict[str, Any] = {"id": fid, "value": float(value) * (10 ** shift), "unit": unit,
                           "decimals": int(decimals), "evidence_ref": pointer}
    code = currency or leaf_currency
    if unit == "currency" and code:
        fig["currency"] = code
    return fig


def report_id(recipe_spec: RecipeSpec, bound: Bound, evidence: Mapping[str, Any]) -> str:
    """Deterministic: the same recipe, parameters and evidence give the same id."""
    key = json.dumps([recipe_spec.id, recipe_spec.version, dict(bound.params), evidence],
                     sort_keys=True, default=str)
    return "rpt_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ── assembly ───────────────────────────────────────────────────────────────

def merged_gate_words(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """``GATE_WORDS`` with a recipe's own sentences for the gates it reads
    differently (a closed gate vocabulary, so a recipe that checks an
    assumption the list does not name says what its gate meant)."""
    return {**GATE_WORDS, **(overrides or {})}


def _caveats(gates: GateReport, sampled: bool, words: Mapping[str, str]) -> list[dict[str, str]]:
    out = []
    if sampled:
        out.append({"kind": "sampling", "text": SAMPLING_WORDS})
    for g in gates.gates:
        if g.status == "qualified":
            out.append({"kind": g.caveat_kind, "text": words[g.name]})
    # One caveat per kind: the reader sees the kind, the evidence holds each gate.
    seen: dict[str, dict[str, str]] = {}
    for c in out:
        seen.setdefault(c["kind"], c)
    return list(seen.values())[:12]


def _completion(gates: GateReport, words: Mapping[str, str]) -> dict[str, Any]:
    if gates.verdict is Verdict.PASS:
        return {"state": "answered"}
    missing = [words[g.name][:140] for g in gates.gates if g.status == "qualified"]
    return {"state": "partial", "missing": missing[:6]}


def _slot(text: str, name: str) -> str:
    """``text`` when it fits the slot; a template over the limit is a recipe
    bug and raises, naming the slot and the limit, instead of being cut."""
    limit = SLOT_LIMITS[name]
    if len(text) > limit:
        raise ReportInvalid([f"{name}: template is {len(text)} characters, over the {limit}-character limit"])
    return text


def _next_figure_id(figures: Sequence[Mapping[str, Any]]) -> str:
    taken = [int(f["id"][1:]) for f in figures if str(f.get("id", "")).startswith("f") and f["id"][1:].isdigit()]
    return f"f{max(taken, default=0) + 1}"


def _data_checks(gates: GateReport, figures: list[dict[str, Any]], tree: Mapping[str, Any],
                 gate_refs: Mapping[str, str]) -> list[dict[str, Any]]:
    """``data_checks[]`` for every gate, appending to ``figures`` the figure that
    shows a gate's value: read from the evidence leaf ``gate_refs`` names for
    the gate (a figure already citing that leaf is reused), and held equal to
    the value the gate decided on, so the check shows what the check saw."""
    checks: list[dict[str, Any]] = []
    for g in gates.gates:
        check: dict[str, Any] = {"gate": g.name, "status": g.status}
        if g.status == "qualified" and g.caveat_kind:
            check["caveat_kind"] = g.caveat_kind
        if g.status == "failed" and g.reason:
            check["reason"] = g.reason
        if g.value is not None:
            pointer = gate_refs.get(g.name)
            if pointer is None:
                raise ReportInvalid([f"gate {g.name}: its value is not in the evidence tree (no gate_refs entry)"])
            unit, decimals = _GATE_FIGURE_UNITS[g.unit or "unitless"]
            fig = next((f for f in figures if f.get("evidence_ref") == pointer), None)
            if fig is None:
                try:
                    fig = figure(_next_figure_id(figures), tree, pointer, unit=unit, decimals=decimals)
                except (KeyError, ValueError) as exc:
                    raise ReportInvalid([f"gate {g.name}: no evidence leaf at {pointer} ({exc})"]) from None
                figures.append(fig)
            shift = _format.evidence_shift(fig["unit"], g.unit)
            if shift is None or not _format.figure_matches(float(g.value) * (10 ** shift), fig["decimals"], fig["value"]):
                raise ReportInvalid([f"gate {g.name}: value {g.value} {g.unit} does not match the evidence leaf "
                                     f"{pointer} ({fig['value']} {fig['unit']})"])
            check["figure"] = fig["id"]
        checks.append(check)
    return checks[:24]


def engine_versions() -> dict[str, str]:
    """``provenance.engine``: duckdb's version, plus scipy, statsmodels and
    pandas when the run imported them."""
    out: dict[str, str] = {}
    for name in _ENGINES:
        module = sys.modules.get(name)
        if module is None and name == "duckdb":
            try:
                import duckdb as module  # noqa: N813
            except ImportError:  # the report is still valid without it
                continue
        version = getattr(module, "__version__", None)
        if isinstance(version, str):
            out[name] = version
    return out


def build_report(*, recipe_spec: RecipeSpec, bound: Bound, evidence: Evidence | Mapping[str, Any],
                 gates: GateReport, headline_template: str,
                 bullets: Sequence[Mapping[str, str]], figures: Sequence[Mapping[str, Any]],
                 charts: Sequence[Mapping[str, Any]], method_summary: str,
                 rows_analyzed: int | None, total_rows: int | None,
                 kpis: Sequence[Mapping[str, Any]] = (), created_at: str | None = None,
                 sampled: bool = False, gate_words: Mapping[str, str] | None = None,
                 gate_refs: Mapping[str, str] | None = None) -> dict[str, Any]:
    """One valid Outcome Report 1.1 (``spec.assert_valid`` runs before return).

    Text slots hold placeholders only; every number comes from ``figures``,
    which point into ``evidence``; a template over its slot's limit raises
    :class:`ReportInvalid` rather than being cut. ``provenance.llm_used`` is
    false: nothing here was written by a model. ``gate_words`` replaces the
    caveat sentence of a gate the recipe reads differently from
    ``GATE_WORDS``; ``gate_refs`` names, per gate, the evidence pointer of the
    leaf holding the value the gate decided on, which ``data_checks[]`` cites
    as a figure.
    """
    from avaloka.version import __version__

    words = merged_gate_words(gate_words)
    tree = evidence.tree if isinstance(evidence, Evidence) else evidence
    tree = json.loads(json.dumps(tree, default=float))  # plain JSON, numpy scalars gone
    sampled = sampled or (rows_analyzed is not None and total_rows is not None
                          and rows_analyzed < total_rows)
    figs = [dict(f) for f in figures]
    checks = _data_checks(gates, figs, tree, gate_refs or {})
    report: dict[str, Any] = {
        "schema_version": _spec.SCHEMA_VERSION,
        "report_id": report_id(recipe_spec, bound, tree),
        "kind": "analysis",
        "domain": "generic",
        "template": "generic",
        "audience": "analyst",
        "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis": {"analysis_id": f"recipe-{recipe_spec.id}", "thread_id": "local",
                     "turn_id": "run", "version": 1, "parent_version": None, "dataset_ids": []},
        "headline": {"text": _slot(headline_template, "headline"),
                     "bullets": [{"so_what": _slot(b["so_what"], "so_what"), "now_what": _slot(b["now_what"], "now_what")}
                                 for b in bullets][:3]},
        "figures": figs,
        "kpis": [dict(k) for k in kpis][:8],
        "charts": [dict(c) for c in charts][:12],
        "method": {"summary": _slot(method_summary, "method_summary"),
                   "steps": [f"Recipe {recipe_spec.id}, version {recipe_spec.version}."],
                   "sample": {"rows_analyzed": rows_analyzed, "total_rows": total_rows,
                              "is_sampled": bool(sampled)}},
        "caveats": _caveats(gates, sampled, words),
        "steps": [{"id": recipe_spec.id, "order": 1, "title": recipe_spec.title[:80], "status": "done"}],
        "completion": _completion(gates, words),
        "data_checks": checks,
        "evidence": tree,
        "provenance": {"generator": "avaloka", "generator_version": __version__,
                       "llm_used": False, "llm_model": None,
                       # verify_report has not run yet; the runner sets passed=True after it.
                       "verifier": {"passed": False, "rejections": []},
                       "recipe": {"id": recipe_spec.id, "version": recipe_spec.version},
                       "engine": engine_versions()},
    }
    _spec.assert_valid(report)
    return report


# ── verification ───────────────────────────────────────────────────────────

def problems(report: Mapping[str, Any]) -> list[str]:
    """Every reason the report must not be shown; ``[]`` means it may."""
    found = [str(i) for i in _spec.validate(report)]
    if found:
        return found
    found += [str(i) for i in _spec.unresolved_refs(dict(report))]
    for fig in report.get("figures", []):
        if not _format.figure_matches_evidence(fig, report.get("evidence", {})):
            found.append(f"/figures/{fig.get('id')}: value does not match {fig.get('evidence_ref')}")
    for pointer, text in _spec.text_slots(dict(report)):
        free = _spec.free_numbers(text)
        if free:
            found.append(f"{pointer}: free numbers {free}")
    return found


def verify_report(report: Mapping[str, Any]) -> None:
    """Raise :class:`ReportInvalid` listing every problem; return on a clean report."""
    found = problems(report)
    if found:
        raise ReportInvalid(found)


def rendered_slots(report: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Each text slot with its placeholders filled, as a reader sees it."""
    for pointer, text in _spec.text_slots(dict(report)):
        yield pointer, _format.render_text(text, report.get("figures", []))
