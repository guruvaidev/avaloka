"""Schema 1.1 in a recipe run (design 1.7-INSTANT-ANALYSIS §4.3): the report's
``data_checks`` are the GateReport, one entry per gate the recipe emitted,
each deciding value a figure verified against the evidence; ``provenance``
names the recipe and the engines. Also ``build_report``'s template-length
assert: a slot template over the schema's limit is a recipe bug, raised,
never cut.
"""

from __future__ import annotations

import json
import re

import pytest

from avaloka.outcomes import format as _format
from avaloka.recipes import get
from avaloka.recipes.base import Bound, Gate, GateReport
from avaloka.recipes.library import describe
from avaloka.recipes.report import SLOT_LIMITS, ReportInvalid, build_report, engine_versions, figure
from tests.recipes._golden import case_id, cases, run_case


def reported_cases():
    return [c for c in cases() if json.loads((c / "expected.json").read_text())["verdict"] in ("pass", "qualified")]


# ── every golden: the data checks are the gates ────────────────────────────

@pytest.mark.parametrize("case", reported_cases(), ids=case_id)
def test_the_data_checks_are_the_gate_report(case):
    expected, result = run_case(case)
    report, gates = result.report, result.gates
    assert report is not None and gates is not None, (result.verdict, result.reason)
    assert report["schema_version"] == "1.1"
    assert [(c["gate"], c["status"]) for c in report["data_checks"]] == [(g.name, g.status) for g in gates.gates]
    figs = {f["id"]: f for f in report["figures"]}
    for check, gate in zip(report["data_checks"], gates.gates):
        assert check.get("caveat_kind") == (gate.caveat_kind if gate.status == "qualified" else None)
        assert check.get("reason") == (gate.reason if gate.status == "failed" else None)
        if gate.value is None:
            assert "figure" not in check
            continue
        # The deciding number is a figure: read from the evidence, and equal to what the gate saw.
        fig = figs[check["figure"]]
        assert _format.figure_matches_evidence(fig, report["evidence"])
        shift = _format.evidence_shift(fig["unit"], gate.unit)
        assert shift is not None
        assert _format.figure_matches(float(gate.value) * 10 ** shift, fig["decimals"], fig["value"]), (gate, fig)
    spec = get(expected["recipe"]).spec
    assert report["provenance"]["recipe"] == {"id": spec.id, "version": spec.version}
    assert "duckdb" in report["provenance"]["engine"]


def test_qualified_checks_with_a_value_cite_a_figure_somewhere():
    """The parametrised test above is not vacuous: goldens do carry qualified checks with figures."""
    cited = set()
    for case in reported_cases():
        _, result = run_case(case)
        for check in result.report["data_checks"]:
            if check["status"] == "qualified":
                assert "figure" in check, (case_id(case), check)
                cited.add(check["gate"])
    assert {"duplicates", "missing_measure", "small_groups", "period_gaps", "incomplete_period", "simpson",
            "collinearity", "flag_share", "power"} <= cited


def test_engine_versions_are_names_and_version_strings():
    versions = engine_versions()
    assert "duckdb" in versions
    for name, version in versions.items():
        assert re.fullmatch(r"[a-z_]+", name) and isinstance(version, str) and version


# ── build_report on its own ────────────────────────────────────────────────

TREE = {"rows": {"n": {"value": 3.0, "unit": "count"}, "share": {"value": 0.25, "unit": "fraction"}}}
BOUND = Bound("describe", "1.0", {"measure": "x"})


def _build(**over):
    kw = dict(recipe_spec=describe.SPEC, bound=BOUND, evidence=TREE, gates=GateReport(()),
              headline_template="Rows: {f1}.", bullets=[{"so_what": "So {f1}.", "now_what": "Now."}],
              figures=[figure("f1", TREE, "/rows/n", unit="count", decimals=0)], charts=[],
              method_summary="A count.", rows_analyzed=3, total_rows=3, created_at="2026-11-10T00:00:00Z")
    kw.update(over)
    return build_report(**kw)


@pytest.mark.parametrize("slot,over", [
    ("headline", lambda text: {"headline_template": text}),
    ("so_what", lambda text: {"bullets": [{"so_what": text, "now_what": "Now."}]}),
    ("now_what", lambda text: {"bullets": [{"so_what": "So.", "now_what": text}]}),
    ("method_summary", lambda text: {"method_summary": text}),
])
def test_an_over_long_template_raises_naming_the_slot_and_the_limit(slot, over):
    limit = SLOT_LIMITS[slot]
    assert _build(**over("x" * limit))   # at the limit it is a report
    with pytest.raises(ReportInvalid, match=rf"{slot}: template is {limit + 1} characters, over the {limit}-character limit"):
        _build(**over("x" * (limit + 1)))


def test_a_gate_value_without_an_evidence_leaf_is_a_recipe_bug():
    gates = GateReport((Gate("duplicates", "qualified", caveat_kind="data_quality", value=3, unit="count"),))
    with pytest.raises(ReportInvalid, match="gate duplicates: its value is not in the evidence tree"):
        _build(gates=gates)


def test_a_gate_ref_that_resolves_to_nothing_is_a_recipe_bug():
    gates = GateReport((Gate("duplicates", "qualified", caveat_kind="data_quality", value=3, unit="count"),))
    with pytest.raises(ReportInvalid, match="gate duplicates: no evidence leaf at /rows/missing"):
        _build(gates=gates, gate_refs={"duplicates": "/rows/missing"})
    with pytest.raises(ReportInvalid, match="gate duplicates: no evidence leaf at /rows"):
        _build(gates=gates, gate_refs={"duplicates": "/rows"})


def test_a_gate_value_that_disagrees_with_its_leaf_is_a_recipe_bug():
    gates = GateReport((Gate("duplicates", "qualified", caveat_kind="data_quality", value=4, unit="count"),))
    with pytest.raises(ReportInvalid, match="gate duplicates: value 4 count does not match the evidence leaf /rows/n"):
        _build(gates=gates, gate_refs={"duplicates": "/rows/n"})


def test_a_gate_figure_reuses_the_figure_citing_its_leaf_or_is_appended_after():
    gates = GateReport((Gate("duplicates", "qualified", caveat_kind="data_quality", value=3, unit="count"),
                        Gate("missing_measure", "qualified", caveat_kind="data_quality", value=0.25, unit="fraction"),
                        Gate("constant", "passed")))
    report = _build(gates=gates, gate_refs={"duplicates": "/rows/n", "missing_measure": "/rows/share"})
    assert [f["id"] for f in report["figures"]] == ["f1", "f2"]   # existing ids stay; the new one follows
    assert report["data_checks"] == [
        {"gate": "duplicates", "status": "qualified", "caveat_kind": "data_quality", "figure": "f1"},
        {"gate": "missing_measure", "status": "qualified", "caveat_kind": "data_quality", "figure": "f2"},
        {"gate": "constant", "status": "passed"},
    ]
    assert report["figures"][1] == {"id": "f2", "value": 25.0, "unit": "percent", "decimals": 1,
                                    "evidence_ref": "/rows/share"}
    assert report["provenance"]["recipe"] == {"id": "describe", "version": describe.VERSION}
