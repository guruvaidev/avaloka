"""The recipe contract is closed, and closed against the Outcome spec.

These tests pin the vocabularies of ``avaloka.recipes.base`` to the schema the
renderers read, so a recipe cannot raise a caveat the export cannot show or
claim an evidence kind the wording rules do not know.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avaloka.recipes import (CAVEAT_KINDS, EVIDENCE_KINDS, GATES, PARAM_NAMES, PRECONDITIONS,
                             REASONS, RECIPE_IDS, Ask, Gate, GateReport, Intent, ParamSpec,
                             RecipeSpec, Refusal, Verdict)

SCHEMA = json.loads((Path("avaloka/outcomes/outcome-report.schema.json")).read_text())


def test_caveat_kinds_are_the_schemas():
    schema_kinds = SCHEMA["properties"]["caveats"]["items"]["properties"]["kind"]["enum"]
    assert list(CAVEAT_KINDS) == schema_kinds


def test_evidence_kinds_are_the_schemas():
    assert "associational" in EVIDENCE_KINDS and "experimental" in EVIDENCE_KINDS
    kinds = json.dumps(SCHEMA)
    for k in EVIDENCE_KINDS:
        assert f'"{k}"' in kinds


def test_vocabularies_have_no_duplicates():
    for vocab in (RECIPE_IDS, PARAM_NAMES, PRECONDITIONS, GATES, REASONS):
        assert len(set(vocab)) == len(vocab)


def test_recipe_ids_are_the_designs_ten():
    assert RECIPE_IDS == ("profile", "describe", "trend", "compare_periods", "segment",
                          "cohort_retention", "correlate", "regress", "anomalies", "ab_test")


def _spec(**over):
    base = dict(id="describe", version="1.0", title="Describe",
                params=(ParamSpec("measure", "measure"), ParamSpec("dimension", "dimension", required=False)),
                preconditions=("has_measure",), quality_gates=("missing_measure", "constant"),
                evidence_kind="associational")
    base.update(over)
    return RecipeSpec(**base)


def test_spec_accepts_a_well_formed_recipe():
    s = _spec()
    assert s.major == 1 and s.explain_slots == ("headline",)


@pytest.mark.parametrize("over", [
    {"id": "forecast"},                       # not in RECIPE_IDS
    {"version": "1"},                         # not MAJOR.MINOR
    {"version": "01.2"},
    {"preconditions": ("has_rows",)},         # not a precondition
    {"quality_gates": ("vibes",)},            # not a gate
    {"evidence_kind": "causal"},              # not an evidence kind
    {"params": (ParamSpec("measure", "measure"), ParamSpec("measure", "measure"))},
])
def test_spec_rejects_out_of_vocabulary_values(over):
    with pytest.raises(ValueError):
        _spec(**over)


def test_param_spec_rejects_unknown_name_and_role():
    with pytest.raises(ValueError):
        ParamSpec("metric", "measure")
    with pytest.raises(ValueError):
        ParamSpec("measure", "number")


def test_intent_rejects_unknown_parameter_and_source():
    with pytest.raises(ValueError):
        Intent("q", "describe", {"metric": "x"})
    with pytest.raises(ValueError):
        Intent("q", "describe", {}, source="guess")
    i = Intent("q", "describe", {"measure": "revenue"}, 0.9, "fixture")
    assert i.params["measure"] == "revenue"


def test_refusal_reason_is_closed():
    with pytest.raises(ValueError):
        Refusal("because", "text")
    assert Refusal("insufficient_rows", "Fewer than 30 rows.").reason == "insufficient_rows"


def test_gate_requires_a_caveat_when_qualified_and_a_reason_when_failed():
    with pytest.raises(ValueError):
        Gate("duplicates", "qualified")
    with pytest.raises(ValueError):
        Gate("duplicates", "failed")
    with pytest.raises(ValueError):
        Gate("duplicates", "passed", unit="rows")
    g = Gate("duplicates", "qualified", caveat_kind="data_quality", value=0.08, unit="fraction")
    assert g.value == 0.08


def test_gate_report_verdict_is_the_worst_gate():
    passed = Gate("constant", "passed")
    qual = Gate("duplicates", "qualified", caveat_kind="data_quality", value=12, unit="count")
    failed = Gate("missing_measure", "failed", reason="no_measure")
    assert GateReport((passed,)).verdict is Verdict.PASS
    assert GateReport((passed, qual)).verdict is Verdict.QUALIFIED
    assert GateReport((passed, qual)).caveat_kinds == ("data_quality",)
    r = GateReport((qual, failed))
    assert r.verdict is Verdict.REFUSE and r.reason == "no_measure"


def test_every_gate_a_recipe_emits_is_in_its_spec():
    """Run every golden case through the recipe; each Gate name check() emits
    must be declared in that recipe's SPEC.quality_gates, so the report's
    data-checks section never shows a name the spec does not know."""
    from tests.recipes._golden import cases, run_case

    seen: dict[str, set[str]] = {}
    for case in cases():
        expected, result = run_case(case)
        if result.gates is not None:
            seen.setdefault(result.recipe_id, set()).update(g.name for g in result.gates.gates)
    import avaloka.recipes.library  # noqa: F401
    from avaloka.recipes import all_recipes
    for recipe in all_recipes():
        emitted = seen.get(recipe.spec.id, set())
        assert emitted, f"no golden exercised {recipe.spec.id}'s check()"
        assert emitted <= set(recipe.spec.quality_gates), (recipe.spec.id, emitted - set(recipe.spec.quality_gates))
        for name in emitted:
            assert name in GATES


def test_ask_names_a_parameter():
    a = Ask("measure", "Which column holds the amount?", ("revenue", "cost"))
    assert a.parameter == "measure" and len(a.options) == 2
