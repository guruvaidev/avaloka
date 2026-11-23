"""Outcome Report spec v1.1: schema, validator, formatter and scenario math.

docs/design/2.0-OUTCOMES.md §4.1 (schema), 1.7-INSTANT-ANALYSIS.md §4.3 (the 1.1 additions), §3.2 rule 3 (matching a figure to
its evidence), §2.2 (scenario model), §7 S1-S3. The golden vectors under
avaloka/outcomes/golden/ are shared with ui/tests/unit/outcome-runtime.test.ts,
so the Python and JavaScript sides are held to the same strings and floats.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from avaloka.outcomes import format as fmt
from avaloka.outcomes import scenario, spec

REPO = Path(__file__).resolve().parents[3]
OUTCOMES = REPO / "avaloka" / "outcomes"
GOLDEN = OUTCOMES / "golden"


def _load(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


RETAIL = _load("example_retail.json")
FINTECH = _load("example_fintech.json")
FORMAT = _load("format.json")
SCENARIO = _load("scenario.json")


def _codes(issues):
    return {(i.path, i.code) for i in issues}


def _retail():
    return copy.deepcopy(RETAIL)


# ═══ S1 — the schema ═══════════════════════════════════════════════════════

def test_published_schema_file_is_the_dict():
    """outcome-report.schema.json is generated from spec.SCHEMA, never edited apart."""
    assert spec.SCHEMA_PATH.read_text(encoding="utf-8") == spec.published_schema_text()
    assert json.loads(spec.SCHEMA_PATH.read_text())["properties"]["schema_version"] == \
        {"const": spec.SCHEMA_VERSION}


@pytest.mark.parametrize("report", [RETAIL, FINTECH], ids=["retail", "fintech"])
def test_both_doc_examples_validate(report):
    """§4.1 Examples A and B, verbatim, are valid and every figure reference resolves."""
    assert spec.validate(report) == []
    assert spec.unresolved_refs(report) == []
    for _, text in spec.text_slots(report):
        fmt.render_text(text, report["figures"])


@pytest.mark.parametrize("where", [
    (),
    ("figures", 0),
    ("kpis", 0),
    ("charts", 0),
    ("charts", 0, "data_table"),
    ("maps", 0),
    ("whatif", 0, "drivers", 0),
    ("strategies", 0, "design"),
    ("provenance", "verifier"),
    ("method", "sample"),
], ids=lambda w: "/".join(map(str, w)) or "root")
def test_spec_rejects_unknown_field(where):
    """Every object of v1.0 is closed: one unknown key anywhere is rejected, at its path."""
    report = _retail()
    node = report
    for step in where:
        node = node[step]
    node["surprise"] = 1
    pointer = "".join(f"/{p}" for p in where) + "/surprise"
    assert ("unknown_field" in {i.code for i in spec.validate(report)}
            and (pointer, "unknown_field") in _codes(spec.validate(report)))


def test_spec_rejects_unknown_figure_key():
    """§3.5 V-a's named bite: delete additionalProperties from $defs/figure → red."""
    report = _retail()
    report["figures"][0]["value_text"] = "8.4%"
    assert _codes(spec.validate(report)) == {("/figures/0/value_text", "unknown_field")}


@pytest.mark.parametrize("section", ["whatif", "strategies"])
def test_spec_rejects_missing_evidence_kind(section):
    """§0A (a): evidence_kind is required on every what-if and strategy."""
    report = _retail()
    del report[section][0]["evidence_kind"]
    assert _codes(spec.validate(report)) == {(f"/{section}/0", "missing_field")}


def test_spec_rejects_scenario_without_evidence_kind():
    report = _retail()
    report["kind"] = "what_if"
    report["scenario"] = scenario.scenario_block(
        _linear_whatif(), {"discount": 40}, base_version=3)
    assert spec.validate(report) == []
    del report["scenario"]["evidence_kind"]
    assert _codes(spec.validate(report)) == {("/scenario", "missing_field")}


@pytest.mark.parametrize("planted,code", [
    ({"formatter": "function (p) { return p.value; }"}, "code_in_option"),
    ({"formatter": "(p) => p.value"}, "code_in_option"),
    ({"title": {"text": "</script><script>alert(1)</script>"}}, "code_in_option"),
    ({"<img src=x onerror=alert(1)>": 1}, "code_in_option"),
    ({"formatter": print}, "not_json"),
])
def test_spec_rejects_code_in_an_echarts_option(planted, code):
    """§7 S1: a function, '=>' or '<' anywhere in an option is rejected."""
    report = _retail()
    report["charts"][0]["option"].update(planted)
    assert code in {i.code for i in spec.validate(report)}
    assert all(i.path.startswith("/charts/0/option") for i in spec.validate(report))


def test_spec_rejects_a_raw_number_where_a_figure_reference_belongs():
    """§7 S1: kpis[].value is an fN reference, never the number itself."""
    report = _retail()
    report["kpis"][0]["value"] = 2.6
    assert _codes(spec.validate(report)) == {("/kpis/0/value", "type")}


@pytest.mark.parametrize("mutate,expected", [
    (lambda r: r.update(report_id="rpt_3f9a0c1d2e4b5a69\n"), ("/report_id", "pattern")),
    (lambda r: r.update(created_at="18 Oct 2026"), ("/created_at", "format")),
    (lambda r: r["analysis"].update(version=True), ("/analysis/version", "type")),
    (lambda r: r["analysis"].update(version=0), ("/analysis/version", "range")),
    (lambda r: r["figures"][0].update(value=float("nan")), ("/figures/0/value", "not_json")),
    (lambda r: r["figures"][0].update(decimals=5), ("/figures/0/decimals", "range")),
    (lambda r: r["figures"][0].update(unit="percentage"), ("/figures/0/unit", "enum")),
    (lambda r: r["headline"]["bullets"].extend([r["headline"]["bullets"][0]] * 2),
     ("/headline/bullets", "items")),
    (lambda r: r["side_notes"][0].update(dismissable=False),
     ("/side_notes/0/dismissable", "const")),
    (lambda r: r["whatif"][0]["drivers"][0]["knots"][0].pop(),
     ("/whatif/0/drivers/0/knots/0", "items")),
    (lambda r: r["charts"][0].update(geo_kind="us_state"), ("/charts/0/geo_kind", "unknown_field")),
    (lambda r: r["maps"][0].pop("k_floor"), ("/maps/0", "missing_field")),
    (lambda r: r["charts"][0].update(alt_text="A chart."), ("/charts/0/alt_text", "length")),
    (lambda r: r.update(steps=[]), ("/steps", "items")),
], ids=["pattern-newline", "date-time", "bool-is-not-int", "minimum", "nan", "decimals-max",
        "unit-enum", "max-bullets", "dismissable-const", "knot-arity", "chart-is-closed",
        "map-requires-geo", "alt-text-min", "steps-min"])
def test_spec_rejects(mutate, expected):
    report = _retail()
    mutate(report)
    assert expected in _codes(spec.validate(report))


def test_validator_refuses_a_schema_keyword_it_does_not_implement():
    """A keyword the walker ignores would be a check that silently does nothing."""
    with pytest.raises(RuntimeError, match="unevaluatedProperties"):
        spec._check_keywords({"type": "object", "unevaluatedProperties": False})
    with pytest.raises(RuntimeError, match="email"):
        spec._check_keywords({"properties": {"a": {"format": "email"}}})


# ═══ versioning ════════════════════════════════════════════════════════════

def test_newer_minor_ignores_unknown_fields_but_keeps_every_other_rule():
    """1.2 adds optional fields: a 1.1 reader validates the rest and drops the new ones."""
    report = _retail()
    report["schema_version"] = "1.2"
    report["glossary"] = [{"term": "sell-through"}]
    report["figures"][0]["precision_note"] = "new in 1.2"
    assert spec.validate(report) == []
    seen = spec.accept(report)
    assert "glossary" not in seen and "precision_note" not in seen["figures"][0]
    assert seen["schema_version"] == "1.2" and report["glossary"]  # input untouched

    report["kind"] = "forecast"
    assert _codes(spec.validate(report)) == {("/kind", "enum")}
    del report["headline"]
    assert ("", "missing_field") in _codes(spec.validate(report))


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_same_unknown_field_is_rejected_at_this_minor_and_below(version):
    """Only a minor newer than this module's is read leniently."""
    report = _retail()
    report["schema_version"] = version
    report["glossary"] = []
    assert _codes(spec.validate(report)) == {("/glossary", "unknown_field")}
    with pytest.raises(spec.SpecError, match="unknown field 'glossary'"):
        spec.accept(report)


# ═══ 1.1: data_checks, provenance.recipe, provenance.engine ════════════════

def _retail_1_1():
    """A 1.1 document: the retail example plus every optional 1.1 field."""
    report = _retail()
    report["schema_version"] = "1.1"
    report["data_checks"] = [
        {"gate": "duplicates", "status": "qualified", "figure": "f1", "caveat_kind": "data_quality"},
        {"gate": "missing_measure", "status": "passed"},
        {"gate": "min_rows", "status": "failed", "figure": "f2", "reason": "insufficient_rows"},
    ]
    report["provenance"]["recipe"] = {"id": "describe", "version": "1.0"}
    report["provenance"]["engine"] = {"duckdb": "1.2.0", "scipy": "1.14.1", "statsmodels": "0.14.4"}
    return report


def test_this_module_writes_1_1():
    assert spec.SCHEMA_VERSION == "1.1"
    assert spec.SCHEMA["$id"].endswith("/1.1")


def test_a_1_1_document_with_data_checks_validates_in_full():
    report = _retail_1_1()
    assert spec.validate(report) == []
    assert spec.unresolved_refs(report) == []
    assert spec.accept(report) == report   # nothing is dropped: every 1.1 field is known


@pytest.mark.parametrize("report", [RETAIL, FINTECH], ids=["retail", "fintech"])
def test_a_1_0_document_validates_unchanged(report):
    """The 1.1 additions are optional: a 1.0 golden is valid and is read as it is."""
    assert report["schema_version"] == "1.0"
    assert spec.validate(report) == []
    assert spec.accept(report) == report


@pytest.mark.parametrize("mutate,expected", [
    (lambda r: r["data_checks"][0].update(note="x"), ("/data_checks/0/note", "unknown_field")),
    (lambda r: r["data_checks"][0].pop("status"), ("/data_checks/0", "missing_field")),
    (lambda r: r["data_checks"][0].update(status="ok"), ("/data_checks/0/status", "enum")),
    (lambda r: r["data_checks"][0].update(gate="Duplicates"), ("/data_checks/0/gate", "pattern")),
    (lambda r: r["data_checks"][0].update(gate="a" * 41), ("/data_checks/0/gate", "length")),
    (lambda r: r["data_checks"][0].update(figure=12.0), ("/data_checks/0/figure", "type")),
    (lambda r: r["data_checks"][0].update(caveat_kind="noise"), ("/data_checks/0/caveat_kind", "enum")),
    (lambda r: r["data_checks"][2].update(reason="too few rows"), ("/data_checks/2/reason", "pattern")),
    (lambda r: r.update(data_checks=[r["data_checks"][1]] * 25), ("/data_checks", "items")),
    (lambda r: r["provenance"]["recipe"].update(version="v1"), ("/provenance/recipe/version", "pattern")),
    (lambda r: r["provenance"]["recipe"].pop("id"), ("/provenance/recipe", "missing_field")),
    (lambda r: r["provenance"]["recipe"].update(title="x"), ("/provenance/recipe/title", "unknown_field")),
    (lambda r: r["provenance"]["engine"].update(DuckDB="1"), ("/provenance/engine/DuckDB", "unknown_field")),
    (lambda r: r["provenance"]["engine"].update(duckdb=1.2), ("/provenance/engine/duckdb", "type")),
    (lambda r: r["provenance"]["engine"].update({f"p{c}": "1" for c in "abcdef"}),
     ("/provenance/engine", "items")),
], ids=["check-closed", "check-status-required", "status-enum", "gate-pattern", "gate-length",
        "figure-is-a-ref", "caveat-enum", "reason-pattern", "max-checks", "recipe-version", "recipe-id-required",
        "recipe-closed", "engine-key-pattern", "engine-value-string", "engine-max-eight"])
def test_the_1_1_additions_are_closed(mutate, expected):
    report = _retail_1_1()
    mutate(report)
    assert expected in _codes(spec.validate(report))


def test_a_data_check_figure_must_resolve():
    """The deciding number of a check is a figure reference like any other."""
    report = _retail_1_1()
    report["data_checks"][0]["figure"] = "f99"
    assert spec.validate(report) == []
    assert _codes(spec.unresolved_refs(report)) == {("/data_checks/0/figure", "unresolved_figure")}


def test_pattern_and_max_properties_are_implemented_keywords():
    """``provenance.engine`` needs them; a walker that skipped them would accept anything."""
    spec._check_keywords({"type": "object", "patternProperties": {"^a$": {"type": "string"}}, "maxProperties": 2})
    with pytest.raises(RuntimeError, match="minProperties"):
        spec._check_keywords({"type": "object", "minProperties": 1})
    with pytest.raises(RuntimeError, match="email"):
        spec._check_keywords({"patternProperties": {"^a$": {"format": "email"}}})


@pytest.mark.parametrize("version,code", [
    ("2.0", "unsupported_version"), ("0.9", "unsupported_version"),
    ("1", "version"), ("v1.0", "version"), ("1.0.0", "version"), (1.0, "version"),
])
def test_other_versions_are_refused_with_a_plain_message(version, code):
    report = _retail()
    report["schema_version"] = version
    issues = spec.validate(report)
    assert [(i.path, i.code) for i in issues] == [("/schema_version", code)]
    if code == "unsupported_version":
        assert "Upgrade Avaloka" in issues[0].message


def test_missing_schema_version_is_a_missing_field():
    report = _retail()
    del report["schema_version"]
    assert _codes(spec.validate(report)) == {("", "missing_field")}


# ═══ figures and placeholders ══════════════════════════════════════════════

def test_unresolved_figure_references_are_found():
    report = _retail()
    report["kpis"][0]["value"] = "f99"
    report["headline"]["text"] = "Net sales grew {f42}."
    report["strategies"][0]["design"]["mde"] = "f77"
    report["figures"].append(dict(report["figures"][0]))
    assert _codes(spec.unresolved_refs(report)) == {
        ("/kpis/0/value", "unresolved_figure"),
        ("/headline/text", "unresolved_figure"),
        ("/strategies/0/design/mde", "unresolved_figure"),
        ("/figures/11/id", "duplicate_figure"),
    }


def test_free_numbers_flags_raw_digits_and_allows_periods_and_years():
    assert spec.free_numbers("Sales grew {f1} in Q3 2026 and FY25, H1 too.") == []
    assert spec.free_numbers("Top 10 stores hold {f2}.") == ["10"]
    assert spec.free_numbers("Up 4.7% on 1,200 orders") == ["4.7", "1,200"]


@pytest.mark.parametrize("report", [RETAIL, FINTECH], ids=["retail", "fintech"])
def test_golden_examples_carry_no_raw_numbers_in_text_slots(report):
    """§3.2 rule 1 over the fixtures build and verify use: every number is a figure.

    (§4.1's Example B wrote "30+" and "top 10" as raw digits; here they are f6, f7.)
    """
    flagged = {p: spec.free_numbers(t) for p, t in spec.text_slots(report)
               if spec.free_numbers(t)}
    assert flagged == {}


def test_text_slots_are_the_ones_section_3_2_checks():
    report = _retail()
    report["maps"][0]["caption"] = "West leads at {f5}."
    pointers = [p for p, _ in spec.text_slots(report)]
    assert pointers == ["/headline/text",
                        "/headline/bullets/0/so_what", "/headline/bullets/0/now_what",
                        "/headline/bullets/1/so_what", "/headline/bullets/1/now_what",
                        "/maps/0/caption", "/strategies/0/action", "/strategies/0/text"]


def test_resolve_pointer():
    assert spec.resolve_pointer(FINTECH["evidence"], "/kpi/dq30/rate/value") == 0.047
    assert spec.resolve_pointer({"a/b": {"~": [5]}}, "/a~1b/~0/0") == 5
    for bad in ("/kpi/dq30/nope", "kpi", "/kpi/dq30/rate/value/0"):
        with pytest.raises(KeyError):
            spec.resolve_pointer(FINTECH["evidence"], bad)


# ═══ S2 — the formatter, against the shared golden vectors ═════════════════

def test_golden_format_vectors_cover_what_s2_requires():
    """At least 60 cases, every unit, every suffix, negatives, signs and currency codes."""
    cases = FORMAT["format"]
    ok = [c for c in cases if "expected" in c]
    assert len(cases) >= 60
    assert {c["figure"]["unit"] for c in ok} == set(fmt.UNITS)
    assert all(any(c["expected"].endswith(s) for c in ok) for s in "KMBT")
    assert any(c["expected"].startswith("-") for c in ok)
    assert any(c.get("signed") for c in ok)
    assert {c["figure"].get("currency") for c in ok} >= {"USD", "EUR", "CHF", None}
    assert {c["error"] for c in cases if "error" in c} == \
        {"unit", "decimals", "currency", "value", "compact"}
    assert {c["figure"].get("compact") for c in ok} == {None, True, False}
    by_value = {(c["figure"]["value"], c["figure"].get("compact")): c["expected"] for c in ok}
    assert by_value[(999950, None)] == "1M" and by_value[(999949, None)] == "999.9K"


@pytest.mark.parametrize("case", FORMAT["format"], ids=lambda c: c["name"])
def test_formatter_matches_golden(case):
    f = case["figure"]
    call = lambda: fmt.format_number(f["value"], f["unit"], f["decimals"], f.get("currency"),  # noqa: E731
                                     signed=case.get("signed", False), compact=f.get("compact"))
    if "error" in case:
        with pytest.raises(fmt.FormatError) as err:
            call()
        assert err.value.code == case["error"]
    else:
        assert call() == case["expected"]


@pytest.mark.parametrize("case", FORMAT["round"], ids=lambda c: f"{c['value']}@{c['decimals']}")
def test_round_half_even_matches_golden(case):
    assert fmt.round_half_even(case["value"], case["decimals"], shift=case["shift"]) == \
        case["expected"]


@pytest.mark.parametrize("case", FORMAT["substitute"]["cases"], ids=lambda c: c["name"])
def test_render_text_matches_golden(case):
    figures = FORMAT["substitute"]["figures"]
    if "error" in case:
        with pytest.raises(fmt.FormatError) as err:
            fmt.render_text(case["text"], figures)
        assert err.value.code == case["error"]
    else:
        assert fmt.render_text(case["text"], figures) == case["expected"]


def test_rounding_is_half_even_on_the_printed_decimal():
    """Not Python's round(), which works on the binary value."""
    assert round(2.675, 2) == 2.67
    assert fmt.round_half_even(2.675, 2) == "2.68"
    assert [fmt.round_half_even(v, 0) for v in (0.5, 1.5, 2.5, -2.5)] == ["0", "2", "2", "-2"]
    assert fmt.format_number(12450, "count", 0) == "12.4K"


# ═══ S3 — a figure against its evidence (§3.2 rule 3) ══════════════════════

def test_figure_4_6_accepted_for_evidence_0_0462():
    shift = fmt.evidence_shift("percent", "fraction")
    assert shift == 2
    assert fmt.figure_matches(4.6, 1, 0.0462, shift=shift)


def test_figure_4_7_rejected_for_evidence_0_0462():
    """§3.5 V-b's bite: compare with a 5 % tolerance instead of equal-after-rounding → red."""
    assert not fmt.figure_matches(4.7, 1, 0.0462, shift=fmt.evidence_shift("percent", "fraction"))


def test_percent_matches_fraction_evidence_only_when_the_unit_says_so():
    """The evidence leaf's own unit decides the scale: 8.4 % is 0.084 only as a fraction."""
    assert fmt.figure_matches(8.4, 1, {"value": 0.084, "unit": "fraction"}, figure_unit="percent")
    assert not fmt.figure_matches(8.4, 1, 0.084, figure_unit="percent")
    assert not fmt.figure_matches(8.4, 1, {"value": 0.084, "unit": "percent"},
                                  figure_unit="percent")
    assert not fmt.figure_matches(8.4, 1, {"value": 0.084, "unit": "currency"},
                                  figure_unit="percent")
    figure = {"id": "f1", "value": 8.4, "unit": "percent", "decimals": 1,
              "evidence_ref": "/kpi/yoy"}
    assert fmt.figure_matches_evidence(figure, {"kpi": {"yoy": {"value": 0.084,
                                                                 "unit": "fraction"}}})
    assert not fmt.figure_matches_evidence(figure, {"kpi": {"yoy": 0.084}})
    with pytest.raises(fmt.FormatError):
        fmt.figure_matches(8.4, 1, {"value": 0.084, "unit": "fraction"})


@pytest.mark.parametrize("report", [RETAIL, FINTECH], ids=["retail", "fintech"])
def test_every_golden_figure_matches_its_evidence(report):
    """The fixtures are verifiable: §3.2 rule 3 holds for every figure of both examples."""
    assert [f["id"] for f in report["figures"]
            if not fmt.figure_matches_evidence(f, report["evidence"])] == []


def test_evidence_leaves_are_validated():
    report = _retail()
    report["evidence"]["kpi"]["net_sales"]["yoy_pct"]["unit"] = "fractions"
    report["evidence"]["bridge"]["note"] = {"value": "1.9", "unit": "currency"}
    report["evidence"]["bridge"]["mix"] = {"value": -0.1, "unit": "currency", "src": "x"}
    assert _codes(spec.validate(report)) == {
        ("/evidence/kpi/net_sales/yoy_pct/unit", "enum"),
        ("/evidence/bridge/note/value", "type"),
        ("/evidence/bridge/mix/src", "unknown_field"),
    }
    # A key that happens to be called "unit" with a number is data, not a leaf.
    report = _retail()
    report["evidence"]["bridge"]["unit"] = 3
    assert spec.validate(report) == []
    assert spec.evidence_leaf(RETAIL["evidence"], "/kpi/net_sales/yoy_pct") == \
        (0.084, "fraction", None)
    assert spec.evidence_leaf(RETAIL["evidence"], "/bridge/price") == (1900000, None, None)
    with pytest.raises(ValueError):
        spec.evidence_leaf(RETAIL["evidence"], "/bridge")


def test_a_figure_stores_the_real_magnitude():
    """$1.9M is stored as 1900000: it matches that evidence, never a "1.9 in millions"."""
    assert fmt.figure_matches(1900000, 0, 1900000)
    assert not fmt.figure_matches(1900000, 0, 1.9)
    assert not fmt.figure_matches(1.9, 0, 1900000)
    assert fmt.format_number(1900000, "currency", 0, "USD") == "$1.9M"
    assert fmt.format_number(1900000, "currency", 0, "USD", compact=False) == "$1,900,000"


def test_retail_bridge_renders_at_its_real_magnitude():
    """Example A's bridge is dollars; the formatter, not the data, writes the M."""
    bullet = RETAIL["headline"]["bullets"][0]["so_what"]
    assert fmt.render_text(bullet, RETAIL["figures"]) == \
        "Price added $1.9M of the $2.6M rise; volume added $700K."
    assert RETAIL["evidence"]["bridge"] == {"price": 1900000, "volume": 700000, "total": 2600000}


def test_evidence_units_must_agree():
    assert fmt.evidence_shift("pp", "pp") == 0
    assert fmt.evidence_shift("currency", None) == 0
    assert fmt.evidence_shift("percent", "currency") is None
    assert fmt.evidence_shift("pp", "fraction") is None


# ═══ §2.2 — the scenario model, against the shared golden vectors ══════════

def _linear_whatif():
    return {"id": "w1", "kpi": "net_sales", "y0": 100.0, "z": 1.96,
            "evidence_kind": "associational", "evidence": {"turn_id": "t1"},
            "drivers": [{"id": "discount", "label": "Discount", "unit": "percent",
                         "baseline": 12, "lo": 5, "hi": 25, "form": "linear",
                         "beta": 0.8, "se": 0.1}]}


@pytest.mark.parametrize("case", SCENARIO["cases"], ids=lambda c: c["name"])
def test_scenario_matches_golden(case):
    call = lambda: scenario.evaluate(case["whatif"], case["values"],  # noqa: E731
                                     conformal_q=case.get("conformal_q"))
    if "error" in case:
        with pytest.raises(scenario.ScenarioError) as err:
            call()
        assert err.value.code == case["error"]
    else:
        assert call() == case["expected"]


def test_sure_text_matches_golden():
    assert scenario.SURE_TEXT == SCENARIO["sure_text"]


def test_requested_above_p95_is_clamped():
    """§3.5 V-f's bite: remove the clamp from scenario.py → red (vitest twin in the UI)."""
    out = scenario.evaluate(_linear_whatif(), {"discount": 40})
    assert out["applied"] == {"discount": 25}
    assert out["clamped"] == [{"driver": "discount", "requested": 40, "applied": 25,
                               "bound": "hi"}]
    assert out["result"]["y"] == pytest.approx(100 + 0.8 * 13)
    # Unclamped at the same applied value it would be fairly sure; the clamp costs a level.
    assert scenario.evaluate(_linear_whatif(), {"discount": 25})["result"]["sure"] == "fairly_sure"
    assert out["result"]["sure"] == "direction_only"


def test_scenario_is_additive_with_an_rss_band():
    whatif = _linear_whatif()
    whatif["drivers"].append({"id": "price", "label": "Price", "unit": "currency",
                              "baseline": 20, "lo": 10, "hi": 30, "form": "linear",
                              "beta": -2.0, "se": 0.5})
    out = scenario.evaluate(whatif, {"discount": 20, "price": 23})
    assert out["result"]["y"] == pytest.approx(100 + 0.8 * 8 - 2.0 * 3)
    assert out["sigma"] == pytest.approx((0.8 ** 2 + 1.5 ** 2) ** 0.5)
    assert out["result"]["hi"] - out["result"]["y"] == pytest.approx(1.96 * out["sigma"])


def test_scenario_block_is_a_valid_scenario_section():
    block = scenario.scenario_block(_linear_whatif(), {"discount": 1}, base_version=2,
                                    name="Deeper discount")
    assert block["clamped"][0]["bound"] == "lo" and block["values"] == {"discount": 1}
    report = _retail()
    report.update(kind="what_if", scenario=block)
    assert spec.validate(report) == []


# ═══ packaging and isolation ═══════════════════════════════════════════════

def test_schema_and_runtime_ship_in_the_wheel():
    package_data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"][
        "package-data"]["avaloka"]
    for pattern in ("outcomes/*.schema.json", "outcomes/assets/*.js"):
        assert pattern in package_data
        assert list((REPO / "avaloka").glob(pattern)), pattern


def test_outcomes_import_only_the_standard_library():
    """S1.08 for this package, and stricter: no app.*, and no pandas or numpy either."""
    probe = ("import sys, avaloka.outcomes; "
             "print(sorted({m.split('.')[0] for m in sys.modules} & {'app', 'pandas', 'numpy'}))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
