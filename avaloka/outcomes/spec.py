"""The Outcome Report spec, v1.1: schema, validator and figure helpers.

Every renderer (HTML export, Markdown, chat summary, the UI frame, the SDKs) is
a pure function of one JSON document, the Outcome Report. This module holds its
schema (docs/design/2.0-OUTCOMES.md §4.1, with the 1.1 additions of
docs/design/1.7-INSTANT-ANALYSIS.md §4.3) as a Python dict, publishes it as
``outcome-report.schema.json`` beside this file, and validates documents
against it.

Why a hand-written validator. ``jsonschema`` is not one of the eight runtime
dependencies S1.09 fixes, so the validator below interprets exactly the subset
of JSON Schema 2020-12 the schema uses. The subset is closed: a keyword the
walker does not implement fails at import (``_check_keywords``), so a keyword
added to the schema can never be silently ignored.

Versioning. ``schema_version`` is ``"<major>.<minor>"``; this module writes
``1.1``, which added three optional fields to ``1.0`` (``data_checks[]``,
``provenance.recipe``, ``provenance.engine``).

- ``1.0`` and ``1.1`` are validated strictly against this schema: an unknown
  field anywhere is rejected. A ``1.0`` document is a ``1.1`` document with
  none of the additions, so it validates unchanged.
- ``1.N`` with N > 1 was written by a newer Avaloka that added optional fields.
  Known fields are still validated in full; unknown fields are ignored, and
  :func:`accept` returns a copy without them, which is what a renderer reads.
- Any other major is refused with a plain message, never half-read.

Text slots carry ``{fN}`` placeholders, never raw numbers; ``figures[]`` holds
the numbers, each with the JSON Pointer of the evidence it came from.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = "1.1"
SCHEMA_PATH = Path(__file__).with_name("outcome-report.schema.json")

_EVIDENCE_KINDS = ["associational", "experimental", "causal_estimated"]
_DOMAINS = ["healthcare", "finance", "personal_finance", "stock_market", "retail",
            "telecom", "genomics", "robotics", "astronomy", "manufacturing", "hr",
            "education", "logistics", "marketing", "public_sector", "generic"]
_TEMPLATES = ["fintech_credit", "adtech_performance", "marketing_performance",
              "retail_performance", "generic"]
_VERIFIER_RULES = ["schema", "free_number", "number_word", "unresolved_figure",
                   "value_mismatch", "unit_mismatch", "range_order", "causal_wording",
                   "future_certainty", "length", "chart_shape", "geo_unresolved",
                   "whatif_bounds", "export_size", "export_sandbox", "telemetry_names"]
_CAVEAT_KINDS = ["sampling", "associational", "additive_model", "independence",
                 "extrapolation", "clamped", "small_n", "suppressed_geo",
                 "missing_template_data", "currency_unstated", "data_quality", "other"]
_CHART_KINDS = ["line", "area", "bar", "stacked_bar", "pie", "histogram", "boxplot",
                "scatter", "heatmap", "funnel", "sankey", "waterfall", "cohort_heatmap",
                "vintage_lines", "choropleth", "point_map", "kpi_spark"]
_QUESTION_TYPES = ["trend", "composition", "distribution", "relationship", "funnel",
                   "cohort", "geo", "bridge", "flow", "ranking"]

_FIGURE_UNITS = ["count", "currency", "percent", "pp", "ratio", "days", "unitless"]
# An evidence leaf may also be stored as a fraction (0.084), which a percent
# figure reads at 100x (8.4); §3.2 rule 3.
EVIDENCE_UNITS = _FIGURE_UNITS + ["fraction"]

_REF_ID = {"$ref": "#/$defs/id"}
_REF_FIG = {"$ref": "#/$defs/fref"}
_REF_TRACE = {"$ref": "#/$defs/trace_ref"}
_STR = {"type": "string"}
# 1.1: snake_case names from a closed vocabulary (a recipe's gates, its id, an
# engine's package name), never free text.
_SNAKE = {"type": "string", "pattern": "^[a-z_]+$"}


def _closed(required: list[str], properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": required,
            "properties": properties, **extra}


def _array(items: dict[str, Any], max_items: int, **extra: Any) -> dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": items, **extra}


# §4.1 has `maps[]` as `allOf: [chart]` plus geo properties. With a closed
# chart that combination rejects the geo fields, and with an open chart every
# chart accepts any key; draft 2020-12 needs `unevaluatedProperties` to have
# both. Composing the closed map definition here keeps chart and map closed and
# in one place.
_CHART_REQUIRED = ["id", "title", "chart_kind", "question_type", "option", "alt_text",
                   "data_table"]
_CHART_PROPERTIES: dict[str, Any] = {
    "id": _REF_ID,
    "title": {"type": "string", "maxLength": 80},
    "caption": {"type": "string", "maxLength": 200},
    "question_type": {"enum": _QUESTION_TYPES},
    "chart_kind": {"enum": _CHART_KINDS},
    "option": {"type": "object", "x-inert": True,
               "description": "ECharts option. JSON only: no functions; no key or string "
                              "value may contain 'function', '=>' or '<'."},
    "alt_text": {"type": "string", "minLength": 20, "maxLength": 400},
    "data_table": _closed(["columns", "rows"], {
        "columns": _array(_STR, 12),
        "rows": _array({"type": "array"}, 500)}),
    "evidence": _REF_TRACE,
}
_MAP_PROPERTIES: dict[str, Any] = {
    **_CHART_PROPERTIES,
    "geo_kind": {"enum": ["country", "us_state", "latlon", "geohash", "h3", "us_zip"]},
    "asset": {"enum": ["world_110m", "us_states_20m", "none"]},
    "suppressed_regions": {"type": "integer", "minimum": 0},
    "k_floor": {"type": "integer", "minimum": 1},
}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://avaloka.ai/schemas/outcome-report/1.1",
    "title": "Avaloka Outcome Report",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "report_id", "kind", "domain", "template", "created_at",
                 "analysis", "headline", "figures", "kpis", "charts", "method", "caveats",
                 "steps", "completion", "evidence", "provenance"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "report_id": {"type": "string", "pattern": "^rpt_[0-9a-f]{16}$"},
        "kind": {"enum": ["analysis", "what_if", "model", "mission"]},
        "domain": {"enum": _DOMAINS},
        "template": {"enum": _TEMPLATES},
        "audience": {"enum": ["executive", "analyst"]},
        "created_at": {"type": "string", "format": "date-time"},
        "analysis": _closed(["analysis_id", "thread_id", "turn_id", "version"], {
            "analysis_id": _STR, "thread_id": _STR, "turn_id": _STR,
            "mission_id": {"type": ["string", "null"]},
            "version": {"type": "integer", "minimum": 1},
            "parent_version": {"type": ["integer", "null"], "minimum": 1},
            "dataset_ids": _array(_STR, 16)}),
        "headline": _closed(["text", "bullets"], {
            "text": {"$ref": "#/$defs/slot160"},
            "bullets": _array(_closed(["so_what", "now_what"], {
                "so_what": {"$ref": "#/$defs/slot140"},
                "now_what": {"$ref": "#/$defs/slot140"}}), 3)}),
        "figures": _array({"$ref": "#/$defs/figure"}, 400),
        "kpis": _array(_closed(["id", "label", "value", "direction_good"], {
            "id": _REF_ID, "label": {"type": "string", "maxLength": 40},
            "value": _REF_FIG, "baseline": _REF_FIG, "delta": _REF_FIG,
            "delta_kind": {"enum": ["abs", "pct", "pp"]},
            "baseline_label": {"type": "string", "maxLength": 40},
            "direction_good": {"enum": ["up", "down", "neutral"]},
            "spark_chart": _REF_ID}), 8),
        "charts": _array({"$ref": "#/$defs/chart"}, 12),
        "maps": _array({"$ref": "#/$defs/map"}, 4),
        "whatif": _array({"$ref": "#/$defs/whatif"}, 3),
        "strategies": _array({"$ref": "#/$defs/strategy"}, 3),
        "scenario": {"$ref": "#/$defs/scenario"},
        "steps": _array(_closed(["id", "order", "title", "status"], {
            "id": _REF_ID, "order": {"type": "integer", "minimum": 1},
            "title": {"type": "string", "maxLength": 80},
            "status": {"enum": ["done", "needs_data", "needs_decision", "next"]},
            "detail": {"type": "string", "maxLength": 200},
            "needs": {"type": "string", "maxLength": 120},
            "evidence": _REF_TRACE}), 12, minItems=1),
        "completion": _closed(["state"], {
            "state": {"enum": ["answered", "partial", "open"]},
            "missing": _array({"type": "string", "maxLength": 140}, 6),
            "open_questions": _array({"type": "string", "maxLength": 140}, 4)}),
        "follow_ups": _array(_closed(["id", "label", "prompt"], {
            "id": _REF_ID, "label": {"type": "string", "maxLength": 40},
            "prompt": {"type": "string", "maxLength": 200}, "step": _REF_ID}), 4),
        "join_proposals": _array(_closed(
            ["id", "left", "right", "keys", "match_rate", "rows_left", "rows_after",
             "fanout", "verdict", "status"], {
                "id": _REF_ID, "left": _STR, "right": _STR,
                "keys": _array(_array(_STR, 2, minItems=2), 3),
                "match_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "sample_rows": {"type": "integer"}, "rows_left": {"type": "integer"},
                "rows_right": {"type": "integer"}, "rows_after": {"type": "integer"},
                "fanout": {"type": "number", "minimum": 0},
                "verdict": {"enum": ["safe", "low_overlap", "fanout", "type_mismatch"]},
                "status": {"enum": ["proposed", "accepted", "declined"]}}), 3),
        "learned": _array({"type": "string", "maxLength": 200}, 5),
        "side_notes": _array(_closed(["id", "what", "why", "dismissable"], {
            "id": {"type": "string", "pattern": "^sn_[a-z0-9_]{1,40}$"},
            "what": {"type": "string", "maxLength": 120},
            "why": {"type": "string", "maxLength": 200},
            "connector": {"type": ["string", "null"]},
            "dismissable": {"const": True}}), 3),
        "method": _closed(["summary", "sample"], {
            "summary": {"$ref": "#/$defs/slot400"},
            "steps": _array({"type": "string", "maxLength": 200}, 12),
            "model": {"type": ["object", "null"], "additionalProperties": False, "properties": {
                "family": _STR, "metric": _STR, "score": _REF_FIG, "baseline_score": _REF_FIG,
                "beats_baseline": {"type": "boolean"}}},
            "sample": _closed(["rows_analyzed", "total_rows", "is_sampled"], {
                "rows_analyzed": {"type": ["integer", "null"]},
                "total_rows": {"type": ["integer", "null"]},
                "is_sampled": {"type": "boolean"}, "mode": _STR})}),
        "caveats": _array(_closed(["kind", "text"], {
            "kind": {"enum": _CAVEAT_KINDS},
            "text": {"type": "string", "maxLength": 300}}), 12),
        # 1.1: the recipe's GateReport, one entry per quality gate. The deciding
        # number is a figure, so it is verified against the evidence like any other.
        "data_checks": _array(_closed(["gate", "status"], {
            "gate": {**_SNAKE, "maxLength": 40},
            "status": {"enum": ["passed", "qualified", "failed"]},
            "figure": _REF_FIG,
            "caveat_kind": {"enum": _CAVEAT_KINDS},
            "reason": _SNAKE}), 24),
        # The one deliberately open object: the numeric snapshot figures point into.
        # Its leaves are bare numbers (in the unit of the figure that cites them)
        # or {"value", "unit"[, "currency"]} objects; any object in it whose
        # "unit" is a string must be such a leaf (x-evidence).
        "evidence": {"type": "object", "x-evidence": True,
                     "description": "Numeric snapshot the figures point into; values copied "
                                    "from persisted analysis artifacts. A leaf is a number in "
                                    "the citing figure's unit, or an evidence_leaf."},
        "provenance": _closed(["generator", "generator_version", "llm_used", "verifier"], {
            "generator": {"const": "avaloka"}, "generator_version": _STR,
            "llm_used": {"type": "boolean"}, "llm_model": {"type": ["string", "null"]},
            # 1.1: which recipe wrote the report and the engines it computed with.
            "recipe": _closed(["id", "version"], {
                "id": _SNAKE, "version": {"type": "string", "pattern": "^\\d+\\.\\d+$"}}),
            "engine": {"type": "object", "additionalProperties": False, "maxProperties": 8,
                       "patternProperties": {"^[a-z_]+$": _STR}},
            "verifier": _closed(["passed", "rejections"], {
                "passed": {"type": "boolean"},
                "rejections": {"type": "array", "items": _closed(["slot", "rule"], {
                    "slot": _STR, "rule": {"enum": _VERIFIER_RULES},
                    "action": {"enum": ["repaired", "dropped", "labelled"]}})},
                "attempts": {"type": "integer", "minimum": 1, "maximum": 3}})}),
    },
    "$defs": {
        "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"},
        "fref": {"type": "string", "pattern": "^f[0-9]{1,3}$"},
        "slot140": {"type": "string", "maxLength": 140},
        "slot160": {"type": "string", "maxLength": 160},
        "slot400": {"type": "string", "maxLength": 400},
        "figure": _closed(["id", "value", "unit", "decimals", "evidence_ref"], {
            "id": _REF_FIG, "value": {"type": "number"},
            "unit": {"enum": _FIGURE_UNITS},
            "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "decimals": {"type": "integer", "minimum": 0, "maximum": 4},
            "compact": {"type": "boolean"},
            "evidence_ref": {"type": "string", "pattern": "^/"}}),
        "evidence_leaf": _closed(["value", "unit"], {
            "value": {"type": "number"}, "unit": {"enum": EVIDENCE_UNITS},
            "currency": {"type": "string", "pattern": "^[A-Z]{3}$"}}),
        "trace_ref": _closed(["turn_id"], {
            "turn_id": _STR, "step_no": {"type": "integer"}, "agent": _STR, "artifact": _STR}),
        "chart": _closed(_CHART_REQUIRED, _CHART_PROPERTIES),
        "map": _closed(_CHART_REQUIRED + ["geo_kind", "asset", "suppressed_regions", "k_floor"],
                       _MAP_PROPERTIES),
        "driver": _closed(["id", "label", "unit", "baseline", "lo", "hi", "form"], {
            "id": _REF_ID, "label": {"type": "string", "maxLength": 40},
            "unit": {"enum": ["count", "currency", "percent", "ratio", "days", "unitless"]},
            "baseline": {"type": "number"}, "lo": {"type": "number"}, "hi": {"type": "number"},
            "step": {"type": "number", "exclusiveMinimum": 0},
            "form": {"enum": ["linear", "piecewise", "elasticity", "step"]},
            "beta": {"type": "number"}, "se": {"type": "number", "minimum": 0},
            "elasticity": {"type": "number"},
            "knots": _array({"type": "array", "minItems": 4, "maxItems": 4,
                             "prefixItems": [{"type": "number"}] * 4,
                             "description": "[x, g, g_lo, g_hi], x strictly increasing"},
                            24, minItems=2),
            "levels": _array(_STR, 2)}),
        "whatif": _closed(["id", "kpi", "y0", "drivers", "evidence_kind", "evidence"], {
            "id": _REF_ID, "kpi": _REF_ID, "y0": {"type": "number"}, "z": {"const": 1.96},
            "conformal_q": {"type": ["number", "null"], "minimum": 0},
            "drivers": _array({"$ref": "#/$defs/driver"}, 5, minItems=1),
            "evidence_kind": {"enum": _EVIDENCE_KINDS},
            "evidence": _REF_TRACE}),
        "strategy": _closed(["id", "rank", "action", "text", "impact", "confidence",
                             "evidence_kind", "evidence", "verified"], {
            "id": _REF_ID, "rank": {"type": "integer", "minimum": 1, "maximum": 3},
            "tier": {"enum": ["immediate", "intermediate", "complex"]},
            "downgraded_from": {"enum": ["immediate", "intermediate"]},
            "design": _closed(["type", "metric"], {
                "type": {"enum": ["ab_test", "holdout", "threshold_rule", "staged", "data_needed"]},
                "metric": {"enum": ["proportion", "mean", "rmse", "auc", "uplift"]},
                "alpha": {"type": "number"}, "power": {"type": "number"},
                "mde": _REF_FIG, "n_required": _REF_FIG, "n_available": _REF_FIG,
                "duration_days": {"type": "integer"},
                "cv": {"enum": ["kfold5", "stratified5", "timeseries5", "none"]},
                "reason": {"type": "string", "maxLength": 200}}),
            "step": _REF_ID, "side_note": _STR,
            "action": {"$ref": "#/$defs/slot140"}, "text": {"$ref": "#/$defs/slot400"},
            "impact": _closed(["kpi", "low", "mid", "high"], {
                "kpi": _REF_ID, "low": _REF_FIG, "mid": _REF_FIG, "high": _REF_FIG}),
            "confidence": {"enum": ["high", "medium", "low"]},
            "score": {"type": "number", "minimum": 0},
            "evidence_kind": {"enum": _EVIDENCE_KINDS},
            "evidence": _REF_TRACE, "whatif": _REF_ID,
            "verified": {"type": "boolean"}}),
        "scenario": _closed(["whatif", "base_version", "values", "applied", "clamped",
                             "evidence_kind", "result"], {
            "name": {"type": "string", "maxLength": 60}, "whatif": _REF_ID,
            "base_version": {"type": "integer", "minimum": 1},
            "values": {"type": "object", "additionalProperties": {"type": "number"}},
            "applied": {"type": "object", "additionalProperties": {"type": "number"}},
            "clamped": {"type": "array", "items": _closed(
                ["driver", "requested", "applied", "bound"], {
                    "driver": _STR, "requested": {"type": "number"},
                    "applied": {"type": "number"}, "bound": {"enum": ["lo", "hi"]}})},
            "evidence_kind": {"enum": _EVIDENCE_KINDS},
            "result": _closed(["y", "lo", "hi", "sure"], {
                "y": {"type": "number"}, "lo": {"type": "number"}, "hi": {"type": "number"},
                "sure": {"enum": ["fairly_sure", "direction_only", "either_way"]}})}),
    },
}

# ── the validator ────────────────────────────────────────────────────────────

_ANNOTATIONS = {"$schema", "$id", "title", "description", "$comment", "$defs"}
_ASSERTIONS = {"type", "enum", "const", "$ref", "properties", "required",
               "additionalProperties", "items", "prefixItems", "minItems", "maxItems",
               "minLength", "maxLength", "pattern", "format", "minimum", "maximum",
               "exclusiveMinimum", "patternProperties", "maxProperties", "x-inert", "x-evidence"}
_FORMATS = {"date-time"}

# RFC 3339 date-time, the meaning of JSON Schema's "date-time" format.
_DATE_TIME = re.compile(
    r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])[Tt]([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)"
    r"(\.\d+)?([Zz]|[+-]([01]\d|2[0-3]):[0-5]\d)\Z")
_CODE_MARKERS = ("function", "=>", "<")
_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")


@dataclass(frozen=True)
class Issue:
    """One reason a document is not a valid Outcome Report."""

    path: str  # JSON Pointer into the document; "" is the root
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message} [{self.code}]"


class SpecError(ValueError):
    """Raised by :func:`assert_valid` and :func:`accept`; carries every issue."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        shown = "; ".join(str(i) for i in issues[:5])
        more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        super().__init__(f"not a valid Outcome Report: {shown}{more}")


def _check_keywords(schema: Any, where: str = "#") -> None:
    if isinstance(schema, dict):
        for key, sub in schema.items():
            if key not in _ANNOTATIONS | _ASSERTIONS:
                raise RuntimeError(f"schema keyword {key!r} at {where} is not implemented "
                                   "by avaloka.outcomes.spec; implement it or remove it")
            if key == "format" and sub not in _FORMATS:
                raise RuntimeError(f"schema format {sub!r} at {where} is not implemented")
            if key in ("properties", "patternProperties", "$defs"):
                for name, child in sub.items():
                    _check_keywords(child, f"{where}/{key}/{name}")
            elif key in ("items", "additionalProperties") and isinstance(sub, dict):
                _check_keywords(sub, f"{where}/{key}")
            elif key == "prefixItems":
                for i, child in enumerate(sub):
                    _check_keywords(child, f"{where}/{key}/{i}")


_check_keywords(SCHEMA)


def _python_pattern(pattern: str) -> re.Pattern[str]:
    # ECMA-262 `$` (what JSON Schema means) does not match before a trailing
    # newline; Python's does. `\Z` is the ECMA meaning.
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + r"\Z"
    return re.compile(pattern)


_PATTERNS: dict[str, re.Pattern[str]] = {}


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_ok(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return _is_number(value) and (isinstance(value, int) or value.is_integer())
    raise RuntimeError(f"unknown JSON type {name!r}")


def _json_equal(a: Any, b: Any) -> bool:
    # JSON equality: true is not 1, 1.0 is 1.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return a == b


class _Walk:
    def __init__(self, lenient: bool):
        self.lenient = lenient
        self.issues: list[Issue] = []
        self.ignored: list[tuple[Any, ...]] = []
        self.frefs: list[tuple[str, str]] = []

    def add(self, path: tuple[Any, ...], code: str, message: str) -> None:
        self.issues.append(Issue(_pointer(path), code, message))

    def json_only(self, value: Any, path: tuple[Any, ...]) -> bool:
        """Reject what JSON cannot carry: a callable, a set, NaN, a non-string key."""
        if value is None or isinstance(value, (str, bool, int)):
            return True
        if isinstance(value, float):
            if math.isfinite(value):
                return True
            self.add(path, "not_json", "number is not finite")
            return False
        if isinstance(value, list):
            return all([self.json_only(v, path + (i,)) for i, v in enumerate(value)])
        if isinstance(value, dict):
            ok = True
            for k, v in value.items():
                if not isinstance(k, str):
                    self.add(path, "not_json", f"object key {k!r} is not a string")
                    ok = False
                else:
                    ok = self.json_only(v, path + (k,)) and ok
            return ok
        self.add(path, "not_json", f"{type(value).__name__} is not a JSON value")
        return False

    def inert(self, value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, str):
            lowered = value.lower()
            hit = next((m for m in _CODE_MARKERS if m in lowered), None)
            if hit:
                self.add(path, "code_in_option", f"option text contains {hit!r}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.inert(v, path + (i,))
        elif isinstance(value, dict):
            for k, v in value.items():
                self.inert(k, path + (k,))
                self.inert(v, path + (k,))

    def node(self, value: Any, schema: dict[str, Any], path: tuple[Any, ...]) -> None:
        ref = schema.get("$ref")
        if ref is not None:
            if ref == "#/$defs/fref" and isinstance(value, str):
                self.frefs.append((_pointer(path), value))
            name = ref.removeprefix("#/$defs/")
            self.node(value, SCHEMA["$defs"][name], path)

        types = schema.get("type")
        if types is not None:
            names = [types] if isinstance(types, str) else types
            if not any(_type_ok(value, n) for n in names):
                self.add(path, "type", f"expected {' or '.join(names)}, got "
                                       f"{_json_type(value)}")
                return

        if "const" in schema and not _json_equal(value, schema["const"]):
            self.add(path, "const", f"must be {schema['const']!r}")
        if "enum" in schema and not any(_json_equal(value, e) for e in schema["enum"]):
            self.add(path, "enum", f"{value!r} is not one of {schema['enum']}")

        if isinstance(value, str):
            self.string(value, schema, path)
        elif _is_number(value):
            self.number(value, schema, path)
        elif isinstance(value, list):
            self.array(value, schema, path)
        elif isinstance(value, dict):
            self.object(value, schema, path)

        if schema.get("x-inert"):
            self.inert(value, path)
        if schema.get("x-evidence"):
            self.evidence(value, path)

    def evidence(self, value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("unit"), str):
                self.node(value, SCHEMA["$defs"]["evidence_leaf"], path)
                return
            for k, v in value.items():
                self.evidence(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.evidence(v, path + (i,))

    def string(self, value: str, schema: dict[str, Any], path: tuple[Any, ...]) -> None:
        if "minLength" in schema and len(value) < schema["minLength"]:
            self.add(path, "length", f"shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            self.add(path, "length", f"longer than {schema['maxLength']} characters")
        if "pattern" in schema:
            pattern = _PATTERNS.setdefault(schema["pattern"], _python_pattern(schema["pattern"]))
            if not pattern.search(value):
                self.add(path, "pattern", f"{value!r} does not match {schema['pattern']}")
        if schema.get("format") == "date-time" and not _DATE_TIME.match(value):
            self.add(path, "format", f"{value!r} is not an RFC 3339 date-time")

    def number(self, value: float, schema: dict[str, Any], path: tuple[Any, ...]) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            self.add(path, "range", f"{value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            self.add(path, "range", f"{value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            self.add(path, "range", f"{value} must be above {schema['exclusiveMinimum']}")

    def array(self, value: list[Any], schema: dict[str, Any], path: tuple[Any, ...]) -> None:
        if "minItems" in schema and len(value) < schema["minItems"]:
            self.add(path, "items", f"fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            self.add(path, "items", f"more than {schema['maxItems']} items")
        prefix = schema.get("prefixItems", [])
        for i, item in enumerate(value):
            if i < len(prefix):
                self.node(item, prefix[i], path + (i,))
            elif "items" in schema:
                self.node(item, schema["items"], path + (i,))

    def object(self, value: dict[str, Any], schema: dict[str, Any], path: tuple[Any, ...]) -> None:
        for name in schema.get("required", []):
            if name not in value:
                self.add(path, "missing_field", f"required field {name!r} is missing")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            self.add(path, "items", f"more than {schema['maxProperties']} fields")
        properties = schema.get("properties", {})
        patterns = [(_PATTERNS.setdefault(p, _python_pattern(p)), sub)
                    for p, sub in schema.get("patternProperties", {}).items()]
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            matched = [sub for pattern, sub in patterns if pattern.search(name)]
            for sub in matched:
                self.node(item, sub, path + (name,))
            if name in properties:
                self.node(item, properties[name], path + (name,))
            elif matched:
                continue
            elif extra is False:
                if self.lenient:
                    self.ignored.append(path + (name,))
                else:
                    self.add(path + (name,), "unknown_field", f"unknown field {name!r}")
            elif isinstance(extra, dict):
                self.node(item, extra, path + (name,))


def _json_type(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _type_ok(value, name):
            return name
    return type(value).__name__


def _pointer(path: tuple[Any, ...]) -> str:
    return "".join("/" + _escape(str(p)) for p in path)


def parse_version(version: Any) -> tuple[int, int] | None:
    """``"1.0"`` → ``(1, 0)``; anything that is not ``major.minor`` → None."""
    if not isinstance(version, str):
        return None
    match = _VERSION.match(version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _run(report: Any) -> _Walk:
    walk = _Walk(lenient=False)
    if not isinstance(report, dict):
        walk.add((), "type", f"expected object, got {_json_type(report)}")
        return walk
    version = parse_version(report.get("schema_version"))
    if "schema_version" in report and version is None:
        walk.add(("schema_version",), "version",
                 f"schema_version {report['schema_version']!r} is not 'major.minor'")
        return walk
    if version is not None and version[0] != 1:
        walk.add(("schema_version",), "unsupported_version",
                 f"this report uses spec version {report['schema_version']}; this Avaloka "
                 f"reads {SCHEMA_VERSION.split('.')[0]}.x. Upgrade Avaloka to open it.")
        return walk
    if not walk.json_only(report, ()):
        return walk
    current = parse_version(SCHEMA_VERSION)
    newer_minor = version is not None and version[1] > current[1]
    walk.lenient = newer_minor
    # Every 1.x is held to this schema's rules except the version constant
    # itself: an older minor lacks only optional fields, a newer one adds some.
    body = dict(report, schema_version=SCHEMA_VERSION) if version not in (None, current) else report
    walk.node(body, SCHEMA, ())
    return walk


def validate(report: Any) -> list[Issue]:
    """Every reason ``report`` is not a valid Outcome Report; ``[]`` means valid."""
    return _run(report).issues


def is_valid(report: Any) -> bool:
    return not validate(report)


def assert_valid(report: Any) -> None:
    issues = validate(report)
    if issues:
        raise SpecError(issues)


def accept(report: Any) -> dict[str, Any]:
    """What a renderer reads: a valid report, minus fields from a newer minor.

    Raises :class:`SpecError` for an invalid report or an unknown major.
    """
    walk = _run(report)
    if walk.issues:
        raise SpecError(walk.issues)
    out = copy.deepcopy(report)
    for path in sorted(walk.ignored, key=len, reverse=True):
        parent = out
        for step in path[:-1]:
            parent = parent[step]
        del parent[path[-1]]
    return out


def published_schema_text() -> str:
    """The exact text of ``outcome-report.schema.json``; a test holds them equal."""
    import json
    return json.dumps(SCHEMA, indent=2, ensure_ascii=False) + "\n"


# ── figures and placeholders ─────────────────────────────────────────────────

PLACEHOLDER = re.compile(r"\{(f[0-9]{1,3})\}")
# §3.2 rule 1: what a text slot may contain besides placeholders.
_ALLOWED_TOKENS = [re.compile(p) for p in
                   (r"\b(19|20)\d{2}\b", r"\bQ[1-4]\b", r"\bH[12]\b", r"\bFY\d{2,4}\b")]
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def placeholders(text: str) -> list[str]:
    """Figure ids a text slot refers to, in order: ``"{f1} of {f2}"`` → ``["f1", "f2"]``."""
    return PLACEHOLDER.findall(text)


def free_numbers(text: str) -> list[str]:
    """Digits left after removing placeholders and the allowed tokens (§3.2 rule 1).

    A non-empty result means the slot carries a raw number, which only a figure may.
    """
    stripped = PLACEHOLDER.sub(" ", text)
    for token in _ALLOWED_TOKENS:
        stripped = token.sub(" ", stripped)
    return _DIGITS.findall(stripped)


def text_slots(report: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """``(pointer, text)`` for every slot §3.2 checks: headline, bullets, captions,
    strategy action and text."""
    headline = report.get("headline") or {}
    if isinstance(headline.get("text"), str):
        yield "/headline/text", headline["text"]
    for i, bullet in enumerate(headline.get("bullets") or []):
        for key in ("so_what", "now_what"):
            if isinstance(bullet.get(key), str):
                yield f"/headline/bullets/{i}/{key}", bullet[key]
    for section in ("charts", "maps"):
        for i, chart in enumerate(report.get(section) or []):
            if isinstance(chart.get("caption"), str):
                yield f"/{section}/{i}/caption", chart["caption"]
    for i, card in enumerate(report.get("strategies") or []):
        for key in ("action", "text"):
            if isinstance(card.get(key), str):
                yield f"/strategies/{i}/{key}", card[key]


def unresolved_refs(report: dict[str, Any]) -> list[Issue]:
    """References to figures that do not exist, and duplicate figure ids.

    Covers every field the schema types as a figure reference (found from the
    schema, so a new one is covered without a change here) and every
    placeholder in a text slot.
    """
    issues: list[Issue] = []
    known: set[str] = set()
    for i, figure in enumerate(report.get("figures") or []):
        fid = figure.get("id")
        if fid in known:
            issues.append(Issue(f"/figures/{i}/id", "duplicate_figure",
                                f"figure id {fid!r} is used twice"))
        known.add(fid)
    walk = _Walk(lenient=True)
    walk.node(dict(report, schema_version=SCHEMA_VERSION), SCHEMA, ())
    refs = [(p, f) for p, f in walk.frefs if not p.startswith("/figures/")]
    refs += [(p, f) for p, text in text_slots(report) for f in placeholders(text)]
    for pointer, fid in refs:
        if fid not in known:
            issues.append(Issue(pointer, "unresolved_figure", f"{fid!r} is not in figures"))
    return issues


def resolve_pointer(document: Any, pointer: str) -> Any:
    """RFC 6901 lookup, used for ``figures[].evidence_ref``; raises KeyError."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    node = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            if not token.isdigit() or (token != "0" and token.startswith("0")):
                raise KeyError(pointer)
            index = int(token)
            if index >= len(node):
                raise KeyError(pointer)
            node = node[index]
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            raise KeyError(pointer)
    return node


def evidence_leaf(evidence: Any, pointer: str) -> tuple[float, str | None, str | None]:
    """``(value, unit, currency)`` of the evidence leaf at ``pointer``.

    A bare number has no unit of its own (it is in the citing figure's unit):
    ``(0.084, None, None)``. A leaf object gives its declared unit:
    ``{"value": 0.084, "unit": "fraction"}`` → ``(0.084, "fraction", None)``.
    Raises KeyError if the pointer does not resolve and ValueError if it does
    not resolve to a leaf.
    """
    node = resolve_pointer(evidence, pointer)
    if _is_number(node):
        return node, None, None
    if isinstance(node, dict) and isinstance(node.get("unit"), str) \
            and _is_number(node.get("value")):
        return node["value"], node["unit"], node.get("currency")
    raise ValueError(f"{pointer} is not an evidence leaf")
