"""Outcome Reports: the typed, versioned spec every report renderer reads.

Stdlib only, and nothing from ``app`` (S1.08): the CLI, the SDK and the
platform all build and read the same document.
"""

from .format import (FormatError, figure_matches, figure_matches_evidence, format_figure,
                     format_number, render_text, round_half_even)
from .scenario import ScenarioError, evaluate, scenario_block
from .spec import (EVIDENCE_UNITS, SCHEMA, SCHEMA_VERSION, Issue, SpecError, accept, assert_valid,
                   evidence_leaf, free_numbers, is_valid, placeholders, text_slots, unresolved_refs,
                   validate)

__all__ = [
    "EVIDENCE_UNITS", "SCHEMA", "SCHEMA_VERSION", "Issue", "SpecError", "accept", "assert_valid",
    "evidence_leaf", "free_numbers", "is_valid", "placeholders", "text_slots", "unresolved_refs",
    "validate",
    "FormatError", "figure_matches", "figure_matches_evidence", "format_figure", "format_number",
    "render_text", "round_half_even",
    "ScenarioError", "evaluate", "scenario_block",
]
