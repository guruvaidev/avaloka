"""The ten recipe specs as the design declares them (§2.5), in one place.

The intent parser needs a recipe's parameters before the recipe exists as
code: ``recognize`` checks a model's JSON against them, ``heuristic_intent``
knows which column roles to look for, and ``bind`` knows what is required.
A recipe that is implemented owns its spec (``avaloka.recipes.library``);
this module takes it from there, so the parser and the recipe can never
disagree about what a parameter is. A recipe that is not yet implemented is
declared here as the design describes it, and moves to its module when built;
``tests/recipes/test_specs_match_library.py`` holds the two in step.

``WINDOWS`` is the closed vocabulary of the ``window`` parameter: period
labels ``compare_periods`` understands. ``last_period`` means the most recent
complete period at the time column's grain against the one before it.
"""

from __future__ import annotations

from .base import RecipeSpec
from .library import ab_test as _ab_test
from .library import anomalies as _anomalies
from .library import cohort_retention as _cohort_retention
from .library import compare_periods as _compare_periods
from .library import correlate as _correlate
from .library import describe as _describe
from .library import profile as _profile
from .library import regress as _regress
from .library import segment as _segment
from .library import trend as _trend
from .registry import RECIPE_IDS

WINDOWS: tuple[str, ...] = _compare_periods.WINDOWS

#: Column roles a parameter role accepts. A ``target`` is a measure the caller
#: named; a high-cardinality dimension still groups, the recipe's
#: ``high_cardinality`` gate decides whether it groups usefully.
ROLE_ACCEPTS: dict[str, tuple[str, ...]] = {
    "measure": ("measure", "target"),
    "dimension": ("dimension", "high_card_dimension"),
    "time": ("time",),
    "id": ("id",),
    "geo": ("geo",),
    "stage": ("stage",),
    "target": ("target",),
    "text": ("text",),
    "high_card_dimension": ("high_card_dimension",),
}

#: Defaults the binder fills for optional non-column parameters when the
#: question named none. A column parameter never has a default: an unnamed
#: optional dimension stays unbound and the recipe runs ungrouped.
DEFAULTS: dict[tuple[str, str], str] = {
    ("compare_periods", "window"): "last_period",
}


SPECS: dict[str, RecipeSpec] = {s.id: s for s in (
    _profile.SPEC,
    _describe.SPEC,
    _trend.SPEC,
    _compare_periods.SPEC,
    _segment.SPEC,
    _cohort_retention.SPEC,
    _correlate.SPEC,
    _regress.SPEC,
    _anomalies.SPEC,
    _ab_test.SPEC,
)}

assert tuple(SPECS) == RECIPE_IDS, "specs.py must declare every RECIPE_IDS member once, in order"


def spec_for(recipe_id: str) -> RecipeSpec:
    """The spec of a recipe id; ``KeyError`` for an id outside RECIPE_IDS."""
    return SPECS[recipe_id]
