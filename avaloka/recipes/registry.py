"""The closed set of recipe ids, and the registry of implementations.

``RECIPE_IDS`` is a vocabulary, not a list of what is built: an id is here
when the design names it (``docs/design/1.7-INSTANT-ANALYSIS.md`` §2.5), and
``all_recipes()`` returns what is actually registered. The id may be recorded
in telemetry; a parameter (a column name) never may.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from .base import Recipe

RECIPE_IDS: tuple[str, ...] = (
    "profile", "describe", "trend", "compare_periods", "segment", "cohort_retention",
    "correlate", "regress", "anomalies", "ab_test",
)

_REGISTRY: dict[str, "Recipe"] = {}


def register(recipe: "Recipe") -> "Recipe":
    """Register one recipe instance; a second registration of an id is an error."""
    rid = recipe.spec.id
    if rid not in RECIPE_IDS:
        raise ValueError(f"unknown recipe id {rid!r}")
    if rid in _REGISTRY:
        raise ValueError(f"recipe {rid!r} is already registered")
    _REGISTRY[rid] = recipe
    return recipe


def get(recipe_id: str) -> "Recipe":
    try:
        return _REGISTRY[recipe_id]
    except KeyError:
        known = sorted(_REGISTRY)
        raise KeyError(f"no recipe {recipe_id!r} is registered; registered: {known}") from None


def all_recipes() -> Iterator["Recipe"]:
    """Registered recipes in RECIPE_IDS order."""
    for rid in RECIPE_IDS:
        if rid in _REGISTRY:
            yield _REGISTRY[rid]


def _reset_for_tests() -> None:
    _REGISTRY.clear()
