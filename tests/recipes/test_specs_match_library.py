"""One spec per recipe: the parser's table is the implementation's spec.

``specs.SPECS`` is what ``recognize``, ``heuristic_intent`` and ``bind`` read;
a recipe module is what runs. If the two drifted, the binder could bind a
parameter the recipe does not take, or skip a precondition it enforces.
"""

from __future__ import annotations

import avaloka.recipes.library  # noqa: F401  (registers the implemented recipes)
from avaloka.recipes import RECIPE_IDS, all_recipes
from avaloka.recipes.specs import SPECS


def test_every_registered_recipe_is_the_table_entry():
    registered = list(all_recipes())
    assert registered, "no recipe registered"
    for recipe in registered:
        assert SPECS[recipe.spec.id] is recipe.spec


def test_the_table_declares_every_id_once_in_order():
    assert tuple(SPECS) == RECIPE_IDS


def test_trend_declares_the_periods_it_needs():
    assert "min_periods_4" in SPECS["trend"].preconditions
