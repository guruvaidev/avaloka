"""The Analysis Recipe Library: typed, versioned, testable analyses.

A recipe answers one bounded question from a dataset by code — DuckDB SQL and
a statistics library — and emits an Outcome Report (``avaloka.outcomes``) and
a parameterised notebook (``avaloka.notebook``). A language model never writes
a number here; it may only parse the question into a closed vocabulary
(:class:`Intent`) and rephrase template sentences whose figures are fixed.

Design: ``docs/design/1.7-INSTANT-ANALYSIS.md`` §2. Nothing in this package
imports ``app.*`` (S1.08). DuckDB, scipy and statsmodels are the
``avaloka[recipes]`` extra, not base dependencies (S1.09); a recipe that needs
one reports the install command in plain words when it is absent.
"""

from .base import (CAVEAT_KINDS, EVIDENCE_KINDS, GATES, GRAINS, INTENT_FLAGS, PARAM_NAMES,
                   PRECONDITIONS, REASONS, ROLES, Ask, Bound, BudgetReport, Evidence, Gate,
                   GateReport, Intent, ParamSpec, Recipe, RecipeSpec, Refusal, RunResult, Verdict)
from .budget import DEFAULT as DEFAULT_BUDGET
from .budget import Budget
from .registry import RECIPE_IDS, all_recipes, get, register

__all__ = [
    "CAVEAT_KINDS", "EVIDENCE_KINDS", "GATES", "GRAINS", "INTENT_FLAGS", "PARAM_NAMES", "PRECONDITIONS",
    "REASONS", "ROLES", "Ask", "Bound", "Budget", "BudgetReport", "DEFAULT_BUDGET", "Evidence",
    "Gate", "GateReport", "Intent", "ParamSpec", "Recipe", "RecipeSpec", "Refusal", "RunResult", "Verdict",
    "RECIPE_IDS", "all_recipes", "get", "register",
]
