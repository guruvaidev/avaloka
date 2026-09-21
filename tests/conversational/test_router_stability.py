"""The router must give the same answer every time, not most of the time.

Why repeated trials. The reported bug was a *sampling* failure: with the
production model, a bare "run 3" classified as ``train_model`` somewhere
between 40% and 90% of the time depending on the sampling draw, and the user
was answered by the training agent with a refusal about a pivot table. A
single-shot assertion would have passed most runs, failed some, and been
quarantined as flaky -- so the defect would have survived the test that was
supposed to catch it.

An LLM in a control path is only as good as its worst draw. These tests assert
on a *rate over N trials* with zero tolerance, which is the only shape that can
distinguish "correct" from "usually correct".

Skipped unless a key is present, so the offline suite stays hermetic.
"""
import os

import pytest

pytest.importorskip("langchain_core")

TRIALS = int(os.getenv("AVALOKA_ROUTER_TRIALS", "12"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv("GROQ_API_KEY_PLANNING_AGENT") or os.getenv("GROQ_API_KEY")
             or os.getenv("OPENROUTER_API_KEY")),
        reason="router stability needs a planning-model key",
    ),
]

CSV_INFO = ("columns: Activity Domain, Number of Activity Domains, "
            "Target Hours per Domain, Total man hrs, Score")


def _route_many(text, trials=TRIALS):
    from app.agents.planner import _classify_planner_route
    return [_classify_planner_route(text, CSV_INFO) for _ in range(trials)]


def _rate(routes, value):
    return sum(r == value for r in routes) / len(routes)


# -- picks must never be read as training requests -------------------------

@pytest.mark.parametrize("message", ["run 3", "run 5", "run 1", "do 2", "option 4"])
def test_a_bare_pick_never_routes_to_training(message):
    """The exact phrasing the product tells users to type.

    _render_suggestions closes with "say the number ... or 'run 2'", so this is
    not a hypothetical input -- it is the one Avaloka asks for by name.
    """
    routes = _route_many(message)
    misrouted = _rate(routes, "train_model")
    assert misrouted == 0, (
        f"{message!r} routed to the training agent in "
        f"{misrouted:.0%} of {len(routes)} trials -- the user gets "
        f"'I can only help with model-training and inference tasks' "
        f"in response to an analysis request"
    )


# -- and genuine training requests must still get there --------------------

@pytest.mark.parametrize("message", [
    "train a model to predict score",
    "build me a classifier on this data",
    "run the training plan",
])
def test_genuine_training_requests_still_reach_training(message):
    """Guards the over-correction: it would be easy to fix the misroute by
    breaking training routing entirely."""
    routes = _route_many(message, trials=max(4, TRIALS // 3))
    assert _rate(routes, "train_model") == 1, (
        f"{message!r} failed to reach the training agent: {routes}"
    )


# -- analysis requests stay in planning ------------------------------------

@pytest.mark.parametrize("message", [
    "Create a pivot table comparing management effort against target volumes",
    "summarise the columns",
    "plot hours per domain",
])
def test_analysis_requests_stay_in_planning(message):
    routes = _route_many(message, trials=max(4, TRIALS // 3))
    assert _rate(routes, "continue_planning") == 1, f"{message!r} -> {routes}"
