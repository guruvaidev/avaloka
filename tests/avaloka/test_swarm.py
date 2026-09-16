"""Comprehensive swarm tests.

The swarm is Avaloka cloning herself across a mission's steps. These tests pin
the behaviour the spec promises: the right clones activate for the right job (and
only those), they run in the right order, the narration matches the ledger, and
the swarm converges cleanly.
"""

import pytest

from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import (analyze_fireflies, run_analyze, run_train,
                              train_fireflies)
from avaloka.persona import Avaloka
from avaloka.swarm import CLONE, SwarmNarrator, clone_of, make_progress


def _ctx(kind, data, out, **kw):
    return MissionContext(kind=kind, goal=kw.pop("goal", "test"), data_source=str(data),
                          output_dir=out, budget=Budget(limit_usd=kw.pop("budget", None)), **kw)


# --- clone identity -------------------------------------------------------
def test_every_active_firefly_has_a_clone_persona():
    """No firefly may activate without a named clone — the swarm must be coherent."""
    active = {f.name for f in analyze_fireflies()} | {f.name for f in train_fireflies(True)}
    for name in active:
        clone_name, focus = clone_of(name)
        assert clone_name.startswith("Ava-") and focus
        assert name in CLONE, f"{name} activates but has no clone identity"


def test_clone_names_are_stable():
    assert clone_of("data_scout")[0] == "Ava-Scout"
    assert clone_of("validator")[0] == "Ava-Skeptic"
    assert clone_of("unknown_firefly")[0] == "Ava-unknown_firefly"


# --- conditional activation (economic, not theatrical) --------------------
def test_analyze_never_wakes_modelling_clones():
    names = [f.name for f in analyze_fireflies()]
    assert "model_scientist" not in names and "ml_engineer" not in names
    assert names[0] == "data_scout" and names[-1] == "reporter"


def test_train_activates_modelling_but_not_packaging_unless_deployable():
    lean = [f.name for f in train_fireflies(deployable=False)]
    full = [f.name for f in train_fireflies(deployable=True)]
    assert "model_scientist" in lean and "ml_engineer" not in lean
    assert "ml_engineer" in full
    # FinOps weighs compute vs quality only when there's a model to serve.
    assert "finops" in lean


# --- narration matches the ledger ----------------------------------------
def test_narrator_counts_dispatch_and_converge():
    emitted = []
    narrator = SwarmNarrator(emitted.append)
    narrator.announce(3)
    for f in ("data_scout", "validator", "reporter"):
        narrator.dispatch(f)
    from avaloka.mission.ledger import Ledger, WorkUnit
    led = Ledger()
    led.record(WorkUnit("data_scout", "Analyst", "x", manual_minutes=120))
    narrator.converge(led)
    assert any("Cloning myself into 3" in e for e in emitted)
    assert sum("sets off" in e for e in emitted) == 3
    assert any("copies are back" in e for e in emitted)


def test_make_progress_adapter_dispatches_on_start():
    events = []
    narrator = SwarmNarrator(events.append)
    on_progress = make_progress(narrator)
    on_progress("start", "data_scout")
    on_progress("done", "data_scout: profiled 5 columns")
    assert any("Ava-Scout" in e and "sets off" in e for e in events)
    assert any("returns" in e for e in events)


def test_narrator_with_persona_voice_is_safe():
    """Voice rephrasing must never crash when the LLM is disabled."""
    ava = Avaloka(llm=False)
    emitted = []
    narrator = SwarmNarrator(emitted.append, voice=ava)
    narrator.announce(6)
    assert emitted and "Avaloka" in emitted[0]


# --- swarm over a real mission -------------------------------------------
def test_analyze_mission_activates_exactly_the_analyze_swarm(churn_csv, tmp_path):
    ctx = _ctx(MissionKind.ANALYZE, churn_csv, tmp_path / "out", goal="Understand churn")
    run_analyze(ctx)
    activated = [u.firefly for u in ctx.ledger.units]
    assert activated == [f.name for f in analyze_fireflies()]
    # every activation carries measured work
    assert all(u.manual_minutes > 0 for u in ctx.ledger.units)
    assert ctx.blackboard.get("profile") is not None


def test_train_swarm_records_model_and_orders_validator_after_scientist(churn_csv, tmp_path):
    ctx = _ctx(MissionKind.TRAIN, churn_csv, tmp_path / "out", goal="Predict churn",
               target="churned", metric="roc_auc", deployable=True)
    run_train(ctx)
    activated = [u.firefly for u in ctx.ledger.units]
    assert activated.index("model_scientist") < activated.index("validator")
    assert activated.index("validator") < activated.index("ml_engineer")
    assert ctx.blackboard.get("training_summary")


def test_swarm_progress_dispatch_count_matches_fireflies(churn_csv, tmp_path):
    events = []
    narrator = SwarmNarrator(events.append)
    on_progress = make_progress(narrator)
    ctx = _ctx(MissionKind.ANALYZE, churn_csv, tmp_path / "out", goal="g")
    run_analyze(ctx, on_progress)
    assert sum("sets off" in e for e in events) == len(analyze_fireflies())
