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
# --- the count she announces is the count that comes back -------------------

def test_narrator_announces_a_specialist_that_joins_after_the_count():
    """announce(2), then three dispatches: the third says one more is coming and
    corrects the count, so what she announced is what converge() reports back."""
    events = []
    n = SwarmNarrator(events.append)
    n.announce(2)
    for f in ("scout", "profiler", "conformal_calibrator"):
        n.dispatch(f)
    joined = [e for e in events if "copies in all" in e]
    assert len(joined) == 1 and "3 copies in all" in joined[0], events
    assert n._announced == 3 == n._dispatched
    n.dispatch("scout")   # a copy within the announced count says nothing extra
    assert sum("copies in all" in e for e in events) == 2   # 4 > 3: corrected again
    assert "4 copies in all" in events[-2]

_PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY",
                  "GEMINI_API_KEY", "INFERENCE_PROVIDER", "AVALOKA_LLM_MODEL")


def test_train_announces_the_number_of_copies_that_come_back(churn_csv, tmp_path):
    """`avaloka train` said "10 focused copies" and then "All 11 copies are back".

    The planner slots in a specialist (here the conformal calibrator) while the
    mission is running, after the up-front announcement. Runs the real CLI with
    the deterministic voice (no model, no network) and checks the last count she
    announced, the count she says came back and the clones that actually set off
    are one number.
    """
    import os
    import re
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in _PROVIDER_KEYS}
    env["COLUMNS"] = "400"  # one narration line per console line
    res = subprocess.run(
        [sys.executable, "-m", "avaloka", "train", str(churn_csv), "--target", "churned",
         "-o", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    out = " ".join(res.stdout.split())

    assert "Ava-conformal_calibrator sets off" in out, "fixture no longer triggers a mid-run specialist"
    announced = [int(n) for n in re.findall(r"Cloning myself into (\d+) focused copies", out)]
    announced += [int(n) for n in re.findall(r"(\d+) copies in all", out)]
    returned = [int(n) for n in re.findall(r"All (\d+) copies are back", out)]
    set_off = out.count(" sets off")
    assert announced and returned == [set_off], (announced, returned, set_off)
    assert announced[-1] == returned[0] == set_off, (
        f"announced {announced}, said {returned[0]} came back, {set_off} set off")
