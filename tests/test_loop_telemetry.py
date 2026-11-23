"""Loop shape: converged, exhausted, or oscillating — and telling them apart."""

from __future__ import annotations

import pytest

from app.core.loop_telemetry import (Attempt, LoopOutcome, LoopTrace,
                                     summarise, trace_from_state)


def trace(*stages, max_retries=2, loop_id="t"):
    """Build a trace from failing-stage names; None means that attempt passed."""
    t = LoopTrace(loop_id=loop_id, max_retries=max_retries)
    for i, stage in enumerate(stages):
        t.record(Attempt(index=i, failing_stage=stage,
                         feedback=None if stage is None else f"{stage} objected"))
    return t


# --------------------------------------------------------------------------- #
# The distinction the current code cannot make
# --------------------------------------------------------------------------- #

def test_a_loop_that_fixes_itself_is_converged():
    assert trace("syntax", None).outcome is LoopOutcome.CONVERGED


def test_first_time_pass_is_converged():
    assert trace(None).outcome is LoopOutcome.CONVERGED


def test_running_out_of_retries_is_exhausted_not_converged():
    t = trace("syntax", "static", "logical", max_retries=2)
    assert t.outcome is LoopOutcome.EXHAUSTED
    assert t.converged is False


def test_oscillation_is_distinguished_from_exhaustion():
    """The failure mode invisible today: fix A, break B, break A again.

    From outside, this looks identical to an exhausted loop — both stop
    failing after three attempts. One was making progress; this one was not.
    """
    t = trace("syntax", "static", "syntax", max_retries=5)
    assert t.outcome is LoopOutcome.OSCILLATED


def test_steady_progress_through_distinct_stages_is_not_oscillation():
    t = trace("syntax", "static", "logical", max_retries=5)
    assert t.outcome is not LoopOutcome.OSCILLATED


def test_two_attempts_cannot_show_oscillation():
    """Three are needed: fail A, fail B, fail A again."""
    assert trace("syntax", "syntax", max_retries=5).outcome is not LoopOutcome.OSCILLATED


def test_an_empty_loop_is_aborted():
    assert LoopTrace(loop_id="x").outcome is LoopOutcome.ABORTED


def test_giving_up_early_is_hard_failure_not_exhaustion():
    """Stopping below the ceiling means something was unrecoverable."""
    assert trace("syntax", max_retries=5).outcome is LoopOutcome.HARD_FAILED


# --------------------------------------------------------------------------- #
# Which stage actually blocked
# --------------------------------------------------------------------------- #

def test_blocking_stage_is_the_last_objection():
    assert trace("syntax", "logical").blocking_stage() == "logical"


def test_a_converged_loop_has_no_blocking_stage():
    assert trace("syntax", None).blocking_stage() is None


def test_stages_hit_preserves_first_appearance_order():
    assert trace("static", "syntax", "static").stages_hit() == ["static", "syntax"]


# --------------------------------------------------------------------------- #
# Failure signatures ignore incidental detail
# --------------------------------------------------------------------------- #

def test_signatures_match_despite_shifting_line_numbers():
    """Messages embed line numbers; the objection is the same."""
    a = Attempt(0, "syntax", "SyntaxError: unexpected token at line 12")
    b = Attempt(1, "syntax", "SyntaxError: unexpected token at line 12")
    assert a.signature() == b.signature()


def test_different_stages_never_share_a_signature():
    a = Attempt(0, "syntax", "same text")
    b = Attempt(1, "logical", "same text")
    assert a.signature() != b.signature()


def test_a_passing_attempt_signature_is_ok():
    assert Attempt(0, None).signature() == "ok"


# --------------------------------------------------------------------------- #
# Aggregation — what a harness run reports
# --------------------------------------------------------------------------- #

def test_convergence_rate_is_the_headline_number():
    stats = summarise([trace("syntax", None), trace(None),
                       trace("syntax", "static", "logical")])
    assert stats["n_loops"] == 3
    assert stats["convergence_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_blocking_stages_are_ranked_by_frequency():
    stats = summarise([
        trace("syntax", "logical"), trace("static", "logical"), trace("syntax"),
    ])
    assert list(stats["blocking_stages"])[0] == "logical"


def test_converged_loops_report_their_own_mean_attempts():
    """Mixing failed loops into the mean hides how fast success actually is."""
    stats = summarise([trace(None), trace("syntax", None),
                       trace("syntax", "static", "logical")])
    assert stats["mean_attempts_when_converged"] == pytest.approx(1.5)


def test_no_loops_summarises_cleanly():
    assert summarise([])["n_loops"] == 0


def test_all_failing_reports_no_converged_mean():
    assert summarise([trace("syntax", "static", "logical")])["mean_attempts_when_converged"] is None


# --------------------------------------------------------------------------- #
# Serialisation and the state bridge
# --------------------------------------------------------------------------- #

def test_trace_serialises_for_a_report():
    import json
    payload = trace("syntax", None).as_dict()
    json.dumps(payload)
    assert payload["outcome"] == "converged"
    assert payload["attempts_used"] == 2


def test_summary_names_the_blocking_stage():
    assert "logical" in trace("syntax", "logical").summary()


def test_state_bridge_reads_what_coder_already_writes():
    """Until a trace is threaded through the graph, reconstruct from state."""
    t = trace_from_state({"retry_count": 2, "code_validation_feedback": "bad column"})
    assert t.attempts_used == 2
    assert t.converged is False


def test_state_bridge_treats_no_feedback_as_success():
    t = trace_from_state({"retry_count": 1, "code_validation_feedback": None})
    assert t.converged is True


def test_telemetry_never_intervenes():
    """Passive by design: no writes back into agent state, no exceptions."""
    import inspect

    import app.core.loop_telemetry as mod
    src = inspect.getsource(mod)
    for forbidden in ("state.update(", "state[", "raise "):
        assert forbidden not in src, f"telemetry must stay passive ({forbidden})"
