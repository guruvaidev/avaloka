"""The reviewer's rationale must reach the CODER on a strict block.

test_dta_logic_review_blocks.py asserts the validator node sets the rationale, but
the node is not the contract that matters: the repair loop overwrites
``code_validation_feedback`` with ``build_composite_feedback(...)`` before every
regeneration (data_transfer_agent.py Step 5). ``build_composite_feedback`` reads
``logical_review_feedback`` — so a strict block that set only
``code_validation_feedback`` had its rationale erased to None on the way to the
coder: 5 blind repair cycles re-emitting the same code, then an abort whose reason
was empty for both the customer and the operator log.

Blocking is the default (AVALOKA_STRICT_LOGIC_VALIDATION unset), so that path is
the common one — these tests pin the seam between the two modules rather than
either module alone.
"""
import app.agents.data_transfer_agent.daft_validator as dv
from app.agents.data_transfer_agent.daft_validator import logical_semantic_validator_node
from app.agents.data_transfer_agent.data_transfer_agent import build_composite_feedback

_RATIONALE = "column 'total' is a random sample, not the requested sum"


class _WrongVerdictLLM:
    """Reviewer stub that judges the transform logically incorrect."""

    def invoke(self, _messages):
        import json

        class _R:
            content = json.dumps(
                {"is_logically_correct": False, "rationale": _RATIONALE}
            )

        return _R()


def _coder_def():
    return {
        "user_prompt": "sum sales into 'total'",
        "generated_code": "df = df.with_column('total', df['sales'].sample())",
        "execution_stdout": "ran",
        "execution_output_preview": "| total |\n| 3 |",
        "syntax_error": False,
        "static_semantic_error": False,
    }


def test_strict_block_rationale_survives_build_composite_feedback(monkeypatch):
    """The exact seam: validator output → build_composite_feedback → coder prompt."""
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    monkeypatch.setattr(dv, "validator_llm", _WrongVerdictLLM())

    coder_def = _coder_def()
    coder_def.update(logical_semantic_validator_node(coder_def))
    assert coder_def["logical_semantic_error"] is True  # precondition: blocked

    # Exactly what the repair loop does before regenerating (Step 5).
    feedback = build_composite_feedback(coder_def)

    assert feedback is not None, (
        "Reviewer rationale was erased before reaching the coder — the repair loop "
        "would regenerate blind and abort with an empty reason."
    )
    assert "sample" in feedback.lower()
    assert "LOGIC REVIEW FEEDBACK" in feedback


def test_strict_block_sets_the_key_the_repair_loop_reads(monkeypatch):
    """Pin the key name itself: logical_review_feedback is the coder-facing channel.

    code_validation_feedback is build_composite_feedback's OUTPUT and is overwritten
    every cycle, so it cannot be the carrier for the block's own rationale.
    """
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    monkeypatch.setattr(dv, "validator_llm", _WrongVerdictLLM())

    out = logical_semantic_validator_node(_coder_def())

    assert _RATIONALE in (out.get("logical_review_feedback") or "")


def test_strict_block_rationale_reaches_the_operator_log_and_customer(monkeypatch):
    """Step 5 logs, and _surface_logic_review_warning, both read logical_review_feedback.

    Without it both render None — the failure is invisible from inside and outside.
    """
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    monkeypatch.setattr(dv, "validator_llm", _WrongVerdictLLM())

    coder_def = _coder_def()
    coder_def.update(logical_semantic_validator_node(coder_def))

    assert coder_def.get("logical_review_feedback") is not None
