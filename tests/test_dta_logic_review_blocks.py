"""The logic reviewer must BLOCK by default when it judges a transform WRONG — a
logically-incorrect transform must not silently write to the destination. Previously
the non-strict (default) branch returned APPROVED even on a WRONG verdict, and only a
hidden env var (AVALOKA_STRICT_LOGIC_VALIDATION=1) made it block. The default is now
to block; AVALOKA_STRICT_LOGIC_VALIDATION=0 is an explicit opt-out into advisory mode.

Downstream contract (data_transfer_agent.py Step 8): logical_semantic_error=True is a
generation failure → the transfer aborts instead of writing. So asserting the node
sets that flag is asserting the transfer is blocked.
"""
import app.agents.data_transfer_agent.daft_validator as dv
from app.agents.data_transfer_agent.daft_validator import (
    logical_semantic_validator_node,
)


class _VerdictLLM:
    """Reviewer stub returning a fixed correctness verdict."""
    def __init__(self, is_correct):
        self._is_correct = is_correct

    def invoke(self, _messages):
        import json

        class _R:
            content = json.dumps({
                "is_logically_correct": self._is_correct,
                "rationale": "column 'total' is a random sample, not the requested sum",
            })
        return _R()


def _state():
    return {
        "user_prompt": "sum sales into 'total'",
        "generated_code": "df = df.with_column('total', df['sales'].sample())",
        "execution_stdout": "ran",
        "execution_output_preview": "| total |\n| 3 |",
        "syntax_error": False,
        "static_semantic_error": False,
    }


def test_wrong_verdict_blocks_by_default(monkeypatch):
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    monkeypatch.setattr(dv, "validator_llm", _VerdictLLM(is_correct=False))

    out = logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is True          # blocks → repair/abort
    assert out.get("code_validated") is not True
    assert out["code_validation_feedback"] != "APPROVED"
    # Reviewer rationale is preserved for the repair feedback / warning.
    assert "sample" in (out.get("code_validation_feedback") or "").lower()


def test_correct_verdict_approves(monkeypatch):
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    monkeypatch.setattr(dv, "validator_llm", _VerdictLLM(is_correct=True))

    out = logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is False
    assert out["code_validation_feedback"] == "APPROVED"
    assert out["code_validated"] is True


def test_advisory_opt_out_still_proceeds_on_wrong(monkeypatch):
    # Explicit escape hatch: warn but do not block.
    monkeypatch.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "0")
    monkeypatch.setattr(dv, "validator_llm", _VerdictLLM(is_correct=False))

    out = logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is False
    assert out["code_validation_feedback"] == "APPROVED"
    # The reviewer's concern is still surfaced as advisory feedback.
    assert "sample" in (out.get("logical_review_feedback") or "").lower()


def test_explicit_strict_one_also_blocks(monkeypatch):
    # Back-compat: the old opt-in value still means block.
    monkeypatch.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "1")
    monkeypatch.setattr(dv, "validator_llm", _VerdictLLM(is_correct=False))

    out = logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is True
