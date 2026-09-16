"""The DTA logic reviewer is TOLD to check the output columns/preview, but the
prompt template had no ``$preview`` slot — so ``safe_substitute(preview=…)`` silently
discarded it and the reviewer checked columns it never saw. And the preview serialiser
assumed a list, so the DTA's string preview would have been sliced to its first two
CHARACTERS. These tests lock in that the reviewer actually receives the output data.
"""
import app.agents.data_transfer_agent.daft_validator as dv
from app.agents.data_transfer_agent.daft_validator import (
    _REVIEW_PROMPT_TEMPLATE,
    _serialise_output_preview,
    logical_semantic_validator_node,
)


def test_template_has_a_preview_slot():
    assert "$preview" in _REVIEW_PROMPT_TEMPLATE.template


def test_serialise_preview_handles_string_verbatim():
    # DTA path: preview is the already-rendered stdout table (a string).
    table = "| billing_category | Price |\n| High | 30 |\n| Low | 10 |"
    assert _serialise_output_preview(table) == table
    # NOT sliced to the first two characters (the original latent bug).
    assert _serialise_output_preview(table) != table[:2]


def test_serialise_preview_handles_record_list_as_json():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
    out = _serialise_output_preview(rows)
    assert '"a": 1' in out and '"a": 3' in out
    assert '"a": 5' not in out  # only the first two rows


def test_serialise_preview_empty_is_labelled():
    assert _serialise_output_preview([]) == "No output preview available."
    assert _serialise_output_preview("") == "No output preview available."
    assert _serialise_output_preview(None) == "No output preview available."


class _CapturingLLM:
    """Records the prompt the reviewer actually receives; approves everything."""
    def __init__(self):
        self.seen = None

    def invoke(self, messages):
        self.seen = messages[0].content

        class _R:
            content = '{"is_logically_correct": true, "rationale": "ok"}'
        return _R()


def test_reviewer_prompt_actually_contains_the_output_preview(monkeypatch):
    llm = _CapturingLLM()
    monkeypatch.setattr(dv, "validator_llm", llm)

    preview_table = "| billing_category | Price |\n| High | 30 |\n| Low | 10 |"
    state = {
        "user_prompt": "Add billing_category; only output Price and billing_category.",
        "generated_code": "df = df.select('Price', 'billing_category')",
        "execution_stdout": "run ok",
        "execution_output_preview": preview_table,   # DTA stores a string
        "syntax_error": False,
        "static_semantic_error": False,
    }

    result = logical_semantic_validator_node(state)

    assert llm.seen is not None
    # The reviewer must actually SEE the transformed output columns/values.
    assert "billing_category" in llm.seen
    assert preview_table in llm.seen
    # And the slot must be filled — no leftover literal placeholder.
    assert "$preview" not in llm.seen
    # Sanity: approval flows through.
    assert result.get("logical_semantic_error") is False
