"""1.6 release suite: coder retry loop guard rails (plan case AG-04).

Oscillation itself (fix A, break B, fix B, break A) is model behaviour and needs
a live model. What is deterministic, and tested here, is what stops it: the
retry works from the previous code plus the latest validator feedback rather
than regenerating from scratch, and the loop is capped however the errors
alternate. See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import app.agents.coder as coder_mod
import app.api.workflow as wf

_PLAN = "Group by region and sum revenue."
_PREV_CODE = "import pandas as pd\n\ndef main(df):\n    return df.groupby('region')['revenue'].sum()\n"
_FEEDBACK_B = "Static check: column 'Revenue' does not exist; use 'revenue'."


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return AIMessage(content="```python\nimport pandas as pd\n\ndef main(df):\n    return df\n```")


def test_ag04_retry_refines_previous_code_with_latest_feedback_and_loop_is_capped(monkeypatch):
    """AG-04: a refine attempt is sent the previous code and the latest validator
    feedback (no fresh blueprint), and alternating failures stop at the cap."""
    llm = _RecordingLLM()
    monkeypatch.setattr(coder_mod, "coder_llm", llm)

    def _no_fresh_blueprint(_state):
        raise AssertionError("a feedback retry must not regenerate the blueprint")

    monkeypatch.setattr(coder_mod, "_generate_pseudocode", _no_fresh_blueprint)

    state = {
        "plan": _PLAN,
        "retry_count": 1,
        "messages": [HumanMessage(content="sum revenue by region")],
        "generated_code": _PREV_CODE,
        "code_validation_feedback": _FEEDBACK_B,
        "logical_semantic_error": True,
        "schema": {"region": "object", "revenue": "float64"},
        "input_sample_data": [{"region": "n", "revenue": 1.0}],
        "input_data_type": "csv",
    }
    out = coder_mod.coder_node(state)

    assert len(llm.calls) == 1, "the refine attempt must make exactly one model call"
    prompt = "\n".join(str(getattr(m, "content", m)) for m in llm.calls[0])
    assert _PREV_CODE.strip() in prompt
    assert _FEEDBACK_B in prompt
    assert out["retry_count"] == 2

    # Errors alternating A/B/A/B never keep the loop alive past the cap.
    routes = []
    for attempt in range(6):
        err = "syntax_error" if attempt % 2 == 0 else "logical_semantic_error"
        routes.append(wf.check_validation_status({err: True, "retry_count": attempt}))
    assert routes[:3] == ["refine", "refine", "refine"]
    assert set(routes[3:]) == {"end"}, routes
