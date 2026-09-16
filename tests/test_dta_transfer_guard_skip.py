"""The column-oriented planner guards (fabricated-reference, math-on-string) judge a
request against the ACTIVE analysis dataset's columns. A transfer request names
databases/tables/connections (sunny_test2, adult_income, …) — not columns — so with
a dataset loaded the fabrication guard flagged those names as "missing columns" and
blocked the transfer before it could reach the DTA. plan_etl_job now skips those
guards for transfer-intent prompts (same trigger that gates the DTA fast-path).
"""
from unittest import mock

from langchain_core.messages import HumanMessage

import app.agents.planner as planner
from app.agents.planner import plan_etl_job

# The reported prompt: a transfer phrased with "into … in …" (no standalone " to "),
# so the deterministic fast-path can't fully parse it and it falls through to the
# guards — exactly the path that used to be blocked.
_TRANSFER_PROMPT = (
    "Transfer adult_income from sunny_test2 into table adult_income_dest in "
    "sunny_test7, filtering rows where age > 100"
)

# A genuine analysis request (no transfer verb) — the guard MUST still run here.
_ANALYSIS_PROMPT = "show me the fraud_score by region"


def _state(prompt):
    # An active dataset whose columns do NOT include the transfer's table/db names,
    # so the fabrication guard would have fired if it ran.
    return {
        "messages": [HumanMessage(content=prompt)],
        "user_prompt": prompt,
        "schema": {"income": "float", "region": "string"},
        "uploaded_csv_columns": ["income", "region"],
        "dta_database_registry": {},
    }


def _run(prompt, monkeypatch):
    """Drive plan_etl_job with the fabrication guard spied and no LLM (stub path)."""
    fab_spy = mock.MagicMock(return_value=None)
    monkeypatch.setattr(planner, "_detect_fabricated_reference", fab_spy)
    monkeypatch.setattr(planner, "llm", None)  # forces the deterministic stub plan
    with mock.patch("app.agents.planner._handle_initiate_transfer", return_value="DTA_HANDLED"), \
         mock.patch("app.agents.planner._seed_ray_fields"), \
         mock.patch("app.agents.planner._seed_infra_request_if_needed"):
        out = plan_etl_job(_state(prompt))
    return out, fab_spy


def test_transfer_intent_skips_the_fabrication_guard(monkeypatch):
    out, fab_spy = _run(_TRANSFER_PROMPT, monkeypatch)
    fab_spy.assert_not_called()
    # And the turn was not blocked with a fabrication message.
    last = out["messages"][-1].content if out.get("messages") else ""
    assert "does not exist" not in last.lower()
    assert "isn't in the dataset" not in last.lower()


def test_non_transfer_request_still_runs_the_fabrication_guard(monkeypatch):
    _out, fab_spy = _run(_ANALYSIS_PROMPT, monkeypatch)
    fab_spy.assert_called_once()


def test_transfer_trigger_matches_the_reported_phrasing():
    # The transfer-intent signal (same regex that gates the fast-path) recognises
    # the reported phrasing even though it has no standalone " to ".
    assert planner._TRANSFER_TRIGGER_RE.search(_TRANSFER_PROMPT)
    assert not planner._TRANSFER_TRIGGER_RE.search(_ANALYSIS_PROMPT)
