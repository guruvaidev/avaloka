"""rc18 follow-ups to the rweb17 card-truth fixes (coordinator, on #605).

Plain text is code's and must be true; every model-written value is marked
commentary, wherever it is shown. Built from the stored rweb17 fields in
tests/fixtures/card_truth/rweb17_card_truth.json and the UI fixture
ui/tests/fixtures/commentary/rweb17-card-truth.json.

1. A superlative over groups that a minimum-row filter left out says so.
2. The upload overview's domain and readiness badges carry segments.
3. column_stats' meaning and predictive value are the profiling model's.
4. The chart explainer's answer carries segments.
5. "All requested metrics are complete" claimed what only the model reviewer judged.
6. The executive report's narrative and the trace's model-written reasons are marked.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.agents import visualization_agent as va
from app.api import reply_segments as rs

HERE = Path(__file__).parent
CASES = json.loads((HERE / "fixtures" / "card_truth" / "rweb17_card_truth.json").read_text(encoding="utf-8"))
UI = json.loads((HERE.parent / "ui" / "tests" / "fixtures" / "commentary" / "rweb17-card-truth.json")
                .read_text(encoding="utf-8"))
MESSY = CASES["messy_xlsx"]
SUFFIX = " (commentary — not checked)"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _kinds(segments):
    return {s["kind"] for s in segments or [] if s["text"].strip()}


# --------------------------------------------------------------------------- #
# 1. A minimum-row filter behind a superlative is said
# --------------------------------------------------------------------------- #
def _negative_flags():
    """One row per invoice line of the messy file: Country and whether its
    Quantity is negative (the full-file counts in the fixture)."""
    rows = []
    for country, g in MESSY["full_file"]["groups"].items():
        rows += [{"Country": country, "neg": int(i < g["negative"])} for i in range(g["rows"])]
    return rows


def test_a_rate_by_group_names_the_groups_its_lowest_is_about():
    rows = _negative_flags()
    small = {c for c, g in MESSY["full_file"]["groups"].items() if g["rows"] < va.RATE_MIN_BIN_ROWS}
    assert small and all(MESSY["full_file"]["groups"][c]["negative"] == 0 for c in small)
    _points, _meta, text = va._ground_rate_by_group(rows, "Country", "neg", top_k=15)
    # Six countries under 20 rows are at 0% too: "lowest" over the 10 larger
    # ones is not the lowest of all 16, so the sentence says whose it is.
    assert "among the 10 groups with at least 20 rows, highest for Germany (6.6% of 256 rows)" in text, text
    # The bars stop at top_k; "lowest" does not.
    _p, _m, cut = va._ground_rate_by_group(rows, "Country", "neg", top_k=3)
    assert "lowest for 0.0%, a tie between 5 groups" in cut, cut


def test_a_rate_by_group_unchanged_when_the_filter_changes_nothing():
    rows = [{"g": g, "y": int(i < k)} for g, n, k in (("a", 40, 20), ("b", 40, 10), ("c", 5, 2))
            for i in range(n)]
    _p, _m, text = va._ground_rate_by_group(rows, "g", "y")
    assert "highest for a (50.0% of 40 rows), lowest for b (25.0% of 40 rows)" in text
    assert "among" not in text


def test_a_rate_by_bins_says_the_trend_is_over_the_larger_bands():
    rows = [{"x": x, "y": int(i < k)} for x, n, k in ((1, 40, 4), (2, 40, 8), (3, 40, 12), (4, 3, 0))
            for i in range(n)]
    _p, _m, text = va._ground_rate_by_bins(rows, "x", "y", 4)
    assert "rises with each higher x band" in text
    assert "Across the 3 bands with at least 20 rows" in text, text
    assert "among the 3 bands with at least 20 rows" in text, text


def test_the_cli_highest_rate_says_its_minimum_rows():
    from avaloka.goal_analysis import MIN_OUTCOME_ROWS, _binary_detail_of

    e = {"quantity": False, "groups": {"Spain": (5, 0.6), "Germany": (256, 0.066), "France": (197, 0.015)}}
    text = _binary_detail_of("Country", e, 0.015)
    assert text.startswith("highest rate at Country=Germany (6.6% vs 1.5% overall)")
    assert text.endswith(f"among Country values with at least {MIN_OUTCOME_ROWS} rows"), text
    e["groups"].pop("Spain")
    assert "among" not in _binary_detail_of("Country", e, 0.015)


# --------------------------------------------------------------------------- #
# 2. The overview badges
# --------------------------------------------------------------------------- #
def test_the_overview_badges_carry_their_provenance():
    vocab, model = UI["overview_vocabulary"], UI["overview_model_domain"]
    assert vocab["domain_segments"] == [{"text": "retail", "kind": "plain"}]
    assert _kinds(model["domain_segments"]) == {"commentary"}
    for ov in (vocab, model):
        assert ov["readiness_score_segments"] == [{"text": f"{ov['readiness_score']}/100", "kind": "commentary"}]
    # Rebuilt now from the same stored fields, the backend still says so.
    fresh = {k: v for k, v in copy.deepcopy(vocab).items() if not k.endswith("_segments")}
    again = va.mark_profile_commentary(fresh)
    assert again.get("domain_segments") == vocab["domain_segments"]
    assert again.get("readiness_score_segments") == vocab["readiness_score_segments"]


# --------------------------------------------------------------------------- #
# 3. Column meanings and predictive values
# --------------------------------------------------------------------------- #
def test_column_meaning_and_predictive_value_are_the_models():
    stats = [{"name": "StockCode"}, {"name": "Country"}]
    va._enrich_column_stats(stats, {
        "column_explanations": {"StockCode": "Code representing the specific product sold"},
        "predictive_values": {"StockCode": "high", "Country": "low"}})
    code = stats[0]
    assert code.get("meaning_segments") == [{"text": "Code representing the specific product sold", "kind": "commentary"}]
    assert code["meaning"] == "Code representing the specific product sold" + SUFFIX
    assert code.get("meaning_source") == "model" and code.get("predictive_value_source") == "model"
    assert stats[1].get("predictive_value_source") == "model" and "meaning" not in stats[1]
    for row in UI["column_stats"]:
        if row.get("meaning"):
            rs.check(row["meaning"], row["meaning_segments"])
            assert _kinds(row["meaning_segments"]) == {"commentary"}


# --------------------------------------------------------------------------- #
# 4. The chart explainer's answer
# --------------------------------------------------------------------------- #
def test_the_explainers_answer_is_commentary_and_its_fallbacks_plain(monkeypatch):
    from app.agents import chart_explainer as ce

    said = "Germany has the highest return share. The UK is close to the overall rate."

    class _Resp:
        content = json.dumps({"answer": said, "focus_chart_ids": ["llm_0"]})

    monkeypatch.setattr(ce, "viz_llm", object())
    monkeypatch.setattr(ce, "invoke_viz_llm", lambda *_a, **_k: _Resp())
    charts = [{"id": "llm_0", "title": "Return rate by country", "points": []}]
    out = ce.explain_charts("Which country returns most?", charts)
    assert out.get("answer_segments") and _kinds(out["answer_segments"]) == {"commentary"}
    assert rs.text_of(out["answer_segments"]) == said
    rs.check(out["answer"], out["answer_segments"])
    from app.api.schemas import InsightExplainOut
    assert InsightExplainOut(**out).answer_segments == out["answer_segments"]

    _Resp.content = "not json"
    fallback = ce.explain_charts("?", charts)
    assert fallback["answer"] == ce.FALLBACK_ANSWER
    assert fallback["answer_segments"] == [{"text": ce.FALLBACK_ANSWER, "kind": "plain"}]


# --------------------------------------------------------------------------- #
# 5. "All requested metrics are complete"
# --------------------------------------------------------------------------- #
def test_the_metrics_message_claims_only_what_code_checked(tmp_path, monkeypatch):
    from app.agents import multi_action as multi
    from app.api import workflow

    source = tmp_path / "sales.csv"
    source.write_text("region,sales\nEast,10\nWest,30\n")
    plan = multi.ActionPlan(mode="aggregate", actions=[
        multi.MetricAction(id="metric_1", operation="mean", column="sales", group_by=[], output_name="mean_sales"),
        multi.MetricAction(id="metric_2", operation="max", column="sales", group_by=[], output_name="max_sales"),
    ])
    monkeypatch.setattr(workflow, "plan_metric_actions", lambda *_: plan)
    monkeypatch.setattr(workflow, "_detect_ambiguous_prompt_details", lambda _: None)
    state = {"user_prompt": "mean and max sales and the trend by region", "messages": [],
             "schema": {"region": "object", "sales": "int64"}, "input_data_type": "csv",
             "data_source_location": str(source), "output_location": str(tmp_path / "out.csv")}
    out = workflow.execute_structured_actions_node(state | workflow.coding_subgraph_node(state))
    text = out["messages"][-1].content
    # Whether the plan covers "the trend by region" was the reviewer model's call.
    assert "requested" not in text, text
    assert text.startswith("All 2 metrics in the plan were calculated."), text
    assert _kinds(rs.of_message(out["messages"][-1])) == {"plain"}


# --------------------------------------------------------------------------- #
# 6. Report narrative and trace reasons
# --------------------------------------------------------------------------- #
def test_a_model_rewritten_report_narrative_is_marked():
    from avaloka.reports.render import executive_html

    data = {"goal": "g", "mission_id": "m", "kind": "analyze", "kpis": [], "economics": [], "findings": [],
            "verdict": "pass", "max_level": 1, "limitations": [], "units": []}
    rewrite = "Avaloka profiled the retail file. It found a clean dataset."
    html = executive_html(dict(data, narrative=rewrite, narrative_commentary=True))
    assert html.count(SUFFIX.strip()) == 2 and 'class="commentary"' in html
    plain = executive_html(dict(data, narrative="Avaloka profiled 8 columns.", narrative_commentary=False))
    assert SUFFIX.strip() not in plain and "Avaloka profiled 8 columns." in plain


def test_the_reporter_knows_when_the_model_rewrote_the_narrative(monkeypatch):
    from avaloka.fireflies import reporter as rep

    monkeypatch.setattr(rep, "narrate", lambda client, system, prompt, fallback: ("Rewritten by a model.", 0.0))
    monkeypatch.setattr(rep, "LLMClient", lambda: None)
    r = rep.Reporter.__new__(rep.Reporter)

    class _Ctx:
        class execution_mode:
            value = "cloud"
    r.ctx = _Ctx()
    profile = {"quality": {"score": 90}, "dataset": {"n_cols": 8, "n_rows": 20000}, "domain": {}}
    text = r._narrative(profile, {"task": "exploratory"}, None, {"verdict": "pass"})
    base = getattr(r, "_narrative_base", None)
    assert text == "Rewritten by a model." and isinstance(base, str) and text != base
    assert base.startswith("Avaloka profiled 8 columns")


def test_trace_reasons_a_model_wrote_are_commentary():
    steps = {s["node"] if "node" in s else s["agent"]: s for s in UI["trace_steps"]}
    marked = [s for s in UI["trace_steps"] if (s.get("decision") or {}).get("reason_segments")]
    assert len(marked) == 2
    for s in marked:
        d = s["decision"]
        assert rs.text_of(d["reason_segments"]) == d["reason"]
        assert _kinds(d["reason_segments"]) == {"commentary"}
    from app.trace.describe import describe_step

    plan = describe_step("plan_etl", {}, {"plan": "Group by Country and compute the share of negative rows.",
                                          "ready_to_code": True})
    assert plan["decision"]["value"] == "code"                       # the route: code's
    assert _kinds(plan["decision"].get("reason_segments")) == {"commentary"}
    from langchain_core.messages import AIMessage

    said = rs.tag(AIMessage(content="Germany returns most. 6.6% of its rows."),
                  rs.join(rs.commentary("Germany returns most."), rs.plain(" 6.6% of its rows.")))
    answer = describe_step("narrate_result", {}, {"messages": [said]})["calculation"]
    assert answer.get("answer_segments") == rs.of_message(said)
    assert steps  # every stored step rendered


# --------------------------------------------------------------------------- #
# The verifier's step states its count, never "all supported" (owner's ruling)
# --------------------------------------------------------------------------- #
def test_the_verification_step_states_the_count_not_a_verdict():
    from app.trace.build import build_turn, planner_graph
    from app.trace.describe import describe_step

    clean = describe_step("verify_claims", {}, {"verification_report": {
        "safe_to_present": True, "claims_checked": 4, "n_problems": 0, "findings": []}})
    assert clean["decision"]["reason"] == "Checked 4 claim(s) against the numbers"
    assert "supported" not in clean["decision"]["reason"]
    flagged = describe_step("verify_claims", {}, {"verification_report": {
        "safe_to_present": False, "claims_checked": 4, "n_problems": 1, "findings": []}})
    assert flagged["decision"]["reason"] == "1 of 4 claim(s) need attention"      # unchanged

    step = {"kind": "agent", "step_id": "v", "parent_id": None, "seq": 1, "node": "verify_claims",
            "agent": "Claim verifier", "status": "ok", "latency_ms": 3, "inference": [],
            "decision": {"type": "evidence", "value": "safe"},
            "calculation": {"values": {"claims_checked": 4, "problems": 0}}}
    node = next(n for n in planner_graph(build_turn([step]))["nodes"] if n["type"] == "evidence")
    assert node["subtitle"] == "4 claim(s) checked"
    from app.trace.build import _subtitle
    # Training's evidence ("beats_baseline") is not the verifier's count.
    assert _subtitle({"decision": {"type": "evidence", "value": "beats_baseline"},
                      "calculation": {"values": {}}}) == "beats_baseline"
