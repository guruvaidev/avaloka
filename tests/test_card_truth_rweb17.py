"""Code-written plain text on rc17's upload cards must be true (rweb17 regrade).

The fixture (tests/fixtures/card_truth/rweb17_card_truth.json) holds only the
fields these checks read, taken from the stored rweb17 upload states:

1. A model-chosen aggregate the code did not compute was summed and printed
   under the model's name: "Rate_negative Quantity by Country: highest is
   United Kingdom (138,377)"; the share of negative rows is highest for
   Germany (6.6%). A row share is now computed as one; any other unknown name
   draws no chart.
2. "Outcome is most often “Recovered” (20 of 60 rows)" over a 20/20/20 tie.
3. "Visit_date is a row identifier ..." for a column of visit dates.
4. "The gap between groups is small (26% ...)": a size word the template
   cannot justify.
5. c-strict: the vocabulary's domain reasoning is code (plain); the
   profiling model's readiness score is commentary.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app.agents import visualization_agent as va
from app.api import reply_segments as rs

FIXTURE = Path(__file__).parent / "fixtures" / "card_truth" / "rweb17_card_truth.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
WORKBOOK = Path(__file__).parent / "fixtures" / "null_words" / "three_sheets.xlsx"
SUFFIX = " (commentary — not checked)"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _llm_chart(spec):
    """A chart as the spec builder hands it to grounding."""
    ch = copy.deepcopy(spec)
    ch.update(title=ch.get("llm_title"), insight=ch.get("llm_insight"),
              reason=ch.get("llm_insight"), insight_source="llm" if ch.get("llm_insight") else None)
    return ch


def _plain(segments):
    return "".join(s["text"] for s in segments or [] if s["kind"] == rs.PLAIN)


def _commentary(segments):
    return [s["text"] for s in segments or [] if s["kind"] == rs.COMMENTARY]


# --------------------------------------------------------------------------- #
# 1. An aggregate the code cannot compute is never summed under its name
# --------------------------------------------------------------------------- #
MESSY = CASES["messy_xlsx"]
PROFILES = {p["name"]: p for p in MESSY["profiles"]}
SHARE_OF = {  # chart id -> (column, which rows count)
    "llm_0": ("Quantity", lambda v: v is not None and float(v) < 0),
    "llm_3": ("CustomerID", lambda v: v is None or v != v),
}


def _expected_shares(rows, column, hit):
    groups = {}
    for r in rows:
        if r.get("Country") is None:
            continue
        v = r.get(column)
        if column == "Quantity" and v is None:
            continue
        n, k = groups.get(r["Country"], (0, 0))
        groups[r["Country"]] = (n + 1, k + int(hit(v)))
    return {c: round(100 * k / n, 1) for c, (n, k) in groups.items()}


@pytest.mark.parametrize("chart_id", sorted(SHARE_OF))
def test_rate_aggregate_is_the_share_of_each_groups_rows(chart_id):
    spec = next(c for c in MESSY["charts"] if c["id"] == chart_id)
    named = spec["encodings"]["y"]["aggregate"]
    shown = MESSY["shown_insight_facts"][chart_id]
    assert shown.startswith(named.capitalize())          # what rc17 showed, plain

    ch = _llm_chart(spec)
    rows = MESSY["sample_rows"]
    assert va._ground_chart(ch, rows, PROFILES)
    va.mark_chart_commentary(ch)
    facts = ch["insight_facts"]
    assert named.capitalize() not in facts and named not in facts, facts
    assert facts.startswith(f"Share of rows with {'negative Quantity' if chart_id == 'llm_0' else 'no CustomerID'} by Country:")
    assert facts in _plain(ch["insight_segments"])
    column, hit = SHARE_OF[chart_id]
    expected = _expected_shares(rows, column, hit)
    points = ch["derived_data"]["points"]
    assert points and all(p["y"] == expected[p["x"]] for p in points), (points, expected)
    assert all(0 <= p["y"] <= 100 for p in points)
    assert ch["config"]["y_format"] == "percent"
    # The neutral title, if the model's is replaced, is the data sentence's lead-in.
    assert named.capitalize() not in va._neutral_title(facts, "bar", ch["encodings"])


def _full_file_rows():
    rows = []
    for country, g in MESSY["full_file"]["groups"].items():
        for i in range(g["rows"]):
            rows.append({"Country": country,
                         "Quantity": -1 if i < g["negative"] else 3,
                         "CustomerID": None if i < g["missing"] else 12345.0})
    assert len(rows) == MESSY["full_file"]["rows"] == 20_000
    return rows


def test_the_full_file_truth_is_shown_not_a_sum():
    """The owner's truth: Germany's negative-quantity share is the highest at
    6.6%, the UK's 1.5%; the UK's missing CustomerID share is 34.9%."""
    rows = _full_file_rows()
    by_id = {}
    for spec in MESSY["charts"]:
        ch = _llm_chart(spec)
        assert va._ground_chart(ch, rows, PROFILES, scope="in the full file")
        by_id[ch["id"]] = ch["insight_facts"]
    neg, missing = by_id["llm_0"], by_id["llm_3"]
    assert "highest among the 10 groups with at least 20 rows is Germany (6.6% of 256 rows)" in neg, neg
    assert "138,377" not in neg and "1426.6" not in neg
    assert "United Kingdom (34.9% of 19,012 rows)" in missing, missing
    assert "195.34M" not in missing and "7636.0" not in missing
    # Zero is shared by several countries: said as a tie, not one "lowest".
    assert re.search(r"lowest is 0\.0%, a tie between \d+ groups", neg), neg


@pytest.mark.parametrize("named", ["nunique", "pct_change", "rate", "std", "cumulative_sum"])
def test_an_aggregate_nothing_computes_draws_no_chart(named):
    spec = copy.deepcopy(next(c for c in MESSY["charts"] if c["id"] == "llm_0"))
    spec["encodings"]["y"]["aggregate"] = named
    ch = _llm_chart(spec)
    assert va._ground_chart(ch, MESSY["sample_rows"], PROFILES) is False
    assert va._ground_all([ch], MESSY["sample_rows"], PROFILES) == []


@pytest.mark.parametrize("ctype", ["line", "pie"])
def test_a_row_share_is_drawn_as_a_bar_only(ctype):
    spec = copy.deepcopy(next(c for c in MESSY["charts"] if c["id"] == "llm_0"))
    spec["type"] = ctype
    if ctype == "pie":
        spec["encodings"] = {"theta": spec["encodings"]["y"], "color": spec["encodings"]["x"]}
    assert va._ground_chart(_llm_chart(spec), MESSY["sample_rows"], PROFILES) is False


def test_the_trend_check_never_sums_what_the_card_did_not():
    from app.agents.card_quantifiers import ColumnFacts

    rows = [{"d": f"2024-0{m}-0{day}", "v": v} for m in range(1, 5) for day, v in ((1, 1), (2, 9), (3, 5))]
    facts = ColumnFacts(rows)
    assert facts.time_series("d", "v", "median") == [(f"2024-0{m}", 5.0) for m in range(1, 5)]
    assert facts.time_series("d", "v", "max") == [(f"2024-0{m}", 9.0) for m in range(1, 5)]
    assert facts.time_series("d", "v", "rate_negative") is None


# --------------------------------------------------------------------------- #
# 2. Ties are said as ties
# --------------------------------------------------------------------------- #
NULLX = CASES["nullxlsx"]


def test_a_three_way_tie_is_not_most_often_one_value():
    from avaloka.coordinator import _column_finding, merged_findings

    counts = NULLX["outcome_counts"]
    assert sorted(set(counts.values())) == [20] and len(counts) == 3
    col = {"name": "Outcome", "role": "categorical", "count": NULLX["n_rows"],
           "top_values": [{"value": k, "count": v} for k, v in counts.items()]}
    line = _column_finding(col, NULLX["n_rows"])
    assert "most often “Recovered”" in NULLX["shown_swarm_finding"]
    assert "most often" not in line, line
    assert "tie" in line and all(f"“{k}”" in line for k in counts), line
    assert "20 of 60 rows each (33.3%)" in line
    found = merged_findings({"n_rows": 60, "columns": [col]}, goal="what drives the Outcome")
    assert found and "most often" not in found[0]
    # A single top value is unchanged.
    col["top_values"][0]["count"] = 21
    assert _column_finding(col, 60).startswith("Outcome is most often “Recovered” (21 of 60 rows")


def test_highest_over_tied_groups_names_the_tie():
    rows = [{"g": g, "v": v} for g, v in (("a", 5), ("b", 5), ("c", 2), ("c", 2))]
    _points, _meta, text = va._ground_categories(rows, "g", "v", "mean", 15, False)
    assert "highest is a tie between a and b (5 each)" in text, text
    assert "The highest value is 2.5× the lowest." in text


# --------------------------------------------------------------------------- #
# 3. A date column is not a row identifier
# --------------------------------------------------------------------------- #
def test_visit_dates_are_not_called_a_row_identifier():
    from app.agents.claim_verifier import DataUsed, check_identifier_predictor, computed_drivers
    from avaloka.goal_analysis import is_identifier_column

    sheets = pd.read_excel(WORKBOOK, sheet_name=["2022", "2023"])
    for frame in sheets.values():
        assert frame["visit_date"].nunique() >= 0.98 * frame["visit_date"].notna().sum()
        assert not is_identifier_column("visit_date", frame["visit_date"])
        assert is_identifier_column("patient_id", frame["patient_id"])
    used = DataUsed(frames=[(name, frame) for name, frame in sheets.items()])
    ranked = computed_drivers(NULLX["question"], used)
    assert ranked is not None and "visit_date" not in ranked["identifiers"], ranked
    assert "visit date" in NULLX["reply_claim"]
    why = check_identifier_predictor(NULLX["reply_claim"], ranked)
    assert why is None or "visit_date is a row identifier" not in why, why


# --------------------------------------------------------------------------- #
# 4. The difference without a size word
# --------------------------------------------------------------------------- #
def test_the_gap_between_groups_is_stated_not_judged():
    stroke = CASES["stroke"]
    assert "small (26% from lowest to highest)" in stroke["shown_insight_facts"]
    ch = _llm_chart(stroke["chart"])
    profiles = {p["name"]: p for p in stroke["profiles"]}
    rows = stroke["sample_rows"]
    assert va._ground_chart(ch, rows, profiles)
    facts = ch["insight_facts"]
    assert "small" not in facts, facts
    groups = {}
    for r in rows:
        groups.setdefault(str(r["stroke"]), []).append(float(r["avg_glucose_level"]))
    means = sorted(sum(v) / len(v) for v in groups.values())
    ratio = means[-1] / means[0]
    if ratio < 1.5:
        assert f"The highest is {ratio - 1:.0%} above the lowest." in facts, facts
    else:
        assert f"{ratio:.1f}× the lowest" in facts, facts


# --------------------------------------------------------------------------- #
# 5. c-strict: provenance, not string matching
# --------------------------------------------------------------------------- #
def _profiling_result(domain, score=78):
    return {"status": "ok", "domain": domain,
            "quick_insights": {"data_readiness_score": score,
                               "readiness_reasoning": "High completeness but missing CustomerID.",
                               "red_flags": ["CustomerID missing >10% of rows."]}}


def _upload_message(profiling_result, n_rows=20_000, n_cols=8):
    viz = {"charts": [], "profile_overview": va.build_profile_overview(profiling_result)}
    va.mark_upload_commentary(viz, n_rows, n_cols)
    return viz


def test_the_vocabularys_domain_reasoning_is_plain(monkeypatch):
    from app.agents import profiling_agent as pa

    def _no_model(*_a, **_k):
        raise AssertionError("a confident vocabulary match never asks the model")

    monkeypatch.setattr(pa, "_call_domain", _no_model)
    domain, model_ok = pa._resolve_domain(MESSY["columns"], MESSY["sample_rows"][:20])
    assert model_ok is None and domain["category"] == "retail"
    assert domain["reasoning"] == "2 distinct cue(s) matched column names (Quantity, UnitPrice)"
    viz = _upload_message(_profiling_result(domain))
    segs = viz["upload_message_segments"]
    assert domain["reasoning"] in _plain(segs)
    assert "**retail** dataset · 20,000 rows × 8 columns." in _plain(segs)
    ov = viz["profile_overview"]
    assert ov["domain_reasoning_segments"] == [{"text": domain["reasoning"], "kind": "plain"}]
    assert SUFFIX not in ov["domain_reasoning"]
    rs.check(viz["upload_message"], segs)


def test_a_models_domain_and_reasoning_are_commentary(monkeypatch):
    from app.agents import profiling_agent as pa

    said = "Columns describe loan applications and repayment status."
    monkeypatch.setattr(pa, "_call_domain", lambda *_a, **_k: {
        "category": "finance", "confidence": 0.8, "reasoning": said})
    domain, model_ok = pa._resolve_domain(["a", "b", "c"], [{"a": 1, "b": 2, "c": 3}])
    assert model_ok is True and domain["source"] == "model" and domain["category"] != "generic"
    viz = _upload_message(_profiling_result(domain))
    segs = viz["upload_message_segments"]
    plain = _plain(segs)
    assert said not in plain and any(said in t for t in _commentary(segs))
    assert f"**{domain['category']}** dataset" not in plain
    assert viz["profile_overview"]["domain_reasoning"].endswith(SUFFIX)
    # Same words as reasoning, but no recorded provenance: still the model's.
    legacy = dict(domain, reasoning="2 distinct cue(s) matched column names (Quantity, UnitPrice)")
    legacy.pop("reasoning_source")
    assert "distinct cue(s)" not in _plain(_upload_message(_profiling_result(legacy))["upload_message_segments"])


@pytest.mark.parametrize("run", sorted(MESSY["profile_overview"]))
def test_the_profiling_models_readiness_score_is_commentary(run):
    score = MESSY["profile_overview"][run]["readiness_score"]
    domain = {"category": "retail", "source": "vocabulary", "reasoning_source": "vocabulary",
              "reasoning": "2 distinct cue(s) matched column names (Quantity, UnitPrice)"}
    viz = _upload_message(_profiling_result(domain, score))
    segs = viz["upload_message_segments"]
    line = f"**Data readiness: {score}/100.**"
    assert line not in _plain(segs) and "Data readiness" not in _plain(segs)
    assert f"Data readiness: {score}/100." in _commentary(segs), segs
    assert f"**Data readiness: {score}/100.{SUFFIX}**" in viz["upload_message"]
    assert line in rs.text_of(segs)        # the words are unchanged; only the marking is
    rs.check(viz["upload_message"], segs)


# --------------------------------------------------------------------------- #
# Audit: a model's guess is not a plain fact
# --------------------------------------------------------------------------- #
class _Entry:
    args = []

    @staticmethod
    def generate_key(app, task_id):
        return f"redbeat:{task_id}"

    @classmethod
    def from_key(cls, key, app):
        return cls()


def _scheduler(monkeypatch, ids, final=None):
    from app.agents import scheduler

    class _Sched:
        @staticmethod
        def get_task_ids(key, generate_key=False):
            return list(ids)

    class _Result:
        traceback = None

        def __init__(self, task_id, app=None):
            self.state, self.result = "SUCCESS", json.loads(json.dumps(final))

    monkeypatch.setattr(scheduler, "AvalokaEntry", _Entry)
    monkeypatch.setattr(scheduler, "AvalokaScheduler", _Sched)
    monkeypatch.setattr(scheduler, "AsyncResult", _Result)
    return scheduler


def test_has_not_run_yet_comes_from_the_run_list(monkeypatch):
    sched = _scheduler(monkeypatch, ids=[])
    state = {"messages": [], "task_operation": {"task_id": "t-1", "result_index": 0}}
    assert sched.handle_task_result(state)["messages"][-1].content == "Task t-1 hasn't run yet"


def test_a_past_runs_model_reply_stays_commentary(monkeypatch):
    reply = "Sales rose sharply in March."
    final = {"messages": [{"type": "ai", "content": reply}],
             "execution_result": {"status": "success"}, "output_location": "s3://out/x.csv",
             "output_file_data": None, "task_run_statistics": {"start": 0.0, "end": 1.0, "time_spent": 1.0}}
    sched = _scheduler(monkeypatch, ids=["exec-1"], final=final)
    state = {"messages": [], "task_operation": {"task_id": "t-1", "result_index": 0}}
    msg = sched.handle_task_result(state)["messages"][-1]
    segs = rs.of_message(msg)
    assert "ran successfully" in msg.content and reply in msg.content
    assert reply not in _plain(segs) and any(reply in t for t in _commentary(segs))
