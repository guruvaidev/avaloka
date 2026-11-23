"""A superlative or a threshold the computed table contradicts (1.6 demo guide).

The 1.6 demo-guide run (develop-1.6 @ 7fd77f09, gpt-oss-120b) showed two false
chat replies on the YouTube sample, each beside the table it summarised, and
no verifier notice on either:

- "The lowest average is for Autos & Vehicles at about 17.9M." The table's
  lowest average is Travel & Events, 12,500,000 (Autos & Vehicles: 17,850,000).
- "No category exceeds 30 million" over median subscribers by category, where
  Trailers is 39,000,000.

Why they passed: ``check_ordering_contradicted`` locates a claim's *figure* in
a column and never reads the group the claim names, and it skips every scaled
figure ("17.9M", "17.9 million"); no check read a threshold. Both sentences
fell to ``check_invented_numbers``, whose UNSUPPORTED verdict leaves
``verification_safe_to_present`` True, and the web app shows the notice only
when it is False.

Rows are recomputed from ``app/sample_data/Global_YouTube_Statistics.csv``
(``test_the_fixture_rows_are_the_sample_data``); the replies are transcribed
from the run's screenshots (tests/fixtures/verifier_superlatives_16).
"""

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import Verdict, claim_verifier_node, verify_claims

ROOT = Path(__file__).resolve().parents[1]
TURNS = json.loads((Path(__file__).parent / "fixtures" / "verifier_superlatives_16" / "turns.json").read_text())
MEAN, MEDIAN = TURNS["mean"], TURNS["median"]
NNBSP = " "
#: The gap gpt-oss writes before "M"/"million" and around "&", a plain space, and none.
GAPS = [NNBSP, " ", ""]


def _node(turn, reply=None, rows=None, **extra):
    state = {"messages": [HumanMessage(content=turn["question"]), AIMessage(content=reply or turn["reply"])],
             "output_json": turn["rows"] if rows is None else rows, **extra}
    return claim_verifier_node(state)


def _with_gap(text, gap):
    # "Autos & Vehicles" keeps its spaces when the gap is removed: only the
    # figure's gap ("17.9M") goes.
    if gap == "":
        return text.replace(f"{NNBSP}&{NNBSP}", " & ").replace(NNBSP, "")
    return text.replace(NNBSP, gap)


def _check(claim, rows):
    finding = _flags(claim, rows)
    return finding.check if finding is not None else None


def _flags(claim, rows):
    """The contradiction the verifier finds in *claim* over *rows*, or None.

    Through the public ``verify_claims``, so a tree without these checks fails
    these tests on their assertions, not on an import."""
    report = verify_claims(claim, {"execution_output": rows})
    # 1.7 counts only the claims a check judged (claims_checked); a skipped
    # claim is still seen.
    assert report.claims_seen == 1, claim
    return next((f for f in report.findings if f.verdict is Verdict.CONTRADICTED), None)


# --------------------------------------------------------------------------- #
# The fixture is the demo's data
# --------------------------------------------------------------------------- #

def test_the_fixture_rows_are_the_sample_data():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(ROOT / "app" / "sample_data" / "Global_YouTube_Statistics.csv", encoding="latin-1")
    grouped = df.groupby("category", dropna=False)["subscribers"]

    def label(k):
        return None if isinstance(k, float) and math.isnan(k) else k

    assert {label(k): round(v, 2) for k, v in grouped.mean().items()} == \
        {r["category"]: r["average_subscribers"] for r in MEAN["rows"]}
    assert {label(k): v for k, v in grouped.median().items()} == \
        {r["category"]: r["median_subscribers"] for r in MEDIAN["rows"]}
    means = {r["category"]: r["average_subscribers"] for r in MEAN["rows"] if r["category"]}
    medians = {r["category"]: r["median_subscribers"] for r in MEDIAN["rows"] if r["category"]}
    assert min(means, key=means.get) == "Travel & Events" and means["Autos & Vehicles"] == 17_850_000
    assert max(medians.values()) == medians["Trailers"] == 39_000_000


# --------------------------------------------------------------------------- #
# The two demo replies (bites on 7fd77f09)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gap", GAPS, ids=["nnbsp", "space", "none"])
def test_demo_lowest_average_is_not_autos_and_vehicles(gap):
    reply = _with_gap(MEAN["reply"], gap)
    claim = _with_gap(MEAN["false_claim"], gap)
    out = _node(MEAN, reply)
    found = [f for f in out["verification_report"]["findings"] if f["claim"] == claim]
    assert found and found[0]["check"] == "superlative_contradicted", out["verification_report"]["findings"]
    assert found[0]["verdict"] == "contradicted"
    assert out["verification_safe_to_present"] is False
    assert found[0]["detail"] == ("Autos & Vehicles's average_subscribers (17,850,000) is not the lowest: "
                                  "Travel & Events has 12,500,000")        # 1.7's wording (claim_extremes)


@pytest.mark.parametrize("gap", GAPS, ids=["nnbsp", "space", "none"])
def test_demo_no_category_exceeds_30_million_is_broken_by_trailers(gap):
    reply = _with_gap(MEDIAN["reply"], gap)
    claim = _with_gap(MEDIAN["false_claim"], gap)
    out = _node(MEDIAN, reply)
    found = [f for f in out["verification_report"]["findings"] if f["claim"] == claim]
    assert found and found[0]["check"] == "threshold_contradicted", out["verification_report"]["findings"]
    assert found[0]["verdict"] == "contradicted"
    assert out["verification_safe_to_present"] is False
    assert found[0]["detail"] == ("says no category is above 30 million, but Trailers is at 39,000,000 "
                                  "by median_subscribers in the computed result")


def test_the_notice_carries_only_the_two_false_claims_as_contradicted():
    for turn in (MEAN, MEDIAN):
        report = _node(turn)["verification_report"]
        contradicted = [f["claim"] for f in report["findings"] if f["verdict"] == "contradicted"]
        assert contradicted == [turn["false_claim"]], report["findings"]


@pytest.mark.parametrize("gap", GAPS, ids=["nnbsp", "space", "none"])
@pytest.mark.parametrize("name", ["mean", "median"])
def test_every_line_the_notice_shows_for_the_demo_replies_is_true(name, gap):
    """Once a superlative or threshold makes the reply unsafe, the notice lists
    every problem. "cites 41.6, which does not appear in the computed evidence"
    for "41.6\u202fM" (41,615,384.62 is in the table) was false and had been
    hidden only because nothing else made the reply unsafe."""
    turn = TURNS[name]
    report = _node(turn, _with_gap(turn["reply"], gap))["verification_report"]
    for finding in report["findings"]:
        if finding["check"] != "invented_number":
            continue
        cited = finding["detail"].removeprefix("cites ").split(", which")[0]
        # The figure the line cites is the one the sentence says, scale and all:
        # "cites 41.6" for "41.6\u202fM" names a figure the reply never gave.
        assert not re.search(re.escape(cited) + r"[\s\u202f\u00a0]?[KMBT]\b", finding["claim"]), finding


@pytest.mark.parametrize("text,value", [
    ("41.6\u202fM", 41_600_000.0),          # gpt-oss's narrow no-break space
    ("12.5\u00a0M", 12_500_000.0),
    ("41.6 M", 41_600_000.0),               # a decimal figure, then a plain space
    ("28 M", 28_000_000.0),                 # two digits, then a plain space
    ("17.9M", 17_900_000.0),
    ("Type 2 B", 2.0),                      # a label, not two billion
    ("in 2019 M&A deals", 2019.0),
    ("5 m away", 5.0),                      # lower case keeps the old rule
])
def test_a_scale_letter_after_a_space_is_read(text, value):
    from app.agents.claim_verifier import _NUMBER, _parse_number

    assert _parse_number(_NUMBER.search(text).group(0)) == value


# --------------------------------------------------------------------------- #
# True claims on the same tables pass
# --------------------------------------------------------------------------- #

TRUE_ON_MEAN = [
    "Shows have the highest average subscribers at about 41.6M.",
    "Travel & Events has the lowest average subscribers.",
    "The lowest average is for Travel & Events at 12.5M.",
    "The highest average is Shows.",
    "Shows is the highest.",
    "No category exceeds 42 million.",
    "All categories are above 12 million.",
    "None is below 12 million.",
    # The demo reply's own true sentences.
    "- The top three categories (Shows, Trailers, Film & Animation) all exceed 28 M, showing a tight "
    "cluster at the high end.",
    "- The largest gap is between Shows (41.6 M) and Travel & Events (12.5 M).",
]
TRUE_ON_MEDIAN = [
    "- The highest median is Trailers at 39.0 million, followed by Nonprofits & Activism at 27.75 million.",
    "- The lowest median is Travel & Events at 12.5 million.",
    "No category exceeds 39 million.",
    "No category's median is below 12 million.",
    "Every category has a median above 12 million.",
    "- Shows (23.1 million) sits in the middle of the range, higher than most categories but below the "
    "top performers.",
]


@pytest.mark.parametrize("claim", TRUE_ON_MEAN)
def test_true_claims_on_the_mean_table_pass(claim):
    assert _flags(claim, MEAN["rows"]) is None


@pytest.mark.parametrize("claim", TRUE_ON_MEDIAN)
def test_true_claims_on_the_median_table_pass(claim):
    assert _flags(claim, MEDIAN["rows"]) is None


def test_a_reply_of_true_claims_stays_presentable():
    reply = "\n".join(["Here is what the table shows."] + [f"- {c.lstrip('- ')}" for c in TRUE_ON_MEDIAN])
    out = _node(MEDIAN, reply)
    assert out["verification_safe_to_present"] is True, out["verification_report"]["findings"]
    assert not any(f["check"] in ("superlative_contradicted", "threshold_contradicted")
                   for f in out["verification_report"]["findings"])


@pytest.mark.parametrize("claim", [
    "Gaming Shorts has the lowest average subscribers.",       # no such group in the result
    "The lowest average is for Kids & Family.",
    "No channel exceeds 30 million.",                           # channels, not the result's categories
    "No category exceeds 30 million views.",                    # a measure the result does not hold
    "No category exceeds 30%.",
])
def test_claims_the_result_cannot_judge_are_skipped(claim):
    assert _flags(claim, MEDIAN["rows"]) is None


# --------------------------------------------------------------------------- #
# Wording variants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("claim,check", [
    ("Autos & Vehicles is the lowest.", "superlative_contradicted"),
    ("Autos & Vehicles has the lowest average subscribers.", "superlative_contradicted"),
    ("Autos and Vehicles has the least subscribers on average.", "superlative_contradicted"),
    ("Autos & Vehicles sits at the bottom.", "superlative_contradicted"),
    ("Trailers has the most subscribers on average.", "superlative_contradicted"),
    ("The top average is for Trailers.", "superlative_contradicted"),
    ("**Comedy** ranks highest.", "superlative_contradicted"),
    ("No category is above 30 million.", "threshold_contradicted"),
    ("None exceeds 30 million.", "threshold_contradicted"),
    ("None of the categories exceeds 30M.", "threshold_contradicted"),
    ("Not a single category goes above 40 million.", "threshold_contradicted"),
    ("Every category is above 15 million.", "threshold_contradicted"),
    ("All categories exceed 15 million subscribers.", "threshold_contradicted"),
    ("None below 15 million.", "threshold_contradicted"),
    ("No category falls below 15,000,000.", "threshold_contradicted"),
    ("Across categories, none exceeds 30 million.", "threshold_contradicted"),
])
def test_wording_variants_are_flagged(claim, check):
    finding = _flags(claim, MEAN["rows"])
    assert finding is not None and finding.check == check, claim


def test_median_wording_variants_are_flagged():
    assert _check("The lowest median is Autos & Vehicles.", MEDIAN["rows"]) == "superlative_contradicted"
    assert _check("No category has a median above 30 million.", MEDIAN["rows"]) == "threshold_contradicted"
    assert _check("No category's median exceeds 30 million.", MEDIAN["rows"]) == "threshold_contradicted"


# --------------------------------------------------------------------------- #
# Unclear claims are skipped, not guessed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("claim", [
    "Autos & Vehicles has the second lowest average.",          # a rank, not the extreme
    "Autos & Vehicles is one of the lowest.",
    "Autos & Vehicles is not the lowest.",
    "Entertainment is the largest category.",                   # a size the table may not hold
    "The top three categories are Autos & Vehicles, Comedy and Gaming.",
    "Among the categories shown, the lowest is Autos & Vehicles.",
    "The lowest are Autos & Vehicles and Pets & Animals.",      # two groups named
    "Excluding Travel & Events, the lowest is Autos & Vehicles.",
    "The highest-earning channel is in Music.",
    "Almost no category exceeds 30 million.",
    "No category except Trailers exceeds 30 million.",
    "No other category exceeds 30 million.",
])
def test_ambiguous_or_qualified_claims_are_skipped(claim):
    assert _flags(claim, MEAN["rows"]) is None


def test_columns_that_disagree_are_skipped_but_a_measure_word_resolves_them():
    rows = [{"category": "Alpha", "mean_subscribers": 10.0, "n_channels": 900},
            {"category": "Beta", "mean_subscribers": 30.0, "n_channels": 2},
            {"category": "Gamma", "mean_subscribers": 20.0, "n_channels": 50}]
    assert _flags("Alpha is the lowest.", rows) is None          # lowest mean, highest count
    assert _flags("Alpha has the lowest mean subscribers.", rows) is None
    assert _check("Beta has the lowest mean subscribers.", rows) == "superlative_contradicted"
    assert _check("Gamma has the lowest average subscribers.", rows) == "superlative_contradicted"
    assert _flags("No category exceeds 35.", rows) is None        # the mean holds it, the count (900) does not
    assert _check("No category exceeds 25.", rows) == "threshold_contradicted"   # both break it
    assert _check("No category exceeds 25 subscribers.", rows) == "threshold_contradicted"
    assert _flags("No category exceeds 35 subscribers.", rows) is None


def test_results_that_are_not_one_row_per_group_are_skipped():
    # 1.7's rule (claim_extremes, as in the _2x twin of this test): a label that
    # names several rows -- Alpha in two regions, or in two years -- is
    # ambiguous and skipped. A label that names one row is judged, even beside
    # a second label column (1.6 skipped those too).
    two_labels = [{"region": "North", "category": "Alpha", "v": 1.0},
                  {"region": "North", "category": "Beta", "v": 3.0},
                  {"region": "South", "category": "Alpha", "v": 5.0},
                  {"region": "South", "category": "Beta", "v": 2.0}]
    repeated = [{"category": "Alpha", "year": 2020, "v": 1.0}, {"category": "Alpha", "year": 2021, "v": 5.0},
                {"category": "Beta", "year": 2020, "v": 3.0}]
    one_label = [{"category": "Alpha", "v": 1.0}, {"category": "Beta", "v": 3.0}]
    # The same claims bite on the one-row-per-group shape, so the skips below are not vacuous.
    assert _check("Alpha has the highest v.", one_label) == "superlative_contradicted"
    assert _check("No category exceeds 2.", one_label) == "threshold_contradicted"
    for rows in (two_labels, repeated):
        assert _flags("Alpha has the highest v.", rows) is None
        assert _flags("No category exceeds 2.", rows) is None


def test_a_total_row_and_the_missing_label_row_are_not_groups():
    rows = [{"category": "Alpha", "sales": 10.0}, {"category": "Beta", "sales": 20.0},
            {"category": "Total", "sales": 30.0}, {"category": None, "sales": 1.0}]
    assert _flags("Beta has the highest sales.", rows) is None
    assert _flags("Alpha has the lowest sales.", rows) is None
    assert _flags("No category exceeds 25.", rows) is None
    assert _check("Alpha has the highest sales.", rows) == "superlative_contradicted"
    assert _check("No category exceeds 15.", rows) == "threshold_contradicted"


def test_a_bound_is_read_at_the_precision_it_is_stated():
    rows = [{"category": "Alpha", "v": 30_400_000.0}, {"category": "Beta", "v": 12.0}]
    assert _flags("No category exceeds 30 million.", rows) is None          # 30.4M rounds to 30M
    assert _check("No category exceeds 30.0 million.", rows) == "threshold_contradicted"


def test_cells_read_from_the_csv_data_uri_are_numbers():
    """output_file_data decodes to strings; the check reads them as numbers."""
    rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in MEDIAN["rows"]]
    assert _check(MEDIAN["false_claim"], rows) == "threshold_contradicted"
    assert _flags("The lowest median is Travel & Events at 12.5 million.", rows) is None


# --------------------------------------------------------------------------- #
# End to end: the production graph's verify_claims node, as the chat path runs it
# --------------------------------------------------------------------------- #

def _csv_data_uri(rows):
    pd = pytest.importorskip("pandas")
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    return {"filename": "out.csv", "content": "data:text/csv;base64," + base64.b64encode(csv_bytes).decode(),
            "size": len(csv_bytes)}


@pytest.mark.parametrize("name", ["mean", "median"])
@pytest.mark.parametrize("evidence", ["output_json", "output_file_data"])
def test_end_to_end_through_the_production_graph(name, evidence):
    """The chat path: the turn's reply and its executed result reach the graph's
    verify_claims node, whose two fields the API returns and the web notice reads."""
    pytest.importorskip("celery", reason="production workflow requires the API scheduler dependencies")
    from app.api import workflow

    turn = TURNS[name]
    produced = {"output_json": turn["rows"]} if evidence == "output_json" \
        else {"output_file_data": _csv_data_uri(turn["rows"])}

    def conversational(_state):
        return {"avaloka_mode": "analyzing", "avaloka_intent": "statistical_analysis",
                "delegate_to_planner": True}

    planner = Mock(return_value={
        "messages": [AIMessage(content=turn["reply"])],
        "execution_result": {"status": "success"},
        "ready_to_summarize": False,
        "ready_to_code": False,
        **produced,
    })
    with patch.object(workflow, "memory_injection_node", return_value={}), \
            patch.object(workflow, "avaloka_agent_node", side_effect=conversational), \
            patch.object(workflow, "plan_etl_job", planner):
        graph = workflow.build_graph().compile()
        result = graph.invoke({"messages": [HumanMessage(content=turn["question"])], "planner_definition": {}})

    planner.assert_called_once()
    assert result["verification_safe_to_present"] is False
    findings = result["verification_report"]["findings"]
    contradicted = [f for f in findings if f["verdict"] == "contradicted"]
    assert [f["claim"] for f in contradicted] == [turn["false_claim"]], findings


def test_end_to_end_narrator_reply_is_what_the_verifier_reads():
    """result_narrator_node appends the model's reply; claim_verifier_node checks
    that reply against the same output_file_data the narrator summarised."""
    from app.agents import result_narrator

    turn = MEAN
    state = {"messages": [HumanMessage(content=turn["question"])],
             "execution_result": {"status": "success"},
             "output_file_data": _csv_data_uri(turn["rows"])}
    with patch.object(result_narrator, "narrator_llm", object()), \
            patch.object(result_narrator, "_narrate", return_value=turn["reply"]):
        state.update(result_narrator.result_narrator_node(state))
    out = claim_verifier_node(state)
    assert out["verification_safe_to_present"] is False
    assert [f["check"] for f in out["verification_report"]["findings"] if f["verdict"] == "contradicted"] == \
        ["superlative_contradicted"]


def test_verify_claims_counts_the_turn():
    report = verify_claims(MEDIAN["reply"], {"execution_output": MEDIAN["rows"]})
    assert report.claims_checked == 5
    assert not report.safe_to_present
