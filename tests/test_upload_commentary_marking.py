"""2.0 (c-strict): every model-written sentence on the upload path is shown
as "commentary — not checked"; text written by code from the data stays
plain. Card leads and headlines, chart titles, key findings, watch-outs
(red flags, concerns) and the readiness reasoning are the model's.

The card sentences are the ones rc15's live check found shown unmarked
(rweb15 logs/shown15*.txt), on frames where the card_quantifiers checks
let them through: c-strict marks them anyway. Each case runs the real
upload assembly (build_visualization_config_from_sample) with the insight
model's reply stubbed. No LLM.
"""

from __future__ import annotations

import copy

import pytest

from app.agents import visualization_agent as va
from app.api import reply_segments as rs

SUFFIX = " (commentary — not checked)"
assert SUFFIX == rs.COMMENTARY_SUFFIX


def _check_contract(viz):
    """Every marked field renders to its plain string under C1's rules."""
    rs.check(viz["upload_message"], viz["upload_message_segments"])
    for ch in viz["charts"]:
        for key in ("insight", "headline", "title"):
            if ch.get(key + "_segments"):
                rs.check(ch[key], ch[key + "_segments"])
    ov = viz.get("profile_overview") or {}
    for key in ("readiness_reasoning", "domain_reasoning"):
        if ov.get(key + "_segments"):
            rs.check(ov[key], ov[key + "_segments"])
    for key in ("red_flags", "concerns", "strengths", "quick_wins", "analysis_suggestions"):
        for text, segs in zip(ov.get(key) or [], ov.get(key + "_segments") or []):
            rs.check(text, segs)


def _kinds(segments):
    return [s["kind"] for s in segments or []]


def _build(monkeypatch, rows, suggestions, profiling_result=None, use_llm=True):
    monkeypatch.setattr(va, "viz_llm", object())
    monkeypatch.setattr(va, "_suggest_for_plan", lambda plans, uc, effort: (copy.deepcopy(suggestions), None))
    viz = va.build_visualization_config_from_sample(
        "ds-commentary", rows, total_rows=len(rows), profiling_result=profiling_result, use_llm=use_llm)
    return viz


def _suggestion(ctype, title, x, sentence, y=None, **config):
    return {"type": ctype, "title": title, "x_field": x, "y_field": y, "insight": sentence,
            "intent": "comparison", "config": config}


def _llm_card(viz, sentence):
    found = [ch for ch in viz["charts"] if ch.get("llm_insight") == sentence]
    assert found, f"the card for {sentence!r} must be drawn"
    return found[0]


# rc15 sentences shown unmarked, with frames where #585's checks pass them.
URL_SENTENCE = ("A handful of recipe URLs account for the bulk of the records, suggesting that a few dishes "
                "are heavily over‑represented and may skew any analysis of recipe popularity.")
ONCE_SENTENCE = "A few instruction texts repeat while most are seen only once."
TREND_SENTENCE = "Monthly sales volume shows a steady increase over the period."
ASSOC_SENTENCE = "Higher unit prices are associated with lower quantities, implying price sensitivity among buyers."


def _url_rows():
    rows = [{"Url": u} for u, n in (("a", 400), ("b", 250), ("c", 100)) for _ in range(n)]
    return rows + [{"Url": f"other {i}"} for i in range(250)]


def _once_rows():
    rows = [{"Instructions": f"text {i}"} for i in range(90)]
    return rows + [{"Instructions": f"common {i}"} for i in range(3) for _ in range(20)]


def _trend_rows():
    rows = []
    for m in range(1, 9):
        for h in range(10):
            rows.append({"InvoiceDate": f"2010-{m:02d}-15 {8 + h:02d}:15:00", "Quantity": 100 * m})
    return rows


def _assoc_rows():
    rows = []
    for i in range(400):
        price = 1.0 + (i * 37 % 100) / 10.0
        rows.append({"UnitPrice": price, "Quantity": 20 - 3.0 * price + ((i * 7919) % 41) - 20})
    return rows


HANDFUL_SENTENCE = ("A small handful of products account for a disproportionate share of line items, "
                    "suggesting a focus on a few best‑selling items.")
TIP_SENTENCE = "Tip amounts are almost always zero when payment_type is 2."


def _stock_rows():
    counts = [900, 800, 700] + [16] * 97 + [8]                    # top 3 = 60% of lines
    return [{"StockCode": f"{84000 + i}{'A' if i % 7 == 3 else ''}", "Quantity": 3}
            for i, n in enumerate(counts) for _ in range(n)]


def _taxi_rows():
    rows = []
    for pay, n, zero in ((1.0, 3000, 200), (2.0, 850, 849)):     # cash: one tipped ride
        rows += [{"payment_type": pay, "tip_amount": 0.0}] * zero
        rows += [{"payment_type": pay, "tip_amount": 2.5}] * (n - zero)
    return rows


RC15_CASES = {
    "small_handful_of_products": (_stock_rows, _suggestion("bar", "Few products dominate", "StockCode",
                                                           HANDFUL_SENTENCE, y="Quantity", top_k=15)),
    "tips_zero_by_payment": (_taxi_rows, _suggestion("bar", "Cash rides are not tipped", "payment_type",
                                                     TIP_SENTENCE, y="tip_amount")),
    "handful_of_urls_bulk": (_url_rows, _suggestion("bar", "A few URLs dominate", "Url", URL_SENTENCE, top_k=10)),
    "most_seen_only_once": (_once_rows, _suggestion("bar", "Most texts are unique", "Instructions",
                                                    ONCE_SENTENCE, top_k=5)),
    "steady_increase": (_trend_rows, _suggestion("line", "Sales keep rising", "InvoiceDate", TREND_SENTENCE,
                                                 y="Quantity")),
    "associated_with": (_assoc_rows, _suggestion("scatter", "Price vs quantity", "UnitPrice", ASSOC_SENTENCE,
                                                 y="Quantity")),
}


@pytest.mark.parametrize("case", sorted(RC15_CASES))
def test_rc15_card_sentence_that_passed_the_checks_is_commentary(monkeypatch, case):
    make_rows, suggestion = RC15_CASES[case]
    viz = _build(monkeypatch, make_rows(), [suggestion])
    sentence = suggestion["insight"]
    card = _llm_card(viz, sentence)
    assert card["insight_source"] == "llm", "#585's checks let this one through"

    # The plain strings old readers show carry the suffix right after the model's words.
    assert card["headline"] == sentence + SUFFIX
    assert sentence + SUFFIX in card["insight"]
    assert card["reason"] == card["insight"]
    # Segments: the model's sentence is commentary, the data-written numbers plain.
    segs = card["insight_segments"]
    assert [s for s in segs if s["kind"] == "commentary"], segs
    assert all(sentence.rstrip(". ") in s["text"] for s in segs if s["kind"] == "commentary")
    assert card["headline_segments"] == [{"text": sentence, "kind": "commentary"}]
    # The data sentence is code's: plain, never suffixed.
    assert card["insight_facts"] and SUFFIX not in card["insight_facts"]
    for s in segs:
        if s["kind"] == "plain":
            assert s["text"].strip() == card["insight_facts"]
    # The model's title survived the checks: commentary too.
    assert card["title_segments"] == [{"text": suggestion["title"], "kind": "commentary"}]
    assert card["title"] == suggestion["title"] + SUFFIX
    # And it is listed under Key findings as commentary.
    msg = viz["upload_message_segments"]
    assert {"text": sentence, "kind": "commentary"} in msg
    assert sentence + SUFFIX in viz["upload_message"]
    _check_contract(viz)


def test_withheld_sentence_stays_withheld_and_its_card_is_plain(monkeypatch):
    """#585 withhold/replace is unchanged: the data sentence replaces it, plain."""
    rows = [{"Country": "United Kingdom"}] * 900 + [{"Country": c} for c in ("Germany", "France") for _ in range(50)]
    sentence = "No other country appears in the data, so the UK is the sole leader."
    viz = _build(monkeypatch, rows, [_suggestion("bar", "The UK stands alone", "Country", sentence, top_k=10)])
    card = next(ch for ch in viz["charts"] if (ch.get("derived_data") or {}).get("llm_insight_replaced") == sentence)
    assert card["headline"] is None and card["headline_segments"] is None
    assert card["insight_segments"] == [{"text": card["insight_facts"], "kind": "plain"}]
    assert card["insight"] == card["insight_facts"]
    assert card["title_segments"][0]["kind"] == "plain" and SUFFIX not in card["title"]
    assert "No other country" not in viz["upload_message"]


def _profiling_result():
    return {
        "status": "success",
        "domain": {"category": "Healthcare", "reasoning": "Columns describe patients and strokes."},
        "quick_insights": {
            "data_readiness_score": 72,
            "readiness_reasoning": "Mostly complete, but bmi has gaps.",
            "red_flags": ["No other Residence type is present", "No other work type is listed",
                          "work_type is about 90% Private"],
            "quick_wins": ["Impute bmi."],
        },
        "data_quality_insights": {"concerns": ["Class imbalance in stroke."], "strengths": ["No duplicates."]},
        "analysis_suggestions": ["Which work type has the highest stroke rate?"],
    }


def _stroke_rows():
    return [{"work_type": w, "Residence_type": "Urban", "age": 20 + (i % 60)} for i, (w, n) in enumerate(
        (("Private", 2925), ("Self-employed", 819), ("children", 687), ("Govt_job", 657), ("Never_worked", 22)))
        for _ in range(n)]
    _check_contract(viz)


def test_profile_watch_outs_and_readiness_are_commentary(monkeypatch):
    viz = _build(monkeypatch, _stroke_rows(), [], profiling_result=_profiling_result())
    ov = viz["profile_overview"]
    # #585 still withholds the false one and corrects the wrong number...
    assert not any("work type is listed" in f for f in ov["red_flags"])
    assert {"field": "red_flags", "original": "work_type is about 90% Private",
            "corrected": "work_type is 57.2% Private"} in ov["corrections"]
    assert "work_type is 57.2% Private" + SUFFIX in ov["red_flags"]
    # ...and every sentence that is shown, corrected or not, is the model's: commentary.
    assert ov["red_flags"] and all(f.endswith(SUFFIX) for f in ov["red_flags"])
    assert all(_kinds(s) == ["commentary"] for s in ov["red_flags_segments"])
    assert [s[0]["text"] + SUFFIX for s in ov["red_flags_segments"]] == ov["red_flags"]
    for key in ("concerns", "strengths", "quick_wins", "analysis_suggestions"):
        assert ov[key] and all(t.endswith(SUFFIX) for t in ov[key]), key
        assert all(_kinds(s) == ["commentary"] for s in ov[key + "_segments"]), key
    assert ov["readiness_reasoning"] == "Mostly complete, but bmi has gaps." + SUFFIX
    assert ov["readiness_reasoning_segments"] == [{"text": "Mostly complete, but bmi has gaps.", "kind": "commentary"}]
    assert ov["domain_reasoning_segments"][0]["kind"] == "commentary"
    # The score and the domain label are not sentences: unchanged.
    assert ov["readiness_score"] == 72 and ov["domain"] == "Healthcare"

    msg = viz["upload_message_segments"]
    assert "".join(s["text"] + (SUFFIX if s["kind"] == "commentary" else "") for s in msg) == viz["upload_message"]
    assert {"text": "Mostly complete, but bmi has gaps.", "kind": "commentary"} in msg
    assert {"text": "No other Residence type is present", "kind": "commentary"} in msg
    # The frame (headings, sizes) is code's; the score is the profiling
    # model's (78/78/85 on one file across rweb17 runs): commentary.
    plain = "".join(s["text"] for s in msg if s["kind"] == "plain")
    assert "Data readiness" not in plain and "**Watch out for**" in plain
    assert {"text": "Data readiness: 72/100.", "kind": "commentary"} in msg
    assert "rows × 3 columns" in plain
    assert SUFFIX not in plain
    _check_contract(viz)


def test_data_written_text_stays_plain(monkeypatch):
    """No model in the loop: nothing is commentary, nothing is suffixed."""
    viz = _build(monkeypatch, _stroke_rows(), [], use_llm=False)
    assert viz["charts"]
    for ch in viz["charts"]:
        assert SUFFIX not in str(ch.get("insight")) and SUFFIX not in str(ch.get("title"))
        assert ch["insight_segments"] == [{"text": ch["insight"], "kind": "plain"}]
        assert ch["title_segments"] == [{"text": ch["title"], "kind": "plain"}]
    assert _kinds(viz["upload_message_segments"]) == ["plain"]
    for line in viz["dataset_summary"].get("insights") or []:
        assert SUFFIX not in line
    _check_contract(viz)


def test_marking_twice_does_not_stack_the_suffix(monkeypatch):
    make_rows, suggestion = RC15_CASES["handful_of_urls_bulk"]
    viz = _build(monkeypatch, make_rows(), [suggestion], profiling_result=_profiling_result())
    card = _llm_card(viz, URL_SENTENCE)
    before = (copy.deepcopy(card), copy.deepcopy(viz["profile_overview"]))
    va.mark_chart_commentary(card)
    va.mark_profile_commentary(viz["profile_overview"])
    assert (card, viz["profile_overview"]) == before
    _check_contract(viz)
