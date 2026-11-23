"""Upload cards must not state quantifier claims the profile contradicts.

rc14 regrade (rweb14 rw7, truths in logs/truthB.txt): the insight model's
findings and the profiling model's red flags were shown on the upload card
unmarked although the statistics computed at upload contradicted them:

- taxi: "Tip amounts are frequently zero when ... credit card" (6.6% are)
- taxi: "Airport_fee always zero" (24,265 rows are not)
- taxi: "Extra and improvement_surcharge constant across rows" (26 / 4 values)
- recipes: "Instructions ... the vast majority are unique" (30% of 4,291 occur once)
- stroke: smokers "markedly higher stroke occurrence" than never-smokers (5.3% vs 4.8%)

The frames below are synthetic, but reproduce the exact counts of the real
files (taxi_rowgroups.parquet, recipes.jsonl, healthcare-dataset-stroke-data.csv)
for the columns each claim is about. Every test goes through the code path the
upload uses: _ground_all for chart findings, verify_profile_claims for the
profile's red flags, build_upload_message for the card text.

No LLM is called; the model's sentences are the ones it wrote, verbatim.
"""

from __future__ import annotations

import pytest

from app.agents import visualization_agent as va


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _profiles(rows):
    return {p["name"]: p for p in va.profile_columns(rows)}


def _llm_chart(ctype, encodings, sentence, **config):
    return {"title": "LLM finding", "type": ctype, "encodings": encodings, "config": config,
            "insight": sentence, "reason": sentence, "insight_source": "llm",
            "llm_insight": sentence}


def _shown(chart):
    """Every text of a chart the card can show."""
    return " ".join(str(chart.get(k) or "") for k in ("insight", "reason", "headline", "title"))


def _ground(chart, rows, profile_rows=None, scope="in the full file"):
    kept = va._ground_all([chart], rows, _profiles(profile_rows or rows), scope=scope)
    assert kept, "the chart itself must still be drawn"
    return kept[0]


# ---------------------------------------------------------------------------
# Taxi (200,000 rows)
# ---------------------------------------------------------------------------

# payment_type -> (rows, rows with tip_amount == 0), from taxi_rowgroups.parquet
_TAXI_TIPS = {1: (152005, 10018), 2: (42345, 42337), 3: (1715, 1685), 4: (3935, 3897)}
_TAXI_AIRPORT = {0.0: 175735, 1.75: 23843, -1.75: 422}
_TAXI_EXTRA = {0.0: 90976, 2.5: 43563, 1.0: 33899, 5.0: 13938, 3.5: 8570, 6.0: 1773, 7.5: 1696,
               4.25: 1088, 9.25: 961, -1.0: 834, 1.75: 729, 6.75: 450, -2.5: 403, 2.75: 344,
               11.75: 201, 10.25: 192, 7.75: 163, -5.0: 106, 10.0: 29, -6.0: 24, -7.5: 23,
               8.5: 19, 3.2: 12, 5.25: 4, 3.25: 2, 0.75: 1}
_TAXI_IMPROVEMENT = {1.0: 196898, -1.0: 3021, 0.0: 63, 0.3: 18}


def _expand(counts):
    return [v for v, n in counts.items() for _ in range(n)]


@pytest.fixture(scope="module")
def taxi_rows():
    pay, tip = [], []
    for p, (n, zero) in _TAXI_TIPS.items():
        pay += [float(p)] * n
        tip += [0.0] * zero + [2.5] * (n - zero)
    airport, extra, impr = _expand(_TAXI_AIRPORT), _expand(_TAXI_EXTRA), _expand(_TAXI_IMPROVEMENT)
    assert len(pay) == len(airport) == len(extra) == len(impr) == 200000
    return [{"payment_type": a, "tip_amount": b, "Airport_fee": c, "extra": d,
             "improvement_surcharge": e} for a, b, c, d, e in zip(pay, tip, airport, extra, impr)]


@pytest.fixture(scope="module")
def taxi_stats(taxi_rows):
    return va.build_column_stats(va.profile_columns(taxi_rows))


TIP_SENTENCE = ("Tip amounts are frequently zero when the payment method is credit card, "
                "suggesting a possible recording error that could understate driver earnings.")


def test_taxi_frequently_zero_tips_for_card_payments_is_withheld(taxi_rows):
    chart = _llm_chart("bar", {"x": {"field": "payment_type"},
                               "y": {"field": "tip_amount", "aggregate": "mean"}}, TIP_SENTENCE)
    chart = _ground(chart, taxi_rows, taxi_rows[::25])
    assert "frequently zero" not in _shown(chart)
    assert chart["headline"] is None
    assert chart["insight"] == chart["insight_facts"]          # the data's sentence instead
    msg = va.build_upload_message(None, [chart], 200000, 19)
    assert "frequently zero" not in msg
    assert "Average tip_amount by payment_type" in msg


def test_taxi_frequently_zero_tips_for_payment_type_1_is_withheld(taxi_rows):
    """Resolvable condition, still false: 10,018 of 152,005 card rows (6.6%)."""
    sentence = "Tip amounts are frequently zero when payment_type is 1."
    chart = _llm_chart("bar", {"x": {"field": "payment_type"},
                               "y": {"field": "tip_amount", "aggregate": "mean"}}, sentence)
    chart = _ground(chart, taxi_rows, taxi_rows[::25])
    assert "frequently zero" not in _shown(chart)
    assert "6.6%" in chart["derived_data"]["llm_insight_replaced_because"]


def test_taxi_true_zero_tip_claim_for_cash_still_shows(taxi_rows):
    """42,337 of 42,345 cash rows (payment_type 2) have no tip: a true claim."""
    sentence = "Tip amounts are almost always zero when payment_type is 2."
    chart = _llm_chart("bar", {"x": {"field": "payment_type"},
                               "y": {"field": "tip_amount", "aggregate": "mean"}}, sentence)
    chart = _ground(chart, taxi_rows, taxi_rows[::25])
    assert chart["headline"] == sentence
    assert chart["insight"].startswith("Tip amounts are almost always zero")


def test_taxi_airport_fee_always_zero_red_flag_is_withheld(taxi_stats):
    ov = va.verify_profile_claims({"red_flags": [
        "Zero tip with payment_type=1 (credit card) may be data error",
        "Airport_fee always zero despite possible airport trips",
        "Extra and improvement_surcharge constant across rows",
    ], "concerns": [
        "Constant extra and surcharge values need verification",
        "Airport_fee always zero may miss airport trips",
    ]}, taxi_stats)
    shown = " | ".join(ov["red_flags"] + ov["concerns"])
    assert "always zero" not in shown
    # Hedged and quantifier-free: kept as written.
    assert ov["red_flags"] == ["Zero tip with payment_type=1 (credit card) may be data error"]
    withheld = {c["original"] for c in ov["corrections"] if c["corrected"] == ""}
    assert "Airport_fee always zero despite possible airport trips" in withheld

    msg = va.build_upload_message(ov, [], 200000, 19)
    assert "Airport_fee always zero" not in msg


def test_taxi_extra_and_improvement_surcharge_constant_is_withheld(taxi_stats):
    ov = va.verify_profile_claims(
        {"red_flags": ["Extra and improvement_surcharge constant across rows"],
         "concerns": ["Constant extra and surcharge values need verification"]}, taxi_stats)
    assert ov["red_flags"] == [] and ov["concerns"] == []
    reasons = " ".join(c.get("reason", "") for c in ov["corrections"])
    assert "26 distinct values" in reasons
    msg = va.build_upload_message(ov, [], 200000, 19)
    assert "constant" not in msg.lower()


def test_taxi_airport_fee_almost_always_zero_chart_finding_is_withheld(taxi_rows):
    """87.9% zero is not 'almost always' (>= 95%)."""
    sentence = ("Airport fees are almost always zero, even though some trips likely involve "
                "the airport, indicating missing surcharge data.")
    chart = _llm_chart("pie", {"theta": {"field": "count", "aggregate": "sum"},
                               "color": {"field": "Airport_fee"}}, sentence, top_k=10)
    chart = _ground(chart, taxi_rows, taxi_rows[::25])
    assert "almost always zero" not in _shown(chart)


def test_known_issues_sent_to_the_insight_model_drop_the_false_ones(taxi_stats):
    ctx = va.verify_known_issues({"known_issues": [
        "Airport_fee always zero despite possible airport trips",
        "Extra and improvement_surcharge constant across rows",
        "Missing timezone in datetime fields",
    ]}, taxi_stats)
    assert ctx["known_issues"] == ["Missing timezone in datetime fields"]


# ---------------------------------------------------------------------------
# Recipes (12,190 rows, 4,291 distinct Instructions)
# ---------------------------------------------------------------------------

# occurrences -> number of distinct Instructions texts with that many rows
_RECIPE_FREQ = {1: 1273, 2: 1265, 3: 832, 4: 473, 5: 207, 6: 85, 7: 55, 8: 24, 9: 14, 10: 4,
                11: 5, 12: 6, 14: 5, 15: 3, 16: 1, 17: 1, 18: 3, 19: 3, 20: 2, 24: 4, 25: 2,
                27: 2, 29: 2, 30: 2, 32: 3, 36: 3, 37: 1, 39: 1, 40: 2, 41: 1, 42: 2, 45: 1,
                46: 1, 62: 1, 121: 1, 208: 1}

INSTRUCTIONS_SENTENCE = (
    "The Instructions column shows an extreme long‑tail pattern: only a few instruction "
    "texts appear repeatedly while the vast majority are unique, making categorical grouping "
    "difficult.")


@pytest.fixture(scope="module")
def recipe_rows():
    rows, k = [], 0
    for occurs, n_texts in _RECIPE_FREQ.items():
        for _ in range(n_texts):
            k += 1
            rows += [{"Instructions": f"Schritt {k}: alles verruehren und backen.", "Day": 1}] * occurs
    assert (len(rows), k) == (12190, 4291)
    return rows


def test_recipes_vast_majority_unique_instructions_is_withheld(recipe_rows):
    chart = _llm_chart("bar", {"x": {"field": "Instructions"}, "y": {"aggregate": "count"}},
                       INSTRUCTIONS_SENTENCE, top_k=5)
    chart = _ground(chart, recipe_rows, scope="in the sample")
    assert "vast majority are unique" not in _shown(chart)
    assert "29.7%" in chart["derived_data"]["llm_insight_replaced_because"]


def test_recipes_true_uniqueness_claim_still_shows():
    """95 of 100 distinct texts occur once: 'the vast majority are unique' holds."""
    rows = [{"Instructions": f"Text {i}"} for i in range(95)]
    rows += [{"Instructions": f"Repeated {i}"} for i in range(5) for _ in range(3)]
    chart = _llm_chart("bar", {"x": {"field": "Instructions"}, "y": {"aggregate": "count"}},
                       INSTRUCTIONS_SENTENCE, top_k=5)
    chart = _ground(chart, rows, scope="in the sample")
    assert chart["headline"] == INSTRUCTIONS_SENTENCE


def test_recipes_constant_day_column_red_flag_still_shows(recipe_rows):
    stats = va.build_column_stats(va.profile_columns(recipe_rows))
    ov = va.verify_profile_claims({"red_flags": ["Constant Day column"],
                                   "concerns": ["Day column constant"],
                                   "readiness_reasoning": "Data is complete but needs "
                                   "preprocessing of constant column, month format."}, stats)
    assert ov["red_flags"] == ["Constant Day column"]
    assert ov["concerns"] == ["Day column constant"]
    assert ov["readiness_reasoning"].startswith("Data is complete")


# ---------------------------------------------------------------------------
# Stroke (5,110 rows)
# ---------------------------------------------------------------------------

# group -> (strokes, rows), from healthcare-dataset-stroke-data.csv
_SMOKING = {"Unknown": (47, 1544), "formerly smoked": (70, 885),
            "never smoked": (90, 1892), "smokes": (42, 789)}
_HYPERTENSION = {1: (66, 498), 0: (183, 4612)}

SMOKE_SENTENCE = ("Individuals who currently smoke have a markedly higher stroke occurrence "
                  "compared with never‑smokers, highlighting smoking as a major modifiable risk.")
HYPERTENSION_SENTENCE = (
    "Patients with hypertension experience a noticeably higher proportion of strokes than those "
    "without hypertension, indicating a key risk factor to monitor.")


@pytest.fixture(scope="module")
def stroke_rows():
    stroke_smoking, healthy_smoking = [], []
    for g, (k, n) in _SMOKING.items():
        stroke_smoking += [g] * k
        healthy_smoking += [g] * (n - k)
    stroke_hyp = [1] * 66 + [0] * 183
    healthy_hyp = [1] * (498 - 66) + [0] * (4612 - 183)
    rows = [{"smoking_status": s, "hypertension": h, "stroke": 1}
            for s, h in zip(stroke_smoking, stroke_hyp)]
    rows += [{"smoking_status": s, "hypertension": h, "stroke": 0}
             for s, h in zip(healthy_smoking, healthy_hyp)]
    assert len(rows) == 5110
    return rows


def _rate_chart(x, sentence):
    return _llm_chart("bar", {"x": {"field": x}, "y": {"field": "stroke", "aggregate": "mean"}},
                      sentence, top_k=15, rate_by_group=True, y_format="percent")


def test_stroke_smokers_markedly_higher_than_never_smokers_is_withheld(stroke_rows):
    chart = _ground(_rate_chart("smoking_status", SMOKE_SENTENCE), stroke_rows, scope="in the sample")
    assert "markedly higher" not in _shown(chart)
    assert chart["headline"] is None
    msg = va.build_upload_message(None, [chart], 5110, 12)
    assert "markedly higher" not in msg
    assert "stroke rate by smoking_status" in msg


def test_stroke_true_hypertension_comparison_still_shows(stroke_rows):
    """13.3% vs 4.0%: 3.3x and 9.3 points apart."""
    chart = _ground(_rate_chart("hypertension", HYPERTENSION_SENTENCE), stroke_rows,
                    scope="in the sample")
    assert chart["headline"] == HYPERTENSION_SENTENCE


def test_stroke_true_former_smoker_comparison_still_shows(stroke_rows):
    """7.9% vs 4.8%: 1.66x and 3.1 points apart."""
    sentence = ("Former smokers have a markedly higher stroke occurrence compared with "
                "never\u2011smokers.")
    chart = _ground(_rate_chart("smoking_status", sentence), stroke_rows, scope="in the sample")
    assert chart["headline"] == sentence


def test_full_file_statistics_carry_the_zero_share_the_check_needs(tmp_path):
    """Min -1.75 / max 1.75 cannot decide 'almost always zero'; the full-file
    statistics now carry zero_pct (87.9%), so the red flag is withheld there
    too, and a true one ('frequently zero') is kept."""
    path = tmp_path / "taxi.csv"
    lines = ["Airport_fee,Instructions"]
    lines += [f"0,text {i % 700}" for i in range(879)] + [f"1.75,text {i % 700}" for i in range(117)]
    lines += [f"-1.75,text {i % 700}" for i in range(4)]
    path.write_text("\n".join(lines) + "\n")
    rows = [dict(zip(("Airport_fee", "Instructions"), ln.split(","))) for ln in lines[1:]]
    stats = va.build_column_stats_full(str(path), "csv", va.profile_columns(rows))
    assert stats is not None
    ov = va.verify_profile_claims({"red_flags": ["Airport_fee is almost always zero",
                                                 "Airport_fee is frequently zero"]}, stats)
    assert ov["red_flags"] == ["Airport_fee is frequently zero"]
