"""Upload-card claims the rc15 live check found shown unmarked (rweb15
logs/shown15*.txt, truths in truthC/D/E.txt), and the shapes added for them
in card_quantifiers: concentration by "a handful", "most ... seen only once",
"steady increase", "no other X", and "associated with" between two numbers.

Synthetic frames reproduce the counts the claims are about. Every test goes
through the upload path (_ground_all / verify_profile_claims). No LLM.
"""

from __future__ import annotations

import pytest

from app.agents import visualization_agent as va


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _llm_chart(ctype, encodings, sentence, **config):
    return {"title": "LLM finding", "type": ctype, "encodings": encodings, "config": config,
            "insight": sentence, "reason": sentence, "insight_source": "llm", "llm_insight": sentence}


def _ground(chart, rows, scope="in the full file"):
    profiles = {p["name"]: p for p in va.profile_columns(rows)}
    kept = va._ground_all([chart], rows, profiles, scope=scope)
    assert kept, "the chart itself must still be drawn"
    return kept[0]


def _withheld(chart, phrase):
    shown = " ".join(str(chart.get(k) or "") for k in ("insight", "reason", "headline", "title"))
    return phrase not in shown and chart["headline"] is None


def _count_bar(field):
    return {"x": {"field": field}, "y": {"aggregate": "count"}}


# ---------------------------------------------------------------------------
# Recipes (12,190 rows): Url top 3 = 6.8%, Name top 5 = 8.6%,
# Instructions: 1,273 of 4,291 distinct texts occur once (29.7%; 10.4% of rows)
# ---------------------------------------------------------------------------

_RECIPE_FREQ = {1: 1273, 2: 1265, 3: 832, 4: 473, 5: 207, 6: 85, 7: 55, 8: 24, 9: 14, 10: 4,
                11: 5, 12: 6, 14: 5, 15: 3, 16: 1, 17: 1, 18: 3, 19: 3, 20: 2, 24: 4, 25: 2,
                27: 2, 29: 2, 30: 2, 32: 3, 36: 3, 37: 1, 39: 1, 40: 2, 41: 1, 42: 2, 45: 1,
                46: 1, 62: 1, 121: 1, 208: 1}


def _spread(top, n_rest, total):
    """Counts: the given top counts, then n_rest groups sharing what is left evenly."""
    left = total - sum(top)
    base, extra = divmod(left, n_rest)
    return list(top) + [base + 1] * extra + [base] * (n_rest - extra)


@pytest.fixture(scope="module")
def recipe_rows():
    instr = []
    k = 0
    for occurs, n_texts in _RECIPE_FREQ.items():
        for _ in range(n_texts):
            k += 1
            instr += [f"Schritt {k}: verruehren und backen."] * occurs
    urls = [f"https://www.chefkoch.de/rezepte/{i}/" for i, n in
            enumerate(_spread([374, 329, 120], 190, 12190)) for _ in range(n)]
    names = [f"Rezept {i}" for i, n in enumerate(_spread([374, 329, 127, 120, 86], 187, 12190)) for _ in range(n)]
    assert len(instr) == len(urls) == len(names) == 12190
    # Interleave so no column is sorted like another.
    return [{"Url": u, "Instructions": t, "Name": n} for u, t, n in zip(urls, instr[::-1], names[1:] + names[:1])]


URL_SENTENCE = ("A handful of recipe URLs account for the bulk of the records, suggesting that a few dishes "
                "are heavily over‑represented and may skew any analysis of recipe popularity.")
INSTR_ONCE_SENTENCE = ("A few instruction texts dominate the dataset while most are seen only once, "
                       "showing very low repetition of cooking steps.")
INSTR_OWN_SENTENCE = ("A few instruction texts appear many times while most recipes have their own unique "
                      "steps, indicating that a small set of popular methods dominates the collection.")
TITLES_SENTENCE = ("A handful of recipe titles dominate the collection, with a few names appearing far more "
                   "frequently than the rest, indicating a focus on popular dishes.")


def test_handful_of_urls_account_for_the_bulk_is_withheld(recipe_rows):
    chart = _ground(_llm_chart("bar", _count_bar("Url"), URL_SENTENCE, top_k=10), recipe_rows)
    assert _withheld(chart, "account for the bulk")
    assert "top 5" in chart["derived_data"]["llm_insight_replaced_because"]


def test_few_instructions_dominate_while_most_seen_once_is_withheld(recipe_rows):
    chart = _ground(_llm_chart("bar", _count_bar("Instructions"), INSTR_ONCE_SENTENCE, top_k=5), recipe_rows)
    assert _withheld(chart, "most are seen only once")


def test_most_recipes_have_their_own_unique_steps_is_withheld(recipe_rows):
    chart = _ground(_llm_chart("bar", _count_bar("Instructions"), INSTR_OWN_SENTENCE, top_k=5), recipe_rows)
    assert _withheld(chart, "their own unique steps")


def test_handful_of_recipe_titles_dominate_is_withheld(recipe_rows):
    chart = _ground(_llm_chart("bar", _count_bar("Name"), TITLES_SENTENCE, top_k=10), recipe_rows)
    assert _withheld(chart, "titles dominate")


def test_true_concentration_still_shows():
    rows = [{"Url": u} for u, n in (("a", 400), ("b", 250), ("c", 100)) for _ in range(n)]
    rows += [{"Url": f"other {i}"} for i in range(250)]
    chart = _ground(_llm_chart("bar", _count_bar("Url"), URL_SENTENCE, top_k=10), rows)
    assert chart["headline"] == URL_SENTENCE


def test_true_most_seen_once_still_shows():
    rows = [{"Instructions": f"text {i}"} for i in range(90)]
    rows += [{"Instructions": f"common {i}"} for i in range(3) for _ in range(20)]
    sentence = "A few instruction texts repeat while most are seen only once."
    chart = _ground(_llm_chart("bar", _count_bar("Instructions"), sentence, top_k=5), rows)
    assert chart["headline"] == sentence


# ---------------------------------------------------------------------------
# Messy e-commerce (2010 Q4, 20,000 lines)
# ---------------------------------------------------------------------------

_DAILY_QTY = {"2010-12-01": 26814, "2010-12-02": 21023, "2010-12-03": 14830, "2010-12-05": 16395,
              "2010-12-06": 21419, "2010-12-07": 24995, "2010-12-08": 22741, "2010-12-09": 7101}
TREND_SENTENCE = ("Daily sales volume shows a steady increase across the recorded days, suggesting growing "
                  "buying activity over the period.")


def _daily_rows(daily):
    rows = []
    for day, qty in daily.items():
        for h in range(10):                      # ten lines a day, summing to the day's total
            rows.append({"InvoiceDate": f"{day} {8 + h:02d}:15:00", "Quantity": qty // 10 + (qty % 10 if h == 0 else 0)})
    return rows


def test_steady_daily_increase_is_withheld():
    rows = _daily_rows(_DAILY_QTY)
    chart = _llm_chart("line", {"x": {"field": "InvoiceDate"}, "y": {"field": "Quantity", "aggregate": "sum"}},
                       TREND_SENTENCE)
    chart = _ground(chart, rows)
    assert _withheld(chart, "steady increase")
    assert "Spearman" in chart["derived_data"]["llm_insight_replaced_because"]


def test_true_steady_increase_still_shows():
    daily = {f"2010-{m:02d}-15": 1000 * m for m in range(1, 9)}
    sentence = "Monthly sales volume shows a steady increase over the period."
    chart = _llm_chart("line", {"x": {"field": "InvoiceDate"}, "y": {"field": "Quantity", "aggregate": "sum"}},
                       sentence)
    chart = _ground(chart, _daily_rows(daily))
    assert chart["headline"] == sentence


HANDFUL_SENTENCE = ("A small handful of products account for a disproportionate share of line items, "
                    "suggesting a focus on a few best‑selling items.")


def _stock_rows(top, n_rest, total):
    # Real StockCodes mix digits and letters ("85123A"), so the column is a category.
    return [{"StockCode": f"{84000 + i}{'A' if i % 7 == 3 else ''}", "Quantity": 3}
            for i, n in enumerate(_spread(top, n_rest, total)) for _ in range(n)]


def test_small_handful_of_products_disproportionate_share_is_withheld():
    rows = _stock_rows([108, 102, 95, 95, 88], 2353, 20000)       # top 5 = 2.4% of lines
    chart = _llm_chart("bar", {"x": {"field": "StockCode"}, "y": {"field": "Quantity", "aggregate": "sum"}},
                       HANDFUL_SENTENCE, top_k=15)
    chart = _ground(chart, rows)
    assert _withheld(chart, "disproportionate share")


def test_true_disproportionate_share_still_shows():
    rows = _stock_rows([900, 800, 700], 97, 4000)                 # top 3 = 60% of lines
    chart = _llm_chart("bar", {"x": {"field": "StockCode"}, "y": {"field": "Quantity", "aggregate": "sum"}},
                       HANDFUL_SENTENCE, top_k=15)
    chart = _ground(chart, rows)
    assert chart["headline"] == HANDFUL_SENTENCE


NO_OTHER_SENTENCE = "No other country appears in the data, so the UK is the sole leader."


def _country_rows():
    rows = [{"Country": "United Kingdom"}] * 19012
    others = ["Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal",
              "Australia", "Norway", "Italy", "Channel Islands", "Finland", "Cyprus", "Sweden"]
    for i, n in enumerate(_spread([256, 197], 13, 988)):
        rows += [{"Country": others[i]}] * n
    return rows


def test_no_other_country_appears_is_withheld():
    chart = _ground(_llm_chart("bar", _count_bar("Country"), NO_OTHER_SENTENCE, top_k=10), _country_rows())
    assert _withheld(chart, "No other country")
    assert "16 distinct values" in chart["derived_data"]["llm_insight_replaced_because"]


def test_no_other_work_type_red_flag_is_withheld_and_a_true_one_kept():
    rows = [{"work_type": w, "Residence_type": "Urban"} for w, n in
            (("Private", 2925), ("Self-employed", 819), ("children", 687), ("Govt_job", 657),
             ("Never_worked", 22)) for _ in range(n)]
    stats = va.build_column_stats(va.profile_columns(rows))
    ov = va.verify_profile_claims({"red_flags": ["No other work type is listed",
                                                 "No other Residence type is present"]}, stats)
    assert ov["red_flags"] == ["No other Residence type is present"]


ASSOC_SENTENCE = "Higher unit prices are associated with lower quantities, implying price sensitivity among buyers."


def _price_qty_rows(slope):
    rows = []
    for i in range(400):
        price = 1.0 + (i * 37 % 100) / 10.0
        noise = ((i * 7919) % 41) - 20
        rows.append({"UnitPrice": price, "Quantity": 20 + slope * price + noise})
    return rows


def test_price_associated_with_lower_quantity_at_r_zero_is_withheld():
    chart = _llm_chart("scatter", {"x": {"field": "UnitPrice"}, "y": {"field": "Quantity"}}, ASSOC_SENTENCE)
    chart = _ground(chart, _price_qty_rows(0.0))
    assert _withheld(chart, "associated with lower quantities")


def test_true_negative_association_still_shows():
    chart = _llm_chart("scatter", {"x": {"field": "UnitPrice"}, "y": {"field": "Quantity"}}, ASSOC_SENTENCE)
    chart = _ground(chart, _price_qty_rows(-3.0))
    assert chart["headline"] == ASSOC_SENTENCE
