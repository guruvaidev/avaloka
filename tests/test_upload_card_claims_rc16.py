"""Upload-card claims the rc16 regrade found shown unmarked (rweb16
logs/shown16b.txt and shown16c.txt, truths in truthC-G.txt), and the rules
added to card_quantifiers for them (#585):

* recipes, "most entries appearing only once": the rule existed, but it was
  checked on the 4,018-row sample, where 65.1% of the distinct Instructions
  occur once; in the 12,190-row file 29.7% do (10.4% of rows);
* recipes, "a large portion of the collection" for a recipe holding 3.1%;
* taxi, "consistently generates higher total fares": higher on 3 of 6 days;
* taxi, "credit cards ... little or no tip, while cash payments show higher
  average tips": reversed (card 4.62, cash 0.0012), and payment_type holds
  codes, so the comparison is withheld as unmappable;
* taxi, "the standard extra fee of one dollar": 16.9% are $1, 45.5% are $0;
* stroke, "Mixed case in Residence_type": it holds Urban and Rural only.

The sentences are verbatim. Synthetic rows reproduce the shares the claims
are about. The cards go through the upload path,
build_visualization_config_from_sample, with the insight model's suggestions
stubbed; the watch-out goes through verify_profile_claims. No LLM.
"""

from __future__ import annotations

import csv

import pytest

from app.agents import visualization_agent as va

INSTRUCTIONS_SENTENCE = ("The Instructions field is extremely varied, with most entries appearing only once, "
                         "which makes grouping recipes by instruction difficult.")
PORTION_SENTENCE = ("Herzhafte Pfannkuchen stands out as the most common recipe, accounting for a large "
                    "portion of the collection.")
VENDOR_SENTENCE = ("One taxi vendor consistently generates higher total fares than the other, suggesting a "
                   "revenue advantage that could be explored for best practices.")
TIP_SENTENCE = ("Trips paid with credit cards tend to have little or no tip, while cash payments show higher "
                "average tips, suggesting payment method influences driver earnings.")
EXTRA_SENTENCE = ("Most trips carry the standard extra fee of one dollar, while a small minority show a higher "
                  "extra fee of three and a half dollars, which may reflect special pricing or a data entry issue.")
MIXED_CASE_FLAG = "Mixed case in Residence_type"


def _upload(monkeypatch, rows, suggestions, **kw):
    """The upload path with the insight model's suggestions stubbed."""
    monkeypatch.setattr(va, "viz_llm", object())
    monkeypatch.setattr(va, "_suggest_for_plan", lambda plans, user_context, effort: (suggestions, None))
    return va.build_visualization_config_from_sample("t", rows, use_llm=True, **kw)


def _shown(cfg, sentence):
    """Whether *sentence* reaches the user: on a card or in the upload message."""
    cards = " ".join(str(c.get(k) or "") for c in cfg["charts"] for k in ("insight", "reason", "headline"))
    return sentence in cards or sentence in str(cfg.get("upload_message") or "")


def _why(cfg, sentence):
    for c in cfg["charts"]:
        dd = c.get("derived_data") or {}
        if dd.get("llm_insight_replaced") == sentence:
            return dd.get("llm_insight_replaced_because")
    return None


def _bar(x, sentence, y=None, title="LLM finding"):
    s = {"type": "bar", "x_field": x, "title": title, "insight": sentence}
    if y:
        s["y_field"] = y
    return s


# ---------------------------------------------------------------------------
# Recipes: Instructions in a sample, Name shares
# ---------------------------------------------------------------------------

def _recipe_rows(n_once=160, n_twice=60, n_four=30, top_name_share=0.033):
    """400 sampled recipes: 160 of 250 distinct Instructions occur once (64%)."""
    texts = [f"Schritt {k}: alles verruehren und im Ofen backen." for k in range(n_once + n_twice + n_four)]
    instr = texts[:n_once] + [t for t in texts[n_once:n_once + n_twice] for _ in range(2)] + \
        [t for t in texts[n_once + n_twice:] for _ in range(4)]
    n = len(instr)
    top = round(top_name_share * n)
    others = [f"Rezept {i % 40} mit Gemuese" for i in range(n - top)]
    names = ["Herzhafte Pfannkuchen"] * top + others
    return [{"Instructions": t, "Name": nm, "Year": 2005 + i % 15} for i, (t, nm) in enumerate(zip(instr, names[::-1]))]


def test_most_entries_once_on_a_sample_is_withheld(monkeypatch):
    rows = _recipe_rows()
    cfg = _upload(monkeypatch, rows, [_bar("Instructions", INSTRUCTIONS_SENTENCE)], total_rows=12190)
    assert not _shown(cfg, INSTRUCTIONS_SENTENCE), cfg["upload_message"]
    assert "only a sample of 400 of the file's 12,190 rows" in (_why(cfg, INSTRUCTIONS_SENTENCE) or "")


def test_most_entries_once_on_the_whole_file_still_shows(monkeypatch):
    """True claim: 64% of the distinct texts occur once, and the rows are every row."""
    rows = _recipe_rows()
    cfg = _upload(monkeypatch, rows, [_bar("Instructions", INSTRUCTIONS_SENTENCE)], total_rows=len(rows))
    assert _shown(cfg, INSTRUCTIONS_SENTENCE)


def test_most_entries_once_is_rechecked_on_the_full_file(monkeypatch, tmp_path):
    """Withheld on the sample, then checked on every row once the file is re-read."""
    rows = _recipe_rows()
    path = tmp_path / "recipes.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = _upload(monkeypatch, rows[:300], [_bar("Instructions", INSTRUCTIONS_SENTENCE)],
                  total_rows=len(rows), full_data_path=str(path), full_data_type="csv")
    assert cfg["dataset"]["charts_scope"] == "full_file"
    assert _shown(cfg, INSTRUCTIONS_SENTENCE)


def test_large_portion_for_a_three_percent_recipe_is_withheld(monkeypatch):
    cfg = _upload(monkeypatch, _recipe_rows(), [_bar("Name", PORTION_SENTENCE)], total_rows=12190)
    assert not _shown(cfg, PORTION_SENTENCE)
    assert "(needs >= 20.0%)" in (_why(cfg, PORTION_SENTENCE) or "")


def test_large_portion_for_a_quarter_still_shows(monkeypatch):
    cfg = _upload(monkeypatch, _recipe_rows(top_name_share=0.25), [_bar("Name", PORTION_SENTENCE)],
                  total_rows=12190)
    assert _shown(cfg, PORTION_SENTENCE)


# ---------------------------------------------------------------------------
# Taxi: vendors by day, tips by payment code, the extra fee
# ---------------------------------------------------------------------------

def _taxi_rows(stray_days=True, vendor1_wins_days=()):
    """Vendor 2's mean total fare is higher on 2024-01-01..03; with stray_days,
    three more days carry only a handful of vendor-2 trips (as the file's
    2002-12-31, 2009-01-01 and 2023-12-31 do)."""
    rows = []
    for d, day in enumerate(["2024-01-01", "2024-01-02", "2024-01-03"]):
        hi, lo = (28.0, 31.0) if day in vendor1_wins_days else (31.0, 28.0)
        for i in range(60):
            rows.append({"VendorID": 2, "tpep_pickup_datetime": f"{day} 0{i % 10}:15:00", "total_amount": hi + i % 3})
            if i < 30:
                rows.append({"VendorID": 1, "tpep_pickup_datetime": f"{day} 1{i % 10}:20:00",
                             "total_amount": lo + i % 3})
    if stray_days:
        rows += [{"VendorID": 2, "tpep_pickup_datetime": "2002-12-31 23:50:00", "total_amount": 0.0}] * 2
        rows += [{"VendorID": 2, "tpep_pickup_datetime": "2009-01-01 00:10:00", "total_amount": 9.4}]
        rows += [{"VendorID": 2, "tpep_pickup_datetime": "2023-12-31 23:40:00", "total_amount": 22.5}] * 10
    return rows


def test_consistently_higher_on_three_of_six_days_is_withheld(monkeypatch):
    cfg = _upload(monkeypatch, _taxi_rows(), [_bar("VendorID", VENDOR_SENTENCE, y="total_amount")])
    assert not _shown(cfg, VENDOR_SENTENCE)
    assert "holds in 3 of 6 periods" in (_why(cfg, VENDOR_SENTENCE) or "")


def test_consistently_higher_on_every_day_still_shows(monkeypatch):
    cfg = _upload(monkeypatch, _taxi_rows(stray_days=False), [_bar("VendorID", VENDOR_SENTENCE, y="total_amount")])
    assert _shown(cfg, VENDOR_SENTENCE)


def _tip_rows():
    """payment_type 1 (card) tips 4.62 on average, 2 (cash) almost never."""
    rows = [{"payment_type": 1, "tip_amount": [3.62, 4.62, 5.62][i % 3]} for i in range(300)]
    rows += [{"payment_type": 2, "tip_amount": 0.5 if i == 0 else 0.0} for i in range(120)]
    rows += [{"payment_type": 3, "tip_amount": 0.0} for _ in range(10)]
    rows += [{"payment_type": 4, "tip_amount": 0.1 if i % 2 else 0.0} for i in range(20)]
    return rows


def test_card_versus_cash_tips_on_payment_codes_is_withheld(monkeypatch):
    cfg = _upload(monkeypatch, _tip_rows(), [_bar("payment_type", TIP_SENTENCE, y="tip_amount")])
    assert not _shown(cfg, TIP_SENTENCE)
    assert "names none of its values" in (_why(cfg, TIP_SENTENCE) or "")


def test_reversed_comparison_of_named_codes_is_withheld(monkeypatch):
    sentence = "Payment type 2 shows higher average tips than payment type 1."
    cfg = _upload(monkeypatch, _tip_rows(), [_bar("payment_type", sentence, y="tip_amount")])
    assert not _shown(cfg, sentence)


def test_true_comparison_of_named_codes_still_shows(monkeypatch):
    sentence = "Payment type 1 shows higher average tips than payment type 2."
    cfg = _upload(monkeypatch, _tip_rows(), [_bar("payment_type", sentence, y="tip_amount")])
    assert _shown(cfg, sentence)


def _extra_rows(counts):
    rows = []
    for value, n in counts.items():
        rows += [{"extra": value, "fare_amount": 10.0 + i % 7} for i in range(n)]
    return rows


#: The file's shares: 0 45.5%, 2.5 21.8%, 1 16.9%, 5 7.0%, 3.5 and others the rest.
EXTRA_COUNTS = {0.0: 455, 2.5: 218, 1.0: 169, 5.0: 70, 3.5: 50, 6.0: 38}


def test_standard_extra_fee_that_is_not_the_mode_is_withheld(monkeypatch):
    cfg = _upload(monkeypatch, _extra_rows(EXTRA_COUNTS), [_bar("extra", EXTRA_SENTENCE)])
    assert not _shown(cfg, EXTRA_SENTENCE)
    assert "the most common extra is 0 (45.5% of rows); 1 is 16.9%" in (_why(cfg, EXTRA_SENTENCE) or "")


def test_standard_extra_fee_that_is_the_mode_still_shows(monkeypatch):
    counts = {1.0: 455, 2.5: 218, 0.0: 169, 5.0: 70, 3.5: 50, 6.0: 38}
    cfg = _upload(monkeypatch, _extra_rows(counts), [_bar("extra", EXTRA_SENTENCE)])
    assert _shown(cfg, EXTRA_SENTENCE)


# ---------------------------------------------------------------------------
# Stroke: a case-variance watch-out
# ---------------------------------------------------------------------------

def _stats(rows):
    return va.build_column_stats(va.profile_columns(rows))


def test_mixed_case_in_a_column_without_case_variants_is_withheld():
    rows = [{"Residence_type": "Urban" if i % 2 else "Rural", "age": 40 + i % 30} for i in range(200)]
    ov = va.verify_profile_claims({"red_flags": [MIXED_CASE_FLAG, "Potential stroke imbalance"]}, _stats(rows))
    assert ov["red_flags"] == ["Potential stroke imbalance"]
    assert "none of its 2 values (Urban, Rural) differ only in case" in ov["corrections"][0]["reason"] or \
        "none of its 2 values (Rural, Urban) differ only in case" in ov["corrections"][0]["reason"]


def test_inconsistent_casing_with_case_variants_still_shows():
    rows = [{"Outcome": ["Recovered", "RECOVERED", "Readmitted"][i % 3], "Glucose": 100 + i % 40}
            for i in range(60)]
    ov = va.verify_profile_claims({"red_flags": ["Inconsistent outcome casing"]}, _stats(rows))
    assert ov["red_flags"] == ["Inconsistent outcome casing"]
