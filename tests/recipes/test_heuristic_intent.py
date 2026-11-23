"""The deterministic intent pass against the golden questions.

Every item under ``golden_questions/<recipe>.json`` is a question, the
columns it was asked over (name → role) and the Intent a careful reader
would produce. Recipe choice must be right on at least 90 % overall; a flag
the golden expects must be set on every item (a missed ``causal_question``
would let a report read as an effect, so there is no tolerance on flags).
Parameter accuracy is reported, not gated: the binder asks for what the
heuristics leave unbound.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avaloka.outcomes.semantics import ColumnSemantics
from avaloka.recipes import INTENT_FLAGS, RECIPE_IDS
from avaloka.recipes.intent import (_CAUSAL_TERMS, _TIE_ORDER, causal_term, explicit_grain,
                                    heuristic_intent, mentioned_columns)

GOLDEN = Path(__file__).parent / "golden_questions"
MIN_RECIPE_ACCURACY = 0.90
MIN_PER_RECIPE = 10


def load_goldens() -> dict[str, list[dict]]:
    return {rid: json.loads((GOLDEN / f"{rid}.json").read_text()) for rid in RECIPE_IDS}


def semantics_for(columns: dict[str, str]) -> dict[str, ColumnSemantics]:
    return {name: ColumnSemantics(name=name, role=role) for name, role in columns.items()}


def _listed(params) -> dict:
    return {k: (sorted(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()}


def test_golden_set_is_well_formed():
    goldens = load_goldens()
    total = 0
    for rid, items in goldens.items():
        assert len(items) >= MIN_PER_RECIPE, f"{rid}: {len(items)} goldens, need {MIN_PER_RECIPE}"
        for it in items:
            assert set(it) == {"question", "columns", "expected"}
            assert it["expected"]["recipe"] in RECIPE_IDS
            assert set(it["expected"]["flags"]) <= set(INTENT_FLAGS)
            for v in it["expected"]["params"].values():
                for col in (v if isinstance(v, list) else [v]):
                    assert col in it["columns"] or col in ("day", "week", "month", "quarter", "year") \
                        or col.startswith(("last_", "ytd", "all"))
        total += len(items)
    assert total >= 100
    causal = sum("causal_question" in it["expected"]["flags"] for items in goldens.values() for it in items)
    ambiguous = sum("ambiguous" in it["expected"]["flags"] for items in goldens.values() for it in items)
    assert causal >= 8 and ambiguous >= 5


def test_heuristic_accuracy_and_flags(capsys):
    goldens = load_goldens()
    rows, missed_flags = [], []
    total_hit = total_n = total_params = 0
    for rid, items in goldens.items():
        hit = params_ok = 0
        for it in items:
            intent = heuristic_intent(it["question"], semantics_for(it["columns"]), {})
            exp = it["expected"]
            hit += intent.recipe == exp["recipe"]
            params_ok += _listed(intent.params) == _listed(exp["params"])
            for flag in exp["flags"]:
                if flag not in intent.flags:
                    missed_flags.append((it["question"], flag, intent.flags))
        rows.append((rid, hit, len(items), params_ok))
        total_hit, total_n, total_params = total_hit + hit, total_n + len(items), total_params + params_ok
    with capsys.disabled():
        print("\n  recipe             recipe-acc   params-acc")
        for rid, hit, n, p in rows:
            print(f"  {rid:18} {hit:3}/{n:<3} {hit / n:5.2f}  {p:3}/{n:<3} {p / n:5.2f}")
        print(f"  {'overall':18} {total_hit:3}/{total_n:<3} {total_hit / total_n:5.2f}  "
              f"{total_params:3}/{total_n:<3} {total_params / total_n:5.2f}")
    assert not missed_flags, f"expected flags not set: {missed_flags}"
    assert total_hit / total_n >= MIN_RECIPE_ACCURACY


def test_causal_lexicon_matches_the_verifiers():
    """The copied words must be the verifier's own; app/ is read as text, not imported."""
    src = Path(__file__).resolve().parents[2] / "app" / "agents" / "claim_verifier.py"
    if not src.exists():
        pytest.skip("app/ is not part of this tree")
    text = src.read_text()
    for term in _CAUSAL_TERMS:
        assert f'"{term}"' in text, f"{term!r} is not in claim_verifier's lexicon"
    assert causal_term("root cause analysis of errors") is None
    assert causal_term("does price increase the likelihood of churn") == "increase the likelihood"


def test_tie_order_covers_every_recipe_once():
    assert sorted(_TIE_ORDER) == sorted(RECIPE_IDS)


def test_no_cue_means_no_recipe_and_zero_confidence():
    sem = semantics_for({"revenue": "measure", "order_date": "time"})
    i = heuristic_intent("revenue please", sem, {})
    assert i.recipe is None and i.confidence == 0.0 and i.params == {} and i.source == "heuristic"


def test_confidence_scheme():
    sem = semantics_for({"revenue": "measure", "cost": "measure", "order_date": "time", "region": "dimension"})
    assert heuristic_intent("How has revenue trended monthly?", sem, {}).confidence == 1.0
    two_families = heuristic_intent("Compare revenue this quarter vs last quarter by region", sem, {})
    assert two_families.confidence == pytest.approx(0.8)
    tie = heuristic_intent("Give me the trend and the breakdown of revenue", sem, {})
    assert "ambiguous" in tie.flags and tie.confidence == pytest.approx(0.5)
    unbound = heuristic_intent("Show me the trend", sem, {})
    assert "measure" not in unbound.params and unbound.confidence == pytest.approx(0.7)


def test_mentions_prefer_the_longer_name_and_measure_synonyms_only_for_measures():
    sem = semantics_for({"order_date": "time", "order_id": "id", "sales": "dimension", "revenue": "measure"})
    assert mentioned_columns("revenue by order date", sem)[0] == ("order_date", 2)
    # "sales" is a synonym of the measure revenue, but here sales is a dimension column named verbatim
    hits = dict(mentioned_columns("sales by region", sem))
    assert "sales" in hits and "revenue" in hits


def test_explicit_grain_reads_a_step_not_a_window():
    assert explicit_grain("revenue by month") == "month"
    assert explicit_grain("weekly errors") == "week"
    assert explicit_grain("revenue over the last month") is None
    assert explicit_grain("daily and monthly") is None


def test_multi_question_records_the_first_clause():
    sem = semantics_for({"revenue": "measure", "cost": "measure", "units": "measure",
                         "order_date": "time", "region": "dimension"})
    i = heuristic_intent("Revenue by region; then the correlation of cost with units", sem, {})
    assert i.recipe == "segment" and "multi_question" in i.flags
    assert i.params == {"measure": "revenue", "dimension": "region"}
