"""``recognize`` is a recognizer: every out-of-vocabulary value is a named rejection."""

from __future__ import annotations

import pytest

from avaloka.recipes import Intent
from avaloka.recipes.intent import REJECTIONS, IntentRejected, recognize

COLUMNS = {"order_date": "time", "region": "dimension", "revenue": "measure", "cost": "measure",
           "customer_id": "id"}
Q = "revenue trend by month"


def _payload(**over):
    base = {"recipe": "trend", "params": {"measure": "revenue", "time": "order_date", "grain": "month"},
            "flags": [], "confidence": 0.9}
    base.update(over)
    return base


def test_well_formed_payload_round_trips():
    i = recognize(_payload(flags=["causal_question", "causal_question"]), question=Q, columns=COLUMNS, source="fixture")
    assert isinstance(i, Intent)
    assert i.recipe == "trend" and i.params == {"measure": "revenue", "time": "order_date", "grain": "month"}
    assert i.flags == ("causal_question",) and i.confidence == 0.9 and i.source == "fixture"
    assert i.question == Q


def test_many_parameter_takes_a_tuple_of_columns():
    i = recognize({"recipe": "correlate", "params": {"measures": ["revenue", "cost"]}, "flags": [], "confidence": 1},
                  question=Q, columns=COLUMNS, source="model")
    assert i.params["measures"] == ("revenue", "cost")


@pytest.mark.parametrize("payload, reason", [
    (["trend"], "not_an_object"),
    ({"recipe": "trend", "params": {}}, "missing_key"),
    (_payload(extra="x"), "unknown_key"),
    (_payload(recipe="forecast"), "unknown_recipe"),
    (_payload(params={"metric": "revenue"}), "unknown_parameter"),
    (_payload(params={"arm": "region"}), "parameter_not_in_recipe"),
    (_payload(params={"measure": "profit"}), "unknown_column"),
    (_payload(params={"measure": "region"}), "role_mismatch"),
    (_payload(params={"time": "revenue"}), "role_mismatch"),
    (_payload(params={"grain": "fortnight"}), "unknown_grain"),
    ({"recipe": "compare_periods", "params": {"window": "last_decade"}, "flags": [], "confidence": 1}, "unknown_window"),
    (_payload(flags=["hunch"]), "unknown_flag"),
    (_payload(flags="causal_question"), "bad_value_shape"),
    (_payload(params={"measure": ["revenue", "cost"]}), "bad_value_shape"),
    ({"recipe": "correlate", "params": {"measures": "revenue"}, "flags": [], "confidence": 1}, "bad_value_shape"),
    ({"recipe": "correlate", "params": {"measures": []}, "flags": [], "confidence": 1}, "bad_value_shape"),
    (_payload(params=["revenue"]), "bad_value_shape"),
    (_payload(confidence=1.5), "bad_confidence"),
    (_payload(confidence="high"), "bad_confidence"),
    (_payload(confidence=True), "bad_confidence"),
])
def test_every_rejection_path_is_named(payload, reason):
    with pytest.raises(IntentRejected) as exc:
        recognize(payload, question=Q, columns=COLUMNS, source="model")
    assert exc.value.reason == reason
    assert reason in REJECTIONS


def test_nothing_is_coerced():
    """A missing optional parameter stays missing; a wrong one is not dropped."""
    i = recognize({"recipe": "describe", "params": {"measure": "revenue"}, "flags": [], "confidence": 0.5},
                  question=Q, columns=COLUMNS, source="model")
    assert "dimension" not in i.params
    with pytest.raises(IntentRejected):
        recognize({"recipe": "describe", "params": {"measure": "revenue", "dimension": "nope"}, "flags": [],
                   "confidence": 0.5}, question=Q, columns=COLUMNS, source="model")


def test_rejection_reason_vocabulary_is_closed():
    with pytest.raises(ValueError):
        IntentRejected("vibes")
