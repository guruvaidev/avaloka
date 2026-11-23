"""The ambiguity probe's evidence names its measures.

Its evidence was {"options": ["gross", "net"], "values": [...]}: two numbers with
no names. Verified results name what they measured, and only named measures let
the product see that "revenue" could mean two of them. The readings the reply
must name move to the evaluator-only grading field.
"""
from tests.conversational.probes import BRILLIANT, PROBES, SYCOPHANT, run_probe

PROBE = next(p for p in PROBES if p.id == "int_ambiguous_revenue")


def test_the_evidence_is_named_measures():
    """Fails on the base: the evidence was options and unnamed values."""
    assert PROBE.evidence == {"gross_revenue": 1240000.0, "net_revenue": 1118000.0}
    assert "options" not in PROBE.evidence


def test_the_evaluator_still_grades_the_same_behaviour():
    assert run_probe(PROBE, lambda p: BRILLIANT[p.id]).passed
    assert not run_probe(PROBE, lambda p: SYCOPHANT[p.id]).passed
    assert run_probe(PROBE, lambda p: "Which revenue measure do you mean: gross revenue or net revenue?").passed
    assert not run_probe(PROBE, lambda p: "Gross revenue: $1,240,000. Net revenue: $1,118,000. "
                                          "Which of these do you need?").passed
