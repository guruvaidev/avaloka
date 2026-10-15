"""The reviewer's verdict is read by type, not by truthiness.

JSON mode guarantees valid JSON, not correct types. A reviewer that answers
``"is_logically_correct": "false"`` has said "not correct"; read by
truthiness, that non-empty string was an approval.

The rule (``validator._read_verdict_flag``), applied to all four reads
(``is_logically_correct`` and ``is_fabricated`` on the first reply and on the
fabrication recheck):

* a JSON boolean is taken as it is;
* "true" / "false" (any case, surrounding whitespace ignored) and the integers
  1 / 0 are read as the booleans they spell;
* an absent key or ``null`` is unchanged: the model did not say, which reads
  as "not true";
* anything else is no verdict, neither answer.

Replies are written as raw JSON fragments so the shape under test is exactly
what the reviewer would send.
"""
import logging
from types import SimpleNamespace

import pytest

import app.agents.validator as validator_mod

RATIONALE = "Column patient_ssn_hash is misused."
UNPARSABLE = "Reviewer response unparsable; skipping logical validation."
ABSENT = None  # the key is left out of the reply

APPROVED = {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}
SOFT = {**APPROVED, "logical_review_feedback": RATIONALE}
BLOCKED = {"logical_semantic_error": True, "code_validation_feedback": RATIONALE}
NO_VERDICT = {"logical_semantic_error": False, "code_validation_feedback": UNPARSABLE}

VERDICT_KEYS = (
    "logical_review_decided_by",
    "logical_review_judged_correct",
    "logical_review_judged_fabricated",
    "logical_review_recheck_ran",
    "logical_review_recheck_confirmed",
    "logical_review_overruled",
)

# A recheck that confirms fabrication, and one that withdraws it and approves.
CONFIRMS = ("false", "true")
WITHDRAWS_AND_APPROVES = ("true", "false")


def _raw(correct, fabricated, rationale=RATIONALE):
    parts = []
    if correct is not ABSENT:
        parts.append(f'"is_logically_correct": {correct}')
    if fabricated is not ABSENT:
        parts.append(f'"is_fabricated": {fabricated}')
    parts.append(f'"rationale": "{rationale}"')
    return "{" + ", ".join(parts) + "}"


class ScriptedReviewer:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self._replies.pop(0))


def _review(monkeypatch, first, recheck=CONFIRMS, strict=None):
    """Run the node; ``first`` / ``recheck`` are (correct, fabricated) fragments,
    or a whole raw reply string."""
    if strict is None:
        monkeypatch.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
    else:
        monkeypatch.setenv(validator_mod.STRICT_LOGIC_ENV, strict)
    replies = [r if isinstance(r, str) else _raw(*r) for r in (first, recheck)]
    reviewer = ScriptedReviewer(replies)
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)
    out = validator_mod.logical_semantic_validator_node({
        "generated_code": "def main(df):\n    return df\n",
        "user_prompt": "Total a column",
        "schema": {"patient_ssn_hash": "float64"},
        "syntax_error": False,
        "static_semantic_error": False,
        "execution_output_preview": [{"patient_ssn_hash": 1.0}],
    })
    return out, reviewer.calls


def _fields(decided_by, correct=None, fabricated=None, ran=False, confirmed=None, overruled=False):
    return dict(zip(VERDICT_KEYS, (decided_by, correct, fabricated, ran, confirmed, overruled)))


def _check(out, calls, pre_existing, fields, expected_calls):
    assert {k: v for k, v in out.items() if k not in VERDICT_KEYS} == pre_existing
    assert {k: out[k] for k in VERDICT_KEYS} == fields
    assert calls == expected_calls


# ---------------------------------------------------------------------------
# is_logically_correct on the first reply
# ---------------------------------------------------------------------------

READ_TRUE = ["true", '"true"', '"True"', '" TRUE "', "1"]
READ_FALSE = ["false", '"false"', '"False"', '" false "', '"FALSE"', "0"]
NOT_SAID = ["null", ABSENT]
MALFORMED = ['"yes"', '"maybe"', '"False."', '""', "[]", "[true]", "{}", '{"v": true}',
             "0.5", "2", "-1", "1.0", "0.0"]


@pytest.mark.parametrize("fabricated", ["false", "true"])
@pytest.mark.parametrize("shape", READ_TRUE)
def test_correct_read_as_true_approves(monkeypatch, shape, fabricated):
    out, calls = _review(monkeypatch, (shape, fabricated))
    _check(out, calls, APPROVED,
           _fields("approved", correct=True, fabricated=fabricated == "true"), 1)


@pytest.mark.parametrize("shape", READ_FALSE + NOT_SAID)
def test_correct_read_as_false_is_an_overruled_soft_flag(monkeypatch, shape):
    out, calls = _review(monkeypatch, (shape, "false"))
    _check(out, calls, SOFT,
           _fields("soft_flag_default_mode", correct=False, fabricated=False, overruled=True), 1)


@pytest.mark.parametrize("shape", READ_FALSE + NOT_SAID)
def test_correct_read_as_false_with_fabrication_reaches_the_recheck_and_blocks(monkeypatch, shape):
    out, calls = _review(monkeypatch, (shape, "true"), recheck=CONFIRMS)
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_confirmed", correct=False, fabricated=True,
                   ran=True, confirmed=True), 2)


@pytest.mark.parametrize("shape", READ_FALSE)
def test_correct_read_as_false_blocks_in_strict_mode(monkeypatch, shape):
    out, calls = _review(monkeypatch, (shape, "false"), strict="1")
    _check(out, calls, BLOCKED,
           _fields("blocked_strict_mode", correct=False, fabricated=False), 1)


@pytest.mark.parametrize("strict", [None, "1"])
@pytest.mark.parametrize("fabricated, recorded", [
    ("false", False), ('"false"', False), ("0", False), ("null", False), (ABSENT, False),
    ('"yes"', None), ("[]", None),  # both keys unreadable: nothing to act on
])
@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_without_fabrication_is_no_verdict(
        monkeypatch, shape, fabricated, recorded, strict):
    """Neither approval nor rejection, in default and in strict mode: let
    through like an unparseable reply. is_fabricated is recorded as read."""
    out, calls = _review(monkeypatch, (shape, fabricated), strict=strict)
    _check(out, calls, NO_VERDICT,
           _fields("no_verdict_malformed_value", fabricated=recorded), 1)


# Each verdict is read on its own: a readable is_fabricated=true is acted on
# as before even when is_logically_correct is unreadable.
FABRICATED = ["true", '"true"', "1"]


@pytest.mark.parametrize("fabricated", FABRICATED)
@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_with_fabrication_reaches_the_recheck_and_blocks(
        monkeypatch, shape, fabricated):
    out, calls = _review(monkeypatch, (shape, fabricated), recheck=CONFIRMS)
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_confirmed", correct=None, fabricated=True,
                   ran=True, confirmed=True), 2)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_with_fabrication_blocks_when_the_recheck_crashes(monkeypatch, shape):
    out, calls = _review(monkeypatch, (shape, "true"), recheck="[1]")
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_recheck_failed", correct=None, fabricated=True,
                   ran=True), 2)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_with_fabrication_can_be_approved_on_recheck(monkeypatch, shape):
    out, calls = _review(monkeypatch, (shape, "true"), recheck=WITHDRAWS_AND_APPROVES)
    _check(out, calls, APPROVED,
           _fields("approved_on_recheck", correct=None, fabricated=True,
                   ran=True, confirmed=False), 2)


@pytest.mark.parametrize("recheck_correct", READ_TRUE)
@pytest.mark.parametrize("recheck_fabricated", ['"yes"', "[]", "0.5", "2", '""'])
@pytest.mark.parametrize("shape", ['"yes"', "[]", "0.5", "2", '""', "{}"])
def test_malformed_correct_with_fabrication_and_unreadable_recheck_flag_approves_on_recheck(
        monkeypatch, shape, recheck_fabricated, recheck_correct):
    """Each key on its own, on both replies: the recheck's readable approval is
    acted on although its fabrication flag, and the first reply's correctness,
    were unreadable."""
    out, calls = _review(monkeypatch, (shape, "true"),
                         recheck=(recheck_correct, recheck_fabricated))
    _check(out, calls, APPROVED,
           _fields("approved_on_recheck", correct=None, fabricated=True,
                   ran=True, confirmed=None), 2)


@pytest.mark.parametrize("recheck, confirmed", [
    (("false", "false"), False), (("false", '"yes"'), None), ("not json", False),
])
@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_with_unconfirmed_fabrication_is_no_verdict_not_an_overrule(
        monkeypatch, shape, recheck, confirmed):
    """Nothing readable said "not correct", so there is nothing to overrule."""
    out, calls = _review(monkeypatch, (shape, "true"), recheck=recheck)
    _check(out, calls, NO_VERDICT,
           _fields("no_verdict_malformed_value", correct=None, fabricated=True,
                   ran=True, confirmed=confirmed, overruled=False), 2)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_correct_with_fabrication_blocks_in_strict_mode(monkeypatch, shape):
    """Strict mode acts on a readable fabrication verdict as before: block, no
    recheck. An unreadable correctness verdict alone does not block (above)."""
    out, calls = _review(monkeypatch, (shape, "true"), strict="1")
    _check(out, calls, BLOCKED,
           _fields("blocked_strict_mode", correct=None, fabricated=True), 1)


def test_whole_reply_empty_object_is_unchanged(monkeypatch):
    out, calls = _review(monkeypatch, "{}")
    _check(out, calls, {**APPROVED, "logical_review_feedback": "No rationale provided."},
           _fields("soft_flag_default_mode", correct=False, fabricated=False, overruled=True), 1)


def test_headline_string_false_is_not_an_approval(monkeypatch):
    out, _ = _review(monkeypatch, ('"false"', "false"))
    # what the pipeline sees is the existing default-mode rule: unchanged
    assert out["code_validation_feedback"] == "APPROVED"
    assert out["logical_semantic_error"] is False
    # what is recorded: a negative verdict that was overruled, not an approval
    assert out["logical_review_decided_by"] == "soft_flag_default_mode"
    assert out["logical_review_judged_correct"] is False
    assert out["logical_review_overruled"] is True


# ---------------------------------------------------------------------------
# is_fabricated on the first reply (reviewer said not correct)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["true", '"true"', '" True "', "1"])
def test_fabricated_read_as_true_reaches_the_recheck(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", shape), recheck=CONFIRMS)
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_confirmed", correct=False, fabricated=True,
                   ran=True, confirmed=True), 2)


@pytest.mark.parametrize("shape", ["false", '"false"', '"False"', '" false "', "0", "null", ABSENT])
def test_fabricated_read_as_false_runs_no_recheck(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", shape))
    _check(out, calls, SOFT,
           _fields("soft_flag_default_mode", correct=False, fabricated=False, overruled=True), 1)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_fabricated_is_not_a_fabrication_verdict(monkeypatch, shape):
    """The "not correct" verdict stands; fabrication is unresolved, so no
    recheck and no block. Let through by the default-mode rule, recorded apart
    from a clean overrule, with the flag as None rather than False."""
    out, calls = _review(monkeypatch, ("false", shape))
    _check(out, calls, SOFT,
           _fields("soft_flag_fabrication_unreadable", correct=False, fabricated=None,
                   overruled=True), 1)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_fabricated_still_blocks_in_strict_mode(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", shape), strict="1")
    _check(out, calls, BLOCKED,
           _fields("blocked_strict_mode", correct=False, fabricated=None), 1)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_fabricated_does_not_unsettle_an_approval(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("true", shape))
    _check(out, calls, APPROVED, _fields("approved", correct=True, fabricated=None), 1)


# ---------------------------------------------------------------------------
# The recheck reply (first reply: not correct, fabricated)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["true", '"true"', '"TRUE"', "1"])
def test_recheck_fabricated_read_as_true_confirms(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=("false", shape))
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_confirmed", correct=False, fabricated=True,
                   ran=True, confirmed=True), 2)


@pytest.mark.parametrize("shape", ["false", '"false"', '" False "', "0", "null", ABSENT])
def test_recheck_fabricated_read_as_false_does_not_confirm(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=("false", shape))
    _check(out, calls, SOFT,
           _fields("soft_flag_default_mode", correct=False, fabricated=True,
                   ran=True, confirmed=False, overruled=True), 2)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_recheck_fabricated_neither_confirms_nor_approves(monkeypatch, shape):
    """A reply, not a crash: it does not confirm, so nothing blocks (the same
    outcome as a recheck reply that did not parse), and it is recorded apart."""
    out, calls = _review(monkeypatch, ("false", "true"), recheck=("false", shape))
    _check(out, calls, SOFT,
           _fields("soft_flag_fabrication_unreadable", correct=False, fabricated=True,
                   ran=True, confirmed=None, overruled=True), 2)


@pytest.mark.parametrize("shape", MALFORMED)
def test_malformed_recheck_fabricated_with_a_usable_approval_approves(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=("true", shape))
    _check(out, calls, APPROVED,
           _fields("approved_on_recheck", correct=False, fabricated=True,
                   ran=True, confirmed=None), 2)


def test_a_crashed_recheck_still_keeps_the_first_verdict_and_blocks(monkeypatch):
    """Unchanged: only a recheck that raises (or is not an object) is a crash."""
    out, calls = _review(monkeypatch, ("false", "true"), recheck="[1]")
    _check(out, calls, BLOCKED,
           _fields("blocked_fabrication_recheck_failed", correct=False, fabricated=True,
                   ran=True), 2)


@pytest.mark.parametrize("shape", READ_TRUE)
def test_recheck_correct_read_as_true_approves_on_recheck(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=(shape, "false"))
    _check(out, calls, APPROVED,
           _fields("approved_on_recheck", correct=False, fabricated=True,
                   ran=True, confirmed=False), 2)


@pytest.mark.parametrize("shape", READ_FALSE + NOT_SAID + MALFORMED)
def test_recheck_correct_not_true_is_not_an_approval(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=(shape, "false"))
    _check(out, calls, SOFT,
           _fields("soft_flag_default_mode", correct=False, fabricated=True,
                   ran=True, confirmed=False, overruled=True), 2)


@pytest.mark.parametrize("shape", MALFORMED + ['"false"'])
def test_confirmed_recheck_ignores_its_other_key(monkeypatch, shape):
    out, calls = _review(monkeypatch, ("false", "true"), recheck=(shape, "true"))
    assert out["logical_review_decided_by"] == "blocked_fabrication_confirmed"
    assert out["logical_semantic_error"] is True and calls == 2


# ---------------------------------------------------------------------------
# The helper itself: a bool is not an int, in either direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), (1, True), (0, False),
    ("true", True), ("FALSE", False), ("  True\n", True),
    (None, None),
])
def test_helper_reads(value, expected):
    got = validator_mod._read_verdict_flag({"k": value}, "k")
    assert got is expected  # identity: 1 comes back as True, never as the int 1


def test_helper_absent_key_is_none():
    assert validator_mod._read_verdict_flag({}, "k") is None


@pytest.mark.parametrize("value", [
    2, -1, 0.0, 1.0, 0.5, float("nan"), "", "yes", "no", "False.", "0", "1", "null",
    [], [True], {}, {"v": True},
])
def test_helper_malformed(value):
    assert validator_mod._read_verdict_flag({"k": value}, "k") is validator_mod._MALFORMED_VERDICT


# ---------------------------------------------------------------------------
# Counters: a malformed verdict is a review with no verdict
# ---------------------------------------------------------------------------

def test_malformed_value_is_counted_as_no_verdict(monkeypatch, caplog):
    from app.agents.logical_review_verdict import (
        logical_review_counts,
        reset_logical_review_counts,
    )

    reset_logical_review_counts()
    caplog.set_level(logging.INFO, logger="app.agents.logical_review_verdict")
    mix = (
        [(('"yes"', "false"), CONFIRMS)] * 3      # malformed -> no verdict
        + [(("0.5", "false"), CONFIRMS)]          # malformed -> no verdict
        + ["not json at all"]                     # unparseable -> no verdict
        + [(('"false"', "false"), CONFIRMS)] * 2  # string false -> overruled
        + [(('"true"', "false"), CONFIRMS)]       # string true -> approved
        + [(("true", "false"), CONFIRMS)]         # approved
        + [(('"false"', "true"), CONFIRMS)]       # string false + fabricated -> blocked
        + [(("false", '"yes"'), CONFIRMS)]        # fabrication unreadable -> overruled, apart
        + [(("[]", "true"), CONFIRMS)]            # correctness unreadable, fabrication confirmed -> blocked
        + [(("[]", "true"), ("false", "false"))]  # ... not confirmed -> no verdict
    )
    for item in mix:
        with monkeypatch.context() as m:
            if isinstance(item, str):
                _review(m, item)
            else:
                _review(m, item[0], recheck=item[1])

    c = logical_review_counts()
    assert c["calls"] == c["reviews"] == 13
    assert c["no_verdict"] == 6
    assert c["by_decision"]["no_verdict_malformed_value"] == 5
    assert c["by_decision"]["no_verdict_unparseable"] == 1
    assert c["overruled"] == 3
    assert c["by_decision"]["soft_flag_default_mode"] == 2          # clean overrules
    assert c["by_decision"]["soft_flag_fabrication_unreadable"] == 1
    assert c["fabrication_flagged"] == 3 and c["fabrication_confirmed"] == 2
    assert c["approved_by_reviewer"] == 2
    assert c["blocked"] == 2
    assert (c["approved_by_reviewer"] + c["overruled"] + c["no_verdict"] + c["blocked"]
            == c["reviews"])
    # a malformed verdict is neither correct nor incorrect
    assert c["judged_correct"] == 2
    assert c["judged_not_correct"] == 4
    # ... and the one block that rests on fabrication alone has no readable
    # correctness verdict either, so it is in neither count.
    assert c["judged_correct"] + c["judged_not_correct"] + c["no_verdict"] == c["reviews"] - 1

    lines = [r.getMessage() for r in caplog.records
             if r.name == "app.agents.logical_review_verdict"]
    assert "decided_by=no_verdict_malformed_value judged_correct=None" in lines[0]
    assert "yes" not in lines[0]
