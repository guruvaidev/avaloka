"""The CLI's copy of the null-word rule, pinned to the platform's.

``avaloka/`` is standalone and may not import ``file_handler``, so
``avaloka/io/loader.py`` carries a hand-written second implementation of the
CSV half of ``file_handler/null_policy.py``. Two copies drift. This is the only
file that imports both, and it does two things with the one table in
``tests/shared_cases/null_word_labels.py``:

* reads each case through the platform's reader and checks the expected
  outcome (the CLI's reader gets the same table in
  ``tests/avaloka/test_loader_null_words.py``);
* reads each case through BOTH readers and compares what they return with each
  other, cell for cell. Change a rule in one copy and this fails by naming the
  case and showing the two outcomes side by side.

One difference is DECLARED in the table, with its reason, rather than
tolerated: dates. A difference the table does not declare fails, and so does a
declared one once the two implementations agree.
"""

from __future__ import annotations

import pandas as pd
import pytest

from avaloka.io import loader
from file_handler import null_policy
from tests.shared_cases.null_word_labels import CASE_IDS, CASES, MISSING, outcome


def _write(tmp_path, case):
    path = tmp_path / "case.csv"
    path.write_text(case.csv)
    return path


def _cli(path):
    return outcome(loader._read_csv(path, ","))


def _platform(path):
    return outcome(null_policy.read_csv_best_effort(str(path)))


def test_same_word_list():
    assert loader.NULL_WORDS == null_policy.NULL_WORDS


def test_same_sniff_depth():
    assert loader.CSV_SNIFF_ROWS == null_policy.CSV_SNIFF_ROWS


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_platform_reader_returns_the_expected_outcome(tmp_path, case):
    assert _platform(_write(tmp_path, case)) == (case.platform_returns or case.expected)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_two_implementations_agree(tmp_path, case):
    """The drift check: same file, both readers, outcomes compared directly."""
    path = _write(tmp_path, case)
    cli, platform = _cli(path), _platform(path)
    if not case.declared_difference:
        assert cli == platform, (
            f"UNDECLARED DIFFERENCE: the CLI's copy of the null-word rule and "
            f"file_handler/null_policy.py disagree on {case.name!r}")
        return
    # A declared difference is held in both directions.
    assert cli != platform, (
        f"STALE DECLARATION: {case.name!r} declares a difference between the CLI and "
        f"the platform, but the two now return the same thing. Remove the declaration "
        f"from tests/shared_cases/null_word_labels.py. It said: {case.declared_difference}")
    assert cli == case.expected and platform == case.platform_returns, (
        f"{case.name!r} declares a difference, but not this one. "
        f"Declared: {case.declared_difference}")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_two_implementations_pick_the_same_text_columns(tmp_path, case):
    head = pd.read_csv(_write(tmp_path, case))
    cli, platform = loader._csv_text_columns(head), null_policy.csv_text_columns(head)
    if not case.declared_difference:
        assert cli == platform
    else:
        # the declared difference: the platform also calls the date column text
        assert cli == ["method"] and platform == ["order_date", "method"]


def test_exactly_one_difference_is_declared():
    """A second declared difference is a decision, so it has to be made here."""
    declared = [case for case in CASES if case.declared_difference]
    assert [case.name for case in declared] == ["date column: gaps written as NA stay missing"]
    assert "temporal split" in declared[0].declared_difference


def test_a_declaration_needs_both_its_reason_and_its_difference():
    from tests.shared_cases.null_word_labels import Case

    with pytest.raises(ValueError, match="go together"):
        Case("x", "a\n1\n", {"a": [1]}, platform_returns={"a": [2]})
    with pytest.raises(ValueError, match="go together"):
        Case("x", "a\n1\n", {"a": [1]}, declared_difference="because")
    with pytest.raises(ValueError, match="records none"):
        Case("x", "a\n1\n", {"a": [1]}, declared_difference="because",
             platform_returns={"a": [1]})


def test_the_table_is_not_trivial(tmp_path):
    """If no case kept a label or dropped a gap, the comparisons compare nothing."""
    kept = [c for c in CASES if any(
        cell in null_policy.NULL_WORDS
        for cells in c.expected.values() for cell in cells if isinstance(cell, str))]
    gaps = [c for c in CASES if any(
        cell is MISSING for cells in c.expected.values() for cell in cells)]
    assert len(kept) >= 5 and len(gaps) >= 5
    # and a plain pandas read does NOT satisfy the table
    plain_wrong = [c.name for c in CASES
                   if outcome(pd.read_csv(_write(tmp_path, c))) != c.expected]
    assert len(plain_wrong) >= 5
