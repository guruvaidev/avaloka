"""The benchmark runs that fail because a defect they describe is live, declared.

A benchmark task that fails is the point of the benchmark: each one below
measures something the product gets wrong today, and the scorecard keeps
counting it as a failure -- ``n_passed`` and the overall score do not change
because a task is listed here. What the list changes is the exit code, and it
does that the way ``pytest.mark.xfail(strict=True)`` does:

* a listed task that fails does not fail the run;
* a failure that is **not** listed fails the run, as every failure did before;
* a listed task that **passes** fails the run too, until its entry is removed.

The third rule is what keeps this from being a mute button. A defect that is
fixed and left on the list would hide its own regression; strictness makes the
list exactly as long as the set of live defects, and no longer.

Before this existed the run exited 1 whenever any of these failed, so the
pull-request Benchmark Scorecard step and the public CI benchmark job were red
on every pull request, for the same eleven runs, and a twelfth -- a real
regression -- would have looked identical. Use ``--strict`` for the old rule.

Each reason names what is wrong, not what the fix should be. Add an entry only
for a defect that is confirmed and recorded (CHANGELOG, Known issues); a task
added to land red ahead of its fix belongs here in the same pull request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

#: task name -> what is wrong. Keyed by task, so every source scheme of a
#: listed task is covered; none of the tasks below runs on more than one.
KNOWN_DEFECTS: dict[str, str] = {
    # The CLI's spreadsheet reader (avaloka/io/loader.py::_read_excel).
    "xlsx_decoy_sheet_outweighs_the_answer":
        "the sheet with the most non-empty cells is read, whatever the question is about",
    "xlsx_sec_financial_report":
        "period columns of a 10-K statement stay text, and the stated unit (thousands) is not carried",
    "xlsx_footnote_markers_on_labels":
        "footnote markers (*, (a), †) stay on column names and the footnote block is read as rows",
    "xlsx_merged_header_cells":
        "the second row of a merged header is never read, so its columns come back unnamed",
    "xlsx_stacked_two_row_header":
        "a two-row header is read as one row, so period sub-headings are lost",
    "xlsx_second_table_below":
        "a second table under the first is read as more rows of the first",
    "xlsx_subtotals_and_grand_total":
        "subtotal and grand-total rows are read as data rows",
    "xlsx_totals_row_at_bottom":
        "a totals row at the bottom is read as a data row",
    "xlsx_parenthesised_negatives":
        "accounting negatives written (1,234) are not read as numbers",
    "xlsx_thousands_scaling_in_title":
        "a scale stated in the title (GBP thousands) is not applied or reported",
    # The conversational family grades probe scorers, not the product.
    "conv_avaloka_behaviour_is_unmeasured":
        "no path drives the conversational probes through Avaloka's own agent graph",
}


@dataclass
class GateVerdict:
    """What the exit code is decided from."""

    new_failures: list[str] = field(default_factory=list)       # failed, not listed
    known_failing: list[str] = field(default_factory=list)      # failed, listed
    listed_but_passing: list[str] = field(default_factory=list) # listed, passed: remove it

    @property
    def ok(self) -> bool:
        return not self.new_failures and not self.listed_but_passing

    def as_dict(self) -> dict:
        return {"ok": self.ok, "new_failures": self.new_failures,
                "known_failing": self.known_failing,
                "listed_but_passing": self.listed_but_passing,
                "declared_in": "avaloka/benchmark/known_defects.py"}


def _skipped(result) -> bool:
    return bool(result.error and "unavailable" in result.error)


def gate(results: Iterable, known: dict[str, str] = KNOWN_DEFECTS) -> GateVerdict:
    """Classify each run that counts. A skipped run neither passes nor fails."""
    verdict = GateVerdict()
    for r in results:
        if _skipped(r):
            continue
        label = f"{r.task} ({r.source_scheme})"
        if r.passed:
            if r.task in known:
                verdict.listed_but_passing.append(label)
        elif r.task in known:
            verdict.known_failing.append(label)
        else:
            verdict.new_failures.append(label)
    return verdict
