"""One table of null-word cases, read by both implementations of the rule.

The rule -- an empty cell is always missing; a null-like word is missing in a
typed column and a label in a text column -- is implemented twice:

* ``file_handler/null_policy.py`` for the platform, the canonical one;
* ``avaloka/io/loader.py`` for the CLI, a hand-written second copy, because
  the CLI is standalone and may not import ``file_handler``.

Two copies drift. Every test of either side takes its cases from HERE, so there
is one list and not two that happen to look alike:

* ``tests/test_cli_null_policy_parity.py`` -- the platform reader against the
  expected outcomes, and the two readers against each other;
* ``tests/avaloka/test_loader_null_words.py`` -- the CLI reader against the
  same expected outcomes.

This module imports nothing from the product: standard library only, so that
importing it cannot pull ``app`` or ``file_handler`` into the CLI's tests.

Each case is CSV text and what a correct reader returns for it, as
``{column: [cell, ...]}`` with ``MISSING`` for a missing cell. ``outcome``
turns a frame into that shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

MISSING = None


@dataclass(frozen=True)
class Case:
    name: str
    csv: str
    expected: dict[str, list[Any]]
    # A DECLARED DIFFERENCE between the two implementations. Empty for every
    # row but the ones where the CLI and the platform are known, and approved,
    # to return different things; then it holds the reason, and
    # ``platform_returns`` holds what the platform returns instead of
    # ``expected``. It is a declaration, not a tolerance, and it is checked in
    # both directions by tests/test_cli_null_policy_parity.py:
    #   * a difference on a row that does not declare one fails;
    #   * a row that declares one fails if the two implementations agree, so a
    #     declaration cannot outlive the difference it describes.
    declared_difference: str = ""
    platform_returns: Optional[dict[str, list[Any]]] = None

    def __post_init__(self) -> None:
        if bool(self.declared_difference.strip()) != (self.platform_returns is not None):
            raise ValueError(
                f"case {self.name!r}: declared_difference and platform_returns go together "
                "-- a difference needs its reason, and a reason needs the difference")
        if self.platform_returns is not None and self.platform_returns == self.expected:
            raise ValueError(f"case {self.name!r} declares a difference but records none")


def outcome(frame) -> dict[str, list[Any]]:
    """A frame as plain cells: ``MISSING`` for a missing cell, else the value."""
    result = {}
    for column in frame.columns:
        cells = []
        for cell in frame[column].tolist():
            missing = cell is None or cell != cell        # NaN is the one value != itself
            cells.append(MISSING if missing else cell)
        result[str(column)] = cells
    return result


_DATES = [MISSING if i % 10 == 0 else f"2024-01-{i % 28 + 1:02d}" for i in range(40)]
_DATES_CSV = "order_date,method\n" + "".join(
    f"{'NA' if day is MISSING else day},{'None' if i % 3 else 'card'}\n"
    for i, day in enumerate(_DATES))

CASES = [
    Case(
        name="text column: None, NA and N/A are real categories",
        csv="method,n\ncard,1\nNone,2\nNA,3\nN/A,4\ncash,5\n",
        expected={"method": ["card", "None", "NA", "N/A", "cash"], "n": [1, 2, 3, 4, 5]},
    ),
    Case(
        name="text column: every other null-like word is a label too",
        csv="method\ncard\nnull\nNULL\nnan\nNaN\nn/a\n#N/A\n<NA>\n",
        expected={"method": ["card", "null", "NULL", "nan", "NaN", "n/a", "#N/A", "<NA>"]},
    ),
    Case(
        name="numeric column: gaps written as NA stay missing and numeric",
        csv="amount\n1.5\nNA\n2.5\nN/A\nnull\n",
        expected={"amount": [1.5, MISSING, 2.5, MISSING, MISSING]},
    ),
    Case(
        name="integer column with one gap",
        csv="count\n1\nNone\n3\n",
        expected={"count": [1.0, MISSING, 3.0]},
    ),
    Case(
        name="boolean column: gaps written as NA stay missing",
        csv="paid,n\nTrue,1\nNA,2\nFalse,3\n",
        expected={"paid": [True, MISSING, False], "n": [1, 2, 3]},
    ),
    Case(
        name="text and numeric side by side, each by its own rule",
        csv="method,amount\ncard,1.5\nNone,NA\nNA,2.5\ncash,N/A\n",
        expected={"method": ["card", "None", "NA", "cash"],
                  "amount": [1.5, MISSING, 2.5, MISSING]},
    ),
    Case(
        name="a column that is entirely null-like words stays missing",
        csv="name,note\nann,NA\nbob,None\ncy,N/A\n",
        expected={"name": ["ann", "bob", "cy"], "note": [MISSING, MISSING, MISSING]},
    ),
    Case(
        name="mixed case is ordinary text, in either kind of reader",
        csv="method\nNONE\nNa\nnone\ncard\nNA\n",
        expected={"method": ["NONE", "Na", "none", "card", "NA"]},
    ),
    Case(
        name="surrounding whitespace is kept as written and is not a null word",
        csv="method,n\n NA ,1\ncard,2\n None,3\nNA,4\n",
        expected={"method": [" NA ", "card", " None", "NA"], "n": [1, 2, 3, 4]},
    ),
    Case(
        name="empty cell and empty quoted string are missing in a text column",
        csv='method,n\ncard,1\n,2\n"",3\nNone,4\n',
        expected={"method": ["card", MISSING, MISSING, "None"], "n": [1, 2, 3, 4]},
    ),
    Case(
        name="genuinely missing cells in a numeric column",
        csv="method,amount\ncard,1.5\ncash,\nNone,2.5\n",
        expected={"method": ["card", "cash", "None"], "amount": [1.5, MISSING, 2.5]},
    ),
    Case(
        name="quoted null-like words are still labels in a text column",
        csv='method,n\n"None",1\n"card",2\n"NA",3\n',
        expected={"method": ["None", "card", "NA"], "n": [1, 2, 3]},
    ),
    Case(
        name="numbers with stray text: untypable, so a text column",
        csv="amount\n1.5\nunknown\nNA\n2.5\n",
        expected={"amount": ["1.5", "unknown", "NA", "2.5"]},
    ),
    Case(
        name="whitespace-only cells are not real strings: the column stays typed",
        csv="pad,n\n  ,1\nNA,2\n  ,3\n",
        expected={"pad": ["  ", MISSING, "  "], "n": [1, 2, 3]},
    ),
    Case(
        name="no null-like words at all",
        csv="a,b\n1,x\n2,y\n",
        expected={"a": [1, 2], "b": ["x", "y"]},
    ),
    Case(
        name="date column: gaps written as NA stay missing",
        csv=_DATES_CSV,
        expected={"order_date": _DATES,
                  "method": ["None" if i % 3 else "card" for i in range(40)]},
        platform_returns={"order_date": ["NA" if day is MISSING else day for day in _DATES],
                          "method": ["None" if i % 3 else "card" for i in range(40)]},
        declared_difference=(
            "The platform reader keeps a date column's NA gaps as labels, "
            "because pandas hands dates over as text and that reader asks only "
            "whether pandas typed the column. The CLI treats a column that "
            "reads as dates as typed, because otherwise stats.infer_role turns "
            "it categorical and the planner loses its temporal split."),
    ),
]

CASE_IDS = [case.name for case in CASES]
