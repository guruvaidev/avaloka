"""Every upload format the product declares must have a benchmark task.

The API's upload dialog accepts nine extensions (``SUPPORTED_UPLOAD_EXTS`` in
``app/api/server.py``). Until this check existed, exactly one of them had a
benchmark task: ``DatasetSpec.builder`` wrote a CSV, and the three source
schemes re-read that same CSV, so eight declared formats were never read by the
suite at all -- while the suite reported 1.00.

Two of those eight do not fail loudly. Handed an XML or an Avro file, the CLI
loader's delimiter sniffer accepts it and returns a one-column frame named
after the container's magic bytes. A profile of that frame looks exactly like a
profile of a real table, so the gap could not be seen from the scorecard.

This is a declaration-drift check, built like the one on tool_choice: the
product's declaration and the suite's coverage are compared against a table
that must be kept current by hand, and the check fails in **both** directions.

* **UNDECLARED DIFFERENCE** -- the API accepts a format that no benchmark task
  carries, and no deferral has been declared for it. Add a task, or declare the
  deferral in ``DEFERRED_FORMATS`` with the reason.
* **STALE DECLARATION** -- a deferral is declared for a format that is now
  covered, or that the API no longer accepts. Delete the deferral; a reason
  that no longer applies is worse than none, because the next reader believes
  it.

A deferral belongs in code, with its reason, where review can see it. An
absence is how eight formats went uncovered without anyone noticing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from avaloka.benchmark.datasets_formats import DEFERRED_FORMATS
from avaloka.benchmark.suite import all_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SOURCE = REPO_ROOT / "app" / "api" / "server.py"

#: The constant is read out of the API's source with ``ast`` rather than by
#: importing it. Importing ``app.api.server`` would drag in FastAPI and the
#: whole application graph, and the CLI's own contract forbids ``avaloka``
#: depending on ``app``; parsing keeps this check honest about that boundary
#: and makes it runnable without the app's dependencies installed.
CONSTANT_NAME = "SUPPORTED_UPLOAD_EXTS"


def _declared_upload_formats() -> set[str]:
    """The extensions the API says it accepts, read from its source."""
    tree = ast.parse(SERVER_SOURCE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if CONSTANT_NAME not in names:
            continue
        value = node.value
        if not isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            raise AssertionError(
                f"{CONSTANT_NAME} in {SERVER_SOURCE} is no longer a literal "
                f"set/list/tuple but {type(value).__name__}. This check reads it "
                "statically; teach it the new shape rather than deleting it.")
        out = set()
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                raise AssertionError(
                    f"{CONSTANT_NAME} contains a non-literal element "
                    f"({ast.dump(element)[:80]}); this check cannot read it statically.")
            out.add(element.value.lower().lstrip("."))
        return out
    raise AssertionError(
        f"{CONSTANT_NAME} was not found in {SERVER_SOURCE}. It was the single "
        "source of truth for what the product accepts; if it moved or was "
        "renamed, point this check at the new one. Do not delete this check -- "
        "without it a format can be declared supported and never read.")


def _covered_formats() -> dict[str, list[str]]:
    """Declared format -> the benchmark tasks that carry a file of that format."""
    covered: dict[str, list[str]] = {}
    for task in all_tasks():
        fmt = task.dataset.declared_format
        if fmt:
            covered.setdefault(fmt.lower().lstrip("."), []).append(task.name)
    return covered


# --------------------------------------------------------------------------
# Guards against this check passing without testing anything. The failure mode
# this repo keeps producing is a control that silently does nothing, and each
# of the three inputs below can go empty in a way that would make the
# comparison below trivially true.
# --------------------------------------------------------------------------
def test_the_declared_format_list_is_readable_and_plausible():
    declared = _declared_upload_formats()
    assert declared, f"{CONSTANT_NAME} parsed as empty -- the check would pass vacuously"
    assert "csv" in declared, (
        f"{CONSTANT_NAME} does not contain 'csv' ({sorted(declared)}); this is "
        "almost certainly a parsing failure rather than a product change")
    assert len(declared) >= 5, (
        f"{CONSTANT_NAME} parsed as only {sorted(declared)}. If the product "
        "really dropped that many formats, update this floor deliberately.")


def test_the_coverage_mechanism_is_wired_at_all():
    """``declared_format`` is what makes coverage visible; if nothing sets it,
    every format reads as uncovered and the real signal is lost in the noise."""
    covered = _covered_formats()
    assert covered, (
        "no benchmark task sets DatasetSpec.declared_format, so coverage cannot "
        "be measured. This check reports every declared format as uncovered "
        "when the mechanism is unwired, which is indistinguishable from a suite "
        "that genuinely covers nothing -- wire it.")


def test_deferrals_carry_a_reason():
    for ext, reason in DEFERRED_FORMATS.items():
        assert isinstance(reason, str) and len(reason.strip()) >= 40, (
            f"the deferral for '{ext}' has no usable reason ({reason!r}). A "
            "deferral without a reason is an absence with extra steps.")


# --------------------------------------------------------------------------
# The check itself.
# --------------------------------------------------------------------------
def test_every_declared_upload_format_has_benchmark_coverage():
    declared = _declared_upload_formats()
    covered = _covered_formats()
    deferred = {k.lower().lstrip(".") for k in DEFERRED_FORMATS}

    problems: list[str] = []

    uncovered = sorted(declared - set(covered) - deferred)
    if uncovered:
        problems.append(
            "UNDECLARED DIFFERENCE: the API accepts "
            f"{len(uncovered)} format(s) that no benchmark task reads and that "
            f"are not declared as deferred: {uncovered}. "
            "Add a task with DatasetSpec(declared_format=...), or declare the "
            "deferral with its reason in "
            "avaloka/benchmark/datasets_formats.py::DEFERRED_FORMATS.")

    both = sorted(deferred & set(covered))
    if both:
        problems.append(
            "STALE DECLARATION: these formats are declared deferred but are now "
            f"covered by a task: {both}. Delete the deferral -- a reason that no "
            "longer applies is believed by the next reader.")

    gone = sorted(deferred - declared)
    if gone:
        problems.append(
            "STALE DECLARATION: these formats are declared deferred but the API "
            f"no longer accepts them: {gone}. Delete the deferral.")

    extra = sorted(set(covered) - declared)
    if extra:
        problems.append(
            "UNDECLARED DIFFERENCE: these formats have benchmark tasks but the "
            f"API does not accept them: {extra}. Either the product dropped a "
            "format and the task should go, or the declaration is wrong.")

    if problems:
        detail = "\n\n".join(problems)
        summary = (f"\n\ndeclared={sorted(declared)}"
                   f"\ncovered={ {k: v for k, v in sorted(covered.items())} }"
                   f"\ndeferred={sorted(deferred)}")
        pytest.fail(detail + summary, pytrace=False)
