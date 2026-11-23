"""The golden runner: discover ``golden/<recipe>/<case>/``, run the recipe
without a model, and assert the case. ``assert_golden`` is shared with
``test_bite.py`` so a mutation is judged by exactly the assertions a golden
makes.

A case is three files: ``make_data.py`` (``make() -> DataFrame``, seeded),
``intent.json`` (the parsed Intent) and ``expected.json``. An expected
figure's ``id`` is the figure's ``evidence_ref`` pointer (stable across
re-ordering); a plain ``fN`` id is accepted too.

``assert_notebook_parity`` is the design's parity test (§2.4): the case's
notebook is executed on the case's data and every figure of the report must
match the notebook's leaf at its ``evidence_ref``, by the report's own rule.
It is shared with ``test_bite.py`` so a notebook that drifts is judged by
exactly the assertion the parity test makes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

import avaloka.recipes.library  # noqa: F401  (registers the recipes)
from avaloka.outcomes import format as _format
from avaloka.outcomes import semantics as _sem
from avaloka.outcomes import spec as _spec
from avaloka.recipes import Intent, get, notebook_exec
from avaloka.recipes.engine import connect
from avaloka.recipes.report import rendered_slots, verify_report
from avaloka.recipes.run import build_profile, run

GOLDEN = Path(__file__).resolve().parent / "golden"
REPO = Path(__file__).resolve().parents[2]


def cases() -> list[Path]:
    return sorted(p for p in GOLDEN.glob("*/*") if (p / "expected.json").is_file())


def case_id(case: Path) -> str:
    return f"{case.parent.name}/{case.name}"


def make_frame(case: Path) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(f"golden_{case.parent.name}_{case.name}", case / "make_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.make()


def run_case(case: Path):
    intent = json.loads((case / "intent.json").read_text())
    expected = json.loads((case / "expected.json").read_text())
    df = make_frame(case)
    recipe = get(expected["recipe"])
    result = run(recipe, Intent(**intent), connect(df), _sem.infer(df), build_profile(df, source="fixture.csv"))
    return expected, result


def _figure_by_id(report: dict, fid: str) -> dict | None:
    for fig in report["figures"]:
        if fid == fig["id"] or fid == fig["evidence_ref"]:
            return fig
    return None


def assert_golden(case: Path) -> None:
    expected, result = run_case(case)
    assert result.recipe_id == expected["recipe"]
    assert int(result.version.split(".")[0]) == expected["version_major"]
    assert result.verdict.value == expected["verdict"], (result.verdict, result.reason, result.refusal, result.ask)
    assert result.reason == expected.get("reason")
    if result.verdict.value == "ask":
        assert result.ask is not None and result.ask.parameter == expected["ask_parameter"]
    if result.report is None:
        assert expected.get("figures", []) == [] and expected.get("caveats", []) == []
        return
    report = result.report
    verify_report(report)
    assert report["provenance"]["llm_used"] is False
    for want in expected["figures"]:
        got = _figure_by_id(report, want["id"])
        assert got is not None, f"no figure {want['id']} in {[f['evidence_ref'] for f in report['figures']]}"
        assert got["unit"] == want["unit"], (want["id"], got["unit"], want["unit"])
        assert _format.figure_matches(want["value"], want["decimals"], got["value"]), \
            f"{want['id']}: expected {want['value']} at {want['decimals']} decimals, got {got['value']}"
    assert {c["kind"] for c in report["caveats"]} == set(expected["caveats"])
    for pointer, text in rendered_slots(report):
        for term in expected.get("must_not_say", []):
            assert term.lower() not in text.lower(), f"{pointer} says {term!r}: {text}"
    for pointer, text in _spec.text_slots(report):
        assert not _spec.free_numbers(text), f"{pointer} carries a free number: {text}"
    notebook = result.notebook
    assert notebook is not None
    json.dumps(notebook)  # valid JSON
    tagged = [c for c in notebook["cells"] if "parameters" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1


# ── notebook parity ────────────────────────────────────────────────────────

def execute_notebook(case: Path, notebook: dict, *, frame: pd.DataFrame | None = None) -> dict:
    """Run ``notebook`` in a fresh kernel on the case's frame (or ``frame``)
    written to a CSV in a temporary directory, with ``DATA_SOURCE`` pointed at
    it, and return the evidence it printed. The kernel is the test's own
    interpreter with this repository on its path, since the ``profile``
    notebook imports ``avaloka.stats``."""
    df = make_frame(case) if frame is None else frame
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "data.csv"
        df.to_csv(csv, index=False)
        return notebook_exec.execute(notebook, parameters={"DATA_SOURCE": str(csv)}, env={"PYTHONPATH": path})


def assert_figures_reproduced(report: dict, evidence: dict, label: str) -> None:
    """Every figure of ``report`` matches the executed notebook's ``evidence``
    leaf at the figure's ``evidence_ref``, by ``figure_matches`` at the
    figure's decimals; a missing leaf fails the same way."""
    for fig in report["figures"]:
        pointer = fig["evidence_ref"]
        try:
            value, unit, _ = _spec.evidence_leaf(evidence, pointer)
        except (KeyError, ValueError) as exc:
            raise AssertionError(f"{label}: the notebook has no evidence leaf at {pointer} ({exc})") from None
        leaf = value if unit is None else {"value": value, "unit": unit}
        assert _format.figure_matches(fig["value"], fig["decimals"], leaf, figure_unit=fig["unit"]), \
            (f"{label}: {pointer} is {fig['value']} {fig['unit']} in the report but "
             f"{value} {unit or fig['unit']} in the notebook (at {fig['decimals']} decimals)")


def assert_notebook_parity(case: Path) -> None:
    """The case's report is reproduced by its executed notebook (:func:`assert_figures_reproduced`)."""
    expected, result = run_case(case)
    assert result.report is not None and result.notebook is not None, (expected["verdict"], result.verdict, result.reason)
    assert_figures_reproduced(result.report, execute_notebook(case, result.notebook), case_id(case))
