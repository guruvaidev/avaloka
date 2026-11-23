"""The Billing tab's plan matrix is generated from app/core/editions.py.

ui/src/lib/edition-matrix.gen.ts is committed so the UI never imports Python.
These tests regenerate it and fail on any difference, then prove the comparison
can fail at all by changing a real copy of editions.py. Pure file and subprocess
checks: no app import, no network.
"""

from __future__ import annotations

import dataclasses
import difflib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gen_edition_matrix.py"
_spec = importlib.util.spec_from_file_location("gen_edition_matrix", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _changed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith(("- ", "+ "))
    ]


def _flip_in_source(tmp_path: Path, plan: str, capability: str) -> Path:
    """A copy of the real editions.py with one granted flag turned off."""
    source = gen.EDITIONS_PY.read_text(encoding="utf-8")
    head, marker, tail = source.partition(f'    "{plan}": Capabilities(\n')
    assert marker, f"no {plan} block in editions.py"
    grant = f"        {capability}=True,\n"
    assert grant in tail.split("    ),\n", 1)[0], f"{plan} does not grant {capability}"
    copy = tmp_path / "editions.py"
    copy.write_text(
        head + marker + tail.replace(grant, f"        {capability}=False,\n", 1),
        encoding="utf-8",
    )
    return copy


def _check(editions: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--editions", str(editions)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_committed_file_is_what_the_generator_writes():
    assert gen.OUTPUT.read_text(encoding="utf-8") == gen.render(gen.load_editions()), (
        "ui/src/lib/edition-matrix.gen.ts is stale: run python scripts/gen_edition_matrix.py"
    )


def test_check_passes_on_the_real_editions_file():
    result = _check(gen.EDITIONS_PY)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "plan,capability",
    [
        ("professional", "team_collaboration"),
        ("professional", "share_analysis"),
        ("enterprise", "org_structure"),
        ("free", "data_analysis"),
    ],
)
def test_changing_a_flag_in_editions_makes_the_committed_file_stale(tmp_path, plan, capability):
    flipped = _flip_in_source(tmp_path, plan, capability)

    assert _changed_lines(gen.render(gen.load_editions()), gen.render(gen.load_editions(flipped))) == [
        f"-     {capability}: true,",
        f"+     {capability}: false,",
    ]
    result = _check(flipped)
    assert result.returncode == 1
    assert "is stale" in result.stderr


def test_changing_a_status_in_editions_makes_the_committed_file_stale(tmp_path):
    source = gen.EDITIONS_PY.read_text(encoding="utf-8")
    entry = '    "sso_audit": "coming",\n'
    assert source.count(entry) == 1
    copy = tmp_path / "editions.py"
    copy.write_text(source.replace(entry, '    "sso_audit": "shipped",\n'), encoding="utf-8")

    assert _changed_lines(gen.render(gen.load_editions()), gen.render(gen.load_editions(copy))) == [
        '-   sso_audit: "coming",',
        '+   sso_audit: "shipped",',
    ]
    assert _check(copy).returncode == 1


def test_adding_a_capability_makes_the_committed_file_stale(tmp_path):
    source = gen.EDITIONS_PY.read_text(encoding="utf-8")
    field = "    sso_audit: bool = False\n"
    status = '    "sso_audit": "coming",\n'
    assert field in source and status in source
    copy = tmp_path / "editions.py"
    copy.write_text(
        source.replace(field, field + "    brand_new: bool = False\n").replace(
            status, status + '    "brand_new": "shipped",\n'
        ),
        encoding="utf-8",
    )

    assert "brand_new" in gen.render(gen.load_editions(copy))
    assert _check(copy).returncode == 1


def test_every_plan_and_every_capability_reach_the_output():
    """Each cell of the matrix, and each status, moves the output, so none is dropped."""
    editions = gen.load_editions()
    baseline = gen.render(editions)
    for plan in gen.PLANS:
        for field in dataclasses.fields(editions.Capabilities):
            granted = editions._HOSTED[plan]
            editions._HOSTED[plan] = dataclasses.replace(
                granted, **{field.name: not getattr(granted, field.name)}
            )
            assert gen.render(editions) != baseline, f"{plan}.{field.name} is not in the output"
            editions._HOSTED[plan] = granted
    for name, status in list(editions.CAPABILITY_STATUS.items()):
        editions.CAPABILITY_STATUS[name] = "shipped" if status == "coming" else "coming"
        assert gen.render(editions) != baseline, f"status of {name} is not in the output"
        editions.CAPABILITY_STATUS[name] = status
    assert gen.render(editions) == baseline
