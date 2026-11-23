"""T16.12: the editions matrix. ``dataset_catalog`` is on in every edition and
build; ``shared_catalog`` only in Enterprise with the commercial build; the
EDITIONS doc rows match the flags. Bite: grant shared_catalog to Professional."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.editions import EDITIONS, GATE_REASON, GateReason, explain, resolve_capabilities

DOC = Path(__file__).resolve().parents[2] / "docs" / "EDITIONS.md"


def caps(deployment, edition, *, commercial):
    return resolve_capabilities(deployment, edition, has_commercial_build=commercial)


@pytest.mark.parametrize("edition", EDITIONS)
@pytest.mark.parametrize("deployment", ["self_hosted", "avaloka_hosted"])
@pytest.mark.parametrize("commercial", [False, True])
def test_the_personal_catalog_is_open_everywhere(deployment, edition, commercial):
    assert GATE_REASON["dataset_catalog"] is GateReason.OPEN
    assert caps(deployment, edition, commercial=commercial).dataset_catalog is True


def test_the_shared_catalog_is_enterprise_only_and_needs_the_commercial_build():
    assert GATE_REASON["shared_catalog"] is GateReason.COMMERCIAL
    got = {e: caps("avaloka_hosted", e, commercial=True).shared_catalog for e in EDITIONS}
    assert got == {"oss": False, "free": False, "professional": False, "enterprise": True}
    assert caps("avaloka_hosted", "enterprise", commercial=False).shared_catalog is False
    assert caps("self_hosted", "enterprise", commercial=True).shared_catalog is True
    assert caps("self_hosted", "oss", commercial=True).shared_catalog is False
    assert explain("shared_catalog", caps("avaloka_hosted", "professional", commercial=True)) == \
        "shared_catalog is available in Avaloka Enterprise."


def _doc_row(flag: str):
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and f"`{flag}`" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return cells[1:5]
    return None


def test_the_editions_doc_rows_match_the_flags():
    """The doc is the user-facing contract; the flags are the code's."""
    marks = {True: "●", False: "○"}
    for flag in ("dataset_catalog", "shared_catalog"):
        row = _doc_row(flag)
        assert row is not None, f"docs/EDITIONS.md has no row for `{flag}`"
        expected = [marks[getattr(caps("avaloka_hosted", e, commercial=True), flag)]
                    for e in ("oss", "free", "professional", "enterprise")]
        assert row == expected, (flag, row, expected)
    text = DOC.read_text(encoding="utf-8")
    assert re.search(r"personal.*catalog", text, re.IGNORECASE | re.DOTALL)
