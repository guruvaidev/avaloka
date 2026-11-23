"""Guard: the published 1.6 traceability matrix stays public-safe.

docs/test-reports/1.6-traceability.csv is generated from internal bug sheets that
carry assignee and reporter names. The public tree may only carry IDs,
descriptions, expected behaviour, mappings, test names and references. The OSS
manifest scans for credential shapes, not for personal data, so this test is
the only check on that file.

The name list is kept as salted SHA-256 hashes of lower-cased name tokens (single
words of 4+ letters and full names), so the names themselves are not committed.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pytest

CSV_PATH = Path(__file__).resolve().parents[2] / "docs" / "test-reports" / "1.6-traceability.csv"

ALLOWED_COLUMNS = [
    "source",
    "id",
    "description",
    "expected_behaviour",
    "recorded_status",
    "mapping",
    "framework",
    "xfail_strict",
    "test",
    "blocker",
    "defect_ref",
    "inventory_ref",
    "duplicate_of",
    "status",
    "note",
]

# Columns from the source sheets that must never be published, whatever they
# are renamed to. Checked case-insensitively against the header.
FORBIDDEN_HEADER_WORDS = ("owner", "assignee", "assigned", "reporter", "identified", "tester", "email", "executed")

SALT = "avaloka-traceability:"
NAME_TOKEN_HASHES = frozenset({
    "0d7f107ff9c2191c72af16b9b010a8b92fae0e293ac85d3868e9e7cdd8904a39",
    "1d1b5de2931280289d246700d68a9ecbafa59e0c8ed15b5431e26dbaf1cbcd94",
    "1d46e6084cbb8d452c534a1568c94a387a415e324d11e066e8b0e4f9a5397c0b",
    "1f8cba81912316f31612d03a525ba9787d9248b14926bbc9cc4dbb65ab4ebdaf",
    "1fd9897b9a3f89b68aadc172cb74166649266fe9afd71d1a3063f1d6822c04b3",
    "33cfe180f3f52dba2f5cad863b34f3d3794641505a6bd8f4350ce55738488e6e",
    "385e6e4760770dd4404ee0c172d40197efacbab9faf665d87d48ff726f90c78a",
    "386d37a3d95ed290d37f9a219ad21541e2bc557a1f084fcf853675d1fb3f5e6b",
    "44cabd4880614290878087bc7de25f820b369efbfafe7109e9c2bcf1653ee6bb",
    "51e7e96a5676186e3852254ef72d908dec68665abfe92db667261f8954e9e224",
    "561cc5f4ea97988e371776fa4daab2c596bf920f8ccb22bb97be8b67b694ffb1",
    "570058b66a6732f035cbe65d87f403710f4d570fe048ddcb5c4eeb323f690d95",
    "605ec248d33b66c05c5e5bc635cf96e1f12e88becc3b705bbefbb859807c458d",
    "6c1368a093aaa49246086ddf535fcad21dfd96660f2647492c96c94db9b56fc0",
    "701232af6096729b69f0d0c9c7dc2faad615f545aa9f621bb014ce69bbf28e33",
    "71adba1dc70a5ef9a9f240657d0470645f3e6107577c3eebb176c0d774ab1fdf",
    "81c4734a9f95a76c170202c5ec550cc94b81a0dea216dc02e2c8705fb94d6fbb",
    "87e9729387e5d7aaffbc97c0ed8004e4910a63848dba49508e45834b82e73529",
    "952c3304e4ffaff1a3d0b01da621749e70c96ccc1101bb958e48fc01876f3de3",
    "9ea4bb05d43a726d5c2fb9c9d9757bca5c2f3ad75d464dd5f34f89fb00915537",
    "b154d9748d208e5d9a107507d980a34bf0fd1ff7ba169ef4b3d15916d31f19a4",
    "b9f8300bc5a545ac852751ea45ed0f4c7818a531213c1ebe867fdcdd0a24fe22",
    "c63215365c146030a8330e90fc1dc8d55d3bf1fa15bdc8110fa3aeda9b281797",
    "c6496d33ad9252331ef888b593ea74ccbf36baf3fa098ae9142920cedf565744",
    "ccbf88a6cb9406391f6f641939a366cdef8a69fe0ef414613ebc5204f2b800d5",
    "d3081cf5fc4d47930c44392bf0b6af23a881ae8c72221b38c164501b3b0a5714",
    "d51d263c0ae1584c5c66c41d7e02f206c66a0bc410ae853bb3802ca9df3f5ec1",
    "e0a25296273662922ea9924b0e06d82a1305d6c891dadff0446251afca9bea6f",
    "ecb962a99d5c027b5944f345fa8517074aeacb320a07ad1cf7f277e734b768d0",
    "ee34f9334b58cec0c4e9f7bf9f0000c9459f80b24f9fb46a7491170223413e1c",
    "f6bf74c55a729095a942bd9c940d0da33f7d87fff31c12d9392279bb79970a24",
    "f9fa5902fffd7d9ab55893f565a3921cdfd6791ca70747c93dac1ed8c041a43a",
    "fa8a75367a0896d68430f8cfd22c337fcbd7f2b083ff8ee08ea45f779a47e1c5",
})

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?i)\b(?:https?|ftp)://|\bwww\.|docs\.google\.|\bapp\.avaloka\b|bitbucket\.org/|\.internal\b")
SESSION = re.compile(r"(?i)\bsession=\w+|\bai-playground-\d+")


def _hash(token: str) -> str:
    return hashlib.sha256((SALT + token).encode()).hexdigest()


def _rows() -> tuple[list[str], list[dict[str, str]]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def find_name_tokens(text: str) -> list[str]:
    """Return the tokens (and adjacent word pairs) of ``text`` whose hash is listed."""
    words = re.findall(r"[a-z][a-z'\-]+", text.lower())
    hits = [w for w in words if len(w) >= 4 and _hash(w) in NAME_TOKEN_HASHES]
    hits += [f"{a} {b}" for a, b in zip(words, words[1:]) if _hash(f"{a} {b}") in NAME_TOKEN_HASHES]
    return hits


def find_contact_or_url(text: str) -> list[str]:
    hits = [m for m in EMAIL.findall(text) if not m.lower().endswith(("example.com", "example.org"))]
    hits += URL.findall(text) + SESSION.findall(text)
    return hits


def test_traceability_matrix_exists_and_is_not_empty():
    """The matrix is published and covers every source sheet (plan, Bugs-1.6, -New, UI) plus findings made while writing the suite."""
    _, rows = _rows()
    assert len(rows) >= 200
    assert {r["source"] for r in rows} == {"Master Test Plan", "Bugs-1.6", "Bugs-1.6-New", "UI Bugs", "Release suite finding"}


def test_traceability_matrix_has_only_allowed_columns():
    """Only whitelisted columns; no owner/assignee/reporter/tester/email column under any name."""
    header, _ = _rows()
    assert header == ALLOWED_COLUMNS
    for col in header:
        assert not any(w in col.lower() for w in FORBIDDEN_HEADER_WORDS), col


def test_traceability_matrix_has_no_emails_urls_or_session_ids():
    """No email address, URL, internal hostname or session/project identifier in any cell."""
    _, rows = _rows()
    offending = [(r["id"], h) for r in rows for v in r.values() for h in find_contact_or_url(v or "")]
    assert offending == []


def test_traceability_matrix_has_no_listed_person_names():
    """No cell contains a name token from the source sheets' people columns (checked by hash)."""
    _, rows = _rows()
    offending = [(r["id"], len(h)) for r in rows for v in r.values() for h in find_name_tokens(v or "")]
    assert offending == []


@pytest.mark.parametrize(
    "text, expect_hit",
    [
        ("reach me at someone@company.io", True),
        ("see https://docs.google.com/x", True),
        ("example fixture user@example.com", False),
        ("plain description of a bug", False),
    ],
)
def test_contact_and_url_detector_bites(text, expect_hit):
    """The detectors themselves must fire on a planted email/URL and stay quiet on clean text."""
    assert bool(find_contact_or_url(text)) is expect_hit


def test_name_detector_bites_on_a_listed_token():
    """A token whose salted hash is listed is reported (proves the hash check is wired)."""
    import tests.release_1_6.test_traceability_public_safe as mod

    original = mod.NAME_TOKEN_HASHES
    try:
        mod.NAME_TOKEN_HASHES = frozenset(original | {_hash("planted")})
        assert find_name_tokens("assigned to Planted today") == ["planted"]
    finally:
        mod.NAME_TOKEN_HASHES = original
