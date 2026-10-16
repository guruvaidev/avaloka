"""Every place that installs pandas keeps it below 3.

1.6 was gated on pandas 2. On 3.0.6 the hermetic gate fails 13 more tests
across the platform -- the preparation agent, PII pseudonymisation, the null
policy, local inference -- because pandas 3 reads text as its own ``str``
dtype and much of the code asks ``dtype == object``. An uncapped ``>=2.0``
let a clean install (and the next image build) pick 3.0.6 with nothing
noticing, so the cap is checked here, in every requirements file that names
pandas, not only in pyproject.toml. Lift it when the gate is green on 3.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = re.compile(r"""^\s*"?pandas(\[[^\]]*\])?\s*(?P<spec>[<>=!~][^"#;\s]*)""", re.M)


#: Directories that hold other projects' requirements, not ours.
_NOT_OURS = {".git", "node_modules", ".venv", "venv", "site-packages", "__pycache__"}


def _declaring_files() -> list[Path]:
    candidates = [ROOT / "pyproject.toml"] + [
        p for p in ROOT.rglob("requirements*.txt")
        if not _NOT_OURS & set(p.relative_to(ROOT).parts)]
    return sorted(p for p in candidates
                  if p.is_file() and SPEC.search(p.read_text(errors="replace")))


def test_pandas_is_declared_somewhere():
    """Without this, finding no file would make the check below pass."""
    files = {p.relative_to(ROOT).as_posix() for p in _declaring_files()}
    assert {"pyproject.toml", "requirements.txt"} <= files, files


@pytest.mark.parametrize("path", _declaring_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_pandas_is_capped_below_3(path):
    for match in SPEC.finditer(path.read_text(errors="replace")):
        spec = match.group("spec")
        assert re.search(r"<\s*3(\.0)*(?![\d.])", spec), (
            f"{path.relative_to(ROOT)} declares pandas{spec}, which admits pandas 3")
