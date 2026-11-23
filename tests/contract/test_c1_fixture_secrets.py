"""
Contract suite C1 -- plan suite E11.05 (repo hygiene), committed test fixtures.

``test_c1_repo_hygiene.py`` scans tracked source for credential shapes but skips
``tests/fixtures/`` and ``tests/data/`` wholesale. Those directories are where a
captured graph state, a recorded API response or a copied ``.env`` lands when a
fixture is built from a live run, so they get their own scan here. It is the
Python port of the gate's ``fixture_secret_scan.sh``: same file set, same four
rules, same output.

File set: every git-tracked file under ``tests/`` or ``ui/tests/`` that is
``.json``, ``.jsonl``, ``.yaml``, ``.yml``, ``.txt`` or ``.csv``, or that sits under
a ``fixtures/`` path. Only ``git ls-files`` output counts; an untracked scratch file
never fails the suite.

Rules, each line checked independently:

* ``dta_database_registry`` -- the DTA's registry of database credentials,
  which has no business in a fixture at any value;
* a ``password``/``passwd``/``pwd``/``db_password``/``pg_password`` JSON field with
  a value that is not a placeholder (empty, ``<...>``, ``***``, redacted, xxx,
  changeme, ``"test"``, dummy);
* a ``scheme://user:pass@`` connection string;
* a secret-like env name (``...API_KEY``, ``SECRET``, ``TOKEN``, ``PASSWORD``,
  ``PRIVATE_KEY``, ``ENCRYPTION_KEY``) assigned a value of 8+ characters that is
  not a placeholder or ``${...}``.

Secret hygiene of the suite itself: a finding is "<path>:<line>: <rule name>" and
nothing else. No assertion in this file compares an expression that holds a
matched value, so pytest's assertion rewriting has nothing secret to display.
The positive control's planted values are assembled at runtime and written only
into a temporary repository, so neither this scan nor C1's can match this file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The scanner -- rules and file selection match fixture_secret_scan.sh exactly.
# ---------------------------------------------------------------------------

FIXTURE_ROOTS = ("tests/", "ui/tests/")
FIXTURE_SUFFIX_OR_DIR = re.compile(r"\.(json|jsonl|yaml|yml|txt|csv)$|/fixtures/")

FIXTURE_SECRET_RULES: Tuple[Tuple[str, re.Pattern], ...] = (
    ("dta_database_registry", re.compile(r"dta_database_registry")),
    (
        "password field with a value",
        re.compile(
            r'"(?:password|passwd|pwd|db_password|pg_password)"\s*:\s*"'
            r'(?!\s*"|<|\*{3}|redacted|REDACTED|xxx|changeme|test"|dummy)[^"]{1,}"',
            re.I,
        ),
    ),
    (
        "connection string with credentials",
        re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@\"']+:[^\s@\"'/]+@", re.I),
    ),
    (
        "secret-like env name with a value",
        re.compile(
            r"\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE_KEY|ENCRYPTION_KEY)\b"
            r"\s*[\"']?\s*[:=]\s*[\"']?"
            r"(?!<|\$\{|redacted|REDACTED|xxx|dummy|changeme|test\b)[A-Za-z0-9_\-+/=.]{8,}"
        ),
    ),
)

#: rc16 tracks 48 fixture files. A scan over far fewer means the selection broke
#: (wrong cwd, ls-files failed silently), and a clean result would mean nothing.
MIN_FIXTURE_FILES = 20


def _git_ls_files(root: Path) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git is unusable in this environment: {type(exc).__name__}")
    if proc.returncode != 0:
        pytest.skip(f"{root} is not a git work tree; the fixture scan needs git ls-files")
    return [p for p in proc.stdout.split("\0") if p]


def _fixture_paths(root: Path) -> List[str]:
    return sorted(
        {
            p
            for p in _git_ls_files(root)
            if p.startswith(FIXTURE_ROOTS) and FIXTURE_SUFFIX_OR_DIR.search(p)
        }
    )


def _scan_fixtures(root: Path) -> Tuple[int, List[str]]:
    """Return (files scanned, findings). A finding is "<path>:<line>: <rule>"."""
    paths = _fixture_paths(root)
    findings: List[str] = []
    for relpath in paths:
        try:
            text = (root / relpath).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in FIXTURE_SECRET_RULES:
                if pattern.search(line):
                    findings.append(f"{relpath}:{lineno}: {name}")
    return len(paths), findings


def _report(scanned: int, findings: Sequence[str]) -> str:
    lines = [f"[fixture-scan] {f}" for f in findings]
    lines.append(f"[fixture-scan] {scanned} files scanned, {len(findings)} hit(s)")
    return "\n".join(lines)


def _rule_of(finding: str) -> str:
    return finding.rsplit(": ", 1)[1]


def _temp_repo(root: Path, files: Sequence[Tuple[str, str]]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=60)
    for relpath, body in files:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)
    return root


# ---------------------------------------------------------------------------
# E11.05 -- committed fixtures carry no credential
# ---------------------------------------------------------------------------


def test_e11_05_tracked_fixtures_carry_no_credentials() -> None:
    """Every tracked data fixture under tests/ and ui/tests/ is free of credentials."""
    scanned, findings = _scan_fixtures(REPO_ROOT)
    print(_report(scanned, findings))
    assert scanned >= MIN_FIXTURE_FILES, (
        f"only {scanned} fixture files selected; the scan's file selection is broken"
    )
    assert not findings, (
        "committed fixtures carry credentials (file:line: rule; values withheld). "
        "Scrub or replace with a placeholder:\n" + _report(scanned, findings)
    )


def test_e11_05_fixture_scanner_bites(tmp_path: Path) -> None:
    """One planted value per rule is found, by exactly that rule, and never echoed."""
    # Assembled at runtime so no committed file -- this one included -- holds them.
    pw_value = "ZZ-SYNTH-" + "PW-1"
    uri_password = "ZZ" + "pw"
    env_value = "ZZsynth" + "12345"
    registry_value = "ZZ-SYNTH-" + "REGISTRY"
    planted_values = (pw_value, uri_password, env_value, registry_value)

    repo = _temp_repo(
        tmp_path / "planted",
        [
            ("tests/fixtures/state.json",
             '{"dta_database_registry": {"warehouse": "' + registry_value + '"}}\n'),
            ("tests/fixtures/conn.json", '{"host": "db", "password": "' + pw_value + '"}\n'),
            ("tests/data/dsn.txt", "postgresql://zz:" + uri_password + "@db/x\n"),
            ("ui/tests/env.yaml", "GROQ_API_KEY=" + env_value + "\n"),
        ],
    )
    scanned, findings = _scan_fixtures(repo)
    report = _report(scanned, findings)

    # The echo check runs first: once it holds, nothing compared below can carry a
    # planted value into an assertion message.
    echoed = [i for i, value in enumerate(planted_values) if value in report]
    assert not echoed, f"scanner output echoes planted value(s) number {echoed}"
    assert scanned == 4
    rules = sorted(_rule_of(f) for f in findings)
    assert rules == sorted(name for name, _ in FIXTURE_SECRET_RULES)


def test_e11_05_fixture_scanner_accepts_placeholders(tmp_path: Path) -> None:
    """Placeholder values in every rule's field shape produce no finding."""
    password_lines = [
        '{"password": ""}',
        '{"password": "<your-password>"}',
        '{"passwd": "***"}',
        '{"pwd": "redacted"}',
        '{"db_password": "REDACTED"}',
        '{"pg_password": "xxx"}',
        '{"password": "changeme"}',
        '{"password": "test"}',
        '{"password": "dummy"}',
    ]
    env_lines = [
        "GROQ_API_KEY=",
        "GROQ_API_KEY=<groq-key>",
        "OPENAI_API_KEY=${OPENAI_API_KEY}",
        "SUPABASE_SECRET=redacted-value",
        "ACCESS_TOKEN: xxxxxxxxxx",
        "DB_PASSWORD=changeme123",
        "PRIVATE_KEY=dummy-key-value",
        "ENCRYPTION_KEY=test",
        "API_KEY=short",
    ]
    uri_lines = [
        "postgresql://db.internal:5432/app",
        "postgresql://reader@db/app",
        "https://example.com/path",
    ]
    repo = _temp_repo(
        tmp_path / "placeholders",
        [
            ("tests/fixtures/passwords.jsonl", "\n".join(password_lines) + "\n"),
            ("tests/fixtures/env.txt", "\n".join(env_lines) + "\n"),
            ("tests/data/uris.csv", "\n".join(uri_lines) + "\n"),
        ],
    )
    scanned, findings = _scan_fixtures(repo)

    assert scanned == 3
    assert not findings, "placeholders were flagged:\n" + _report(scanned, findings)
