"""Suite C8 (internal names) -- the public tree never names the internal hosting.

Two published docs named the internal repository (its host and workspace) in
prose, and nothing checked for it: the boundary is path-based, so a file that
publishes carries whatever it says. ``assert_no_words`` in oss/manifest.yaml is
the content check: case-insensitive patterns that must match no published
text file, with exact allowed phrases that each carry a reason.

The behavioural cases run the real generator against a throwaway repository,
as the rest of C8 does; nothing touches a remote or this checkout's refs.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.contract.test_c8_public_tree_generator import (REPO_ROOT, _clean_env, _git,
                                                          _require, _run_generator,
                                                          _script_text, _write)

MANIFEST = REPO_ROOT / "oss" / "manifest.yaml"
#: The words, assembled so this file does not need allowing.
HOST, WORKSPACE = "bit" + "bucket", "kamali" + "incteam"


def _section() -> str:
    m = re.search(r"^assert_no_words:\n((?:(?![a-z_]+:).*\n)*)", MANIFEST.read_text(), re.M)
    assert m, "oss/manifest.yaml has no assert_no_words section; the generator would skip the check"
    return m.group(1)


def test_c8_20_manifest_forbids_both_internal_names():
    body = _section()
    pats = re.findall(r'^    - "([^"]+)"', body.split("  allow:")[0], re.M)
    assert pats, "assert_no_words declares no patterns"
    for word in (HOST, WORKSPACE, HOST.upper(), WORKSPACE.capitalize()):
        assert any(re.search(p, word, re.I) for p in pats), f"no pattern matches {word!r}"


def test_c8_21_every_allowed_phrase_has_a_reason():
    body = _section()
    phrases = re.findall(r'^    - phrase: "([^"]+)"\n      reason: (.+)$', body, re.M)
    declared = re.findall(r"^    - phrase:", body, re.M)
    assert len(phrases) == len(declared), "an allowed phrase has no reason line"
    assert all(len(reason.split()) >= 6 for _, reason in phrases)
    assert not any(WORKSPACE in p.lower() for p, _ in phrases), "the workspace is never allowed"


_FIXTURE_MANIFEST = f"""\
exclude:
  - path: withheld/
    reason: Not part of the public tree.
assert_no_secrets:
  min_tail: 20
  patterns:
    - "ghp_"
assert_no_words:
  patterns:
    - "b[i]tbucket"
    - "kamal[i]incteam"
  allow:
    - phrase: "{HOST}-pipelines.yml"
      reason: the name of a withheld CI file, which the generator must name to withhold it
assert_unreachable:
  - .env
"""


@pytest.fixture()
def repo(tmp_path: Path):
    _require("git"), _require("bash"), _require("python3")
    env = _clean_env(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    _git(src, env, "init", "-q", "-b", "work")
    _write(src, "README.md", "# Demo\n\n## Quickstart\n\nsteps\n\n## Next\n\nmore\n")
    _write(src, "oss/manifest.yaml", _FIXTURE_MANIFEST)
    _write(src, "oss/overlay/README.quickstart.md", "## Quickstart\n\npublic quickstart\n")
    _write(src, "docs/ci.md", f"CI lives in {HOST}-pipelines.yml, which is withheld.\n")
    _write(src, f"withheld/{HOST}.md", f"internal: git@{HOST}.org:{WORKSPACE}/x.git\n")
    _write(src, "scripts/generate-oss.sh", _script_text())
    # The address invariant the script also runs (oss/manifest.yaml assert_no_addresses).
    _write(src, "oss/address_scan.py", (REPO_ROOT / "oss" / "address_scan.py").read_text())

    def commit(extra: dict | None = None):
        for rel, text in (extra or {}).items():
            _write(src, rel, text)
        _git(src, env, "add", "-A")
        _git(src, env, "commit", "-q", "--allow-empty", "-m", "source")
        _git(src, env, "update-ref", "refs/remotes/origin/develop-1.6", "HEAD")
        return _run_generator(src, env)
    return commit


def test_c8_22_clean_tree_passes_and_says_what_it_scanned(repo):
    run = repo()
    assert run.returncode == 0, run.stdout + run.stderr
    assert "== internal names ==" in run.stdout
    assert "ok   no file matches /b[i]tbucket/i" in run.stdout
    assert "ok   no file matches /kamal[i]incteam/i" in run.stdout
    assert "1 allowed phrase(s)" in run.stdout


@pytest.mark.parametrize("planted", [
    f"PR numbers are {HOST.capitalize()} (`{WORKSPACE}/avaloka-dev`).",
    f"origin is git@{HOST}.org:x/y.git",
    f"the {WORKSPACE.upper()} workspace",
    f"home: https://{HOST.upper()}.ORG/x",
    f"see {HOST}-pipelines.yaml",      # near an allowed phrase is not the phrase
])
def test_c8_23_a_planted_name_fails_the_generation(repo, planted):
    run = repo({"docs/releases/notes.md": f"Release notes.\n{planted}\n"})
    assert run.returncode != 0, run.stdout
    assert "internal name(s) in the public tree -- refusing to write" in run.stdout + run.stderr
    assert "docs/releases/notes.md:2" in run.stdout


def test_c8_24_a_manifest_without_patterns_fails_rather_than_passing_vacuously(repo, tmp_path):
    manifest = tmp_path / "src" / "oss" / "manifest.yaml"
    text = manifest.read_text()
    for line in ('    - "b[i]tbucket"\n', '    - "kamal[i]incteam"\n'):
        assert line in text
        text = text.replace(line, "")
    manifest.write_text(text)
    run = repo()
    assert run.returncode != 0
    assert "declares no patterns" in run.stdout + run.stderr
