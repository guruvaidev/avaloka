"""Suite C8 -- the public-tree generator has two modes and no intermediate branch.

scripts/generate-oss.sh used to have a third hop: ``--write`` committed the
generated tree onto a branch (``oss/1.6``) that sat between the source branch
and the public repository. Publishing became a squash onto the public tip, and
after that nothing read the intermediate branch. It went on existing, fell
behind what was actually published, and still looked like a branch one could
target -- a second copy of the public tree, and the wrong one.

These cases keep it retired. The static ones read the script; the behavioural
ones run the real script against a throwaway repository built in ``tmp_path``,
so nothing here touches a remote, the network, or this checkout's refs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate-oss.sh"
BRANCH_MODEL_DOC = REPO_ROOT / "docs" / "oss-branch-model.md"

#: A shell default that points a variable at an ``oss/...`` branch, e.g.
#: ``${DST_BRANCH:-oss/1.6}``. The variable name is deliberately not fixed:
#: renaming it must not be a way past this check.
_OSS_BRANCH_DEFAULT_RE = re.compile(r"\$\{\w+:[-=]\s*[\"']?oss[/-]")

#: The retired knobs. Their presence means the intermediate branch is back.
_RETIRED_NAMES = ("DST_BRANCH", "GEN_BRANCH", "oss-sync/generated", "oss/1.6")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        # Same rule as the rendered-chart cases in test_c2: a skip reads as a
        # pass in a summary line, which is tolerable on a laptop and not on the
        # run that gates a merge.
        message = f"{tool} is not installed; the generator's behaviour was not exercised"
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
        pytest.skip(message)
    return path


def _clean_env(home: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The caller's environment minus anything that would aim git or the script elsewhere."""
    drop = ("SRC_BRANCH", "REMOTE", "PUBLISH_REMOTE", "PUBLISH_BRANCH", "PUBLISH_BASE",
            "PUBLISH_CONFIRM", "DST_BRANCH", "GEN_BRANCH")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in drop}
    env.update({
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    })
    env.update(extra or {})
    return env


def _git(cwd: Path, env: Dict[str, str], *args: str) -> str:
    run = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert run.returncode == 0, f"git {' '.join(args)} failed: {run.stderr}"
    return run.stdout


_MANIFEST = """\
exclude:
  - path: withheld/
    reason: Not part of the public tree.
overlay:
  - DCO
splice:
  - file: README.md
    section: "## Quickstart"
    from: oss/overlay/README.quickstart.md
assert_present:
  README.md:
    - "public quickstart"
assert_no_secrets:
  min_tail: 20
  prefixes:
    - "ghp_"
assert_unreachable:
  - .env
"""


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def source_repo(tmp_path: Path):
    """A repository shaped like the source branch, carrying the real script.

    ``origin/develop-1.6`` is a local ref, not a remote: the script only ever
    archives it. No ``github`` remote is configured; a case that wants the
    published tree adds one pointing at another directory under ``tmp_path``.
    """
    _require("git")
    _require("bash")
    _require("python3")
    env = _clean_env(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    _git(src, env, "init", "-q", "-b", "work")
    _write(src, "README.md", "# Demo\n\n## Quickstart\n\ninternal steps\n\n## Next\n\nmore\n")
    _write(src, "oss/manifest.yaml", _MANIFEST)
    _write(src, "oss/overlay/README.quickstart.md", "## Quickstart\n\npublic quickstart\n")
    _write(src, "oss/overlay/DCO", "sign your work\n")
    _write(src, "withheld/notes.md", "internal\n")
    _write(src, "unchanged.txt", "same on both sides\n")
    _write(src, "new_module.py", "x = 1\n")
    # Ignored, and tracked only because it was force-added -- the shape of the
    # sample datasets in the real tree.
    _write(src, ".gitignore", "*.csv\n")
    _write(src, "sample.csv", "a,b\n1,2\n")
    _write(src, "scripts/generate-oss.sh", _script_text())
    # The address invariant the script runs (oss/manifest.yaml assert_no_addresses).
    _write(src, "oss/address_scan.py", (REPO_ROOT / "oss" / "address_scan.py").read_text())
    _git(src, env, "add", "-A")
    _git(src, env, "add", "-f", "sample.csv")
    _git(src, env, "commit", "-q", "-m", "source")
    _git(src, env, "update-ref", "refs/remotes/origin/develop-1.6", "HEAD")
    return src, env


def _run_generator(src: Path, env: Dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/generate-oss.sh", *args], cwd=src, env=env,
                          capture_output=True, text=True, timeout=120)


# ---------------------------------------------------------------------------
# Static: the script no longer knows about an intermediate branch
# ---------------------------------------------------------------------------

def test_c8_01_no_default_points_at_an_oss_branch() -> None:
    """No variable in the generator defaults to an ``oss/...`` branch."""
    text = _script_text()
    defaults = _OSS_BRANCH_DEFAULT_RE.findall(text)
    assert not defaults, (
        f"scripts/generate-oss.sh defaults a variable to an oss/ branch again: {defaults}. "
        "The public repository's own main is the only published copy; compare and publish "
        "against $PUBLISH_REMOTE/$PUBLISH_BASE.")
    present = [name for name in _RETIRED_NAMES if name in text]
    assert not present, (
        f"scripts/generate-oss.sh mentions retired intermediate-branch names: {present}")


def test_c8_02_dry_run_and_publish_share_one_base() -> None:
    """The dry run compares against the base ``--publish`` builds on, and says when it cannot."""
    text = _script_text()
    assert 'echo "== compare vs $PUBLISH_REMOTE/$PUBLISH_BASE' in text, (
        "the dry run no longer names $PUBLISH_REMOTE/$PUBLISH_BASE as what it compares against")
    assert 'echo "== publish (squash onto $PUBLISH_REMOTE/$PUBLISH_BASE' in text, (
        "--publish no longer builds on $PUBLISH_REMOTE/$PUBLISH_BASE")
    assert text.count('git fetch -q "$PUBLISH_REMOTE" "$PUBLISH_BASE"') == 2, (
        "both modes must fetch the published base before reading it; a stale "
        "remote-tracking ref is how a comparison goes quietly wrong")


def test_c8_03_force_add_survives_in_publish() -> None:
    """``--publish`` adds with ``-f``; without it every force-added file is dropped."""
    text = _script_text()
    assert 'git -C "$work" add -A -f' in text, (
        "--publish must `git add -A -f`: .gitignore lists *.csv and app/sample_data/, and "
        "those files are tracked only because they were force-added")


def test_c8_04_branch_model_doc_describes_two_hops() -> None:
    doc = BRANCH_MODEL_DOC.read_text(encoding="utf-8")
    assert not re.search(r"generate-oss\.sh\s+--write", doc), (
        "docs/oss-branch-model.md still tells the reader to run --write")
    flow = [line for line in doc.splitlines() if line.startswith("develop-1.6  ──")]
    assert flow, "docs/oss-branch-model.md lost its flow line"
    assert all("oss/1.6" not in line for line in flow), (
        f"the flow line still routes through the retired branch: {flow}")


# ---------------------------------------------------------------------------
# Behavioural: run the real script
# ---------------------------------------------------------------------------

def test_c8_05_write_is_refused_before_any_work(tmp_path: Path) -> None:
    """``--write`` exits non-zero with an explanation, and does nothing first.

    Run from a directory that is not a repository: the refusal has to come
    before the script looks for one, so an old habit cannot get as far as
    generating a tree, let alone committing it.
    """
    _require("bash")
    run = subprocess.run(["bash", str(SCRIPT), "--write"], cwd=tmp_path,
                         env=_clean_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert run.returncode == 2, (
        f"--write exited {run.returncode}, expected the refusal's 2.\n{run.stdout}\n{run.stderr}")
    assert "--write is retired" in run.stderr
    assert "--publish" in run.stderr, "the refusal must name the mode that replaces it"
    assert run.stdout == "", f"--write did work before refusing:\n{run.stdout}"


def test_c8_06_write_creates_no_branch(source_repo) -> None:
    """In a real repository ``--write`` leaves every ref exactly as it was."""
    src, env = source_repo
    before = _git(src, env, "for-each-ref")
    run = _run_generator(src, env, "--write")
    assert run.returncode == 2, f"{run.stdout}\n{run.stderr}"
    assert _git(src, env, "for-each-ref") == before, "--write moved or created a ref"


def test_c8_07_unknown_argument_is_refused(tmp_path: Path) -> None:
    """A typo must not fall through to a dry run that looks like it did what was asked."""
    _require("bash")
    run = subprocess.run(["bash", str(SCRIPT), "--pubish"], cwd=tmp_path,
                         env=_clean_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert run.returncode == 2
    assert "unknown argument" in run.stderr


def test_c8_08_dry_run_without_the_public_remote_skips_visibly(source_repo) -> None:
    """No public remote: the checks still run, the comparison is skipped out loud, exit 0."""
    src, env = source_repo
    run = _run_generator(src, env)
    out = run.stdout
    assert run.returncode == 0, f"dry run exited {run.returncode}\n{out}\n{run.stderr}"
    assert "== invariants ==" in out and "ok   README.md present: public quickstart" in out
    assert "== secret scan ==" in out and "ok   ghp_" in out
    assert "== compare vs github/main" in out
    assert "SKIPPED -- this checkout has no remote named 'github'" in out, out
    assert "invariants and secret scan above did run" in out
    # Nothing that could be mistaken for the result of a comparison.
    assert "public tree now:" not in out
    assert "no change" not in out
    assert not re.search(r"\d+ added, \d+ modified, \d+ removed", out)


def test_c8_09_dry_run_compares_against_the_published_tree(source_repo, tmp_path: Path) -> None:
    """With the public remote present, the dry run reports added, modified and removed paths."""
    src, env = source_repo
    pub = tmp_path / "published"
    pub.mkdir()
    _git(pub, env, "init", "-q", "-b", "main")
    _write(pub, "README.md", "# Demo\n\n## Quickstart\n\nan older public quickstart\n\n## Next\n\nmore\n")
    _write(pub, "unchanged.txt", "same on both sides\n")
    _write(pub, "retired.txt", "published once, gone from the source since\n")
    _git(pub, env, "add", "-A")
    _git(pub, env, "commit", "-q", "-m", "published")
    _git(src, env, "remote", "add", "github", str(pub))

    def local_refs() -> str:
        # The fetch updates refs/remotes/github/main, as any fetch does. That is
        # the comparison's input, not something the dry run produced.
        return _git(src, env, "for-each-ref", "refs/heads", "refs/tags", "refs/remotes/origin")

    refs_before = local_refs()

    run = _run_generator(src, env)
    out = run.stdout
    assert run.returncode == 0, f"dry run exited {run.returncode}\n{out}\n{run.stderr}"
    assert "SKIPPED" not in out
    changes = dict(reversed(m.groups()) for m in re.finditer(r"^   ([AMD])\t(.+)$", out, re.M))
    assert changes.get("README.md") == "M", out
    assert changes.get("retired.txt") == "D", out
    assert changes.get("new_module.py") == "A", out
    assert changes.get("DCO") == "A", "the overlay file is part of the generated tree"
    # Ignored but force-added on the source branch: it is in the tree --publish
    # would build, so the comparison has to see it too.
    assert changes.get("sample.csv") == "A", out
    assert "unchanged.txt" not in changes
    assert not any(path.startswith("withheld/") for path in changes), "an excluded path was generated"
    counts = re.search(r"(\d+) added, (\d+) modified, (\d+) removed", out)
    assert counts, out
    assert (int(counts.group(2)), int(counts.group(3))) == (1, 1), out
    assert int(counts.group(1)) == sum(1 for status in changes.values() if status == "A")
    # A dry run is a dry run.
    assert local_refs() == refs_before, "the dry run moved or created a branch"
    assert _git(src, env, "status", "--porcelain") == "", "the dry run dirtied the checkout"


def test_c8_10_dry_run_matches_what_publish_would_build(source_repo, tmp_path: Path) -> None:
    """The dry run's list is the diff between the published tip and the ``--publish`` tree."""
    src, env = source_repo
    pub = tmp_path / "published"
    pub.mkdir()
    _git(pub, env, "init", "-q", "-b", "main")
    _write(pub, "unchanged.txt", "same on both sides\n")
    _write(pub, "retired.txt", "gone\n")
    _git(pub, env, "add", "-A")
    _git(pub, env, "commit", "-q", "-m", "published")
    _git(src, env, "remote", "add", "github", str(pub))

    dry = _run_generator(src, env)
    assert dry.returncode == 0, f"{dry.stdout}\n{dry.stderr}"
    listed = sorted(f"{s}\t{p}" for s, p in re.findall(r"^   ([AMD])\t(.+)$", dry.stdout, re.M))

    publish = _run_generator(src, env, "--publish")
    assert publish.returncode == 0, f"{publish.stdout}\n{publish.stderr}"
    assert "dry run. Set PUBLISH_CONFIRM=yes to push:" in publish.stdout
    commit = re.search(r"git push github ([0-9a-f]{40}):refs/heads/main", publish.stdout).group(1)
    base = _git(src, env, "rev-parse", f"{commit}^").strip()
    real = sorted(_git(src, env, "diff", "--no-renames", "--name-status", base, commit).splitlines())
    assert listed == real
    # And without PUBLISH_CONFIRM nothing reached the published repository.
    assert _git(pub, env, "rev-parse", "main").strip() == base


# ---------------------------------------------------------------------------
# Splices and invariants: the public tree can differ from the source line by
# line, and every such difference is checked rather than assumed
# ---------------------------------------------------------------------------

_VERSIONED_MANIFEST = _MANIFEST.replace("""assert_present:
  README.md:
    - "public quickstart"
""", """  - file: pyproject.toml
    match: 'version = "[^"]*"'
    with: 'version = "1.0.0"'
  - file: CHANGES.md
    before: "## [Unreleased]"
    from: oss/overlay/CHANGES.release.md
  - file: CHANGES.md
    line: "Internal note about 1.6.0."
    with: "Public note."
assert_present:
  README.md:
    - "public quickstart"
  pyproject.toml:
    - 'version = "1.0.0"'
  CHANGES.md:
    - "## [1.0.0]"
assert_absent:
  pyproject.toml:
    - "version = \\"1.6.0\\""
  CHANGES.md:
    - "1.6.0"
""")


def _commit_source(src: Path, env: Dict[str, str], manifest: str) -> None:
    _write(src, "oss/manifest.yaml", manifest)
    _write(src, "oss/overlay/CHANGES.release.md", "## [1.0.0]\n\nFirst public release.\n")
    _write(src, "pyproject.toml", '[project]\nname = "demo"\nversion = "1.6.0"\n')
    _write(src, "CHANGES.md", "# Changes\n\n## [Unreleased]\n\nInternal note about 1.6.0.\n")
    _git(src, env, "add", "-A")
    _git(src, env, "commit", "-q", "-m", "versioned source")
    _git(src, env, "update-ref", "refs/remotes/origin/develop-1.6", "HEAD")


def test_c8_11_splices_set_the_public_version(source_repo, tmp_path: Path) -> None:
    """line, match and before splices apply, and quoted needles are unescaped.

    The absent needle is written with escaped quotes. Before the scalar was
    unescaped it searched for a backslash no file carries, so it could not fail.
    """
    src, env = source_repo
    _commit_source(src, env, _VERSIONED_MANIFEST)
    out_dir = tmp_path / "kept"
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(out_dir)})
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert 'ok   pyproject.toml absent: version = "1.6.0"' in run.stdout, run.stdout
    assert (out_dir / "pyproject.toml").read_text() == '[project]\nname = "demo"\nversion = "1.0.0"\n'
    changes = (out_dir / "CHANGES.md").read_text()
    assert "First public release.\n\n## [Unreleased]" in changes, changes
    assert "1.6.0" not in changes and "Public note." in changes


def test_c8_12_a_splice_that_misses_fails_the_generation(source_repo, tmp_path: Path) -> None:
    """An anchor that matches no line refuses, rather than publishing the source text."""
    src, env = source_repo
    _commit_source(src, env, _VERSIONED_MANIFEST.replace(
        'line: "Internal note about 1.6.0."', 'line: "A line that is not there."'))
    out_dir = tmp_path / "kept"
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(out_dir)})
    assert run.returncode != 0, run.stdout
    assert "FAIL CHANGES.md line: A line that is not there. -- anchor matches 0 lines" in run.stdout
    assert "1 splice(s) failed -- refusing to write" in run.stderr
    assert not out_dir.exists(), "a refused tree was kept"


def test_c8_13_removing_the_version_splice_trips_the_invariant(source_repo) -> None:
    """The version invariants bite: drop the splice and generation is refused."""
    src, env = source_repo
    _commit_source(src, env, _VERSIONED_MANIFEST.replace("""  - file: pyproject.toml
    match: 'version = "[^"]*"'
    with: 'version = "1.0.0"'
""", ""))
    run = _run_generator(src, env)
    assert run.returncode != 0, run.stdout
    assert 'FAIL pyproject.toml present: version = "1.0.0"  (found 0)' in run.stdout
    assert 'FAIL pyproject.toml absent: version = "1.6.0"  (found 1)' in run.stdout
    assert "2 invariant(s) failed -- refusing to write" in run.stderr


# ---------------------------------------------------------------------------
# Public text: a withheld replacement table, and pattern invariants
# ---------------------------------------------------------------------------

_TEXT_MANIFEST = _MANIFEST.replace("""assert_present:""", """replace_table: oss/internal/public-text.tsv
assert_present:""") + """assert_no_match:
  - glob: "**/*.md"
    pattern: '(?<![\\\\w&#/])#[1-9][0-9]{2,3}(?!\\\\w)'
    unless_after: "GitHub "
"""


def _commit_text_source(src: Path, env: Dict[str, str], table: str, notes: str) -> None:
    _write(src, "oss/manifest.yaml", _TEXT_MANIFEST)
    _write(src, "oss/internal/public-text.tsv", table)
    _write(src, "NOTES.md", notes)
    (src / "report.csv").write_bytes(b"id,note\r\n1,fixed by #663\r\n")
    _git(src, env, "add", "-A")
    _git(src, env, "add", "-f", "report.csv")   # *.csv is ignored, as in the real tree
    _git(src, env, "commit", "-q", "-m", "text source")
    _git(src, env, "update-ref", "refs/remotes/origin/develop-1.6", "HEAD")


def test_c8_14_replacement_table_applies_and_keeps_line_endings(source_repo, tmp_path: Path) -> None:
    """Rows apply exactly, a CRLF file stays CRLF, and a GitHub reference is allowed."""
    src, env = source_repo
    table = ("# comment\n"
             "NOTES.md\tFixed in #687.\tFixed in 1.0.0.\n"
             "report.csv\tfixed by #663\tfixed\n")
    _commit_text_source(src, env, table, "# Notes\n\nFixed in #687. See GitHub #482.\n")
    out_dir = tmp_path / "kept"
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(out_dir)})
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert (out_dir / "NOTES.md").read_text() == "# Notes\n\nFixed in 1.0.0. See GitHub #482.\n"
    assert (out_dir / "report.csv").read_bytes() == b"id,note\r\n1,fixed\r\n"
    assert "ok   no match for" in run.stdout, run.stdout


def test_c8_15_a_row_that_misses_fails_the_generation(source_repo, tmp_path: Path) -> None:
    """A row whose text is not there refuses, rather than publishing the old text."""
    src, env = source_repo
    _commit_text_source(src, env, "NOTES.md\tFixed in #999.\tFixed.\n", "# Notes\n\nFixed in #687.\n")
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(tmp_path / "kept")})
    assert run.returncode != 0, run.stdout
    assert "FAIL row 1: NOTES.md -- text occurs 0 times, expected 1" in run.stdout
    assert not (tmp_path / "kept").exists()


def test_c8_16_an_internal_change_number_trips_the_pattern_invariant(source_repo) -> None:
    """A three-digit change number left in Markdown fails; the GitHub form does not."""
    src, env = source_repo
    _commit_text_source(src, env, "# no rows\n", "# Notes\n\nFixed in #687. See GitHub #482.\n")
    run = _run_generator(src, env)
    assert run.returncode != 0, run.stdout
    assert "FAIL no match for" in run.stdout and "(1: NOTES.md:3)" in run.stdout, run.stdout
    assert "1 pattern invariant(s) failed -- refusing to write" in run.stderr


def test_c8_17_an_optional_row_may_be_absent_but_not_miscounted(source_repo, tmp_path: Path) -> None:
    """0|N: absent passes and is listed; present at any other count still refuses."""
    src, env = source_repo
    table = ("NOTES.md\tFixed in #687.\tFixed in 1.0.0.\n"
             "NOTES.md\tPending text #701.\tPending text.\t0|1\n")
    _commit_text_source(src, env, table, "# Notes\n\nFixed in #687.\n")
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(tmp_path / "absent")})
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert ". 1 optional row(s) not present: 2" in run.stdout, run.stdout

    _commit_text_source(src, env, table, "# Notes\n\nFixed in #687.\nPending text #701.\nPending text #701.\n")
    run = _run_generator(src, {**env, "OSS_OUT_DIR": str(tmp_path / "twice")})
    assert run.returncode != 0, run.stdout
    assert "FAIL row 2: NOTES.md -- text occurs 2 times, expected 0|1" in run.stdout, run.stdout
