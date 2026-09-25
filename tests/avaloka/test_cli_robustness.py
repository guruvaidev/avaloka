"""What the CLI does when the user gets it wrong.

Every case here was reproduced against the CLI before being fixed. Two classes
of failure showed up, and the second is the dangerous one:

* input pandas rejects -> a raw traceback, with no statement of which input was
  at fault or what to do about it;
* input pandas *accepts* -> a complete mission, a validation verdict and an
  economic multiplier computed over a header with no rows, or over binary
  content parsed as one column of mojibake.

A tool that fabricates a confident analysis of nothing is worse than one that
crashes, so the loader now refuses and the CLI reports the refusal in a
sentence.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.io.loader import UnusableDataset, load_dataset

runner = CliRunner()


def flat(output: str) -> str:
    """CLI output with its line wrapping undone.

    Rich wraps prose to the terminal width, which is 80 columns under
    ``CliRunner`` and wider on most developer machines. That is the right
    behaviour for an error sentence -- the alternative is text running off the
    screen -- but it means a phrase can be split at any point, so an assertion
    that a sentence was printed must not depend on where the break landed.
    Paths and commands are a different matter: those must survive unwrapped,
    and are asserted against the raw output.
    """
    return " ".join(output.split())


@pytest.fixture
def good_csv(tmp_path):
    path = tmp_path / "churn.csv"
    pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(12)],
        "region": ["north", "south"] * 6,
        "monthly_spend": [29.0, 99.0] * 6,
        "tenure_months": list(range(1, 13)),
        "churned": [1, 0] * 6,
    }).to_csv(path, index=False)
    return path


# ── Datasets that parse but hold nothing to analyse ─────────────────────────

def test_header_with_no_rows_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n")

    with pytest.raises(UnusableDataset) as excinfo:
        load_dataset(str(path))

    message = str(excinfo.value)
    assert "no rows" in message
    assert "a, b" in message, "name the headers so the user can see what was read"
    assert len(message) < 400


@pytest.mark.parametrize("blob", [
    pytest.param(gzip.compress(b"customer,spend\n" * 200), id="gzipped"),
    pytest.param(bytes(range(128, 256)) * 8, id="high-byte-binary"),
])
def test_binary_content_named_csv_is_refused(tmp_path, blob):
    """The realistic mistake is a compressed or renamed file with a .csv name."""
    path = tmp_path / "junk.csv"
    path.write_bytes(blob)

    with pytest.raises(UnusableDataset) as excinfo:
        load_dataset(str(path))

    message = str(excinfo.value)
    assert "unreadable" in message, (
        "binary must be reported as unreadable, not as 'no rows' — the latter "
        "sends the user to their export settings instead of to the file type"
    )
    assert len(message) < 400, "an error that scrolls the terminal is not read"


def test_a_real_single_column_csv_is_not_mistaken_for_binary(tmp_path):
    """The binary check tests column *names*, not column count.

    A one-column CSV is perfectly ordinary and must still load.
    """
    path = tmp_path / "one.csv"
    path.write_text("customer_id\nC001\nC002\n")

    handle = load_dataset(str(path))

    assert handle.n_rows == 2
    assert list(handle.frame.columns) == ["customer_id"]


def test_a_good_dataset_still_loads(good_csv):
    handle = load_dataset(str(good_csv))
    assert handle.n_rows == 12
    assert handle.n_cols == 5


# ── Mistakes in the arguments ───────────────────────────────────────────────

def test_unknown_target_column_suggests_the_right_one(good_csv, tmp_path):
    """A typo used to surface as pandas' KeyError several agents deep."""
    result = runner.invoke(app, ["train", str(good_csv), "--target", "churnd",
                                 "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "no column named 'churnd'" in flat(result.output)
    assert "churned" in result.output, "must suggest the near-miss"
    assert "Traceback" not in result.output


def test_unknown_metric_is_rejected_before_any_work(good_csv, tmp_path):
    """An unknown metric produced a None score, which then crashed formatting.

    The crash surfaced in the persona's closing line, a long way from the
    argument that caused it.
    """
    result = runner.invoke(app, ["train", str(good_csv), "--target", "churned",
                                 "--metric", "rocauc", "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "roc_auc" in result.output, "must suggest the correct spelling"
    assert "Traceback" not in result.output


def test_missing_file_says_what_is_accepted(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.csv"), "--goal", "x",
                                 "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "can't reach that dataset" in flat(result.output)


def test_a_directory_is_not_a_dataset(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path), "--goal", "x",
                                 "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "directory, not a dataset" in flat(result.output)


def test_unwritable_output_fails_before_the_work(good_csv, tmp_path):
    """Checked up front: discovering it after the mission wastes the whole run."""
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory")

    result = runner.invoke(app, ["analyze", str(good_csv), "--goal", "x",
                                 "-o", str(blocker / "out")])

    assert result.exit_code == 1
    assert "can't write results" in flat(result.output)


# ── A successful run has to actually say something ──────────────────────────

def test_a_successful_analysis_reports_its_findings(good_csv, tmp_path):
    """The bundle always held the findings; the console did not.

    A run that prints an economic multiplier and a bare verdict, with no
    statement of what was found, has answered a question nobody asked.
    """
    out = tmp_path / "out"
    result = runner.invoke(app, ["analyze", str(good_csv), "--goal",
                                 "who is at risk of churning?", "-o", str(out), "--quiet"])

    assert result.exit_code == 0, result.output
    assert "What I found" in result.output
    assert "Deliverables" in result.output
    assert "executive_report.html" in result.output, "point at what to open next"


def test_a_warn_verdict_says_which_check_warned(good_csv, tmp_path):
    """"WARN" on its own is not actionable; the failing check has the reason."""
    out = tmp_path / "out"
    result = runner.invoke(app, ["analyze", str(good_csv), "--goal", "x",
                                 "-o", str(out), "--quiet"])

    assert "Validation verdict" in result.output
    verdict = (result.output.split("Validation verdict")[1]).splitlines()
    if "WARN" in verdict[0] or "FAIL" in verdict[0]:
        assert any(":" in line for line in verdict[1:4]), (
            "a non-passing verdict must name the check that produced it"
        )


# ── Help is the first thing a new user reads ────────────────────────────────

def test_help_leads_with_a_runnable_example():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "avaloka analyze" in result.output
    assert "Release 1" not in result.output, "internal release numbering is not user-facing"


def test_version_flag_exists():
    """`avaloka version` existed; `--version` is where people look."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip()
