"""The swarm's edition limits, enforced where clones are actually dispatched.

The open-source and Free editions get a preview: two clones per mission, run
locally. These tests drive the real coordinator and the real CLI, because a
limit that is only documented, or only applied in one caller, is the failure
this file exists to catch.
"""

from __future__ import annotations

import threading
import time

import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.coordinator import MainAvaloka, ShardOutcome, load_coordination
from avaloka.io import load_dataset
from avaloka.mission.ledger import WorkUnit
from avaloka.ray_batch import _partition_stats
from avaloka.swarm_limits import (PREVIEW_MAX_CLONES, UNLIMITED, grant,
                                  limits_for, limits_from_env)


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


def _fake_run(record):
    """A stand-in sub-Avaloka that records how many run at once."""
    lock = threading.Lock()
    state = {"active": 0}

    def run(self, *, shard_id, shard, shard_dir, **kwargs):
        with lock:
            state["active"] += 1
            record.append(state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return (
            ShardOutcome(shard_id=shard_id, n_rows=shard.n_rows,
                         output_dir=str(shard_dir / "mission"), verdict="pass",
                         max_safe_deployment_level=1, manual_minutes=1, cost_usd=0,
                         sampling={"sampled": False, "n_working_rows": shard.n_rows,
                                   "strategy": "none"},
                         partition_stats=_partition_stats(load_dataset(str(shard.path)).frame)),
            [WorkUnit("data_scout", "Data analyst", "profile", manual_minutes=1)],
        )
    return run


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_open_source_and_free_get_the_preview_limit():
    for edition in ("oss", "community", "free", None, "nonsense"):
        limits = limits_for(edition, has_commercial_build=False)
        assert limits.max_clones_per_mission == PREVIEW_MAX_CLONES == 2, edition
        assert limits.max_parallel_clones == 2
        assert limits.execution == "local" and limits.cluster_fan_out is False
        assert limits.preview is True


def test_a_claimed_paid_edition_without_its_build_gets_the_preview():
    """Limits follow installed code, never configuration."""
    for edition in ("professional", "enterprise"):
        assert limits_for(edition, has_commercial_build=False).max_clones_per_mission == 2


def test_commercial_editions_raise_or_remove_the_limit():
    assert limits_for("professional", has_commercial_build=True).max_clones_per_mission == 16
    ent = limits_for("enterprise", has_commercial_build=True)
    assert ent.max_clones_per_mission is None and ent.max_parallel_clones is None
    # A commercial build still honours a Free plan's preview.
    assert limits_for("free", has_commercial_build=True).max_clones_per_mission == 2


def test_no_edition_advertises_cluster_fan_out_this_build_does_not_have():
    for edition in ("oss", "free", "professional", "enterprise"):
        assert limits_for(edition, has_commercial_build=True).cluster_fan_out is False


def test_grant_says_plainly_what_it_capped():
    g = grant(8, None, limits_for("oss", has_commercial_build=False))
    assert (g.clones, g.parallel, g.limited) == (2, 2, True)
    assert g.message.startswith("Swarm limited to 2 clones on the free open-source edition")
    assert "you asked for 8" in g.message


def test_grant_within_the_limit_is_silent():
    g = grant(2, None, limits_for("oss", has_commercial_build=False))
    assert (g.clones, g.limited, g.message) == (2, False, "")


# --------------------------------------------------------------------------- #
# The coordinator: the place clones are really dispatched
# --------------------------------------------------------------------------- #

def test_default_coordinator_caps_the_fleet_at_two(churn_csv, tmp_path, monkeypatch):
    """No edition configured, no commercial build: eight requested, two run."""
    assert limits_from_env().max_clones_per_mission == 2
    peaks: list[int] = []
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run(peaks))
    result = MainAvaloka().coordinate(
        str(churn_csv), goal="fleet", output_dir=tmp_path / "f", workers=8,
        max_parallel=8, on_progress=lambda e, m: events.append((e, m)),
    )
    assert result.workers == 2 and len(result.shards) == 2
    assert max(peaks) <= 2
    assert result.execution["parallelism"] == 2
    swarm = result.execution["swarm"]
    assert swarm["requested_clones"] == 8 and swarm["clones"] == 2 and swarm["limited"] is True
    # Announced before dispatch, never silently shrunk.
    limit_events = [m for e, m in events if e == "limit"]
    assert limit_events and "Swarm limited to 2 clones" in limit_events[0]
    assert [e for e, _ in events].index("limit") < [e for e, _ in events].index("announce")
    # And recorded in the persisted report.
    assert load_coordination(tmp_path / "f")["execution"]["swarm"]["limited"] is True


def test_forged_enterprise_edition_is_still_capped(churn_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "enterprise")
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run([]))
    result = MainAvaloka().coordinate(str(churn_csv), goal="g",
                                      output_dir=tmp_path / "f", workers=6)
    assert result.workers == 2


def test_explicit_policy_lifts_the_cap(churn_csv, tmp_path, monkeypatch):
    peaks: list[int] = []
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run(peaks))
    result = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="g", output_dir=tmp_path / "f", workers=6, max_parallel=6)
    assert result.workers == 6 and result.execution["swarm"]["limited"] is False
    assert result.execution["parallelism"] == 6


# --------------------------------------------------------------------------- #
# The CLI says so, even when told to be quiet
# --------------------------------------------------------------------------- #

def test_cli_announces_the_cap_even_with_quiet(churn_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run([]))
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(churn_csv), "--goal", "x",
                                      "--workers", "5", "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "Swarm limited to 2 clones" in text and "you asked for 5" in text
    assert load_coordination(out)["workers"] == 2


def test_cli_default_fleet_fits_the_edition_without_a_warning(churn_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run([]))
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(churn_csv), "--goal", "x",
                                      "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Swarm limited" not in result.output
    assert load_coordination(out)["workers"] == 2


# --------------------------------------------------------------------------- #
# The document states what the code enforces
# --------------------------------------------------------------------------- #

def test_editions_doc_states_the_enforced_limits():
    """docs/EDITIONS.md declares the swarm as a PREVIEW LIMIT; hold it to the code.

    The owner kept the 2-clone cap on self-hosted open source, which the
    COST/COMMERCIAL categories cannot describe. The document names it as its own
    category, gives the numbers and quotes the announced message. If either the
    table in avaloka/swarm_limits.py or the document changes alone, this fails.
    """
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[2] / "docs" / "EDITIONS.md").read_text()
    assert "| **PREVIEW LIMIT** |" in doc, "the reasons table lost its PREVIEW LIMIT row"
    assert "avaloka/swarm_limits.py" in doc

    row = next(line for line in doc.splitlines() if line.startswith("| Clones per mission |"))
    cells = [c.strip() for c in row.strip("|").split("|")][1:]
    stated = dict(zip(("oss", "free", "professional", "enterprise"), cells))
    for edition, text in stated.items():
        cap = limits_for(edition, has_commercial_build=True).max_clones_per_mission
        assert text == ("unlimited" if cap is None else str(cap)), (edition, text, cap)

    message = grant(8, None, limits_for("oss", has_commercial_build=False)).message
    assert f"> {message}" in doc, "EDITIONS.md must quote the message users actually see"


def _swarm_prose(doc: str) -> str:
    """The prose that states the swarm limits, as one line of text.

    The PREVIEW LIMIT paragraph and the Swarm section, without table rows and
    without the quoted example message (its "8" is the example request, pinned
    exactly by the test above).
    """
    start = doc.index("There is one preview limit today")
    end = doc.index("We do not ship gates or limits")
    section = doc[doc.index("### Swarm — open, with a preview limit"):]
    section = section[: section.index("\n### ", 1)]
    lines = (doc[start:end] + "\n" + section).splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith(("|", ">"))]
    return " ".join(" ".join(kept).split())


def test_editions_prose_states_the_enforced_numbers():
    """The sentences, not only the table, must say what the code enforces.

    Each phrase that states a limit is checked against avaloka/swarm_limits.py.
    Then every number in that prose, in digits or in words, must be one the
    code actually uses, so a number added or changed anywhere in it is caught.
    """
    import re
    from pathlib import Path

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16, "twenty": 20}
    num = r"(\d+|" + "|".join(words) + r")"

    def value(token: str) -> int:
        return int(token) if token.isdigit() else words[token.lower()]

    prose = _swarm_prose((Path(__file__).resolve().parents[2] / "docs" / "EDITIONS.md").read_text())
    preview = limits_for("oss", has_commercial_build=False).max_clones_per_mission
    pro = limits_for("professional", has_commercial_build=True).max_clones_per_mission
    assert limits_for("enterprise", has_commercial_build=True).max_clones_per_mission is None

    claims = {
        rf"capped at {num} clones per mission on the open-source and Free": preview,
        rf"{num} on Professional": pro,
        rf"Free editions get {num} clones per mission": preview,
        rf"Professional raises it to {num}": pro,
    }
    for pattern, expected in claims.items():
        found = re.findall(pattern, prose, flags=re.I)
        assert found, f"EDITIONS.md no longer states: {pattern!r}"
        for token in found:
            assert value(token) == expected, (pattern, token, expected)
    assert re.search(r"unlimited on Enterprise", prose) and re.search(r"Enterprise removes it", prose)

    # "Apache 2.0" is a licence version, and "one" reads as English ("only one",
    # "one answer") rather than a limit; neither is a clone count.
    counted = re.sub(r"Apache 2\.0", "", prose)
    tokens = [t for t in re.findall(rf"\b{num}\b", counted, flags=re.I) if t.lower() != "one"]
    stated = {value(t) for t in tokens}
    assert stated <= {preview, pro}, f"a number in the swarm prose is not one the code uses: {stated}"
