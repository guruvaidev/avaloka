"""Tests for the reporting layer.

These check that the numbers answer the questions honestly — particularly that
a report never states a rate it does not have the evidence for.
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from telemetry_service import reports
from telemetry_service.store import Store

DAY = 86_400


@pytest.fixture()
def store(tmp_path):
    s = Store(path=tmp_path / "reports.db")
    yield s
    s.close()


def iid(label):
    """An install id for a readable label. The ingest accepts only hex or UUID
    ids, so the tests name installations and store what a client would send."""
    return hashlib.md5(label.encode()).hexdigest()


def agent_event(index, install, agent="data_scout", outcome="success",
                duration=100, error=None):
    record = {
        "event_id": f"{index:032x}",
        "event_type": "agent.completed",
        "install_id": iid(install),
        "agent": agent,
        "outcome": outcome,
        "duration_ms": duration,
    }
    if error:
        record["error_class"] = error
    return record


def install_event(index, install, **extra):
    return {
        "event_id": f"1{index:031x}",
        "event_type": "install",
        "install_id": iid(install),
        "avaloka_version": "0.3.0",
        "edition": "oss",
        "deployment": "laptop",
        "cloud": "local",
        "os": "linux",
        "arch": "x86_64",
        "python_version": "3.11",
        "cpu_count": 8,
        **extra,
    }


class TestTheFixtures:
    def test_the_records_built_here_are_ones_the_ingest_accepts(self):
        """As sent, that is: an agent record carries no install id of its own.
        The store attributes it to the batch's install record, and these
        fixtures stand for records after that attribution."""
        from telemetry_service.schema import validate_batch

        sent = [install_event(0, "user-0")]
        for record in (agent_event(1, "user-0"),
                       agent_event(2, "user-0", agent="validator", outcome="failed",
                                   error="KeyError")):
            sent.append({k: v for k, v in record.items() if k != "install_id"})
        assert len(validate_batch(sent)) == 3


# ------------------------------------------------------------------- adoption

class TestAdoption:
    def test_it_separates_many_users_from_one_enthusiast(self, store):
        """1000 events from one install is a very different fact from 1000 installs."""
        now = time.time()
        store.store([agent_event(i, "solo") for i in range(50)], received_at=now)
        report = reports.adoption(store)
        row = report.rows[0]
        assert row["installations"] == 1
        assert row["events"] == 50
        assert row["events_per_install_max"] == 50

    def test_it_counts_installations_not_just_events(self, store):
        now = time.time()
        store.store(
            [agent_event(i, f"install-{i % 10}") for i in range(30)], received_at=now
        )
        row = reports.adoption(store).rows[0]
        assert row["installations"] == 10 and row["events"] == 30

    def test_it_surfaces_people_who_tried_once_and_left(self, store):
        """The most important adoption signal, and the easiest to miss."""
        now = time.time()
        records = [agent_event(i, f"once-{i}") for i in range(8)]
        records += [agent_event(100 + i, "regular") for i in range(20)]
        store.store(records, received_at=now)
        assert reports.adoption(store).rows[0]["single_event_installs"] == 8

    def test_an_empty_store_says_so_rather_than_showing_zeroes(self, store):
        report = reports.adoption(store)
        assert not report.rows and report.caveat


# --------------------------------------------------------------- capabilities

class TestCapabilityUsage:
    def test_it_reports_runs_and_installations_separately(self, store):
        """A feature used 100 times by one person is not a popular feature."""
        now = time.time()
        records = [agent_event(i, "solo", agent="model_scientist") for i in range(40)]
        records += [agent_event(100 + i, f"user-{i}", agent="reporter") for i in range(9)]
        store.store(records, received_at=now)

        rows = {r["agent"]: r for r in reports.capability_usage(store).rows}
        assert rows["model_scientist"]["runs"] == 40
        assert rows["model_scientist"]["installations"] == 1
        assert rows["reporter"]["runs"] == 9
        assert rows["reporter"]["installations"] == 9


# ---------------------------------------------------------------- reliability

class TestReliability:
    def test_failures_and_their_error_types_are_surfaced(self, store):
        now = time.time()
        records = [
            agent_event(i, f"user-{i}", agent="validator",
                        outcome="failed", error="KeyError")
            for i in range(6)
        ]
        records += [agent_event(100 + i, f"user-{i}", agent="validator") for i in range(4)]
        store.store(records, received_at=now)

        row = reports.reliability(store).rows[0]
        assert row["agent"] == "validator"
        assert row["failures"] == 6 and row["runs"] == 10
        assert row["top_error"] == "KeyError"
        assert row["failure_rate_pct"] == 60.0

    def test_no_rate_is_reported_on_too_few_installations(self, store):
        """Three failures out of four installs is not a 75% failure rate,
        it is four data points."""
        now = time.time()
        store.store(
            [agent_event(i, f"user-{i}", outcome="failed", error="ValueError")
             for i in range(3)],
            received_at=now,
        )
        row = reports.reliability(store).rows[0]
        assert row["failures"] == 3
        assert "failure_rate_pct" not in row
        assert "noise" in reports.reliability(store).caveat

    def test_timeouts_count_as_failures(self, store):
        now = time.time()
        store.store(
            [agent_event(i, f"user-{i}", outcome="timeout") for i in range(6)],
            received_at=now,
        )
        assert reports.reliability(store).rows[0]["failures"] == 6


# ---------------------------------------------------------------- performance

class TestPerformance:
    def test_percentiles_not_averages(self, store):
        """One user in twenty waiting four minutes is invisible in an average."""
        now = time.time()
        records = [agent_event(i, "u", duration=100) for i in range(19)]
        records.append(agent_event(999, "u", duration=240_000))
        store.store(records, received_at=now)

        row = reports.performance(store).rows[0]
        assert row["p50_ms"] == 100
        assert row["max_ms"] == 240_000
        assert row["samples"] == 20

    def test_slowest_agents_come_first(self, store):
        now = time.time()
        store.store(
            [agent_event(i, "u", agent="reporter", duration=10) for i in range(5)]
            + [agent_event(100 + i, "u", agent="train_models", duration=5000) for i in range(5)],
            received_at=now,
        )
        assert reports.performance(store).rows[0]["agent"] == "train_models"


# --------------------------------------------------------------- environments

class TestEnvironments:
    def test_each_installation_is_counted_once(self, store):
        """An install that uploads ten times is still one machine."""
        now = time.time()
        store.store([install_event(i, "same-machine") for i in range(10)], received_at=now)
        rows = {r["dimension"]: r["distribution"] for r in reports.environments(store).rows}
        assert rows["os"] == {"linux": 1}

    def test_the_distribution_is_reported(self, store):
        now = time.time()
        records = [install_event(i, f"linux-{i}") for i in range(7)]
        records += [install_event(100 + i, f"win-{i}", os="windows") for i in range(3)]
        store.store(records, received_at=now)
        assert reports.environments(store).rows[0]["distribution"] == {"linux": 7, "windows": 3}

    def test_an_unlisted_cloud_is_its_own_group_not_part_of_local(self, store):
        """`local` means not in a cloud. Folding an unrecognised provider into
        it would count a cloud deployment as a laptop."""
        records = [install_event(i, f"laptop-{i}") for i in range(4)]
        records += [install_event(100 + i, f"elsewhere-{i}", cloud="other") for i in range(2)]
        records += [install_event(200, "on-gcp", cloud="gcp")]
        store.store(records, received_at=time.time())
        rows = {r["dimension"]: r["distribution"] for r in reports.environments(store).rows}
        assert rows["cloud"] == {"local": 4, "other": 2, "gcp": 1}


# ------------------------------------------------------------------ freshness

class TestFreshness:
    def test_stale_data_is_visible(self, store):
        """A dashboard that stopped updating looks authoritative while lying."""
        now = time.time()
        store.store([agent_event(0, "old")], received_at=now - (30 * DAY))
        store.store([agent_event(1, "new")], received_at=now)
        row = reports.freshness(store, now=now).rows[0]
        assert row["events_held"] == 2
        assert row["received_in_last_7_days"] == 1
        assert row["older_than_7_days"] == 1

    def test_an_empty_store_says_so(self, store):
        assert reports.freshness(store).caveat


# ------------------------------------------------------------------- run_all

class TestRunAll:
    def test_every_report_runs_on_an_empty_store(self, store):
        """None of them may crash on no data, which is the state on day one."""
        result = reports.run_all(store)
        assert set(result) == set(reports.ALL_REPORTS)
        for payload in result.values():
            assert "title" in payload and "rows" in payload

    def test_every_report_runs_on_real_data(self, store):
        now = time.time()
        store.store(
            [install_event(i, f"user-{i}") for i in range(5)]
            + [agent_event(100 + i, f"user-{i % 5}") for i in range(25)],
            received_at=now,
        )
        result = reports.run_all(store)
        assert result["adoption"]["rows"][0]["installations"] == 5
        assert result["capabilities"]["rows"]


# ------------------------------------------------- the second producer's events

def analytics_events(deployment="d" * 32, session="5" * 32):
    """Product-analytics records as the ingest accepts them. They carry fields
    the process reports read -- install_id, duration_ms, outcome, error_class,
    session_id, seq -- which is exactly why they must be kept out."""
    common = {"install_id": deployment, "session_id": session, "actor_id": "a" * 32,
              "schema_version": "1.1"}
    turn = "7" * 32
    return [
        {"event_id": "e" * 31 + "0", "event_type": "analytics.session.started",
         "seq": 0, "t_ms": 0, "entry": "login", **common},
        {"event_id": "e" * 31 + "1", "event_type": "analytics.page.viewed",
         "seq": 1, "t_ms": 10, "route": "/analysis", **common},
        {"event_id": "e" * 31 + "2", "event_type": "analytics.question.submitted",
         "seq": 2, "t_ms": 20, "turn_id": turn, "input_method": "typed",
         "prompt_chars_bucket": "21-80", **common},
        {"event_id": "e" * 31 + "3", "event_type": "analytics.question.completed",
         "seq": 3, "t_ms": 9000, "turn_id": turn, "outcome": "failed",
         "error_class": "TimeoutError", "duration_ms": 86_000_000, **common},
        {"event_id": "e" * 31 + "4", "event_type": "analytics.dataset.connect_completed",
         "seq": 4, "t_ms": 9500, "source_kind": "upload", "outcome": "success",
         "file_format": "csv", "row_count_bucket": "1e3-1e4", "duration_ms": 1200,
         **common},
    ]


class TestAnalyticsEventsDoNotDistortTheReports:
    """These reports describe installations and agent runs. The store now also
    holds product-analytics events, which describe a browser session. Reports
    shaped for those do not exist yet; until they do, the ones here must give
    the same answers whether or not such events are present."""

    def _process_data(self, store, now):
        store.store(
            [install_event(i, f"user-{i}") for i in range(5)]
            + [agent_event(100 + i, f"user-{i % 5}", outcome="failed" if i % 4 == 0 else "success",
                           error="KeyError" if i % 4 == 0 else None) for i in range(25)]
            + [{"event_id": "d" * 32, "event_type": "dataset", "install_id": iid("user-0"),
                "row_count_bucket": "1e4-1e5", "file_format": "parquet", "sampled": True}],
            received_at=now - 2 * DAY,
        )

    def test_the_samples_are_records_the_ingest_really_accepts(self):
        from telemetry_service.schema import validate_batch

        assert len(validate_batch(analytics_events())) == 5

    def test_every_report_gives_the_same_answer_with_analytics_events_present(self, store, tmp_path):
        now = time.time()
        self._process_data(store, now)
        before = {name: fn(store).as_dict() for name, fn in reports.ALL_REPORTS.items()
                  if name != "freshness"}

        store.store(analytics_events(), received_at=now - 2 * DAY)
        after = {name: fn(store).as_dict() for name, fn in reports.ALL_REPORTS.items()
                 if name != "freshness"}

        assert after == before
        # ...and the comparison is not between two empty answers.
        assert before["adoption"]["rows"][0]["installations"] == 5
        assert before["reliability"]["rows"][0]["failures"] == 7
        assert before["data-shapes"]["rows"][0]["formats"] == {"parquet": 1}

    def test_a_deployment_is_not_counted_as_an_installation(self, store):
        """The analytics export mints its own install id. Counted here, one
        deployment would appear as two installations."""
        now = time.time()
        self._process_data(store, now)
        store.store(analytics_events(), received_at=now)
        row = reports.adoption(store).rows[0]
        assert row["installations"] == 5
        assert row["events"] == 31

    def test_a_page_view_is_not_an_agent_run_and_a_question_is_not_a_slow_agent(self, store):
        store.store(analytics_events(), received_at=time.time())
        assert reports.capability_usage(store).rows == []
        assert reports.reliability(store).rows == []
        assert reports.performance(store).rows == []
        assert reports.data_shapes(store).rows[0]["formats"] == {}
        assert reports.sessions(store).rows == []
        assert reports.interrupted(store).rows == []

    def test_a_store_holding_only_analytics_events_crashes_nothing(self, store):
        store.store(analytics_events(), received_at=time.time())
        result = reports.run_all(store)
        assert set(result) == set(reports.ALL_REPORTS)
        assert result["adoption"]["caveat"] == "No events received yet."

    def test_freshness_counts_everything_held_and_says_so(self, store):
        """The one report about the store rather than the installations."""
        now = time.time()
        self._process_data(store, now)
        store.store(analytics_events(), received_at=now)
        row = reports.freshness(store, now=now).rows[0]
        assert row["events_held"] == 31 + 5
        assert "analytics" in reports.freshness.__doc__


# ------------------------------------------------------------ understanding

def understanding_event(index, install, **extra):
    record = {
        "event_id": f"7{index:031x}",
        "event_type": "agent.understanding",
        "install_id": iid(install),
        "agent": "data_scout",
        "role": "understanding",
        "outcome": "success",
        "duration_ms": 120,
        "row_count_bucket": "1e3-1e4",
        "column_count_bucket": "5-10",
        "column_type_counts": {"numeric": 4, "categorical": 2},
        "null_rate_counts": {"0": 5, "10-50%": 1},
        "cardinality_counts": {"2": 1, "11-100": 5},
        "file_format": "csv",
        "size_bucket": "1-10MB",
        **extra,
    }
    return record


def swarm_understanding_event(index, install, **extra):
    return {
        "event_id": f"8{index:031x}",
        "event_type": "swarm.understanding",
        "install_id": iid(install),
        "clones": 2,
        "parallel": 2,
        "steps_total": 12,
        "steps_per_clone_min": 6,
        "steps_per_clone_max": 6,
        "role_counts": {"understanding": 2, "verification": 2},
        "outcome": "success",
        "verdict": "warn",
        "merge_agreement": "unanimous",
        "model_selected": False,
        "candidates": 0,
        "checks_run": 8, "checks_passed": 7, "checks_warned": 1, "checks_failed": 0,
        **extra,
    }


class TestUnderstanding:
    def _data(self, store, now):
        from telemetry_service.schema import validate_batch

        records = (
            [install_event(i, f"user-{i}") for i in range(2)]
            + [understanding_event(10, "user-0"),
               understanding_event(11, "user-0", agent="sampling_specialist", role="sampling", sampled=True,
                                   sampling_strategy="random", sample_fraction_bucket="10-50%"),
               understanding_event(12, "user-1", agent="data_engineer", role="feature_engineering",
                                   operation_count=3, op_counts={"drop_duplicate_rows": 1, "strip_whitespace": 2}),
               understanding_event(13, "user-1", agent="validator", role="verification", checks_run=4,
                                   checks_passed=3, checks_warned=1, checks_failed=0, verdict="warn",
                                   llm_provider="groq", llm_model="other", tokens_in=200, tokens_out=30)]
            + [swarm_understanding_event(20, "user-0"), swarm_understanding_event(21, "user-1", clones=4,
                                                                                  merge_agreement="split")]
        )
        sent = [records[0]] + [{k: v for k, v in r.items() if k != "install_id"} for r in records[1:]]
        validate_batch(sent)  # the samples are records the ingest really accepts
        store.store(records, received_at=now)

    def test_it_sums_histograms_and_counts_roles_never_per_column(self, store):
        from telemetry_service.reports import understanding

        self._data(store, time.time())
        row, = understanding(store).rows
        assert row["steps"] == 4 and row["missions"] == 2 and row["installations"] == 2
        assert row["roles"] == {"feature_engineering": 1, "sampling": 1, "understanding": 1, "verification": 1}
        assert row["sampled_steps"] == 1 and row["sampling_decisions"] == 1
        assert row["sample_fraction_buckets"] == {"10-50%": 1}
        assert row["operations"] == {"drop_duplicate_rows": 1, "strip_whitespace": 2}
        assert row["checks"] == {"run": 4, "passed": 3, "warned": 1, "failed": 0}
        assert row["column_kinds"] == {"categorical": 8, "numeric": 16}
        assert row["null_rate_bands"] == {"0": 20, "10-50%": 4}
        assert row["size_buckets"] == {"1-10MB": 4}
        assert row["tokens_in"] == 200 and row["tokens_out"] == 30
        assert row["fleet_sizes"] == {"2": 1, "4": 1}
        assert row["merged_verdicts"] == {"warn": 2} and row["merge_agreement"] == {"split": 1, "unanimous": 1}
        text = json.dumps(understanding(store).as_dict())
        assert "data_scout" not in text  # roles, not agents: the report is about what was done

    def test_an_empty_store_says_so(self, store):
        from telemetry_service.reports import understanding

        report = understanding(store)
        assert report.rows == [] and "No understanding" in report.caveat
