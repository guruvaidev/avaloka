"""Tests for reconstructing a session from its start and stop records.

The pair exists so a run can be replayed and so a killed process is visible.
These check both, and check that the two kinds of failure — an agent that
raised and a process that was killed — never get counted as the same thing.
"""

from __future__ import annotations

import hashlib
import time

import pytest

from telemetry_service import reports
from telemetry_service.store import Store


#: Comfortably past reports.UNFINISHED_GRACE_SECONDS, so a start with no
#: completion is old enough to be called a kill rather than assumed in flight.
_LONG_AGO = time.time() - 7 * 86400


@pytest.fixture()
def store(tmp_path):
    s = Store(path=tmp_path / "sessions.db")
    yield s
    s.close()


class _Session:
    """Builds the records one run would produce, in the order it produces them."""

    def __init__(self, session_id="5" * 32):
        self.session_id = session_id
        self.records: list[dict] = []
        self._seq = 0
        self._n = 0

    def _next(self):
        self._seq += 1
        self._n += 1
        return self._seq

    def _event_id(self, what):
        # Hex, as the ingest requires, and derived from the whole session id.
        return hashlib.md5(f"{self.session_id}-{what}-{self._n:04d}".encode()).hexdigest()

    def start(self, agent):
        seq = self._next()
        self.records.append({
            # The full session id, not a prefix. An earlier version used the
            # first eight characters and every session in a loop produced the
            # same event ids, so the store's dedup collapsed twenty-five runs
            # into one -- correct behaviour, useless test data.
            "event_id": self._event_id("start"),
            "event_type": "agent.started",
            "session_id": self.session_id,
            "seq": seq,
            "agent": agent,
        })
        return seq

    def finish(self, agent, started_seq, outcome="success", duration=100):
        self.records.append({
            "event_id": self._event_id("done"),
            "event_type": "agent.completed",
            "session_id": self.session_id,
            "seq": self._next(),
            "started_seq": started_seq,
            "agent": agent,
            "outcome": outcome,
            "duration_ms": duration,
        })

    def ran(self, agent, outcome="success", duration=100):
        self.finish(agent, self.start(agent), outcome, duration)
        return self

    def killed_during(self, agent):
        """A start with no completion — what a SIGKILL leaves behind."""
        self.start(agent)
        return self


class TestTheFixtures:
    def test_the_records_built_here_are_ones_the_ingest_accepts(self):
        """Reports tested on records the endpoint would refuse prove little."""
        from telemetry_service.schema import validate_batch

        run = _Session()
        outer = run.start("code_etl")
        run.ran("coder", outcome="failed")
        run.finish("code_etl", outer, duration=900)
        run.killed_during("train_models")
        assert len(validate_batch(run.records)) == len(run.records)


class TestReconstruction:
    def test_a_run_is_replayed_in_the_order_it_happened(self, store):
        run = _Session()
        for agent in ["memory_injection", "plan_etl", "code_etl", "visualize"]:
            run.ran(agent)
        store.store(run.records, received_at=time.time())

        row = reports.sessions(store).rows[0]
        assert row["order"] == ["memory_injection", "plan_etl", "code_etl", "visualize"]
        assert row["agents_started"] == 4
        assert row["agents_completed"] == 4
        assert row["never_finished"] == []

    def test_records_are_ordered_by_seq_not_arrival(self, store):
        """Batches upload out of order and a redelivery can arrive much later.
        Ordering by received_at would scramble the reconstruction."""
        run = _Session()
        for agent in ["first", "second", "third"]:
            run.ran(agent)

        # Deliver the last agent's records first, an hour earlier.
        now = time.time()
        store.store(run.records[4:], received_at=now - 3600)
        store.store(run.records[:4], received_at=now)

        assert reports.sessions(store).rows[0]["order"] == ["first", "second", "third"]

    def test_nested_agents_are_reconstructed(self, store):
        """The coding sub-graph runs inside code_etl, so the records interleave."""
        run = _Session()
        outer = run.start("code_etl")
        inner = run.start("coder")
        run.finish("coder", inner)
        run.finish("code_etl", outer, duration=900)
        store.store(run.records, received_at=time.time())

        row = reports.sessions(store).rows[0]
        assert row["order"] == ["code_etl", "coder"]
        assert row["never_finished"] == []

    def test_separate_runs_stay_separate(self, store):
        first, second = _Session("a" * 32), _Session("b" * 32)
        first.ran("plan_etl")
        second.ran("visualize")
        store.store(first.records + second.records, received_at=time.time())

        rows = reports.sessions(store).rows
        assert len(rows) == 2
        assert {tuple(r["order"]) for r in rows} == {("plan_etl",), ("visualize",)}

    def test_an_empty_store_says_so(self, store):
        assert reports.sessions(store).caveat


class TestKilledProcesses:
    """The failure that was invisible before start records existed.

    Every store() here uses a timestamp well in the past. A start is only judged
    once the grace period has passed, so records that just arrived are treated
    as still in flight -- correct behaviour, but it would make these tests
    assert nothing.
    """

    def test_a_start_with_no_completion_is_surfaced(self, store):
        run = _Session()
        run.ran("plan_etl")
        run.killed_during("train_models")
        store.store(run.records, received_at=_LONG_AGO)

        row = reports.sessions(store).rows[0]
        assert row["never_finished"] == ["train_models"]
        assert row["agents_started"] == 2
        assert row["agents_completed"] == 1

    def test_a_kill_is_not_counted_as_a_failure(self, store):
        """A KeyError is a bug in our code; a kill is the workload exceeding the
        machine. Different fixes, so they must not share a number."""
        run = _Session()
        run.ran("validator", outcome="failed")
        run.killed_during("train_models")
        store.store(run.records, received_at=_LONG_AGO)

        row = reports.sessions(store).rows[0]
        assert row["failed"] == ["validator"]
        assert row["never_finished"] == ["train_models"]

    def test_interrupted_agents_are_counted_across_sessions(self, store):
        records = []
        for index in range(6):
            run = _Session(f"{index:032d}")
            run.killed_during("train_models")
            records += run.records
        for index in range(6, 10):
            run = _Session(f"{index:032d}")
            run.ran("train_models")
            records += run.records
        store.store(records, received_at=_LONG_AGO)

        row = reports.interrupted(store).rows[0]
        assert row["agent"] == "train_models"
        assert row["started"] == 10
        assert row["never_finished"] == 6
        assert row["interrupted_pct"] == 60.0

    def test_no_rate_is_claimed_on_too_few_runs(self, store):
        """Three kills out of four is not a 75% rate, it is four data points."""
        records = []
        for index in range(3):
            run = _Session(f"{index:032d}")
            run.killed_during("train_models")
            records += run.records
        store.store(records, received_at=_LONG_AGO)

        row = reports.interrupted(store).rows[0]
        assert row["never_finished"] == 3
        assert "interrupted_pct" not in row
        assert reports.interrupted(store).caveat

    def test_sessions_with_kills_are_listed_first(self, store):
        healthy = _Session("c" * 32)
        for agent in ["a", "b", "c"]:
            healthy.ran(agent)
        broken = _Session("d" * 32)
        broken.killed_during("train_models")
        store.store(healthy.records + broken.records, received_at=_LONG_AGO)

        assert reports.sessions(store).rows[0]["never_finished"] == ["train_models"]


class TestFindingsFromReview:
    """Each of these pins a defect an adversarial review confirmed."""

    def test_a_report_covers_more_than_the_newest_hundred_records(self, store):
        """store.records() defaults to 100. Using that default meant every
        counter was computed over whatever hundred rows arrived last, fleet-wide
        — so real kills vanished and healthy runs were reported as crashes."""
        records = []
        for index in range(120):
            run = _Session(f"{index:032d}")
            run.ran("plan_etl")
            records += run.records
        old = time.time() - 10 * 86400
        store.store(records, received_at=old)

        assert len(reports.sessions(store, limit=999).rows) == 120
        assert reports.interrupted(store).rows[0]["started"] == 120

    def test_an_in_flight_run_is_not_reported_as_a_crash(self, store):
        """A run writes both records milliseconds apart, but the client uploads
        every six hours — so a batch boundary can deliver a start without its
        completion. Calling that a kill invents the exact signal this exists to
        detect."""
        run = _Session()
        run.killed_during("train_models")
        store.store(run.records, received_at=time.time())      # just arrived

        assert reports.sessions(store).rows[0]["never_finished"] == []
        assert reports.interrupted(store).rows == []

    def test_an_old_unmatched_start_is_still_reported(self, store):
        """The grace period must not swallow real crashes."""
        run = _Session()
        run.killed_during("train_models")
        store.store(run.records, received_at=time.time() - 3 * 86400)

        assert reports.sessions(store).rows[0]["never_finished"] == ["train_models"]
        assert reports.interrupted(store).rows[0]["never_finished"] == 1

    def test_out_of_order_delivery_does_not_fabricate_kills(self, store):
        """Completions delivered in an earlier batch than their starts — the
        normal periodic upload pattern. Every one of these runs succeeded."""
        records = []
        for index in range(10):
            run = _Session(f"{index:032d}")
            run.ran("train_models")
            records += run.records

        old = time.time() - 5 * 86400
        starts = [r for r in records if r["event_type"] == "agent.started"]
        completions = [r for r in records if r["event_type"] == "agent.completed"]
        store.store(completions, received_at=old - 3600)
        store.store(starts, received_at=old)

        assert reports.interrupted(store).rows[0]["never_finished"] == 0
        assert all(not r["never_finished"] for r in reports.sessions(store).rows)

    def test_a_record_the_server_accepts_never_crashes_a_report(self, store):
        """The ingest schema requires only event_id and event_type. A record
        without `agent` or `seq` is therefore accepted by the public endpoint,
        and indexing it took down run_all() along with this report."""
        store.store([
            {"event_id": "1" * 32, "event_type": "agent.started"},
            {"event_id": "2" * 32, "event_type": "agent.completed"},
            {"event_id": "3" * 32, "event_type": "agent.started",
             "session_id": "5" * 32},
        ], received_at=time.time() - 5 * 86400)

        reports.sessions(store)          # must not raise
        reports.interrupted(store)
        reports.run_all(store)

    def test_nested_time_is_not_counted_twice(self, store):
        """The coding sub-graph's duration is already inside code_etl's."""
        run = _Session()
        outer = run.start("code_etl")
        inner = run.start("coder")
        run.finish("coder", inner, duration=400)
        run.finish("code_etl", outer, duration=1000)
        store.store(run.records, received_at=time.time() - 5 * 86400)

        assert reports.sessions(store).rows[0]["total_ms"] == 1000

    def test_sibling_top_level_agents_are_both_counted(self, store):
        """The counterpart: two agents at the same level are not nested, so both
        their durations belong in the total."""
        run = _Session()
        run.ran("plan_etl", duration=300)
        run.ran("visualize", duration=200)
        store.store(run.records, received_at=time.time() - 5 * 86400)

        assert reports.sessions(store).rows[0]["total_ms"] == 500


class TestItStaysHonest:
    def test_every_report_still_runs_on_an_empty_store(self, store):
        result = reports.run_all(store)
        assert set(result) == set(reports.ALL_REPORTS)
        for payload in result.values():
            assert "title" in payload and "rows" in payload

    def test_the_new_reports_are_registered(self):
        assert "sessions" in reports.ALL_REPORTS
        assert "interrupted" in reports.ALL_REPORTS

    def test_a_truncated_listing_says_it_was_truncated(self, store):
        """A capped list that does not say so reads as the whole picture."""
        records = []
        for index in range(25):
            run = _Session(f"{index:032d}")
            run.ran("plan_etl")
            records += run.records
        store.store(records, received_at=time.time())

        report = reports.sessions(store, limit=5)
        assert len(report.rows) == 5
        assert report.caveat and "25" in report.caveat
