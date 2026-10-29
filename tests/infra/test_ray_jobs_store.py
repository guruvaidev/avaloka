"""ray_jobs_store: what reaches the Supabase ray_jobs table, against a fake client."""

import sys
import types
import unittest
from unittest import mock

from app.infra import ray_jobs_store as store


class FakeQuery:
    def __init__(self, log, data=None, fail=False):
        self.log, self.data, self.fail = log, data or [], fail

    def __getattr__(self, name):
        def step(*args, **kwargs):
            self.log.append((name, args, kwargs))
            return self
        return step

    def execute(self):
        if self.fail:
            raise RuntimeError('relation "public.ray_jobs" does not exist')
        return types.SimpleNamespace(data=self.data)


class StoreTests(unittest.TestCase):
    def _patch(self, data=None, fail=False):
        log = []
        query = FakeQuery(log, data, fail)
        client = types.SimpleNamespace(table=lambda name: (log.append(("table", (name,), {})), query)[1])
        mod = types.ModuleType("app.agents.sampling_persistence")
        mod.get_supabase_client = lambda: client
        p = mock.patch.dict(sys.modules, {"app.agents.sampling_persistence": mod})
        p.start()
        self.addCleanup(p.stop)
        return log

    def test_submission_drops_unknown_keys(self):
        log = self._patch()
        store.record_submission({"submission_id": "dta-1", "user_id": "u1", "kind": "data_transfer",
                                 "password": "x", "dashboard_url": "http://ray:8265"})
        upsert = next(e for e in log if e[0] == "upsert")
        row = upsert[1][0]
        self.assertNotIn("password", row)
        self.assertEqual(row["user_id"], "u1")
        self.assertEqual(upsert[2], {"on_conflict": "submission_id"})
        self.assertIn("created_at", row)

    def test_submission_needs_user(self):
        self._patch()
        with self.assertRaises(ValueError):
            store.record_submission({"submission_id": "dta-1"})

    def test_update_never_changes_owner(self):
        log = self._patch()
        store.record_update("dta-1", {"status": "SUCCEEDED", "user_id": "attacker"})
        payload = next(e for e in log if e[0] == "update")[1][0]
        self.assertNotIn("user_id", payload)
        self.assertEqual(payload["status"], "SUCCEEDED")
        self.assertIn(("eq", ("submission_id", "dta-1"), {}), log)

    def test_reads_filter_by_user(self):
        log = self._patch(data=[{"submission_id": "dta-1", "user_id": "u1"}])
        self.assertEqual(store.get_for_user("u1", "dta-1")["submission_id"], "dta-1")
        self.assertIn(("eq", ("user_id", "u1"), {}), log)

    def test_missing_table_is_store_unavailable(self):
        self._patch(fail=True)
        with self.assertRaises(store.StoreUnavailable):
            store.list_for_user("u1")


if __name__ == "__main__":
    unittest.main()
