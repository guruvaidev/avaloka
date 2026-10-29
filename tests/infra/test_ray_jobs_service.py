"""ray_jobs_service: ownership, fallbacks and Ray calls behind /api/ray/*.

The job-records store and the Ray client are faked. Runs under pytest or
`python -m unittest`.
"""

import enum
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


def _install_fake_ray():
    if "ray.job_submission" in sys.modules and getattr(sys.modules["ray.job_submission"], "_FAKE", False):
        return sys.modules["ray.job_submission"]

    class JobStatus(str, enum.Enum):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        STOPPED = "STOPPED"
        SUCCEEDED = "SUCCEEDED"
        FAILED = "FAILED"

    mod = types.ModuleType("ray.job_submission")
    mod._FAKE = True
    mod.JobStatus = JobStatus
    mod.JobSubmissionClient = object
    ray = sys.modules.setdefault("ray", types.ModuleType("ray"))
    ray.job_submission = mod
    sys.modules["ray.job_submission"] = mod
    return mod


fake = _install_fake_ray()

from app.infra import ray_jobs_service as svc  # noqa: E402
from app.infra import ray_jobs_store  # noqa: E402

URL = "http://ray-head:8265"


def rec(sid, user="u1", status="SUCCEEDED", **kw):
    return {"submission_id": sid, "user_id": user, "status": status, "kind": "data_transfer",
            "dashboard_url": URL, "created_at": "2026-10-24T10:47:06+00:00", **kw}


def info(sid, user="u1", status="RUNNING", **kw):
    return SimpleNamespace(submission_id=sid, status=getattr(fake.JobStatus, status),
                           message=kw.get("message", ""), start_time=1792838826000,
                           end_time=kw.get("end_time"), metadata={"user_id": user, "kind": "data_transfer"})


class FakeStore:
    def __init__(self, rows=(), unavailable=False):
        self.rows = {r["submission_id"]: dict(r) for r in rows}
        self.unavailable = unavailable
        self.updates = []

    def get_for_user(self, user_id, sid):
        if self.unavailable:
            raise ray_jobs_store.StoreUnavailable("relation ray_jobs does not exist")
        r = self.rows.get(sid)
        return r if r and r["user_id"] == user_id else None

    def list_for_user(self, user_id, kind=None, status=None, limit=50):
        if self.unavailable:
            raise ray_jobs_store.StoreUnavailable("down")
        return [r for r in self.rows.values() if r["user_id"] == user_id][:limit]

    def record_update(self, sid, fields):
        self.updates.append((sid, dict(fields)))


class FakeClient:
    def __init__(self, jobs=(), logs="line1\nline2\nline3\n", stop_result=True):
        self.jobs = {j.submission_id: j for j in jobs}
        self.logs = logs
        self.stop_result = stop_result
        self.stopped = []

    def get_job_info(self, sid):
        if sid not in self.jobs:
            raise RuntimeError(f"Job {sid} does not exist")
        return self.jobs[sid]

    def list_jobs(self):
        return list(self.jobs.values())

    def get_job_logs(self, sid):
        self.get_job_info(sid)
        return self.logs

    def stop_job(self, sid):
        self.stopped.append(sid)
        self.jobs[sid].status = fake.JobStatus.STOPPED
        return self.stop_result

    def get_job_status(self, sid):
        return self.jobs[sid].status


class Base(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.client = FakeClient()
        patches = [
            mock.patch.object(svc, "ray_jobs_store", self._store_module()),
            mock.patch.object(svc, "_client", lambda url: self.client),
            mock.patch.dict("os.environ", {"RAY_DASHBOARD_URL": URL}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _store_module(self):
        test = self
        return SimpleNamespace(
            StoreUnavailable=ray_jobs_store.StoreUnavailable,
            get_for_user=lambda *a: test.store.get_for_user(*a),
            list_for_user=lambda *a, **k: test.store.list_for_user(*a, **k),
            record_update=lambda *a: test.store.record_update(*a),
        )


class OwnershipTests(Base):
    def test_list_returns_only_callers_jobs_without_cluster_url(self):
        self.store = FakeStore([rec("dta-a"), rec("dta-b", user="u2")])
        jobs = svc.list_jobs("u1")
        self.assertEqual([j["submission_id"] for j in jobs], ["dta-a"])
        self.assertNotIn("dashboard_url", jobs[0])
        self.assertNotIn("user_id", jobs[0])

    def test_someone_elses_job_is_404(self):
        self.store = FakeStore([rec("dta-b", user="u2")])
        for call in (svc.get_job, svc.get_logs, svc.stop_job):
            with self.assertRaises(svc.RayJobsError) as ctx:
                call("u1", "dta-b")
            self.assertEqual(ctx.exception.status_code, 404)

    def test_malformed_id_is_404_without_lookup(self):
        with self.assertRaises(svc.RayJobsError) as ctx:
            svc.get_job("u1", "../../etc/passwd")
        self.assertEqual(ctx.exception.status_code, 404)


class LiveStatusTests(Base):
    def test_running_record_gets_live_status_and_final_is_stored(self):
        self.store = FakeStore([rec("dta-a", status="PENDING")])
        self.client = FakeClient([info("dta-a", status="SUCCEEDED", end_time=1792838871571)])
        job = svc.get_job("u1", "dta-a")
        self.assertEqual(job["status"], "SUCCEEDED")
        self.assertEqual(job["runtime_s"], 45.57)
        self.assertEqual(self.store.updates[0][0], "dta-a")

    def test_finished_record_is_not_sent_to_ray(self):
        self.store = FakeStore([rec("dta-a", status="FAILED", message="boom")])
        self.client = None  # any Ray call would fail
        self.assertEqual(svc.get_job("u1", "dta-a")["message"], "boom")

    def test_ray_unreachable_keeps_stored_status(self):
        self.store = FakeStore([rec("dta-a", status="RUNNING")])
        self.client = FakeClient([])  # Ray no longer knows the job
        self.assertEqual(svc.get_job("u1", "dta-a")["status"], "RUNNING")


class FallbackTests(Base):
    def test_store_down_lists_from_ray_metadata(self):
        self.store = FakeStore(unavailable=True)
        self.client = FakeClient([info("dta-a"), info("dta-b", user="u2")])
        jobs = svc.list_jobs("u1")
        self.assertEqual([j["submission_id"] for j in jobs], ["dta-a"])
        self.assertEqual(jobs[0]["status"], "RUNNING")

    def test_store_down_ownership_from_ray_metadata(self):
        self.store = FakeStore(unavailable=True)
        self.client = FakeClient([info("dta-b", user="u2")])
        with self.assertRaises(svc.RayJobsError) as ctx:
            svc.get_job("u1", "dta-b")
        self.assertEqual(ctx.exception.status_code, 404)


class LogsAndStopTests(Base):
    def test_logs_tail(self):
        self.store = FakeStore([rec("dta-a")])
        self.client = FakeClient([info("dta-a", status="SUCCEEDED")])
        out = svc.get_logs("u1", "dta-a", tail=2)
        self.assertEqual(out["lines"], ["line2", "line3"])
        self.assertTrue(out["truncated"])

    def test_logs_gone_after_head_restart_is_410(self):
        self.store = FakeStore([rec("dta-a")])
        self.client = FakeClient([])
        with self.assertRaises(svc.RayJobsError) as ctx:
            svc.get_logs("u1", "dta-a")
        self.assertEqual(ctx.exception.status_code, 410)

    def test_stop_running_job(self):
        self.store = FakeStore([rec("dta-a", status="RUNNING")])
        self.client = FakeClient([info("dta-a", status="RUNNING")])
        out = svc.stop_job("u1", "dta-a")
        self.assertEqual(out, {"submission_id": "dta-a", "stopped": True, "status": "STOPPED"})
        self.assertEqual(self.client.stopped, ["dta-a"])
        self.assertEqual(self.store.updates[-1][1]["status"], "STOPPED")

    def test_stop_finished_job_is_409(self):
        self.store = FakeStore([rec("dta-a", status="SUCCEEDED")])
        with self.assertRaises(svc.RayJobsError) as ctx:
            svc.stop_job("u1", "dta-a")
        self.assertEqual(ctx.exception.status_code, 409)


class StatusTests(unittest.TestCase):
    def test_not_configured(self):
        with mock.patch.dict("os.environ", {"RAY_DASHBOARD_URL": ""}):
            self.assertEqual(svc.cluster_status(), {"configured": False, "reachable": False})

    def test_unreachable(self):
        with mock.patch.dict("os.environ", {"RAY_DASHBOARD_URL": "http://127.0.0.1:9"}), \
             mock.patch.object(svc.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertEqual(svc.cluster_status(), {"configured": True, "reachable": False})

    def test_reachable_with_nodes(self):
        bodies = iter([b'{"ray_version": "2.49.2"}', b'{"data": {"summary": [{}, {}]}}'])

        class Resp:
            def __init__(self):
                self.body = next(bodies)

            def read(self):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch.dict("os.environ", {"RAY_DASHBOARD_URL": URL}), \
             mock.patch.object(svc.urllib.request, "urlopen", side_effect=lambda *a, **k: Resp()):
            self.assertEqual(svc.cluster_status(),
                             {"configured": True, "reachable": True, "ray_version": "2.49.2", "nodes": 2})


if __name__ == "__main__":
    unittest.main()
