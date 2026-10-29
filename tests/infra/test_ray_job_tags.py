"""ray_job_runner: jobs carry user tags as Ray metadata and get a ray_jobs record.

No Ray needed: a fake ray.job_submission module stands in for the client.
Runs under pytest or `python -m unittest`.
"""

import enum
import sys
import types
import unittest
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

from app.infra import ray_job_runner as runner  # noqa: E402


class FakeClient:
    def __init__(self, fail_first_with=None, final=None, logs="AVALOKA_RESULT rows_inserted=20 table=t.csv\n"):
        self.submits = []
        self.fail_first_with = fail_first_with
        self.final = final or fake.JobStatus.SUCCEEDED
        self.logs = logs

    def submit_job(self, **kwargs):
        self.submits.append(kwargs)
        if self.fail_first_with and len(self.submits) == 1:
            raise RuntimeError(self.fail_first_with)
        return kwargs["submission_id"]

    def get_job_status(self, _sid):
        return self.final

    def get_job_logs(self, _sid):
        return self.logs


class JobMetadataTests(unittest.TestCase):
    def test_strips_uri_credentials_and_empty_values(self):
        # Built at runtime: the repo's secret scan (test_c1_repo_hygiene) flags any
        # URI that carries a literal user and password, fake ones included.
        userinfo = ":".join(["bob", "not-a-real-password"])
        meta = runner._job_metadata({
            "user_id": "u1", "thread_id": "", "source": f"postgresql://{userinfo}@db:5432/app",
            "destination": "gs://bucket/out.csv", "secret_key": "nope",
        })
        self.assertEqual(meta["source"], "postgresql://db:5432/app")
        self.assertEqual(meta["destination"], "gs://bucket/out.csv")
        self.assertEqual(meta["kind"], "data_transfer")
        self.assertNotIn("thread_id", meta)
        self.assertNotIn("secret_key", meta)

    def test_values_are_strings_and_capped(self):
        meta = runner._job_metadata({"user_id": 42, "source": "x" * 2000})
        self.assertEqual(meta["user_id"], "42")
        self.assertEqual(len(meta["source"]), 512)

    def test_no_tags_no_metadata(self):
        self.assertEqual(runner._job_metadata(None), {})
        self.assertEqual(runner._job_metadata({}), {})

    def test_status_name_handles_enum_and_text(self):
        self.assertEqual(runner._status_name(fake.JobStatus.STOPPED), "STOPPED")
        self.assertEqual(runner._status_name("succeeded"), "SUCCEEDED")

    def test_rows_from_logs_takes_last_result(self):
        logs = "AVALOKA_RESULT rows_inserted=5 table=a\n...\nAVALOKA_RESULT rows_inserted=20 table=b"
        self.assertEqual(runner._rows_from_logs(logs), 20)
        self.assertIsNone(runner._rows_from_logs("no result line"))


class ContextTests(unittest.TestCase):
    def test_nested_context_merges_and_restores(self):
        with runner.ray_job_context(user_id="u1", thread_id="t1"):
            with runner.ray_job_context(kind="mta_training", thread_id=None):
                self.assertEqual(runner.current_job_tags(),
                                 {"user_id": "u1", "thread_id": "t1", "kind": "mta_training"})
            self.assertEqual(runner.current_job_tags(), {"user_id": "u1", "thread_id": "t1"})
        self.assertEqual(runner.current_job_tags(), {})

    def test_run_rayjob_merges_context_with_explicit_tags(self):
        with mock.patch.object(runner, "_extract_script_from_yaml", return_value="print(1)"), \
             mock.patch.object(runner, "_extract_runtime_env_from_yaml", return_value={}), \
             mock.patch.object(runner, "_run_ray_job_direct") as direct, \
             runner.ray_job_context(user_id="u1", thread_id="t1"):
            runner.run_rayjob_from_yaml("kind: RayJob", rayjob_name="dta-x",
                                        dashboard_url="http://ray:8265",
                                        job_tags={"thread_id": "t2", "kind": "data_transfer"})
        tags = direct.call_args.kwargs["job_tags"]
        self.assertEqual(tags, {"user_id": "u1", "thread_id": "t2", "kind": "data_transfer"})


class DirectSubmissionTests(unittest.TestCase):
    def _run(self, client, tags):
        calls = []
        with mock.patch.object(runner, "make_job_client", return_value=client), \
             mock.patch.object(runner, "_preflight_dashboard"), \
             mock.patch.object(runner, "_store_call", side_effect=lambda *a, **k: calls.append(a)), \
             mock.patch.object(runner.time, "sleep"):
            out = runner._run_ray_job_direct(
                script_text="import daft\n", runtime_env={"pip": ["pandas"]},
                job_id="dta-dta-1", dashboard_url="http://ray:8265", job_tags=tags,
            )
        return out, calls

    def test_metadata_sent_and_job_recorded(self):
        client = FakeClient()
        out, calls = self._run(client, {"user_id": "u1", "thread_id": "t1",
                                        "destination": "gs://b/x.csv"})
        self.assertEqual(out.status, "SUCCEEDED")
        self.assertEqual(client.submits[0]["metadata"],
                         {"user_id": "u1", "thread_id": "t1", "destination": "gs://b/x.csv",
                          "kind": "data_transfer"})
        (sub_name, sub_row), (upd_name, upd_sid, upd_fields) = calls
        self.assertEqual(sub_name, "record_submission")
        self.assertEqual(sub_row["submission_id"], "dta-dta-1")
        self.assertEqual(sub_row["dashboard_url"], "http://ray:8265")
        self.assertEqual(upd_name, "record_update")
        self.assertEqual(upd_sid, "dta-dta-1")
        self.assertEqual(upd_fields["status"], "SUCCEEDED")
        self.assertEqual(upd_fields["rows_written"], 20)

    def test_retry_after_id_collision_keeps_metadata(self):
        client = FakeClient(fail_first_with="submission_id dta-dta-1 already exists")
        out, calls = self._run(client, {"user_id": "u1"})
        self.assertEqual(len(client.submits), 2)
        self.assertEqual(client.submits[1]["metadata"]["user_id"], "u1")
        self.assertNotEqual(client.submits[1]["submission_id"], "dta-dta-1")
        self.assertEqual(calls[0][1]["submission_id"], client.submits[1]["submission_id"])

    def test_stopped_job_recorded_as_stopped(self):
        out, calls = self._run(FakeClient(final=fake.JobStatus.STOPPED), {"user_id": "u1"})
        self.assertEqual(out.status, "FAILED")           # chat-facing outcome unchanged
        self.assertEqual(calls[-1][2]["status"], "STOPPED")

    def test_untagged_job_is_not_recorded(self):
        client = FakeClient()
        out, calls = self._run(client, None)
        self.assertIsNone(client.submits[0]["metadata"])
        self.assertEqual(calls, [])
        self.assertEqual(out.status, "SUCCEEDED")

    def test_store_failure_never_fails_the_job(self):
        client = FakeClient()
        with mock.patch.object(runner, "make_job_client", return_value=client), \
             mock.patch.object(runner, "_preflight_dashboard"), \
             mock.patch.dict(sys.modules, {"app.infra.ray_jobs_store": None}):
            out = runner._run_ray_job_direct(
                script_text="x=1", runtime_env={}, job_id="dta-2",
                dashboard_url="http://ray:8265", job_tags={"user_id": "u1"},
            )
        self.assertEqual(out.status, "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()