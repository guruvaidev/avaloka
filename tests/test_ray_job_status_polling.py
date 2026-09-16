from types import SimpleNamespace

from app.agents.mta_v2.gcp import wait_for_ray_job_completion as polling


def test_get_status_distinguishes_missing_job(monkeypatch):
    response = SimpleNamespace(status_code=404)
    monkeypatch.setattr(polling.requests, "get", lambda *args, **kwargs: response)

    assert polling.get_ray_job_status("http://ray:8265", "missing") == "NOT_FOUND"


def test_wait_stops_after_three_consecutive_missing_responses(monkeypatch):
    statuses = iter(["UNKNOWN", "NOT_FOUND", "NOT_FOUND", "NOT_FOUND"])
    monkeypatch.setattr(polling, "get_ray_job_status", lambda *_: next(statuses))
    monkeypatch.setattr(polling.time, "sleep", lambda *_: None)

    assert polling.wait_for_ray_job_completion("http://ray:8265", "lost", 100) == "NOT_FOUND"


def test_one_missing_response_does_not_end_a_recovering_job(monkeypatch):
    statuses = iter(["NOT_FOUND", "RUNNING", "SUCCEEDED"])
    monkeypatch.setattr(polling, "get_ray_job_status", lambda *_: next(statuses))
    monkeypatch.setattr(polling.time, "sleep", lambda *_: None)

    assert polling.wait_for_ray_job_completion("http://ray:8265", "job", 10) == "SUCCEEDED"
