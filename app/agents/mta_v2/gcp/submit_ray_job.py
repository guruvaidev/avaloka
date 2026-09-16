import time
import requests


def submit_ray_job(
    dashboard_url: str,
    env_vars: dict,
    entry_point: str = "python /app/ray_job.py",
    max_retries: int = 3,
) -> str:
    base_url = dashboard_url.rstrip("/")

    payload = {"entrypoint": entry_point, "runtime_env": {"env_vars": env_vars}}

    retries = 0
    while retries < max_retries:
        try:
            r = requests.post(f"{base_url}/api/jobs/", json=payload, timeout=30)
            r.raise_for_status()

            job_data = r.json()
            # Ray 2.53 returns ``submission_id``. Older Ray versions exposed
            # the same identifier as ``job_id``; accept both while clusters
            # are upgraded independently.
            job_id = job_data.get("submission_id") or job_data.get("job_id")
            if not job_id:
                raise RuntimeError(
                    "Ray Jobs API response did not include submission_id or job_id"
                )
            return job_id
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= max_retries:
                raise e
            time.sleep(10)
