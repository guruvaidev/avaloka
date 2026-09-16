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
            job_id = job_data["job_id"]

            return job_id
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= max_retries:
                raise e
            time.sleep(10)
