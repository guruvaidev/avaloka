import requests
import time


def get_ray_job_status(dashboard_url: str, job_id: str):
    base_url = dashboard_url.rstrip("/")
    try:
        r = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=10)
        if r.status_code == 404:
            return "NOT_FOUND"
        r.raise_for_status()

        data = r.json()
        return data.get("status")
    except (requests.RequestException, ValueError, TypeError):
        return "UNKNOWN"


def wait_for_ray_job_completion(dashboard_url: str, job_id: str, max_retries: int = 60):
    terminal_statuses = ["SUCCEEDED", "FAILED", "STOPPED"]

    retries = 0
    consecutive_not_found = 0
    while retries < max_retries:
        status = get_ray_job_status(dashboard_url, job_id)
        if status in terminal_statuses:
            return status
        if status == "NOT_FOUND":
            consecutive_not_found += 1
            # A single 404 can occur during dashboard propagation. Repeated
            # 404s mean the head restarted or the submission disappeared, so
            # waiting for the full training timeout cannot recover the job.
            if consecutive_not_found >= 3:
                return status
        else:
            consecutive_not_found = 0

        retries += 1
        time.sleep(1)
    else:
        return "UNKNOWN"
