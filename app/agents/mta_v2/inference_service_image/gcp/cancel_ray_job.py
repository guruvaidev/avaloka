import requests
import time

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}

def cancel_ray_job(
    dashboard_url: str,
    job_id: str,
    max_retries: int = 3,
) -> bool:
    base_url = dashboard_url.rstrip("/")

    try:
        r = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        status = r.json().get("status", "UNKNOWN")
        if status in TERMINAL_STATUSES:
            print(f"Job {job_id} is already in terminal state: {status}. Nothing to cancel.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch job status: {e}")

    for attempt in range(max_retries):
        try:
            r = requests.delete(f"{base_url}/api/jobs/{job_id}", timeout=10)
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries - 1:
                raise e
            time.sleep(5)

    return False