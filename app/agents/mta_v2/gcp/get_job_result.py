import requests
import time
import re
import json
import logging


def get_job_logs(
    dashboard_url: str, job_id: str, retries: int = 3, delay: int = 5
) -> str:
    """Return the Ray job's raw log output for failure diagnostics."""
    for attempt in range(retries):
        try:
            response = requests.get(f"{dashboard_url}/api/jobs/{job_id}/logs", timeout=10)
            response.raise_for_status()
            return str(response.json().get("logs", ""))
        except requests.exceptions.RequestException as exc:
            logging.warning(
                "Unable to retrieve logs for Ray job %s (attempt %s/%s): %s",
                job_id,
                attempt + 1,
                retries,
                exc,
            )
            if attempt < retries - 1:
                time.sleep(delay)
    return ""


def get_job_result(
    dashboard_url: str, job_id: str, retries: int = 3, delay: int = 5
) -> dict | None:
    for i in range(retries):
        try:
            logs = get_job_logs(dashboard_url, job_id, retries=1, delay=delay)
            if not logs:
                time.sleep(delay)
                continue
            pattern = r"```json\s*(.*?)\s*```"
            result_data = logs.strip()
            matche = re.findall(pattern, result_data, re.DOTALL)
            if matche:
                result_data = matche[0].strip()
            else:
                result_data = ""

            if result_data:
                return json.loads(result_data)
            else:
                return None
        except requests.exceptions.RequestException as e:
            time.sleep(delay)
        except (json.JSONDecodeError, IndexError) as e:
            return None
    return None
