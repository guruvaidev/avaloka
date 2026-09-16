import requests
import time
import re
import json
import logging

def get_job_result(
    dashboard_url: str, job_id: str, retries: int = 3, delay: int = 5
) -> dict | None:
    for i in range(retries):
        try:
            r = requests.get(f"{dashboard_url}/api/jobs/{job_id}/logs", timeout=10)
            r.raise_for_status()
            data = r.json()
            logs = data.get("logs", "")
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
