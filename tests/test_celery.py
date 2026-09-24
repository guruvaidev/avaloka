import io
import json
import os
import socket
import time
from datetime import datetime
from urllib.parse import urlparse

import pytest
import requests

API_URL = os.getenv("AVALOKA_API_URL", "http://localhost:8010")
HEADERS = {"X-User-Id": "test_user"}

# This suite drives the real scheduling stack end to end: it uploads a dataset
# to a running Avaloka API, asks the planner to schedule a recurring task, and
# then reads that task back through Celery + Redis + RedBeat. None of that can
# be stood up in-process, so with nothing listening every test here failed at
# the module fixture with a bare ConnectionError -- reported as five broken
# tests rather than as an absent service.
#
# Marked `integration` so the hermetic CI stage excludes it by marker, and
# guarded by a probe so that running this file directly still exercises the
# real stack for anyone who has it up (docker compose up, then pytest
# tests/test_celery.py). Point AVALOKA_API_URL elsewhere if the server is not
# on the default port.


def _api_is_listening(url: str, timeout: float = 1.0) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _api_is_listening(API_URL),
        reason=(
            f"no Avaloka API listening on {API_URL} — this suite needs the "
            "server plus Celery/Redis (docker compose up), and "
            "AVALOKA_API_URL can point it elsewhere"
        ),
    ),
]
CSV_FILE_CONTENT = """OrderID,OrderDate,CustomerID,Region,Product,Quantity,UnitPrice,Revenue
1,South,Laptop,2,740.81,1481.62
2,East,Laptop,1,843.38,843.38
3,,Smartphone,2,190.57,381.14
4,West,Monitor,1,315.89,315.89

"""

@pytest.fixture(scope="module")
def test_data():
    csv_content = io.BytesIO(CSV_FILE_CONTENT.encode("utf-8"))

    r = requests.post(
        API_URL + "/api/upload",
        headers=HEADERS,
        files={"file": ("x.csv", csv_content, "text/csv")},
    )

    assert r.status_code == 200
    data = r.json()

    return {
        "dataset_id": data["dataset_id"],
        "thread_id": data["thread_id"],
        "session_id": data["session_id"],
        "task_id": None
    }

def test_schedule_task(test_data):
    print("--------------Schedule Test--------------")
    message = "Group by Region then sum the Revenue. Schedule this to run every minute"
    r = requests.post(
        f"{API_URL}/threads/{test_data['thread_id']}/messages",
        headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']},
        json={
            "role": "user",
            "content": message,
            "metadata": {"dataset_id": test_data['dataset_id']},
            "stream": False,
        },
    )

    assert r.status_code == 200, r.text
    
    data = r.json()
    task_info = data.get("task_info") or {}

    assert task_info.get("task_id") is not None, "Task ID is none"
    assert task_info['next_due_at'] < 62

    test_data["task_id"] = task_info["task_id"]

def test_check_task(test_data):
    print("--------------Status Test--------------")
    if test_data["task_id"] is None:
        pytest.skip("Task ID is none")
    TASK_ID = test_data['task_id']

    while True:
        with requests.get(
            f"{API_URL}/tasks/{TASK_ID}/status",
            headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']},
            stream=True
        ) as r:
            assert r.status_code == 200, r.text

            now = time.time()
            for raw_data in r.iter_lines():
                raw_data: str = raw_data.decode()
                if raw_data == "event: done":
                    break
                if raw_data == "event: message" or not raw_data.startswith("data: "):
                    continue

                data = json.loads(raw_data.split("data: ")[1])
                assert data.get("status") in ("SCHEDULED", "REVOKED", "SUCCESS", "FAILED", "PENDING")

                if data.get("status") == "SCHEDULED":
                    print(f"Sleeping for {data['next_due_at']}")
                    time.sleep(data["next_due_at"])
                    continue

                if data.get("status") == "REVOKED":
                    print(f"Sleeping for {data['next_due_at']}")
                    assert data["next_due_at"] < 62, data["next_due_at"] 
                    return

                if data.get("status") == "SUCCESS":
                    print(f"Task {TASK_ID} finished successfully:", data["result"])
                    assert data["next_due_at"] < 62, data["next_due_at"] 
                    return

                if data.get("status") == "FAILED":
                    print(f"Task {TASK_ID} failed:", data["traceback"])
                    assert data["next_due_at"] < 62, data["next_due_at"] 
                    return
                
                print(f"Task {TASK_ID} is still in progress...\n{data}")
        if time.time() - now > 1800:
            print("Test exceeded 3 minutes")
            break

def test_task_list(test_data):
    print("--------------Task List Test--------------")
    if test_data["task_id"] is None:
        pytest.skip("Task ID is none")
    TASK_ID = test_data['task_id']

    r = requests.get(
        f"{API_URL}/tasks",
        headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']}
    )

    tasks = r.json()
    assert len(tasks) == 1
    assert tasks[-1]["id"] == TASK_ID, str(tasks[-1])
    assert tasks[-1]["total_run_count"] == 1, str(tasks[-1])

def test_query_task_info(test_data):
    if test_data["task_id"] is None:
        pytest.skip("Task ID is none")
    TASK_ID = test_data['task_id']

    max_timestamp = int(datetime.now().timestamp()) + 60

    r = requests.get(
        f"{API_URL}/tasks/{TASK_ID}/info",
        headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']}
    )

    task_info = r.json()
    assert task_info["total_run_count"] == 1

    assert_schedule = {
        "month_of_year": "*",
        "day_of_month": "*",
        "day_of_week": "*",
        "hour": "*",
        "minute": "*",
    }
    assert task_info["schedule"] == assert_schedule, "Schedule is incorrect"

def test_cancel_task(test_data):
    print("--------------Cancel Test--------------")
    if test_data["task_id"] is None:
        pytest.skip("Task ID is none")
    TASK_ID = test_data['task_id']

    r = requests.delete(
        f"{API_URL}/tasks/{TASK_ID}",
        headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']}
    )

    data = r.json()
    assert data["status"] == "CANCELLED", str(data)

    r = requests.get(
        f"{API_URL}/tasks",
        headers={**HEADERS, "X-Avaloka-Session": test_data['session_id']}
    )

    tasks = r.json()
    assert len(tasks) == 0 or tasks[-1] != TASK_ID