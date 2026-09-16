
# app/infra/run_ray_smoketest.py
import os
import time
import uuid
import re
from pathlib import Path

from app.agents.execution_agent import execution_agent_node_ray
from app.infra.rayjob_submitter import submit_ray_job_sync


def run_dryrun():
    # Force the execution_agent_node_ray DRYRUN path
    os.environ["RAYJOB_DRYRUN"] = "1"
    os.environ.setdefault("RAYJOB_DRYRUN_OUTPUT_PATH", str(Path.cwd() / "rayjob.out.yaml"))

    state = {
        "execution_mode": "k8s-ray",
        "ray_namespace": "ray-jobs",
        "ray_workers": 3,
        "ray_timeout_s": 1800,

        # easiest first test (no creds needed for read):
        "data_source_location_cloud": "https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv",

        "coder_definition": {
            "code": """
import pandas as pd
print("data_source_location =", data_source_location)
df = pd.read_csv(data_source_location)
print(df.head())
"""
        },

        "ray_job_name": f"avaloka-dryrun-{int(time.time())}",
        "messages": [],
    }

    out_state = execution_agent_node_ray(state)
    out_path = os.environ["RAYJOB_DRYRUN_OUTPUT_PATH"]
    print(f"\n[DRYRUN] wrote YAML to: {out_path}")
    print("execution_result.status =", out_state.get("execution_result", {}).get("status"))


def run_real():
    connection_id = os.getenv("AVALOKA_CONNECTION_ID", "00237d04-1ffc-4193-bb9a-5b7b98d9cc92")
    dataset_id = str(uuid.uuid4())
    ray_ns = "ray-jobs"

    template_path = str(Path(__file__).resolve().parent / "rayjob.yaml")

    data_source_uri = os.getenv(
        "AVALOKA_DATA_SOURCE_URI",
        "s3://avaloka-s3-testing-bucket/test-input-data/b4f50d53-4cf3-41c8-a3e3-02735abf312c.csv",
    )

    out = submit_ray_job_sync(
        connection_id=connection_id,
        dataset_id=dataset_id,
        data_source_uri=data_source_uri,
        ray_ns=ray_ns,
        template_path=template_path,
    )

    print("\n===== OUTCOME =====")
    print("status =", out.status)
    print("rayjob_name =", out.rayjob_name)
    print("namespace =", out.namespace)
    print("job_status =", out.job_status)
    print("deployment_status =", out.deployment_status)
    print("runtime_s =", out.runtime_s)

    print("\n===== RAW LOGS (tail) =====")
    logs = re.sub(r"\x1b\[[0-9;]*m", "", out.logs or "")
    print(logs)


if __name__ == "__main__":
    # choose mode by env var
    if os.getenv("SMOKE_DRYRUN", "0") == "1":
        run_dryrun()
    else:
        run_real()
