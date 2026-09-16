"""T4 — Agent workloads (integration).

The in-process agent contracts (coder/DTA/MTA) are already covered by the existing
suite (tests/test_coder_integration.py, tests/test_dta_planner_integration.py,
tests/test_mta_integration.py). This module adds the *k8s variants* the plan calls
for — the same contracts against a DEPLOYED backend — plus the DTA GKE render seam.

Gating:
  * GROQ_API_KEY_PLANNING_AGENT / _CODING_AGENT unset -> skip (real agents).
  * AVALOKA_API_URL unset -> deployed-API variants skip loudly.
"""
from __future__ import annotations

import io
import os

import pytest
import requests

GROQ = os.getenv("GROQ_API_KEY_PLANNING_AGENT") and os.getenv("GROQ_API_KEY_CODING_AGENT")
API_URL = os.getenv("AVALOKA_API_URL")
API_TOKEN = os.getenv("AVALOKA_API_TOKEN")

requires_groq = pytest.mark.skipif(not GROQ, reason="GROQ_API_KEY_* unset — real agents unavailable")
requires_api = pytest.mark.skipif(not API_URL, reason="AVALOKA_API_URL unset — no deployed backend")

pytestmark = pytest.mark.integration


def _auth():
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


# ---------------------------------------------------------------- T4a (deployed)
@requires_groq
@requires_api
@pytest.mark.parametrize("prompt", [
    "Read the CSV file and save it to the output location.",
    "Read the sales data, filter for sales in the 'USA', and save the result.",
    "Calculate the total price of products for each country and save the result.",
    "First filter the sales data for orders in the 'USA'. Then group the remaining "
    "rows by Category and calculate the total Price.",
])
def test_t4a_coder_on_deployed_api(prompt):
    """The deployed backend generates coder code for the verbatim suite prompts."""
    csv = b"Country,Category,Price\nUSA,A,10\nUSA,B,20\nUK,A,30\n"
    up = requests.post(f"{API_URL}/api/upload", headers=_auth(),
                       files={"file": ("sales_data.csv", io.BytesIO(csv), "text/csv")}, timeout=60)
    assert up.status_code == 200, up.text
    thread_id = up.json()["thread_id"]
    msg = requests.post(f"{API_URL}/threads/{thread_id}/messages", headers=_auth(),
                        json={"role": "user", "content": prompt}, timeout=180)
    assert msg.status_code == 200, msg.text
    code = (msg.json().get("coder_definition") or {}).get("code", "")
    assert code and "import pandas" in code, f"no coder code for prompt: {prompt!r}"


# ---------------------------------------------------------------- T4c (deployed)
@requires_groq
@requires_api
def test_t4c_mta_flagship_on_deployed_api():
    """MTA on the deployed backend: upload a training CSV, send a training-intent
    message, and prove the deployed API exposes the MTA response contract and runs
    the real graph end to end.

    Note: whether a single bare message populates ``training_task`` is a *planner*
    decision (enable_training is set in app/agents/planner.py, not from message
    keywords) and depends on dataset/target context + the LLM — so it is not
    deterministic on turn 1. We assert the reliable contract: the request is
    processed by the real graph and the MTA response fields are present; if the
    planner did commit to training, we additionally check the pending-task shape.
    """
    csv = b"feature1,feature2,feature3,target\n1,2,3,0\n4,5,6,1\n7,8,9,0\n"
    up = requests.post(f"{API_URL}/api/upload", headers=_auth(),
                       files={"file": ("train_data.csv", io.BytesIO(csv), "text/csv")}, timeout=60)
    assert up.status_code == 200, up.text
    thread_id = up.json()["thread_id"]
    msg = requests.post(f"{API_URL}/threads/{thread_id}/messages", headers=_auth(),
                        json={"role": "user", "content": "I want to train a machine learning model"},
                        timeout=300)
    assert msg.status_code == 200, msg.text
    body = msg.json()
    # The deployed API exposes the MTA response contract (keys present even if null).
    for key in ("training_task", "ready_to_train", "training_status"):
        assert key in body, f"deployed ChatResponse missing MTA field {key!r}: {sorted(body)}"
    # If the planner enabled training this turn, the pending-task shape must hold.
    if body.get("training_task") is not None:
        assert body.get("ready_to_train") is True
        assert body.get("training_status") == "pending"


# ---------------------------------------------------------------- T4b DTA render seam
@pytest.mark.cluster
@pytest.mark.kuberay
def test_t4b_dta_rayjob_dry_run_server():
    """With live CRDs, the DTA RayJob template validates server-side."""
    from pathlib import Path
    from tests.k8s.helpers import k8s

    if k8s.kubectl("get", "crd", "rayjobs.ray.io", check=False).returncode != 0:
        pytest.skip("rayjobs.ray.io CRD not installed (run T3 first)")
    manifest = Path(__file__).resolve().parents[2] / "app/infra/rayjob_data_transfer.yaml"
    if not manifest.exists():
        pytest.skip(f"{manifest} not found")

    # The file is a TEMPLATE with __PLACEHOLDER__ tokens that ray_job_runner fills in;
    # substitute them (as the runner does) so the RENDERED RayJob+ConfigMap is what we
    # validate against the live CRDs. __INJECTION_SCRIPT__ is a ConfigMap block-scalar
    # body, so it must be indented under `sample_code.py: |`.
    import os
    import tempfile

    rendered = (manifest.read_text()
                .replace("__RAYJOB_NAME__", "dta-dryrun")
                .replace("__JOB_ID__", "dryrun-1")
                .replace("__GCS_SECRET_NAME__", "gcs-sa-key")
                .replace("__INJECTION_SCRIPT__", "    print('ok')"))
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    Path(tmp).write_text(rendered)
    try:
        r = k8s.kubectl_ns("apply", "--dry-run=server", "-f", tmp, check=False)
    finally:
        os.unlink(tmp)
    assert r.returncode == 0, f"server dry-run failed:\n{r.stdout}\n{r.stderr}"
