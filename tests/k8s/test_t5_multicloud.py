"""T5 — Multi-cloud matrix (cloud, slow).

Parameterised over providers using D6's ephemeral lifecycle. Gated behind
AVALOKA_TEST_ALLOW_CLOUD=1 plus the relevant credentials; skips loudly otherwise.
Each leg creates an ephemeral cluster -> deploys -> runs a health check -> destroys.

Cost/time: each cloud leg is ~25–45 min and bills real money. Prefer helm
template-level parity (T1.7) for routine runs; run these only on demand.
"""
from __future__ import annotations

import os

import pytest

ALLOW = os.getenv("AVALOKA_TEST_ALLOW_CLOUD") == "1"

pytestmark = [pytest.mark.cloud, pytest.mark.slow]

# (provider, extra-credential env that must be present)
LEGS = [
    ("gcp", "GCP_PROJECT_ID"),
    ("aws", "AWS_REGION"),
    ("azure", "AZURE_RESOURCE_GROUP"),
]


@pytest.mark.parametrize("provider,cred_env", LEGS)
def test_t5_provider_leg(provider, cred_env):
    if not ALLOW:
        pytest.skip("AVALOKA_TEST_ALLOW_CLOUD=1 not set — cloud legs are opt-in and billable")
    if not os.getenv(cred_env):
        pytest.skip(f"{cred_env} unset — no credentials for the {provider} leg")

    from tests.k8s.helpers.ephemeral import ephemeral_cluster
    from app.infra import deploy_stack
    from app.infra.providers.factory import get_provider

    # A short, deterministic runid (no Math.random/Date in this process is fine here;
    # a real run passes one in via the harness/env).
    runid = os.getenv("AVALOKA_TEST_RUNID", "manual")
    with ephemeral_cluster(provider, runid=runid):
        get_provider(provider).configure_kubectl()
        out = deploy_stack.deploy_avaloka(
            namespace="avaloka-test", service_type="LoadBalancer",
            image_pull_policy="Always",
        )
        assert out["status"] == "SUCCESS", out
    # ephemeral_cluster tears the cluster down unconditionally on exit.
