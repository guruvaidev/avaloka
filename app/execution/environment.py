"""Cloud-environment awareness — one detection point, many consumers.

The vision requires Avaloka to *behave with the knowledge of the cloud it runs
in*: GCP primitives on GCP, AWS on AWS, Azure on Azure, and sane local/on-prem
behaviour otherwise. Rather than scatter ``if platform == "gcp"`` branches, this
module resolves the environment **once** and exposes it as structured knowledge
that three consumers share:

  * the **execution router** (which full-data engine / registry / object store);
  * the **workload estimator** (cloud knowledge is one input, not the point);
  * the **agents** (a cloud-specific system-prompt fragment + service catalog so
    the Planner/Coder reason in the right primitives).

Detection order (cheapest, most explicit first):

  1. explicit override — ``AVALOKA_CLOUD=gcp|aws|azure|local``;
  2. hermetic env-var fingerprints (no network) — the deployment injects these;
  3. optional instance-metadata probe (IMDS / metadata server), off by default
     so tests and local runs never touch the network or hang;
  4. fallback → on-prem / local.

Everything here is dependency-free and network-free unless ``allow_network`` is
explicitly requested, which keeps it unit-testable without provisioning cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CloudProvider(str, Enum):
    GCP = "gcp"
    AWS = "aws"
    AZURE = "azure"
    LOCAL = "local"  # on-prem / laptop / kind / k3s


# Per-cloud service map. This is the "environment knowledge" the agents reason
# with — object store, warehouse, batch/Spark engine, container registry, secret
# manager, workload-identity mechanism, and the managed-LLM a BYOC customer in
# that cloud will typically expect (matters for the LLM-provider abstraction).
_SERVICE_CATALOG: Dict[CloudProvider, Dict[str, object]] = {
    CloudProvider.GCP: {
        "object_store": "gs://",
        "object_store_name": "Google Cloud Storage",
        "warehouse": "BigQuery",
        "batch_engine": "Dataproc / Dataproc Serverless",
        "container_registry": "Artifact Registry",
        "secret_manager": "Secret Manager",
        "workload_identity": "Workload Identity",
        "managed_llm": "Vertex AI",
        "managed_k8s": "GKE",
    },
    CloudProvider.AWS: {
        "object_store": "s3://",
        "object_store_name": "Amazon S3",
        "warehouse": "Redshift",
        "batch_engine": "EMR / Glue",
        "container_registry": "ECR",
        "secret_manager": "Secrets Manager",
        "workload_identity": "IRSA",
        "managed_llm": "Amazon Bedrock",
        "managed_k8s": "EKS",
    },
    CloudProvider.AZURE: {
        "object_store": "abfss://",
        "object_store_name": "ADLS Gen2",
        "warehouse": "Synapse",
        "batch_engine": "Synapse Spark / HDInsight",
        "container_registry": "ACR",
        "secret_manager": "Key Vault",
        "workload_identity": "Managed Identity",
        "managed_llm": "Azure OpenAI",
        "managed_k8s": "AKS",
    },
    CloudProvider.LOCAL: {
        "object_store": "file://",
        "object_store_name": "local filesystem",
        "warehouse": "DuckDB / local",
        "batch_engine": "local Spark / single-node",
        "container_registry": "local / private registry",
        "secret_manager": "environment / .env (dotenv)",
        "workload_identity": "static credentials",
        "managed_llm": "self-hosted / Groq",
        "managed_k8s": "kind / k3s",
    },
}

# URI schemes that pin a mission's source to a specific cloud regardless of where
# Avaloka itself is running (a mission over s3:// is an AWS-flavoured mission).
_SCHEME_PROVIDER = {
    "gs": CloudProvider.GCP,
    # fsspec/gcsfs registers "gcs" as an alias of "gs", so storage happily reads a
    # gcs:// URI. Without it here the router silently fell through to LOCAL.
    "gcs": CloudProvider.GCP,
    "s3": CloudProvider.AWS,
    "s3a": CloudProvider.AWS,
    "abfs": CloudProvider.AZURE,
    "abfss": CloudProvider.AZURE,
    "adl": CloudProvider.AZURE,
    "wasb": CloudProvider.AZURE,
    "wasbs": CloudProvider.AZURE,
    "file": CloudProvider.LOCAL,
}


@dataclass
class CloudEnvironment:
    """Resolved environment knowledge shared by router, estimator and agents."""

    provider: CloudProvider
    source: str  # how we decided: "override" | "env" | "metadata" | "default"
    region: Optional[str] = None
    service_catalog: Dict[str, object] = field(default_factory=dict)

    @property
    def is_cloud(self) -> bool:
        return self.provider is not CloudProvider.LOCAL

    def system_prompt_fragment(self) -> str:
        """Cloud-specific context injected into agent state (agent-awareness)."""
        c = self.service_catalog
        if self.provider is CloudProvider.LOCAL:
            return (
                "You are running on-premises / locally. Prefer local execution, "
                "single-node engines, and filesystem paths; do not assume any "
                "managed cloud service is available."
            )
        return (
            f"You are running on {self.provider.value.upper()}"
            + (f" in region {self.region}" if self.region else "")
            + ". Reason in this cloud's native primitives: object storage "
            f"{c.get('object_store_name')} ({c.get('object_store')}), warehouse "
            f"{c.get('warehouse')}, batch engine {c.get('batch_engine')}, "
            f"registry {c.get('container_registry')}, secrets via "
            f"{c.get('secret_manager')}, identity via {c.get('workload_identity')}, "
            f"and managed LLM {c.get('managed_llm')}. Do not reference other clouds' "
            "services unless the mission's data explicitly lives there."
        )


def _from_env_fingerprint() -> Optional[tuple]:
    """Detect the cloud from injected env vars — hermetic, no network."""
    env = os.environ
    # GCP: Cloud Run / GKE / functions / gcloud all set one of these.
    if any(k in env for k in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID", "GCLOUD_PROJECT", "K_SERVICE")):
        return CloudProvider.GCP, env.get("GOOGLE_CLOUD_REGION") or env.get("FUNCTION_REGION")
    # Azure before AWS: Azure App Service / Functions / AKS.
    if any(k in env for k in ("AZURE_SUBSCRIPTION_ID", "MSI_ENDPOINT", "IDENTITY_ENDPOINT", "WEBSITE_SITE_NAME", "AKS_CLUSTER_NAME")):
        return CloudProvider.AZURE, env.get("REGION_NAME") or env.get("AZURE_REGION")
    # AWS: Lambda / ECS / EKS / any aws CLI session.
    if any(k in env for k in ("AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME", "ECS_CONTAINER_METADATA_URI", "AWS_REGION", "AWS_DEFAULT_REGION")):
        return CloudProvider.AWS, env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    return None


def _from_metadata_probe(timeout: float = 0.3) -> Optional[tuple]:
    """Best-effort instance-metadata probe. Only called when allow_network=True.

    Uses very short timeouts and swallows every error so a probe on the wrong
    cloud (or on-prem) fails fast to the next candidate rather than hanging.
    """
    import urllib.request  # local import: never paid for unless probing

    def _get(url: str, headers: Dict[str, str]) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read().decode("utf-8", "ignore")
        except Exception:
            return None

    # GCP metadata server.
    if _get("http://metadata.google.internal/computeMetadata/v1/instance/zone",
            {"Metadata-Flavor": "Google"}):
        return CloudProvider.GCP, None
    # Azure IMDS.
    az = _get("http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text",
              {"Metadata": "true"})
    if az:
        return CloudProvider.AZURE, az.strip() or None
    # AWS IMDS (v1 read is enough to fingerprint the provider).
    if _get("http://169.254.169.254/latest/meta-data/placement/region", {}):
        return CloudProvider.AWS, None
    return None


def detect_environment(allow_network: bool = False) -> CloudEnvironment:
    """Resolve the cloud environment once. Cheap, explicit, hermetic by default."""
    override = os.environ.get("AVALOKA_CLOUD")
    if override:
        try:
            provider = CloudProvider(override.strip().lower())
            return CloudEnvironment(provider, "override",
                                    service_catalog=_SERVICE_CATALOG[provider])
        except ValueError:
            pass  # unknown value — fall through to real detection

    fp = _from_env_fingerprint()
    if fp:
        provider, region = fp
        return CloudEnvironment(provider, "env", region=region,
                                service_catalog=_SERVICE_CATALOG[provider])

    if allow_network:
        probed = _from_metadata_probe()
        if probed:
            provider, region = probed
            return CloudEnvironment(provider, "metadata", region=region,
                                    service_catalog=_SERVICE_CATALOG[provider])

    return CloudEnvironment(CloudProvider.LOCAL, "default",
                            service_catalog=_SERVICE_CATALOG[CloudProvider.LOCAL])


def provider_for_uri(uri: str) -> Optional[CloudProvider]:
    """Which cloud a data URI belongs to, by scheme (gs:// → GCP, s3:// → AWS…)."""
    if "://" not in uri:
        return None
    scheme = uri.split("://", 1)[0].lower()
    return _SCHEME_PROVIDER.get(scheme)


def resolve_cloud_target(
    pinned: Optional[str],
    source_uri: Optional[str] = None,
    env: Optional[CloudEnvironment] = None,
) -> CloudEnvironment:
    """Decide the cloud a mission runs against, most-explicit-wins.

    Priority: an explicit ``cloud_target`` on the mission > the cloud implied by
    the data's URI scheme > the detected runtime environment. This is what makes
    a plan environment-aware without the interfaces knowing anything about cloud.
    """
    if pinned:
        try:
            provider = CloudProvider(str(pinned).strip().lower())
            return CloudEnvironment(provider, "mission",
                                    service_catalog=_SERVICE_CATALOG[provider])
        except ValueError:
            pass
    if source_uri:
        by_uri = provider_for_uri(source_uri)
        if by_uri:
            return CloudEnvironment(by_uri, "source_uri",
                                    service_catalog=_SERVICE_CATALOG[by_uri])
    return env or detect_environment()
