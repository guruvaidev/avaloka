# app/infra/providers/factory.py
"""Resolve a provider key ("local"|"gcp"|"aws"|"azure") to a ClusterProvider instance.

Only ``local`` is available in an open-source build. Creating and paying for
managed cloud clusters on a customer's behalf is what the commercial editions
do, and ``cloud_provisioning`` is marked ``COMMERCIAL`` in
:mod:`app.core.editions` accordingly. That gate used to be declared and never
checked: the GKE, EKS and AKS providers shipped in the open-source
distribution and ``get_provider("gcp")`` returned one. This module is where the
declaration and the behaviour are reconciled.
"""
from __future__ import annotations

import os

from app.infra.providers.base import ClusterProvider

SUPPORTED_PLATFORMS = ("local", "gcp", "aws", "azure")

#: Platforms that stand up managed cloud infrastructure, and so bill the user's
#: cloud account. Gated on ``cloud_provisioning``.
CLOUD_PLATFORMS = ("gcp", "aws", "azure")

_COMMERCIAL_MESSAGE = (
    "Provisioning a managed {platform} cluster is a Professional and Enterprise "
    "capability, and this is an open-source build.\n\n"
    "The open-source edition targets a laptop or a small Kubernetes cluster you "
    "run yourself. Use platform 'local' (kind), or point Avaloka at a cluster "
    "you have already provisioned — 'connect' works against any cluster your "
    "kubeconfig can reach, including a cloud one, and distributed Ray execution "
    "and the scheduler are open-source capabilities on it.\n\n"
    "For provisioned and autoscaled clusters, see https://avaloka.ai or email "
    "support@avaloka.ai."
)


class CommercialCapabilityRequired(RuntimeError):
    """Raised when an open-source build is asked for a commercial capability."""


def get_provider(platform: str) -> ClusterProvider:
    if platform == "local":
        from app.infra.providers.local_kind import LocalKindProvider
        return LocalKindProvider()

    if platform in CLOUD_PLATFORMS:
        from app.core.editions import resolve_capabilities

        capabilities = resolve_capabilities(
            os.getenv("AVALOKA_DEPLOYMENT"), os.getenv("AVALOKA_EDITION")
        )
        if not capabilities.cloud_provisioning:
            raise CommercialCapabilityRequired(
                _COMMERCIAL_MESSAGE.format(platform=platform.upper())
            )

    if platform == "gcp":
        from app.infra.providers.gcp_gke import GcpGkeProvider
        return GcpGkeProvider()
    if platform == "aws":
        from app.infra.providers.aws_eks import AwsEksProvider
        return AwsEksProvider()
    if platform == "azure":
        from app.infra.providers.azure_aks import AzureAksProvider
        return AzureAksProvider()
    raise ValueError(
        f"Unsupported platform: {platform!r}. Supported: {', '.join(SUPPORTED_PLATFORMS)}."
    )
