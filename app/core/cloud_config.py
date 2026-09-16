"""Cloud identity and image locations, resolved per provider — never guessed.

Avaloka provisions clusters on four platforms (``app/infra/providers/``), but the
model-serving and training path did not follow: project id, container registry
and image tags were read with **hardcoded defaults pointing at the project this
was first built in**::

    INFERENCE_SERVICE_GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    INFERENCE_SERVICE_DOCKER_IMAGE   = os.getenv(
        "INFERENCE_SERVICE_DOCKER_IMAGE",
        "gcr.io/INFERENCE_SERVICE_GCP_PROJECT_ID/inference-service-image:test-v0.0.9")

An operator who did not set those variables did not get an error. They got a
deployment quietly aimed at someone else's GCP project, pulling from a private
registry they cannot read, failing with a 403 that names a project they have
never heard of. For an open-source release that is worse than not starting.

The rule this module enforces: **a missing cloud parameter is an error that
names the variable, never a default that points somewhere private.** Nothing
here has a fallback value for identity. :class:`MissingCloudConfig` is raised
with the exact variable to set and the provider it applies to.

Provider selection is ``CLOUD_PROVIDER`` (``local`` | ``gcp`` | ``aws`` |
``azure``), matching ``app.infra.providers.factory.SUPPORTED_PLATFORMS`` so the
serving path and the provisioning path cannot disagree about where they are.
When it is unset the provider is *inferred* from whichever credentials are
actually present, and an ambiguous environment is an error rather than a coin
flip — being wrong about which cloud you are on is not a recoverable mistake.

Registries follow the provider, because image addresses are not portable:

===========  ====================================================
provider     registry
===========  ====================================================
``gcp``      ``gcr.io/<project>`` (or Artifact Registry via env)
``aws``      ``<account>.dkr.ecr.<region>.amazonaws.com``
``azure``    ``<registry-name>.azurecr.io``
``local``    no registry; images are side-loaded into kind
===========  ====================================================
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LOCAL = "local"
GCP = "gcp"
AWS = "aws"
AZURE = "azure"

#: Kept identical to ``app.infra.providers.factory.SUPPORTED_PLATFORMS``. A test
#: pins the two together so the serving path and the provisioning path can never
#: drift into supporting different clouds.
SUPPORTED_PROVIDERS: Tuple[str, ...] = (LOCAL, GCP, AWS, AZURE)

#: Which env var carries the account/project identity, per provider.
_PROJECT_ENV = {
    GCP: ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"),
    AWS: ("AWS_ACCOUNT_ID",),
    AZURE: ("AZURE_SUBSCRIPTION_ID",),
}

#: Credentials whose presence implies a provider, used only when CLOUD_PROVIDER
#: is unset.
_CREDENTIAL_HINTS = {
    GCP: ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"),
    AWS: ("AWS_ACCOUNT_ID", "AWS_ACCESS_KEY_ID", "AWS_REGION", "AWS_DEFAULT_REGION"),
    AZURE: ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACCOUNT"),
}


class MissingCloudConfig(RuntimeError):
    """A required cloud parameter is unset. The message names the variable."""


class AmbiguousCloudProvider(RuntimeError):
    """Credentials for more than one cloud are present and none was selected."""


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def provider() -> str:
    """The active cloud, from ``CLOUD_PROVIDER`` or inferred from credentials.

    Inference is a convenience for the single-cloud case, not a guess: if two
    clouds' credentials are present the environment is ambiguous and that is
    raised rather than resolved, because silently picking one deploys real
    infrastructure to the wrong account.
    """
    explicit = _env("CLOUD_PROVIDER")
    if explicit:
        normalised = explicit.lower()
        if normalised not in SUPPORTED_PROVIDERS:
            raise MissingCloudConfig(
                f"CLOUD_PROVIDER={explicit!r} is not supported. "
                f"Set one of: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        return normalised

    present = [p for p, hints in _CREDENTIAL_HINTS.items() if _env(*hints)]
    if len(present) == 1:
        logger.info("[cloud] CLOUD_PROVIDER unset; inferred %r from credentials", present[0])
        return present[0]
    if len(present) > 1:
        raise AmbiguousCloudProvider(
            "Credentials for more than one cloud are present "
            f"({', '.join(sorted(present))}). Set CLOUD_PROVIDER explicitly to "
            f"one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return LOCAL


def project_id(*, required: bool = True) -> Optional[str]:
    """The account/project this deployment belongs to. No default identity."""
    active = provider()
    if active == LOCAL:
        return None
    names = _PROJECT_ENV[active]
    value = _env(*names)
    if value:
        return value
    if not required:
        return None
    raise MissingCloudConfig(
        f"No project/account id configured for CLOUD_PROVIDER={active!r}. "
        f"Set {' or '.join(names)}. Avaloka does not assume a project — an "
        f"assumed one deploys to somebody else's account."
    )


def region(*, required: bool = False) -> Optional[str]:
    active = provider()
    value = {
        GCP: lambda: _env("GCP_REGION", "GCP_ZONE"),
        AWS: lambda: _env("AWS_REGION", "AWS_DEFAULT_REGION"),
        AZURE: lambda: _env("AZURE_LOCATION"),
        LOCAL: lambda: None,
    }[active]()
    if value or not required or active == LOCAL:
        return value
    raise MissingCloudConfig(
        f"No region configured for CLOUD_PROVIDER={active!r}. "
        f"Set {'GCP_REGION' if active == GCP else 'AWS_REGION' if active == AWS else 'AZURE_LOCATION'}."
    )


def container_registry(*, required: bool = True) -> Optional[str]:
    """Where images live for this provider.

    ``CONTAINER_REGISTRY`` overrides the derivation outright, which is what an
    operator using Artifact Registry, Harbor, or a mirror needs.
    """
    override = _env("CONTAINER_REGISTRY")
    if override:
        return override.rstrip("/")

    active = provider()
    if active == LOCAL:
        return None
    if active == GCP:
        return f"gcr.io/{project_id()}"
    if active == AWS:
        acct = project_id()
        aws_region = region(required=True)
        return f"{acct}.dkr.ecr.{aws_region}.amazonaws.com"
    if active == AZURE:
        name = _env("AZURE_CONTAINER_REGISTRY")
        if not name:
            if not required:
                return None
            raise MissingCloudConfig(
                "No registry configured for CLOUD_PROVIDER='azure'. Set "
                "AZURE_CONTAINER_REGISTRY (the ACR name) or CONTAINER_REGISTRY "
                "(a full registry host)."
            )
        return f"{name}.azurecr.io"
    return None


def image(env_var: str, repository: str, tag: str) -> str:
    """Resolve a container image address.

    ``env_var`` wins outright so an operator can pin any image without touching
    code. Otherwise the address is built from the provider's registry — never
    from a literal that embeds somebody else's project.
    """
    override = _env(env_var)
    if override:
        return override
    registry = container_registry()
    if not registry:
        # kind: images are side-loaded, so a bare name is the correct address.
        return f"{repository}:{tag}"
    return f"{registry}/{repository}:{tag}"


def describe() -> dict:
    """Diagnostics for ``install.sh --check`` and startup logs. Never raises."""
    def _safe(fn):
        try:
            return fn()
        except (MissingCloudConfig, AmbiguousCloudProvider) as exc:
            return f"UNSET ({type(exc).__name__})"
    return {
        "provider": _safe(provider),
        "project_id": _safe(lambda: project_id(required=False)),
        "region": _safe(lambda: region(required=False)),
        "container_registry": _safe(lambda: container_registry(required=False)),
    }
