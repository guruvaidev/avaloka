"""Cloud identity must never be assumed.

The bug these pin: model serving read `os.getenv("GCP_PROJECT_ID")`,
so an operator who set nothing did not get an error —
they got a deployment aimed at the project Avaloka was first built in, failing
with a 403 naming a project they had never heard of.
"""

import pytest

from app.core import cloud_config as cc

CLOUD_ENV = (
    "CLOUD_PROVIDER", "CONTAINER_REGISTRY",
    "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS",
    "GCP_REGION", "GCP_ZONE",
    "AWS_ACCOUNT_ID", "AWS_ACCESS_KEY_ID", "AWS_REGION", "AWS_DEFAULT_REGION",
    "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ACCOUNT",
    "AZURE_LOCATION", "AZURE_CONTAINER_REGISTRY",
    "INFERENCE_SERVICE_DOCKER_IMAGE", "RAY_TRAINING_DOCKER_URI",
)


@pytest.fixture(autouse=True)
def clean_cloud_env(monkeypatch):
    for name in CLOUD_ENV:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# The regression this module exists for
# --------------------------------------------------------------------------- #

def test_no_private_project_is_ever_returned(monkeypatch):
    """The old default must not survive anywhere in the resolution path."""
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "customer-project-123")
    assert cc.project_id() == "customer-project-123"
    assert "ai-playground" not in cc.container_registry()


def test_unset_project_raises_and_names_the_variable(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    with pytest.raises(cc.MissingCloudConfig) as exc:
        cc.project_id()
    message = str(exc.value)
    assert "GCP_PROJECT_ID" in message, "the error must say what to set"
    assert "ai-playground" not in message


def test_source_tree_has_no_hardcoded_project_default():
    """A literal default for identity is the bug; keep it from coming back."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders = []
    pattern = re.compile(r'getenv\(\s*["\'][A-Z_]*PROJECT[A-Z_]*["\']\s*,\s*["\'][^"\']+["\']')
    for path in root.rglob("*.py"):
        # cloud_config's own docstring quotes the old buggy line as the example
        # of what not to do; it is the definition of the rule, not a breach.
        if path.name == "cloud_config.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root.parent)}:{lineno}")
    assert not offenders, (
        "A project id must never have a literal default — it addresses someone "
        f"else's account: {offenders}"
    )


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #

def test_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-gcp-project")
    assert cc.provider() == "aws"


def test_unsupported_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "digitalocean")
    with pytest.raises(cc.MissingCloudConfig) as exc:
        cc.provider()
    assert "digitalocean" in str(exc.value)


def test_provider_inferred_from_a_single_cloud_credential(monkeypatch):
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-1")
    assert cc.provider() == "azure"


def test_two_clouds_present_is_an_error_not_a_coin_flip(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "p")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "1234")
    with pytest.raises(cc.AmbiguousCloudProvider):
        cc.provider()


def test_no_credentials_means_local():
    assert cc.provider() == "local"


def test_local_needs_no_project_or_registry():
    assert cc.project_id() is None
    assert cc.container_registry() is None


# --------------------------------------------------------------------------- #
# Registries follow the provider
# --------------------------------------------------------------------------- #

def test_gcp_registry(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "acme-prod")
    assert cc.container_registry() == "gcr.io/acme-prod"


def test_aws_registry_is_account_and_region_scoped(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert cc.container_registry() == "123456789012.dkr.ecr.eu-west-1.amazonaws.com"


def test_aws_registry_requires_a_region(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "aws")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "123456789012")
    with pytest.raises(cc.MissingCloudConfig) as exc:
        cc.container_registry()
    assert "AWS_REGION" in str(exc.value)


def test_azure_registry(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-1")
    monkeypatch.setenv("AZURE_CONTAINER_REGISTRY", "avalokaacr")
    assert cc.container_registry() == "avalokaacr.azurecr.io"


def test_explicit_registry_overrides_derivation(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "acme-prod")
    monkeypatch.setenv("CONTAINER_REGISTRY", "europe-docker.pkg.dev/acme/images/")
    assert cc.container_registry() == "europe-docker.pkg.dev/acme/images"


# --------------------------------------------------------------------------- #
# Image resolution
# --------------------------------------------------------------------------- #

def test_image_is_built_from_the_providers_registry(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "acme-prod")
    assert cc.image("X_IMAGE", "inference-service-image", "v1") == (
        "gcr.io/acme-prod/inference-service-image:v1")


def test_image_env_override_wins_outright(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "acme-prod")
    monkeypatch.setenv("INFERENCE_SERVICE_DOCKER_IMAGE", "harbor.internal/team/svc:2.1")
    assert cc.image("INFERENCE_SERVICE_DOCKER_IMAGE", "inference-service-image",
                    "v1") == "harbor.internal/team/svc:2.1"


def test_local_images_are_bare_names_for_side_loading():
    assert cc.image("X_IMAGE", "avaloka-inference", "latest") == "avaloka-inference:latest"


# --------------------------------------------------------------------------- #
# Agreement with the provisioning layer
# --------------------------------------------------------------------------- #

def test_supported_providers_match_the_cluster_factory():
    """Serving and provisioning must not drift into different cloud support."""
    from app.infra.providers.factory import SUPPORTED_PLATFORMS
    assert set(cc.SUPPORTED_PROVIDERS) == set(SUPPORTED_PLATFORMS)


def test_describe_never_raises_on_an_unconfigured_environment():
    out = cc.describe()
    assert out["provider"] == "local"
    assert set(out) == {"provider", "project_id", "region", "container_registry"}
