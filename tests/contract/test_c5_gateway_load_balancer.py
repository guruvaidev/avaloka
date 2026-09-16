"""C5 — cloud Application Load Balancer (Gateway API) contract.

Clusterless, like the rest of the contract tier: values files are parsed with
``yaml.safe_load``; the templates carry ``{{ }}`` and are scanned line-wise.

What this pins is the set of mistakes that produce a *silently wrong* load
balancer — one that comes up green and still breaks the product:

  * a cloud overlay that keeps ``service.type: LoadBalancer`` hands out a second
    public IP straight to the pod, bypassing the Gateway's TLS, WAF and logs;
  * ``certificateRefs`` alongside GCP's certmap annotation is rejected outright by
    the GKE controller, and AWS ignores ``certificateRefs`` altogether;
  * the GCP default backend timeout is 30s, which 504s long agent requests while
    the pod is still working;
  * a hostname with no certificate behind it serves the wrong cert.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART_ROOT = REPO_ROOT / "deploy" / "helm" / "avaloka"
TEMPLATE_DIR = CHART_ROOT / "templates"
OVERLAY_DIR = CHART_ROOT / "values"

GATEWAY_TEMPLATE = TEMPLATE_DIR / "gateway.yaml"
GCP_POLICY_TEMPLATE = TEMPLATE_DIR / "gateway-policies-gcp.yaml"
AWS_CONFIG_TEMPLATE = TEMPLATE_DIR / "gateway-config-aws.yaml"
PREREQ_SCRIPT = REPO_ROOT / "deploy" / "gcp-lb-prereqs.sh"

# Overlays that front the release with a cloud load balancer, and the provider each
# one must declare. on-prem/minikube are deliberately absent: they have no managed
# ALB and keep their NodePorts.
LB_OVERLAYS = {
    "values-gke.yaml": "gcp",
    "values-eks.yaml": "aws",
    "values-aks.yaml": "azure",
}

# The production app.avaloka.ai release. Layered on top of values-gke.yaml rather
# than standing alone, so it is validated as a merge, not on its own.
APP_OVERLAY = "values-gke-app.yaml"
GKE_INSTANCES = ("values-gke.yaml", APP_OVERLAY)

# Anything below this and a mission/DTA request is cut off mid-flight by the load
# balancer rather than by the app.
MIN_API_TIMEOUT_SEC = 600


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_values(path: pathlib.Path) -> dict:
    data = yaml.safe_load(_read(path))
    return data if isinstance(data, dict) else {}


def _gateway(values: dict) -> dict:
    return values.get("gateway") or {}


@pytest.fixture(scope="module")
def base_values() -> dict:
    return _load_values(CHART_ROOT / "values.yaml")


# --------------------------------------------------------------- template shape
def test_c5_01_gateway_template_exists_and_is_values_gated() -> None:
    """The Gateway ships as an opt-in template so local/kind installs are untouched."""
    assert GATEWAY_TEMPLATE.is_file(), f"missing {GATEWAY_TEMPLATE}"
    head = _read(GATEWAY_TEMPLATE).lstrip().splitlines()[0]
    assert head.startswith("{{- if .Values.gateway.enabled }}"), (
        f"gateway.yaml is not gated on .Values.gateway.enabled: {head!r}"
    )


def test_c5_02_gateway_is_disabled_by_default(base_values: dict) -> None:
    """values.yaml ships the Gateway off — turning it on is an explicit cloud choice."""
    assert _gateway(base_values).get("enabled") is False


@pytest.mark.parametrize("kind", ["Gateway", "HTTPRoute"])
def test_c5_03_uses_the_ga_gateway_api_group(kind: str) -> None:
    """Gateway/HTTPRoute are the v1 (GA) types, not v1alpha2/v1beta1."""
    text = _read(GATEWAY_TEMPLATE)
    assert f"kind: {kind}" in text, f"gateway.yaml declares no {kind}"
    versions = set(re.findall(r"^apiVersion:\s*(gateway\.networking\.k8s\.io/\S+)\s*$", text, re.M))
    assert versions == {"gateway.networking.k8s.io/v1"}, f"non-GA Gateway API versions: {versions}"


def test_c5_04_http_listener_redirects_instead_of_serving() -> None:
    """Port 80 carries a RequestRedirect to https, bound to the http listener only."""
    text = _read(GATEWAY_TEMPLATE)
    assert "type: RequestRedirect" in text and "scheme: https" in text
    assert "sectionName: http" in text, "the redirect route is not pinned to the http listener"


# --------------------------------------------------------------- overlay wiring
@pytest.mark.parametrize("overlay,provider", sorted(LB_OVERLAYS.items()))
def test_c5_05_cloud_overlay_enables_the_gateway(overlay: str, provider: str) -> None:
    """Each cloud overlay turns the Gateway on and names its provider."""
    gateway = _gateway(_load_values(OVERLAY_DIR / overlay))
    assert gateway.get("enabled") is True, f"{overlay} does not enable the Gateway"
    assert gateway.get("provider") == provider, (
        f"{overlay} declares provider {gateway.get('provider')!r}, expected {provider!r}"
    )


@pytest.mark.parametrize("overlay", sorted(LB_OVERLAYS))
@pytest.mark.parametrize("path", [("service",), ("webui", "service")])
def test_c5_06_gateway_overlays_do_not_also_publish_a_service_ip(
    overlay: str, path: tuple[str, ...]
) -> None:
    """Behind a Gateway the backing Services are ClusterIP.

    A `LoadBalancer` Service here is the regression this whole change exists to
    remove: it provisions a second public address that reaches pods directly,
    unprotected by the load balancer's TLS termination, Cloud Armor policy or
    access logging — and nothing about the deployment looks broken.
    """
    values: object = _load_values(OVERLAY_DIR / overlay)
    for key in path:
        assert isinstance(values, dict)
        values = values.get(key) or {}
    assert isinstance(values, dict)
    service_type = values.get("type")
    assert service_type == "ClusterIP", (
        f"{overlay} sets {'.'.join(path)}.type={service_type!r}; behind a Gateway it must be "
        "ClusterIP or the pod gets its own public IP that bypasses the load balancer"
    )


@pytest.mark.parametrize("overlay", sorted(LB_OVERLAYS))
def test_c5_07_every_gateway_host_is_a_fqdn(overlay: str) -> None:
    """gateway.hosts are concrete FQDNs — a Gateway with no hostname matches nothing."""
    hosts = _gateway(_load_values(OVERLAY_DIR / overlay)).get("hosts") or []
    assert hosts, f"{overlay} enables the Gateway with no hosts"
    for host in hosts:
        assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9-]+)+", host), (
            f"{overlay} host {host!r} is not a plain lowercase FQDN"
        )


@pytest.mark.parametrize("overlay", sorted(LB_OVERLAYS))
def test_c5_08_tls_is_terminated_with_a_usable_certificate_source(overlay: str) -> None:
    """TLS is on and the mode's required field is filled in for that provider.

    The three providers disagree about where certificates live, and picking the
    wrong one fails at apply time (GCP) or silently serves nothing (AWS).
    """
    gateway = _gateway(_load_values(OVERLAY_DIR / overlay))
    tls = gateway.get("tls") or {}
    assert tls.get("enabled") is True, f"{overlay} disables TLS on a public load balancer"

    mode = tls.get("mode")
    provider = gateway.get("provider")
    if mode == "certmap":
        assert provider == "gcp", "certmap is a GCP Certificate Manager mechanism"
        assert tls.get("certMapName"), f"{overlay} uses certmap with no certMapName"
    elif mode == "secret":
        assert tls.get("secretName"), f"{overlay} uses mode=secret with no secretName"
    elif mode == "external":
        assert provider == "aws", (
            "mode=external exists for AWS, whose controller reads ACM ARNs from "
            "LoadBalancerConfiguration rather than certificateRefs"
        )
        assert (gateway.get("aws") or {}).get("certificateArns"), (
            f"{overlay} uses mode=external but supplies no gateway.aws.certificateArns, so the "
            "chart renders no LoadBalancerConfiguration and the listener has no certificate"
        )
    else:
        pytest.fail(f"{overlay} uses unknown gateway.tls.mode {mode!r}")


def test_c5_09_certmap_and_certificate_refs_are_mutually_exclusive() -> None:
    """certificateRefs renders only in mode=secret.

    GKE rejects a Gateway that carries both the networking.gke.io/certmap annotation
    and listeners[].tls.certificateRefs, and the AWS controller ignores
    certificateRefs entirely — so the template must emit them for neither.
    """
    text = _read(GATEWAY_TEMPLATE)
    assert 'if eq $g.tls.mode "secret"' in text, (
        "the certificateRefs block is not guarded on tls.mode == secret"
    )
    guard_at = text.index('if eq $g.tls.mode "secret"')
    refs_at = text.index("certificateRefs:")
    assert guard_at < refs_at, "certificateRefs is emitted outside the mode=secret guard"


# --------------------------------------------------------------- GCP specifics
def test_c5_10_gcp_policies_are_provider_scoped() -> None:
    """The networking.gke.io policies render only for provider=gcp."""
    assert GCP_POLICY_TEMPLATE.is_file(), f"missing {GCP_POLICY_TEMPLATE}"
    head = _read(GCP_POLICY_TEMPLATE).lstrip().splitlines()[0]
    assert '.Values.gateway.enabled' in head and 'eq .Values.gateway.provider "gcp"' in head, (
        f"GCP policy template is not scoped to provider=gcp: {head!r}"
    )


@pytest.mark.parametrize("kind", ["HealthCheckPolicy", "GCPBackendPolicy", "GCPGatewayPolicy"])
def test_c5_11_gcp_policy_kinds_are_present(kind: str) -> None:
    """All three GKE Gateway policy CRDs are wired, each on networking.gke.io/v1."""
    text = _read(GCP_POLICY_TEMPLATE)
    assert f"kind: {kind}" in text, f"{kind} is not rendered"
    versions = set(re.findall(r"^apiVersion:\s*(networking\.gke\.io/\S+)\s*$", text, re.M))
    assert versions == {"networking.gke.io/v1"}, f"unexpected policy apiVersions: {versions}"


def test_c5_12_api_backend_timeout_outlives_an_agent_request(base_values: dict) -> None:
    """The API backend timeout is far above GCP's 30s default.

    Mission planning and DTA transfers routinely run for minutes. On the stock
    timeout the load balancer answers 504 while the pod is still working, which
    reads to a user as the product hanging and then failing for no reason.
    """
    gcp = _gateway(base_values).get("gcp") or {}
    api_timeout = (gcp.get("api") or {}).get("timeoutSec")
    assert isinstance(api_timeout, int) and api_timeout >= MIN_API_TIMEOUT_SEC, (
        f"gateway.gcp.api.timeoutSec is {api_timeout!r}; needs >= {MIN_API_TIMEOUT_SEC}s"
    )


def test_c5_13_health_checks_probe_the_real_health_endpoints(base_values: dict) -> None:
    """The API health check targets /health; the web UI targets a path nginx serves."""
    text = _read(GCP_POLICY_TEMPLATE)
    assert "requestPath:" in text and "portSpecification: USE_FIXED_PORT" in text
    webui_health = ((_gateway(base_values).get("gcp") or {}).get("webui") or {}).get("healthPath")
    assert webui_health, "no gateway.gcp.webui.healthPath configured"
    webui_template = _read(TEMPLATE_DIR / "webui.yaml")
    assert webui_health in webui_template, (
        f"health check path {webui_health!r} is served by nothing in webui.yaml; the load "
        "balancer would mark every web UI backend unhealthy and serve 502"
    )


def test_c5_14_gke_overlay_pins_a_named_static_address() -> None:
    """The GKE Gateway claims a pre-reserved global IP by name.

    Without it GKE allocates an ephemeral address, and every reinstall changes the
    IP the avaloka.ai DNS records point at.
    """
    gateway = _gateway(_load_values(OVERLAY_DIR / "values-gke.yaml"))
    addresses = gateway.get("addresses") or []
    assert addresses, "values-gke.yaml pins no static address"
    assert all(a.get("type") == "NamedAddress" and a.get("value") for a in addresses), (
        f"expected NamedAddress entries with a value, got {addresses}"
    )


def test_c5_15_prereq_script_covers_the_gke_overlay_resource_names() -> None:
    """deploy/gcp-lb-prereqs.sh defaults match what values-gke.yaml expects to exist.

    The script creates the IP and the certificate map; the overlay references them
    by name. If the two drift, `helm install` succeeds and the Gateway then sits
    unprogrammed pointing at resources that were never created.
    """
    assert PREREQ_SCRIPT.is_file(), f"missing {PREREQ_SCRIPT}"
    script = _read(PREREQ_SCRIPT)
    gateway = _gateway(_load_values(OVERLAY_DIR / "values-gke.yaml"))

    ip_name = (gateway.get("addresses") or [{}])[0].get("value")
    cert_map = (gateway.get("tls") or {}).get("certMapName")
    assert f'IP_NAME="{ip_name}"' in script, f"script default IP name does not match {ip_name!r}"
    assert f'CERT_MAP_NAME="{cert_map}"' in script, (
        f"script default cert map does not match {cert_map!r}"
    )


# --------------------------------------------------------------- AWS specifics
def test_c5_16_aws_load_balancer_configuration_is_provider_scoped() -> None:
    """The LoadBalancerConfiguration renders only for AWS, and only with ACM ARNs."""
    assert AWS_CONFIG_TEMPLATE.is_file(), f"missing {AWS_CONFIG_TEMPLATE}"
    text = _read(AWS_CONFIG_TEMPLATE)
    head = text.lstrip().splitlines()[0]
    assert 'eq .Values.gateway.provider "aws"' in head, f"not scoped to provider=aws: {head!r}"
    assert ".Values.gateway.aws.certificateArns" in head
    assert "apiVersion: gateway.k8s.aws/v1" in text
    assert "protocolPort: HTTPS:443" in text


# --------------------------------------------------- HTTPS-only / two instances
@pytest.mark.parametrize("overlay", GKE_INSTANCES)
def test_c5_18_gke_instances_are_https_only(overlay: str) -> None:
    """Both GKE releases open no HTTP port at all.

    Certificate Manager is unaffected — it validates and renews over 443, not 80.
    """
    gateway = _gateway(_load_values(OVERLAY_DIR / overlay))
    assert gateway.get("httpEnabled") is False, (
        f"{overlay} still opens :80; set gateway.httpEnabled=false for HTTPS-only"
    )


def test_c5_19_http_listener_and_redirect_are_gated_on_http_being_enabled() -> None:
    """With httpEnabled=false the template emits neither the listener nor the redirect.

    A redirect HTTPRoute whose parent listener does not exist would sit
    permanently unresolved rather than failing loudly.
    """
    text = _read(GATEWAY_TEMPLATE)
    assert "{{- if $g.httpEnabled }}" in text, "the http listener is not gated"
    assert "$redirect := and $g.httpEnabled" in text, (
        "the redirect route does not depend on the http listener existing"
    )


def test_c5_20_the_two_gke_instances_share_no_public_infrastructure() -> None:
    """test and app get separate IPs and separate certificate maps.

    The point of a second instance is blast-radius isolation; sharing either of
    these would couple a production outage to a test redeploy.
    """
    test_gw = _gateway(_load_values(OVERLAY_DIR / "values-gke.yaml"))
    app_gw = _gateway(_load_values(OVERLAY_DIR / APP_OVERLAY))

    def _ip(gw: dict) -> str:
        return (gw.get("addresses") or [{}])[0].get("value", "")

    assert _ip(test_gw) and _ip(app_gw), "an instance pins no static address"
    assert _ip(test_gw) != _ip(app_gw), (
        f"both GKE instances claim the same global IP {_ip(test_gw)!r}"
    )

    test_map = (test_gw.get("tls") or {}).get("certMapName")
    app_map = (app_gw.get("tls") or {}).get("certMapName")
    assert test_map and app_map and test_map != app_map, (
        f"both GKE instances share certificate map {test_map!r}"
    )

    assert set(test_gw.get("hosts") or []).isdisjoint(app_gw.get("hosts") or []), (
        "the two instances serve overlapping hostnames; DNS can only point one way"
    )


def test_c5_21_app_overlay_documents_the_namespace_requirement() -> None:
    """The app overlay must warn that a separate namespace is mandatory.

    avaloka.fullname derives from the CHART name, not the release name, so two
    releases in one namespace both try to own a Service called `avaloka` and the
    second install fails. That is surprising enough to require a comment.
    """
    text = _read(OVERLAY_DIR / APP_OVERLAY)
    assert "namespace" in text.lower(), f"{APP_OVERLAY} does not mention the namespace"
    assert "nameOverride" in text, (
        f"{APP_OVERLAY} does not mention nameOverride as the same-namespace alternative"
    )


def test_c5_22_fullname_ignores_release_name() -> None:
    """Pins the reason two instances need separate namespaces.

    If avaloka.fullname ever starts honouring .Release.Name, the namespace
    requirement above stops being true and the app overlay's warning is stale.
    """
    helpers = _read(TEMPLATE_DIR / "_helpers.tpl")
    block = helpers[helpers.index('define "avaloka.fullname"'):]
    block = block[: block.index("{{- end -}}")]
    assert ".Release.Name" not in block, (
        "avaloka.fullname now uses the release name — two releases can share a "
        "namespace, so update values-gke-app.yaml's namespace warning"
    )


@pytest.mark.parametrize("overlay", ["values-eks.yaml", "values-aks.yaml"])
def test_c5_17_non_gcp_overlays_set_a_vendor_neutral_route_timeout(overlay: str) -> None:
    """AWS/Azure have no GCPBackendPolicy, so the long timeout rides on the HTTPRoute."""
    timeout = _gateway(_load_values(OVERLAY_DIR / overlay)).get("routeTimeout")
    assert timeout, f"{overlay} sets no gateway.routeTimeout; agent requests would be cut off"
    match = re.fullmatch(r"(\d+)s", str(timeout))
    assert match and int(match.group(1)) >= MIN_API_TIMEOUT_SEC, (
        f"{overlay} routeTimeout {timeout!r} is below {MIN_API_TIMEOUT_SEC}s"
    )
