"""
create_api_gateway.py
---------------------
Set up a GCP API Gateway in front of the Avaloka inference service.

Provides a public HTTPS endpoint, API key authentication, rate limiting,
and request logging — all without touching the inference container code.

High-level flow
---------------
1.  Enable required GCP APIs (apigateway, servicemanagement, servicecontrol).
2.  Build an OpenAPI 2.0 spec that proxies /inference and /health to the
    LoadBalancer IP obtained from Step 5 (run_inference_service).
3.  Create the API Gateway API resource.
4.  Create an API config from the OpenAPI spec.
5.  Deploy a Gateway instance and return the public ``defaultHostname``.
6.  (Optional) Generate per-client API keys restricted to this gateway's
    managed service.

Update helpers
--------------
``update_inference_backend`` patches the GKE Deployment's MLFLOW_RUN_ID env-var
so a newly trained model is picked up without changing the gateway URL or
rotating client API keys.

``delete_api_gateway`` tears down the Gateway, API config, and API resource.

Usage example
-------------
>>> from gcp.create_api_gateway import create_api_gateway, create_api_key
>>> gateway_url = create_api_gateway(
...     project_id="my-gcp-project",
...     location="us-central1",
...     backend_ip="34.56.78.90",
...     backend_port=8080,
... )
>>> print(f"Public endpoint: https://{gateway_url}/inference")
>>>
>>> key_string = create_api_key(
...     project_id="my-gcp-project",
...     display_name="Avaloka Key - AcmeCorp",
... )
>>> print(f"Client API key: {key_string}")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import google.auth
import google.auth.transport.requests
import requests as http_requests
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_ID        = "avaloka-inference-api"
DEFAULT_CONFIG_ID     = "avaloka-config-v1"
DEFAULT_GATEWAY_ID    = "avaloka-gateway"
DEFAULT_BACKEND_PORT  = 8080

# GCP REST base URLs
_GATEWAY_BASE  = "https://apigateway.googleapis.com/v1"
_SERVICES_BASE = "https://serviceusage.googleapis.com/v1"
_APIKEYS_BASE  = "https://apikeys.googleapis.com/v2"

# Required GCP APIs for API Gateway
_REQUIRED_APIS = [
    "apigateway.googleapis.com",
    "servicemanagement.googleapis.com",
    "servicecontrol.googleapis.com",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_api_gateway(
    project_id: str,
    location: str,
    backend_ip: str,
    *,
    backend_port: int = DEFAULT_BACKEND_PORT,
    api_id: str = DEFAULT_API_ID,
    config_id: str = DEFAULT_CONFIG_ID,
    gateway_id: str = DEFAULT_GATEWAY_ID,
    operation_timeout_s: int = 600,
) -> str:
    """
    Enable GCP APIs, create an API Gateway in front of the inference service,
    and return the public ``defaultHostname``.

    Parameters
    ----------
    project_id:
        GCP project ID.
    location:
        GCP region (e.g. ``"us-central1"``).
    backend_ip:
        External IP of the inference LoadBalancer Service (from
        ``run_inference_service``).
    backend_port:
        Port the inference service listens on (default 8080).
    api_id:
        API Gateway *API* resource name (must be unique per project).
    config_id:
        API config version label.
    gateway_id:
        Gateway instance name.
    operation_timeout_s:
        Seconds to wait for long-running Gateway operations.

    Returns
    -------
    tuple[str, str]
        (hostname, managed_service) e.g.
        ``("avaloka-gateway-xxxx.uc.gateway.dev", "inference-api-xxxx.apigateway.project.cloud.goog")``.
    """
    session = _authed_session()

    # 1. Enable required GCP APIs
    logger.info("Enabling required GCP APIs ...")
    for api in _REQUIRED_APIS:
        _enable_api(session, project_id, api)
    logger.info("All required GCP APIs are enabled.")

    # 2. Resolve backend hostname
    # GCP API Gateway forbids raw IP addresses in x-google-backend URLs.
    # nip.io provides wildcard DNS that maps "<ip>.nip.io" back to <ip>,
    # giving us a valid hostname without any DNS setup.
    backend_host = _ip_to_hostname(backend_ip)
    if backend_host != backend_ip:
        logger.info(
            "Backend IP '%s' converted to hostname '%s' (GCP API Gateway "
            "requires a DNS name, not a bare IP).",
            backend_ip, backend_host,
        )
    backend_base = f"http://{backend_host}:{backend_port}"

    # 3. Create the API resource
    logger.info("Creating API resource '%s' ...", api_id)
    _create_api_resource(session, project_id, api_id)

    # 4. Fetch managed service name from the API resource.
    #    This is available immediately after _create_api_resource and contains
    #    the full hashed name e.g.
    #    "inference-api-e4a8d1f0-2jpximna0lfxu.apigateway.project.cloud.goog"
    #    which must be used in x-google-endpoints for allowCors to work.
    managed_service = _get_managed_service_from_api_resource(session, project_id, api_id)
    logger.info("Managed service name: %s", managed_service)

    # 5. Build OpenAPI spec — now we have the real managed service name
    openapi_spec = _build_openapi_spec(
        api_id=api_id,
        project_id=project_id,
        backend_base=backend_base,
        managed_service=managed_service,
    )
    logger.debug("OpenAPI spec:\n%s", yaml.dump(openapi_spec))

    # 6. Create API config
    logger.info("Creating API config '%s' ...", config_id)
    op = _create_api_config(session, project_id, api_id, config_id, openapi_spec)
    _wait_for_gateway_operation(session, op["name"], operation_timeout_s)
    logger.info("API config '%s' is ready.", config_id)

    # 7. Deploy the Gateway
    logger.info("Deploying gateway '%s' in %s ...", gateway_id, location)
    op = _create_gateway(session, project_id, location, api_id, config_id, gateway_id)
    _wait_for_gateway_operation(session, op["name"], operation_timeout_s)
    logger.info("Gateway '%s' deployed.", gateway_id)

    # 8. Retrieve public hostname
    hostname, _ = _describe_gateway(session, project_id, location, gateway_id)
    logger.info("API Gateway is live at: https://%s", hostname)
    logger.info(
        "Managed service name (use for API key restrictions): %s", managed_service
    )

    # 7. Enable the managed service for this project via Service Management.
    #
    # GCP API Gateway creates a managed service (e.g.
    # "avaloka-inference-api-xxxx.apigateway.project.cloud.goog") via
    # Service Management — NOT via Service Usage.  The service must be
    # enabled on the consumer project before API keys scoped to it are
    # accepted.  Service Management uses a different endpoint and auth
    # flow from serviceusage.googleapis.com.
    # if managed_service:
    #     logger.info(
    #         "Enabling managed service '%s' for project '%s' ...",
    #         managed_service, project_id,
    #     )
    #     _enable_managed_service(session, project_id, managed_service)
    #     logger.info("Managed service '%s' enabled.", managed_service)

    return hostname, managed_service


def create_api_key(
    project_id: str,
    display_name: str,
    *,
    api_id: str = DEFAULT_API_ID,
    gateway_id: str = DEFAULT_GATEWAY_ID,
    location: str = "us-central1",
    managed_service: Optional[str] = None,
    enable_apikeys_api: bool = True,
) -> str:
    """
    Generate a new GCP API key restricted to the Avaloka inference gateway
    using the official ``google-cloud-api-keys`` client library.

    Each client company should receive their own key so usage can be tracked
    and keys can be revoked individually.

    Parameters
    ----------
    project_id:
        GCP project ID.
    display_name:
        Human-readable label (e.g. ``"Avaloka Key - AcmeCorp"``).
    gateway_id:
        Gateway instance name — used to probe key propagation.
    location:
        GCP region of the gateway (e.g. ``"us-central1"``).
    managed_service:
        The GCP-assigned managed service name for the gateway, e.g.
        ``"avaloka-inference-api-xxxx.apigateway.project.cloud.goog"``.
        If omitted, it is fetched automatically from the API resource.
        Pass it explicitly to avoid the extra lookup.
    enable_apikeys_api:
        When ``True`` (default) the function ensures ``apikeys.googleapis.com``
        is enabled and waits for propagation before proceeding.

    Returns
    -------
    str
        The raw key string (e.g. ``"AIzaSy..."``).  Hand this to the client.
    """
    from google.cloud import api_keys_v2

    if enable_apikeys_api:
        session = _authed_session(quota_project_id=project_id)
        logger.info("Ensuring apikeys.googleapis.com is enabled ...")
        _enable_api(session, project_id, "apikeys.googleapis.com")
        logger.info("Waiting for apikeys.googleapis.com propagation ...")
        _wait_for_api_propagation(session, project_id, "apikeys.googleapis.com")
    else:
        session = _authed_session(quota_project_id=project_id)

    # Fetch managed service name from the API resource (not the gateway).
    if not managed_service:
        logger.info(
            "Looking up managed service name from API resource '%s' ...", DEFAULT_API_ID
        )
        managed_service = _get_managed_service_from_api_resource(
            session, project_id, api_id
        )
        if not managed_service:
            raise RuntimeError(
                f"Could not determine managed service name for api_id '{DEFAULT_API_ID}'. "
                "Pass it explicitly via the managed_service= parameter."
            )

    logger.info(
        "Creating API key '%s' restricted to service '%s' ...",
        display_name,
        managed_service,
    )

    client = api_keys_v2.ApiKeysClient()

    key = api_keys_v2.Key()
    key.display_name = display_name

    api_target = api_keys_v2.ApiTarget()
    api_target.service = managed_service

    restrictions = api_keys_v2.Restrictions()
    restrictions.api_targets = [api_target]
    key.restrictions = restrictions

    request = api_keys_v2.CreateKeyRequest()
    request.parent = f"projects/{project_id}/locations/global"
    request.key = key

    response = client.create_key(request=request).result()

    key_string = response.key_string
    logger.info(
        "API key '%s' created (key prefix: %s...).", display_name, key_string[:10]
    )

    logger.info("Waiting for API key to propagate ...")
    _wait_for_key_propagation(session, key_string, project_id, location, gateway_id)
    logger.info("API key is active and accepted by the gateway.")

    return key_string


def update_inference_backend(
    cluster_name: str,
    project_id: str,
    location: str,
    new_mlflow_run_id: str,
    *,
    namespace: str = "inference",
    deployment_name: Optional[str] = None,
    rollout_timeout_s: int = 600,
) -> None:
    """
    Update ``MLFLOW_RUN_ID`` on the GKE inference Deployment so new pods
    download the newly trained model on startup.

    The API Gateway URL and all client API keys remain unchanged — only the
    Deployment env-var is patched and a rolling restart is triggered.

    Parameters
    ----------
    cluster_name:
        GKE cluster where the inference service runs.
    project_id, location:
        GCP coordinates.
    new_mlflow_run_id:
        The ``mlflow_run_id`` of the newly trained model.
    namespace:
        Kubernetes namespace of the inference Deployment.
    deployment_name:
        Defaults to ``"{cluster_name}-deployment"`` (matches
        ``run_inference_service`` naming convention).
    rollout_timeout_s:
        Seconds to wait for the rolling restart to complete.
    """
    from kubernetes import client as k8s_client
    from .get_k8s_api import get_k8s_api

    dep_name = deployment_name or f"{cluster_name}-deployment"
    logger.info(
        "Updating MLFLOW_RUN_ID to '%s' on Deployment '%s/%s' ...",
        new_mlflow_run_id, namespace, dep_name,
    )

    api_client = get_k8s_api(cluster_name, project_id, location)
    apps_v1    = k8s_client.AppsV1Api(api_client)

    dep = apps_v1.read_namespaced_deployment(name=dep_name, namespace=namespace)

    for container in dep.spec.template.spec.containers:
        patched = False
        for env_var in (container.env or []):
            if env_var.name == "MLFLOW_RUN_ID":
                env_var.value = new_mlflow_run_id
                patched = True
                break
        if not patched:
            container.env = (container.env or []) + [
                k8s_client.V1EnvVar(name="MLFLOW_RUN_ID", value=new_mlflow_run_id)
            ]

    apps_v1.replace_namespaced_deployment(name=dep_name, namespace=namespace, body=dep)
    logger.info("Deployment '%s' patched — rolling restart in progress ...", dep_name)

    _wait_for_deployment_rollout(api_client, dep_name, namespace, rollout_timeout_s)
    logger.info(
        "Rollout complete. Inference service is now using run_id='%s'.",
        new_mlflow_run_id,
    )


def delete_api_gateway(
    project_id: str,
    location: str,
    *,
    api_id: str = DEFAULT_API_ID,
    config_id: str = DEFAULT_CONFIG_ID,
    gateway_id: str = DEFAULT_GATEWAY_ID,
    operation_timeout_s: int = 300,
) -> None:
    """
    Tear down the Gateway instance, API config, and API resource in order.

    Parameters
    ----------
    project_id, location:
        GCP coordinates.
    api_id, config_id, gateway_id:
        Resource identifiers (must match those used during
        ``create_api_gateway``).
    operation_timeout_s:
        Seconds to wait per individual delete operation.
    """
    session = _authed_session()

    logger.info("Deleting gateway '%s' ...", gateway_id)
    try:
        op = _delete_resource(
            session,
            f"{_GATEWAY_BASE}/projects/{project_id}/locations/{location}"
            f"/gateways/{gateway_id}",
        )
        _wait_for_gateway_operation(session, op["name"], operation_timeout_s)
        logger.info("Gateway '%s' deleted.", gateway_id)
    except _ResourceNotFoundError:
        logger.warning("Gateway '%s' not found — skipping.", gateway_id)

    logger.info("Deleting API config '%s' ...", config_id)
    try:
        op = _delete_resource(
            session,
            f"{_GATEWAY_BASE}/projects/{project_id}/locations/global"
            f"/apis/{api_id}/configs/{config_id}",
        )
        _wait_for_gateway_operation(session, op["name"], operation_timeout_s)
        logger.info("API config '%s' deleted.", config_id)
    except _ResourceNotFoundError:
        logger.warning("API config '%s' not found — skipping.", config_id)

    logger.info("Deleting API resource '%s' ...", api_id)
    try:
        op = _delete_resource(
            session,
            f"{_GATEWAY_BASE}/projects/{project_id}/locations/global/apis/{api_id}",
        )
        _wait_for_gateway_operation(session, op["name"], operation_timeout_s)
        logger.info("API resource '%s' deleted.", api_id)
    except _ResourceNotFoundError:
        logger.warning("API resource '%s' not found — skipping.", api_id)


# ---------------------------------------------------------------------------
# Internal: OpenAPI spec builder
# ---------------------------------------------------------------------------

def _build_openapi_spec(
    api_id: str,
    project_id: str,
    backend_base: str,
    managed_service: Optional[str] = None,
) -> dict:
    """
    Construct the OpenAPI 2.0 spec dict consumed by API Gateway.

    The ``host`` and ``x-google-endpoints.name`` fields are set to the real
    managed service name so that ``allowCors`` takes effect immediately on
    first deploy — no two-pass deployment needed.
    """
    endpoints_name = managed_service or f"{api_id}.apigateway.{project_id}.cloud.goog"

    return {
        "swagger": "2.0",
        "info": {
            "title": api_id,
            "description": "Avaloka ML Inference Public API",
            "version": "1.0.0",
        },
        "host": endpoints_name,
        "schemes": ["https"],
        "produces": ["application/json"],
        "consumes": ["application/json"],
        "x-google-endpoints": [
            {
                "name": endpoints_name,
                "allowCors": True,
            }
        ],
        "x-google-management": {
            "metrics": [
                {
                    "name": "inference-requests",
                    "displayName": "Inference requests",
                    "valueType": "INT64",
                    "metricKind": "DELTA",
                }
            ],
            "quota": {
                "limits": [
                    {
                        "name": "inference-request-limit",
                        "metric": "inference-requests",
                        "unit": "1/min/{project}",
                        "values": {"STANDARD": 1000},
                    }
                ]
            },
        },
        "paths": {
            "/inference": {
                "post": {
                    "summary": "Run ML model inference",
                    "operationId": "inference",
                    "x-google-backend": {
                        "address": f"{backend_base}/inference",
                        "deadline": 600.0,
                    },
                    "security": [],
                    "responses": {
                        "200": {"description": "Prediction result"},
                        "401": {"description": "Unauthorized — missing or invalid API key"},
                        "429": {"description": "Rate limit exceeded"},
                        "502": {"description": "Bad gateway — inference service unreachable"},
                    },
                },
                "options": {
                    "summary": "CORS preflight for inference",
                    "operationId": "inferenceCorsPreflight",
                    "x-google-backend": {
                        "address": f"{backend_base}/inference",
                        "deadline": 10.0,
                    },
                    "security": [],
                    "responses": {
                        "204": {"description": "CORS preflight response"},
                    },
                },
            },
            "/health": {
                "get": {
                    "summary": "Health check (no auth required)",
                    "operationId": "health",
                    "x-google-backend": {
                        "address": f"{backend_base}/health",
                        "deadline": 10.0,
                    },
                    "security": [],
                    "responses": {
                        "200": {"description": "Service is healthy"},
                        "503": {"description": "Service unavailable"},
                    },
                }
            },
        },
        "securityDefinitions": {
            "api_key": {
                "type": "apiKey",
                "name": "x-api-key",
                "in": "header",
            }
        },
    }


# ---------------------------------------------------------------------------
# Internal: GCP API enablement
# ---------------------------------------------------------------------------

def _enable_api(session: http_requests.Session, project_id: str, api_name: str) -> None:
    """Enable a single GCP service API, waiting for the LRO to complete."""
    url  = f"{_SERVICES_BASE}/projects/{project_id}/services/{api_name}:enable"
    resp = session.post(url, json={})
    _raise_for_status(resp, f"enable API {api_name}")
    op = resp.json()

    op_name = op.get("name", "")
    if op_name and not op.get("done") and "noop" not in op_name.lower():
        _wait_for_service_operation(session, op_name)
    logger.debug("API '%s' is enabled.", api_name)


def _enable_managed_service(
    session: http_requests.Session,
    project_id: str,
    managed_service: str,
    timeout_s: int = 120,
) -> None:
    """
    Enable a GCP API Gateway managed service for a consumer project.

    Gateway-created managed services live under Service Management
    (``servicemanagement.googleapis.com``), not Service Usage.  They cannot
    be enabled via the ``serviceusage.googleapis.com/v1/projects/.../services``
    endpoint — that endpoint only knows about pre-registered public GCP APIs.

    The correct flow is:
      POST servicemanagement.googleapis.com/v1/services/{service}/enable
      body: { "consumerId": "project:<project_id>" }

    This is idempotent — enabling an already-enabled service is a no-op.
    """
    _SM_BASE = "https://servicemanagement.googleapis.com/v1"
    url  = f"{_SM_BASE}/services/{managed_service}:enable"
    body = {"consumerId": f"project:{project_id}"}

    resp = session.post(url, json=body)

    if resp.status_code == 200:
        op = resp.json()
        op_name = op.get("name", "")
        if op_name:
            _wait_for_service_management_operation(session, op_name, timeout_s)
        return

    if resp.status_code in (400, 409):
        body_text = resp.text.lower()
        if "already" in body_text or resp.status_code == 409:
            logger.debug(
                "Managed service '%s' already enabled — skipping.", managed_service
            )
            return

    _raise_for_status(resp, f"enable managed service {managed_service}")


def _wait_for_service_management_operation(
    session: http_requests.Session,
    op_name: str,
    timeout_s: int = 120,
) -> None:
    """Poll a Service Management LRO until done."""
    _SM_BASE = "https://servicemanagement.googleapis.com/v1"
    url      = f"{_SM_BASE}/{op_name}" if not op_name.startswith("http") else op_name
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = session.get(url)
        if not resp.ok:
            logger.debug(
                "Service Management op poll returned %s — retrying ...",
                resp.status_code,
            )
            time.sleep(5)
            continue
        op = resp.json()
        if op.get("done"):
            err = op.get("error")
            if err:
                if err.get("code") == 6:
                    logger.debug("Managed service already enabled (ALREADY_EXISTS).")
                    return
                raise RuntimeError(
                    f"Service Management operation '{op_name}' failed: {err}"
                )
            return
        time.sleep(5)

    logger.warning(
        "Service Management operation '%s' did not complete within %ss — "
        "proceeding anyway.", op_name, timeout_s,
    )


def _wait_for_api_propagation(
    session: http_requests.Session,
    project_id: str,
    api_name: str,
    timeout_s: int = 120,
    poll_interval_s: float = 5.0,
) -> None:
    """Poll the Service Usage API until ``api_name`` reports as ENABLED."""
    url      = f"{_SERVICES_BASE}/projects/{project_id}/services/{api_name}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = session.get(url)
        if resp.ok:
            state = resp.json().get("state", "")
            if state == "ENABLED":
                logger.debug("API '%s' confirmed ENABLED.", api_name)
                return
            logger.debug("API '%s' state=%s — waiting ...", api_name, state)
        else:
            logger.debug(
                "Could not query state of '%s' (HTTP %s) — waiting ...",
                api_name, resp.status_code,
            )
        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"API '{api_name}' did not reach ENABLED state within {timeout_s}s."
    )


def _wait_for_key_propagation(
    session: http_requests.Session,
    key_string: str,
    project_id: str,
    location: str,
    gateway_id: str,
    timeout_s: int = 90,
    poll_interval_s: float = 5.0,
) -> None:
    """Probe the gateway /inference endpoint with the new key until it is accepted."""
    try:
        hostname = _get_gateway_hostname(session, project_id, location, gateway_id)
    except Exception as exc:
        logger.warning(
            "Could not resolve gateway hostname for key probe: %s. "
            "Falling back to fixed 60s sleep.", exc,
        )
        time.sleep(60)
        return

    probe_url = f"https://{hostname}/inference"
    deadline  = time.time() + timeout_s

    while time.time() < deadline:
        try:
            resp = http_requests.post(
                probe_url,
                headers={"x-api-key": key_string},
                json={},
                timeout=10,
            )
            body = resp.text.lower()

            if resp.status_code == 400 and (
                "expired" in body or "api key not valid" in body
            ):
                logger.debug(
                    "Key probe: still propagating (HTTP %s: %s) ...",
                    resp.status_code, resp.text[:120],
                )
                time.sleep(poll_interval_s)
                continue

            logger.debug(
                "Key probe: gateway accepted key (HTTP %s).", resp.status_code
            )
            return

        except Exception as exc:
            logger.debug("Key probe request failed: %s — retrying ...", exc)
            time.sleep(poll_interval_s)

    logger.warning(
        "API key did not propagate within %ss — proceeding anyway.", timeout_s
    )


def _wait_for_service_operation(
    session: http_requests.Session,
    op_name: str,
    timeout_s: int = 120,
) -> dict:
    """Poll a Service Usage LRO until done."""
    url      = f"https://serviceusage.googleapis.com/v1/{op_name}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = session.get(url)
        _raise_for_status(resp, f"poll service operation {op_name}")
        op = resp.json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(
                    f"Service operation '{op_name}' failed: {op['error']}"
                )
            return op
        time.sleep(3)

    raise TimeoutError(
        f"Service operation '{op_name}' did not complete within {timeout_s}s."
    )


# ---------------------------------------------------------------------------
# Internal: API Gateway resource management
# ---------------------------------------------------------------------------

def _get_managed_service_from_api_resource(
    session: http_requests.Session,
    project_id: str,
    api_id: str,
) -> str:
    """
    Fetch the managed service name directly from the API resource.

    This is available immediately after ``_create_api_resource`` completes
    and contains the full hashed name, e.g.
    ``"inference-api-e4a8d1f0-2jpximna0lfxu.apigateway.<GCP_PROJECT_ID>.cloud.goog"``.

    This is the correct value to use in ``x-google-endpoints`` for CORS,
    and for restricting API keys — NOT the gateway hostname hash.
    """
    url  = f"{_GATEWAY_BASE}/projects/{project_id}/locations/global/apis/{api_id}"
    resp = session.get(url)
    _raise_for_status(resp, f"fetch API resource {api_id}")
    data = resp.json()
    managed_service = data.get("managedService", "")
    if not managed_service:
        raise RuntimeError(
            f"API resource '{api_id}' has no managedService field. "
            f"Full response: {data}"
        )
    logger.debug("Fetched managed service from API resource: %s", managed_service)
    return managed_service


def _create_api_resource(
    session: http_requests.Session,
    project_id: str,
    api_id: str,
) -> None:
    """Create the top-level API Gateway *API* resource (idempotent)."""
    url  = (
        f"{_GATEWAY_BASE}/projects/{project_id}/locations/global/apis"
        f"?apiId={api_id}"
    )
    resp = session.post(url, json={"displayName": api_id})

    if resp.status_code == 409:
        logger.info("API resource '%s' already exists — reusing.", api_id)
        return

    _raise_for_status(resp, f"create API resource {api_id}")

    op = resp.json()
    if "name" in op:
        _wait_for_gateway_operation(session, op["name"])


def _create_api_config(
    session: http_requests.Session,
    project_id: str,
    api_id: str,
    config_id: str,
    openapi_spec: dict,
) -> dict:
    """Upload the OpenAPI spec as a new API config version, returning the LRO."""
    import base64

    spec_b64 = base64.b64encode(
        yaml.dump(openapi_spec, default_flow_style=False).encode()
    ).decode()

    url  = (
        f"{_GATEWAY_BASE}/projects/{project_id}/locations/global"
        f"/apis/{api_id}/configs?apiConfigId={config_id}"
    )
    body = {
        "displayName": config_id,
        "openapiDocuments": [
            {
                "document": {
                    "path": "api-config.yaml",
                    "contents": spec_b64,
                }
            }
        ],
    }
    resp = session.post(url, json=body)

    if resp.status_code == 409:
        logger.info("API config '%s' already exists — reusing.", config_id)
        return {
            "name": f"projects/{project_id}/locations/global/operations/noop",
            "done": True,
        }

    _raise_for_status(resp, f"create API config {config_id}")
    return resp.json()


def _create_gateway(
    session: http_requests.Session,
    project_id: str,
    location: str,
    api_id: str,
    config_id: str,
    gateway_id: str,
) -> dict:
    """Deploy the API Gateway instance, returning the LRO."""
    api_config_resource = (
        f"projects/{project_id}/locations/global/apis/{api_id}/configs/{config_id}"
    )
    url  = (
        f"{_GATEWAY_BASE}/projects/{project_id}/locations/{location}"
        f"/gateways?gatewayId={gateway_id}"
    )
    body = {
        "displayName": gateway_id,
        "apiConfig": api_config_resource,
    }
    resp = session.post(url, json=body)

    if resp.status_code == 409:
        logger.info(
            "Gateway '%s' already exists — updating to config '%s' ...",
            gateway_id, config_id,
        )
        patch_url = (
            f"{_GATEWAY_BASE}/projects/{project_id}/locations/{location}"
            f"/gateways/{gateway_id}?updateMask=apiConfig"
        )
        patch_resp = session.patch(patch_url, json={"apiConfig": api_config_resource})
        if patch_resp.ok:
            return patch_resp.json()
        logger.debug(
            "Gateway patch returned %s — treating as noop.", patch_resp.status_code
        )
        return {
            "name": (
                f"projects/{project_id}/locations/{location}/operations/noop"
            ),
            "done": True,
        }

    _raise_for_status(resp, f"create gateway {gateway_id}")
    return resp.json()


def _get_gateway_hostname(
    session: http_requests.Session,
    project_id: str,
    location: str,
    gateway_id: str,
) -> str:
    """Describe the deployed gateway and return its ``defaultHostname``."""
    hostname, _ = _describe_gateway(session, project_id, location, gateway_id)
    return hostname


def _describe_gateway(
    session: http_requests.Session,
    project_id: str,
    location: str,
    gateway_id: str,
) -> tuple[str, str]:
    """
    Describe the deployed gateway and return ``(defaultHostname, managedService)``.

    Note: managedService here is fetched from the gateway describe response
    for convenience, but the authoritative source is the API resource
    (use ``_get_managed_service_from_api_resource`` for CORS / key restriction).
    """
    url  = (
        f"{_GATEWAY_BASE}/projects/{project_id}/locations/{location}"
        f"/gateways/{gateway_id}"
    )
    resp = session.get(url)
    _raise_for_status(resp, f"describe gateway {gateway_id}")
    data = resp.json()

    hostname = data.get("defaultHostname", "")
    if not hostname:
        raise RuntimeError(
            f"Gateway '{gateway_id}' has no defaultHostname yet. "
            f"Full response: {data}"
        )

    managed_service = data.get("managedService", "")
    return hostname, managed_service


def _get_managed_service_from_config(
    session: http_requests.Session,
    project_id: str,
    api_config_resource: str,
) -> str:
    """
    Look up the managedService name from the API config resource.
    Falls back to an empty string if the field is absent.
    """
    if not api_config_resource:
        return ""
    url  = f"{_GATEWAY_BASE}/{api_config_resource}"
    resp = session.get(url)
    if not resp.ok:
        logger.warning(
            "Could not fetch API config '%s' to resolve managed service: HTTP %s",
            api_config_resource, resp.status_code,
        )
        return ""
    data = resp.json()
    return (
        data.get("managedServiceConfigs", [{}])[0].get("managedService", "")
        or data.get("serviceConfigId", "")
    )


def _delete_resource(session: http_requests.Session, url: str) -> dict:
    """DELETE a resource URL and return the resulting LRO dict."""
    resp = session.delete(url)
    if resp.status_code == 404:
        raise _ResourceNotFoundError(url)
    _raise_for_status(resp, f"DELETE {url}")
    return resp.json()


# ---------------------------------------------------------------------------
# Internal: Long-running operation polling
# ---------------------------------------------------------------------------

def _wait_for_gateway_operation(
    session: http_requests.Session,
    op_name: str,
    timeout_s: int = 600,
) -> dict:
    """
    Poll an API Gateway LRO until ``done`` is ``True``.

    Accepts either a resource-path string
    (``"projects/.../locations/.../operations/xxx"``) or a full URL.
    Synthetic ``noop`` operations are resolved immediately.
    """
    if op_name.endswith("/noop"):
        return {"done": True}

    url = op_name if op_name.startswith("https://") else f"{_GATEWAY_BASE}/{op_name}"

    deadline      = time.time() + timeout_s
    poll_interval = 5.0

    while time.time() < deadline:
        resp = session.get(url)
        _raise_for_status(resp, f"poll gateway operation {op_name}")
        op = resp.json()

        if op.get("done"):
            if "error" in op:
                raise RuntimeError(
                    f"Gateway operation '{op_name}' failed: {op['error']}"
                )
            logger.debug("Gateway operation '%s' completed.", op_name)
            return op

        elapsed = timeout_s - (deadline - time.time())
        logger.debug(
            "Gateway operation '%s' still running (%.0fs elapsed) ...",
            op_name, elapsed,
        )
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 30.0)

    raise TimeoutError(
        f"Gateway operation '{op_name}' did not complete within {timeout_s}s."
    )


def _wait_for_apikeys_operation(
    session: http_requests.Session,
    op_name: str,
    timeout_s: int = 120,
) -> dict:
    """Poll an API Keys LRO until done."""
    url      = op_name if op_name.startswith("https://") else f"{_APIKEYS_BASE}/{op_name}"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = session.get(url)
        _raise_for_status(resp, f"poll apikeys operation {op_name}")
        op = resp.json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(
                    f"API keys operation '{op_name}' failed: {op['error']}"
                )
            return op
        time.sleep(3)

    raise TimeoutError(
        f"API keys operation '{op_name}' did not complete within {timeout_s}s."
    )


# ---------------------------------------------------------------------------
# Internal: Kubernetes rollout waiter
# ---------------------------------------------------------------------------

def _wait_for_deployment_rollout(
    api_client,
    deployment_name: str,
    namespace: str,
    timeout_s: int,
) -> None:
    """Block until all pods in the Deployment are updated and Available."""
    from kubernetes import client as k8s_client

    apps_v1  = k8s_client.AppsV1Api(api_client)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            dep       = apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            desired   = dep.spec.replicas or 1
            updated   = dep.status.updated_replicas or 0
            available = dep.status.available_replicas or 0
            ready     = dep.status.ready_replicas or 0

            logger.debug(
                "Rollout '%s': desired=%d updated=%d available=%d ready=%d",
                deployment_name, desired, updated, available, ready,
            )

            if updated >= desired and available >= desired and ready >= desired:
                return

        except k8s_client.ApiException as exc:
            logger.warning("Error polling deployment '%s': %s", deployment_name, exc)

        time.sleep(10)

    raise TimeoutError(
        f"Deployment '{deployment_name}' did not finish rolling out within {timeout_s}s."
    )


# ---------------------------------------------------------------------------
# Internal: auth + HTTP helpers
# ---------------------------------------------------------------------------

def _authed_session(quota_project_id: Optional[str] = None) -> http_requests.Session:
    """
    Return a ``requests.Session`` pre-loaded with a fresh Google OAuth2
    Bearer token for all ``cloud-platform`` scoped APIs.
    """
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        quota_project_id=quota_project_id,
    )
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    session = http_requests.Session()
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    if quota_project_id:
        headers["x-goog-user-project"] = quota_project_id
    session.headers.update(headers)
    return session


def _ip_to_hostname(address: str) -> str:
    """
    Return a DNS hostname that resolves to ``address``.

    GCP API Gateway rejects bare IPv4/IPv6 addresses in ``x-google-backend``
    URLs.  When ``address`` is already a hostname it is returned unchanged.
    When it is a raw IPv4 address (e.g. ``"35.188.70.201"``) we use the
    public nip.io wildcard-DNS service: ``"35.188.70.201.nip.io"`` resolves
    to ``35.188.70.201`` with no registration or configuration required.
    """
    import re
    ipv4_re = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
    if ipv4_re.match(address):
        return f"{address}.nip.io"
    ipv6_re = re.compile(r"^[0-9a-fA-F:]+$")
    if ipv6_re.match(address) and ":" in address:
        dashed = address.replace(":", "-")
        return f"{dashed}.nip.io"
    return address


def _raise_for_status(resp: http_requests.Response, context: str) -> None:
    """Raise a descriptive ``RuntimeError`` on non-2xx HTTP responses."""
    if not resp.ok:
        raise RuntimeError(
            f"GCP API call failed [{context}]: "
            f"HTTP {resp.status_code} — {resp.text}"
        )


class _ResourceNotFoundError(Exception):
    """Raised internally when a DELETE target returns HTTP 404."""