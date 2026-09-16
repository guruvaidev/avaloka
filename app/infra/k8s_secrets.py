# app/infra/k8s_secrets.py
from __future__ import annotations

import logging
import re
import subprocess
import time
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize_k8s_name(name: str, max_len: int = 63) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "secret"
    return (s[:max_len].strip("-")) or "secret"


def _run_kubectl(args: list[str], *, input_text: Optional[str] = None, timeout_s: int = 30) -> None:
    """
    Runs kubectl without shell=True (Windows safe).
    Raises RuntimeError with kubectl stderr on failure.
    """
    cmd = ["kubectl", *args]
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as e:
        raise RuntimeError(f"kubectl failed to run: {e}") from e

    if proc.returncode != 0:
        # IMPORTANT: do NOT include input_text here (it contains secrets)
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {err}")


def kubectl_apply_yaml(yaml_text: str, *, timeout_s: int = 30) -> None:
    _run_kubectl(["apply", "-f", "-"], input_text=yaml_text, timeout_s=timeout_s)


def kubectl_delete_secret(secret_name: str, namespace: str, *, timeout_s: int = 30) -> None:
    # ignore-not-found so cleanup is always safe
    _run_kubectl(["-n", namespace, "delete", "secret", secret_name, "--ignore-not-found=true"], timeout_s=timeout_s)


def _build_secret_yaml(secret_name: str, namespace: str, string_data: Dict[str, str]) -> str:
    # stringData is convenient (k8s will base64 it into data)
    # secret keys MUST be valid env var names if you use envFrom in the RayJob template.
    lines = [
        "apiVersion: v1",
        "kind: Secret",
        "metadata:",
        f"  name: {secret_name}",
        f"  namespace: {namespace}",
        "type: Opaque",
        "stringData:",
    ]
    for k, v in string_data.items():
        safe_v = (v or "").replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {k}: "{safe_v}"')
        if k == "GCP_SERVICE_ACCOUNT_JSON":
            has_backslash_n = "\\n" in (v or "")
            logger.info(
                "[k8s_secrets] YAML for %s: raw value has \\n=%s, "
                "escaped value len=%d (raw len=%d)",
                k, has_backslash_n, len(safe_v), len(v or ""),
            )
    return "\n".join(lines) + "\n"


def build_cloud_secret_yaml(
    *,
    secret_name: str,
    namespace: str,
    provider: str,
    creds: Dict[str, Any],
) -> str:
    """
    provider: "s3" | "gcs" | "az" (or "azure")
    creds: dict returned by your get_cloud_connection(connection_id)

    NOTE:
    - Because your RayJob YAML uses envFrom: secretRef, secret keys must be valid env var names.
    """
    p = (provider or "").lower().strip()

    # ---- S3 / AWS ----
    if p in {"s3", "aws"}:
        # Adjust these keys if your get_cloud_connection returns different names
        ak = creds.get("access_key") or creds.get("aws_access_key_id") or creds.get("AWS_ACCESS_KEY_ID")
        sk = creds.get("secret_key") or creds.get("aws_secret_access_key") or creds.get("AWS_SECRET_ACCESS_KEY")
        region = creds.get("region") or creds.get("aws_default_region") or creds.get("AWS_DEFAULT_REGION") or "us-east-1"
        token = creds.get("session_token") or creds.get("aws_session_token") or creds.get("AWS_SESSION_TOKEN")

        if not ak or not sk:
            raise ValueError("Missing AWS credentials (access_key/secret_key) for S3 secret")

        sd = {
            "AWS_ACCESS_KEY_ID": str(ak),
            "AWS_SECRET_ACCESS_KEY": str(sk),
            "AWS_DEFAULT_REGION": str(region),
        }
        if token:
            sd["AWS_SESSION_TOKEN"] = str(token)

        return _build_secret_yaml(secret_name, namespace, sd)

    # ---- GCS ----
    if p in {"gcs", "gcp"}:
        # To keep envFrom valid, store JSON in an env var.
        # You’ll write it to a file inside Ray code if needed later.

        sa_json = (
            creds.get("service_account_json")
            or creds.get("gcp_service_account_json")
            or creds.get("secret_key")   # ← your Supabase schema stores SA JSON here
        )
        # if not sa_json:
        #     raise ValueError("Missing GCP service account JSON in credentials")

       
        if not sa_json:
            # fallback: read from local file path env var (for dev/testing)
            sa_path = os.getenv("GCP_SERVICE_ACCOUNT_JSON_PATH", "")
            if sa_path and os.path.exists(sa_path):
                with open(sa_path, "r", encoding="utf-8") as f:
                    sa_json = f.read()
        if not sa_json:
            #raise ValueError("Missing GCP service account json for GCS secret (service_account_json)")
            raise ValueError("Missing GCP service account json for GCS secret")

        sd = {
            "GCP_SERVICE_ACCOUNT_JSON": str(sa_json),
            # helpful standard vars (optional)
            "GOOGLE_CLOUD_PROJECT": str(creds.get("project") or creds.get("gcp_project") or ""),
        }
        # Empty keys are allowed but we can drop if blank
        sd = {k: v for k, v in sd.items() if v}

        return _build_secret_yaml(secret_name, namespace, sd)

    # ---- Azure ----
    if p in {"az", "azure"}:
        # Common ways adlfs authenticates:
        # - AZURE_STORAGE_CONNECTION_STRING
        # - AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
        conn_str = creds.get("connection_string") or creds.get("AZURE_STORAGE_CONNECTION_STRING")
        acct = creds.get("account_name") or creds.get("AZURE_STORAGE_ACCOUNT_NAME")
        key = creds.get("account_key") or creds.get("AZURE_STORAGE_ACCOUNT_KEY")
        sas = creds.get("sas_token") or creds.get("sas") or creds.get("AZURE_STORAGE_SAS_TOKEN")

        sd: Dict[str, str] = {}
        if conn_str:
            sd["AZURE_STORAGE_CONNECTION_STRING"] = str(conn_str)
        if acct and key:
            sd["AZURE_STORAGE_ACCOUNT_NAME"] = str(acct)
            sd["AZURE_STORAGE_ACCOUNT_KEY"] = str(key)
        
        # SAS auth (needs account name to be useful)
        if acct and sas:
            sd["AZURE_STORAGE_ACCOUNT_NAME"] = str(acct)
            sd["AZURE_STORAGE_SAS_TOKEN"] = str(sas)

        if not sd:
            raise ValueError("Missing Azure credentials for secret (connection_string or account_name/account_key)+sas_token")

        return _build_secret_yaml(secret_name, namespace, sd)

    raise ValueError(f"Unsupported provider for cloud secret: {provider!r}")


def create_cloud_secret(
    *,
    namespace: str,
    provider: str,
    creds: Dict[str, Any],
    dataset_id: str,
    prefix: str = "avaloka-raycreds",
) -> str:
    """
    Creates the secret in Kubernetes and returns the secret name.
    """
    base = f"{prefix}-{(dataset_id or '')[:8]}-{int(time.time())}"
    secret_name = _sanitize_k8s_name(base)

    secret_yaml = build_cloud_secret_yaml(
        secret_name=secret_name,
        namespace=namespace,
        provider=provider,
        creds=creds,
    )
    kubectl_apply_yaml(secret_yaml)
    return secret_name
