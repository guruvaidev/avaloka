from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from app.infra.k8s_secrets import create_cloud_secret, kubectl_delete_secret
from app.infra.rayjob_renderer import render_rayjob_yaml_from_file
from app.infra.ray_job_runner import run_rayjob_from_yaml
from app.api.cloud_connections import get_cloud_connection

logger = logging.getLogger("avaloka.rayjob_submitter")


def _normalize_provider(p: Optional[str]) -> str:
    p = (p or "").strip().lower()
    if p in ("aws_s3", "amazon_s3", "aws"):
        return "s3"
    if p in ("google_cloud_platform", "gcp", "google_cloud_storage", "google"):
        return "gcs"
    if p in ("azure_blob_storage", "azure_blob", "microsoft_azure"):
        return "azure"
    return p


def _build_creds(provider: str, conn: Dict[str, Any]) -> Dict[str, Any]:
    """
    get_cloud_connection() returns a flat Supabase row (NOT {"creds": {...}}).
    This function converts the row into a provider-specific creds payload
    for create_cloud_secret().
    """
    provider = _normalize_provider(provider)

    # ---------- S3 / AWS ----------
    if provider in ("s3",):
        access_key = conn.get("access_key") or conn.get("aws_access_key_id")
        secret_key = conn.get("secret_key") or conn.get("aws_secret_access_key")
        session_token = conn.get("session_token") or conn.get("aws_session_token")
        region = conn.get("region") or conn.get("aws_region")

        if not access_key or not secret_key:
            raise ValueError(
                "S3 connection missing access_key/secret_key. "
                f"Got keys={list(conn.keys())}"
            )

        creds: Dict[str, Any] = {
            "access_key": access_key,
            "secret_key": secret_key,
        }
        if session_token:
            creds["session_token"] = session_token
        if region:
            creds["region"] = region

        # keep these if your secret creator wants them
        if conn.get("bucket_name"):
            creds["bucket_name"] = conn.get("bucket_name")
        if conn.get("endpoint_url"):
            creds["endpoint_url"] = conn.get("endpoint_url")

        return creds

    # ---------- Azure ----------
    if provider in ("azure", "az"):
        # Your cloud_connections.py maps these fields like:
        # account = conn.get("access_key") or conn.get("account_name")
        # container = conn.get("bucket_name") or conn.get("container")
        # sas_token = conn.get("secret_key") or conn.get("sas")
        account = conn.get("account_name") or conn.get("access_key")
        container = conn.get("container") or conn.get("bucket_name")
        sas_token = conn.get("sas_token") or conn.get("sas") or conn.get("secret_key")

        if not account or not container or not sas_token:
            raise ValueError(
                "Azure connection missing account/container/sas_token. "
                f"account={bool(account)} container={bool(container)} sas={bool(sas_token)}"
            )

        return {
            "account_name": account,
            "container": container,
            "sas_token": sas_token,
        }

    # ---------- GCS ----------
    if provider in ("gcs", "gs"):
        # Depends on how you store GCS creds. If you store a service account json,
        # keep it here. Otherwise we pass the row and let create_cloud_secret handle it.
        for key in (
            "service_account_json",
            "gcp_service_account_json",
            "credentials_json",
            "gcs_credentials_json",
        ):
            if conn.get(key):
                return {key: conn[key]}

        # fallback (if create_cloud_secret already knows how to handle your row)
        return dict(conn)

    # ---------- Default fallback ----------
    return dict(conn)


async def submit_ray_job(
    *,
    connection_id: str,
    dataset_id: str,
    data_source_uri: str,
    ray_ns: str,
    template_path: str,
):
    # 1) resolve creds server-side (ASYNC)
    conn = await get_cloud_connection(connection_id)

    provider = _normalize_provider(conn.get("provider") or conn.get("backend"))
    if not provider:
        raise ValueError(f"Missing provider in connection row for id={connection_id}")

    creds = _build_creds(provider, conn)

    # 2) create k8s secret
    secret_name = create_cloud_secret(
        namespace=ray_ns,
        provider=provider,
        creds=creds,
        dataset_id=dataset_id,
    )

    try:
        # 3) unique RayJob name per run
        rayjob_name = f"rayjob-{dataset_id[:8]}-{int(time.time())}"

        # 4) render YAML with secret name + unique job name
        yaml_text = render_rayjob_yaml_from_file(
            template_path,
            rayjob_name=rayjob_name,
            ray_namespace=ray_ns,
            data_source_uri=data_source_uri,
            cloud_secret_name=secret_name,
        )

        # 5) apply + wait + logs
        # (Works whether runner accepts rayjob_name or not)
        try:
            return run_rayjob_from_yaml(
                yaml_text,
                namespace=ray_ns,
                rayjob_name=rayjob_name,
            )
        except TypeError:
            return run_rayjob_from_yaml(
                yaml_text,
                namespace=ray_ns,
            )

    finally:
        # 6) always cleanup secret
        kubectl_delete_secret(secret_name, ray_ns)


def submit_ray_job_sync(**kwargs):
    """
    Convenience wrapper for scripts / CLI usage.
    """
    return asyncio.run(submit_ray_job(**kwargs))

