import json
import os
import docker
import logging
from pathlib import Path
from typing import Optional
import uuid

from app.data_transfer_docker.run_result import TransferResult, parse_transfer_result

logger = logging.getLogger(__name__)

# In-container path of the GCS service-account key (under the mounted job dir).
# The generated injection script authenticates to GCS purely via
# GOOGLE_APPLICATION_CREDENTIALS (GCSConfig()/google.cloud.storage.Client() with
# no explicit creds), so we write the key here and point that env var at it.
_GCS_SA_FILENAME = "gcs_sa.json"


def _write_container_gcs_sa(job_dir: Path, job_id: str, gcp_sa_json) -> Optional[str]:
    """Resolve the GCS service-account key for the container and write it into the
    (host-side, then mounted) job dir. Returns the in-container path to expose as
    GOOGLE_APPLICATION_CREDENTIALS, or None when no usable key is found.

    Precedence:
      1. ``gcp_sa_json`` — the connection's OWN stored service-account key (JSON
         string or dict), passed by the planner. Preferred, and works on GKE too.
      2. Fallback — the HOST's ``GOOGLE_APPLICATION_CREDENTIALS`` file, ONLY when
         DTA_ALLOW_AMBIENT_GCP_CREDS=1: on a deployed host that file is the
         PLATFORM's key, so an unattended fallback would run a tenant's transfer
         as the platform. Same gate as ray_job_runner (keep in sync).
    """
    sa_text: Optional[str] = None
    origin = None
    if gcp_sa_json:
        sa_text = json.dumps(gcp_sa_json) if isinstance(gcp_sa_json, dict) else str(gcp_sa_json)
        origin = "connection service-account key"
    elif os.getenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "0").strip().lower() in ("1", "true", "yes"):
        # Strip surrounding quotes: on Windows `set VAR="C:\path"` keeps the quotes
        # in the value, so the raw env would point at a nonexistent quoted filename.
        host_adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip().strip('"').strip("'")
        if host_adc and os.path.isfile(host_adc):
            try:
                with open(host_adc, "r", encoding="utf-8") as f:
                    sa_text = f.read()
                origin = f"host GOOGLE_APPLICATION_CREDENTIALS ({host_adc})"
            except OSError as e:
                logger.warning("Could not read host GOOGLE_APPLICATION_CREDENTIALS %s: %s", host_adc, e)
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        logger.warning(
            "No connection service-account key for this transfer. The host's "
            "GOOGLE_APPLICATION_CREDENTIALS is set but is NOT used as a fallback (it is "
            "the platform's identity, not the tenant's); the container will be anonymous "
            "to GCS. Set DTA_ALLOW_AMBIENT_GCP_CREDS=1 to opt in on a dev machine."
        )

    if not sa_text:
        return None
    try:
        json.loads(sa_text)  # fail fast if it isn't valid JSON
    except (ValueError, TypeError) as e:
        logger.warning("GCS service-account key is not valid JSON; skipping GCS creds injection: %s", e)
        return None

    sa_path = job_dir / _GCS_SA_FILENAME
    with open(sa_path, "w", encoding="utf-8") as f:
        f.write(sa_text)
    logger.info("Wrote GCS service-account key for container auth from %s: %s", origin, sa_path)
    return f"/mnt/gcs/transfers/{job_id}/{_GCS_SA_FILENAME}"


def launch_docker_pipeline(injection_script: str, job_id: str = None, extra_volumes: dict = None,
                           cloud_env: dict = None, gcp_sa_json=None, dest_gcp_sa_json=None):
    """
    Saves the generated injection script to a mock GCS directory structure,
    then launches the Docker container to execute it.

    The container expects:
      - SCRIPTS_MOUNT_PATH env var pointing to the GCS mount root
      - JOB_ID env var for the subdirectory
      - The script at: {SCRIPTS_MOUNT_PATH}/{JOB_ID}/injection_script.py

    ``cloud_env`` (optional) is a dict of runtime env vars the container needs to
    reach an S3/Azure bucket (e.g. AWS_ACCESS_KEY_ID). GCS auth is handled
    separately; merged into the container environment when provided.

    ``gcp_sa_json`` (optional) is the connection's GCS service-account key (a JSON
    string or already-parsed dict). When provided it is written into the mounted
    job dir and exposed as GOOGLE_APPLICATION_CREDENTIALS so the container reads/
    writes GCS as that service account — mirroring the GKE runner. When the
    connection has no stored key, the runner falls back to the HOST's ambient
    GOOGLE_APPLICATION_CREDENTIALS file (a dev-env export) — see
    ``_write_container_gcs_sa``. Without either, a GCS transfer forced to local
    Docker (e.g. a VPC-private DB destination) falls back to anonymous access and
    fails with 'storage.objects.get denied'.

    ``dest_gcp_sa_json`` (optional) is the *destination* connection's GCS key; when
    set it is exposed as DEST_GCP_SA_JSON so a cloud->cloud write authenticates as
    the destination's own SA rather than the (source) ambient identity above.
    """
    if job_id is None:
        job_id = f"run_{uuid.uuid4().hex[:8]}"

    logger.info(f"--- STARTING DOCKER EXECUTION FOR JOB: {job_id} ---")

    # 1. Build the mock GCS directory structure
    mock_gcs_root = Path(os.getcwd()) / "pipeline_runs"
    job_dir = mock_gcs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    script_path = job_dir / "injection_script.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(injection_script)
    logger.info(f"Saved injection script to: {script_path}")

    # 1b. Write the GCS service-account key into the (mounted) job dir so the
    # container can authenticate to GCS via GOOGLE_APPLICATION_CREDENTIALS. Uses the
    # connection's key if present, else the host's ambient credentials (see helper).
    gcs_creds_container_path = _write_container_gcs_sa(job_dir, job_id, gcp_sa_json)

    # 2. Initialize Docker client
    client = docker.from_env()
    container_name = f"dta_{job_id}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Launching container: {container_name}")

    # 3. Setup Volumes
    volumes = {
        str(mock_gcs_root.absolute()): {
            'bind': '/mnt/gcs/transfers',
            'mode': 'ro'
        }
    }
    
    if extra_volumes:
        volumes.update(extra_volumes)

    # Base container env, plus any S3/Azure creds the generated script reads.
    environment = {
        "JOB_ID": job_id,
        "RAY_HEAD_ADDRESS": "local",
        "SCRIPTS_MOUNT_PATH": "/mnt/gcs/transfers",
        # This runner is a SINGLE container -- a local Ray cluster here adds no
        # parallelism over Daft's native runner (both use the same CPUs) but
        # costs ~45s of startup: measured 35s in ray.init() + 10s in
        # set_runner_ray() on a transfer whose real work took 17s. That pushed
        # the request past the frontend proxy's ceiling, so a successful
        # transfer surfaced to the user as an unknown outcome. The GKE/KubeRay
        # path is unaffected -- it ignores this flag (see ray_enabled() in
        # container_file.py). Set AVALOKA_DTA_DOCKER_USE_RAY=1 to restore Ray,
        # e.g. to reproduce a cluster-only issue locally.
        "DTA_USE_RAY": (
            "1"
            if os.getenv("AVALOKA_DTA_DOCKER_USE_RAY", "0").strip().lower() in ("1", "true", "yes")
            else "0"
        ),
    }
    if cloud_env:
        environment.update({k: str(v) for k, v in cloud_env.items() if v})
    # GCS auth: point the daft/google client at the mounted service-account key.
    if gcs_creds_container_path:
        environment["GOOGLE_APPLICATION_CREDENTIALS"] = gcs_creds_container_path
    # Destination-write SA (cloud->cloud): the upload helper reads DEST_GCP_SA_JSON
    # to write as the destination's own identity instead of the ambient SA above.
    if dest_gcp_sa_json:
        environment["DEST_GCP_SA_JSON"] = (
            json.dumps(dest_gcp_sa_json) if isinstance(dest_gcp_sa_json, dict) else str(dest_gcp_sa_json)
        )

    container = None
    try:
        container = client.containers.run(
            image="avaloka-dta-local:latest",
            name=container_name,

            # --- THE FIX: Run the container as the root admin ---
            user="root",

            environment=environment,
            volumes=volumes,
            extra_hosts={"host.docker.internal": "host-gateway"},

            # Docker gives a container only 64MB of /dev/shm by default. Ray sizes its
            # object store from shared memory and, below ~30% of available RAM, silently
            # falls back to /tmp (disk) while warning "This will harm performance!".
            # Sizing /dev/shm up keeps the object store in memory. It is a tmpfs, so pages
            # are allocated lazily — this is a ceiling, not a reservation.
            shm_size=os.getenv("AVALOKA_DTA_DOCKER_SHM_SIZE", "2g"),

            detach=True
        )

        logger.info(f"Container {container.name} started. Waiting for completion...")
        # Unbounded wait() would hang the whole planner turn on a stalled container.
        # On timeout the finally block force-removes it. 0 disables the ceiling.
        _wait_timeout = int(os.getenv("AVALOKA_DTA_DOCKER_TIMEOUT_SECONDS", "3600") or 0)
        try:
            result = container.wait(timeout=_wait_timeout or None)
        except Exception:
            logger.error(
                "Container %s did not finish within %ss — treating as failed and "
                "removing it.", container.name, _wait_timeout,
            )
            return TransferResult(success=False)

        # errors="replace": binary log bytes must not fail an exit-code-0 transfer.
        logs = container.logs().decode("utf-8", errors="replace")
        print(f"\n--- 🐳 DOCKER CONTAINER LOGS ---\n{logs}\n--------------------------------\n")

        exit_code = result['StatusCode']
        if exit_code != 0:
            logger.error(f"Container exited with code {exit_code}")
            return parse_transfer_result(False, logs)

        logger.info("Container finished successfully!")
        return parse_transfer_result(True, logs)

    except Exception as e:
        logger.error(f"Docker execution error: {e}", exc_info=True)
        return TransferResult(success=False)

    finally:
        # Always remove the container so finished runs don't pile up.
        if container is not None:
            try:
                container.remove(force=True)
                logger.info(f"Removed container: {container_name}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}")