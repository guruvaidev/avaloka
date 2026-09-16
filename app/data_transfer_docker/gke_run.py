import os
import logging
import uuid
from pathlib import Path

# Reuse the Chameleon runner from the other workflow!
from app.infra.ray_job_runner import run_rayjob_from_yaml, RayJobOutcome
from app.data_transfer_docker.run_result import TransferResult, parse_transfer_result

logger = logging.getLogger(__name__)

GCS_SA_SECRET_NAME = "gcs-sa-key"

def _render_cloud_env_vars(cloud_env: dict = None) -> str:
    """Render an env-var dict as YAML lines for the RayJob runtimeEnvYAML block.

    Each key is emitted at the 6-space indent it needs under ``env_vars:``; the
    replace in launch_gke_pipeline consumes the placeholder's own leading spaces.
    Returns ``""`` when there are no cloud creds (GCS uses the mounted secret).
    """
    if not cloud_env:
        return ""

    def _esc(v: str) -> str:
        # YAML double-quoted scalar: escape backslashes then double-quotes.
        return str(v).replace("\\", "\\\\").replace('"', '\\"')

    return "\n".join(f'      {k}: "{_esc(v)}"' for k, v in cloud_env.items() if v)


def launch_gke_pipeline(injection_script: str, job_id: str = None, cloud_env: dict = None,
                        gcp_sa_json: str = None, dest_gcp_sa_json: str = None,
                        dashboard_url: str = None) -> TransferResult:
    """
    Renders the RayJob YAML with the AI-generated script and hands it
    off to the Chameleon runner (which handles Static vs Ephemeral logic).

    ``cloud_env`` (optional) is a dict of runtime env vars the pods need to reach
    an S3/Azure bucket (e.g. AWS_ACCESS_KEY_ID). ``gcp_sa_json`` (optional) is the
    connection's GCS service-account key; it is forwarded to the Ray runner, which
    writes it into the job's working dir and exposes GOOGLE_APPLICATION_CREDENTIALS
    so the job reads/writes GCS as that service account (in direct-submission mode
    the cluster's default identity is used otherwise, which may lack bucket access).
    ``dest_gcp_sa_json`` (optional) is the *destination* connection's GCS key; it is
    forwarded as DEST_GCP_SA_JSON so a cloud->cloud write authenticates as the
    destination's own SA instead of the (source) ambient identity above.
    """

    # Respect the operator-provided RAY_DASHBOARD_URL (e.g. exported by the
    # startup script or set in .env) instead of forcing a hard-coded IP that
    # goes stale. Fail clearly if it is missing.
    ray_dashboard_url = os.environ.get("RAY_DASHBOARD_URL")
    if not ray_dashboard_url:
        raise RuntimeError(
            "RAY_DASHBOARD_URL is not set. Export it (or add it to .env) "
            "pointing at the Ray dashboard"
        )
    logger.info("Using Ray dashboard URL: %s", ray_dashboard_url)

    if job_id is None:
        job_id = f"run-{uuid.uuid4().hex[:8]}"

    # K8s object names must be lower-case alphanumeric or '-'
    rayjob_name = f"dta-{job_id}".lower().replace("_", "-")
    namespace = os.environ.get("KUBE_NAMESPACE", "default")
    
    logger.info(f"--- STARTING GKE EXECUTION FOR JOB: {rayjob_name} ---")

    # 1. Read the raw YAML template
    template_path = Path(__file__).parent.parent / "infra" / "rayjob_data_transfer.yaml"
    with open(template_path, "r", encoding="utf-8") as f:
        yaml_text = f.read()

    # 2. Indent every script line to 4 spaces so it's valid YAML block-scalar
    # content under sample_code.py. The template's "    __INJECTION_SCRIPT__"
    # placeholder is matched WITH its 4 leading spaces (see the replace below) so
    # those are consumed — preventing a double-indent (first line 8 spaces, rest
    # 4 -> block scalar terminates -> invalid YAML).
    indented_script = "\n".join(
        f"    {line}" if line.strip() else "" for line in injection_script.split("\n")
    )

    # 3. Inject variables into the YAML
    yaml_text = yaml_text.replace("__RAYJOB_NAME__", rayjob_name)
    yaml_text = yaml_text.replace("__JOB_ID__", job_id)
    yaml_text = yaml_text.replace("    __INJECTION_SCRIPT__", indented_script)
    yaml_text = yaml_text.replace("__GCS_SECRET_NAME__", GCS_SA_SECRET_NAME)
    yaml_text = yaml_text.replace("      __CLOUD_ENV_VARS__", _render_cloud_env_vars(cloud_env))

    # 4. Hand off to the reusable Chameleon runner. Forward the connection's GCS
    # service-account key so direct-submission mode authenticates as that SA.
    cloud_creds = None
    if gcp_sa_json or dest_gcp_sa_json:
        cloud_creds = {}
        if gcp_sa_json:
            cloud_creds["gcp_service_account_json"] = gcp_sa_json
        if dest_gcp_sa_json:
            cloud_creds["dest_gcp_service_account_json"] = dest_gcp_sa_json

    try:
        outcome: RayJobOutcome = run_rayjob_from_yaml(
            yaml_text=yaml_text,
            namespace=namespace,
            rayjob_name=rayjob_name,
            cloud_creds=cloud_creds,
            dashboard_url=dashboard_url,
        )

        # 5. Output logs fetched natively by the runner
        if outcome.logs:
            print(f"\n--- ☁️ RAY RUNNER LOGS ({outcome.status}) ---\n{outcome.logs}\n" + "-"*40 + "\n")

        # 6. Evaluate success based on the RayJobOutcome dataclass
        if outcome.status == "SUCCEEDED":
            logger.info(f"🎉 Pipeline execution succeeded for {rayjob_name}!")
            return parse_transfer_result(True, outcome.logs)
        else:
            logger.error(f"❌ Pipeline failed with status: {outcome.status}")
            if outcome.details:
                logger.error(f"Error Details: {outcome.details}")
            return parse_transfer_result(False, outcome.logs)

    except Exception as e:
        logger.error(f"Critical failure executing RayJob via runner: {e}", exc_info=True)
        return TransferResult(success=False)