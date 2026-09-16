import os
import uuid
import logging
from datetime import datetime

from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials
from app.data_transfer_docker.docker_run import launch_docker_pipeline
from app.data_transfer_docker.gke_run import launch_gke_pipeline

def execute_pipeline(injection_script, job_id):
    execution_env = os.environ.get("EXECUTION_ENV", "gke")
    if execution_env == "gke":
        print("☁️ Routing execution to GKE Ray Cluster...")
        return launch_gke_pipeline(injection_script, job_id=job_id)
    else:
        print("🐳 Routing execution to Local Docker...")
        return launch_docker_pipeline(injection_script, job_id=job_id)

def run_c2d_test():
    print(f"\n{'='*80}")
    print(f"🚀 STARTING CLOUD-TO-DATABASE TRANSFER TEST (GCS → Postgres)")
    print(f"{'='*80}")

    # ── 1. Define credentials ─────────────────────────────────────────────────
    source_creds = cloud_storage_credentials(
        provider="gcp",
        bucket_name="avaloka-test-user-filestore",
        file_path="patient_data.csv"
    )

    dest_creds = {
        "host": "10.111.16.4",
        "port": 5432,
        "user": "avaloka",
        "password": "avaL0kapa$$word",
        "database": "avaloka_dest",
        "table": "patient_data_c2d", # Table will be created by the pipeline script
        "sslmode": "require"           
    }

    # ── 2. Define prompt ──────────────────────────────────────────────────────
    prompt = (
        "Read the patient data CSV file. "
        "Keep all original columns. "
        "Add a new column called 'risk_status': if Billing Amount is > 20000 set to 'High', else 'Low'. "
        "Write the result to the destination database table."
    )

    # ── 3. Run the pipeline ───────────────────────────────────────────────────
    print("\n📋 Running data transfer pipeline (code generation)...")

    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="csv",
        destination_type="postgres",
        source_credentials=source_creds,
        destination_credentials=dest_creds,
        source_file=source_creds.get_cloud_uri("csv")
    )

    coder_def = final_state.get('coder_definition', {})

    # ── 4. Validation & Execution ─────────────────────────────────────────────
    if any([coder_def.get('syntax_error'),
            coder_def.get('static_semantic_error'),
            coder_def.get('execution_error'),
            coder_def.get('logical_semantic_error')]):
        print("\n❌ Code generation pipeline failed with internal errors.")
        return

    if final_state.get('destination_schema_compatible') is False:
        print("\n⚠️ PIPELINE HALTED: Schema Mismatch")
        print(f"Agent's Output Schema: {coder_def.get('generated_output_schema')}")
        print(f"Destination DB Schema: {final_state.get('destination_schema')}")
        return

    if not injection_script:
        print("\n❌ No injection script was generated. Aborting.")
        return

    print("\n" + "-"*60)
    print("📜 FINAL GENERATED INJECTION SCRIPT")
    print("-" * 60)
    print(injection_script)
    print("-" * 60 + "\n")

    job_id = f"c2d-test-{uuid.uuid4().hex[:6]}"
    print(f"\n🔑 Using Job ID: {job_id}")

    success = execute_pipeline(injection_script, job_id=job_id)

    # ── 5. Report outcome ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    if success:
        print(f"🎉 SUCCESS! Transfer complete.")
        print(f"   Check your destination database table: {dest_creds['table']}")
    else:
        print(f"❌ Execution failed. Check runner logs.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_c2d_test()