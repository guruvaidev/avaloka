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

def run_d2c_test():
    print(f"\n{'='*80}")
    print(f"🚀 STARTING DATABASE-TO-CLOUD TRANSFER TEST (MySQL → GCS)")
    print(f"{'='*80}")

    # ── 1. Define credentials ─────────────────────────────────────────────────
    source_creds = {
        "host": "10.111.16.10",
        "port": 3306,
        "user": "avaloka",
        "password": "avaL0kap@ssword",
        "database": "cali_db",
        "table": "housing",
        "ssl": {"ssl_disabled": False}  
    }

    dest_creds = cloud_storage_credentials(
        provider="gcp",
        bucket_name="avaloka-dta-destination",
        file_path="transfers/housing_data_d2c.parquet"
    )

    # ── 2. Define prompt ──────────────────────────────────────────────────────
    prompt = (
        "Read the housing data from the source database. "
        "Filter the dataset to include rows where 'housing_median_age' is less than 15. "
        "Output the 'longitude', 'latitude', 'population', and 'median_house_value' columns. "
        "Write the result to the destination parquet file."
    )

    # ── 3. Run the pipeline ───────────────────────────────────────────────────
    print("\n📋 Running data transfer pipeline (code generation)...")

    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="mysql",
        destination_type="parquet",
        source_credentials=source_creds,
        destination_credentials=dest_creds,
        destination_file=dest_creds.get_cloud_uri("parquet")
    )

    coder_def = final_state.get('coder_definition', {})

    # ── 4. Validation & Execution ─────────────────────────────────────────────
    if any([coder_def.get('syntax_error'),
            coder_def.get('static_semantic_error'),
            coder_def.get('execution_error'),
            coder_def.get('logical_semantic_error')]):
        print("\n❌ Code generation pipeline failed with internal errors.")
        return

    if not injection_script:
        print("\n❌ No injection script was generated. Aborting.")
        return

    print("\n" + "-"*60)
    print("📜 FINAL GENERATED INJECTION SCRIPT")
    print("-" * 60)
    print(injection_script)
    print("-" * 60 + "\n")

    job_id = f"d2c-test-{uuid.uuid4().hex[:6]}"
    print(f"\n🔑 Using Job ID: {job_id}")

    success = execute_pipeline(injection_script, job_id=job_id)

    # ── 5. Report outcome ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    if success:
        print(f"🎉 SUCCESS! Transfer complete.")
        print(f"   Check your destination bucket at:")
        print(f"   https://console.cloud.google.com/storage/browser/avaloka-dta-destination/transfers")
    else:
        print(f"❌ Execution failed. Check runner logs.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_d2c_test()