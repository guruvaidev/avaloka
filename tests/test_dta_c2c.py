import os
import uuid
from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials
from app.data_transfer_docker.gke_run import launch_gke_pipeline


def run_cloud_to_cloud_test():
    print(f"\n{'='*80}")
    print(f"🚀 STARTING CLOUD-TO-CLOUD TRANSFER TEST (GCS → GCS)")
    print(f"{'='*80}")

    # ── 1. Define cloud credentials ───────────────────────────────────────────
    # No key material here — authentication is handled by the Kubernetes secret
    # mounted into the Ray pods via GOOGLE_APPLICATION_CREDENTIALS.
    # We only need to tell the pipeline where the files live.

    source_creds = cloud_storage_credentials(
        provider="gcp",
        bucket_name="avaloka-test-user-filestore",
        file_path="patient_data.csv"
    )

    # source_creds = cloud_storage_credentials(
    #     provider="gcp",
    #     bucket_name="avaloka-test-user-filestore",
    #     file_path="test_c2c_jyothi/patient_data.csv"
    # )

    dest_creds = cloud_storage_credentials(
        provider="gcp",
        bucket_name="avaloka-dta-destination",
        file_path="transfers/patient_data_transformed.json"
    )

    # dest_creds = cloud_storage_credentials(su
    #     provider="gcp",
    #     bucket_name="avaloka-dta-destination",
    #     file_path="transfers/2019_nov_data_transformed.json"
    # )
    # ── 2. Define prompt ──────────────────────────────────────────────────────
    prompt = (
        "Read the patient data CSV file. "
        "Keep all original columns and all rows. "
        "Add a new column called 'billing_category': "
        "if Billing Amount is greater than 30000 set it to 'High', otherwise set it to 'Low'. "
        "only output patient_id and billing_category columns. "
        "Write the result to the destination file."
    )

    # ── 3. Run the pipeline ───────────────────────────────────────────────────
    print("\n📋 Running data transfer pipeline (code generation)...")

    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="csv",
        destination_type="json",
        source_credentials=source_creds,
        destination_credentials=dest_creds,
        source_file=source_creds.get_cloud_uri("csv"),
        destination_file=dest_creds.get_cloud_uri("json")
    )

    # ── 4. Print diagnostic info regardless of outcome ────────────────────────
    print("\n🔍 PIPELINE DIAGNOSTICS:")
    print(f"   Code generated successfully : {final_state.get('code_generated_successfully')}")
    print(f"   Destination schema compatible: {final_state.get('destination_schema_compatible')}")
    print(f"   Source URI                  : {source_creds.get_cloud_uri('csv')}")
    print(f"   Destination URI             : {dest_creds.get_cloud_uri('json')}")

    if final_state.get('error_message'):
        print(f"\n❌ Pipeline error: {final_state.get('error_message')}")

    if final_state.get('warnings'):
        for w in final_state.get('warnings'):
            print(f"⚠️  Warning: {w}")

    # ── 5. Print the generated script so we can verify URIs before launching ──
    if injection_script:
        print("\n📜 GENERATED INJECTION SCRIPT:")
        print("-" * 60)
        print(injection_script)
        print("-" * 60)
    else:
        print("\n❌ No injection script was generated. Aborting.")
        return

    # ── 6. Gate on schema compatibility before launching ─────────────────────
    if not final_state.get('destination_schema_compatible'):
        print("\n❌ Schema compatibility check failed. Not launching GKE job.")
        print(f"   Error: {final_state.get('error_message')}")
        return

    # ── 7. Launch on GKE ──────────────────────────────────────────────────────
    job_id = f"cloud-test-{uuid.uuid4().hex[:6]}"
    print(f"\n☁️  Launching GKE Ray job: {job_id}")
    print("   (This will take 2-4 minutes for cluster spin-up + execution)")

    success = launch_gke_pipeline(injection_script, job_id=job_id)

    # ── 8. Report outcome ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    if success:
        print(f"🎉 SUCCESS! Transfer complete.")
        print(f"   Check your destination bucket at:")
        print(f"   https://console.cloud.google.com/storage/browser/avaloka-dta-destination/transfers")
        print(f"   Expected file: patient_data_transformed.json")
    else:
        print(f"❌ GKE execution failed.")
        print(f"   Check Ray dashboard at: http://34.30.130.231:8265")
        print(f"   Look for job: dta-{job_id}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_cloud_to_cloud_test()