import torch
import os
import uuid
from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
from app.data_transfer_docker.docker_run import launch_docker_pipeline

def run_file_transfer_test():
    print(f"\n{'='*80}")
    print(f"🚀 STARTING FILE-TO-FILE TRANSFER TEST")
    print(f"{'='*80}")

    # 1. Define Paths 
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    local_base_path = os.path.abspath(os.path.join(current_script_dir, "..", "app", "sample_data"))
    local_source_file = os.path.join(local_base_path, "salaries.csv")

    docker_source_path = "/app/sample_data/salaries.csv"
    docker_dest_path = "/app/sample_data/salaries_transformed.csv"

    source_creds = {"file_path": local_source_file}
    dest_creds = {"file_path": docker_dest_path}

    prompt = (
        "read the csv file, salaries.csv"
        "filter for the columns where salary is greater than 50000"
        "Calculate a new column bonus which is salary multiplied by 0.1 but this is only for the rows where salary is greater than 50000, otherwise set bonus to 0"
        "write the result to a new csv file called salaries_transformed.csv"
        "i want to see all columns and all rows, regardless of the salary"
    )

    # 4. Run the Agent Pipeline
    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="csv",   
        destination_type="csv", 
        source_credentials=source_creds,
        destination_credentials=dest_creds,
        source_file=docker_source_path,
        destination_file=docker_dest_path
    )

    # 5. Validation and Execution
    if final_state.get('destination_schema_compatible') is not False and injection_script:
        print(f'\n✅ Code generation succeeded!')
        print("\n📜 GENERATED SCRIPT SNIPPET:")
        print("-" * 30)
        print(injection_script[:500] + "...") 
        print("-" * 30)

        job_id = f"file-test-{uuid.uuid4().hex[:6]}"
        execution_env = os.environ.get("EXECUTION_ENV", "docker")
        
        print(f"📦 Triggering {execution_env} execution...")
        if execution_env == "docker":
            
            # FIX 2: Mount the local sample data directory into the container
            # Make sure mode is 'rw' (read-write) since we are writing the output file here
            extra_vols = {
                str(local_base_path): {
                    'bind': '/app/sample_data',
                    'mode': 'rw'
                }
            }
            
            success = launch_docker_pipeline(
                injection_script, 
                job_id=job_id, 
                extra_volumes=extra_vols # Pass the volumes here!
            )
        else:
            from app.data_transfer_docker.gke_run import launch_gke_pipeline
            success = launch_gke_pipeline(injection_script, job_id=job_id)

        if success:
            print(f"\n🎉 SUCCESS! Transformed file should be at: {docker_dest_path}")
        else:
            print(f"\n❌ Execution failed.")
    else:
        print("\n❌ Pipeline failed to generate valid code for the file transfer.")

if __name__ == "__main__":
    run_file_transfer_test()