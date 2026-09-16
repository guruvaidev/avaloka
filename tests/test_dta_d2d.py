import torch
import os
import logging
from datetime import datetime
import uuid

from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
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

def setup_logging():
    """Routes all pipeline logs to a file instead of the terminal."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_run_{timestamp}.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Optional: uncomment if you want to suppress console logging entirely
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    return log_file

def run_single_transfer(test_name, prompt, source_creds, dest_creds):
    """Handles the E2E pipeline for a single data transfer request."""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Destination Table: {dest_creds['table']}")
    print("⚙️ Running agent pipeline to generate injection script (this may take a moment)...")
    
    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="mysql",   
        destination_type="postgres", 
        source_credentials=source_creds,
        destination_credentials=dest_creds
    )

    coder_def = final_state.get('coder_definition', {})

    # ==========================================
    # PIPELINE VALIDATION & ROUTING
    # ==========================================
    # 0. Source dataset was empty after applying the filter — abort cleanly
    if coder_def.get('empty_source'):                               # ← NEW CASE 0
        abort_reason = coder_def.get(
            'pipeline_abort_reason',
            "The source dataset returned 0 rows after applying the filter condition."
        )
        print(f"\n⚠️  [{test_name}] PIPELINE HALTED: Empty Source Dataset")
        print("The filter condition in your prompt matched no rows in the source table.")
        print(f"Reason : {abort_reason}")
        print("Action : Verify the filter values exist in the source data before retrying.")
        return False
    # 1. Check if code generation failed entirely
    elif any([coder_def.get('syntax_error'),
            coder_def.get('static_semantic_error'),
            coder_def.get('execution_error'),
            coder_def.get('logical_semantic_error')]):
        
        print(f"\n❌ [{test_name}] Code generation pipeline failed with internal errors.")
        print("Check the generated log file for specific traceback details.")
        return False

    # 2. Check if code generated, but schemas DO NOT match
    elif final_state.get('destination_schema_compatible') is False:
        print(f"\n⚠️ [{test_name}] PIPELINE HALTED: Schema Mismatch")
        print("The agent successfully wrote the ETL code, but the output data doesn't match the destination table.")
        print(f"Agent's Output Schema: {coder_def.get('generated_output_schema')}")
        print(f"Destination DB Schema: {final_state.get('destination_schema')}")
        print("Action Required: Please update your destination database table to match the Agent's Output Schema.")
        return False

    # 3. Everything is perfect! Show script and Run Docker.
    elif final_state.get('destination_schema_compatible') is True and injection_script:
        print(f'\n✅ [{test_name}] Code generation pipeline succeeded AND schemas are compatible!')
        
        print("\n" + "-"*60)
        print("📜 FINAL GENERATED INJECTION SCRIPT")
        print("-"*60)
        print(injection_script)
        print("-"*60 + "\n")
        
        print("🐳 Triggering runner. Showing container logs below:\n")

        job_id = f"demo-{test_name}-{uuid.uuid4().hex[:6]}"
        print(f"🔑 Using Job ID: {job_id}")
        
        success = execute_pipeline(injection_script, job_id=job_id)

        if success:
            print(f"\n🎉 [{test_name}] SUCCESS! Full pipeline ran successfully.")
            return True
        else:
            print(f"\n❌ [{test_name}] Pipeline execution failed. Check runner logs above.")
            return False
            
    # 4. Fallback
    else:
        print(f'\n⚠️ [{test_name}] Pipeline completed, but no injection script was returned and no clear error was caught.')
        return False

def run_sequential_tests():
    log_file = setup_logging()
    print(f"📁 Detailed pipeline logs for all tests are being saved to: {log_file}")

    # The source database is the same for all tests
    source_creds = {
        "host": "10.111.16.10",
        "port": 3306,
        "user": "avaloka",
        "password": "avaL0kap@ssword",
        "database": "cali_db",
        "table": "housing",
        "ssl": {"ssl_disabled": False}  
    }
    # source_creds = {
    #     "user": "root",
    #     "password": "rootpassword",
    #     "host": "host.docker.internal", 
    #     "port": 3309, 
    #     "database": "cali_db",
    #     "table": "housing",
    #     "ssl": {"ssl_disabled": False}
    # }

    
    # Base destination credentials (we will copy and update the 'table' per test)
    base_dest_creds = {
        "host": "10.111.16.4",
        "port": 5432,
        "user": "avaloka",
        "password": "avaL0kapa$$word",
        "database": "avaloka_dest",
        "sslmode": "require"   
        # "user": "avaloka",
        # "password": "avalokapassword",
        # "host": "host.docker.internal",       
        # "port": 5436, # Match destination PostgreSQL port
        # "database": "avaloka_dest",
        # "sslmode": "disable"
    }

    # Define the 3 distinct test cases
    test_cases = [
        {
            "name": "high-income-filter",
            "prompt": "Filter the dataset to include only rows where 'median_income' is greater than 8.0 and 'total_bedrooms' is less than 50. Output the 'longitude', 'latitude', and 'median_house_value' columns.",
            "dest_table": "housing_high_income"
        },
        {
            "name": "new-builds-demographics",
            "prompt": "Filter the dataset to include rows where 'housing_median_age' is less than 10. Output the 'population', 'households', and 'median_house_value' columns.",
            "dest_table": "housing_new_builds"
        },
        {
            "name": "inland-rooms",
            "prompt": "Filter the dataset to only include rows where 'ocean_proximity' is exactly 'INLAND'. Then, output only the 'total_rooms', 'total_bedrooms', and 'median_income' columns.",
            "dest_table": "housing_inland_rooms"
        }
    ]

    # Execute tests sequentially
    for idx, test in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"EXECUTING TRANSFER {idx} OF {len(test_cases)}")
        print(f"#{'#'*79}")
        
        # Create a fresh copy of dest_creds with the specific table for this test
        current_dest_creds = base_dest_creds.copy()
        current_dest_creds["table"] = test["dest_table"]
        
        run_single_transfer(
            test_name=test["name"],
            prompt=test["prompt"],
            source_creds=source_creds,
            dest_creds=current_dest_creds
        )
        
    print("\n🏁 All sequential data transfers have completed.")

if __name__ == "__main__":
    run_sequential_tests()