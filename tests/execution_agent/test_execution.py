""" Testing the execution of the code on the infrastructure """

from tests._quarantine import requires_api

requires_api("app.execution_helper", replacement="OUTPUT_DIR moved out of app.execution_helper, which was deleted; repair against the current execution-agent API or remove this file.")


import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import pytest

load_dotenv("~/Documents/kamaliinc/avaloka-prototypes/avaloka-agentic-workflow/.env")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.execution_agent import execution_agent_node
from app.execution_helper import OUTPUT_DIR
from app.graph.etl_state import ETLState 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("execution_test")

DATA_DIR = Path(__file__).parent.parent / "data"
IRIS_PATH = DATA_DIR / "iris.csv"

# Sample Python code from the coding agent
SAMPLE_CODE = """
import pandas as pd
import os
from datetime import datetime

input_file = os.environ.get('INPUT_FILE', 'input.csv')
output_file = os.environ.get('OUTPUT_FILE', 'output.csv')

df = pd.read_csv(input_file)
result = df.groupby('species').mean().reset_index()
result.to_csv(output_file, index=False)
print('done')
"""

@pytest.fixture(scope="function")
def clean_dir():
    for f in OUTPUT_DIR.glob("*.csv"):
        f.unlink()
    yield
    for f in OUTPUT_DIR.glob("*.csv"):
        logger.info("output: %s", f)


def get_platform():
    """Get the current platform based on environment variables"""
    if os.environ.get("GCP_PROJECT_ID"):
        return "gcp"
    if os.environ.get("AWS_REGION") and os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN"):
        return "aws"
    return None


def check_env_setup():
    """Check if environment variables are properly set up"""
    platform = get_platform()
    if platform is None:
        logger.error("Environment variables not set up properly!")
        logger.error("For GCP: Set GCP_PROJECT_ID")
        logger.error("For AWS: Set AWS_REGION and AWS_EKS_CLUSTER_ROLE_ARN")
        logger.error(f"Expected .env file location: {env_path}")
        return False
    return True


@pytest.mark.skipif(not check_env_setup(), reason="Environment variables not configured. Please export GCP_PROJECT_ID and setup the .env")
def test_e2e(clean_dir):
    # Create a state object with the sample code and the iris data
    state = ETLState(
        messages=[],
        planner_definition={"task": "Group iris.csv by species and avg"},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={"code": SAMPLE_CODE},
        infrastructure_request={},
        infrastructure_provisioned=None,
        data_source_location=str(IRIS_PATH),
        sample_data=str(IRIS_PATH),
        infrastructure_preference=get_platform(),
    )

    start = time.time() # Start time
    result = execution_agent_node(state) # Execute the code
    logger.info("runtime %.2fs", time.time() - start) # Log the runtime

    assert result["execution_result"]["status"] == "success" # Check if the execution was successful
    files = list(OUTPUT_DIR.glob("*.csv")) 
    assert files, "no csv output" 
    df_out = pd.read_csv(files[0]) # Read the output file
    df_src = pd.read_csv(IRIS_PATH) # Read the input file
    assert set(df_out["species"]) == set(df_src["species"]) # Check if the output file has the same species as the input file