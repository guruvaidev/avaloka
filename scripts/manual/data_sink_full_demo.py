'''
Docstring for avaloka-dev.tests.test_data_sink_full

Step 1: initialize the etl state object: user prompt, input csv details, target db config
Step 2: initialize coding state of daft coder
Step 3: run daft coder to generate code for etl
Step 4: daft coder must also output the template for destination check
Step 5: destination schema check
Step 6: the read from csv --> apply transformation --> write to db (the data transfer pipeline)
Step 7: checking if the output is correct
'''

from pathlib import Path
import daft
import os

from app.agents.data_transfer_agent.dta_state import DaftETLState, DaftCodingAgentState
from app.agents.data_transfer_agent.data_transfer_agent import daft_code_generation_pipeline 
from app.agents.data_transfer_agent.datasink.sql_sink import MySQLDataSink

# SETUP: Schema & Prompt Definition
source_schema = {
    "Name": "string",
    "Age": "integer",
    "Gender": "string",
    "Blood Type": "string",
    "Medical Condition": "string",
    "Date of Admission": "date",
    "Doctor": "string",
    "Hospital": "string",
    "Insurance Provider": "string",
    "Billing Amount": "float",
    "Room Number": "integer",
    "Admission Type": "string",
    "Discharge Date": "date",
    "Medication": "string",
    "Test Results": "string"
}

user_prompt = """
Write a Daft query to process patient hospitalization records. 
The query should perform the following steps:
1. Data Cleaning: Filter the dataset to exclude any records where the 'Test Results' are recorded as 'Inconclusive'.
2. Feature Engineering: Calculate the duration of each hospital visit. Create a new column named 'length_of_stay' derived by subtracting the 'Discharge Date' from the 'Date of Admission'.
"""

#this needs to be done right
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_SOURCE_LOCATION = BASE_DIR / "sample_data" / "patient_data.csv"

#initialize etl state and coder state

coder_definition = DaftCodingAgentState(
    user_prompt=user_prompt,
    data_source_location=INPUT_SOURCE_LOCATION,
    schema=source_schema,
    syntax_error=False,
    static_semantic_error=False,
    execution_error=False,
    logical_semantic_error=False
)

etl_object = DaftETLState(
    coder_definition=coder_definition,
    user_query=user_prompt,
    source_schema=source_schema,
    input_source_location=INPUT_SOURCE_LOCATION,
    messages=[],
    input_sample_data=None # Will be loaded in the next step
)

#destination details and configuration
conn_str = "mysql+pymysql://root:password@localhost:3306/hospital_db"
mysql_destination = MySQLDataSink(conn_str, "patients")

# PRE-PROCESSING: Data Loading
print(f">>> Loading sample data from {INPUT_SOURCE_LOCATION}...")

if os.path.exists(INPUT_SOURCE_LOCATION):
    sample_df = pd.read_csv(INPUT_SOURCE_LOCATION, nrows=100)
    etl_object["input_sample_data"] = sample_df
    print(f"   [Success] Loaded {len(sample_df)} rows into state.")
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Input file not found at {INPUT_SOURCE_LOCATION}")

#code generation pipeline
print("\n>>> STARTING DAFT CODE GENERATION PIPELINE ")
final_state = daft_code_generation_pipeline(etl_object)
print(">>> Pipeline Finished.\n")



