import torch
import pandas as pd
from app.agents.data_transfer_agent.dta_state import DaftETLState, DaftCodingAgentState
from app.agents.data_transfer_agent.daft_coder import daft_coder_node
from app.agents.data_transfer_agent.daft_execution import execution_agent_node_local

# 1. SETUP: Schema & Prompt Definition

# Define the schema based on typical patient data
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

# Generate a well-written prompt for the test case
user_prompt = """
Write a Daft query to process patient hospitalization records. 
The query should perform the following steps:
1. Data Cleaning: Filter the dataset to exclude any records where the 'Test Results' are recorded as 'Inconclusive'.
2. Feature Engineering: Calculate the duration of each hospital visit. Create a new column named 'length_of_stay' derived by subtracting the 'Discharge Date' from the 'Date of Admission'.
"""

# PLACEHOLDER: User to substitute file path here
INPUT_SOURCE_LOCATION = "C:/Users/Julia Maddu GS/Desktop/daft-migration-prototype/data/patient-data-raw.csv"


# 2. INSTANTIATION: ETL Object & Coder Agent

# Instantiate the State Object
coder_definition = DaftCodingAgentState(
    user_prompt=user_prompt,
    data_source_location=INPUT_SOURCE_LOCATION,
    schema= source_schema
)

etl_object = DaftETLState(
    coder_definition=coder_definition,
    user_query=user_prompt,
    source_schema=source_schema,
    input_source_location=INPUT_SOURCE_LOCATION,
    messages=[]
)


# 3. INVOCATION: Coder Agent

print(">>> Invoking Coder Agent...")

# Invoke the coder agent to generate the transformation code
coder_definition = daft_coder_node(coder_definition)

# UPDATE 1: Update the Coding Agent State

etl_object["coder_definition"] = coder_definition


# INTERMEDIATE STEP: Data Extraction

print(">>> Reading Input Source & Sampling...")

# Read the file from the location stored in the ETL object
# Assuming CSV for this example, but could be Parquet/JSON
try:
    full_df = pd.read_csv(etl_object["input_source_location"])
    
    # Extract first 1000 rows
    sample_df = full_df.head(1000)
    
    # Update the input_sample_data in the ETL object
    etl_object["input_sample_data"] = sample_df
    print(f"Successfully loaded {len(sample_df)} rows into input_sample_data.")

except FileNotFoundError:
    print(f"Error: File not found at {etl_object.input_source_location}")


# 5. EXECUTION AGENT


print(">>> Invoking Execution Agent...")

execution_agent_output = execution_agent_node_local(etl_object)


# Print the final output
print("\n=== Final State Output ===")
print(execution_agent_output)