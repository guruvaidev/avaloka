
from tests._quarantine import requires_api

requires_api("app.agents.data_transfer_agent.data_transfer_agent", "deduce_db_schema", "deduce_file_schema", replacement="deduce_db_schema/deduce_file_schema were consolidated into deduce_schema with a different signature; repair against deduce_schema or remove this file.")

import torch
import daft
from pathlib import Path

from app.agents.data_transfer_agent.data_transfer_agent import deduce_db_schema, deduce_file_schema, build_connection_string

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "sample_data"

iris_path = str(DATA_DIR / "iris.csv")
patient_path = str(DATA_DIR / "patient_data.csv")
titanic_path = str(DATA_DIR / "titanic-dataset.csv")


#defining schemas
HEALTHCARE_SCHEMA = {
    "Name": "string",
    "Age": "integer",
    "Gender": "string",
    "Blood Type": "string",
    "Medical Condition": "string",
    "Date of Admission": "string",
    "Doctor": "string",
    "Hospital": "string",
    "Insurance Provider": "string",
    "Billing Amount": "float",
    "Room Number": "integer",
    "Admission Type": "string",
    "Discharge Date": "string",
    "Medication": "string",
    "Test Results": "string",
}

TITANIC_SCHEMA = {
    "PassengerId": "integer",
    "Survived": "integer",
    "Pclass": "integer",
    "Name": "string",
    "Sex": "string",
    "Age": "float",
    "SibSp": "integer",
    "Parch": "integer",
    "Ticket": "string",
    "Fare": "float",
    "Cabin": "string",
    "Embarked": "string",
}

IRIS_SCHEMA = {
    "sepal_length": "float",
    "sepal_width": "float",
    "petal_length": "float",
    "petal_width": "float",
    "species": "string",
}

if __name__ == "__main__":

    db_credentials = {
        "user": "root",          
        "password": "rootpassword", 
        "host": "127.0.0.1",
        "port": 3307,             
        "database": "iris_db"
    }

    connection_url = build_connection_string(credentials=db_credentials, source_type="mysql")
    print(f"Built Connection String: {connection_url}\n")

    table_name = 'iris'

    ans = deduce_db_schema(connection_url, table_name)
    print(ans)