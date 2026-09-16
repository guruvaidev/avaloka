import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.data_transfer_agent import daft_coder
from tests.test_daft_pseudocode_integration import PROMPT_CASES


pytestmark = pytest.mark.integration


HEALTHCARE_ROWS = [
    {
        "Name": "john doe",
        "Age": 45,
        "Gender": "M",
        "Blood Type": "O+",
        "Medical Condition": "Asthma",
        "Date of Admission": "2020-01-05",
        "Doctor": "Dr. Alice Smith",
        "Hospital": "General",
        "Insurance Provider": "ProviderA",
        "Billing Amount": 12000.0,
        "Room Number": 12,
        "Admission Type": "Emergency",
        "Discharge Date": "2020-01-10",
        "Medication": "DrugA",
        "Test Results": "Negative",
    },
    {
        "Name": "jane roe",
        "Age": 70,
        "Gender": "F",
        "Blood Type": "A-",
        "Medical Condition": "Diabetes",
        "Date of Admission": "2020-02-15",
        "Doctor": "Dr. Bob Jones",
        "Hospital": "Central",
        "Insurance Provider": "ProviderB",
        "Billing Amount": 8000.0,
        "Room Number": 7,
        "Admission Type": "Routine",
        "Discharge Date": "2020-02-20",
        "Medication": "DrugB",
        "Test Results": "Inconclusive",
    },
]

TITANIC_ROWS = [
    {
        "PassengerId": 1,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Mr. John Smith",
        "Sex": "male",
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 72.5,
        "Cabin": "C85",
        "Embarked": "S",
    },
    {
        "PassengerId": 2,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Miss. Jane Doe",
        "Sex": "female",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PC 17599",
        "Fare": 15.0,
        "Cabin": None,
        "Embarked": "C",
    },
]

IRIS_ROWS = [
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "species": "setosa",
    },
    {
        "sepal_length": 6.4,
        "sepal_width": 3.2,
        "petal_length": 4.5,
        "petal_width": 1.5,
        "species": "versicolor",
    },
]

TAXI_ROWS = [
    {
        "VendorID": 1,
        "tpep_pickup_datetime": "2020-01-01 10:00:00",
        "tpep_dropoff_datetime": "2020-01-01 10:30:00",
        "passenger_count": 2,
        "trip_distance": 3.5,
        "RatecodeID": 1,
        "store_and_fwd_flag": "N",
        "PULocationID": 100,
        "DOLocationID": 200,
        "payment_type": 1,
        "fare_amount": 12.0,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 15.8,
        "congestion_surcharge": 2.5,
    },
    {
        "VendorID": 2,
        "tpep_pickup_datetime": "2020-01-02 18:00:00",
        "tpep_dropoff_datetime": "2020-01-02 18:10:00",
        "passenger_count": 1,
        "trip_distance": 1.2,
        "RatecodeID": 1,
        "store_and_fwd_flag": "N",
        "PULocationID": 110,
        "DOLocationID": 210,
        "payment_type": 2,
        "fare_amount": 6.5,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 0.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 7.8,
        "congestion_surcharge": 2.5,
    },
]


def _sample_rows_for_schema(schema: dict):
    if set(schema.keys()) == set(HEALTHCARE_ROWS[0].keys()):
        return HEALTHCARE_ROWS
    if set(schema.keys()) == set(TITANIC_ROWS[0].keys()):
        return TITANIC_ROWS
    if set(schema.keys()) == set(IRIS_ROWS[0].keys()):
        return IRIS_ROWS
    if set(schema.keys()) == set(TAXI_ROWS[0].keys()):
        return TAXI_ROWS
    raise ValueError("Unknown schema for sample data.")


def _execute_generated_code(code: str, df):
    exec_globals = {"__name__": "__main__"}
    exec(code, exec_globals)
    main = exec_globals.get("main")
    assert main is not None, "Generated code must define main(df)"
    result = main(df)
    return result


@pytest.mark.parametrize("schema,prompt", PROMPT_CASES)
def test_data_transfer_agent_codegen(schema, prompt):
    if daft_coder.coder_llm is None:
        pytest.skip("GROQ_API_KEY_CODING_AGENT is required for integration tests.")

    daft = pytest.importorskip("daft")

    embeddings_path = Path("app/rag/daft_embeddings")
    if not embeddings_path.exists():
        pytest.skip("Daft embeddings not found; run RAG ingestion before integration tests.")

    state = {
        "user_prompt": prompt,
        "schema": schema,
        "input_data_type": "csv",
    }

    result = daft_coder.daft_coder_node(state)
    generated_code = result.get("generated_code") or ""
    assert generated_code, "Code generation failed"

    rows = _sample_rows_for_schema(schema)
    df = daft.from_pydict({key: [row[key] for row in rows] for key in rows[0]})
    output = _execute_generated_code(generated_code, df)
    assert output is not None, "Generated code returned no DataFrame"
    output.collect()
