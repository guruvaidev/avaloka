from types import SimpleNamespace
import os
import re

import pytest

from app.agents.data_transfer_agent import daft_coder
from app.agents.data_transfer_agent.daft_coder import parse_pseudo_code as parse_pseudocode


def test_generate_pseudocode_uses_prompt_and_rules(monkeypatch):
    class StubLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="1. Step one")

    stub_llm = StubLLM()
    monkeypatch.setattr(daft_coder, "coder_llm", stub_llm)

    state = {
        "user_prompt": "Create a new column based on Age.",
        "schema": {"Age": "integer"},
    }

    result = daft_coder._generate_pseudocode_daft(state)

    assert result == "1. Step one"
    assert stub_llm.messages is not None

    system_prompt = stub_llm.messages[0].content
    human_prompt = stub_llm.messages[1].content

    assert "Never rename columns unless the user explicitly requests a rename." in system_prompt
    assert "You MUST include explicit casting steps whenever types are incompatible" in system_prompt
    assert "Create a new column based on Age." in human_prompt
    assert "Schema (column: dtype):" in human_prompt
    assert "Age" in human_prompt


def test_cached_chroma_collection(monkeypatch):
    daft_coder._get_cached_chroma_collection.cache_clear()
    calls = []

    def fake_load(db_path, collection_name, embedding_model):
        calls.append((db_path, collection_name, embedding_model))
        return {"db_path": db_path, "collection_name": collection_name}

    monkeypatch.setattr(daft_coder, "load_chroma_collection", fake_load)

    first = daft_coder._get_cached_chroma_collection("db", "collection", "model")
    second = daft_coder._get_cached_chroma_collection("db", "collection", "model")

    assert first is second
    assert len(calls) == 1


def test_daft_coder_node_builds_prompt_from_inputs(monkeypatch):
    class StubLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="```python\nprint('ok')\n```")

    stub_llm = StubLLM()
    monkeypatch.setattr(daft_coder, "coder_llm", stub_llm)
    monkeypatch.setattr(daft_coder, "_generate_pseudocode_daft", lambda state: "1. Step one")
    monkeypatch.setattr(daft_coder, "retrieve_docs", lambda lines, collection, top_k: [])
    monkeypatch.setattr(daft_coder, "format_chunks_for_prompt", lambda docs: "")
    monkeypatch.setattr(daft_coder, "_get_cached_chroma_collection", lambda *args, **kwargs: object())

    state = {
        "user_prompt": "Filter rows where Age > 30.",
        "schema": {"Age": "integer"},
        "input_data_type": "csv",
    }

    result = daft_coder.daft_coder_node(state)

    assert "generated_code" in result
    assert "print('ok')" in result["generated_code"]

    human_prompt = stub_llm.messages[1].content
    assert "Filter rows where Age > 30." in human_prompt
    assert "Age" in human_prompt
    assert "Pseudocode blueprint" in human_prompt


pytestmark = pytest.mark.integration


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

TAXI_SCHEMA = {
    "VendorID": "integer",
    "tpep_pickup_datetime": "string",
    "tpep_dropoff_datetime": "string",
    "passenger_count": "integer",
    "trip_distance": "float",
    "RatecodeID": "integer",
    "store_and_fwd_flag": "string",
    "PULocationID": "integer",
    "DOLocationID": "integer",
    "payment_type": "integer",
    "fare_amount": "float",
    "extra": "float",
    "mta_tax": "float",
    "tip_amount": "float",
    "tolls_amount": "float",
    "improvement_surcharge": "float",
    "total_amount": "float",
    "congestion_surcharge": "float",
}


PROMPT_CASES = [
    (
        HEALTHCARE_SCHEMA,
        """
Write a Daft query to process patient hospitalization records.
The query should perform the following steps:
1. Filter the dataset to exclude any records where the 'Test Results' are recorded as 'Inconclusive'.
2. Calculate the duration of each hospital visit. Create a new column named 'length_of_stay' derived by subtracting the 'Discharge Date' from the 'Date of Admission'.
3. Format the "Name" column to have first letter of each word as uppercase and the rest lowercase (e.g., "john doe" -> "John Doe").
4. Create a new column named 'age_group' that categorizes patients into age groups based on the 'Age' column. The age groups should be defined as follows:
   - 'Child' for ages 0-12
   - 'Teen' for ages 13-19
   - 'Adult' for ages 20-64
   - 'Senior' for ages 65 and above
5. Sort the resulting dataset by 'length_of_stay' in descending order and then by 'Name' in ascending order.
6. Transform the 'Doctor' column to extract the initial of the first name and the last name of the doctor. Assume the format of the doctor's name is "Dr. Firstname Lastname" and you want to keep only "F Lastname".
7. Finally, select only the following columns to be included in the output: 'Name', 'age_group', 'Medical Condition', 'length_of_stay', and the transformed 'Doctor' column.
""",
    ),
    (
        HEALTHCARE_SCHEMA,
        """
Write a Daft query to process patient hospitalization records.
The query should perform the following steps:
1. Filter the dataset to include records only for patients who were admitted to the hospital in the year 2020.
2. Group the dataset by 'Medical Condition' and calculate the average 'Billing Amount' for each medical condition. Create a new column named 'average_billing' to store these values.
3. Sort the resulting dataset by 'average_billing' in descending order.
4. Create a new column named 'condition_severity' that categorizes medical conditions based on the average billing amount. The categories should be defined as follows:
    - 'Low' for average billing amounts less than $5,000
    - 'Medium' for average billing amounts between $5,000 and $20,000
    - 'High' for average billing amounts greater than $20,000
5. Finally, select only the following columns to be included in the output: 'Medical Condition', 'average_billing', and 'condition_severity'.
""",
    ),
    (
        HEALTHCARE_SCHEMA,
        """
Write a Daft query to process patient hospitalization records.
The query should perform the following steps:
1. Filter the dataset to include only records for patients who were admitted to the hospital in the year 2020 and had a 'Billing Amount' greater than $10,000.
2. Create a new column named 'admission_month' that extracts the month from the 'Date of Admission' column.
3. Create a new column named 'admission_year' that extracts the year from the 'Date of Admission' column.
4. Group the dataset by 'admission_month' and 'admission_year', and calculate the total 'Billing Amount' for each month-year combination. Create a new column named 'total_billing' to store these values.
5. Sort the resulting dataset by 'total_billing' in descending order.
""",
    ),
    (
        HEALTHCARE_SCHEMA,
        """
Write a Daft query to process patient hospitalization records.
The query should perform the following steps:
1. Transform the 'Date of Admission' and 'Discharge Date' columns to a standard date format (e.g., YYYY-MM-DD).
2. Create a new column named 'length_of_stay' that calculates the number of days between the 'Date of Admission' and 'Discharge Date'.
3. Create a new column called 'charge_per_day' that calculates the average daily charge for each patient by dividing the 'Billing Amount' by the 'length_of_stay'.
4. Filter the dataset to include only records where the 'charge_per_day' is greater than $500.
""",
    ),
    (
        HEALTHCARE_SCHEMA,
        """
Write a Daft query to process patient hospitalization records.
The query should perform the following steps:
1. For each insurance provider, hash the 'Insurance Provider' column to create a new column named 'provider_hash' that contains a unique hash value for each provider. Use the SHA-256 hashing algorithm to generate the hash values.
2. Remove the original 'Insurance Provider' column from the dataset after creating the 'provider_hash' column.
3. Group the dataset by 'provider_hash' and calculate the total 'Billing Amount' for each hashed provider. Create a new column named 'total_billing' to store these values.
4. Sort the resulting dataset by 'total_billing' in descending order.
""",
    ),
    (
        TITANIC_SCHEMA,
        """
Write a Daft query to process Titanic passenger records. The query should perform the following steps:
1. Filter the dataset to exclude any records where 'Age' is null or missing.
2. Create a new column named 'age_group' that categorizes passengers into age groups:
   - 'Child' for ages 0-12
   - 'Teen' for ages 13-19
   - 'Adult' for ages 20-64
   - 'Senior' for ages 65 and above
3. Create a new column named 'family_size' by adding 'SibSp' and 'Parch' columns and adding 1 (for the passenger themselves).
4. Create a new column named 'is_alone' that is True if 'family_size' equals 1, otherwise False.
5. Group by 'Pclass' and 'age_group', and calculate the survival rate (average of 'Survived' column) for each group. Name this column 'survival_rate'.
6. Sort the resulting dataset by 'Pclass' in ascending order and then by 'survival_rate' in descending order.
7. Select only the following columns: 'Pclass', 'age_group', 'survival_rate'.
""",
    ),
    (
        TITANIC_SCHEMA,
        """
Write a Daft query to process Titanic passenger records. The query should perform the following steps:
1. Filter the dataset to include only passengers who embarked from Southampton (Embarked = 'S').
2. Create a new column named 'fare_category' that categorizes fares:
   - 'Budget' for fare less than $10
   - 'Economy' for fare between $10 and $30
   - 'Premium' for fare between $30 and $100
   - 'Luxury' for fare greater than $100
3. Extract the title from the 'Name' column (Mr., Mrs., Miss., Master., etc.) and create a new column named 'title'. Assume titles are followed by a period.
4. Group by 'Sex' and 'fare_category', and calculate the count of passengers and average fare for each group.
5. Sort by count in descending order.
6. Select columns: 'Sex', 'fare_category', 'passenger_count', 'average_fare'.
""",
    ),
    (
        TITANIC_SCHEMA,
        """
Write a Daft query to process Titanic passenger records. The query should perform the following steps:
1. Create a new column named 'deck' by extracting the first character from the 'Cabin' column (e.g., 'C85' -> 'C'). If Cabin is null, set deck to 'Unknown'.
2. Filter to include only records where 'Fare' is greater than 0.
3. Create a new column named 'fare_per_person' by dividing 'Fare' by 'family_size' (where family_size = SibSp + Parch + 1).
4. Group by 'deck' and 'Pclass', and calculate the average 'fare_per_person' and survival rate for each group.
5. Sort by 'Pclass' ascending, then by average fare_per_person descending.
6. Select columns: 'deck', 'Pclass', 'avg_fare_per_person', 'survival_rate'.
""",
    ),
    (
        TITANIC_SCHEMA,
        """
Write a Daft query to process Titanic passenger records. The query should perform the following steps:
1. Filter the dataset to include only adult passengers (Age >= 18).
2. Create a new column named 'ticket_prefix' by extracting all alphabetic characters from the 'Ticket' column. If there are no letters, set it to 'NUMERIC'.
3. Create a new column named 'has_cabin' that is True if 'Cabin' is not null, otherwise False.
4. Group by 'ticket_prefix' and calculate the count of passengers, average fare, and survival rate.
5. Filter groups to include only those with at least 10 passengers.
6. Sort by survival_rate in descending order.
7. Select columns: 'ticket_prefix', 'passenger_count', 'avg_fare', 'survival_rate'.
""",
    ),
    (
        TITANIC_SCHEMA,
        """
Write a Daft query to process Titanic passenger records. The query should perform the following steps:
1. Create a new column named 'name_length' that contains the character count of the 'Name' column.
2. Create a new column named 'cabin_count' by counting the number of cabin codes in the 'Cabin' column (separated by spaces). If Cabin is null, set to 0.
3. For each 'Embarked' port, hash the port code to create a new column named 'port_hash' using SHA-256 hashing.
4. Remove the original 'Embarked' column after creating 'port_hash'.
5. Group by 'Sex', 'Pclass', and 'port_hash', and calculate the total count and survival rate.
6. Sort by count in descending order.
7. Select columns: 'Sex', 'Pclass', 'port_hash', 'total_passengers', 'survival_rate'.
""",
    ),
    (
        IRIS_SCHEMA,
        """
Write a Daft query to process Iris flower records. The query should perform the following steps:
1. Create a new column named 'sepal_area' by multiplying 'sepal_length' and 'sepal_width'.
2. Create a new column named 'petal_area' by multiplying 'petal_length' and 'petal_width'.
3. Create a new column named 'total_area' by adding 'sepal_area' and 'petal_area'.
4. Create a new column named 'size_category' that categorizes flowers based on 'total_area':
   - 'Small' for total_area less than 15
   - 'Medium' for total_area between 15 and 25
   - 'Large' for total_area greater than 25
5. Group by 'species' and 'size_category', and calculate the count and average total_area for each group.
6. Sort by species ascending, then by average total_area descending.
7. Select columns: 'species', 'size_category', 'flower_count', 'avg_total_area'.
""",
    ),
    (
        IRIS_SCHEMA,
        """
Write a Daft query to process Iris flower records. The query should perform the following steps:
1. Create a new column named 'sepal_ratio' by dividing 'sepal_length' by 'sepal_width'.
2. Create a new column named 'petal_ratio' by dividing 'petal_length' by 'petal_width'.
3. Filter the dataset to include only records where 'petal_length' is greater than the average petal_length across all flowers.
4. Create a new column named 'aspect_category' based on sepal_ratio:
   - 'Wide' for ratio less than 2.5
   - 'Balanced' for ratio between 2.5 and 3.5
   - 'Narrow' for ratio greater than 3.5
5. Group by 'species' and calculate the average sepal_ratio, petal_ratio, and count.
6. Sort by avg_sepal_ratio in descending order.
7. Select columns: 'species', 'avg_sepal_ratio', 'avg_petal_ratio', 'flower_count'.
""",
    ),
    (
        IRIS_SCHEMA,
        """
Write a Daft query to process Iris flower records. The query should perform the following steps:
1. Normalize the 'species' column to uppercase.
2. Create a new column named 'petal_dominance' by dividing 'petal_area' (petal_length * petal_width) by the sum of sepal_area and petal_area.
3. Create a new column named 'dominance_category':
   - 'Sepal Dominant' if petal_dominance < 0.3
   - 'Balanced' if petal_dominance between 0.3 and 0.6
   - 'Petal Dominant' if petal_dominance > 0.6
4. Group by 'species' and 'dominance_category', and calculate the count and average petal_dominance.
5. Sort by species, then by avg_petal_dominance descending.
6. Select columns: 'species', 'dominance_category', 'count', 'avg_petal_dominance'.
""",
    ),
    (
        IRIS_SCHEMA,
        """
Write a Daft query to process Iris flower records. The query should perform the following steps:
1. Create a new column named 'perimeter' by calculating 2 * (sepal_length + sepal_width + petal_length + petal_width).
2. Create a new column named 'compactness' by dividing the total area (sepal_area + petal_area) by the square of the perimeter.
3. Filter records where 'sepal_length' is greater than 5.0 and 'petal_width' is less than 2.0.
4. Create a new column named 'compactness_level':
   - 'Low' for compactness < 0.01
   - 'Medium' for compactness between 0.01 and 0.02
   - 'High' for compactness > 0.02
5. Group by 'compactness_level' and calculate average sepal_length, petal_length, and count.
6. Sort by count descending.
7. Select columns: 'compactness_level', 'avg_sepal_length', 'avg_petal_length', 'flower_count'.
""",
    ),
    (
        IRIS_SCHEMA,
        """
Write a Daft query to process Iris flower records. The query should perform the following steps:
1. Hash the 'species' column using SHA-256 to create a new column named 'species_hash'.
2. Remove the original 'species' column after creating the hash.
3. Create a new column named 'measurement_variance' by calculating the variance across the four measurement columns (sepal_length, sepal_width, petal_length, petal_width) for each row.
4. Group by 'species_hash' and calculate the average of all four measurements plus the average measurement_variance.
5. Sort by avg_measurement_variance in descending order.
6. Select columns: 'species_hash', 'avg_sepal_length', 'avg_sepal_width', 'avg_petal_length', 'avg_petal_width', 'avg_variance'.
""",
    ),
    (
        TAXI_SCHEMA,
        """
Write a Daft query to process NYC taxi trip records. The query should perform the following steps:
1. Filter the dataset to exclude trips where 'trip_distance' is 0 or 'total_amount' is less than or equal to 0.
2. Calculate the trip duration by creating a new column named 'trip_duration_minutes' that subtracts 'tpep_pickup_datetime' from 'tpep_dropoff_datetime' and converts to minutes.
3. Create a new column named 'speed_mph' by dividing 'trip_distance' by 'trip_duration_minutes' and multiplying by 60.
4. Filter out trips where 'speed_mph' is greater than 100 (likely data errors) or less than 1.
5. Create a new column named 'duration_category':
   - 'Short' for trips less than 15 minutes
   - 'Medium' for trips between 15 and 45 minutes
   - 'Long' for trips greater than 45 minutes
6. Group by 'duration_category' and calculate average trip_distance, average total_amount, and count.
7. Sort by avg_total_amount descending.
8. Select columns: 'duration_category', 'trip_count', 'avg_distance', 'avg_fare'.
""",
    ),
    (
        TAXI_SCHEMA,
        """
Write a Daft query to process NYC taxi trip records. The query should perform the following steps:
1. Extract the hour of day from 'tpep_pickup_datetime' and create a new column named 'pickup_hour'.
2. Extract the day of week from 'tpep_pickup_datetime' and create a new column named 'pickup_day'.
3. Create a new column named 'time_of_day':
   - 'Early Morning' for hours 0-5
   - 'Morning' for hours 6-11
   - 'Afternoon' for hours 12-17
   - 'Evening' for hours 18-23
4. Create a new column named 'tip_percentage' by dividing 'tip_amount' by 'fare_amount' and multiplying by 100.
5. Filter to include only trips with payment_type = 1 (credit card) where tip_percentage is between 0 and 50.
6. Group by 'time_of_day' and calculate average tip_percentage, average total_amount, and count.
7. Sort by avg_tip_percentage descending.
8. Select columns: 'time_of_day', 'trip_count', 'avg_tip_percentage', 'avg_total_amount'.
""",
    ),
    (
        TAXI_SCHEMA,
        """
Write a Daft query to process NYC taxi trip records. The query should perform the following steps:
1. Create a new column named 'total_fees' by summing 'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge', and 'congestion_surcharge'.
2. Create a new column named 'fees_ratio' by dividing 'total_fees' by 'total_amount'.
3. Create a new column named 'distance_category':
   - 'Very Short' for distance less than 1 mile
   - 'Short' for distance between 1 and 3 miles
   - 'Medium' for distance between 3 and 10 miles
   - 'Long' for distance greater than 10 miles
4. Filter records where 'passenger_count' is between 1 and 6.
5. Group by 'distance_category' and 'passenger_count', and calculate average total_amount, average fees_ratio, and count.
6. Sort by distance_category, then by passenger_count.
7. Select columns: 'distance_category', 'passenger_count', 'trip_count', 'avg_total_amount', 'avg_fees_ratio'.
""",
    ),
    (
        TAXI_SCHEMA,
        """
Write a Daft query to process NYC taxi trip records. The query should perform the following steps:
1. Extract the date from 'tpep_pickup_datetime' and create a new column named 'pickup_date'.
2. Create a new column named 'is_weekend' that is True if the day of week is Saturday or Sunday, otherwise False.
3. Create a new column named 'fare_per_mile' by dividing 'fare_amount' by 'trip_distance' (handle division by zero).
4. Filter to include only trips where 'trip_distance' is greater than 0.5 miles and 'fare_per_mile' is less than $20.
5. Group by 'pickup_date' and 'is_weekend', and calculate total trips, total revenue (sum of total_amount), and average fare_per_mile.
6. Sort by total_revenue descending.
7. Select columns: 'pickup_date', 'is_weekend', 'total_trips', 'total_revenue', 'avg_fare_per_mile'.
""",
    ),
    (
        TAXI_SCHEMA,
        """
Write a Daft query to process NYC taxi trip records. The query should perform the following steps:
1. Hash the combination of 'PULocationID' and 'DOLocationID' using SHA-256 to create a new column named 'route_hash'.
2. Remove the original 'PULocationID' and 'DOLocationID' columns after creating the hash.
3. Create a new column named 'profit_margin' by dividing (total_amount - tolls_amount - total_fees) by total_amount.
4. Filter records where 'trip_distance' is greater than 0 and 'total_amount' is greater than 0.
5. Group by 'route_hash' and calculate the count of trips, average trip_distance, average total_amount, and average profit_margin.
6. Filter groups to include only routes with at least 50 trips.
7. Sort by trip_count descending.
8. Select columns: 'route_hash', 'trip_count', 'avg_distance', 'avg_revenue', 'avg_profit_margin'.
""",
    ),
]


def _allowed_new_columns(prompt: str) -> set:
    matches = re.findall(
        r"new column named ['\"]?([^'\"]+)['\"]?",
        prompt,
        flags=re.IGNORECASE,
    )
    matches += re.findall(
        r"new column called ['\"]?([^'\"]+)['\"]?",
        prompt,
        flags=re.IGNORECASE,
    )
    return {match.strip() for match in matches}


@pytest.mark.parametrize("schema,prompt", PROMPT_CASES)
def test_pseudocode_generation_for_all_prompts(schema, prompt):
    if daft_coder.coder_llm is None:
        pytest.skip("GROQ_API_KEY_CODING_AGENT is required for integration tests.")

    state = {
        "user_prompt": prompt,
        "schema": schema,
    }

    pseudocode = daft_coder._generate_pseudocode_daft(state)
    assert pseudocode, "Pseudocode generation returned empty output"

    assert "### TRANSFORMATION STEPS" in pseudocode
    assert "### DERIVED COLUMNS SUMMARY" in pseudocode
    assert "### UDF REQUIREMENTS" in pseudocode

    doc = parse_pseudocode(pseudocode)
    assert doc.steps, "No transformation steps parsed"

    for step in doc.steps:
        has_mapping = bool(step.functions or step.udf_required or step.missing_columns)
        assert has_mapping, f"Missing mapping tags for step {step.index}"

    allowed_new = _allowed_new_columns(prompt)
    schema_columns = set(schema.keys())
    defined_columns = set(doc.defined_column_names())

    extra_columns = {
        col for col in defined_columns if col not in schema_columns and col not in allowed_new
    }
    assert not extra_columns, f"Unexpected derived columns: {sorted(extra_columns)}"
