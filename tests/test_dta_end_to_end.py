"""
================================================================================
 DTA END-TO-END TEST HARNESS  —  DATABASE → DATABASE, DRIVEN THROUGH THE PLANNER
================================================================================

This is the full ``test_daft_coder_pipeline.py`` benchmark (every category:
column ops, null handling, type casting, string manipulation, date/time,
arithmetic, filtering, data masking, UDFs, aggregations) — but instead of
calling ``data_transfer_pipeline`` directly, **every transfer is invoked through
the planner**, exactly the way a real chat request flows:

    planner._handle_register_database(source)   ─┐
    planner._handle_register_database(dest)     ─┤─► planner._handle_initiate_transfer
                                                 │        │ resolve creds
                                                 │        │ data_transfer_pipeline (code-gen + schema)
                                                 └────────┴─► GKE Ray / Docker runner ─► destination

The result is then read back from the destination Postgres table and verified
deterministically — the same pass/fail contract the reference suite uses.

Why the planner path needs a small creds shim
----------------------------------------------
The reference passes ``sslmode: "require"`` on both source and destination creds;
that SSL context is what lets it reach the cloud Postgres. ``RegisterDatabaseParams``
has no ``sslmode`` field, so the planner would otherwise drop it and every
connection would fail. After registering, we inject ``sslmode`` straight into the
registry entry — ``_fetch_full_creds`` copies every registry key into the creds it
hands to ``build_connection_string`` and the sink, so SSL flows through end-to-end
just like the direct test. Nothing else about the planner path is bypassed.

Pre-requisites (identical to the reference)
-------------------------------------------
    * Source table ``mock_crm_data`` (canonical 5-row CRM fixture) exists on the
      source Postgres.
    * Every ``dest_*`` destination table PRE-EXISTS with a schema compatible with
      that transformation's output (the planner will not invent a missing table —
      it reports it, surfaced here as "FAIL (Dest Table Setup)"). Each run
      truncates its destination table first for a deterministic insert.

Requires the coder LLM key (export it, do NOT use `set` on Linux):
    export GROQ_API_KEY_CODING_AGENT=...

Run
---
    EXECUTION_ENV=gke  python -m tests.test_dta_end_to_end
"""

import os
import sys
import uuid
import logging
from datetime import datetime

import pandas as pd

# Planner is the integration entry point under test.
from app.agents.planner import (
    RegisterDatabaseParams,
    InitiateTransferParams,
    _handle_register_database,
    _handle_initiate_transfer,
)

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None


# ==========================================
# TEST SUITE: COLUMN OPERATIONS (10 Tests)
# ==========================================
COLUMN_OPERATIONS_TEST_SUITE = [
    {
        "category": "Column Operations",
        "name": "col_rename_id",
        "dest_table": "dest_col_rename_id",
        "prompt": "Rename the 'id' column to 'customer_id'.",
        "verify": lambda df: "customer_id" in df.columns and "id" not in df.columns
    },
    {
        "category": "Column Operations",
        "name": "col_drop_ssn",
        "dest_table": "dest_col_drop_ssn",
        "prompt": "Drop the 'ssn' column from the dataset.",
        "verify": lambda df: "ssn" not in df.columns
    },
    {
        "category": "Column Operations",
        "name": "col_multi_rename",
        "dest_table": "dest_col_multi_rename",
        "prompt": "Rename 'status' to 'account_status' and 'loyalty_score' to 'points' in one step.",
        "verify": lambda df: "account_status" in df.columns and "points" in df.columns and "status" not in df.columns and "loyalty_score" not in df.columns
    },
    {
        "category": "Column Operations",
        "name": "col_copy_email",
        "dest_table": "dest_col_copy_email",
        "prompt": "Create an exact duplicate of 'email_address' and name it 'contact_email'.",
        "verify": lambda df: "contact_email" in df.columns and (df['contact_email'] == df['email_address']).all()
    },
    {
        "category": "Column Operations",
        "name": "col_drop_multi",
        "dest_table": "dest_col_drop_multi",
        "prompt": "Drop both the 'signup_date' and 'age' columns.",
        "verify": lambda df: "signup_date" not in df.columns and "age" not in df.columns
    },
    {
        "category": "Column Operations",
        "name": "col_add_constant_str",
        "dest_table": "dest_col_add_constant_str",
        "prompt": "Add a new column 'country' and set every row's value to the string 'USA'.",
        "verify": lambda df: "country" in df.columns and (df['country'] == 'USA').all()
    },
    {
        "category": "Column Operations",
        "name": "col_add_constant_num",
        "dest_table": "dest_col_add_constant_num",
        "prompt": "Add a new column 'version' with every row set to the integer value 1.",
        "verify": lambda df: "version" in df.columns and (df['version'] == 1).all()
    },
    {
        "category": "Column Operations",
        "name": "col_add_null",
        "dest_table": "dest_col_add_null",
        "prompt": "Add a new column called 'notes' filled with NULL values.",
        "verify": lambda df: "notes" in df.columns and df['notes'].isna().all()
    },
    {
        "category": "Column Operations",
        "name": "col_reorder",
        "dest_table": "dest_col_reorder",
        "prompt": "Reorder the columns so that 'first_name' comes first, followed by 'id', then all remaining columns in their original order.",
        "verify": lambda df: list(df.columns[:2]) == ['first_name', 'id']
    },
    {
        "category": "Column Operations",
        "name": "col_rename_and_drop",
        "dest_table": "dest_col_rename_and_drop",
        "prompt": "Rename 'id' to 'customer_id' and drop the 'ssn' column in the same operation.",
        "verify": lambda df: "customer_id" in df.columns and "id" not in df.columns and "ssn" not in df.columns
    },
]

# ==========================================
# TEST SUITE: NULL / MISSING VALUE HANDLING (10 Tests)
# ==========================================
NULL_HANDLING_TEST_SUITE = [
    {
        "category": "Null Handling",
        "name": "null_fill_int_constant",
        "dest_table": "dest_null_fill_int_constant",
        "prompt": "Set 'loyalty_score' to NULL where 'status' is 'inactive', then fill all NULL values in 'loyalty_score' with the integer 0.",
        "verify": lambda df: df['loyalty_score'].notna().all() and (df.loc[df['status'] == 'inactive', 'loyalty_score'] == 0).all()
    },
    {
        "category": "Null Handling",
        "name": "null_fill_str_constant",
        "dest_table": "dest_null_fill_str_constant",
        "prompt": "Set 'status' to NULL where 'age' is less than 30, then fill all NULL values in 'status' with the string 'UNKNOWN'.",
        "verify": lambda df: df['status'].notna().all() and (df.loc[df['age'] < 30, 'status'] == 'UNKNOWN').all()
    },
    {
        "category": "Null Handling",
        "name": "null_fill_mean",
        "dest_table": "dest_null_fill_mean",
        "prompt": "Set 'account_balance' to NULL where 'id' is 2 or 4, then fill those NULL values with the mean of the non-null values in 'account_balance'.",
        "verify": lambda df: df['account_balance'].notna().all()
    },
    {
        "category": "Null Handling",
        "name": "null_fill_median",
        "dest_table": "dest_null_fill_median",
        "prompt": "Set 'age' to NULL where 'status' is 'pending', then fill NULL values in 'age' with the median of the non-null 'age' values.",
        "verify": lambda df: df['age'].notna().all()
    },
    {
        "category": "Null Handling",
        "name": "null_forward_fill",
        "dest_table": "dest_null_forward_fill",
        "prompt": "Set 'loyalty_score' to NULL where 'id' is 3, then apply a forward fill on 'loyalty_score' ordered by 'id' to fill the gap.",
        "verify": lambda df: df['loyalty_score'].notna().all() and df.loc[df['id'] == 3, 'loyalty_score'].iloc[0] == 61
    },
    {
        "category": "Null Handling",
        "name": "null_backward_fill",
        "dest_table": "dest_null_backward_fill",
        "prompt": "Set 'loyalty_score' to NULL where 'id' is 3, then apply a backward fill on 'loyalty_score' ordered by 'id' to fill the gap.",
        "verify": lambda df: (df['loyalty_score'].notna().all() and df.loc[df['id'] == 3, 'loyalty_score'].iloc[0] == df.loc[df['id'] == 4, 'loyalty_score'].iloc[0])
    },
    {
        "category": "Null Handling",
        "name": "null_drop_any",
        "dest_table": "dest_null_drop_any",
        "prompt": "Set 'email_address' to NULL where 'id' is 5, then drop any rows that have at least one NULL value in any column.",
        "verify": lambda df: df.notna().all().all() and len(df) == 4
    },
    {
        "category": "Null Handling",
        "name": "null_drop_specific_col",
        "dest_table": "dest_null_drop_specific_col",
        "prompt": "Set 'ssn' to NULL where 'id' is 1, then drop all rows where 'ssn' is NULL.",
        "verify": lambda df: df['ssn'].notna().all() and len(df) == 4
    },
    {
        "category": "Null Handling",
        "name": "null_conditional_fill",
        "dest_table": "dest_null_conditional_fill",
        "prompt": "Set 'account_balance' to NULL where 'status' is 'pending'. Then fill the NULL 'account_balance' values using the value from 'loyalty_score' for that same row.",
        "verify": lambda df: df['account_balance'].notna().all() and (df.loc[df['status'] == 'pending', 'account_balance'] == df.loc[df['status'] == 'pending', 'loyalty_score']).all()
    },
    {
        "category": "Null Handling",
        "name": "null_fill_multi_col",
        "dest_table": "dest_null_fill_multi_col",
        "prompt": "Set 'loyalty_score' to NULL where 'id' is 4 and 'account_balance' to NULL where 'id' is 2. Then fill NULLs in 'loyalty_score' with 0 and NULLs in 'account_balance' with 999.",
        "verify": lambda df: df['loyalty_score'].notna().all() and df['account_balance'].notna().all() and df.loc[df['id'] == 4, 'loyalty_score'].iloc[0] == 0 and df.loc[df['id'] == 2, 'account_balance'].iloc[0] == 999
    },
]

# ==========================================
# TEST SUITE: TYPE CASTING (10 Tests)
# ==========================================
TYPE_CASTING_TEST_SUITE = [
    {
        "category": "Type Casting",
        "name": "cast_id_str",
        "dest_table": "dest_cast_id_str",
        "prompt": "Cast the 'id' column to a string/varchar data type.",
        "verify": lambda df: pd.api.types.is_string_dtype(df['id'])
    },
    {
        "category": "Type Casting",
        "name": "cast_loyalty_float",
        "dest_table": "dest_cast_loyalty_float",
        "prompt": "Cast the 'loyalty_score' column from integer to a float data type.",
        "verify": lambda df: pd.api.types.is_float_dtype(df['loyalty_score'])
    },
    {
        "category": "Type Casting",
        "name": "cast_signup_date",
        "dest_table": "dest_cast_signup_date",
        "prompt": "Convert 'signup_date' from a string in DD/MM/YYYY format into a proper date object.",
        "verify": lambda df: pd.to_datetime(df['signup_date'], errors='coerce').notna().all()
    },
    {
        "category": "Type Casting",
        "name": "cast_balance_int",
        "dest_table": "dest_cast_balance_int",
        "prompt": "Cast 'account_balance' to an integer, truncating any decimals.",
        "verify": lambda df: pd.api.types.is_integer_dtype(df['account_balance'])
    },
    {
        "category": "Type Casting",
        "name": "cast_age_float",
        "dest_table": "dest_cast_age_float",
        "prompt": "Cast the 'age' column to a float data type.",
        "verify": lambda df: pd.api.types.is_float_dtype(df['age'])
    },
    {
        "category": "Type Casting",
        "name": "cast_status_bool",
        "dest_table": "dest_cast_status_bool",
        "prompt": "Create a boolean column 'is_active' that is True where 'status' is 'active', and False otherwise.",
        "verify": lambda df: "is_active" in df.columns and pd.api.types.is_bool_dtype(df['is_active']) and df.loc[df['status'] == 'active', 'is_active'].all() and not df.loc[df['status'] != 'active', 'is_active'].any()
    },
    {
        "category": "Type Casting",
        "name": "cast_id_float",
        "dest_table": "dest_cast_id_float",
        "prompt": "Cast the 'id' column to a float data type.",
        "verify": lambda df: pd.api.types.is_float_dtype(df['id'])
    },
    {
        "category": "Type Casting",
        "name": "cast_balance_str",
        "dest_table": "dest_cast_balance_str",
        "prompt": "Convert 'account_balance' to a string/varchar data type.",
        "verify": lambda df: pd.api.types.is_string_dtype(df['account_balance'])
    },
    {
        "category": "Type Casting",
        "name": "cast_ssn_bigint",
        "dest_table": "dest_cast_ssn_bigint",
        "prompt": "Remove all hyphens from 'ssn' and cast the result to a Big Integer stored in a new column 'ssn_num'.",
        "verify": lambda df: "ssn_num" in df.columns and pd.api.types.is_integer_dtype(df['ssn_num']) and (df['ssn_num'] > 0).all()
    },
    {
        "category": "Type Casting",
        "name": "cast_loyalty_str",
        "dest_table": "dest_cast_loyalty_str",
        "prompt": "Cast the 'loyalty_score' column to a string data type.",
        "verify": lambda df: pd.api.types.is_string_dtype(df['loyalty_score'])
    },
]

# ==========================================
# TEST SUITE: STRING MANIPULATION (15 Tests)
# ==========================================
STRING_MANIPULATION_TEST_SUITE = [
    {
        "category": "String Manipulation",
        "name": "str_upper_first_name",
        "dest_table": "dest_str_upper_first_name",
        "prompt": "Convert all values in 'first_name' to uppercase and save to a new column 'first_name_upper'.",
        "verify": lambda df: "first_name_upper" in df.columns and (df['first_name_upper'] == df['first_name'].str.upper()).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_lower_status",
        "dest_table": "dest_str_lower_status",
        "prompt": "Convert all values in 'status' to lowercase and overwrite the column in place.",
        "verify": lambda df: (df['status'] == df['status'].str.lower()).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_title_name",
        "dest_table": "dest_str_title_name",
        "prompt": "Apply title case to the 'first_name' column, storing the result in a new column 'first_name_title'.",
        "verify": lambda df: "first_name_title" in df.columns and (df['first_name_title'] == df['first_name'].str.title()).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_strip_email",
        "dest_table": "dest_str_strip_email",
        "prompt": "Strip any leading and trailing whitespace from the 'email_address' column and overwrite it in place.",
        "verify": lambda df: (df['email_address'] == df['email_address'].str.strip()).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_prefix_chars",
        "dest_table": "dest_str_prefix_chars",
        "prompt": "Extract the first 3 characters of 'first_name' into a new column 'name_prefix'.",
        "verify": lambda df: "name_prefix" in df.columns and (df['name_prefix'] == df['first_name'].str[:3]).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_substr_email_local",
        "dest_table": "dest_str_substr_email_local",
        "prompt": "Extract the local part of 'email_address' (everything before the '@') into a new column 'email_local'.",
        "verify": lambda df: "email_local" in df.columns and not df['email_local'].str.contains('@').any()
    },
    {
        "category": "String Manipulation",
        "name": "str_extract_domain",
        "dest_table": "dest_str_extract_domain",
        "prompt": "Extract the domain portion of 'email_address' (everything after '@') into a new column 'email_domain'.",
        "verify": lambda df: "email_domain" in df.columns and not df['email_domain'].str.contains('@').any() and df['email_domain'].str.contains(r'\.').all()
    },
    {
        "category": "String Manipulation",
        "name": "str_concat_full_label",
        "dest_table": "dest_str_concat_full_label",
        "prompt": "Concatenate 'first_name' and 'status' with a hyphen separator into a new column 'name_status'.",
        "verify": lambda df: "name_status" in df.columns and (df['name_status'] == df['first_name'] + '-' + df['status']).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_replace_domain",
        "dest_table": "dest_str_replace_domain",
        "prompt": "Replace 'example.com' with 'company.org' in the 'email_address' column and save the result in a new column 'new_email'.",
        "verify": lambda df: "new_email" in df.columns and df['new_email'].str.endswith('company.org').all()
    },
    {
        "category": "String Manipulation",
        "name": "str_prefix_constant",
        "dest_table": "dest_str_prefix_constant",
        "prompt": "Prepend the string 'USR_' to each value in the 'id' column (cast to string first) and store the result in a new column 'user_code'.",
        "verify": lambda df: "user_code" in df.columns and df['user_code'].str.startswith('USR_').all()
    },
    {
        "category": "String Manipulation",
        "name": "str_suffix_constant",
        "dest_table": "dest_str_suffix_constant",
        "prompt": "Append the suffix '_verified' to every value in the 'status' column and store in a new column 'status_flag'.",
        "verify": lambda df: "status_flag" in df.columns and df['status_flag'].str.endswith('_verified').all()
    },
    {
        "category": "String Manipulation",
        "name": "str_char_count",
        "dest_table": "dest_str_char_count",
        "prompt": "Count the number of times the character 'e' appears (case-insensitive) in 'first_name' and store the result in a new integer column 'e_count'.",
        "verify": lambda df: "e_count" in df.columns and (df['e_count'] == df['first_name'].str.lower().str.count('e')).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_length",
        "dest_table": "dest_str_length",
        "prompt": "Calculate the character length of each value in 'first_name' and store it in a new integer column 'name_length'.",
        "verify": lambda df: "name_length" in df.columns and (df['name_length'] == df['first_name'].str.len()).all()
    },
    {
        "category": "String Manipulation",
        "name": "str_mask_email",
        "dest_table": "dest_str_mask_email",
        "prompt": "Replace everything before the '@' in 'email_address' with '***' and save the result as 'masked_email'.",
        "verify": lambda df: "masked_email" in df.columns and df['masked_email'].str.startswith('***@').all()
    },
    {
        "category": "String Manipulation",
        "name": "str_clean_ssn",
        "dest_table": "dest_str_clean_ssn",
        "prompt": "Remove all hyphens from the 'ssn' column and store the result in a new column 'clean_ssn'.",
        "verify": lambda df: "clean_ssn" in df.columns and not df['clean_ssn'].str.contains('-').any() and (df['clean_ssn'].str.len() == 9).all()
    },
]

# ==========================================
# TEST SUITE: DATE & TIME HANDLING (12 Tests)
# ==========================================
DATETIME_HANDLING_TEST_SUITE = [
    {
        "category": "Date & Time Handling",
        "name": "date_parse_to_date",
        "dest_table": "dest_date_parse_to_date",
        "prompt": "Parse the 'signup_date' column from its current DD/MM/YYYYn string format into a proper date/datetime object.",
        "verify": lambda df: (pd.api.types.is_datetime64_any_dtype(df['signup_date']) or pd.api.types.is_object_dtype(df['signup_date']))
    },
    {
        "category": "Date & Time Handling",
        "name": "date_reformat_iso",
        "dest_table": "dest_date_reformat_iso",
        "prompt": "Convert 'signup_date' from DD/MM/YYYY string format to a new string format of YYYY-MM-DD, storing the result in a new column 'signup_date_iso'.",
        "verify": lambda df: "signup_date_iso" in df.columns and df['signup_date_iso'].str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_extract_year",
        "dest_table": "dest_date_extract_year",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and extract just the year as a new integer column 'signup_year'.",
        "verify": lambda df: "signup_year" in df.columns and set(df['signup_year'].tolist()) == {2020, 2021, 2022, 2023}
    },
    {
        "category": "Date & Time Handling",
        "name": "date_extract_month",
        "dest_table": "dest_date_extract_month",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and extract the month as a new integer column 'signup_month'.",
        "verify": lambda df: "signup_month" in df.columns and df['signup_month'].between(1, 12).all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_extract_day",
        "dest_table": "dest_date_extract_day",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and extract the day of the month as a new integer column 'signup_day'.",
        "verify": lambda df: "signup_day" in df.columns and df['signup_day'].between(1, 31).all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_extract_dayofweek",
        "dest_table": "dest_date_extract_dayofweek",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and extract the day of the week as an integer (0=Monday, 6=Sunday) into a new column 'signup_dayofweek'.",
        "verify": lambda df: "signup_dayofweek" in df.columns and df['signup_dayofweek'].between(0, 6).all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_days_since_signup",
        "dest_table": "dest_date_days_since_signup",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and calculate the number of days elapsed from signup to today's date, storing it as an integer column 'days_since_signup'.",
        "verify": lambda df: "days_since_signup" in df.columns and (df['days_since_signup'] > 0).all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_add_days",
        "dest_table": "dest_date_add_days",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY), add 30 days to it, and store the resulting date as a new column 'signup_date_plus_30' in YYYY-MM-DD format.",
        "verify": lambda df: "signup_date_plus_30" in df.columns and df['signup_date_plus_30'].str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_subtract_days",
        "dest_table": "dest_date_subtract_days",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY), subtract 7 days from it, and store the result as a new column 'pre_signup_date' in YYYY-MM-DD format.",
        "verify": lambda df: "pre_signup_date" in df.columns and df['pre_signup_date'].str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_filter_by_year",
        "dest_table": "dest_date_filter_by_year",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY), extract the signup year into a new column 'signup_year', and keep only the rows where signup_year is 2021.",
        "verify": lambda df: len(df) == 2 and all(
            pd.to_datetime(d, dayfirst=True).year == 2021 for d in df['signup_date']
        ) if not pd.api.types.is_datetime64_any_dtype(df['signup_date'])
        else (df['signup_date'].dt.year == 2021).all()
    },
    {
        "category": "Date & Time Handling",
        "name": "date_bucket_era",
        "dest_table": "dest_date_bucket_era",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY) and create a new column 'signup_era': set it to 'early' if the year is before 2022, and 'recent' if 2022 or later.",
        "verify": lambda df: "signup_era" in df.columns and set(df['signup_era'].unique()).issubset({'early', 'recent'}) and len(df[df['signup_era'] == 'early']) == 3 and len(df[df['signup_era'] == 'recent']) == 2
    },
    {
        "category": "Date & Time Handling",
        "name": "date_days_between_derived",
        "dest_table": "dest_date_days_between_derived",
        "prompt": "Parse 'signup_date' (DD/MM/YYYY). Add 'age' multiplied by 365 days to the signup date to approximate a 'birth_date', and compute the difference in days between that birth_date and signup_date, storing it as 'age_in_days'.",
        "verify": lambda df: "age_in_days" in df.columns and (df['age_in_days'] > 0).all()
    },
]

# ==========================================
# TEST SUITE: ARITHMETIC & NUMERIC OPERATIONS (10 Tests)
# ==========================================
ARITHMETIC_OPERATIONS_TEST_SUITE = [
    {
        "category": "Arithmetic Operations",
        "name": "arith_add_100",
        "dest_table": "dest_arith_add_100",
        "prompt": "Add 100 to the 'account_balance' column and store the result in a new column 'adjusted_balance'.",
        "verify": lambda df: "adjusted_balance" in df.columns and (df['adjusted_balance'] == df['account_balance'] + 100).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_mult_loyalty",
        "dest_table": "dest_arith_mult_loyalty",
        "prompt": "Multiply 'loyalty_score' by 1.5 and store the result in a new column 'boosted_loyalty'.",
        "verify": lambda df: "boosted_loyalty" in df.columns and (df['boosted_loyalty'] == df['loyalty_score'] * 1.5).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_sub_age",
        "dest_table": "dest_arith_sub_age",
        "prompt": "Subtract 'age' from 100 and store the result in a new column 'years_to_100'.",
        "verify": lambda df: "years_to_100" in df.columns and (df['years_to_100'] == 100 - df['age']).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_div_balance_age",
        "dest_table": "dest_arith_div_balance_age",
        "prompt": "Divide 'account_balance' by 'age' and save the result as a new column 'balance_per_year'.",
        "verify": lambda df: "balance_per_year" in df.columns and (round(df['balance_per_year'], 4) == round(df['account_balance'] / df['age'], 4)).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_mod_id",
        "dest_table": "dest_arith_mod_id",
        "prompt": "Calculate 'id' modulo 2 and store the result in a new column 'id_parity'.",
        "verify": lambda df: "id_parity" in df.columns and (df['id_parity'] == df['id'] % 2).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_add_two_cols",
        "dest_table": "dest_arith_add_two_cols",
        "prompt": "Add 'loyalty_score' and 'age' together into a new column named 'combined_metric'.",
        "verify": lambda df: "combined_metric" in df.columns and (df['combined_metric'] == df['loyalty_score'] + df['age']).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_percent_increase",
        "dest_table": "dest_arith_percent_increase",
        "prompt": "Increase 'account_balance' by 5 percent and store the result in a new column 'new_balance'.",
        "verify": lambda df: "new_balance" in df.columns and (round(df['new_balance'], 2) == round(df['account_balance'] * 1.05, 2)).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_square_age",
        "dest_table": "dest_arith_square_age",
        "prompt": "Square the 'age' column and store the result in a new column 'age_squared'.",
        "verify": lambda df: "age_squared" in df.columns and (df['age_squared'] == df['age'] ** 2).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_diff_cols",
        "dest_table": "dest_arith_diff_cols",
        "prompt": "Subtract 'loyalty_score' from 'account_balance' and save the result in a new column 'balance_loyalty_diff'.",
        "verify": lambda df: "balance_loyalty_diff" in df.columns and (df['balance_loyalty_diff'] == df['account_balance'] - df['loyalty_score']).all()
    },
    {
        "category": "Arithmetic Operations",
        "name": "arith_half_balance",
        "dest_table": "dest_arith_half_balance",
        "prompt": "Divide 'account_balance' by 2 and store the result in a new column 'half_balance'.",
        "verify": lambda df: "half_balance" in df.columns and (df['half_balance'] == df['account_balance'] / 2).all()
    },
]

# ==========================================
# TEST SUITE: FILTERING & CONDITIONAL LOGIC (15 Tests)
# ==========================================
FILTERING_LOGIC_TEST_SUITE = [
    {
        "category": "Filtering Logic",
        "name": "filt_age_gt_30",
        "dest_table": "dest_filt_age_gt_30",
        "prompt": "Filter the dataset to only include rows where 'age' is strictly greater than 30.",
        "verify": lambda df: (df['age'] > 30).all() and len(df) > 0
    },
    {
        "category": "Filtering Logic",
        "name": "filt_age_lt_40",
        "dest_table": "dest_filt_age_lt_40",
        "prompt": "Keep only the rows where 'age' is strictly less than 40.",
        "verify": lambda df: (df['age'] < 40).all() and len(df) > 0
    },
    {
        "category": "Filtering Logic",
        "name": "filt_status_active",
        "dest_table": "dest_filt_status_active",
        "prompt": "Filter for rows where 'status' is exactly 'active'.",
        "verify": lambda df: (df['status'] == 'active').all() and len(df) == 2
    },
    {
        "category": "Filtering Logic",
        "name": "filt_bal_lt_500",
        "dest_table": "dest_filt_bal_lt_500",
        "prompt": "Keep only the rows where 'account_balance' is less than 500.",
        "verify": lambda df: (df['account_balance'] < 500).all() and len(df) > 0
    },
    {
        "category": "Filtering Logic",
        "name": "filt_loyalty_between",
        "dest_table": "dest_filt_loyalty_between",
        "prompt": "Filter for rows where 'loyalty_score' is between 50 and 90 (inclusive on both ends).",
        "verify": lambda df: df['loyalty_score'].between(50, 90).all() and len(df) > 0
    },
    {
        "category": "Filtering Logic",
        "name": "filt_name_starts_a",
        "dest_table": "dest_filt_name_starts_a",
        "prompt": "Keep only the rows where 'first_name' starts with the letter 'A'.",
        "verify": lambda df: df['first_name'].str.startswith('A').all() and len(df) == 1
    },
    {
        "category": "Filtering Logic",
        "name": "filt_email_contains_num",
        "dest_table": "dest_filt_email_contains_num",
        "prompt": "Filter for rows where 'email_address' contains the substring '474'.",
        "verify": lambda df: df['email_address'].str.contains('474').all() and len(df) == 1
    },
    {
        "category": "Filtering Logic",
        "name": "filt_id_isin",
        "dest_table": "dest_filt_id_isin",
        "prompt": "Filter the dataset to only include rows where 'id' is in the list [1, 3, 5].",
        "verify": lambda df: df['id'].isin([1, 3, 5]).all() and len(df) == 3
    },
    {
        "category": "Filtering Logic",
        "name": "filt_multi_and",
        "dest_table": "dest_filt_multi_and",
        "prompt": "Filter for rows where 'status' is 'pending' AND 'age' is less than 30.",
        "verify": lambda df: ((df['status'] == 'pending') & (df['age'] < 30)).all() and len(df) == 1
    },
    {
        "category": "Filtering Logic",
        "name": "filt_multi_or",
        "dest_table": "dest_filt_multi_or",
        "prompt": "Filter for rows where 'account_balance' is greater than 500 OR 'loyalty_score' is greater than 85.",
        "verify": lambda df: ((df['account_balance'] > 500) | (df['loyalty_score'] > 85)).all() and len(df) > 0
    },
    {
        "category": "Filtering Logic",
        "name": "filt_not_inactive",
        "dest_table": "dest_filt_not_inactive",
        "prompt": "Exclude all rows where 'status' is 'inactive'.",
        "verify": lambda df: (~(df['status'] == 'inactive')).all() and len(df) == 4
    },
    {
        "category": "Filtering Logic",
        "name": "filt_case_when_tier",
        "dest_table": "dest_filt_case_when_tier",
        "prompt": "Add a new column 'balance_tier': set it to 'low' if 'account_balance' is less than 300, 'mid' if between 300 and 700 inclusive, and 'high' if greater than 700.",
        "verify": lambda df: "balance_tier" in df.columns and df.loc[df['account_balance'] < 300, 'balance_tier'].eq('low').all() and df.loc[(df['account_balance'] >= 300) & (df['account_balance'] <= 700), 'balance_tier'].eq('mid').all() and df.loc[df['account_balance'] > 700, 'balance_tier'].eq('high').all()
    },
    {
        "category": "Filtering Logic",
        "name": "filt_case_when_loyalty_label",
        "dest_table": "dest_filt_case_when_loyalty_label",
        "prompt": "Add a new column 'loyalty_label': 'bronze' if 'loyalty_score' < 50, 'silver' if between 50 and 80 inclusive, 'gold' if greater than 80.",
        "verify": lambda df: "loyalty_label" in df.columns and df.loc[df['loyalty_score'] < 50, 'loyalty_label'].eq('bronze').all() and df.loc[(df['loyalty_score'] >= 50) & (df['loyalty_score'] <= 80), 'loyalty_label'].eq('silver').all() and df.loc[df['loyalty_score'] > 80, 'loyalty_label'].eq('gold').all()
    },
    {
        "category": "Filtering Logic",
        "name": "filt_case_when_status_flag",
        "dest_table": "dest_filt_case_when_status_flag",
        "prompt": "Create a new column 'priority': set to 'high' if 'status' is 'active' and 'account_balance' > 500, 'medium' if 'status' is 'active' and 'account_balance' <= 500, and 'low' for all other statuses.",
        "verify": lambda df: "priority" in df.columns and df.loc[(df['status'] == 'active') & (df['account_balance'] > 500), 'priority'].eq('high').all() and df.loc[(df['status'] == 'active') & (df['account_balance'] <= 500), 'priority'].eq('medium').all() and df.loc[df['status'] != 'active', 'priority'].eq('low').all()
    },
    {
        "category": "Filtering Logic",
        "name": "filt_date_year_contains",
        "dest_table": "dest_filt_date_year_contains",
        "prompt": "Keep only the rows where 'signup_date' contains the year '2021' as a substring.",
        "verify": lambda df: df['signup_date'].str.contains('2021').all() and len(df) == 2
    },
]

# ==========================================
# TEST SUITE: DATA MASKING / PII HANDLING (12 Tests)
# ==========================================
DATA_MASKING_TEST_SUITE = [
    {
        "category": "Data Masking",
        "name": "mask_ssn_last4",
        "dest_table": "dest_mask_ssn_last4",
        "prompt": "Mask the 'ssn' column so that only the last 4 digits are visible, replacing the rest with 'XXX-XX-'. Store the result in a new column 'masked_ssn'.",
        "verify": lambda df: "masked_ssn" in df.columns and df['masked_ssn'].str.startswith('XXX-XX-').all() and (df['masked_ssn'].str.len() == 11).all()
    },
    {
        "category": "Data Masking",
        "name": "mask_ssn_partial_format",
        "dest_table": "dest_mask_ssn_partial_format",
        "prompt": "Reformat 'ssn' to show only the last 4 digits in the format '***-**-NNNN', stored in a new column 'ssn_partial'.",
        "verify": lambda df: "ssn_partial" in df.columns and df['ssn_partial'].str.startswith('***-**-').all()
    },
    {
        "category": "Data Masking",
        "name": "mask_ssn_md5",
        "dest_table": "dest_mask_ssn_md5",
        "prompt": "Hash the 'ssn' column using MD5 and store the hex digest in a new column 'ssn_hash'.",
        "verify": lambda df: "ssn_hash" in df.columns and (df['ssn_hash'].str.len() == 32).all() and df['ssn_hash'].str.match(r'^[a-f0-9]{32}$').all()
    },
    {
        "category": "Data Masking",
        "name": "mask_email_sha256",
        "dest_table": "dest_mask_email_sha256",
        "prompt": "Hash the 'email_address' column using SHA-256 and store the hex digest in a new column 'email_hash'.",
        "verify": lambda df: "email_hash" in df.columns and (df['email_hash'].str.len() == 64).all() and df['email_hash'].str.match(r'^[a-f0-9]{64}$').all()
    },
    {
        "category": "Data Masking",
        "name": "mask_email_local_stars",
        "dest_table": "dest_mask_email_local_stars",
        "prompt": "Replace the local part of 'email_address' (everything before '@') with '***' and store as 'masked_email'.",
        "verify": lambda df: "masked_email" in df.columns and df['masked_email'].str.startswith('***@').all()
    },
    {
        "category": "Data Masking",
        "name": "mask_name_redact",
        "dest_table": "dest_mask_name_redact",
        "prompt": "Replace every value in the 'first_name' column with the string 'REDACTED'.",
        "verify": lambda df: (df['first_name'] == 'REDACTED').all()
    },
    {
        "category": "Data Masking",
        "name": "mask_balance_zero",
        "dest_table": "dest_mask_balance_zero",
        "prompt": "Replace every value in 'account_balance' with 0 to mask the financial data.",
        "verify": lambda df: (df['account_balance'] == 0).all()
    },
    {
        "category": "Data Masking",
        "name": "mask_drop_ssn",
        "dest_table": "dest_mask_drop_ssn",
        "prompt": "Remove the 'ssn' column entirely as a PII masking measure.",
        "verify": lambda df: "ssn" not in df.columns
    },
    {
        "category": "Data Masking",
        "name": "mask_name_md5",
        "dest_table": "dest_mask_name_md5",
        "prompt": "Hash the 'first_name' column using MD5 and store the result in a new column 'name_hash', replacing the original.",
        "verify": lambda df: "name_hash" in df.columns and (df['name_hash'].str.len() == 32).all()
    },
    {
        "category": "Data Masking",
        "name": "mask_age_decade_bucket",
        "dest_table": "dest_mask_age_decade_bucket",
        "prompt": "Generalize the 'age' column into decade buckets stored in a new column 'age_group': ages 20-29 become '20s', 30-39 become '30s', 40-49 become '40s', 50-59 become '50s'.",
        "verify": lambda df: "age_group" in df.columns and set(df['age_group'].unique()).issubset({'20s', '30s', '40s', '50s'}) and df.loc[df['age'] == 28, 'age_group'].iloc[0] == '20s' and df.loc[df['age'] == 55, 'age_group'].iloc[0] == '50s'
    },
    {
        "category": "Data Masking",
        "name": "mask_email_tokenize",
        "dest_table": "dest_mask_email_tokenize",
        "prompt": "Replace each 'email_address' with a deterministic token by computing the MD5 hash of the email and appending '@token.internal', stored in a new column 'tokenized_email'.",
        "verify": lambda df: "tokenized_email" in df.columns and df['tokenized_email'].str.endswith('@token.internal').all() and df['tokenized_email'].nunique() == 5
    },
    {
        "category": "Data Masking",
        "name": "mask_id_last2",
        "dest_table": "dest_mask_id_last2",
        "prompt": "Create a truncated identifier 'truncated_id' by taking the string representation of 'id' and keeping only the last 2 characters, left-padded with '*' to a fixed width of 4.",
        "verify": lambda df: "truncated_id" in df.columns and (df['truncated_id'].str.len() == 4).all() and df['truncated_id'].str.startswith('**').all()
    },
]

# ==========================================
# TEST SUITE: UDFs — USER DEFINED FUNCTIONS (10 Tests)
# ==========================================
UDF_TEST_SUITE = [
    {
        "category": "UDF",
        "name": "udf_weighted_score",
        "dest_table": "dest_udf_weighted_score",
        "prompt": "Apply a custom Python UDF that computes a 'composite_score' for each row as: (loyalty_score * 0.7) + (age * 0.3), rounded to 2 decimal places.",
        "verify": lambda df: "composite_score" in df.columns and (round(df.loc[df['id'] == 1, 'composite_score'].iloc[0], 2) == round(84 * 0.7 + 28 * 0.3, 2))
    },
    {
        "category": "UDF",
        "name": "udf_balance_tier",
        "dest_table": "dest_udf_balance_tier",
        "prompt": "Apply a custom Python UDF to create a 'balance_tier' column: return 'low' if account_balance < 200, 'mid' if between 200 and 600 inclusive, and 'high' if above 600.",
        "verify": lambda df: "balance_tier" in df.columns and df.loc[df['account_balance'] == 20, 'balance_tier'].iloc[0] == 'low' and df.loc[df['account_balance'] == 880, 'balance_tier'].iloc[0] == 'high'
    },
    {
        "category": "UDF",
        "name": "udf_name_normalizer",
        "dest_table": "dest_udf_name_normalizer",
        "prompt": "Apply a custom Python UDF that normalizes 'first_name' by stripping whitespace, converting to title case, and replacing any non-alphabetic characters with an underscore. Store in 'normalized_name'.",
        "verify": lambda df: "normalized_name" in df.columns and (df['normalized_name'] == df['first_name'].str.strip().str.title()).all()
    },
    {
        "category": "UDF",
        "name": "udf_date_parser",
        "dest_table": "dest_udf_date_parser",
        "prompt": "Apply a custom Python UDF that parses 'signup_date' from the non-standard format DD/MM/YYYY and returns the corresponding ISO format string YYYY-MM-DD, stored in 'parsed_date'.",
        "verify": lambda df: "parsed_date" in df.columns and df['parsed_date'].str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    },
    {
        "category": "UDF",
        "name": "udf_ssn_validator",
        "dest_table": "dest_udf_ssn_validator",
        "prompt": "Apply a custom Python UDF that checks whether 'ssn' matches the regex pattern XXX-XX-XXXX (3 digits, hyphen, 2 digits, hyphen, 4 digits). Store True/False in a new boolean column 'ssn_valid'.",
        "verify": lambda df: "ssn_valid" in df.columns and pd.api.types.is_bool_dtype(df['ssn_valid']) and df['ssn_valid'].all()
    },
    {
        "category": "UDF",
        "name": "udf_email_domain_safe",
        "dest_table": "dest_udf_email_domain_safe",
        "prompt": "Apply a custom Python UDF to extract the domain from 'email_address' (everything after '@'). If there is no '@' in the value, return 'unknown'. Store in 'safe_domain'.",
        "verify": lambda df: "safe_domain" in df.columns and (df['safe_domain'] == 'example.com').all()
    },
    {
        "category": "UDF",
        "name": "udf_age_bucket_complex",
        "dest_table": "dest_udf_age_bucket_complex",
        "prompt": "Apply a custom Python UDF that categorizes 'age' into 'young' (under 30), 'middle-aged' (30 to 50 inclusive), or 'senior' (over 50), stored in 'age_category'.",
        "verify": lambda df: "age_category" in df.columns and df.loc[df['age'] == 28, 'age_category'].iloc[0] == 'young' and df.loc[df['age'] == 45, 'age_category'].iloc[0] == 'middle-aged' and df.loc[df['age'] == 55, 'age_category'].iloc[0] == 'senior'
    },
    {
        "category": "UDF",
        "name": "udf_composite_key",
        "dest_table": "dest_udf_composite_key",
        "prompt": "Apply a custom Python UDF that generates a 'composite_key' by combining 'id' (zero-padded to 4 digits), an underscore, and the first 3 uppercase letters of 'first_name'. E.g., id=1, first_name='Alice' → '0001_ALI'.",
        "verify": lambda df: "composite_key" in df.columns and df.loc[df['id'] == 1, 'composite_key'].iloc[0] == '0001_ALI' and df.loc[df['id'] == 5, 'composite_key'].iloc[0] == '0005_EVE'
    },
    {
        "category": "UDF",
        "name": "udf_greeting_message",
        "dest_table": "dest_udf_greeting_message",
        "prompt": "Apply a custom Python UDF to generate a 'greeting' column with the message: 'Hello, {first_name}! Your account is currently {status}' filled in per row",
        "verify": lambda df: "greeting" in df.columns and df.loc[df['id'] == 1, 'greeting'].iloc[0] == 'Hello, Alice! Your account is currently active' and df.loc[df['id'] == 2, 'greeting'].iloc[0] == 'Hello, Bob! Your account is currently pending'
    },
    {
        "category": "UDF",
        "name": "udf_loyalty_tier_complex",
        "dest_table": "dest_udf_loyalty_tier_complex",
        "prompt": "Apply a custom Python UDF that computes a 'reward_points' column using the following business logic: if loyalty_score >= 80, multiply account_balance by 0.1; if loyalty_score between 50 and 79, multiply by 0.05; otherwise multiply by 0.01. Round to 2 decimal places.",
        "verify": lambda df: "reward_points" in df.columns and round(df.loc[df['id'] == 1, 'reward_points'].iloc[0], 2) == round(880 * 0.1, 2) and round(df.loc[df['id'] == 2, 'reward_points'].iloc[0], 2) == round(230 * 0.05, 2) and round(df.loc[df['id'] == 4, 'reward_points'].iloc[0], 2) == round(450 * 0.01, 2)
    },
]

# ==========================================
# TEST SUITE: AGGREGATIONS & GROUP BY (12 Tests)
# ==========================================
AGGREGATION_TEST_SUITE = [
    {
        "category": "Aggregation",
        "name": "agg_count_by_status",
        "dest_table": "dest_agg_count_by_status",
        "prompt": "Group the data by 'status' and count the number of rows in each group. The result should have columns 'status' and 'count'.",
        "verify": lambda df: "status" in df.columns and "count" in df.columns and len(df) == 3 and df.loc[df['status'] == 'active', 'count'].iloc[0] == 2
    },
    {
        "category": "Aggregation",
        "name": "agg_sum_balance_by_status",
        "dest_table": "dest_agg_sum_balance_by_status",
        "prompt": "Group by 'status' and compute the sum of 'account_balance' as 'account_balance'. The result should have columns 'status' and 'account_balance'.",
        "verify": lambda df:
            "status" in df.columns and
            "account_balance" in df.columns and
            df.loc[df['status'] == 'active', 'account_balance'].iloc[0] == 900
    },
    {
        "category": "Aggregation",
        "name": "agg_mean_loyalty_by_status",
        "dest_table": "dest_agg_mean_loyalty_by_status",
        "prompt": "Group by 'status' and compute the mean 'loyalty_score' per group, stored as 'avg_loyalty'.",
        "verify": lambda df: "avg_loyalty" in df.columns and abs(df.loc[df['status'] == 'active', 'avg_loyalty'].iloc[0] - 88.0) < 0.01
    },
    {
        "category": "Aggregation",
        "name": "agg_max_balance_by_status",
        "dest_table": "dest_agg_max_balance_by_status",
        "prompt": "Group by 'status' and find the maximum 'account_balance' per group, stored as 'max_balance'.",
        "verify": lambda df: "max_balance" in df.columns and df.loc[df['status'] == 'active', 'max_balance'].iloc[0] == 880
    },
    {
        "category": "Aggregation",
        "name": "agg_min_age_by_status",
        "dest_table": "dest_agg_min_age_by_status",
        "prompt": "Group by 'status' and find the minimum 'age' per group, stored as 'min_age'.",
        "verify": lambda df: "min_age" in df.columns and df.loc[df['status'] == 'active', 'min_age'].iloc[0] == 28
    },
    {
        "category": "Aggregation",
        "name": "agg_count_distinct_names_by_status",
        "dest_table": "dest_agg_count_distinct_names_by_status",
        "prompt": "Group by 'status' and count the number of distinct 'first_name' values per group, stored as 'unique_names'.",
        "verify": lambda df: "unique_names" in df.columns and df.loc[df['status'] == 'active', 'unique_names'].iloc[0] == 2
    },
    {
        "category": "Aggregation",
        "name": "agg_stddev_balance_by_status",
        "dest_table": "dest_agg_stddev_balance_by_status",
        "prompt": "Group by 'status' and compute the standard deviation of 'account_balance' per group, stored as 'balance_stddev'.",
        "verify": lambda df: "balance_stddev" in df.columns and df['balance_stddev'].notna().all()
    },
    {
        "category": "Aggregation",
        "name": "agg_multi_agg",
        "dest_table": "dest_agg_multi_agg",
        "prompt": "Group by 'status' and compute all three of: sum of 'account_balance' as 'total_balance', mean of 'loyalty_score' as 'avg_loyalty', and count of rows as 'member_count' — all in one operation.",
        "verify": lambda df: all(c in df.columns for c in ['status', 'total_balance', 'avg_loyalty', 'member_count']) and df.loc[df['status'] == 'active', 'member_count'].iloc[0] == 2
    },
    {
        "category": "Aggregation",
        "name": "agg_having_count_gt_1",
        "dest_table": "dest_agg_having_count_gt_1",
        "prompt": "Group by 'status', count the rows per group, and then keep only the groups where the count is greater than 1. The result should have columns 'status' and 'count'.",
        "verify": lambda df: (df['count'] > 1).all() and len(df) == 2
    },
    {
        "category": "Aggregation",
        "name": "agg_rank_within_group",
        "dest_table": "dest_agg_rank_within_group",
        "prompt": "Add a column 'balance_rank' that ranks each row by 'account_balance' within its 'status' group (rank 1 = highest balance), without aggregating or reducing rows.",
        "verify": lambda df: "balance_rank" in df.columns and len(df) == 5 and df.loc[df['id'] == 1, 'balance_rank'].iloc[0] == 1
    },
    {
        "category": "Aggregation",
        "name": "agg_cumulative_sum",
        "dest_table": "dest_agg_cumulative_sum",
        "prompt": "Compute a running/cumulative sum of 'account_balance' ordered by 'id', storing the result in a new column 'cumulative_balance'.",
        "verify": lambda df: "cumulative_balance" in df.columns and df.sort_values('id')['cumulative_balance'].iloc[0] == 880 and df.sort_values('id')['cumulative_balance'].iloc[-1] == (880 + 230 + 590 + 450 + 20)
    },
    {
        "category": "Aggregation",
        "name": "agg_global_stats",
        "dest_table": "dest_agg_global_stats",
        "prompt": "Compute global (no groupby) summary statistics: total sum of 'account_balance' as 'total_balance', overall mean of 'loyalty_score' as 'avg_loyalty', and total row count as 'total_rows'. The result should be a single-row table.",
        "verify": lambda df: len(df) == 1 and df['total_balance'].iloc[0] == 2170 and abs(df['avg_loyalty'].iloc[0] - 68.8) < 0.01 and df['total_rows'].iloc[0] == 5
    },
]


# ==========================================
# ACTIVE TEST SUITE — every category, driven through the planner.
# Swap to a single suite (e.g. STRING_MANIPULATION_TEST_SUITE) to run just one.
# ==========================================
TEST_SUITE = (
    COLUMN_OPERATIONS_TEST_SUITE
    + NULL_HANDLING_TEST_SUITE
    + TYPE_CASTING_TEST_SUITE
    + STRING_MANIPULATION_TEST_SUITE
    + DATETIME_HANDLING_TEST_SUITE
    + ARITHMETIC_OPERATIONS_TEST_SUITE
    + FILTERING_LOGIC_TEST_SUITE
    + DATA_MASKING_TEST_SUITE
    + UDF_TEST_SUITE
    + AGGREGATION_TEST_SUITE
)


# ==========================================
# CONNECTION CONFIG  (identical to test_daft_coder_pipeline.py)
# ==========================================
# Same private hosts the reference reaches over GKE. The planner otherwise forces
# local Docker for VPC-private hosts; keep it on GKE like the reference.
os.environ.setdefault("EXECUTION_ENV", "gke")
os.environ.setdefault("DTA_GKE_REACHES_PRIVATE", "1")

SOURCE_CREDS = {
    "host": "10.111.16.4",
    "port": 5432,
    "user": "avaloka",
    "password": "avaL0kapa$$word",
    "database": "avaloka_dest",
    "sslmode": "require",
    "table": "mock_crm_data",
}

# Destination lives on the same Postgres; `table` is set per test.
BASE_DEST_CREDS = {
    "host": "10.111.16.4",
    "port": 5432,
    "user": "avaloka",
    "password": "avaL0kapa$$word",
    "database": "avaloka_dest",
    "sslmode": "require",
}

SOURCE_ALIAS = "e2e_crm_source"
DEST_ALIAS = "e2e_warehouse_dest"


# ==========================================
# CUSTOM TERMINAL LOGGER  (mirrors the reference harness)
# ==========================================
class LoggerTee:
    """Duplicates sys.stdout/sys.stderr to both the terminal and a log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class LLMBehaviorFilter(logging.Filter):
    """Allows only critical LLM process logs, suppressing raw data / RAG dumps."""

    def filter(self, record):
        msg = record.getMessage()
        if "--- CODER STATE ---" in msg or "=== RETRIEVED DAFT DOCUMENTATION ===" in msg:
            return False
        if "Batches:" in msg or "Loading weights:" in msg or "HTTP Request" in msg:
            return False
        if "Using sqlalchemy" in msg or "PhysicalScan->" in msg or "[ray_job_runner]" in msg:
            if record.levelno < logging.ERROR:
                return False
        if "---SCHEMA_START---" in msg or "╭───" in msg:
            return False
        return True


def setup_logging():
    """Routes all Python logs AND terminal print statements to a file."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dta_e2e_d2d_{timestamp}.log")

    noisy_loggers = [
        "httpx", "httpcore", "urllib3", "sqlalchemy.engine.Engine",
        "daft.sql.sql_connection", "ray", "sentence_transformers", "chromadb",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.addFilter(LLMBehaviorFilter())
    root_logger.addHandler(file_handler)

    sys.stdout = LoggerTee(log_file)
    sys.stderr = sys.stdout

    return log_file


# ==========================================
# PLANNER + VERIFICATION HELPERS
# ==========================================
def _fresh_state():
    """A fresh planner conversation state (empty DB registry)."""
    return {"dta_database_registry": {}, "messages": []}


def _register_db(state, alias, creds):
    """Register a Postgres endpoint through the planner's register_database tool.

    RegisterDatabaseParams has no ``sslmode`` field, but the reference creds set
    ``sslmode: "require"`` (the cloud Postgres needs SSL). We inject it into the
    registry entry AFTER registration — ``_fetch_full_creds`` copies every
    registry key into the creds handed to ``build_connection_string`` / the sink,
    so SSL flows through for both source read and destination write.
    """
    params = RegisterDatabaseParams(
        alias=alias,
        db_type="postgresql",
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        user=creds["user"],
        password=creds["password"],
        table=creds.get("table"),
    )
    msg = _handle_register_database(state, params)
    entry = state["dta_database_registry"][alias]
    if creds.get("sslmode"):
        entry["sslmode"] = creds["sslmode"]
    print(f"   📌 register `{alias}` → {msg.splitlines()[0].strip()}")
    return entry


def _transfer_succeeded(planner_msg: str) -> bool:
    """True when the planner reports a completed transfer.

    The reliable cross-version marker is the ``✅ **Transfer complete:`` line
    (older planner builds omit the ``🎉 **SUCCESS!`` header that newer ones add).
    Every failure path prefixes a ``❌``, so success == "Transfer complete" present
    AND no ``❌`` anywhere in the message.
    """
    return "Transfer complete:" in planner_msg and "❌" not in planner_msg


def get_pg_connection(creds):
    """Raw psycopg2 connection for destination read-back / truncate."""
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required but is not installed.")
    return psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        dbname=creds["database"],
        sslmode=creds.get("sslmode", "prefer"),
    )


# ==========================================
# SINGLE TRANSFER:  register → planner.initiate_transfer → deterministic verify
# ==========================================
def run_single_transfer(test):
    test_name = test["name"]
    prompt = test["prompt"]
    dest_table = test["dest_table"]

    print(f"\n{'=' * 80}")
    print(f"🚀 STARTING TEST: {test_name}   [{test.get('category', '')}]")
    print(f"{'=' * 80}")
    print(f"Prompt            : {prompt}")
    print(f"Destination Table : {dest_table}")

    dest_pg = {**BASE_DEST_CREDS, "table": dest_table}

    # 0. Pre-run cleanup: truncate destination for a clean, deterministic run.
    try:
        conn = get_pg_connection(dest_pg)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {dest_table};")
        conn.close()
        print("🧹 Truncated destination table for a clean test run.")
    except Exception as e:
        print(f"⚠️ Could not truncate destination table `{dest_table}`: {e}")
        return "FAIL (Setup: dest table missing?)"

    # 1. Build a planner conversation and register source + destination.
    #    IMPORTANT: register the destination WITH its table baked in. The deployed
    #    planner reads the destination table from the registered connection's
    #    `table` field, so a table-less registration makes it emit `INSERT INTO
    #    "None"`. Setting it here works across planner versions (the `dest_table=`
    #    param below covers the newer planner that reads it from params instead).
    state = _fresh_state()
    _register_db(state, SOURCE_ALIAS, SOURCE_CREDS)
    _register_db(state, DEST_ALIAS, {**BASE_DEST_CREDS, "table": dest_table})

    # 2. Drive the transfer THROUGH THE PLANNER — this resolves creds, generates
    #    the code AND executes the runner (append, mirroring the reference).
    print("⚙️ Planner: initiate_transfer (code-gen + schema check + execution)...")
    params = InitiateTransferParams(
        source_alias=SOURCE_ALIAS,
        destination_alias=DEST_ALIAS,
        user_prompt=prompt,
        write_mode="append",
        dest_table=dest_table,  # named table wins over any registered default
    )
    try:
        planner_msg = _handle_initiate_transfer(state, params)
    except Exception as e:
        print(f"❌ [{test_name}] PLANNER CRASHED: {e}")
        return "ERROR (Planner)"

    print("\n----- PLANNER RESPONSE -----")
    print(planner_msg)
    print("----------------------------")

    if not _transfer_succeeded(planner_msg):
        low = planner_msg.lower()
        if "not registered" in low:
            return "FAIL (Registration)"
        if "which table" in low or "does not exist" in low:
            return "FAIL (Dest Table Setup)"
        if "schema mismatch" in low:
            return "FAIL (Schema)"
        if "setup failed" in low or "system error" in low:
            return "FAIL (Code Gen / Setup)"
        if "failed during execution" in low or "execution crashed" in low:
            return "FAIL (Execution)"
        return "FAIL (Unknown)"

    # 3. DETERMINISTIC DATA VERIFICATION against the destination table.
    print(f"\n🔍 Verifying deterministically against `{dest_table}`...")
    try:
        conn = get_pg_connection(dest_pg)
        df = pd.read_sql(f"SELECT * FROM {dest_table}", conn)
        conn.close()

        if df.empty:
            print(f"❌ [{test_name}] VERIFICATION FAILED: destination table is empty.")
            return "FAIL (Empty Data)"

        if test["verify"](df):
            print(f"🎉 [{test_name}] SUCCESS! Logic verified deterministically.")
            return "PASS"
        print(f"❌ [{test_name}] VERIFICATION FAILED: ran but produced incorrect output.")
        return "FAIL (Logic Incorrect)"
    except Exception as e:
        print(f"❌ [{test_name}] VERIFICATION CRASHED: {e}")
        return "ERROR (Verification)"


# ==========================================
# MAIN RUNNER
# ==========================================
def run_sequential_tests():
    log_file = setup_logging()
    print(f"📁 Detailed pipeline logs for all tests are being saved to: {log_file}")
    print(f"🧭 D2D via PLANNER  |  EXECUTION_ENV = {os.environ.get('EXECUTION_ENV')}")

    if not os.environ.get("GROQ_API_KEY_CODING_AGENT"):
        print(
            "\n❌ GROQ_API_KEY_CODING_AGENT is not set — the coder LLM will be None "
            "and every transfer fails with 'NoneType object has no attribute invoke'.\n"
            "   On Linux use `export GROQ_API_KEY_CODING_AGENT=...` (NOT `set`), then re-run.\n"
        )
        return

    results = []
    for idx, test in enumerate(TEST_SUITE, 1):
        print(f"\n\n{'#' * 80}")
        print(f"EXECUTING TRANSFER {idx} OF {len(TEST_SUITE)}   —  [{test.get('category', '')}]")
        print(f"#{'#' * 79}")
        status = run_single_transfer(test)
        results.append({"name": test["name"], "category": test.get("category", ""), "status": status})

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" 📊 DTA END-TO-END BENCHMARK REPORT  (D2D via PLANNER)")
    print("=" * 70)
    report_df = pd.DataFrame(results)
    passed = len(report_df[report_df["status"] == "PASS"])
    print(f"Total Tests: {len(report_df)} | Passed: {passed} | Failed: {len(report_df) - passed}")
    print(report_df.to_string(index=False))
    print("=" * 70)

    # Per-category summary — handy when scanning 100+ cases.
    print("\nPer-category:")
    for cat, grp in report_df.groupby("category"):
        p = len(grp[grp["status"] == "PASS"])
        print(f"  {cat:<24} {p}/{len(grp)} passed")
    print("=" * 70)


if __name__ == "__main__":
    run_sequential_tests()
