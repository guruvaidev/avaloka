"""End-to-end regression suite for the Healthcare Dataset prompts.

Same shape as ``tests/test_global_youtube_analysis_statistics.py`` -- one
dataset, several tiers of prompt, so that a schema change breaks every tier
together instead of leaving one of them asserting against data nobody uses:

  SINGLE    short conversational asks with no column names and no output shape
            ("what does this dataset talk about ?", "show me the patients older
            than 60"). These exercise what the planner can INFER, and they
            produce the two outcomes the MULTI tier never does: a prose answer,
            and a refusal.

  MULTI     the ten reported analysis prompts -- several clauses, explicit
            columns, an explicit output shape. These exercise instruction-
            FOLLOWING.

  OPERATIONS the transformation families benchmarked in
            tests/test_dta_end_to_end.py -- column ops, null handling, type
            casting, string manipulation, date/time, arithmetic, filtering,
            masking, UDFs, aggregation -- plus joins, asked here as ANALYSIS
            prompts. Each carries a `verify` that inspects the result table,
            because "code ran and rows came back" passes just as happily when
            the agent ignores the instruction and returns the input unchanged.
            Every verify below was checked to REJECT the untransformed frame,
            and test_every_operation_verify_rejects_the_untransformed_dataset
            keeps it that way.

Not every correct answer is a table, so each prompt declares what a correct run
looks like (``Expect.TABLE`` / ``ANSWER`` / ``REFUSAL``) and only the matching
assertions run against it. A suite that demands a result table
from every prompt fails the agent for declining to invent one, which is the
behaviour it should be protecting.

Each prompt is driven through the *full* graph starting at the planner
(``ready_to_code=False`` / ``ready_to_summarize=False``), so the run covers
plan_etl -> summarize_etl -> generate_planner_graph -> code_etl -> execute ->
visualize. That is deliberate: skipping straight to the coder (as
``tests/test_e2e_kaggle.py`` does) hides planner-side regressions.

Each prompt is executed exactly once; the resulting final state is cached and
shared by every assertion family, so the suite costs one graph run per prompt
rather than one per test.

Running::

    dev-env.bat                       # activates venv + sets GROQ_* keys
    set CHROMA_HOST=127.0.0.1         # see the Layer 2 note below
    set CHROMA_PORT=59999
    pytest tests/test_health_care_dataset.py -v

    pytest tests/test_health_care_dataset.py -m multi -v      # one tier

Layer 2 note: ``retrieve_memory`` abandons its worker thread after a 3s circuit
breaker, and the orphan then calls into ChromaDB's Rust bindings while the graph
has moved on. On Windows that reliably takes the whole interpreter down with an
access violation (0xC0000005) in ``chromadb/api/rust.py``, mid-suite. Pointing
CHROMA_HOST at a dead port forces Layer 2 onto its supported in-process fallback
and keeps the run alive. Remove the override once the orphaned-thread teardown
is fixed.

Dataset location resolution order:
  1. ``HEALTHCARE_CSV`` environment variable
  2. ``$AVALOKA_DATA_DIR/healthcare_dataset.csv`` (defaults to ~/Downloads)
  3. ``app/sample_data/healthcare_dataset.csv``

The fixture is 55,500 rows x 15 columns and has NO nulls anywhere, which shapes
several checks below -- an imputation prompt would be a no-op no verify could
tell from success, so the null-handling cases assert on output SHAPE instead.
Its real defects are the ones the prompts are aimed at: 534 exactly duplicated
rows, 108 negative Billing Amounts, 15-decimal-place billing floats, patient
names in scrambled case ("Bobby JacksOn" -- 55,467 of 55,500 rows), and
free-text Hospital values with trailing commas that make 39,876 "distinct"
hospitals out of 55,500 admissions.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from app.api.workflow import build_graph
from app.graph.etl_state import ETLState


def _downloads_dir() -> Path:
    """Where a contributor keeps downloaded datasets.

    This used to be one developer's home directory, hardcoded. That published a
    username and, more to the point, made the test unrunnable by anyone else:
    it looked for a path that exists on exactly one machine.
    """
    import os

    return Path(os.getenv("AVALOKA_DATA_DIR", Path.home() / "Downloads")).expanduser()




pytestmark = [pytest.mark.integration, pytest.mark.slow]


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).parent / "health_care_dataset.log"

# Append, do not truncate. basicConfig runs at IMPORT, so filemode="w" would mean
# that anything importing this module -- `--collect-only`, an IDE test discovery
# pass, a `-k` run of two prompts -- silently destroys the log of the run you
# were trying to diagnose. A run boundary is written below instead, which is all
# the separation a grep actually needs.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filename=LOG_FILE,
    filemode="a",
    force=True,
)
LOGGER = logging.getLogger(__name__)
LOGGER.info("=" * 30 + " NEW SESSION " + "=" * 30)

# Number of real data rows handed to the planner/coder as the uploaded preview.
PREVIEW_ROWS = 20
RECURSION_LIMIT = 60


# ---------------------------------------------------------------------------
# Prompt catalogue
# ---------------------------------------------------------------------------


class Expect(str, Enum):
    """What a correct run looks like for this prompt.

    Not every good answer is a table. Asking "what does this dataset talk
    about?" should produce prose, and asking for a column that does not exist
    should produce a refusal -- asserting a result table for either would fail
    the suite for behaving correctly.
    """

    TABLE = "table"                # runs code and returns rows
    ANSWER = "answer"              # answers in prose; no result table expected
    REFUSAL = "refusal"  # must decline, and must not invent the field


class Tier(str, Enum):
    """How much the prompt spells out, and what it is aimed at."""

    SINGLE = "single"
    MULTI = "multi"
    OPERATION = "operations"


@dataclass(frozen=True)
class AnalysisPrompt:
    """One reported prompt, with a stable id used for the pytest node name."""

    id: str
    title: str
    prompt: str
    tier: Tier = Tier.MULTI
    expect: Expect = Expect.TABLE
    #: For REFUSAL: the field the prompt asks for that the dataset does not have.
    #: The reply must name it, so a generic "I can't do that" does not pass.
    missing_field: str = ""
    #: For OPERATION: which transformation family this exercises, so a failure
    #: report says "string manipulation is broken" rather than naming one prompt.
    category: str = ""
    #: For OPERATION: a deterministic check on the result table. "The pipeline
    #: ran and produced rows" is not evidence the operation was performed --
    #: this is what distinguishes a rename that happened from one that did not.
    verify: Optional[Callable[[Any], bool]] = None
    #: A confirmed product defect this prompt trips. Set it rather than reword
    #: the prompt: rewording makes the suite green by asking a different
    #: question, which is how a defect stops being tracked. See _params().

# ---------------------------------------------------------------------------
# SINGLE-prompt tier
#
# Short, conversational, no column names, no output shape. These exercise the
# planner's inference rather than its instruction-following, and they cover the
# two shapes the MULTI tier never produces: a prose answer, and a refusal.
# ---------------------------------------------------------------------------

SINGLE_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="single_describe_dataset",
        title="What is this dataset about",
        prompt="what does this dataset talk about ?",
        tier=Tier.SINGLE,
        expect=Expect.ANSWER,
    ),
    AnalysisPrompt(
        id="single_filter_elderly",
        title="Filter by an age threshold",
        prompt="show me the patients older than 60",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_how_many_hospitals",
        title="Simple count question",
        prompt="how many hospitals are in this data?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_commonest_condition",
        title="Superlative with no metric named",
        prompt="which medical condition is the most common?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_average_bill",
        title="Aggregate stated in plain words",
        prompt="what is the average billing amount per patient?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_emergency_only",
        title="Filter by a value, not a column",
        prompt="show only the emergency admissions",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_biggest_bills",
        title="Top-N, unqualified",
        prompt="show me the top 10 biggest bills",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_patient_satisfaction",
        title="Asks for a field that does not exist",
        prompt=(
            "Analyse patient satisfaction by hospital. Return a summary DataFrame "
            "with the columns Hospital, Avg_Patient_Satisfaction_Score and "
            "Patient_count, sorted by Avg_Patient_Satisfaction_Score descending."
        ),
        tier=Tier.SINGLE,
        # Nothing in the schema measures satisfaction and nothing derives it --
        # Test Results is a lab outcome, not a rating. Declining and naming the
        # field is the correct outcome. Inventing a proxy would hand a hospital
        # administrator confident satisfaction rankings with nothing behind them.
        expect=Expect.REFUSAL,
        missing_field="Patient_Satisfaction_Score",
    ),
]


# ---------------------------------------------------------------------------
# MULTI-prompt tier
#
# The ten reported prompts, verbatim in substance: several clauses, explicit
# column names, an explicit output shape.
# ---------------------------------------------------------------------------

MULTI_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="overview_quality_check",
        title="Data overview and quality check",
        prompt=(
            "Using dataframe df with columns (Name, Age, Gender, Blood Type, "
            "Medical Condition, Date of Admission, Doctor, Hospital, Insurance "
            "Provider, Billing Amount, Room Number, Admission Type, Discharge "
            "Date, Medication, Test Results), give me: total row count, dtypes "
            "for each column, number of missing values per column, and basic "
            "summary stats for Age and Billing Amount. Then briefly describe any "
            "obvious data-quality issues you see."
        ),
    ),
    AnalysisPrompt(
        id="patient_demographics",
        title="Patient demographics",
        prompt=(
            "From df, analyze patient demographics: the distribution of Age (min, "
            "max, mean, median, standard deviation), counts and percentages by "
            "Gender, and counts and percentages by Blood Type. Provide a short "
            "narrative describing the typical patient profile."
        ),
    ),
    AnalysisPrompt(
        id="medical_conditions_overview",
        title="Medical conditions overview",
        prompt=(
            "Using df, group by Medical Condition and compute for each condition "
            "the number of patients, the average Age and the average Billing "
            "Amount. Sort conditions by number of patients and highlight the top 5 "
            "most common conditions with a short interpretation."
        ),
    ),
    AnalysisPrompt(
        id="admission_type_patterns",
        title="Admission type patterns",
        prompt=(
            "In df, analyze Admission Type (Emergency, Urgent, Elective): overall "
            "counts and percentages for each Admission Type, and for each Medical "
            "Condition show the distribution of Admission Type. Point out which "
            "conditions are most frequently admitted as Emergency versus Elective."
        ),
    ),
    AnalysisPrompt(
        id="billing_by_insurance",
        title="Billing and insurance insights",
        prompt=(
            "Using df, analyze Billing Amount by Insurance Provider: total Billing "
            "Amount per provider, average Billing Amount per provider, and number "
            "of patients per provider. Rank providers by total Billing Amount and "
            "comment on any providers with unusually high average bills."
        ),
    ),
    AnalysisPrompt(
        id="doctor_hospital_workload",
        title="Doctor and hospital workload",
        # Length_of_Stay is NOT a column in this dataset -- it is the field the
        # length_of_stay prompt above derives, and each prompt runs in its own
        # session, so nothing carries over. This is deliberately a TABLE and not a
        # REFUSAL: unlike Patient_Satisfaction_Score it is fully derivable from two
        # columns that ARE in the schema, so re-deriving it is the correct answer
        # and declining would be the regression.
        prompt=(
            "From df, calculate the number of patients, average age, and average Billing Amount" 
            "separately for each Doctor and for each Hospital. Identify the top 5 Doctors"
            "by patient count and the top 5 Hospitals by total Billing Amount,"
            "and provide a short comparison of the results."
        ),
    ),
    AnalysisPrompt(
        id="blood_group_relationships",
        title="Blood group relationships",
        prompt=(
            "Using df, investigate how Blood Type relates to other variables: "
            "cross-tabulate Blood Type by Gender (counts and row-wise percentages), "
            "cross-tabulate Blood Type by Medical Condition for the top 5 "
            "conditions, and compare average Billing Amount across Blood Types. "
            "Describe any notable patterns you see."
        ),
    ),
    AnalysisPrompt(
        id="medication_vs_test_results",
        title="Medication vs test results",
        prompt=(
            "From df, analyze the relationship between Medication and Test Results: "
            "for each Medication show counts and percentages of Test Results "
            "(Normal, Abnormal, Inconclusive), and highlight medications with a "
            "relatively high proportion of Abnormal results. Provide a short "
            "summary of which medications appear most associated with abnormal "
            "outcomes."
        ),
    ),
    AnalysisPrompt(
        id="monthly_admission_trends",
        title="Time trends in admissions and billing",
        prompt=(
            "Using Date of Admission in df, create a monthly summary showing the "
            "number of admissions per month, the total Billing Amount per month, "
            "and the most common Medical Condition each month. Describe any visible "
            "trends or seasonality in admissions or billing over time."
        ),
    ),
]


# ---------------------------------------------------------------------------
# OPERATIONS tier
#
# The transformation families from tests/test_dta_end_to_end.py -- column ops,
# null handling, type casting, string manipulation, date/time, arithmetic,
# filtering, masking, UDFs, aggregation -- plus joins, asked here as ANALYSIS
# prompts against the healthcare schema.
#
# Every case carries a `verify` that inspects the result table. Without one the
# suite only asserts "some code ran and some rows came back", which passes just
# as happily when the agent ignores the instruction and returns the input
# unchanged. The checks are deliberately tolerant about naming -- the agent
# chooses its own output column names, and failing a correct answer for calling a
# column `avg_bill` instead of `Avg_Billing_Amount` would make the suite noise.
# ---------------------------------------------------------------------------


def _norm(name: Any) -> str:
    """Column names, comparable across the agent's naming choices."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _has_col(df, *fragments: str) -> bool:
    """True when some column name contains every fragment (normalised)."""
    cols = [_norm(c) for c in df.columns]
    return any(all(_norm(f) in col for f in fragments) for col in cols)


def _col(df, *fragments: str):
    """The first column whose name contains every fragment, else None."""
    for c in df.columns:
        if all(_norm(f) in _norm(c) for f in fragments):
            return df[c]
    return None


def _no_nulls(df, *fragments: str) -> bool:
    series = _col(df, *fragments)
    return series is not None and not series.isna().any()


def _numeric(df, *fragments: str):
    """The named column coerced to numbers, or None when it is absent/unusable."""
    series = _col(df, *fragments)
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return values if not values.empty else None


def _text(df, *fragments: str):
    """The named column as non-null strings, or None when it is absent/empty."""
    series = _col(df, *fragments)
    if series is None:
        return None
    values = series.dropna().astype(str)
    return values if not values.empty else None


OPERATION_PROMPTS: List[AnalysisPrompt] = [
    # --- Column Operations -------------------------------------------------
    AnalysisPrompt(
        id="op_col_rename",
        title="Rename a column",
        prompt="Rename the Billing Amount column to bill_total and return the table.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: _has_col(df, "billtotal") and not _has_col(df, "billingamount"),
    ),
    AnalysisPrompt(
        id="op_col_select_subset",
        title="Select a subset of columns",
        prompt="Return only the Name, Hospital and Billing Amount columns.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: len(df.columns) == 3 and _has_col(df, "hospital"),
    ),
    AnalysisPrompt(
        id="op_col_drop",
        title="Drop columns",
        prompt="Drop the Room Number and Doctor columns and return everything else.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: not _has_col(df, "roomnumber") and not _has_col(df, "doctor"),
    ),

    # --- Null Handling ------------------------------------------------------
    #
    # This fixture has no nulls at all, so "fill the missing X" is a no-op that
    # every verify would pass whether or not the agent did anything. Both cases
    # below therefore assert on output SHAPE, which the untransformed 15-column
    # frame fails outright.
    AnalysisPrompt(
        id="op_null_count_per_column",
        title="Count nulls per column",
        prompt=(
            "Count the missing values in every column of df and return a two-column "
            "table of column name and missing count."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        # 15 columns in, so a correct answer is ~15 rows by 2 columns. The input
        # frame is 55,500 x 15 and fails both bounds.
        verify=lambda df: len(df.columns) <= 3 and 1 < len(df) <= 20,
    ),
    AnalysisPrompt(
        id="op_null_fill_unknown",
        title="Fill nulls with a placeholder",
        prompt=(
            "Replace any missing Test Results with the text 'Unknown' and return "
            "Name and Test Results."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        # The width bound is the half that does the rejecting here -- see the note
        # on test_every_operation_verify_rejects_the_untransformed_dataset.
        verify=lambda df: _no_nulls(df, "testresults") and len(df.columns) <= 3,
    ),

    # --- Type Casting -------------------------------------------------------
    AnalysisPrompt(
        id="op_cast_int",
        title="Cast a float column to integer",
        prompt=(
            "Convert Billing Amount to a whole number (integer) and return Name and "
            "Billing Amount."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        # Survives the CSV round-trip: pandas re-infers int64 only if every value
        # written was integral, which is exactly the property being asserted.
        verify=lambda df: (
            (v := _col(df, "billing")) is not None
            and str(v.dtype).startswith(("int", "uint"))
        ),
    ),
    AnalysisPrompt(
        id="op_cast_string",
        title="Cast a numeric column to text",
        # A dtype check cannot work here: the result is written to CSV and read
        # back, and CSV carries no type information -- "328" written as text comes
        # back as int64 no matter what the agent did. Asking for a format that is
        # not numeric is what survives the round-trip.
        prompt=(
            "Convert Room Number to text formatted as 'Room 328', and return Name "
            "and Room Number."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        verify=lambda df: (
            (v := _text(df, "room")) is not None
            and v.str.strip().str.lower().str.startswith("room").mean() > 0.8
        ),
    ),

    # --- String Manipulation ------------------------------------------------
    AnalysisPrompt(
        id="op_str_title_case_names",
        title="Normalise scrambled name casing",
        # The dataset's headline text defect: 55,467 of 55,500 names are not in
        # title case ("Bobby JacksOn", "LesLie TErRy"), so an untouched column
        # fails this by a wide margin.
        prompt=(
            "The Name column has inconsistent capitalisation, for example "
            "'Bobby JacksOn'. Convert every Name to proper title case and return "
            "Name and Hospital."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _text(df, "name")) is not None and (v == v.str.title()).all()
        ),
    ),
    AnalysisPrompt(
        id="op_str_upper",
        title="Uppercase a string column",
        prompt=(
            "Convert every Medical Condition to uppercase and return Name and "
            "Medical Condition."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _text(df, "condition")) is not None and (v == v.str.upper()).all()
        ),
    ),
    AnalysisPrompt(
        id="op_str_last_name",
        title="Derive a column from a string split",
        prompt=(
            "Add a column called last_name holding the surname from Name, and "
            "return Name and last_name."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: _has_col(df, "lastname"),
    ),
    AnalysisPrompt(
        id="op_str_contains_filter",
        title="Filter by a substring",
        # "Ltd" matches 3,126 of 55,500 Hospital values. "Clinic" matches none --
        # these are generated company-style names ("Sons and Miller", "Kim Inc") --
        # and a filter that can only return an empty table would fail this tier for
        # a correct answer.
        prompt="Return only the patients whose Hospital name contains 'Ltd'.",
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _text(df, "hospital")) is not None
            and v.str.contains("ltd", case=False).all()
        ),
    ),

    # --- Date & Time --------------------------------------------------------
    AnalysisPrompt(
        id="op_date_length_of_stay",
        title="Derive a duration from two dates",
        prompt=(
            "Convert Date of Admission and Discharge Date to datetime, add a column "
            "called Length_of_Stay holding the number of days between them, and "
            "return Name, Date of Admission, Discharge Date and Length_of_Stay."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        # Real stays run 1-30 days. The bounds catch the reversed subtraction
        # (every value negative), which a mere column-presence check would pass.
        verify=lambda df: (
            (v := _numeric(df, "lengthofstay")) is not None and v.between(0, 400).all()
        ),
    ),
    AnalysisPrompt(
        id="op_date_month_bucket",
        title="Bucket a date into months",
        prompt=(
            "Add a column called admission_month holding the year and month of Date "
            "of Admission formatted as YYYY-MM, and return Name and admission_month."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: (
            (v := _text(df, "admissionmonth")) is not None
            and v.str.match(r"^\d{4}-\d{2}").mean() > 0.8
        ),
    ),
    AnalysisPrompt(
        id="op_date_filter_year",
        title="Filter on a year",
        # Admissions span 2019-05-08 to 2024-05-07; 14,880 rows are 2023 or later,
        # so the filter both bites and returns something.
        prompt="Return only the patients admitted in or after 2023.",
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: (
            (v := _col(df, "admission")) is not None
            and (years := pd.to_datetime(v, errors="coerce").dt.year.dropna()).size > 0
            and (years >= 2023).all()
        ),
    ),

    # --- Arithmetic ---------------------------------------------------------
    AnalysisPrompt(
        id="op_arith_billing_per_day",
        title="Derive a rate column",
        prompt=(
            "Add a column called billing_per_day equal to Billing Amount divided by "
            "the number of days between Date of Admission and Discharge Date, and "
            "return Name, Billing Amount and billing_per_day."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        verify=lambda df: _has_col(df, "billingperday") or _has_col(df, "billperday"),
    ),
    AnalysisPrompt(
        id="op_arith_round_billing",
        title="Round a float column",
        # Not one of the 55,500 billing values is already at 2dp -- they carry 15
        # decimal places -- so an untouched column fails this outright.
        prompt=(
            "Round Billing Amount to 2 decimal places and return Name and Billing "
            "Amount."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        verify=lambda df: (
            (v := _numeric(df, "billing")) is not None and (v.round(2) == v).all()
        ),
    ),

    # --- Filtering Logic ----------------------------------------------------
    AnalysisPrompt(
        id="op_filter_threshold",
        title="Numeric threshold",
        prompt="Return the patients older than 70.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (v := _numeric(df, "age")) is not None and (v > 70).all(),
    ),
    AnalysisPrompt(
        id="op_filter_compound",
        title="Two conditions combined",
        # Admission Type is a filter here, not a requested output column, so it is
        # only checked when the agent happens to return it -- requiring it would
        # fail a correct answer that returned Name and Age.
        prompt="Return the Emergency admissions of patients older than 60.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (a := _numeric(df, "age")) is not None
            and (a > 60).all()
            and ((t := _text(df, "admissiontype")) is None
                 or t.str.contains("emergency", case=False).all())
        ),
    ),
    AnalysisPrompt(
        id="op_filter_isin",
        title="Membership filter",
        prompt="Return only the patients whose Blood Type is O+, O- or AB+.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (v := _text(df, "bloodtype")) is not None
            and v.str.strip().str.upper().isin({"O+", "O-", "AB+"}).all()
        ),
    ),

    # --- Data Masking -------------------------------------------------------
    #
    # The family that matters most on this schema: every row pairs a patient name
    # with a diagnosis.
    AnalysisPrompt(
        id="op_mask_patient_name",
        title="Mask a text column",
        prompt=(
            "Mask the patient names so only the first two characters are visible and "
            "the rest are replaced with asterisks. Return Name, Medical Condition "
            "and Billing Amount."
        ),
        tier=Tier.OPERATION, category="Data Masking",
        # No name in the fixture contains an asterisk, so any hit is the agent's
        # work. Masking applies to every row, so require the overwhelming majority
        # -- not all, since a one or two character name has nothing left to mask.
        verify=lambda df: (
            (v := _text(df, "name")) is not None and v.str.contains(r"\*").mean() > 0.8
        ),
    ),
    AnalysisPrompt(
        id="op_mask_deidentify",
        title="Drop the direct identifiers",
        prompt=(
            "Return a de-identified version of the table with Name and Doctor "
            "removed, keeping Age, Gender, Medical Condition, Admission Type and "
            "Billing Amount."
        ),
        tier=Tier.OPERATION, category="Data Masking",
        verify=lambda df: (
            not _has_col(df, "name")
            and not _has_col(df, "doctor")
            and _has_col(df, "condition")
        ),
    ),

    # --- UDF ----------------------------------------------------------------
    AnalysisPrompt(
        id="op_udf_age_band",
        title="Custom bucketing function",
        prompt=(
            "Classify each patient as Senior if Age is over 65, Adult if over 18, "
            "otherwise Minor. Put the label in a column called age_band and return "
            "Name, Age and age_band."
        ),
        tier=Tier.OPERATION, category="UDF",
        verify=lambda df: (
            (v := _text(df, "ageband")) is not None
            and set(v.str.strip().str.lower().unique()) <= {"senior", "adult", "minor"}
        ),
    ),

    # --- Aggregation --------------------------------------------------------
    AnalysisPrompt(
        id="op_agg_condition_metrics",
        title="Several aggregates in one group-by",
        prompt=(
            "For each Medical Condition return the number of patients, the average "
            "Age and the average Billing Amount."
        ),
        tier=Tier.OPERATION, category="Aggregation",
        # Exactly 6 distinct conditions against 55,500 input rows.
        verify=lambda df: (
            _has_col(df, "condition") and 1 < len(df) <= 10 and len(df.columns) >= 4
        ),
    ),
    AnalysisPrompt(
        id="op_agg_insurance_totals",
        title="Group-by with a total",
        prompt=(
            "For each Insurance Provider return the total Billing Amount and the "
            "number of patients."
        ),
        tier=Tier.OPERATION, category="Aggregation",
        # Exactly 5 distinct providers.
        verify=lambda df: (
            _has_col(df, "insurance") and 1 < len(df) <= 10 and len(df.columns) >= 3
        ),
    ),

    # --- Joins --------------------------------------------------------------
    AnalysisPrompt(
        id="op_join_hospital_totals",
        title="Join an aggregate back onto the rows",
        prompt=(
            "Compute the total Billing Amount per Hospital, then join that total "
            "back onto every patient row. Return Name, Hospital, Billing Amount and "
            "the hospital total in a column called hospital_total_billing."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "hospitaltotal") and _has_col(df, "name"),
    ),
    AnalysisPrompt(
        id="op_join_share_of_condition",
        title="Join then derive a share",
        prompt=(
            "For each patient work out what share of their Medical Condition's total "
            "Billing Amount they account for. Return Name, Medical Condition and a "
            "column called share_of_condition_billing."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "share"),
    ),
    AnalysisPrompt(
        id="op_join_rank_in_condition",
        title="Join a group rank back onto the rows",
        # Ranked within Medical Condition rather than within Hospital: there are
        # 39,876 distinct hospital names across 55,500 admissions, so a per-hospital
        # rank is almost always 1 and proves nothing.
        prompt=(
            "Rank patients by Billing Amount within their own Medical Condition, "
            "join that rank back onto each row, and return Name, Medical Condition, "
            "Billing Amount and a column called rank_in_condition."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "rankincondition"),
    ),
]


#: Everything, in the order the suite reports it.
ANALYSIS_PROMPTS: List[AnalysisPrompt] = (
    SINGLE_PROMPTS + MULTI_PROMPTS + OPERATION_PROMPTS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate_dataset_paths() -> List[Path]:
    paths = []
    env_path = os.environ.get("HEALTHCARE_CSV")
    if env_path:
        paths.append(Path(env_path))
    paths.append(_downloads_dir() / "healthcare_dataset.csv")
    paths.append(REPO_ROOT / "app" / "sample_data" / "healthcare_dataset.csv")
    return paths


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    for path in _candidate_dataset_paths():
        if path.exists():
            LOGGER.info("Using dataset: %s", path)
            return path
    pytest.skip(
        "healthcare_dataset.csv not found. Set HEALTHCARE_CSV or place the file in "
        "app/sample_data/."
    )


@pytest.fixture(scope="session")
def llm_keys_present() -> None:
    """The planner and coder both need their own Groq key; skip loudly without them."""
    missing = [
        name
        for name in ("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT")
        if not os.environ.get(name)
    ]
    if missing:
        pytest.skip(
            f"Missing LLM credentials: {', '.join(missing)}. Run dev-env.bat (or export "
            "the keys) before running this suite — without them the planner is disabled "
            "and every prompt fails for the wrong reason."
        )


@pytest.fixture(scope="session")
def dataset_frame(dataset_path: Path) -> pd.DataFrame:
    """Head of the CSV, used to derive the schema and the uploaded preview."""
    return pd.read_csv(dataset_path, nrows=PREVIEW_ROWS, encoding_errors="replace")


@pytest.fixture(scope="session")
def dataset_schema(dataset_frame: pd.DataFrame) -> Dict[str, str]:
    return {str(col): str(dtype) for col, dtype in dataset_frame.dtypes.items()}


@pytest.fixture(scope="session")
def uploaded_preview(dataset_frame: pd.DataFrame) -> List[List[Any]]:
    """Header row followed by real value rows (the list[list] preview shape)."""
    header = [str(col) for col in dataset_frame.columns]
    rows = dataset_frame.astype(object).where(pd.notna(dataset_frame), "").values.tolist()
    return [header] + rows


@pytest.fixture(scope="session")
def compiled_graph(llm_keys_present):
    return build_graph().compile()


# ---------------------------------------------------------------------------
# Graph runner (one run per prompt, cached)
# ---------------------------------------------------------------------------

_RUN_CACHE: Dict[str, Dict[str, Any]] = {}


def _run_prompt(
    entry: AnalysisPrompt,
    compiled_graph,
    dataset_path: Path,
    dataset_schema: Dict[str, str],
    uploaded_preview: List[List[Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Invoke the full graph for one prompt, starting at the planner."""
    if entry.id in _RUN_CACHE:
        return _RUN_CACHE[entry.id]

    output_location = output_dir / f"{entry.id}.csv"
    initial_state = ETLState(
        user_id="test_healthcare_analysis_user",
        session_id=f"test_healthcare_analysis_{entry.id}",
        user_prompt=entry.prompt,
        messages=[HumanMessage(content=entry.prompt)],
        plan="",
        # Planner-first: these two flags must stay False so plan_etl actually runs.
        ready_to_summarize=False,
        ready_to_code=False,
        data_source_location=str(dataset_path),
        output_location=str(output_location),
        schema=dataset_schema,
        input_data_type="csv",
        uploaded_csv_columns=list(uploaded_preview[0]),
        uploaded_csv_preview=uploaded_preview,
        # Keep execution local; the cloud/Ray path needs a connection_id.
        analysis_fidelity="quick_sample",
        execution_mode="local",
        dataset_id=f"ds_healthcare_{entry.id}",
        infrastructure_request=None,
        infrastructure_provisioned=None,
    )

    LOGGER.info("[%s] START — %s", entry.id, entry.prompt)
    try:
        final_state = compiled_graph.invoke(
            initial_state, config={"recursion_limit": RECURSION_LIMIT}
        )
        error: Optional[BaseException] = None
    except Exception as exc:  # noqa: BLE001 — recorded and re-asserted per stage
        LOGGER.exception("[%s] graph raised", entry.id)
        final_state, error = {}, exc

    result = {
        "entry": entry,
        "state": final_state,
        "error": error,
        "output_location": output_location,
    }
    _RUN_CACHE[entry.id] = result
    LOGGER.info("[%s] END — stages: %s", entry.id, _stage_report(result))
    return result


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("healthcare_analysis_outputs")


@pytest.fixture
def prompt_run(
    request,
    compiled_graph,
    dataset_path,
    dataset_schema,
    uploaded_preview,
    output_dir,
):
    """Cached graph run for the AnalysisPrompt passed as the test parameter."""
    entry: AnalysisPrompt = request.param
    return _run_prompt(
        entry, compiled_graph, dataset_path, dataset_schema, uploaded_preview, output_dir
    )


# ---------------------------------------------------------------------------
# Stage inspection
# ---------------------------------------------------------------------------


def _generated_code(state: Dict[str, Any]) -> str:
    return (
        state.get("generated_code")
        or (state.get("coder_definition") or {}).get("code")
        or ""
    )


def _result_records(state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Decode ``output_file_data`` — the data-URI CSV the visualization agent charts."""
    ofd = state.get("output_file_data")
    if not isinstance(ofd, dict) or not ofd.get("content"):
        return None
    try:
        from app.api.helpers import _datauri_csv_to_records

        return _datauri_csv_to_records(ofd["content"])
    except Exception:  # noqa: BLE001 — absence of records is handled by the caller
        return None


def _output_rows(result: Dict[str, Any]) -> Optional[int]:
    """Row count of whatever the run produced, or None if it produced nothing."""
    state = result["state"]
    records = _result_records(state)
    if records:
        return len(records)
    for key in ("execution_output_data", "execution_output_preview"):
        data = state.get(key)
        if isinstance(data, pd.DataFrame):
            return len(data)
        if isinstance(data, list) and data:
            return len(data)
    exec_result = state.get("execution_result") or {}
    for key in ("output_json", "output_data"):
        data = exec_result.get(key)
        if isinstance(data, list) and data:
            return len(data)
    path: Path = result["output_location"]
    if path.exists() and path.stat().st_size > 0:
        try:
            return len(pd.read_csv(path))
        except Exception:  # noqa: BLE001 — a malformed file is a failure, not a crash
            return None
    return None


def _result_columns(result: Dict[str, Any]) -> List[str]:
    state = result["state"]
    records = _result_records(state)
    if records:
        return [str(c) for c in records[0].keys()]
    data = state.get("execution_output_data")
    if isinstance(data, pd.DataFrame):
        return [str(c) for c in data.columns]
    exec_result = state.get("execution_result") or {}
    rows = exec_result.get("output_json")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return [str(c) for c in rows[0].keys()]
    path: Path = result["output_location"]
    if path.exists() and path.stat().st_size > 0:
        try:
            return [str(c) for c in pd.read_csv(path, nrows=0).columns]
        except Exception:  # noqa: BLE001
            return []
    return []


def _charts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    config = state.get("visualization_config") or {}
    charts = config.get("charts") if isinstance(config, dict) else None
    return charts if isinstance(charts, list) else []


def _stage_report(result: Dict[str, Any]) -> str:
    state = result["state"]
    return (
        f"raised={type(result['error']).__name__ if result['error'] else None} "
        f"plan={bool(state.get('plan'))} "
        f"code={bool(_generated_code(state))} "
        f"exec_error={state.get('execution_error')!r} "
        f"exec_status={(state.get('execution_result') or {}).get('status')!r} "
        f"rows={_output_rows(result)} "
        f"viz_status={state.get('visualization_status')!r} "
        f"charts={len(_charts(state))}"
    )


def _diagnostics(result: Dict[str, Any]) -> str:
    """Compact failure context: which stage broke, plus the code that ran."""
    state = result["state"]
    code = _generated_code(state)
    snippet = code if len(code) <= 2000 else code[:2000] + "\n… (truncated)"
    return (
        f"\nprompt: {result['entry'].prompt}"
        f"\nstages: {_stage_report(result)}"
        f"\nvisualization_status: {state.get('visualization_status')!r}"
        f"\nexecution_error: {state.get('execution_error')!r}"
        f"\nexecution_result: {str(state.get('execution_result'))[:800]!r}"
        f"\ngenerated_code:\n{snippet or '<none>'}"
        f"\nfull log: {LOG_FILE}"
    )



def _params(prompts: List[AnalysisPrompt]):
    """Parametrize over prompts, tagging each case with its tier.

    The tier marker is what makes `-m operations` work. Without it the only way to
    run one tier is a substring match on the node id, which silently depends on
    every id in that tier happening to share a prefix.
    """
    return pytest.mark.parametrize(
        "prompt_run",
        [pytest.param(p, marks=getattr(pytest.mark, p.tier.value)) for p in prompts],
        ids=[p.id for p in prompts],
        indirect=True,
    )


#: Prompts whose correct outcome is a result table. The table, chart and
#: row-count assertions apply only to these -- a prose answer has no rows to
#: count, and a refusal deliberately runs no code.
_TABLE_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.TABLE]
_ANSWER_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.ANSWER]
_REFUSAL_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.REFUSAL]
_OPERATION_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.verify is not None]

_PARAMS = _params(_TABLE_PROMPTS)
_PARAMS_ALL = _params(ANALYSIS_PROMPTS)
_PARAMS_ANSWER = _params(_ANSWER_PROMPTS)
_PARAMS_REFUSAL = _params(_REFUSAL_PROMPTS)
_PARAMS_OPERATION = _params(_OPERATION_PROMPTS)


# ---------------------------------------------------------------------------
# Tests — analysis pipeline
# ---------------------------------------------------------------------------


@_PARAMS
def test_prompt_completes_pipeline(prompt_run):
    """Planner -> coder -> local execution must produce a non-empty result table."""
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    code = _generated_code(state)
    assert code.strip(), f"No code was generated{_diagnostics(result)}"

    assert not state.get("syntax_error"), (
        f"Syntax error in generated code{_diagnostics(result)}"
    )
    assert not state.get("static_semantic_error"), (
        f"Static semantic error in generated code{_diagnostics(result)}"
    )
    assert state.get("execution_error") in (None, ""), (
        f"Execution failed{_diagnostics(result)}"
    )

    exec_status = (state.get("execution_result") or {}).get("status")
    assert exec_status not in ("error", "failed"), (
        f"Execution reported status={exec_status!r}{_diagnostics(result)}"
    )

    rows = _output_rows(result)
    assert rows, f"Run produced no output rows{_diagnostics(result)}"


@_PARAMS
def test_prompt_reaches_planner(prompt_run):
    """The planner must actually run and leave a plan behind (not be short-circuited).

    TABLE prompts only. A refusal reaches the planner and correctly leaves NO
    analysis plan -- asserting a plan for it would fail the prompt for behaving
    exactly as designed.
    """
    result = prompt_run
    state = result["state"]
    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )
    assert state.get("plan"), f"Planner produced no plan{_diagnostics(result)}"


# ---------------------------------------------------------------------------
# Tests — prose answers and refusals
#
# The MULTI tier only ever asserts "code ran, rows came back". These are the
# shapes a short conversational ask actually produces, and neither is covered by
# a result-table assertion.
# ---------------------------------------------------------------------------


def _assistant_reply(state: Dict[str, Any]) -> str:
    """The last thing the assistant said, whatever stage produced it."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) in ("ai", "assistant"):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content
    return ""


@_PARAMS_ANSWER
def test_prompt_answers_in_prose(prompt_run):
    """A question about the data must be answered, not turned into a table.

    "what does this dataset talk about?" has no rows to return. Reaching the
    coder at all would mean the planner mistook a question for a transformation.
    """
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    reply = _assistant_reply(state)
    assert reply.strip(), f"No assistant reply at all{_diagnostics(result)}"
    assert len(reply.strip()) > 40, (
        f"Reply is too short to be an answer: {reply!r}{_diagnostics(result)}"
    )

    lowered = reply.lower()
    assert "temporary problem" not in lowered, (
        f"Planner fell back to its error message{_diagnostics(result)}"
    )
    # The answer should describe THIS dataset, not any dataset.
    assert any(
        term in lowered
        for term in ("patient", "hospital", "medical", "admission", "healthcare")
    ), (
        f"Reply does not mention what the data is about: {reply[:200]!r}"
        f"{_diagnostics(result)}"
    )


@_PARAMS_REFUSAL
def test_prompt_declines_and_names_the_missing_field(prompt_run):
    """Asking for a field the dataset does not have must produce a refusal that
    names the field -- and must not produce numbers.

    Inventing a plausible formula for a column nobody has is the worst possible
    outcome here: the user gets confident figures with nothing behind them. A
    generic "I can't help with that" is not much better, because it gives them no
    way to rephrase, so the reply has to name the field it could not map.
    """
    entry: AnalysisPrompt = prompt_run["entry"]
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    reply = _assistant_reply(state)
    assert reply.strip(), f"Refusal produced no reply at all{_diagnostics(result)}"

    # Compare on alphanumerics only. The prompt writes the field as
    # `Patient_Satisfaction_Score`; a good reply may well say "patient
    # satisfaction score", and failing that answer over an underscore would make
    # this assertion about formatting rather than about naming the field.
    assert _norm(entry.missing_field) in _norm(reply), (
        f"Reply does not name the missing field {entry.missing_field!r}, so the "
        f"user cannot tell what to rephrase: {reply[:300]!r}{_diagnostics(result)}"
    )

    # No fabricated result. A refusal that still emits rows has invented them.
    rows = _output_rows(result)
    assert not rows, (
        f"Declined the request but still produced {rows} rows of output"
        f"{_diagnostics(result)}"
    )


@_PARAMS_ALL
def test_prompt_never_reports_an_internal_error_to_the_user(prompt_run):
    """Whatever the outcome, the user must not be shown a stack-trace artefact or
    the planner's generic apology."""
    result = prompt_run
    reply = _assistant_reply(result["state"]).lower()
    if not reply:
        return
    # Look for a RAISED error, not the mere mention of one. The summarizer
    # legitimately discusses error handling in prose -- "defensive checks could
    # prevent a KeyError if Discharge Date were malformed" is good writing, and a
    # bare substring match would fail the turn for it. A real leak looks like a
    # traceback or "SomeError: detail", which is what these patterns require.
    leaks = [
        r"traceback \(most recent call last\)",
        r"(?:key|value|type|attribute|index|unicode\w*)error\s*:",
        r"'nonetype' object",
        r'file "[^"]+", line \d+',
        r"temporary problem while planning",
    ]
    for pattern in leaks:
        assert not re.search(pattern, reply), (
            f"Internal detail leaked into the user-visible reply "
            f"(matched {pattern!r}): {reply[:300]!r}{_diagnostics(result)}"
        )


# ---------------------------------------------------------------------------
# Tests - OPERATIONS tier
# ---------------------------------------------------------------------------


def _result_frame(result: Dict[str, Any]):
    """The result table as a DataFrame, or None when nothing was written."""
    path: Path = result["output_location"]
    if not (path.exists() and path.stat().st_size > 0):
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


@_PARAMS_OPERATION
def test_operation_was_actually_performed(prompt_run):
    """The transformation must show up in the result, not just run without error.

    Every other assertion in this file is satisfied by a pipeline that ignores the
    instruction and returns the input unchanged: code was generated, it executed,
    rows came back. Only this one can tell a rename that happened from a rename
    that did not.
    """
    entry: AnalysisPrompt = prompt_run["entry"]
    result = prompt_run

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    df = _result_frame(result)
    assert df is not None and not df.empty, (
        f"[{entry.category}] produced no result table{_diagnostics(result)}"
    )

    try:
        ok = bool(entry.verify(df))
    except Exception as exc:  # noqa: BLE001 - a raising check is a failed check
        pytest.fail(
            f"[{entry.category}] verification raised {exc!r} on columns "
            f"{list(df.columns)}{_diagnostics(result)}"
        )

    assert ok, (
        f"[{entry.category}] the operation is not visible in the result. "
        f"Columns: {list(df.columns)}\n"
        f"First row: {df.head(1).to_dict(orient='records')}"
        f"{_diagnostics(result)}"
    )


@_PARAMS_OPERATION
def test_operation_returns_more_than_a_single_cell(prompt_run):
    """A transformation over the dataset should return the dataset, not a scalar.

    Collapsing "rename this column" into a one-cell answer is a real failure mode
    and one the row-count assertion alone lets through.
    """
    entry: AnalysisPrompt = prompt_run["entry"]
    df = _result_frame(prompt_run)
    if df is None:
        pytest.skip("no result table; covered by test_operation_was_actually_performed")
    assert df.shape != (1, 1), (
        f"[{entry.category}] collapsed the whole dataset into one cell"
        f"{_diagnostics(prompt_run)}"
    )


# ---------------------------------------------------------------------------
# Suite self-checks
#
# A `verify` that accepts the untransformed input silently turns its case into a
# no-op: the prompt still runs, still costs an LLM call, and still passes when the
# agent does nothing at all. These two run without credentials, so a plain
# `pytest tests/test_health_care_dataset.py` on a laptop still checks them.
# ---------------------------------------------------------------------------


def test_every_operation_verify_rejects_the_untransformed_dataset(dataset_frame):
    """Each OPERATION check must fail against the raw fixture.

    Worth stating for `op_null_fill_unknown`: this fixture has no nulls anywhere,
    so its null-freeness clause passes on the raw frame and the width bound is the
    half that does the rejecting. Anything that PASSES here is a case that can
    never fail.
    """
    accepted = []
    for entry in _OPERATION_PROMPTS:
        try:
            if entry.verify(dataset_frame.copy()):
                accepted.append(entry.id)
        except Exception:  # noqa: BLE001 - raising on the raw frame is a rejection
            continue
    assert not accepted, (
        "These verify() checks pass against the UNTRANSFORMED dataset, so their "
        f"cases assert nothing: {accepted}"
    )


def test_prompt_ids_are_unique():
    """Duplicate ids collide in _RUN_CACHE, and one prompt silently reuses the
    other's run."""
    ids = [p.id for p in ANALYSIS_PROMPTS]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"Duplicate prompt ids: {duplicates}"


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def _sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="session", autouse=True)
def _summary():
    yield
    if not _RUN_CACHE:
        return
    LOGGER.info("=" * 78)
    LOGGER.info("SUMMARY — %d prompt(s) executed", len(_RUN_CACHE))
    for prompt_id, result in _RUN_CACHE.items():
        LOGGER.info("  %-28s %s", prompt_id, _stage_report(result))
    LOGGER.info("=" * 78)
