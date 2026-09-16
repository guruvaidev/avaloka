"""End-to-end regression suite for the Kaggle NYC Yellow Taxi Trip prompts.

Same shape as ``tests/test_health_care_dataset.py`` and
``tests/test_global_youtube_analysis_statistics.py`` -- one dataset, several
tiers of prompt, so that a schema change breaks every tier together instead of
leaving one of them asserting against data nobody uses:

  SINGLE    short conversational asks with no column names and no output shape
            ("what does this dataset talk about ?", "show the trips longer than
            20 miles"). These exercise what the planner can INFER, and they
            produce the two outcomes the MULTI tier never does: a prose answer,
            and a refusal.

  MULTI     fully specified analysis prompts -- several clauses, explicit
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
            and test_selfcheck_every_operation_verify_rejects_the_untransformed_dataset
            keeps it that way.

  TRANSFER  the same dataset moved rather than analysed, from
            gs://avaloka-test-user-filestore/test_c2c_jyothi/ to
            gs://avaloka-dta-destination/transfers. Routing runs everywhere;
            moving real bytes needs AVALOKA_DTA_LIVE=1 plus credentials.

Not every correct answer is a tableNot every correct answer is a table, so each prompt declares what a correct run
looks like (``Expect.TABLE`` / ``ANSWER`` / ``REFUSAL`` / ``TRANSFER``) and
only the matching assertions run against it. A suite that
demands a result table from every prompt fails the agent for declining to invent
one, which is the behaviour it should be protecting.

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
    pytest tests/kaggle_yellow_taxi_trip_data.py -v

    pytest tests/kaggle_yellow_taxi_trip_data.py -m multi -v      # one tier
    pytest tests/kaggle_yellow_taxi_trip_data.py -m transfer -v

The file name is the one that was asked for and it does NOT match pytest's
``test_*.py`` discovery pattern, so a bare ``pytest tests/`` will not pick it
up -- name the file on the command line as above (pytest always collects an
explicitly named file) or rename it to ``test_kaggle_yellow_taxi_trip_data.py``
to have it run with the rest of the suite.

Layer 2 note: ``retrieve_memory`` abandons its worker thread after a 3s circuit
breaker, and the orphan then calls into ChromaDB's Rust bindings while the graph
has moved on. On Windows that reliably takes the whole interpreter down with an
access violation (0xC0000005) in ``chromadb/api/rust.py``, mid-suite. Pointing
CHROMA_HOST at a dead port forces Layer 2 onto its supported in-process fallback
and keeps the run alive. Remove the override once the orphaned-thread teardown
is fixed. (tests/conftest.py already does this for you.)

Dataset location resolution order:
  1. ``YELLOW_TAXI_DATASET`` environment variable (.csv or .xlsx)
  2. the 50k basename, then the 200k one, each looked for as .xlsx then .csv in
     ``$AVALOKA_DATA_DIR/yellow_taxi_trip_data/`` (defaults to ~/Downloads) and then in
     ``app/sample_data/`` -- see DATASET_BASENAMES.

``app/sample_data/yellow_tripdata_2015-01_dataset_50k.csv`` is committed, so a
fresh clone runs this suite with no Kaggle download.

The shipped fixture is an .xlsx, and the local execution path reads the source
with ``read_csv_best_effort`` (app/agents/execution_agent.py) -- handing it a
workbook fails every prompt with a pandas parse error, which is a failure of the
harness and not of the agent. The ``dataset_path`` fixture therefore converts the
workbook to CSV once per session and points the graph at that, exactly as if the
same rows had been uploaded as a CSV.

The committed fixture is 50,426 rows x 19 columns covering January 2015 and has
NO nulls anywhere, which shapes several checks below -- an imputation prompt
would be a no-op no verify could tell from success, so the null-handling cases
assert on output SHAPE instead. Its real defects are the ones the prompts are
aimed at: 16 trips with a negative fare_amount and total_amount, a tip_amount as
low as -14.33, 297 trips with a trip_distance of exactly 0, a passenger_count of
0 on 33 trips, and RateCodeID 99 on 1 trip (the code book stops at 6).

It is regenerated from the Kaggle .xlsx rather than exported from Excel. An
Excel-exported copy was committed once and cost a full 30-minute run: it carried
49,574 trailing all-blank rows (so every numeric column read back as float64 and
op_cast_int could not produce an integer dtype) and locale DD-MM-YYYY timestamps
with the seconds dropped (so op_str_date_prefix returned "15-01-2015"). Both
looked like agent regressions and neither was. If this file is ever regenerated,
do it with pandas, not a spreadsheet.

No assertion depends on the row count, so the 200k export is interchangeable.
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




pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.kaggle]


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).parent / "kaggle_yellow_taxi_trip_data.log"

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
    REFUSAL = "refusal"            # must decline, and must not invent the field
    TRANSFER = "transfer"    # routes to the transfer agent, not to code generation


class Tier(str, Enum):
    """How much the prompt spells out."""

    SINGLE = "single"
    MULTI = "multi"
    OPERATION = "operations"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class AnalysisPrompt:
    """One prompt, with a stable id used for the pytest node name."""

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
        id="single_filter_long_trips",
        title="Filter by a distance threshold",
        prompt="show the trips longer than 20 miles",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_top_fares",
        title="Superlative with the metric implied",
        prompt="show me the 10 most expensive trips",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_how_many_trips",
        title="Simple count question",
        prompt="how many trips are in this data?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_busiest_hour",
        title="Needs a derivation the prompt never spells out",
        # There is no hour column: the planner has to get it out of
        # tpep_pickup_datetime on its own.
        prompt="which hour of the day is the busiest?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_average_tip",
        title="Aggregate stated in plain words",
        prompt="what is the average tip?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_cash_trips",
        title="Filter by a value, not a column",
        # "cash" is payment_type 2 -- a code book the prompt does not supply.
        prompt="show only the trips that were paid in cash",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_missing_column",
        title="Asks for a field that does not exist",
        # The trip records carry no driver identity and no rating of any kind.
        # The right answer is to say so and name the field, not to invent a
        # proxy out of tip_amount.
        prompt="rank the trips by the driver's rating",
        tier=Tier.SINGLE,
        expect=Expect.REFUSAL,
        missing_field="rating",
    ),
]


# ---------------------------------------------------------------------------
# MULTI-prompt tier
#
# Several clauses, explicit column names, an explicit output shape.
# ---------------------------------------------------------------------------

MULTI_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="revenue_by_payment_type",
        title="Revenue split by payment type",
        prompt=(
            "For each payment_type, compute the number of trips, the total "
            "fare_amount, the total tip_amount and the average total_amount. "
            "Return the table ranked by total revenue."
        ),
    ),
    AnalysisPrompt(
        id="tipping_behaviour",
        title="Tipping behaviour by party size",
        # Cash tips are not recorded, so restricting to payment_type 1 is the
        # difference between a real answer and an average dragged to zero by
        # 75,383 cash trips that all report a 0.00 tip.
        prompt=(
            "Using only the credit card trips (payment_type 1), express each "
            "trip's tip_amount as a percentage of its fare_amount, then return "
            "the average tip percentage for each passenger_count together with "
            "the number of trips in that group."
        ),
    ),
    AnalysisPrompt(
        id="hourly_demand_profile",
        title="Demand profile across the day",
        prompt=(
            "Derive the pickup hour from tpep_pickup_datetime and return, for "
            "each hour of the day, the number of trips, the average "
            "trip_distance and the average total_amount. Rank the hours by "
            "number of trips."
        ),
    ),
    AnalysisPrompt(
        id="distance_bands",
        title="Fare economics by distance band",
        prompt=(
            "Bucket the trips by trip_distance into the bands 0-1, 1-3, 3-10 "
            "and more than 10 miles. For each band return the number of trips, "
            "the average fare_amount, the average tip_amount and the average "
            "total_amount, in the logical band order."
        ),
    ),
    AnalysisPrompt(
        id="vendor_comparison",
        title="Vendor comparison",
        prompt=(
            "Compare the two VendorID values: for each vendor return the number "
            "of trips, the total revenue from total_amount, the average "
            "trip_distance and the share of its trips that were paid by credit "
            "card (payment_type 1)."
        ),
    ),
    AnalysisPrompt(
        id="trip_speed",
        title="Trip duration and average speed",
        prompt=(
            "Compute each trip's duration in minutes from tpep_pickup_datetime "
            "and tpep_dropoff_datetime, then its average speed in miles per "
            "hour using trip_distance. Excluding trips with zero duration or "
            "zero distance, return the 20 fastest trips with their pickup time, "
            "dropoff time, trip_distance, duration and speed."
        ),
    ),
    AnalysisPrompt(
        id="airport_trips",
        title="Airport trips against the standard rate",
        prompt=(
            "RateCodeID 2 is a JFK trip and RateCodeID 3 is a Newark trip. "
            "Compare airport trips against standard rate trips (RateCodeID 1) "
            "on number of trips, average trip_distance, average fare_amount, "
            "average tolls_amount and average total_amount."
        ),
    ),
    AnalysisPrompt(
        id="surcharge_composition",
        title="What total_amount is made of",
        prompt=(
            "For every trip work out what share of total_amount comes from "
            "fare_amount, extra, mta_tax, tip_amount, tolls_amount and "
            "improvement_surcharge. Return the average share of each of those "
            "six components across all trips."
        ),
    ),
    AnalysisPrompt(
        id="data_quality_audit",
        title="Data quality audit",
        # Every count here is a real defect in the fixture: 65 negative fares,
        # 1,298 zero-distance trips, 128 trips carrying nobody.
        prompt=(
            "Audit the quality of this dataset. Count the trips with a negative "
            "fare_amount, the trips with a negative total_amount, the trips "
            "with a trip_distance of zero, the trips with a passenger_count of "
            "zero, and the trips whose tpep_dropoff_datetime is not after "
            "tpep_pickup_datetime. Return one row per issue with the issue name "
            "and the count."
        ),
    ),
    AnalysisPrompt(
        id="passenger_count_effect",
        title="Party size against fare and tip",
        prompt=(
            "For each passenger_count return the number of trips, the average "
            "trip_distance, the average fare_amount and the average tip_amount, "
            "and rank the groups by average tip_amount."
        ),
    ),
    AnalysisPrompt(
        id="busiest_pickup_zones",
        title="Busiest pickup zones from raw coordinates",
        prompt=(
            "Round pickup_latitude and pickup_longitude to 3 decimal places to "
            "form a pickup zone. Return the 15 busiest zones with the number of "
            "trips, the average trip_distance and the average total_amount, "
            "ignoring rows where either coordinate is 0."
        ),
    ),
    AnalysisPrompt(
        id="driver_shift_earnings",
        title="Driver shift earnings (asks for fields that do not exist)",
        prompt=(
            "Analyse driver performance across shifts. For each Driver_ID "
            "return Shift, Trips_Completed, Avg_Rating and Total_Earnings, "
            "sorted by Total_Earnings."
        ),
        # The trip records are anonymous: no driver, no shift, no rating. Naming
        # Driver_ID and declining is the correct outcome; deriving a plausible
        # "driver" from VendorID would hand back confident figures about people
        # who are not in the data.
        expect=Expect.REFUSAL,
        missing_field="Driver_ID",
    ),
]


# ---------------------------------------------------------------------------
# TRANSFER tier
#
# The same dataset, but moved rather than analysed. It lives in this file on
# purpose: the transfers below name the very columns and thresholds the analysis
# prompts above use, so a schema change breaks both together instead of leaving
# the transfer suite quietly asserting against a dataset nobody uses any more.
#
# What runs everywhere is the ROUTING: a transfer phrasing must reach the
# transfer agent with the right destination, and must NOT be turned into
# generated pandas that writes a local CSV. Actually moving the bytes needs GCS
# credentials and a Docker/GKE runner, so that assertion skips unless
# AVALOKA_DTA_LIVE=1 is set.
# ---------------------------------------------------------------------------

#: Override with AVALOKA_DTA_SOURCE_BUCKET / _DEST_BUCKET to point at your own.
DTA_SOURCE_BUCKET = os.environ.get("AVALOKA_DTA_SOURCE_BUCKET", "avaloka-test-user-filestore")
DTA_SOURCE_OBJECT = os.environ.get(
    "AVALOKA_DTA_SOURCE_OBJECT", "test_c2c_jyothi/yellow_tripdata_2015-01_dataset_200k.csv"
)
DTA_DEST_BUCKET = os.environ.get("AVALOKA_DTA_DEST_BUCKET", "avaloka-dta-destination")
DTA_DEST_PREFIX = os.environ.get("AVALOKA_DTA_DEST_PREFIX", "transfers")

#: Cloud CONNECTION names -- the labels shown in Cloud Dataset Connections --
#: rather than raw bucket paths. Naming a connection is its own resolution path
#: in the planner (`_resolve_cloud_conn_by_name`), separate from the raw-bucket
#: phrasing above, and it is the path users actually take because the alternative
#: is pasting a UUID. It is also where transfers have broken silently: the
#: by-name lookup used to build the endpoint from a credential-stripped row, so
#: the runner authenticated as nobody and died with "storage.objects.get denied"
#: -- while the by-UUID phrasing kept working.
DTA_SOURCE_CONNECTION = os.environ.get("AVALOKA_DTA_SOURCE_CONNECTION", "Source_C2C_GCP")
DTA_DEST_CONNECTION = os.environ.get("AVALOKA_DTA_DEST_CONNECTION", "Destination_C2C_GCP")
DTA_SOURCE_FILE = os.environ.get("AVALOKA_DTA_SOURCE_FILE", "Yellow_Taxi_Trips_Cloud.csv")
DTA_DEST_FILE = os.environ.get("AVALOKA_DTA_DEST_FILE", "Yellow_Taxi_Trips_Cloud.json")


TRANSFER_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="transfer_filter_then_send",
        title="Filter, then transfer the result as JSON",
        # An analysis clause and a transfer clause in one sentence. The transfer
        # verb has to win -- generating pandas that writes a local CSV would look
        # like success and put the data nowhere near GCS.
        prompt=(
            "filter the trips whose total_amount is more than 100 and "
            f"transfer to {DTA_DEST_BUCKET}/{DTA_DEST_PREFIX} in .json format"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_implicit_source",
        title="Transfer the active dataset, destination only",
        # No source named: the source is the dataset the user is looking at.
        # Inventing one here moves data the user never pointed at.
        prompt=f"transfer to {DTA_DEST_BUCKET}/{DTA_DEST_PREFIX}",
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_named_bucket_to_bucket",
        title="Names both buckets and both objects",
        prompt=(
            f"transfer from {DTA_SOURCE_BUCKET} to {DTA_DEST_BUCKET}, "
            f"from {DTA_SOURCE_OBJECT} to {DTA_DEST_PREFIX}/yellow_taxi_trips.json"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_top_trips_copy",
        title="'copy' phrasing with a named output object",
        prompt=(
            "copy the top 100 trips by total_amount to "
            f"{DTA_DEST_BUCKET}/{DTA_DEST_PREFIX} as yellow_taxi_top100.json"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_named_connections_c2c",
        title="Filter, then cloud-to-cloud between NAMED connections",
        # The shape reported from the UI, and the one the other four miss: both
        # endpoints referenced by CONNECTION NAME, both objects named, and a
        # CSV -> JSON format change, with an analysis clause in front.
        #
        # Four things have to line up at once, and each has failed on its own:
        #   1. the transfer verb beats the leading "filter ..." clause;
        #   2. "Source_C2C_GCP" resolves by display name, not as a bucket path;
        #   3. the resolved endpoints carry CREDENTIALS (the by-name lookup
        #      returned a credential-stripped row, so the runner authenticated
        #      as nobody);
        #   4. non-finite numerics survive the CSV -> JSON conversion --
        #      json.dumps emits a bare NaN, which is not valid JSON, so the write
        #      "succeeds" and the NEXT append to that destination cannot parse it.
        #
        # 200 is deliberate: it keeps 4 trips out of 50,426, so a live run
        # finishes quickly but still moves real rows. It was 500 while the
        # fixture was the 200k export (max 587.93); on the committed file
        # total_amount tops out at 270.13, so 500 selected NOTHING and a live
        # transfer would have "succeeded" having moved an empty result.
        prompt=(
            "filter the trips whose total_amount is more than 200 and "
            f"transfer from {DTA_SOURCE_CONNECTION} to {DTA_DEST_CONNECTION} "
            f"from {DTA_SOURCE_FILE} to {DTA_DEST_FILE}"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
]


# ---------------------------------------------------------------------------
# OPERATIONS tier# ---------------------------------------------------------------------------
# OPERATIONS tier
#
# The transformation families from tests/test_dta_end_to_end.py -- column ops,
# null handling, type casting, string manipulation, date/time, arithmetic,
# filtering, masking, UDFs, aggregation -- plus joins, asked here as ANALYSIS
# prompts against the taxi schema. No transfers: the five transfer prompts above
# already cover that path.
#
# Every case carries a `verify` that inspects the result table. Without one the
# suite only asserts "some code ran and some rows came back", which passes just
# as happily when the agent ignores the instruction and returns the input
# unchanged. The checks are deliberately tolerant about naming -- the agent
# chooses its own output column names, and failing a correct answer for calling a
# column `avg_fare` instead of `Avg_fare_amount` would make the suite noise.
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


def _pickup_datetime(df):
    """The pickup timestamp column, whatever the agent renamed it to.

    Matching on the fragment "pickup" alone is not safe here -- pickup_latitude
    and pickup_longitude share the prefix and come back as perfectly good
    numbers, so a naive lookup silently checks a coordinate for a date. Skip the
    coordinate columns and require the values to actually parse as timestamps.
    """
    for c in df.columns:
        name = _norm(c)
        if "pickup" not in name or "tude" in name:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce").dropna()
        if not parsed.empty:
            return parsed
    return None


OPERATION_PROMPTS: List[AnalysisPrompt] = [
    # --- Column Operations -------------------------------------------------
    AnalysisPrompt(
        id="op_col_rename",
        title="Rename a column",
        prompt=(
            "Rename the tpep_pickup_datetime column to pickup_time and return "
            "the table."
        ),
        tier=Tier.OPERATION, category="Column Operations",
        # "tpep_pickup_datetime" normalises to "tpeppickupdatetime", which does
        # NOT contain "pickuptime" -- so the raw frame fails the first clause.
        verify=lambda df: _has_col(df, "pickuptime") and not _has_col(df, "tpeppickup"),
    ),
    AnalysisPrompt(
        id="op_col_select_subset",
        title="Select a subset of columns",
        prompt="Return only the VendorID, trip_distance and total_amount columns.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: len(df.columns) == 3 and _has_col(df, "tripdistance"),
    ),
    AnalysisPrompt(
        id="op_col_drop",
        title="Drop columns",
        prompt=(
            "Drop the pickup_latitude, pickup_longitude, dropoff_latitude and "
            "dropoff_longitude columns and return everything else."
        ),
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: not _has_col(df, "latitude") and not _has_col(df, "longitude"),
    ),

    # --- Null Handling ------------------------------------------------------
    #
    # This fixture has no nulls in any column, so a fill or a dropna is a no-op
    # every verify would pass whether or not the agent did anything. Both cases
    # below therefore assert on output SHAPE, which the untransformed 19-column
    # frame fails outright.
    AnalysisPrompt(
        id="op_null_count_per_column",
        title="Count nulls per column",
        prompt=(
            "Count the missing values in every column of df and return a "
            "two-column table of column name and missing count."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        # 19 columns in, so a correct answer is ~19 rows by 2 columns. The input
        # frame is 200,000 x 19 and fails both bounds.
        verify=lambda df: len(df.columns) <= 3 and 1 < len(df) <= 25,
    ),
    AnalysisPrompt(
        id="op_null_fill_zero",
        title="Fill nulls with zero",
        prompt=(
            "Replace any missing passenger_count with 0 and return "
            "trip_distance and passenger_count."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        # The width bound is the half that does the rejecting -- see the note on
        # test_selfcheck_every_operation_verify_rejects_the_untransformed_dataset.
        verify=lambda df: _no_nulls(df, "passengercount") and len(df.columns) <= 3,
    ),

    # --- Type Casting -------------------------------------------------------
    AnalysisPrompt(
        id="op_cast_int",
        title="Cast a float column to integer",
        prompt=(
            "Round trip_distance to the nearest whole mile as an integer and "
            "return VendorID and trip_distance."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        # Survives the CSV round-trip: pandas re-infers int64 only if every value
        # written was integral, which is exactly the property being asserted.
        verify=lambda df: (
            (v := _col(df, "tripdistance")) is not None
            and str(v.dtype).startswith(("int", "uint"))
        ),
    ),
    AnalysisPrompt(
        id="op_cast_string",
        title="Cast a numeric column to text",
        # A dtype check cannot work here: the result is written to CSV and read
        # back, and CSV carries no type information -- "1" written as text comes
        # back as int64 no matter what the agent did. Asking for a format that is
        # not numeric is what survives the round-trip.
        prompt=(
            "Convert payment_type to text formatted as 'Payment 1', and return "
            "VendorID and payment_type."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        verify=lambda df: (
            (v := _text(df, "paymenttype")) is not None
            and v.str.strip().str.lower().str.startswith("payment").mean() > 0.8
        ),
    ),

    # --- String Manipulation ------------------------------------------------
    AnalysisPrompt(
        id="op_str_expand_flag",
        title="Map a coded string to words",
        prompt=(
            "In store_and_fwd_flag replace Y with 'Yes' and N with 'No', and "
            "return VendorID and store_and_fwd_flag."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _text(df, "storeandfwd")) is not None
            and set(v.str.strip().str.lower().unique()) <= {"yes", "no"}
        ),
    ),
    AnalysisPrompt(
        id="op_str_date_prefix",
        title="Extract a substring",
        prompt=(
            "Add a column called pickup_date_text holding the first 10 "
            "characters of tpep_pickup_datetime, and return "
            "tpep_pickup_datetime and pickup_date_text."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        # "datetext", not "pickupdate": the source column tpep_pickup_datetime
        # normalises to "tpeppickupdatetime", which CONTAINS "pickupdate" and is
        # returned alongside the derived one -- so the looser fragment matched the
        # full timestamp column and failed a correct answer. The raw frame has no
        # "datetext" column at all, so the self-check still rejects it.
        verify=lambda df: (
            (v := _text(df, "datetext")) is not None
            and v.str.strip().str.match(r"^\d{4}-\d{2}-\d{2}$").mean() > 0.8
        ),
    ),
    AnalysisPrompt(
        id="op_str_concat_id",
        title="Concatenate two fields into one string",
        prompt=(
            "Add a column called trip_id that joins VendorID and "
            "tpep_pickup_datetime as text separated by a hyphen, and return "
            "trip_id and total_amount."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _text(df, "tripid")) is not None
            and v.str.contains("-").mean() > 0.8
        ),
    ),

    # --- Date & Time --------------------------------------------------------
    AnalysisPrompt(
        id="op_date_pickup_hour",
        title="Derive an hour from a timestamp",
        prompt=(
            "Using tpep_pickup_datetime, add a column called pickup_hour "
            "holding the hour of the day as a number from 0 to 23. Return "
            "tpep_pickup_datetime, pickup_hour and total_amount."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: (
            (v := _numeric(df, "pickuphour")) is not None and v.between(0, 23).all()
        ),
    ),
    AnalysisPrompt(
        id="op_date_trip_duration",
        title="Difference between two timestamps",
        prompt=(
            "Add a column called trip_duration_minutes equal to the number of "
            "minutes between tpep_pickup_datetime and tpep_dropoff_datetime, "
            "and return tpep_pickup_datetime, tpep_dropoff_datetime and "
            "trip_duration_minutes."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: _has_col(df, "duration"),
    ),
    AnalysisPrompt(
        id="op_date_filter_week",
        title="Filter on a date range",
        prompt=(
            "Return only the trips picked up between 2015-01-01 and 2015-01-07 "
            "inclusive."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        # The fixture runs to 2015-01-31, so the untransformed frame fails.
        verify=lambda df: (
            (v := _pickup_datetime(df)) is not None
            and (v < pd.Timestamp("2015-01-08")).all()
        ),
    ),

    # --- Arithmetic ---------------------------------------------------------
    AnalysisPrompt(
        id="op_arith_tip_percentage",
        title="Derive a ratio column",
        prompt=(
            "Add a column called tip_percentage equal to tip_amount divided by "
            "fare_amount times 100, and return fare_amount, tip_amount and "
            "tip_percentage."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        verify=lambda df: _has_col(df, "tip", "percent"),
    ),
    AnalysisPrompt(
        id="op_arith_surcharge_total",
        title="Sum several columns into one",
        prompt=(
            "Add a column called total_surcharges equal to extra plus mta_tax "
            "plus tolls_amount plus improvement_surcharge, and return "
            "fare_amount, total_surcharges and total_amount."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        # "improvement_surcharge" alone contains "surcharge" but not "total", so
        # the raw frame does not satisfy this.
        verify=lambda df: _has_col(df, "surcharge", "total"),
    ),

    # --- Filtering Logic ----------------------------------------------------
    AnalysisPrompt(
        id="op_filter_threshold",
        title="Numeric threshold",
        prompt="Return the trips whose total_amount is greater than 100.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (v := _numeric(df, "totalamount")) is not None and (v > 100).all()
        ),
    ),
    AnalysisPrompt(
        id="op_filter_compound",
        title="Two conditions combined",
        prompt=(
            "Return the trips paid by credit card (payment_type 1) that carried "
            "more than 2 passengers, including VendorID, passenger_count, "
            "payment_type and total_amount."
        ),
        tier=Tier.OPERATION, category="Filtering Logic",
        # passenger_count is the observable half and is explicitly asked for.
        # payment_type is checked only when it comes back, so a correct answer
        # that drops the now-constant column is not failed for it.
        verify=lambda df: (
            (p := _numeric(df, "passengercount")) is not None
            and (p > 2).all()
            and ((t := _numeric(df, "paymenttype")) is None or (t == 1).all())
        ),
    ),
    AnalysisPrompt(
        id="op_filter_isin",
        title="Membership filter",
        prompt="Return only the trips whose RateCodeID is 2, 3 or 4.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (v := _numeric(df, "ratecode")) is not None and v.isin({2, 3, 4}).all()
        ),
    ),

    # --- Data Masking -------------------------------------------------------
    AnalysisPrompt(
        id="op_mask_coordinates",
        title="Mask the location columns",
        # The one genuinely identifying thing in this schema. A coordinate pair
        # plus a timestamp is a home address with a time on it.
        prompt=(
            "Mask the pickup location by replacing every pickup_latitude and "
            "pickup_longitude value with the text REDACTED. Return VendorID, "
            "pickup_latitude, pickup_longitude and total_amount."
        ),
        tier=Tier.OPERATION, category="Data Masking",
        verify=lambda df: (
            (v := _text(df, "pickuplatitude")) is not None
            and (v.str.strip().str.upper() == "REDACTED").all()
        ),
    ),

    # --- UDF ----------------------------------------------------------------
    AnalysisPrompt(
        id="op_udf_distance_band",
        title="Custom bucketing function",
        prompt=(
            "Classify each trip as Short if trip_distance is under 1 mile, "
            "Medium if it is under 5 miles, otherwise Long. Put the label in a "
            "column called trip_length_band and return trip_distance and "
            "trip_length_band."
        ),
        tier=Tier.OPERATION, category="UDF",
        verify=lambda df: (
            (v := _text(df, "band")) is not None
            and set(v.str.strip().str.lower().unique()) <= {"short", "medium", "long"}
        ),
    ),

    # --- Aggregation --------------------------------------------------------
    AnalysisPrompt(
        id="op_agg_multi_metric",
        title="Several aggregates in one group-by",
        prompt=(
            "For each payment_type return the number of trips, the total "
            "total_amount and the average tip_amount."
        ),
        tier=Tier.OPERATION, category="Aggregation",
        # 4 distinct payment types against 200,000 input rows.
        verify=lambda df: (
            _has_col(df, "payment") and 1 < len(df) <= 10 and len(df.columns) >= 4
        ),
    ),

    # --- Joins --------------------------------------------------------------
    AnalysisPrompt(
        id="op_join_vendor_totals",
        title="Join an aggregate back onto the rows",
        prompt=(
            "Compute the total revenue per VendorID from total_amount, then "
            "join that total back onto every trip row. Return VendorID, "
            "total_amount and the vendor total in a column called "
            "vendor_total_revenue."
        ),
        tier=Tier.OPERATION, category="Joins",
        # "VendorID" contains "vendor" but not "total", so the raw frame fails.
        verify=lambda df: _has_col(df, "vendor", "total") and len(df) > 1,
    ),
    AnalysisPrompt(
        id="op_join_share_of_vendor",
        title="Join then derive a share",
        prompt=(
            "For each trip work out what share of its vendor's total revenue it "
            "accounts for. Return VendorID, total_amount and a column called "
            "share_of_vendor_revenue."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "share"),
    ),
    AnalysisPrompt(
        id="op_join_hour_rank",
        title="Join a group rank back onto the rows",
        prompt=(
            "Rank the trips by total_amount within their own pickup hour, join "
            "that rank back onto each row, and return tpep_pickup_datetime, "
            "total_amount and a column called rank_in_hour."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "rank"),
    ),
]


#: Everything, in the order the suite reports it.
ANALYSIS_PROMPTS: List[AnalysisPrompt] = (
    SINGLE_PROMPTS + MULTI_PROMPTS + OPERATION_PROMPTS + TRANSFER_PROMPTS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Row cap for a quicker pass (TAXI_MAX_ROWS=20000). None of the assertions above
#: depend on the exact row count, so trimming only costs coverage of scale.
#: Applies to both the committed CSV and a downloaded workbook.
MAX_ROWS = int(os.environ.get("TAXI_MAX_ROWS", "0") or 0)

#: Basenames to look for, most-preferred first. The 100k CSV is the copy committed
#: to app/sample_data/ so a fresh clone can run this suite with no download; the
#: 200k workbook is the fuller export and wins when it is present locally.
#:
#: Both names are listed on purpose. When the committed fixture was renamed from
#: 200k to 100k and this list still said 200k, the suite kept passing here -- the
#: Downloads copy resolved first -- while a fresh clone found nothing, hit
#: pytest.skip, and reported all 287 tests as skipped. A green run that asserted
#: nothing is worse than a red one.
DATASET_BASENAMES = (
    "yellow_tripdata_2015-01_dataset_50k",
    "yellow_tripdata_2015-01_dataset_200k",
)

#: Where a locally downloaded Kaggle export tends to live.
DOWNLOADS_DIR = _downloads_dir() / "yellow_taxi_trip_data"


def _candidate_dataset_paths() -> List[Path]:
    paths: List[Path] = []
    env_path = os.environ.get("YELLOW_TAXI_DATASET")
    if env_path:
        paths.append(Path(env_path))
    sample_dir = REPO_ROOT / "app" / "sample_data"
    for name in DATASET_BASENAMES:
        for directory in (DOWNLOADS_DIR, sample_dir):
            for suffix in (".xlsx", ".csv"):
                paths.append(directory / f"{name}{suffix}")
    return paths


@pytest.fixture(scope="session")
def dataset_source() -> Path:
    """The dataset as shipped -- a .csv, or an .xlsx workbook if that is what is there."""
    for path in _candidate_dataset_paths():
        if path.exists():
            LOGGER.info("Using dataset: %s", path)
            return path
    pytest.skip(
        "No yellow-taxi dataset found. Expected one of "
        + ", ".join(f"{n}.csv/.xlsx" for n in DATASET_BASENAMES)
        + " in app/sample_data/ (the 100k CSV is committed there) or "
        f"{DOWNLOADS_DIR}, or set YELLOW_TAXI_DATASET to its path."
    )


@pytest.fixture(scope="session")
def dataset_path(dataset_source: Path, tmp_path_factory) -> Path:
    """A CSV the graph can actually read, converting the workbook once if needed.

    The local execution path loads the source with ``read_csv_best_effort``
    (app/agents/execution_agent.py), so pointing ``data_source_location`` at an
    .xlsx fails every prompt with a pandas parse error -- a harness failure that
    looks exactly like an agent failure. Converting here keeps the run honest:
    the graph sees the same rows it would have seen from a CSV upload.
    """
    is_workbook = dataset_source.suffix.lower() in {".xlsx", ".xls"}
    # A CSV is already readable, so it is handed over untouched unless the caller
    # asked for a row cap -- honouring TAXI_MAX_ROWS only for workbooks would make
    # the flag silently do nothing now that the committed fixture is a CSV.
    if not is_workbook and not MAX_ROWS:
        return dataset_source

    converted = tmp_path_factory.mktemp("taxi_dataset") / f"{dataset_source.stem}.csv"
    LOGGER.info("Materializing %s -> %s (max_rows=%s)", dataset_source, converted,
                MAX_ROWS or "all")
    if is_workbook:
        frame = pd.read_excel(dataset_source, nrows=MAX_ROWS or None)
    else:
        frame = pd.read_csv(dataset_source, nrows=MAX_ROWS or None,
                            encoding_errors="replace")
    frame.to_csv(converted, index=False)
    LOGGER.info("Materialized %d rows x %d columns", len(frame), len(frame.columns))
    return converted


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
        user_id="test_yellow_taxi_analysis_user",
        session_id=f"test_yellow_taxi_analysis_{entry.id}",
        user_prompt=entry.prompt,
        messages=[HumanMessage(content=entry.prompt)],
        plan="",
        # Planner-first: these two flags must stay False so plan_etl actually runs.
        ready_to_summarize=False,
        ready_to_code=False,
        data_source_location=str(dataset_path),
        output_location=str(output_location),
        schema=dataset_schema,
        # The workbook was converted to CSV by the dataset_path fixture, so this
        # is "csv" and not "xlsx" -- see that fixture for why.
        input_data_type="csv",
        uploaded_csv_columns=list(uploaded_preview[0]),
        uploaded_csv_preview=uploaded_preview,
        # Keep execution local; the cloud/Ray path needs a connection_id.
        analysis_fidelity="quick_sample",
        execution_mode="local",
        dataset_id=f"ds_yellow_taxi_{entry.id}",
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
    return tmp_path_factory.mktemp("yellow_taxi_analysis_outputs")


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

    The tier marker is what makes `-m transfer` work. Without it the only way to
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
#: count and a refusal deliberately runs no code.
_TABLE_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.TABLE]
_ANSWER_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.ANSWER]
_REFUSAL_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.REFUSAL]
_TRANSFER_TIER_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.TRANSFER]
_OPERATION_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.verify is not None]

_PARAMS = _params(_TABLE_PROMPTS)
_PARAMS_ALL = _params(ANALYSIS_PROMPTS)
_PARAMS_ANSWER = _params(_ANSWER_PROMPTS)
_PARAMS_REFUSAL = _params(_REFUSAL_PROMPTS)
_PARAMS_TRANSFER = _params(_TRANSFER_TIER_PROMPTS)
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

    assert not state.get("syntax_error"), f"Syntax error in generated code{_diagnostics(result)}"
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

    TABLE prompts only. A refusal and a transfer both reach the planner and both
    correctly leave NO analysis plan -- a refusal because it declined, a transfer
    because it routed to the DTA. Asserting a plan for those would fail them for
    behaving exactly as designed.
    """
    result = prompt_run
    state = result["state"]
    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )
    assert state.get("plan"), f"Planner produced no plan{_diagnostics(result)}"


# ---------------------------------------------------------------------------
# Tests — visualization charts
# ---------------------------------------------------------------------------


@_PARAMS
def test_prompt_produces_visualization_charts(prompt_run):
    """Every successful analysis must come back with a renderable chart config."""
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    status = state.get("visualization_status")
    assert status == "ready", (
        f"visualization_status is {status!r}, expected 'ready'{_diagnostics(result)}"
    )

    config = state.get("visualization_config")
    assert isinstance(config, dict) and config, (
        f"visualization_config missing or empty{_diagnostics(result)}"
    )

    charts = _charts(state)
    assert charts, f"visualization_config carries no charts{_diagnostics(result)}"

    for chart in charts:
        for key in ("id", "title", "type", "intent", "encodings", "rank", "reason"):
            assert key in chart, (
                f"Chart {chart.get('id') or chart.get('title')!r} is missing "
                f"'{key}'{_diagnostics(result)}"
            )
        encodings = chart["encodings"]
        assert isinstance(encodings, dict) and encodings, (
            f"Chart {chart['id']!r} has empty encodings{_diagnostics(result)}"
        )
        assert str(chart["reason"]).strip(), (
            f"Chart {chart['id']!r} has an empty reason{_diagnostics(result)}"
        )

    ranks = sorted(chart["rank"] for chart in charts)
    assert ranks == list(range(1, len(charts) + 1)), (
        f"Chart ranks are not contiguous 1..N: {ranks}{_diagnostics(result)}"
    )


@_PARAMS
def test_chart_fields_exist_in_result_table(prompt_run):
    """Chart encodings must reference columns that are actually in the output.

    A chart pointing at a column the result table does not have renders blank in
    the UI, and nothing else in this file notices.
    """
    result = prompt_run
    state = result["state"]
    charts = _charts(state)
    if not charts:
        pytest.skip("No charts produced; covered by test_prompt_produces_visualization_charts")

    columns = _result_columns(result)
    if not columns:
        pytest.skip("Output columns unavailable for this run")

    known = {str(c) for c in columns}
    dangling = []
    for chart in charts:
        for channel, encoding in (chart.get("encodings") or {}).items():
            if not isinstance(encoding, dict):
                continue
            field = encoding.get("field")
            if field and str(field) not in known:
                dangling.append(f"{chart.get('id')}.{channel} -> {field!r}")

    assert not dangling, (
        f"Charts reference columns absent from the result table {sorted(known)}: "
        f"{dangling}{_diagnostics(result)}"
    )


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


# ---------------------------------------------------------------------------
# Tests — SINGLE tier: prose answers and refusals
#
# The MULTI tier only ever asserts "code ran, rows came back". These two shapes
# are the ones a short conversational ask actually produces.
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
    assert any(term in lowered for term in ("trip", "taxi", "fare", "ride", "passenger")), (
        f"Reply does not mention what the data is about: {reply[:200]!r}"
        f"{_diagnostics(result)}"
    )


@_PARAMS_REFUSAL
def test_prompt_declines_and_names_the_missing_field(prompt_run):
    """Asking for a field the dataset does not have must produce a refusal that
    names the field -- and must not produce numbers.

    Inventing a plausible formula for a column nobody has is the worst possible
    outcome here: the user gets confident figures with nothing behind them. A
    generic "I can't help with that" is not much better, because it gives them
    no way to rephrase, so the reply has to name the field it could not map.
    """
    entry: AnalysisPrompt = prompt_run["entry"]
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )

    reply = _assistant_reply(state)
    assert reply.strip(), f"Refusal produced no reply at all{_diagnostics(result)}"

    assert entry.missing_field.lower() in reply.lower(), (
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
    """Whatever the outcome, the user must not be shown a stack-trace artefact
    or the planner's generic apology."""
    result = prompt_run
    reply = _assistant_reply(result["state"]).lower()
    if not reply:
        return
    # Look for a RAISED error, not the mere mention of one. The summarizer
    # legitimately discusses error handling in prose -- "defensive checks could
    # prevent a KeyError if the input CSV were malformed" is good writing, and a
    # bare substring match failed the turn for it. A real leak looks like a
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
# Tests — TRANSFER tier
# ---------------------------------------------------------------------------

#: Actually moving bytes needs GCS credentials and a Docker/GKE runner. Routing
#: does not, and routing is where every reported transfer defect has lived.
DTA_LIVE = os.environ.get("AVALOKA_DTA_LIVE", "").strip().lower() in {"1", "true", "yes"}


@_PARAMS_TRANSFER
def test_transfer_prompt_routes_to_the_transfer_agent(prompt_run):
    """A transfer phrasing must reach the transfer agent, not the code generator.

    This is the failure that looks like success: the planner writes pandas that
    saves a local CSV, every stage reports green, and the data never goes near
    the destination bucket. Asserting `is_dta_request` is what separates the two.
    """
    result = prompt_run
    state = result["state"]

    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )
    assert state.get("is_dta_request") is True, (
        "Prompt was not routed to the transfer agent; the planner treated a "
        f"transfer as an analysis{_diagnostics(result)}"
    )
    assert not state.get("ready_to_code"), (
        f"Transfer prompt was sent to code generation{_diagnostics(result)}"
    )


#: Phrases the DTA uses when it cannot begin: no active dataset, or an endpoint
#: that was never registered in this session. Both are correct answers here --
#: the test session registers no connections -- and neither can name a
#: destination, because the transfer never got that far.
_CANNOT_START = (
    "don't see an active source", "do not see an active source",
    "not found", "not registered", "could not find", "couldn't find",
    "which table", "register",
)


@_PARAMS_TRANSFER
def test_transfer_prompt_gives_a_usable_answer(prompt_run):
    """Either name the destination, or say clearly why the transfer cannot start.

    Demanding the destination unconditionally fails replies like "Source
    avaloka-test-user-filestore not found" -- which is the RIGHT answer in a
    session with no registered connections. What actually matters is that the
    user is never left without either a destination or a reason.
    """
    result = prompt_run
    reply = _assistant_reply(result["state"])
    assert reply.strip(), f"Transfer produced no reply{_diagnostics(result)}"

    names_destination = DTA_DEST_BUCKET.lower() in reply.lower()
    explains_why_not = any(p in reply.lower() for p in _CANNOT_START)
    assert names_destination or explains_why_not, (
        f"Reply neither names the destination {DTA_DEST_BUCKET!r} nor says why "
        f"the transfer cannot start: {reply[:300]!r}{_diagnostics(result)}"
    )


@_PARAMS_TRANSFER
def test_transfer_prompt_does_not_write_a_local_result(prompt_run):
    """A transfer must not quietly leave its output in the local analysis path.

    If rows land in output_location the request was handled as an analysis, and
    the user has a local file where they asked for a bucket object.
    """
    result = prompt_run
    rows = _output_rows(result)
    assert not rows, (
        f"Transfer prompt produced {rows} rows of LOCAL analysis output"
        f"{_diagnostics(result)}"
    )


@pytest.mark.skipif(
    not DTA_LIVE,
    reason="Set AVALOKA_DTA_LIVE=1 (plus GCS credentials and a Docker/GKE runner) "
           "to move real bytes; routing is covered without them.",
)
@_PARAMS_TRANSFER
def test_transfer_prompt_completes_end_to_end(prompt_run):
    """The real thing: the transfer runs and reports a completed job."""
    result = prompt_run
    state = result["state"]
    reply = _assistant_reply(state).lower()

    for failure in ("failed", "could not", "unable to", "error"):
        assert failure not in reply, (
            f"Transfer reported a failure: {reply[:300]!r}{_diagnostics(result)}"
        )
    assert any(ok in reply for ok in ("transfer", "moved", "copied", "complete")), (
        f"Reply does not confirm a transfer happened: {reply[:300]!r}"
        f"{_diagnostics(result)}"
    )


# ---------------------------------------------------------------------------
# Tests - OPERATIONS tier# ---------------------------------------------------------------------------
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
# agent does nothing at all. These need no LLM credentials -- only the dataset
# file -- so `pytest tests/kaggle_yellow_taxi_trip_data.py -k selfcheck` on a
# laptop still checks them.
# ---------------------------------------------------------------------------


def test_selfcheck_every_operation_verify_rejects_the_untransformed_dataset(dataset_frame):
    """Each OPERATION check must fail against the raw fixture.

    Worth stating for `op_null_fill_zero`: this fixture has no nulls anywhere, so
    its null-freeness clause passes on the raw frame and the width bound is the
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


def test_selfcheck_prompt_ids_are_unique():
    """Duplicate ids collide in _RUN_CACHE, and one prompt silently reuses the
    other's run."""
    ids = [p.id for p in ANALYSIS_PROMPTS]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"Duplicate prompt ids: {duplicates}"


def test_selfcheck_refusal_prompts_name_a_missing_field():
    """A REFUSAL case with no `missing_field` asserts nothing beyond "it replied"."""
    unnamed = [p.id for p in _REFUSAL_PROMPTS if not p.missing_field.strip()]
    assert not unnamed, f"REFUSAL prompts with no missing_field declared: {unnamed}"


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
