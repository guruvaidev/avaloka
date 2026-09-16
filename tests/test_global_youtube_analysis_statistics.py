"""End-to-end regression suite for the Global YouTube Statistics prompts.

One dataset, four tiers of prompt, so that a schema change breaks every tier
together instead of leaving one of them asserting against data nobody uses:

  SINGLE    short conversational asks with no column names and no output shape
            ("what does this dataset talk about ?", "filter the channels whose
            subscribers is more than 20L"). These exercise what the planner can
            INFER, and they produce the two outcomes the MULTI tier never does:
            a prose answer, and a refusal.

  MULTI     the originally reported prompts -- several clauses, explicit columns,
            an explicit output shape. These exercise instruction-FOLLOWING.

  OPERATIONS the transformation families benchmarked in
            tests/test_dta_end_to_end.py -- column ops, null handling, type
            casting, string manipulation, date/time, arithmetic, filtering,
            masking, UDFs, aggregation -- plus joins, asked here as ANALYSIS
            prompts. Each carries a `verify` that inspects the result table,
            because "code ran and rows came back" passes just as happily when
            the agent ignores the instruction and returns the input unchanged.
            Every verify is checked to REJECT the untransformed dataframe.

  TRANSFER  the same dataset moved rather than analysed, from
            gs://avaloka-test-user-filestore/test_c2c_jyothi/ to
            gs://avaloka-dta-destination/transfers. Routing runs everywhere;
            moving real bytes needs AVALOKA_DTA_LIVE=1 plus credentials.

Not every correct answer is a table, so each prompt declares what a correct run
looks like (``Expect.TABLE`` / ``ANSWER`` / ``REFUSAL`` / ``TRANSFER``) and only
the matching assertions run against it. Before that, ``channel_age_success`` was
counted as a suite failure for correctly refusing to invent an ``Age_bucket``
column -- the suite punished the behaviour it should have been protecting.

Each prompt is driven through the *full* graph starting at the planner
(``ready_to_code=False`` / ``ready_to_summarize=False``), so the run covers
plan_etl -> summarize_etl -> generate_planner_graph -> code_etl -> execute ->
visualize. That is deliberate: skipping straight to the coder (as
``tests/test_e2e_kaggle.py`` does) hides planner-side regressions.

Each prompt is executed exactly once; the resulting final state is cached and
shared by the pipeline assertions and the visualization assertions, so the two
test families cost one graph run per prompt rather than two.

Running::

    dev-env.bat                       # activates venv + sets GROQ_* keys
    set CHROMA_HOST=127.0.0.1         # see the Layer 2 note below
    set CHROMA_PORT=59999
    pytest tests/test_global_youtube_analysis_statistics.py -v

Layer 2 note: ``retrieve_memory`` abandons its worker thread after a 3s circuit
breaker, and the orphan then calls into ChromaDB's Rust bindings while the graph
has moved on. On Windows that reliably takes the whole interpreter down with an
access violation (0xC0000005) in ``chromadb/api/rust.py``, mid-suite. Pointing
CHROMA_HOST at a dead port forces Layer 2 onto its supported in-process fallback
and keeps the run alive. Remove the override once the orphaned-thread teardown
is fixed.

Dataset location resolution order:
  1. ``YOUTUBE_STATS_CSV`` environment variable
  2. ``$AVALOKA_DATA_DIR/Global_YouTube_Statistics.csv`` (defaults to ~/Downloads)
  3. ``app/sample_data/Global_YouTube_Statistics.csv``
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
LOG_FILE = Path(__file__).parent / "global_youtube_analysis_statistics.log"

# Append, do not truncate. basicConfig runs at IMPORT, so filemode="w" meant
# that anything importing this module -- `--collect-only`, an IDE test
# discovery pass, a `-k` run of two prompts -- silently destroyed the log of the
# run you were trying to diagnose. That happened twice while debugging this
# suite. A run boundary is written below instead, which is all the separation a
# grep actually needs.
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
    the suite for behaving correctly, which is what `channel_age_success` was
    doing before this.
    """

    TABLE = "table"      # runs code and returns rows
    ANSWER = "answer"    # answers in prose; no result table expected
    REFUSAL = "refusal"  # must decline, and must not invent the missing field
    TRANSFER = "transfer"  # routes to the transfer agent, not to code generation


class Tier(str, Enum):
    """How much the prompt spells out.

    SINGLE is the coverage this suite was missing: short, conversational asks
    with no column names and no output shape, where the planner has to infer
    the intent from the schema rather than be handed it.
    """

    SINGLE = "single"
    MULTI = "multi"
    OPERATION = "operations"
    TRANSFER = "transfer"


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
        id="single_filter_subscribers",
        title="Filter by subscriber threshold",
        # "20L" is 20 lakh = 2,000,000. Indian numbering is how the ask arrives
        # in practice, and it is a real parse the planner has to get right.
        prompt="filter the channels whose subscribers is more than 20L",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_top_channels",
        title="Top channels, unqualified",
        prompt="show me the top 10 channels",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_how_many_countries",
        title="Simple count question",
        prompt="how many countries are in this data?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_biggest_category",
        title="Superlative with no metric named",
        prompt="which category is the biggest?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_average_uploads",
        title="Aggregate stated in plain words",
        prompt="what is the average number of uploads per channel?",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_india_channels",
        title="Filter by a value, not a column",
        prompt="show only the channels from India",
        tier=Tier.SINGLE,
    ),
    AnalysisPrompt(
        id="single_missing_column",
        title="Asks for a field that does not exist",
        # The dataset has no engagement rate and no way to derive one. The right
        # answer is to say so and name the field, not to invent a formula.
        prompt="rank the channels by their engagement rate",
        tier=Tier.SINGLE,
        expect=Expect.REFUSAL,
        missing_field="engagement",
    ),
]


# ---------------------------------------------------------------------------
# MULTI-prompt tier
#
# The originally reported prompts: several clauses, explicit column names, an
# explicit output shape.
# ---------------------------------------------------------------------------

MULTI_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="growth_momentum",
        title="Growth / momentum",
        prompt=(
            "Identify the fastest-growing channels by comparing "
            "subscribers_for_last_30_days to total subscribers. List the top 20 "
            "channels and include their Youtuber, category, Country, subscribers, "
            "and subscribers_for_last_30_days."
        ),
    ),
    AnalysisPrompt(
        id="country_opportunity",
        title="Country-level opportunity",
        prompt=(
            "For each Country, compute the total subscribers, total video views, "
            "and number of channels. Rank countries by average subscribers per "
            "channel and highlight the top 10 and bottom 10 countries."
        ),
    ),
    AnalysisPrompt(
        id="earnings_by_category",
        title="Earnings across categories",
        prompt="Compare earnings across different categories.",
    ),
    AnalysisPrompt(
        id="underserved_niches",
        title="Underserved niches",
        prompt=(
            "Treat each row as a channel and define a niche as the combination of "
            "category, channel_type, and Country. For each niche, compute number of "
            "channels, total subscribers, and total video views. Identify niches that "
            "have few channels but high total video views per channel and return the "
            "top 20 underserved niches."
        ),
    ),
    AnalysisPrompt(
        id="category_by_country",
        title="Category performance by country",
        prompt=(
            "For each Country and category, calculate the average video views and "
            "average subscribers per channel. Return a table of the top 15 "
            "(Country, category) pairs ranked by average video views per channel."
        ),
    ),
    AnalysisPrompt(
        id="earnings_vs_audience",
        title="Earnings vs audience size",
        prompt=(
            "Using lowest_yearly_earnings, highest_yearly_earnings, and subscribers, "
            "estimate an earnings-per-subscriber range for each channel. Find the 20 "
            "channels with the highest estimated earnings per subscriber and the 20 "
            "with the lowest."
        ),
    ),
    AnalysisPrompt(
        id="channel_type_comparison",
        title="Channel type comparison",
        prompt=(
            "Compare different channel_type values globally: for each channel_type, "
            "compute number of channels, total subscribers, total video views, and "
            "average video_views_for_the_last_30_days. Rank channel_type by average "
            "video views per channel."
        ),
    ),
    AnalysisPrompt(
        id="macro_vs_youtube",
        title="Macroeconomics vs YouTube",
        prompt=(
            "Aggregate by Country and compute total video views and total subscribers. "
            "Then join this with Population, Urban_population, and Unemployment rate to "
            "analyze how YouTube penetration (subscribers per capita and views per "
            "capita) varies with these macro indicators."
        ),
    ),
    AnalysisPrompt(
        id="channel_age_success",
        title="Channel age and success (asks for a field that does not exist)",
        prompt=(
            "Analyse channel age and success. Bucket channels by age and return a "
            "summary DataFrame with the columns Age_bucket, Avg_subscribers, "
            "Avg_video_views, Channel_count, sorted in the logical bucket order."
        ),
        # There is no Age_bucket column and no age to bucket by -- created_year
        # exists but the prompt names a field, not a derivation. Declining and
        # naming Age_bucket is the correct outcome; this used to be counted as a
        # suite failure for getting it right.
        expect=Expect.REFUSAL,
        missing_field="Age_bucket",
    ),
    AnalysisPrompt(
        id="top_creators_per_country",
        title="Top creators per country",
        prompt=(
            "For each Country, identify the top 5 channels by subscribers and by video "
            "views. Return a combined table showing Country, Youtuber, subscribers, "
            "video views, category, and channel_type."
        ),
    ),
    AnalysisPrompt(
        id="view_efficiency",
        title="View efficiency",
        prompt=(
            "For each channel, compute a simple view efficiency metric as video views "
            "divided by uploads. Find the top 30 channels with the highest view "
            "efficiency, and show their Youtuber, Country, category, channel_type, "
            "uploads, video views, and the computed efficiency."
        ),
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
# The source copy is the one uploaded to
#   gs://avaloka-test-user-filestore/test_c2c_jyothi/Global_YouTube_Statistics.csv
# and the destination is gs://avaloka-dta-destination/transfers.
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
    "AVALOKA_DTA_SOURCE_OBJECT", "test_c2c_jyothi/Global_YouTube_Statistics.csv"
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
#: -- while the by-UUID phrasing kept working. Nothing above covers it.
DTA_SOURCE_CONNECTION = os.environ.get("AVALOKA_DTA_SOURCE_CONNECTION", "Source_C2C_GCP")
DTA_DEST_CONNECTION = os.environ.get("AVALOKA_DTA_DEST_CONNECTION", "Destination_C2C_GCP")
DTA_SOURCE_FILE = os.environ.get(
    "AVALOKA_DTA_SOURCE_FILE", "Global_YouTube_Statistics_Cloud.csv"
)
DTA_DEST_FILE = os.environ.get(
    "AVALOKA_DTA_DEST_FILE", "Global_YouTube_Statistics_Cloud.json"
)


TRANSFER_PROMPTS: List[AnalysisPrompt] = [
    AnalysisPrompt(
        id="transfer_filter_then_send",
        title="Filter, then transfer the result as JSON",
        # The reported shape: an analysis clause and a transfer clause in one
        # sentence. The transfer verb has to win -- generating pandas that writes
        # a local CSV would look like success and put the data nowhere near GCS.
        prompt=(
            "filter the channels whose subscribers is more than 20L and "
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
            f"from {DTA_SOURCE_OBJECT} to {DTA_DEST_PREFIX}/youtube_stats.json"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_top_channels_copy",
        title="Move phrasing with a named output object",
        prompt=(
            "copy the top 100 channels by subscribers to "
            f"{DTA_DEST_BUCKET}/{DTA_DEST_PREFIX} as youtube_top100.json"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
    AnalysisPrompt(
        id="transfer_named_connections_c2c",
        title="Filter, then cloud-to-cloud between NAMED connections",
        # The exact shape reported from the UI, and the one the other four miss:
        # both endpoints referenced by CONNECTION NAME, both objects named, and a
        # CSV -> JSON format change, with an analysis clause in front.
        #
        # Four things have to line up at once, and each has failed on its own:
        #   1. the transfer verb beats the leading "filter ..." clause;
        #   2. "Source_C2C_GCP" resolves by display name, not as a bucket path;
        #   3. the resolved endpoints carry CREDENTIALS (the by-name lookup
        #      returned a credential-stripped row, so the runner authenticated
        #      as nobody);
        #   4. NaN in a numeric column survives the CSV -> JSON conversion --
        #      json.dumps emits a bare NaN, which is not valid JSON, so the write
        #      "succeeds" and the NEXT append to that destination cannot parse it.
        #
        # 119000000 is deliberate: it keeps only a handful of channels, so a
        # live run finishes quickly, and the surviving rows still include ones
        # with blank numeric cells -- which is what makes (4) reproducible.
        prompt=(
            "filter the channels whose subscribers is more than 119000000 and "
            f"transfer from {DTA_SOURCE_CONNECTION} to {DTA_DEST_CONNECTION} "
            f"from {DTA_SOURCE_FILE} to {DTA_DEST_FILE}"
        ),
        tier=Tier.TRANSFER,
        expect=Expect.TRANSFER,
    ),
]


# ---------------------------------------------------------------------------
# OPERATIONS tier
#
# The transformation families from tests/test_dta_end_to_end.py -- column ops,
# null handling, type casting, string manipulation, date/time, arithmetic,
# filtering, masking, UDFs, aggregation -- plus joins, asked here as ANALYSIS
# prompts against the YouTube schema. No transfers: the four transfer prompts
# above already cover that path.
#
# Every case carries a `verify` that inspects the result table. Without one the
# suite only asserts "some code ran and some rows came back", which passes just
# as happily when the agent ignores the instruction and returns the input
# unchanged. The checks are deliberately tolerant about naming -- the agent
# chooses its own output column names, and failing a correct answer for calling
# a column `avg_subs` instead of `Avg_subscribers` would make the suite noise.
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


OPERATION_PROMPTS: List[AnalysisPrompt] = [
    # --- Column Operations -------------------------------------------------
    AnalysisPrompt(
        id="op_col_rename",
        title="Rename a column",
        prompt="Rename the Youtuber column to channel_name and return the table.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: _has_col(df, "channelname") and not _has_col(df, "youtuber"),
    ),
    AnalysisPrompt(
        id="op_col_select_subset",
        title="Select a subset of columns",
        prompt="Return only the Youtuber, Country and subscribers columns.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: len(df.columns) == 3 and _has_col(df, "country"),
    ),
    AnalysisPrompt(
        id="op_col_drop",
        title="Drop columns",
        prompt="Drop the Latitude and Longitude columns and return everything else.",
        tier=Tier.OPERATION, category="Column Operations",
        verify=lambda df: not _has_col(df, "latitude") and not _has_col(df, "longitude"),
    ),

    # --- Null Handling ------------------------------------------------------
    AnalysisPrompt(
        id="op_null_fill_zero",
        title="Fill nulls with zero",
        prompt=(
            "Fill the missing values in subscribers_for_last_30_days with 0 and "
            "return Youtuber and subscribers_for_last_30_days."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        verify=lambda df: _no_nulls(df, "subscribers", "30"),
    ),
    AnalysisPrompt(
        id="op_null_drop_rows",
        title="Drop rows with nulls",
        prompt="Remove every channel with a missing Country and return Youtuber and Country.",
        tier=Tier.OPERATION, category="Null Handling",
        verify=lambda df: _no_nulls(df, "country"),
    ),
    AnalysisPrompt(
        id="op_null_impute_mean",
        title="Impute with the mean",
        # video_views_for_the_last_30_days has 56 nulls; "video views" has none,
        # so imputing that one would be a no-op no check could tell from success.
        prompt=(
            "Replace missing video_views_for_the_last_30_days with the average "
            "across all channels, and return Youtuber and "
            "video_views_for_the_last_30_days."
        ),
        tier=Tier.OPERATION, category="Null Handling",
        verify=lambda df: _no_nulls(df, "videoviews", "30"),
    ),

    # --- Type Casting -------------------------------------------------------
    AnalysisPrompt(
        id="op_cast_int",
        title="Cast a float column to integer",
        prompt=(
            "Convert video views to a whole number (integer) and return Youtuber "
            "and video views."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        verify=lambda df: (
            (v := _col(df, "videoviews")) is not None
            and str(v.dtype).startswith(("int", "uint"))
        ),
    ),
    AnalysisPrompt(
        id="op_cast_string",
        title="Cast a numeric column to text",
        # A dtype check cannot work here: the result is written to CSV and read
        # back, and CSV carries no type information -- "2006" written as text
        # returns as int64 no matter what the agent did. Verified live, and it
        # made the original version of this case unsatisfiable. Asking for a
        # format that is not numeric is what survives the round-trip.
        prompt=(
            "Convert created_year to text formatted as 'Year 2006', and return "
            "Youtuber and created_year."
        ),
        tier=Tier.OPERATION, category="Type Casting",
        verify=lambda df: (
            (v := _col(df, "createdyear")) is not None
            and (vals := v.dropna().astype(str)).size > 0
            and vals.str.strip().str.lower().str.startswith("year").mean() > 0.8
        ),
    ),

    # --- String Manipulation ------------------------------------------------
    AnalysisPrompt(
        id="op_str_upper",
        title="Uppercase a string column",
        prompt="Convert every Country name to uppercase and return Youtuber and Country.",
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _col(df, "country")) is not None
            and (vals := v.dropna().astype(str)).size > 0
            and (vals == vals.str.upper()).all()
        ),
    ),
    AnalysisPrompt(
        id="op_str_length",
        title="Derive a string length column",
        prompt=(
            "Add a column called name_length holding the number of characters in "
            "each Youtuber name, and return Youtuber and name_length."
        ),
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: _has_col(df, "namelength"),
    ),
    AnalysisPrompt(
        id="op_str_contains_filter",
        title="Filter by a substring",
        prompt="Return only the channels whose category contains the word Music.",
        tier=Tier.OPERATION, category="String Manipulation",
        verify=lambda df: (
            (v := _col(df, "category")) is not None
            and v.dropna().astype(str).str.contains("music", case=False).all()
        ),
    ),

    # --- Date & Time --------------------------------------------------------
    AnalysisPrompt(
        id="op_date_channel_age",
        title="Derive an age from a year column",
        prompt=(
            "Using created_year, add a column called channel_age_years holding how "
            "many years old each channel is as of 2026. Return Youtuber, "
            "created_year and channel_age_years."
        ),
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: _has_col(df, "channelage"),
    ),
    AnalysisPrompt(
        id="op_date_filter_year",
        title="Filter on a year",
        prompt="Return only the channels created in or after the year 2015.",
        tier=Tier.OPERATION, category="Date & Time Handling",
        verify=lambda df: (
            (v := _col(df, "createdyear")) is not None
            and (pd.to_numeric(v, errors="coerce").dropna() >= 2015).all()
        ),
    ),

    # --- Arithmetic ---------------------------------------------------------
    AnalysisPrompt(
        id="op_arith_derived_ratio",
        title="Derive a ratio column",
        prompt=(
            "Add a column called views_per_upload equal to video views divided by "
            "uploads, and return Youtuber, uploads, video views and views_per_upload."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        verify=lambda df: _has_col(df, "viewsperupload"),
    ),
    AnalysisPrompt(
        id="op_arith_earnings_midpoint",
        title="Average of two columns",
        prompt=(
            "Add a column called avg_yearly_earnings that is the midpoint of "
            "lowest_yearly_earnings and highest_yearly_earnings, and return "
            "Youtuber with that column."
        ),
        tier=Tier.OPERATION, category="Arithmetic Operations",
        verify=lambda df: _has_col(df, "avgyearlyearnings"),
    ),

    # --- Filtering Logic ----------------------------------------------------
    AnalysisPrompt(
        id="op_filter_threshold",
        title="Numeric threshold",
        prompt="Return the channels with more than 50000000 subscribers.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (v := _col(df, "subscribers")) is not None
            and (pd.to_numeric(v, errors="coerce").dropna() > 50_000_000).all()
        ),
    ),
    AnalysisPrompt(
        id="op_filter_compound",
        title="Two conditions combined",
        prompt=(
            "Return the channels from India that have more than 20000000 subscribers."
        ),
        tier=Tier.OPERATION, category="Filtering Logic",
        # Country is a filter here, not an output column, and the prompt never
        # asked for it back -- requiring it failed a correct answer that
        # returned Youtuber and subscribers. Check the condition that IS
        # observable, and check Country only when it happens to be returned.
        verify=lambda df: (
            (s := _col(df, "subscribers")) is not None
            and (pd.to_numeric(s, errors="coerce").dropna() > 20_000_000).all()
            and ((c := _col(df, "country")) is None
                 or c.dropna().astype(str).str.contains("india", case=False).all())
        ),
    ),
    AnalysisPrompt(
        id="op_filter_isin",
        title="Membership filter",
        prompt="Return only the channels whose Country is India, Brazil or Japan.",
        tier=Tier.OPERATION, category="Filtering Logic",
        verify=lambda df: (
            (c := _col(df, "country")) is not None
            and c.dropna().astype(str).str.lower().isin(
                {"india", "brazil", "japan"}).all()
        ),
    ),

    # --- Data Masking -------------------------------------------------------
    AnalysisPrompt(
        id="op_mask_channel_name",
        title="Mask a text column",
        prompt=(
            "Mask the Youtuber names so only the first two characters are visible "
            "and the rest are replaced with asterisks. Return Youtuber and subscribers."
        ),
        tier=Tier.OPERATION, category="Data Masking",
        # `.any()` was not enough: one real channel name already contains an
        # asterisk, so an untouched column passed. Masking applies to every row,
        # so require the overwhelming majority -- not all, since a one or two
        # character name has nothing left to mask.
        verify=lambda df: (
            (v := _col(df, "youtuber")) is not None
            and (names := v.dropna().astype(str)).size > 0
            and names.str.contains(r"\*").mean() > 0.8
        ),
    ),

    # --- UDF ----------------------------------------------------------------
    AnalysisPrompt(
        id="op_udf_size_band",
        title="Custom bucketing function",
        prompt=(
            "Classify each channel as Mega if it has over 50000000 subscribers, "
            "Large if over 20000000, otherwise Standard. Put the label in a column "
            "called size_band and return Youtuber, subscribers and size_band."
        ),
        tier=Tier.OPERATION, category="UDF",
        verify=lambda df: (
            (v := _col(df, "sizeband")) is not None
            and set(v.dropna().astype(str).str.lower().unique())
            <= {"mega", "large", "standard"}
        ),
    ),

    # --- Aggregation --------------------------------------------------------
    AnalysisPrompt(
        id="op_agg_multi_metric",
        title="Several aggregates in one group-by",
        prompt=(
            "For each Country return the number of channels, the total subscribers "
            "and the average video views."
        ),
        tier=Tier.OPERATION, category="Aggregation",
        # 49 distinct countries against ~995 input rows.
        verify=lambda df: (
            _has_col(df, "country") and 1 < len(df) <= 60 and len(df.columns) >= 4
        ),
    ),

    # --- Joins --------------------------------------------------------------
    AnalysisPrompt(
        id="op_join_country_totals",
        title="Join an aggregate back onto the rows",
        prompt=(
            "Compute the total subscribers per Country, then join that total back "
            "onto every channel row. Return Youtuber, Country, subscribers and the "
            "country total in a column called country_total_subscribers."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "countrytotal") and _has_col(df, "youtuber"),
    ),
    AnalysisPrompt(
        id="op_join_share_of_country",
        title="Join then derive a share",
        prompt=(
            "For each channel work out what share of its country's total video "
            "views it accounts for. Return Youtuber, Country and a column called "
            "share_of_country_views."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "share"),
    ),
    AnalysisPrompt(
        id="op_join_category_rank",
        title="Join a group rank back onto the rows",
        prompt=(
            "Rank channels by subscribers within their own category, join that rank "
            "back onto each row, and return Youtuber, category, subscribers and a "
            "column called rank_in_category."
        ),
        tier=Tier.OPERATION, category="Joins",
        verify=lambda df: _has_col(df, "rankincategory"),
    ),
]


#: Everything, in the order the suite reports it.
ANALYSIS_PROMPTS: List[AnalysisPrompt] = (
    SINGLE_PROMPTS + MULTI_PROMPTS + OPERATION_PROMPTS + TRANSFER_PROMPTS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate_dataset_paths() -> List[Path]:
    paths = []
    env_path = os.environ.get("YOUTUBE_STATS_CSV")
    if env_path:
        paths.append(Path(env_path))
    paths.append(_downloads_dir() / "Global_YouTube_Statistics.csv")
    paths.append(REPO_ROOT / "app" / "sample_data" / "Global_YouTube_Statistics.csv")
    return paths


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    for path in _candidate_dataset_paths():
        if path.exists():
            LOGGER.info("Using dataset: %s", path)
            return path
    pytest.skip(
        "Global_YouTube_Statistics.csv not found. Set YOUTUBE_STATS_CSV or place the "
        "file in app/sample_data/."
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
        user_id="test_youtube_analysis_user",
        session_id=f"test_youtube_analysis_{entry.id}",
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
        dataset_id=f"ds_youtube_{entry.id}",
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
    return tmp_path_factory.mktemp("youtube_analysis_outputs")


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
_TRANSFER_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.expect is Expect.TRANSFER]
_OPERATION_PROMPTS = [p for p in ANALYSIS_PROMPTS if p.verify is not None]

_PARAMS = _params(_TABLE_PROMPTS)
_PARAMS_ALL = _params(ANALYSIS_PROMPTS)
_PARAMS_ANSWER = _params(_ANSWER_PROMPTS)
_PARAMS_REFUSAL = _params(_REFUSAL_PROMPTS)
_PARAMS_TRANSFER = _params(_TRANSFER_PROMPTS)
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
    because it routed to the DTA. Asserting a plan for those failed five prompts
    for behaving exactly as designed.
    """
    result = prompt_run
    state = result["state"]
    assert result["error"] is None, (
        f"Graph raised {result['error']!r}{_diagnostics(result)}"
    )
    assert state.get("plan"), f"Planner produced no plan{_diagnostics(result)}"


# ---------------------------------------------------------------------------
# Tests — visualization charts (bugfix/visualization_charts)
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
        # A chart pointing at a column the result table does not have renders blank
        # in the UI — the failure mode this branch exists to catch.
        assert str(chart["reason"]).strip(), (
            f"Chart {chart['id']!r} has an empty reason{_diagnostics(result)}"
        )

    ranks = sorted(chart["rank"] for chart in charts)
    assert ranks == list(range(1, len(charts) + 1)), (
        f"Chart ranks are not contiguous 1..N: {ranks}{_diagnostics(result)}"
    )


@_PARAMS
def test_chart_fields_exist_in_result_table(prompt_run):
    """Chart encodings must reference columns that are actually in the output."""
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
# are the ones a short conversational ask actually produces, and neither was
# covered: `channel_age_success` was being counted as a suite failure for
# correctly declining to invent a column.
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
    assert any(term in lowered for term in ("channel", "youtub", "subscriber")), (
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
        r"(?:key|value|type|attribute|index|unicode\w*)error\s*:",
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
#
# Same dataset, moved instead of analysed. Kept in this file with the analysis
# prompts because they share the schema: the transfers below filter on the same
# `subscribers` column the analysis prompts rank by, so one schema change breaks
# both together rather than leaving a transfer suite asserting against a dataset
# nobody uses any more.
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

    The earlier version demanded the destination unconditionally, and failed on
    replies like "Source avaloka-test-user-filestore not found" -- which is the
    RIGHT answer in a session with no registered connections. What actually
    matters is that the user is never left without either a destination or a
    reason, so both are accepted and a vague reply is not.
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

    Every other assertion in this file is satisfied by a pipeline that ignores
    the instruction and returns the input unchanged: code was generated, it
    executed, rows came back. Only this one can tell a rename that happened from
    a rename that did not.
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

    Collapsing "rename this column" into a one-cell answer is a real failure
    mode and one the row-count assertion alone lets through.
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
