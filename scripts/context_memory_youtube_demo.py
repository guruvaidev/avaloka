"""
Local demo: show how Avaloka context memory accumulates across YouTube prompts.

This script is intentionally local/demo-only. It does not require live Redis,
Chroma, Milvus, Groq, or LangChain services. Missing framework packages are
stubbed so the real MemoryOrchestrator logic can be exercised with in-process
fallback stores and a deterministic fake memory LLM.

Run:
    python scripts/context_memory_youtube_demo.py
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Resolve the dataset from the first CLI arg or YOUTUBE_CSV_PATH; when neither
# is given (or the file is missing) the demo falls back to a small synthetic
# dataset so it runs on any machine without external files.
CSV_PATH = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("YOUTUBE_CSV_PATH", "")).strip()
SESSION_ID = "youtube_context_memory_demo"


def _install_local_stubs() -> None:
    """Provide small stubs for packages not installed in this local environment."""
    if "langchain_core" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        langchain_core.__path__ = []
        messages = types.ModuleType("langchain_core.messages")

        class _Message:
            def __init__(self, content="", **kwargs):
                self.content = content
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class SystemMessage(_Message):
            pass

        class HumanMessage(_Message):
            type = "human"

        class AIMessage(_Message):
            type = "ai"

        messages.SystemMessage = SystemMessage
        messages.HumanMessage = HumanMessage
        messages.AIMessage = AIMessage
        messages.BaseMessage = _Message
        langchain_core.messages = messages
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.messages"] = messages

    if "langchain_groq" not in sys.modules:
        langchain_groq = types.ModuleType("langchain_groq")
        langchain_groq.ChatGroq = MagicMock
        sys.modules["langchain_groq"] = langchain_groq


class _FakeMemoryResponse:
    def __init__(self, payload):
        self.content = json.dumps(payload)


class FakeMemoryLLM:
    """Deterministic memory extractor for this demo."""

    def invoke(self, messages):
        text = messages[-1].content.lower()
        hints = []

        if "fastest-growing" in text or "subscribers_for_last_30_days" in text:
            hints.append(
                "Growth momentum = subscribers_for_last_30_days / subscribers."
            )
        if "country" in text and "average subscribers" in text:
            hints.append(
                "Country opportunity = aggregate subscribers, views, and channel count by Country."
            )
        if "earnings" in text and "categor" in text:
            hints.append(
                "Category earnings = compare yearly earnings across category groups."
            )
        if "underserved" in text or "niche" in text:
            hints.append(
                "Underserved niche = category + channel_type + Country with high views per channel."
            )
        if "country and category" in text or "country, category" in text:
            hints.append(
                "Country-category performance = rank Country/category pairs by average views."
            )
        if "earnings-per-subscriber" in text or "earnings per subscriber" in text:
            hints.append(
                "Earnings efficiency = yearly earnings divided by subscribers."
            )
        if "channel_type" in text or "channel type" in text:
            hints.append(
                "Channel type comparison = group by channel_type and compare reach metrics."
            )
        if "population" in text or "unemployment" in text or "urban_population" in text:
            hints.append(
                "Macro analysis = compare subscribers/views per capita with population and labor indicators."
            )
        if "age_bucket" in text or "channel age" in text:
            hints.append(
                "Channel age success = bucket created_year and summarize subscribers/views."
            )
        if "top 5 channels" in text or "top creators" in text:
            hints.append(
                "Top creators = top channels per Country by subscribers and video views."
            )
        if "view efficiency" in text or "uploads" in text:
            hints.append("View efficiency = video views / uploads, with zero-safe division.")

        if not hints:
            hints.append(
                "YouTube analytics context uses Country, category, channel_type, subscribers, and video views."
            )

        return _FakeMemoryResponse(
            {
                "new_hints": hints,
                "logic_signature": (
                    "Use pandas groupby, numeric-safe ratios, top-N rankings, "
                    "and concise sorted DataFrames for YouTube analytics."
                ),
            }
        )


def _read_csv_with_encoding(path: str) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path), "utf-8"
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1"), "latin-1"


def _synthetic_youtube_dataframe() -> pd.DataFrame:
    """Small in-process dataset so the demo runs without an external CSV."""
    return pd.DataFrame(
        {
            "Youtuber": [f"Channel {i}" for i in range(1, 11)],
            "category": ["Music", "Gaming", "Education", "News", "Comedy"] * 2,
            "Country": ["US", "IN", "BR", "GB", "KR", "US", "IN", "JP", "DE", "FR"],
            "subscribers": [250, 180, 140, 120, 95, 90, 80, 70, 60, 50],
            "subscribers_for_last_30_days": [12, 3, 8, 1, 5, 2, 4, 1, 3, 2],
            "created_year": [2010, 2012, 2015, 2011, 2016, 2013, 2014, 2017, 2018, 2019],
        }
    )


def _load_demo_dataframe() -> tuple[pd.DataFrame, str, str]:
    """Return (df, encoding, dataset_name), reading CSV_PATH if it exists."""
    if CSV_PATH and os.path.exists(CSV_PATH):
        df, encoding = _read_csv_with_encoding(CSV_PATH)
        return df, encoding, os.path.basename(CSV_PATH)
    return _synthetic_youtube_dataframe(), "utf-8", "synthetic_youtube_statistics.csv"


def _schema_payload(df: pd.DataFrame, encoding: str, dataset: str) -> dict:
    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "encoding": encoding,
        "columns": list(df.columns),
        "numeric_columns": [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ],
        "categorical_columns": [
            col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
        ],
    }


PROMPTS = [
    (
        "Growth / momentum",
        "Identify the fastest-growing channels by comparing subscribers_for_last_30_days "
        "to total subscribers. List the top 20 channels and include their Youtuber, "
        "category, Country, subscribers, and subscribers_for_last_30_days.",
    ),
    (
        "Country-level opportunity",
        "For each Country, compute the total subscribers, total video views, and number "
        "of channels. Rank countries by average subscribers per channel and highlight "
        "the top 10 and bottom 10 countries.",
    ),
    (
        "Category earnings",
        "Compare earnings across different categories.",
    ),
    (
        "Underserved niches",
        "Treat each row as a channel and define a niche as the combination of category, "
        "channel_type, and Country. For each niche, compute number of channels, total "
        "subscribers, and total video views. Identify niches that have few channels but "
        "high total video views per channel and return the top 20 underserved niches.",
    ),
    (
        "Category performance by country",
        "For each Country and category, calculate the average video views and average "
        "subscribers per channel. Return a table of the top 15 Country, category pairs "
        "ranked by average video views per channel.",
    ),
    (
        "Earnings vs audience size",
        "Using lowest_yearly_earnings, highest_yearly_earnings, and subscribers, "
        "estimate an earnings-per-subscriber range for each channel. Find the 20 "
        "channels with the highest estimated earnings per subscriber and the 20 with "
        "the lowest.",
    ),
    (
        "Channel type comparison",
        "Compare different channel_type values globally: for each channel_type, compute "
        "number of channels, total subscribers, total video views, and average "
        "video_views_for_the_last_30_days. Rank channel_type by average video views "
        "per channel.",
    ),
    (
        "Macroeconomics vs YouTube",
        "Aggregate by Country and compute total video views and total subscribers. Then, "
        "join this with Population, Urban_population, and Unemployment rate to analyze "
        "how YouTube penetration, subscribers per capita and views per capita, varies "
        "with these macro indicators.",
    ),
    (
        "Channel age and success",
        "Return a summary DataFrame with the columns: Age_bucket, Avg_subscribers, "
        "Avg_video_views, Channel_count, sorted in the logical bucket order.",
    ),
    (
        "Top creators per country",
        "For each Country, identify the top 5 channels by subscribers and by video "
        "views. Return a combined table showing Country, Youtuber, subscribers, "
        "video views, category, and channel_type.",
    ),
    (
        "View efficiency",
        "For each channel, compute a simple view efficiency metric as video views "
        "divided by uploads. Find the top 30 channels with the highest view efficiency, "
        "and show their Youtuber, Country, category, channel_type, uploads, video views, "
        "and the computed efficiency.",
    ),
]


def main() -> None:
    _install_local_stubs()

    import app.services.memory_plane as memory_plane
    from app.services.memory_plane import _orchestrator

    df, encoding, dataset_id = _load_demo_dataframe()
    data_source = CSV_PATH if (CSV_PATH and os.path.exists(CSV_PATH)) else dataset_id

    memory_plane.memory_llm = FakeMemoryLLM()
    _orchestrator._past_queries_store.clear()
    _orchestrator._session_hints_store.clear()

    _orchestrator.redis.connect()
    _orchestrator.redis.set_schema(
        "default",
        dataset_id,
        _schema_payload(df, encoding, dataset_id),
        ttl_seconds=86400,
    )

    state = {
        "session_id": SESSION_ID,
        "data_source_location": data_source,
        "active_mcp_servers": [],
    }

    print("\nCONTEXT MEMORY DEMO")
    print("=" * 80)
    print(f"Dataset: {data_source}")
    print(f"Rows: {len(df)}")
    print(f"Encoding used: {encoding}")
    print(f"Columns: {', '.join(df.columns)}")
    print("=" * 80)

    for index, (title, prompt) in enumerate(PROMPTS, start=1):
        result = _orchestrator.retrieve_memory(prompt, SESSION_ID, dataset_id, state)

        print(f"\nSTEP {index}: {title}")
        print("-" * 80)
        print(f"Memory unavailable: {result['memory_context_unavailable']}")
        print(f"Accumulated hints stored: {len(result['accumulated_memory_hints'])}")
        print("Planner receives top-3 memory_hints:")

        for hint_index, hint in enumerate(result["memory_hints"], start=1):
            compact = hint.replace("\n", " ")
            if len(compact) > 240:
                compact = compact[:237] + "..."
            print(f"  {hint_index}. {compact}")

    print("\nFINAL SESSION MEMORY STORE")
    print("=" * 80)
    for hint_index, hint in enumerate(
        _orchestrator._session_hints_store.get(SESSION_ID, []),
        start=1,
    ):
        print(f"{hint_index}. {hint}")


if __name__ == "__main__":
    main()
