"""
Dataset registry for the Avaloka benchmark.

Each entry pins a *real* Kaggle dataset (so the benchmark exercises real-world,
messy data) together with a self-contained synthetic schema used to generate an
equivalent offline dataset when Kaggle credentials / network are unavailable.

The synthetic fallback intentionally mirrors the real dataset's columns, dtypes
and target so the same task spec and scoring thresholds apply to both paths --
this mirrors how the existing tests in ``tests/`` fall back to tmp CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class TaskKind(str, Enum):
    """The ML problem the dataset's target supports."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"
    NONE = "none"  # pure data-engineering dataset, no modelling target


@dataclass(frozen=True)
class ColumnSpec:
    """One column in a synthetic schema."""

    name: str
    dtype: str  # "int" | "float" | "category" | "string" | "datetime"
    # For category columns: the discrete levels. For numeric: (low, high).
    categories: Optional[List[str]] = None
    low: float = 0.0
    high: float = 1.0
    null_fraction: float = 0.0


@dataclass(frozen=True)
class DatasetSpec:
    """A benchmark dataset: a real Kaggle ref + an offline-equivalent schema."""

    key: str
    title: str
    kaggle_ref: str  # "owner/dataset-slug" for kagglehub.dataset_download
    primary_file: str  # the csv inside the kaggle archive we benchmark on
    task_kind: TaskKind
    target: Optional[str]
    # Columns the synthetic generator emits (must include `target`).
    synthetic_columns: List[ColumnSpec] = field(default_factory=list)
    # Approx rows for the synthetic fallback. Real Kaggle files are used as-is.
    synthetic_rows: int = 5_000
    # Natural-language brief handed to the planner sub-agent.
    brief: str = ""
    # Optional post-load hook to coerce the real Kaggle frame into the schema
    # the tasks expect (rename columns, pick subset, etc.).
    normalize: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The registry. Kaggle refs are real, widely-used datasets; the synthetic
# schemas are faithful enough that DE/DS tasks behave identically offline.
# --------------------------------------------------------------------------- #

REGISTRY: Dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    REGISTRY[spec.key] = spec
    return spec


register(
    DatasetSpec(
        key="titanic",
        title="Titanic - Machine Learning from Disaster",
        kaggle_ref="yasserh/titanic-dataset",
        primary_file="Titanic-Dataset.csv",
        task_kind=TaskKind.BINARY,
        target="Survived",
        brief=(
            "Predict passenger survival. Engineer features from name titles, "
            "family size and cabin deck; handle missing Age/Embarked."
        ),
        synthetic_rows=891,
        synthetic_columns=[
            ColumnSpec("PassengerId", "int", low=1, high=891),
            ColumnSpec("Survived", "category", categories=["0", "1"]),
            ColumnSpec("Pclass", "category", categories=["1", "2", "3"]),
            ColumnSpec("Sex", "category", categories=["male", "female"]),
            ColumnSpec("Age", "float", low=0.4, high=80.0, null_fraction=0.2),
            ColumnSpec("SibSp", "int", low=0, high=8),
            ColumnSpec("Parch", "int", low=0, high=6),
            ColumnSpec("Fare", "float", low=0.0, high=512.0),
            ColumnSpec(
                "Embarked", "category", categories=["S", "C", "Q"], null_fraction=0.01
            ),
        ],
        tags=["classification", "tabular", "missing-values", "feature-engineering"],
    )
)

register(
    DatasetSpec(
        key="telco_churn",
        title="Telco Customer Churn",
        kaggle_ref="blastchar/telco-customer-churn",
        primary_file="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        task_kind=TaskKind.BINARY,
        target="Churn",
        brief=(
            "Predict customer churn. Encode categorical contract/payment fields, "
            "coerce TotalCharges to numeric, balance the target."
        ),
        synthetic_rows=7_043,
        synthetic_columns=[
            ColumnSpec("customerID", "string"),
            ColumnSpec("gender", "category", categories=["Male", "Female"]),
            ColumnSpec("SeniorCitizen", "category", categories=["0", "1"]),
            ColumnSpec("tenure", "int", low=0, high=72),
            ColumnSpec("MonthlyCharges", "float", low=18.0, high=120.0),
            ColumnSpec("TotalCharges", "float", low=0.0, high=8700.0, null_fraction=0.02),
            ColumnSpec(
                "Contract",
                "category",
                categories=["Month-to-month", "One year", "Two year"],
            ),
            ColumnSpec(
                "PaymentMethod",
                "category",
                categories=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card",
                ],
            ),
            ColumnSpec("Churn", "category", categories=["Yes", "No"]),
        ],
        tags=["classification", "tabular", "imbalanced", "type-coercion"],
    )
)

register(
    DatasetSpec(
        key="house_prices",
        title="House Prices - Advanced Regression",
        kaggle_ref="yasserh/housing-prices-dataset",
        primary_file="Housing.csv",
        task_kind=TaskKind.REGRESSION,
        target="price",
        brief=(
            "Predict house sale price (regression). Scale skewed numeric features, "
            "one-hot encode furnishing/amenity flags."
        ),
        synthetic_rows=545,
        synthetic_columns=[
            ColumnSpec("price", "float", low=1_750_000, high=13_300_000),
            ColumnSpec("area", "float", low=1650, high=16200),
            ColumnSpec("bedrooms", "int", low=1, high=6),
            ColumnSpec("bathrooms", "int", low=1, high=4),
            ColumnSpec("stories", "int", low=1, high=4),
            ColumnSpec("mainroad", "category", categories=["yes", "no"]),
            ColumnSpec("guestroom", "category", categories=["yes", "no"]),
            ColumnSpec("parking", "int", low=0, high=3),
            ColumnSpec(
                "furnishingstatus",
                "category",
                categories=["furnished", "semi-furnished", "unfurnished"],
            ),
        ],
        tags=["regression", "tabular", "skewed", "scaling"],
    )
)

register(
    DatasetSpec(
        key="nyc_taxi",
        title="NYC Yellow Taxi Trips (DE-heavy)",
        kaggle_ref="elemento/nyc-yellow-taxi-trip-data",
        primary_file="yellow_tripdata.csv",
        task_kind=TaskKind.REGRESSION,
        target="total_amount",
        brief=(
            "Large data-engineering workload: parse timestamps, derive trip "
            "duration/speed, partition by pickup date, aggregate fare by zone."
        ),
        synthetic_rows=50_000,
        synthetic_columns=[
            ColumnSpec("VendorID", "category", categories=["1", "2"]),
            ColumnSpec("passenger_count", "int", low=1, high=6),
            ColumnSpec("trip_distance", "float", low=0.0, high=40.0),
            ColumnSpec("PULocationID", "int", low=1, high=265),
            ColumnSpec("DOLocationID", "int", low=1, high=265),
            ColumnSpec("fare_amount", "float", low=2.5, high=200.0),
            ColumnSpec("tip_amount", "float", low=0.0, high=80.0),
            ColumnSpec("total_amount", "float", low=3.0, high=300.0),
            ColumnSpec("payment_type", "category", categories=["1", "2", "3", "4"]),
        ],
        tags=["data-engineering", "large", "timeseries", "aggregation", "partitioning"],
    )
)


def get(key: str) -> DatasetSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown dataset '{key}'. Known: {sorted(REGISTRY)}"
        ) from exc


def all_keys() -> List[str]:
    return sorted(REGISTRY)
