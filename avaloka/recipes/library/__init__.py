"""The built recipes. Importing this package registers each one, so
``avaloka.recipes.get("describe")`` works after ``import avaloka.recipes.library``."""

from __future__ import annotations

from ..registry import register
from .ab_test import ABTest
from .anomalies import Anomalies
from .cohort_retention import CohortRetention
from .compare_periods import ComparePeriods
from .correlate import Correlate
from .describe import Describe
from .profile import Profile
from .regress import Regress
from .segment import Segment
from .trend import Trend

PROFILE = register(Profile())
DESCRIBE = register(Describe())
TREND = register(Trend())
COMPARE_PERIODS = register(ComparePeriods())
SEGMENT = register(Segment())
COHORT_RETENTION = register(CohortRetention())
CORRELATE = register(Correlate())
REGRESS = register(Regress())
ANOMALIES = register(Anomalies())
AB_TEST = register(ABTest())

__all__ = ["PROFILE", "DESCRIBE", "TREND", "COMPARE_PERIODS", "SEGMENT", "COHORT_RETENTION", "CORRELATE", "REGRESS",
           "ANOMALIES", "AB_TEST"]
