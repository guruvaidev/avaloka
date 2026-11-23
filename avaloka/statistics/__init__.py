"""Typed deterministic statistical analysis."""

from .core import (StatisticalPlan, StatisticalResult, analytic_mean_interval,
                   bootstrap_mean_interval, compute_statistic)

__all__ = [
    "StatisticalPlan", "StatisticalResult", "analytic_mean_interval", "bootstrap_mean_interval",
    "compute_statistic",
]
