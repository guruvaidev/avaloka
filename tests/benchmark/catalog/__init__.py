"""Concrete benchmark task catalogs + swarm scenarios."""

from .de_tasks import DE_TASKS
from .ds_tasks import DS_TASKS
from .swarm_scenarios import SWARM_SCENARIOS

ALL_TASKS = DE_TASKS + DS_TASKS

__all__ = ["DE_TASKS", "DS_TASKS", "ALL_TASKS", "SWARM_SCENARIOS"]
