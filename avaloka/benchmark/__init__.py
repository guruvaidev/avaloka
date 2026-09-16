"""The Avaloka benchmark — data-engineering + data-science, source-agnostic.

A defensible way to answer "does the Avaloka CLI actually analyse and train
correctly, from any source?" It exercises the whole stack — source connectors,
workload routing, the firefly swarm, validation, economics and the multi-Avaloka
coordinator — against tasks with *known ground truth*, and scores the outcome
rather than trusting the narration.

Two families of tasks:

* **Data engineering** — ingestion parity across file/database/object-storage,
  workload routing, schema/role inference, data-quality detection, reproducible
  transformation, and *exact* batch convergence.
* **Data science** — model quality beating a naive baseline, target-leakage
  detection, deployment-level gating, economics sanity, correct firefly
  activation, and multi-Avaloka convergence.

Run it::

    python -m avaloka.benchmark run                 # synthetic tasks, all sources
    python -m avaloka.benchmark run --kaggle        # add downloaded Kaggle datasets
"""

from avaloka.benchmark.spec import (BenchmarkTask, CheckResult, DatasetSpec,
                                     Expectations, Scorecard, TaskResult)
from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.suite import data_engineering_tasks, data_science_tasks, all_tasks

__all__ = [
    "BenchmarkTask",
    "CheckResult",
    "DatasetSpec",
    "Expectations",
    "Scorecard",
    "TaskResult",
    "BenchmarkRunner",
    "data_engineering_tasks",
    "data_science_tasks",
    "all_tasks",
]
