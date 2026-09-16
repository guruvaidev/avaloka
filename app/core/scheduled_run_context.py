"""Execution-local bridge between scheduled Celery runs and nested Ray jobs.

Celery owns the user-visible schedule/run identity, while Ray creates a second
submission identity inside the training call. A context variable lets the Ray
trainer report that identity without putting callbacks or internal bookkeeping
fields into the serialised ETL state.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

RayJobCallback = Callable[[str, str, Mapping[str, str]], None]


@dataclass(frozen=True)
class ScheduledRunContext:
    schedule_id: str
    execution_id: str
    ray_job_callback: Optional[RayJobCallback] = None


_CURRENT_RUN: ContextVar[Optional[ScheduledRunContext]] = ContextVar(
    "avaloka_scheduled_run",
    default=None,
)


@contextmanager
def scheduled_run_context(
    schedule_id: str,
    execution_id: str,
    *,
    ray_job_callback: Optional[RayJobCallback] = None,
) -> Iterator[None]:
    """Make scheduled-run identity available for the duration of one task."""
    token = _CURRENT_RUN.set(
        ScheduledRunContext(
            schedule_id=schedule_id,
            execution_id=execution_id,
            ray_job_callback=ray_job_callback,
        )
    )
    try:
        yield
    finally:
        _CURRENT_RUN.reset(token)


def notify_ray_job_submitted(metadata: Mapping[str, str]) -> bool:
    """Attach a newly submitted Ray job to the active scheduled run, if any."""
    current = _CURRENT_RUN.get()
    if current is None or current.ray_job_callback is None:
        return False
    try:
        current.ray_job_callback(
            current.schedule_id,
            current.execution_id,
            metadata,
        )
    except Exception:
        # Tracking must never turn a successfully submitted training job into a
        # failed one. The final Celery result remains the fallback source.
        logger.warning("Could not persist Ray job submission metadata", exc_info=True)
        return False
    return True
