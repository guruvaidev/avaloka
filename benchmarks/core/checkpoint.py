"""Durable, append-only trial execution and evaluation journal."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Optional

from .. import __version__
from .config import RunConfig
from .task import (BenchmarkTask, FailureKind, MissionOutcome, TrialResult,
                   classify_failure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, FailureKind):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _task_fingerprint(task: BenchmarkTask) -> str:
    payload = {
        "id": task.id,
        "track": task.track,
        "mission": task.mission,
        "dataset": task.dataset.id,
        "dataset_fingerprint": task.dataset.fingerprint,
        "split": task.split.value,
        "tags": task.tags,
        "params": task.params,
        "reference": task.reference,
    }
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _outcome_record(outcome: MissionOutcome) -> Dict[str, Any]:
    return _json_safe(asdict(outcome))


_OUTCOME_FIELDS = {field.name for field in dataclass_fields(MissionOutcome)}


def _outcome_from_record(record: Dict[str, Any]) -> MissionOutcome:
    # Drop keys this harness version does not know about so a journal written by a
    # newer build stays loadable instead of aborting the run.
    payload = {key: value for key, value in record.items() if key in _OUTCOME_FIELDS}
    raw_kind = payload.get("error_kind")
    if raw_kind:
        try:
            payload["error_kind"] = FailureKind(raw_kind)
        except ValueError:
            payload["error_kind"] = classify_failure(payload.get("error"))
    return MissionOutcome(**payload)


class TrialJournal:
    """JSONL journal that resumes work only when task and config identities match."""

    schema_version = "benchmark-trial-journal-v1"

    def __init__(self, path: str | Path, *, suite: str, resume: bool = True) -> None:
        self.path = Path(path)
        self.suite = suite
        self.resume = resume
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._executions: Dict[str, MissionOutcome] = {}
        self._evaluated: Dict[str, TrialResult] = {}
        #: Journal lines this build could not read; their trials re-run.
        self.skipped_records = 0
        if resume and self.path.is_file():
            self._load()

    def key(self, task: BenchmarkTask, config: RunConfig, trial_index: int) -> str:
        raw = json.dumps({
            "schema": self.schema_version,
            "harness_version": __version__,
            "suite": self.suite,
            "task": _task_fingerprint(task),
            "config": config.fingerprint(),
            "trial": trial_index,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def execution(self, task: BenchmarkTask, config: RunConfig,
                  trial_index: int) -> Optional[MissionOutcome]:
        return self._executions.get(self.key(task, config, trial_index))

    def evaluated(self, task: BenchmarkTask, config: RunConfig,
                  trial_index: int) -> Optional[TrialResult]:
        return self._evaluated.get(self.key(task, config, trial_index))

    def record_execution(self, task: BenchmarkTask, config: RunConfig,
                         trial_index: int, outcome: MissionOutcome) -> None:
        key = self.key(task, config, trial_index)
        event = self._base_event(key, task, config, trial_index)
        event.update({"stage": "executed", "outcome": _outcome_record(outcome)})
        self._append(event)
        self._executions[key] = outcome

    def record_evaluated(self, task: BenchmarkTask, config: RunConfig,
                         trial: TrialResult) -> None:
        key = self.key(task, config, trial.trial_index)
        event = self._base_event(key, task, config, trial.trial_index)
        event.update({
            "stage": "evaluated",
            "quality": _json_safe(trial.quality),
            "outcome": _outcome_record(trial.outcome),
        })
        self._append(event)
        self._executions[key] = trial.outcome
        self._evaluated[key] = trial

    def _base_event(self, key: str, task: BenchmarkTask, config: RunConfig,
                    trial_index: int) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "harness_version": __version__,
            "key": key,
            "suite": self.suite,
            "task_id": task.id,
            "track": task.track,
            "task_fingerprint": _task_fingerprint(task),
            "config_id": config.id,
            "config_fingerprint": config.fingerprint(),
            "trial_index": trial_index,
        }

    def _append(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, default=str,
                          separators=(",", ":")) + "\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self) -> None:
        """Load resumable records, skipping any line this build cannot read.

        A journal exists so an expensive run survives a crash. Refusing to load it
        because of one unreadable line — a torn final write from ``kill -9`` or a
        record written by a different harness version — would destroy exactly the
        work it is meant to protect. Unreadable lines are counted in
        ``skipped_records`` and ignored; their trials simply re-run.
        """
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                key = event["key"]
                outcome = _outcome_from_record(event["outcome"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.skipped_records += 1
                continue
            self._executions[key] = outcome
            if event.get("stage") == "evaluated":
                try:
                    trial = TrialResult(
                        task_id=event["task_id"],
                        track=event["track"],
                        config_id=event["config_id"],
                        trial_index=int(event["trial_index"]),
                        quality={name: float(value)
                                 for name, value in event.get("quality", {}).items()},
                        outcome=outcome,
                    )
                except (KeyError, TypeError, ValueError):
                    self.skipped_records += 1
                    continue
                self._evaluated[key] = trial
