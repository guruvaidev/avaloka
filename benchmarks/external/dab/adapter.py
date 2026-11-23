"""Official DataAgentBench task loading, isolated evaluation, and run policy."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, List, Tuple

from ...core.config import Budget, RunConfig
from ...core.registry import register_suite
from ...core.task import BenchmarkTask, DatasetSpec, Split

CANONICAL_DATASETS = (
    "agnews", "bookreview", "crmarenapro", "DEPS_DEV_V1", "GITHUB_REPOS",
    "googlelocal", "music_brainz_20k", "PANCANCER_ATLAS", "PATENTS",
    "stockindex", "stockmarket", "yelp",
)
_DATABASE_TYPE = re.compile(r"^\s*db_type:\s*([a-zA-Z0-9_-]+)", re.MULTILINE)


class DABNotProvisioned(RuntimeError):
    pass


def _root() -> Path:
    value = os.environ.get("DAB_ROOT")
    if not value:
        raise DABNotProvisioned(
            "DAB_ROOT is not set. Clone the official DataAgentBench repository, run "
            "download.sh, then export DAB_ROOT=/absolute/path/to/DataAgentBench."
        )
    root = Path(value).expanduser().resolve()
    if not (root / "SUBMISSION_RUBRIC.md").is_file():
        raise DABNotProvisioned(f"{root} is not an official DataAgentBench checkout")
    return root


def _answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for key in ("answer", "last_message", "output_json", "output_file_data"):
            if answer.get(key) not in (None, ""):
                value = answer[key]
                return value if isinstance(value, str) else json.dumps(value, default=str)
    return json.dumps(answer, default=str) if answer is not None else ""


@contextlib.contextmanager
def _validator_import_path(root: Path) -> Iterator[None]:
    """Expose official validator dependencies only while the parent evaluates."""
    value = str(root)
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _official_evaluator(validate_path: Path, root: Path):
    def evaluate(_task: BenchmarkTask, answer: Any):
        name = f"dab_validate_{validate_path.parent.parent.name}_{validate_path.parent.name}"
        spec = importlib.util.spec_from_file_location(name, validate_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load DAB validator {validate_path}")
        module = importlib.util.module_from_spec(spec)
        with _validator_import_path(root):
            spec.loader.exec_module(module)
            passed, reason = module.validate(_answer_text(answer))
        return {
            "pass": float(bool(passed)),
            "score": float(bool(passed)),
            "validator_reason": str(reason),
        }
    return evaluate


def _manifest_fingerprint(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def load_dab_tasks(*, use_hints: bool | None = None) -> List[BenchmarkTask]:
    root = _root()
    if use_hints is None:
        use_hints = os.environ.get("DAB_USE_HINTS", "0").lower() in {"1", "true", "yes"}
    tasks: List[BenchmarkTask] = []
    for dataset in CANONICAL_DATASETS:
        dataset_dir = root / f"query_{dataset}"
        if not dataset_dir.is_dir():
            raise DABNotProvisioned(f"missing official dataset directory: {dataset_dir}")
        # Upstream appends the hint file to the description (run_agent.py:
        # `db_description += "\n\n" + hints`). The hint file holds only the
        # hints, so using it in place of the description removed every schema
        # and database name from hinted runs.
        description_path = dataset_dir / "db_description.txt"
        config_path = dataset_dir / "db_config.yaml"
        description = description_path.read_text(encoding="utf-8").strip()
        if use_hints:
            hints = (dataset_dir / "db_description_withhint.txt").read_text(encoding="utf-8")
            description += "\n\n" + hints.strip()
        config_text = config_path.read_text(encoding="utf-8")
        database_types = tuple(sorted(set(_DATABASE_TYPE.findall(config_text))))
        fingerprint = _manifest_fingerprint(description, config_text)
        for query_dir in sorted(
            dataset_dir.glob("query[0-9]*"),
            key=lambda item: int(item.name.removeprefix("query")),
        ):
            query_path = query_dir / "query.json"
            validate_path = query_dir / "validate.py"
            if not query_path.is_file() or not validate_path.is_file():
                continue
            query_id = query_dir.name.removeprefix("query")
            mission = json.loads(query_path.read_text(encoding="utf-8"))
            validator_fingerprint = hashlib.sha256(
                validate_path.read_bytes()
            ).hexdigest()[:20]
            tasks.append(BenchmarkTask(
                id=f"dab:{dataset}:{query_id}",
                track="dab",
                mission=mission,
                dataset=DatasetSpec(
                    id=dataset,
                    description=description,
                    fingerprint=fingerprint,
                ),
                evaluator=_official_evaluator(validate_path, root),
                split=Split.TEST,
                params={
                    "dab_root": str(root),
                    "dataset": dataset,
                    "query": query_id,
                    "dataset_dir": str(dataset_dir),
                    "db_description": description,
                    "db_config_path": str(config_path),
                    "database_types": database_types,
                    "use_hints": use_hints,
                    "_validator_fingerprint": validator_fingerprint,
                },
                tags=(dataset, *database_types, "official",
                      "hints" if use_hints else "no_hints"),
            ))
    if len(tasks) != 54:
        raise DABNotProvisioned(f"expected 54 canonical DAB queries, found {len(tasks)}")
    return tasks


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise DABNotProvisioned(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise DABNotProvisioned(f"{name} must be numeric") from exc


def _configs() -> List[RunConfig]:
    return [RunConfig(
        id="dab_avaloka_native",
        sut="avaloka_dab",
        llm=os.environ.get("AVALOKA_DAB_MODEL", "gemini-2.5-flash"),
        sampling_mode="entire_dataset",
        environment="dab-local",
        trials=5,
        internet_allowed=False,
        budget=Budget(
            max_cost_usd=_env_float("AVALOKA_DAB_MAX_COST_USD", 2.0),
            max_latency_s=_env_float("AVALOKA_DAB_MAX_LATENCY_S", 600.0),
            max_tokens=_env_int("AVALOKA_DAB_MAX_TOKENS", 2_000_000),
        ),
        extra={
            "agent_command_env": "AVALOKA_DAB_AGENT_CMD",
            "protocol_version": "dab-command-v2",
            "max_iterations": _env_int("AVALOKA_DAB_MAX_ITERATIONS", 25),
            "tool_timeout_s": _env_int("AVALOKA_DAB_TOOL_TIMEOUT", 120),
            "python_timeout_s": _env_int("AVALOKA_DAB_PYTHON_TIMEOUT", 120),
            "max_output_tokens": _env_int("AVALOKA_DAB_MAX_OUTPUT_TOKENS", 8_192),
            "max_prompt_tool_result_bytes": _env_int(
                "AVALOKA_DAB_MAX_PROMPT_TOOL_RESULT_BYTES", 100_000
            ),
        },
    )]


@register_suite(
    "dab",
    kind="external",
    description="Official Berkeley Data Agent Benchmark: 54 queries, 12 datasets, "
                "official validators, 5 trials/query (requires DAB_ROOT).",
)
def build_dab() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    return load_dab_tasks(), _configs()
