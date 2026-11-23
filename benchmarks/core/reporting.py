"""JSON records, atomic persistence, repository identity, and scorecards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from .. import __version__
from .config import RunConfig
from .task import TaskResult


def repository_metadata(repo: str | Path | None = None) -> Dict[str, Any]:
    """Capture reproducibility metadata without including file contents or secrets."""
    cwd = str(repo) if repo is not None else None

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=cwd, text=True, capture_output=True,
                check=False, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--porcelain")
    commit = git("rev-parse", "HEAD")
    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
    }
    # A container without git, or a worktree whose .git file points outside the
    # mount, records None for every field above and the record loses its
    # provenance. The launcher can assert it through the environment instead;
    # the record then says the value was asserted, not observed.
    if commit is None and os.getenv("AVALOKA_BENCH_GIT_COMMIT"):
        dirty = os.getenv("AVALOKA_BENCH_GIT_DIRTY")
        meta.update({
            "git_commit": os.getenv("AVALOKA_BENCH_GIT_COMMIT"),
            "git_branch": os.getenv("AVALOKA_BENCH_GIT_BRANCH") or None,
            "git_dirty": None if dirty is None else dirty.lower() in ("1", "true", "yes"),
            "git_source": "asserted_by_launcher",
        })
    return meta


def results_to_record(results: List[TaskResult], configs: List[RunConfig],
                      *, suite: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config_by_id = {config.id: config for config in configs}
    per_config: Dict[str, Any] = {}
    for result in results:
        bucket = per_config.setdefault(result.config_id, {
            "config": config_by_id[result.config_id].scorecard_config()
            if result.config_id in config_by_id else {"config_id": result.config_id},
            "tasks": {},
        })
        bucket["tasks"][result.task_id] = {
            "track": result.track,
            "stats": result.stats,
            "trials": [asdict(trial) for trial in result.trials],
        }
    return {
        "harness_version": __version__,
        "suite": suite,
        "meta": meta or {},
        "configs": [config.scorecard_config() for config in configs],
        "results": per_config,
    }


def _atomic_write(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json(record: Dict[str, Any], path: str | Path) -> None:
    _atomic_write(path, json.dumps(record, indent=2, default=_json_default) + "\n")


def write_text(content: str, path: str | Path) -> None:
    _atomic_write(path, content)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    try:
        return asdict(value)
    except TypeError:
        return str(value)


def render_scorecard(record: Dict[str, Any]) -> str:
    """A compact Markdown scorecard: one row per (config, task)."""
    lines: List[str] = []
    lines.append(f"# Scorecard — suite `{record['suite']}` "
                 f"(harness {record['harness_version']})")
    lines.append("")

    # Run-level banner. A sweep that stopped early — budget exhausted, ceiling
    # hit — produces a scorecard whose later configs are absent or all-errored.
    # Without this line the reader assumes they simply performed badly.
    meta = record.get("meta") or {}
    spend, aborted = meta.get("spend") or {}, meta.get("aborted_reason")
    if spend:
        lines.append(f"- Observed spend: **${spend.get('spent_usd', 0):.4f}** over "
                     f"{spend.get('calls', 0)} calls"
                     + (f" · ceiling ${spend['ceiling_usd']:.2f}"
                        if spend.get("ceiling_usd") is not None else " · no ceiling")
                     + (f" · **{spend['unpriced_calls']} unpriced**"
                        if spend.get("unpriced_calls") else ""))
    if aborted:
        lines.append("")
        lines.append(f"> 🛑 **RUN INCOMPLETE — {aborted}**")
        lines.append("> Configs below this point may be missing or all-errored "
                     "because the run STOPPED, not because they scored badly. "
                     "This scorecard must not be published as a model comparison.")
    if spend or aborted:
        lines.append("")

    for config_id, block in record["results"].items():
        cfg = block["config"]
        tstats = [t["stats"] for t in block["tasks"].values()]
        # A config in which EVERY task fully errored was never measured at all.
        # This is the shape budget exhaustion and rate limiting both take, and
        # it is the row that otherwise gets published as "Opus 5 scores zero".
        config_dead = bool(tstats) and all(
            (st.get("n_errors") or 0) >= (st.get("n_trials") or cfg.get("trials") or 1)
            and (st.get("n_errors") or 0) > 0
            for st in tstats)
        lines.append(f"## Config `{config_id}`" + ("  🛑 NOT MEASURED" if config_dead else ""))
        lines.append("")
        if config_dead:
            reasons = sorted({e.split(":")[0] for st in tstats for e in (st.get("errors") or [])})
            lines.append(f"> 🛑 **Every trial of every task errored "
                         f"({', '.join(reasons) or 'unknown'}). This model was never "
                         f"successfully measured. Its 0.00 scores below are an "
                         f"absence of data, not a result — exclude the whole config "
                         f"from any ranking.**")
            lines.append("")
        lines.append(f"- sut: `{cfg.get('sut')}` · llm: `{cfg.get('llm')}` · "
                     f"sampling: `{cfg.get('sampling_mode')}` · env: `{cfg.get('environment')}` · "
                     f"agents: `{cfg.get('agents_enabled')}` · trials: {cfg.get('trials')} · "
                     f"internet: {cfg.get('internet_allowed')}")
        lines.append("")
        lines.append("| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | "
                     "Latency s (mean) | Data scanned (mean) | Errors | Price src |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for task_id, t in block["tasks"].items():
            s = t["stats"]
            q = s.get("quality", {})
            score = q.get("score", q.get("pass", {}))
            score_mean = score.get("mean") if isinstance(score, dict) else None
            tup = s.get("tuple", {})
            # A 0.00 that came from every trial erroring is not a measured zero,
            # and must never be read as one: an unreachable or rate-limited model
            # would otherwise publish as "worse than a 30B". n_trials is absent
            # from stats, so fall back to the config's declared trial count.
            n_errors = s.get("n_errors", 0) or 0
            n_trials = s.get("n_trials") or cfg.get("trials") or 0
            all_errored = bool(n_errors) and n_trials and n_errors >= n_trials
            task_cell = f"**{task_id}** ⚠️" if all_errored else str(task_id)
            # Failure kinds (budget_exceeded, timeout, sut_start_error, ...) say
            # WHY a trial failed; the count alone cannot tell a crash from a
            # correct-but-too-expensive answer.
            failures = s.get("failure_kinds") or {}
            failure_text = ", ".join(f"{key}:{value}" for key, value in failures.items())
            if all_errored:
                err_cell = f"**{n_errors}/{n_trials} ALL**" + (
                    f" ({failure_text})" if failure_text else "")
            else:
                err_cell = failure_text or (str(n_errors) if n_errors else "—")
            srcs = s.get("cost_sources") or []
            src_cell = ", ".join(f"`{x}`" for x in srcs) if srcs else "—"
            lines.append(
                f"| {task_cell} | {s.get('pass_at_1', 0):.2f} | {s.get('best_of_n', 0):.2f} | "
                f"{_fmt(score_mean)} | {_fmt(tup.get('cost_usd', {}).get('mean'))} | "
                f"{_fmt(tup.get('latency_s', {}).get('mean'))} | "
                f"{_fmt(tup.get('data_scanned_ratio', {}).get('mean'))} | "
                f"{err_cell} | {src_cell} |"
            )
        lines.append("")
        if any((t["stats"].get("n_errors") or 0) for t in block["tasks"].values()):
            lines.append("> ⚠️  Rows marked **ALL** scored 0.00 because every trial "
                         "errored — the model was never successfully measured. Do not "
                         "read those as quality results.")
            lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        if number != number:
            return "—"
        return f"{number:.4g}"
    except (TypeError, ValueError):
        return str(value)
