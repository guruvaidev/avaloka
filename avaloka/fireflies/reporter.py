"""Reporter — replaces the analyst or consultant.

Economic output: executive and technical reports, plus the reproducibility and
governance artifacts that make the mission auditable (assumptions, lineage,
environment lock, README). Numbers come from upstream fireflies; the Reporter
only frames them — optionally enriching prose with an LLM, never inventing data.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from avaloka.fireflies.base import Firefly
from avaloka.llm import LLMClient, narrate
from avaloka.mission.economics import compute_economics
from avaloka.mission.ledger import WorkUnit
from avaloka.reports import (evaluation_html, executive_html, model_card_md,
                             technical_html)
from avaloka.util import write_json, write_text, write_yaml

_LOCK_PACKAGES = ["pandas", "numpy", "scikit-learn", "pyarrow", "jinja2", "PyYAML", "typer", "rich"]


class Reporter(Firefly):
    name = "reporter"
    human_role = "Analyst / consultant"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        bb = self.ctx.blackboard
        profile = bb["profile"]
        plan = bb.get("plan", {})
        validation = bb.get("validation", {})
        training_summary = bb.get("training_summary")
        finops = bb.get("finops")

        limitations = self._limitations(profile, plan, validation)
        findings = self._findings(profile, plan, training_summary)
        # The console prints what the mission actually concluded, so publish
        # both here rather than only into the HTML bundle: a terminal run that
        # reports a verdict and an economic multiplier but not a single finding
        # has answered a question nobody asked.
        bb["findings"] = findings
        bb["limitations"] = limitations

        # Provisional economics including this reporter unit so the HTML matches
        # the final console summary to within cents.
        provisional = WorkUnit(firefly=self.name, human_role=self.human_role,
                               output="reports", manual_minutes=120.0)
        units = self.ctx.ledger.units + [provisional]
        snapshot = _SnapshotLedger(units)
        econ = compute_economics(snapshot, tier=self.ctx.tier,
                                 loaded_hourly_rate=self.ctx.loaded_hourly_rate,
                                 wall_clock_seconds=self.ctx.wall_clock_seconds())

        narrative = self._narrative(profile, plan, training_summary, validation)

        common = {
            "goal": self.ctx.goal, "mission_id": self.ctx.mission_id,
            "kind": self.ctx.kind.value, "ds": profile["dataset"],
            "quality": profile["quality"], "columns": profile["columns"],
            "correlations": profile.get("correlations", []), "plan": plan,
            "validation": validation, "model": training_summary, "finops": finops,
        }
        exec_data = {
            **common,
            "kpis": [("Quality", f"{profile['quality']['score']}/100"),
                     ("Rows", f"{profile['dataset']['n_rows']:,}"),
                     ("Verdict", validation.get("verdict", "n/a").upper()),
                     ("Multiplier", f"{econ.economic_multiplier:.1f}x")],
            "economics": econ.rows(), "narrative": narrative, "findings": findings,
            "verdict": validation.get("verdict", "warn"),
            "max_level": validation.get("max_safe_deployment_level", 1),
            "limitations": limitations, "units": [u.as_dict() for u in self.ctx.ledger.units],
        }
        write_text(self.ctx.path("executive_report.html"), executive_html(exec_data))
        write_text(self.ctx.path("technical_report.html"), technical_html(common))

        # Train-only deliverables.
        if training_summary is not None:
            card = {**common, "limitations": limitations, "target": self.ctx.target}
            write_text(self.ctx.path("model_card.md"), model_card_md(card))
            write_text(self.ctx.path("evaluation_report.html"), evaluation_html(common))

        self._assumptions(profile, plan, validation)
        self._lineage(profile)
        self._environment_lock()
        self._readme(econ, findings, limitations)

        return ("Wrote executive & technical reports, assumptions, lineage, environment lock "
                "and README.", 120.0, {"model_cost_usd": self._narrate_cost})

    # --- content helpers --------------------------------------------------
    _narrate_cost = 0.0

    def _narrative(self, profile, plan, training, validation) -> str:
        q = profile["quality"]["score"]
        base = (f"Avaloka profiled {profile['dataset']['n_cols']} columns across "
                f"{profile['dataset']['n_rows']:,} rows (quality {q}/100), prepared a reproducible "
                f"pipeline and planned a '{plan.get('task', 'exploratory')}' approach. ")
        if training:
            base += (f"It trained {len(training['candidates'])} candidate model(s) and selected "
                     f"'{training['selected']}'. ")
        base += (f"Validation returned '{validation.get('verdict', 'n/a')}' with a maximum safe "
                 f"deployment level of {validation.get('max_safe_deployment_level', 1)} of 4.")
        client = LLMClient() if self.ctx.execution_mode.value != "local" else None
        system = ("You are a precise data-science consultant. Rewrite the summary in 2-4 sentences "
                  "for an executive. Do not invent numbers; only rephrase what is given.")
        text, cost = narrate(client, system, base, base)
        self._narrate_cost = cost
        return text

    def _findings(self, profile, plan, training) -> list[str]:
        out = []
        for p in profile.get("correlations", [])[:2]:
            if p["corr"] >= 0.5:
                out.append(f"'{p['a']}' and '{p['b']}' are strongly associated (|corr|={p['corr']}).")
        for issue in profile["quality"]["issues"][:2]:
            out.append(issue["detail"])
        if training:
            best = next(c for c in training["candidates"] if c["name"] == training["selected"])
            out.append(f"Best model '{training['selected']}' scored "
                       f"{best['primary_score']:.4f} {training['metric']} on holdout.")
        for h in plan.get("hypotheses", [])[:1]:
            out.append(h)
        return out or ["No dominant signal detected; treat results as exploratory."]

    def _limitations(self, profile, plan, validation) -> list[str]:
        lims = []
        suff = plan.get("data_sufficiency", {})
        if suff and not suff.get("sufficient", True):
            lims.append(suff.get("reason", "Data may be insufficient for stable conclusions."))
        if validation.get("leakage"):
            lims.append("Potential target leakage detected; resolve before trusting model metrics.")
        if profile["quality"]["high_missing_columns"]:
            lims.append("High-missingness columns require an explicit imputation policy.")
        if profile["dataset"]["n_rows"] < 200:
            lims.append("Small sample size limits statistical confidence.")
        lims.append("Estimated manual effort and value are heuristics pending customer calibration.")
        return lims

    def _assumptions(self, profile, plan, validation) -> None:
        assumptions = {
            "mission": {"id": self.ctx.mission_id, "goal": self.ctx.goal, "kind": self.ctx.kind.value},
            "sampling": self.ctx.blackboard.get("sampling", {}),
            "transformations": self.ctx.blackboard.get("transformations", []),
            "task_choice": {"task": plan.get("task"), "reason": plan.get("task_reason")},
            "split": plan.get("split", {}),
            "human_approval_gates": plan.get("human_approval_gates", []),
            "refuse_if": plan.get("refuse_if", []),
            "data_quality_issues": profile["quality"]["issues"],
            "deferred_decisions": [
                t for t in self.ctx.blackboard.get("transformations", [])
                if isinstance(t, dict) and t.get("applied") is False
            ],
        }
        write_yaml(self.ctx.path("assumptions.yaml"), assumptions)

    def _lineage(self, profile) -> None:
        lineage = {
            "mission_id": self.ctx.mission_id,
            "kind": self.ctx.kind.value,
            "started_at_epoch": self.ctx.started_at,
            "source": {"path": profile["dataset"]["source"], "sha256": profile["dataset"]["sha256"],
                       "format": profile["dataset"]["format"]},
            "environment": self.ctx.environment(),
            "transformations": self.ctx.blackboard.get("transformations", []),
            "work_units": self.ctx.ledger.as_list(),
            "outputs": sorted(p.name for p in self.ctx.output_dir.iterdir() if p.is_file()),
        }
        write_json(self.ctx.path("lineage.json"), lineage)

    def _environment_lock(self) -> None:
        lines = [f"# Avaloka environment lock — mission {self.ctx.mission_id}",
                 f"python=={self.ctx.environment()['python']}"]
        for pkg in _LOCK_PACKAGES:
            try:
                lines.append(f"{pkg}=={version(pkg)}")
            except PackageNotFoundError:
                continue
        write_text(self.ctx.path("environment.lock"), "\n".join(lines) + "\n")

    def _readme(self, econ, findings, limitations) -> None:
        train = self.ctx.blackboard.get("training_summary") is not None
        deliverables = [
            "`executive_report.html` — the decision-maker summary",
            "`technical_report.html` — full methodology, schema, quality and validation",
            "`analysis.ipynb` / `analysis.py` — reproducible pipeline",
            "`transformed_dataset.parquet` — cleaned working dataset",
            "`data_quality.json`, `validation_report.json`, `planner_graph.json` — evidence",
            "`assumptions.yaml`, `lineage.json`, `environment.lock` — governance & reproducibility",
        ]
        if train:
            deliverables += [
                "`model/` + `model_card.md` — the validated model and its card",
                "`evaluation_report.html`, `cost_quality.json` — candidate comparison & economics",
                "`service.py`, `Dockerfile`, `deployment/` — the deployment package",
                "`feature_contract.yaml`, `inference_schema.json`, `monitoring_config.yaml` — the serving contract",
            ]
        md = [f"# Avaloka mission `{self.ctx.mission_id}`", "",
              f"**Goal:** {self.ctx.goal}", "",
              f"This bundle was produced by Avaloka as a governed *Data Mission* — an outcome, not "
              f"an interaction. Economic multiplier: **{econ.economic_multiplier:.1f}x** "
              f"(≈{econ.estimated_manual_effort_hours:.1f} manual hours compressed).", "",
              "## Deliverables", ""]
        md += [f"- {d}" for d in deliverables]
        md += ["", "## Key findings", ""] + [f"- {f}" for f in findings]
        md += ["", "## Limitations", ""] + [f"- {l}" for l in limitations]
        md += ["", "## Reproduce", "", "```bash", "python analysis.py", "```", ""]
        if train:
            md += ["## Serve locally", "", "```bash",
                   "pip install -r requirements.lock", "uvicorn service:app --port 8080",
                   "# then: avaloka deploy . --target local", "```", ""]
        md += ["---", "_Reproduce or adapt this mission with `avaloka`._"]
        write_text(self.ctx.path("README.md"), "\n".join(md))


class _SnapshotLedger:
    """Read-only ledger view over a fixed unit list (for provisional economics)."""

    def __init__(self, units: list[WorkUnit]) -> None:
        self._u = units

    @property
    def manual_minutes(self): return sum(u.manual_minutes for u in self._u)
    @property
    def compute_cost_usd(self): return sum(u.compute_cost_usd for u in self._u)
    @property
    def model_cost_usd(self): return sum(u.model_cost_usd for u in self._u)
    @property
    def duration_seconds(self): return sum(u.duration_seconds for u in self._u)
