"""Reporter — replaces the analyst or consultant.

Economic output: executive and technical reports, plus the reproducibility and
governance artifacts that make the mission auditable (assumptions, lineage,
environment lock, README). Numbers come from upstream fireflies; the Reporter
only frames them — optionally enriching prose with an LLM, never inventing data.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import shlex
from typing import Any

from avaloka.fireflies.base import Firefly
from avaloka.llm import LLMClient, narrate, usage_metadata
from avaloka.mission.economics import MANUAL_EFFORT_CAVEAT, compute_economics
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
        goal_analysis = bb.get("goal_analysis")

        limitations = self._limitations(profile, plan, validation)
        findings = self._findings(profile, plan, training_summary, goal_analysis)
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

        narrative = self._narrative(profile, plan, training_summary, validation, goal_analysis)
        # A model's rewrite of the deterministic summary is commentary (c-strict).
        narrative_commentary = narrative != self._narrative_base

        common = {
            "goal": self.ctx.goal, "mission_id": self.ctx.mission_id,
            "kind": self.ctx.kind.value, "ds": profile["dataset"],
            "quality": profile["quality"], "columns": profile["columns"],
            "correlations": profile.get("correlations", []), "plan": plan,
            "validation": validation, "model": training_summary, "finops": finops,
            "goal_analysis": goal_analysis, "domain": profile.get("domain"),
            "script_declines_to_reread": self.ctx.blackboard.get("script_declines_to_reread"),
        }
        exec_data = {
            **common,
            "kpis": [("Quality", f"{profile['quality']['score']}/100"),
                     ("Rows", f"{profile['dataset']['n_rows']:,}"),
                     ("Verdict", validation.get("verdict", "n/a").upper()),
                     ("Multiplier", f"{econ.economic_multiplier:.1f}x")],
            "economics": econ.rows(), "narrative": narrative,
            "narrative_commentary": narrative_commentary, "findings": findings,
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
        self._readme(econ, findings, limitations, profile.get("domain"))

        return ("Wrote executive & technical reports, assumptions, lineage, environment lock "
                "and README.", 120.0, {"model_cost_usd": self._narrate_cost, **self._narrate_usage})

    # --- content helpers --------------------------------------------------
    _narrate_cost = 0.0
    _narrate_usage: dict = {}
    _narrative_base = ""

    def _narrative(self, profile, plan, training, validation, goal_analysis=None) -> str:
        q = profile["quality"]["score"]
        base = (f"Avaloka profiled {profile['dataset']['n_cols']} columns across "
                f"{profile['dataset']['n_rows']:,} rows (quality {q}/100), prepared a reproducible "
                f"pipeline and planned a '{plan.get('task', 'exploratory')}' approach. ")
        domain = profile.get("domain") or {}
        if domain.get("domain") and domain["domain"] != "generic":
            base += f"The columns read as {domain['label'].lower()} data. "
        if training:
            base += (f"It trained {len(training['candidates'])} candidate model(s) and selected "
                     f"'{training['selected']}'. ")
        base += (f"Validation returned '{validation.get('verdict', 'n/a')}' with a maximum safe "
                 f"deployment level of {validation.get('max_safe_deployment_level', 1)} of 4.")
        if goal_analysis:
            base += (f" Goal-specific analysis ({goal_analysis['scope']}): "
                     f"{goal_analysis['findings'][0]}")
        client = LLMClient() if self.ctx.execution_mode.value != "local" else None
        system = ("You are a precise data-science consultant. Rewrite the summary in 2-4 sentences "
                  "for an executive. Do not invent numbers; only rephrase what is given.")
        text, cost = narrate(client, system, base, base)
        self._narrative_base = base
        self._narrate_cost = cost
        self._narrate_usage = usage_metadata(client)
        return text

    def _findings(self, profile, plan, training, goal_analysis=None) -> list[str]:
        # The specialists' verdicts lead: an A/B goal is answered by the
        # lift and its interval, not by the column summaries that follow.
        specialist = self._specialist_findings()
        if goal_analysis:
            # One basis per figure: when the goal's figures were computed
            # over every row, its "Computed over all N rows" line closes
            # them, and a specialist figure from the working sample follows
            # it under its own sample label, never above that claim.
            if (goal_analysis.get("basis") or {}).get("full_pass") and self._statistic_sampled():
                return list(goal_analysis["findings"]) + specialist
            return specialist + list(goal_analysis["findings"])
        out = list(specialist)
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

    def _specialist_findings(self) -> list[str]:
        """The lift with its CI, the SRM check and the power; the statistic;
        the conformal coverage. Each only when its specialist ran."""
        bb = self.ctx.blackboard
        out: list[str] = []
        result = bb.get("experiment_result")
        plan = bb.get("experiment_plan") or {}
        if result is not None:
            get = (plan.get if isinstance(plan, dict) else lambda k, d=None: getattr(plan, k, d))
            arm, outcome = get("treatment_column", "arm"), get("outcome", "outcome")
            treatment, control = get("treatment"), get("control")
            if result.effect is None:
                detail = "; ".join(d.get("detail", "") for d in result.diagnostics) or result.status
                out.append(f"A/B on '{outcome}' ({arm}: {treatment} vs {control}) could not be "
                           f"verified: {detail}")
            else:
                low, high = result.confidence_interval
                unit = "share" if get("metric") == "proportion" else "mean"
                relative = (f", relative {result.relative_effect:+.1%}"
                            if result.relative_effect is not None else "")
                out.append(f"A/B on '{outcome}': '{treatment}' vs '{control}' lift {result.effect:+.4f} "
                           f"in {unit} (95% CI {low:+.4f} to {high:+.4f}{relative}; "
                           f"n={result.control_n}/{result.treatment_n}).")
                if low > 0:
                    out.append(f"The interval excludes zero: '{treatment}' is better than "
                               f"'{control}' on '{outcome}' at the 95% level.")
                elif high < 0:
                    out.append(f"The interval excludes zero: '{treatment}' is worse than "
                               f"'{control}' on '{outcome}' at the 95% level.")
                else:
                    out.append(f"The interval includes zero: the data does not show "
                               f"'{treatment}' beating '{control}' on '{outcome}'.")
                srm = next((d for d in result.diagnostics if d.get("kind") == "sample_ratio"), None)
                if result.sample_ratio_mismatch is None:
                    out.append("Sample ratio check: not run (no expected arm share).")
                elif result.sample_ratio_mismatch:
                    out.append(f"Sample ratio mismatch (chi-square {srm['chi_square']:.2f} > "
                               f"{srm['threshold']}): the assignment is suspect; treat the lift "
                               "as unverified.")
                else:
                    out.append(f"Sample ratio check passed (chi-square {srm['chi_square']:.2f} "
                               f"below {srm['threshold']}).")
                power = result.power or {}
                if power.get("achieved_power") is not None:
                    line = (f"Power: {power['achieved_power']:.0%} to detect the observed effect at "
                            f"alpha {power['alpha']}; these arms resolve an effect of "
                            f"{power['minimum_detectable_effect']:.4f} at "
                            f"{power['target_power']:.0%} power")
                    if power.get("required_n_per_arm_for_observed_effect"):
                        line += (f"; the observed effect would need about "
                                 f"{power['required_n_per_arm_for_observed_effect']:,} per arm")
                    out.append(line + ".")
        statistic = bb.get("statistical_result")
        if statistic is not None:
            estimate = statistic.estimate
            shown = f"{estimate:.4f}" if isinstance(estimate, float) else str(estimate)
            stat_plan = bb.get("statistical_plan") or {}
            what = (stat_plan.get("estimand") if isinstance(stat_plan, dict)
                    else getattr(stat_plan, "estimand", "")) or statistic.method
            line = f"{what}: {shown} over {statistic.sample_size:,} valid rows"
            sampling = bb.get("sampling") or {}
            notes = []
            if self._statistic_sampled():
                # The Statistician read the prepared working frame, which on
                # a sampled run is the sample, and the figure says so.
                notes.append(f"sample of {int(sampling.get('n_working_rows') or 0):,} of "
                             f"{int(sampling.get('n_input_rows') or 0):,} rows")
            elif (bb.get("statistical_basis") or {}).get("full_pass"):
                # Every source row was in memory, so the figure is theirs.
                notes.append("computed over every row")
            if statistic.confidence_interval:
                lo, hi = statistic.confidence_interval
                notes.append(f"95% CI {lo:.4f} to {hi:.4f}")
            if notes:
                line += f" ({'; '.join(notes)})"
            if statistic.p_value is not None:
                line += f", p={statistic.p_value:.4g}"
            out.append(line + ".")
        conformal = bb.get("conformal_result")
        if conformal is not None:
            kind = "interval" if conformal.task == "regression" else "label set"
            width = (f"mean width {conformal.efficiency:.4f}" if conformal.task == "regression"
                     else f"mean set size {conformal.efficiency:.2f}")
            covered = (f"; empirical coverage {conformal.empirical_coverage:.1%} on held-out rows"
                       if conformal.empirical_coverage is not None else "")
            out.append(f"Conformal {kind} at {conformal.target_coverage:.0%} target coverage: "
                       f"{width}{covered}.")
        return out

    def _statistic_sampled(self) -> bool:
        """Whether the Statistician's figure rests on the working sample."""
        bb = self.ctx.blackboard
        if bb.get("statistical_result") is None:
            return False
        basis = bb.get("statistical_basis")
        if basis is not None:
            return bool(basis.get("sampled"))
        return bool((bb.get("sampling") or {}).get("sampled"))

    def _limitations(self, profile, plan, validation) -> list[str]:
        lims = []
        suff = plan.get("data_sufficiency", {})
        if suff and not suff.get("sufficient", True):
            lims.append(suff.get("reason", "Data may be insufficient for stable conclusions."))
        if validation.get("leakage"):
            lims.append("Potential target leakage detected; resolve before trusting model metrics.")
        if profile["quality"]["high_missing_columns"]:
            lims.append("High-missingness columns require an explicit imputation policy.")
        goal_analysis = self.ctx.blackboard.get("goal_analysis")
        if goal_analysis and goal_analysis["status"] != "answered":
            lims.append("The stated goal was not fully answered by a supported calculation; "
                        "see goal_analysis.json for the reason.")
        if goal_analysis and goal_analysis.get("basis", {}).get("sampled", self.ctx.blackboard.get("sampling", {}).get("sampled")):
            lims.append("Goal-specific figures use the working sample, not every source row.")
        if profile["dataset"]["n_rows"] < 200:
            lims.append("Small sample size limits statistical confidence.")
        specialists = plan.get("specialists", {})
        for assumption in specialists.get("experiment", {}).get("assumptions", []):
            lims.append(assumption)
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

    def _readme(self, econ, findings, limitations, domain=None) -> None:
        train = self.ctx.blackboard.get("training_summary") is not None
        declined = self.ctx.blackboard.get("script_declines_to_reread")
        deliverables = [
            "`executive_report.html` — the decision-maker summary",
            "`technical_report.html` — full methodology, schema, quality and validation",
            ("`analysis.ipynb` / `analysis.py` — reproducible pipeline" if not declined else
             f"`analysis.ipynb` / `analysis.py` — the preparation steps as code; they do not "
             f"re-read the source ({declined}) and say so when run"),
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
        else:
            deliverables.append("`goal_analysis.json` — measured answer to the stated goal")
        md = [f"# Avaloka mission `{self.ctx.mission_id}`", "",
              f"**Goal:** {self.ctx.goal}", "",
              f"This bundle was produced by Avaloka as a governed *Data Mission* — an outcome, not "
              f"an interaction. Economic multiplier: **{econ.economic_multiplier:.1f}x** "
              f"(≈{econ.estimated_manual_effort_hours:.1f} manual hours compressed — "
              f"{MANUAL_EFFORT_CAVEAT}).", "",
              "## Deliverables", ""]
        md += [f"- {d}" for d in deliverables]
        md += self._domain_section(domain)
        md += ["", "## Key findings", ""] + [f"- {f}" for f in findings]
        md += ["", "## Limitations", ""] + [f"- {l}" for l in limitations]
        if train and declined:
            reproduce = ("# analysis.py does not re-read the source; the prepared data is here:\n"
                         "python -c \"import pandas as pd; "
                         "print(pd.read_parquet('transformed_dataset.parquet').shape)\"")
        elif train:
            reproduce = "python analysis.py  # reproduce data preparation"
        else:
            reproduce = (f"avaloka analyze {shlex.quote(self.ctx.data_source)} "
                         f"--goal {shlex.quote(self.ctx.goal)} --output ./reproduced-analysis")
        md += ["", "## Reproduce", "", "```bash", reproduce, "```", ""]
        if train:
            md += ["## Serve locally", "", "```bash",
                   "pip install -r requirements.lock", "uvicorn service:app --port 8080",
                   "# then: avaloka deploy . --target local", "```", ""]
        md += ["---", "_Reproduce or adapt this mission with `avaloka`._"]
        write_text(self.ctx.path("README.md"), "\n".join(md))


    @staticmethod
    def _domain_section(domain) -> list[str]:
        """The vocabulary's read of the data and the one check it asks for first.

        Names decided the domain (``avaloka.domains.classify``); the section
        quotes the matched column names, never a value.
        """
        if not domain:
            return []
        if domain.get("domain") == "generic":
            lines = ["", "## Domain", "",
                     f"No domain was recognised from the column names ({domain.get('reason', '')}); "
                     "the general checks apply.", ""]
        else:
            cues = sorted({h["column"] for h in domain.get("matched_cues", [])})
            lines = ["", "## Domain", "",
                     f"Read as **{domain['label']}** data (confidence {domain['confidence']:.2f}; "
                     f"matched on {', '.join(cues[:4])}).", ""]
        checks = domain.get("hints", {}).get("sanity_checks") or []
        if checks:
            lines.append(f"Sanity check for this domain: {checks[0]}")
        questions = domain.get("questions") or []
        if questions:
            lines += ["", f"A question worth answering first: {questions[0]}"]
        return lines


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
