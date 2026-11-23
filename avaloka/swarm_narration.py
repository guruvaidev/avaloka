"""The debug narration: a coordinated mission told back as what the team did.

``avaloka coordinate --debug`` prints, as the run proceeds, short plain
sentences explaining what Avaloka and her clones actually did — who read the
file and what shape it had, how it was sharded, which specialist each clone
sent next and what she came back with, how the results were merged and what
the final numbers rest on. Every sentence here is produced from one trace
record (:mod:`avaloka.swarm_trace`); there is no script to fall back on, so a
line that is not in the records is not in the story.

The narrator is a pure function of the records: :func:`narrate` folds a list
of records into a list of lines, and :class:`SwarmNarrative` is the same fold
exposed one record at a time so the CLI can print while the recorder's sink is
still firing. Feeding the records one by one yields exactly the lines
:func:`narrate` yields for the whole list.

Two levels. ``VERBOSE`` tells the team's story — the brief, the shards, each
clone leaving and returning, the merge and the close. ``DEBUG`` adds every
specialist step each clone ran, with the step's own one-line output. The
``agent.understanding`` and ``swarm.understanding`` records (the richer
agent-level trace) are narrated when present and simply absent otherwise;
any other record type is passed over, not guessed at — register a handler in
:data:`HANDLERS` to narrate it.

Everything here is local stdout and may show the user's column names. None of
it is telemetry; nothing here sends anything anywhere. Nothing from ``app`` (S1.08).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from avaloka.mission.economics import MANUAL_EFFORT_CAVEAT
from avaloka.swarm import CLONE

#: Narration levels. ``QUIET`` yields nothing; the CLI maps no flag, ``-v``
#: and ``--debug`` onto these three.
QUIET, VERBOSE, DEBUG = 0, 1, 2

#: What each specialist does, in the words the swarm already uses for her.
#: Read from :data:`avaloka.swarm.CLONE` so a firefly added there narrates
#: here without a second table; a firefly in neither is described by its name
#: alone — nothing is invented for it.
_FOCUS = {firefly: focus for firefly, (_name, focus) in CLONE.items()}

Handler = Callable[["SwarmNarrative", dict[str, Any]], list[str]]


def _title(firefly: Optional[str]) -> str:
    return (firefly or "specialist").replace("_", " ")


def _dur(ms: Any) -> str:
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "an unknown time"
    if ms < 1000:
        return f"{int(round(ms))} ms"
    if ms < 90_000:
        return f"{ms / 1000:.1f} s"
    return f"{ms / 60_000:.1f} min"


def _n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "$?"


def _money_at_least(value: Any) -> str:
    """_money, except that a positive amount too small for four decimals is
    "under $0.0001", never "$0"."""
    try:
        if 0 < float(value) < 0.00005:
            return "under $0.0001"
    except (TypeError, ValueError):
        pass
    return _money(value)


def _columns(columns: Any, limit: int = 6) -> str:
    cols = [str(c) for c in (columns or [])]
    if not cols:
        return "no columns"
    shown = ", ".join(cols[:limit])
    more = len(cols) - limit
    return shown if more <= 0 else f"{shown} and {more} more"


def _plural(n: Any, word: str) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return f"{n} {word}s"
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _first_line(text: Any) -> str:
    """A failure's message, never its traceback."""
    return str(text or "").strip().splitlines()[0] if text else ""


class SwarmNarrative:
    """The fold. ``feed`` one record and get the sentences it produces."""

    def __init__(self, level: int = DEBUG) -> None:
        self.level = level
        # The little the narrator remembers between records: how many columns
        # the dataset had, each clone's name, and how many clones left.
        self._n_columns: Optional[int] = None
        self._names: dict[str, str] = {}
        self._dispatched = 0
        self._steps_done = 0
        self._manual_minutes = 0.0
        self._model_calls: dict[str, int] = {}   # model id -> calls, from agent.understanding
        self._model_agents: list[str] = []

    # -- the fold ----------------------------------------------------------
    def feed(self, record: dict[str, Any]) -> list[str]:
        if self.level <= QUIET:
            return []
        handler = HANDLERS.get(str(record.get("type")))
        if handler is None:
            return []
        return handler(self, record)

    def _name(self, record: dict[str, Any]) -> str:
        cid = str(record.get("clone_id") or "")
        # ``name`` on clone.dispatched, ``clone_name`` on agent.understanding.
        return (self._names.get(cid) or record.get("name") or record.get("clone_name")
                or cid or "a clone")

    # -- one method per record type ----------------------------------------
    def requested(self, r: dict[str, Any]) -> list[str]:
        kind = r.get("kind")
        verb = "train a model on" if kind == "train" else "analyse"
        goal = r.get("goal")
        lines = [f"Avaloka took the brief: {verb} {r.get('source')}"
                 + (f" to answer “{goal}”." if goal else ".")]
        limits = r.get("limits") or {}
        label = limits.get("label") or limits.get("edition") or "this edition"
        cap = limits.get("max_clones_per_mission")
        granted = r.get("granted_clones")
        requested = r.get("requested_clones")
        if r.get("limited") and requested is not None and granted is not None and granted < requested:
            # "You asked for" only when someone typed the number (--swarm N);
            # a fleet sized from the data says so, as SwarmGrant.message does.
            asked = (f"the data sized the fleet at {requested}" if r.get("derived_from")
                     else f"you asked for {requested}")
            lines.append(f"Team size: {granted} clones will run — {asked}, "
                         f"and {label} allows {cap} per mission.")
        elif cap is None:
            lines.append(f"Team size: {granted} clones ({label} sets no per-mission limit).")
        else:
            lines.append(f"Team size: {granted} clones, within the {cap} that {label} allows per mission.")
        parallel = r.get("granted_parallel")
        if parallel is not None and granted is not None and parallel < granted:
            lines.append(f"They run {parallel} at a time.")
        return lines

    def sharded(self, r: dict[str, Any]) -> list[str]:
        columns = r.get("columns") or []
        self._n_columns = len(columns)
        total = r.get("total_rows")
        per = list(r.get("rows_per_clone") or [])
        lines = [f"She read the dataset once: {_n(total)} rows × {len(columns)} columns "
                 f"({_columns(columns)})."]
        if per:
            rows = " and ".join(_n(x) for x in per) if len(per) <= 2 else ", ".join(_n(x) for x in per)
            disjoint = ""
            try:
                if sum(int(x) for x in per) == int(total):
                    disjoint = ", so every row sits in exactly one shard"
            except (TypeError, ValueError):
                pass
            lines.append(f"She cut it into {len(per)} shards of {rows} rows — one per clone{disjoint}.")
        return lines

    def dispatched(self, r: dict[str, Any]) -> list[str]:
        cid = str(r.get("clone_id") or "")
        name = r.get("name") or cid or "A clone"
        self._names[cid] = name
        self._dispatched += 1
        received = r.get("received") or {}
        cols = received.get("columns")
        ncols = len(cols) if cols else self._n_columns
        line = f"{name} picked up shard {r.get('shard_id')}: {_n(received.get('rows'))} rows"
        if ncols:
            line += f", all {ncols} columns"
        if received.get("target"):
            line += f", target “{received['target']}”"
        return [line + "."]

    def step_started(self, r: dict[str, Any]) -> list[str]:
        if self.level < DEBUG:
            return []
        firefly = r.get("firefly")
        focus = _FOCUS.get(str(firefly))
        tail = f": she {focus}." if focus else "."
        return [f"{self._name(r)} → step {r.get('step_no')}, her {_title(firefly)}{tail}"]

    def step_completed(self, r: dict[str, Any]) -> list[str]:
        # Counted at every level: the closing line says what the hours rest on.
        self._steps_done += 1
        try:
            self._manual_minutes += float(r.get("manual_minutes") or 0.0)
        except (TypeError, ValueError):
            pass
        if self.level < DEBUG:
            return []
        name = self._name(r)
        role = r.get("human_role")
        who = f"{_title(r.get('firefly'))}" + (f" ({role})" if role else "")
        if r.get("outcome") == "ok":
            output = str(r.get("output") or "").strip()
            line = f"{name} ← {who} done in {_dur(r.get('duration_ms'))}"
            return [line + (f": {output}" if output else ".")]
        err = _first_line(r.get("error"))
        klass = r.get("error_class") or "an error"
        return [f"{name} ← {who} failed after {_dur(r.get('duration_ms'))}: {klass}"
                + (f" — {err}" if err else "") + "."]

    def returned(self, r: dict[str, Any]) -> list[str]:
        name = self._name(r)
        dur = _dur(r.get("duration_ms"))
        if r.get("status") != "done":
            err = _first_line(r.get("error"))
            return [f"{name} came back without a result after {dur}: "
                    f"{r.get('error_class') or 'an error'}" + (f" — {err}" if err else "") + "."]
        bits = []
        verdict = r.get("verdict")
        if verdict:
            lvl = r.get("max_safe_deployment_level")
            bits.append(f"verdict {str(verdict).upper()}"
                        + (f", safe to deploy up to level {lvl}/4" if lvl is not None else ""))
        sampling = r.get("sampling") or {}
        if sampling:
            if sampling.get("sampled"):
                bits.append(f"she worked on a {sampling.get('strategy')} sample of "
                            f"{_n(sampling.get('n_working_rows'))} of her "
                            f"{_n(sampling.get('n_input_rows'))} rows")
            elif sampling.get("n_working_rows") is not None:
                bits.append(f"she worked on all {_n(sampling.get('n_working_rows'))} of her rows")
        model = r.get("model") or {}
        if isinstance(model, dict) and model.get("selected"):
            score = model.get("score")
            metric = model.get("metric")
            bits.append(f"her best model is {model['selected']}"
                        + (f" ({metric} {score:.4f})" if isinstance(score, (int, float)) and metric else ""))
        cost = r.get("cost_usd")
        compute, model_cost = r.get("compute_cost_usd"), r.get("model_cost_usd")
        if isinstance(compute, (int, float)) or isinstance(model_cost, (int, float)):
            # The compute figure is a nominal local-CPU rate per second
            # (fireflies.base.LOCAL_CPU_USD_PER_SECOND), not money spent: said
            # bare as "cost $0.0001" it sat next to "No model was called".
            if model_cost:
                bits.append(f"model calls {_money_at_least(model_cost)}")
            if compute:
                bits.append(f"compute {_money_at_least(compute)} at a nominal CPU rate")
        elif cost:
            bits.append(f"cost {_money(cost)}, a nominal compute rate plus any model calls")
        steps = r.get("steps")
        head = f"{name} is back after {dur}" + (f" with {steps} steps done" if steps is not None else "")
        return [head + (": " + "; ".join(bits) if bits else "") + "."]

    def merged(self, r: dict[str, Any]) -> list[str]:
        rules = r.get("rules") or {}
        contributions = list(r.get("contributions") or [])
        n = len(contributions) or self._dispatched
        verdict = r.get("verdict")
        lvl = r.get("max_safe_deployment_level")
        parts = []
        if rules.get("profile"):
            parts.append(f"the profile by {rules['profile']} over {_n(r.get('n_rows'))} rows")
        if verdict:
            parts.append(f"the verdict by {rules.get('verdict', 'rule')} → {str(verdict).upper()}"
                         + (f" (safe to level {lvl}/4)" if lvl is not None else ""))
        selected = r.get("selected_model") or {}
        if isinstance(selected, dict) and selected.get("selected"):
            parts.append(f"the model by {rules.get('model', 'rule')} → {selected['selected']}"
                         + (f" from shard {selected.get('from_shard')}"
                            if selected.get("from_shard") is not None else ""))
        lines = [f"Avaloka merged the {n} results into one answer: " + ", ".join(parts) + "."
                 if parts else f"Avaloka merged the {n} results into one answer."]
        # What the merged data says, from the exact merged profile
        # (avaloka.coordinator.merged_findings): the story names a finding,
        # not only the plumbing that produced it.
        findings = [str(f).strip().rstrip(".") for f in (r.get("findings") or []) if str(f).strip()]
        if findings:
            lines.append("What the merged data shows: " + "; ".join(findings) + ".")
        if contributions:
            lines.append("Each clone's share: " + "; ".join(
                f"{self._names.get(str(c.get('clone_id')), c.get('clone_id'))} "
                f"({_n(c.get('rows'))} rows, {str(c.get('verdict') or '—').lower()}"
                + (", model selected" if c.get("selected") else "") + ")"
                for c in contributions) + ".")
        answer = r.get("answer") or {}
        if answer.get("headline"):
            # The answer to the goal, and what each clone put into it.
            lines.append(str(answer["headline"]))
            lines.extend(str(c) for c in answer.get("contributions") or [])
            if answer.get("approximate"):
                lines.append("Approximate, not exact: " + "; ".join(map(str, answer["approximate"])) + ".")
        return lines

    def finished(self, r: dict[str, Any]) -> list[str]:
        dur = _dur(r.get("duration_ms"))
        if r.get("status") != "done":
            return [f"The mission stopped after {dur}: {r.get('error_class') or 'an error'}."]
        econ = r.get("economics") or {}
        hours = econ.get("estimated_manual_effort_hours")
        line = f"Finished in {dur}."
        if isinstance(hours, (int, float)):
            basis = (f": the fixed manual minutes each of the {self._steps_done} specialist steps is "
                     f"rated at, added up" if self._steps_done else "")
            line += (f" Manual effort: about {hours:.1f} hours, a {MANUAL_EFFORT_CAVEAT}{basis}.")
        # Whether a model took part is known from the understanding records
        # when they are in the stream (a local model costs nothing, so the
        # cost figure alone would say "no model" after gemma phrased the
        # report); the cost is the fallback when they are not.
        model_cost = econ.get("model_cost_usd")
        if self._model_calls:
            calls = sum(self._model_calls.values())
            ids = ", ".join(self._model_calls)
            by = " and ".join(self._model_agents[:2])
            line += (f" The {by} made {_plural(calls, 'call')} to {ids}"
                     + (f" (cost {_money(model_cost)})" if isinstance(model_cost, (int, float)) and model_cost > 0 else "")
                     + "; every number above was computed from the data, not by a model.")
        elif isinstance(model_cost, (int, float)) and model_cost > 0:
            line += f" Model calls cost {_money(model_cost)}."
        elif model_cost is not None:
            line += " No model was called: every number above was computed from the data."
        return [line]


    # -- the understanding records -----------------------------------------
    # ``agent.understanding`` follows each ``clone.step.completed`` and carries
    # the decision in words; ``swarm.understanding`` precedes ``swarm.finished``
    # with the team's totals. The conclusion itself was already told from the
    # step record, so only what is new is said here.
    def agent_understanding(self, r: dict[str, Any]) -> list[str]:
        # A failed step was already told, with its error, from clone.step.completed.
        if self.level < DEBUG or r.get("outcome") == "failed":
            return []
        name = self._name(r)
        who = _title(r.get("agent"))
        decided = r.get("decided") or {}
        lines: list[str] = []
        if decided.get("task") and decided.get("task_reason"):
            why = str(decided["task_reason"]).strip().rstrip(".")
            line = f"{name}'s {who} chose the task {decided['task']} because {why}"
            if decided.get("split") and decided.get("split_reason"):
                line += f"; split {decided['split']}: {str(decided['split_reason']).strip().rstrip('.')}"
            lines.append(line + ".")
        findings = decided.get("findings")
        if isinstance(findings, list) and findings:
            shown = "; ".join(str(f).rstrip(".") for f in findings[:2])
            more = len(findings) - 2
            lines.append(f"{name} found: {shown}" + (f" (and {more} more)." if more > 0 else "."))
        # Keyed on the agent, not the role: data_engineer and feature_engineer
        # share a role, and the optimizer shares the planner's.
        if r.get("agent") == "workload_optimizer" and decided.get("strategy"):
            line = f"{name}'s {who} chose the {decided['strategy']} path"
            extras = []
            if decided.get("n_pushdown_filters"):
                extras.append(f"{decided['n_pushdown_filters']} filters pushed down")
            if decided.get("n_projection_columns"):
                extras.append(f"{decided['n_projection_columns']} columns projected")
            if decided.get("estimated_rows_after_filters") is not None:
                extras.append(f"about {_n(decided['estimated_rows_after_filters'])} rows after filtering")
            lines.append(line + (": " + ", ".join(extras) if extras else "") + ".")
        if r.get("agent") == "feature_engineer" and decided.get("n_steps") is not None \
                and not decided["n_steps"]:
            lines.append(f"{name}'s {who} left the columns as they were.")
        elif r.get("agent") == "feature_engineer" and decided.get("n_steps") is not None:
            ops = [str(o) for o in (decided.get("operations") or [])]
            line = f"{name}'s {who} applied {_plural(decided['n_steps'], 'feature step')}"
            if ops:
                shown = ", ".join(ops[:3]) + (f" and {len(ops) - 3} more" if len(ops) > 3 else "")
                line += f" ({shown})"
            diagnostics = decided.get("diagnostics") or []
            if diagnostics:
                line += f"; {_plural(len(diagnostics), 'diagnostic')}"
            lines.append(line + ".")
        if decided.get("selected") and decided.get("metric"):
            score, base = decided.get("score"), decided.get("baseline_score")
            line = f"{name}'s {who} picked {decided['selected']}"
            if isinstance(score, (int, float)):
                line += f" at {decided['metric']} {score:.4f}"
            if isinstance(base, (int, float)):
                line += f" against a baseline of {base:.4f}"
            beats = decided.get("beats_baseline")
            if beats is not None:
                line += ", which beats it" if beats else ", which does not beat it"
            lines.append(line + ".")
        ver = r.get("verification") or {}
        if ver.get("checks_run"):
            line = (f"{name}'s {who} ran {_plural(ver['checks_run'], 'check')}: {ver.get('passed', 0)} passed, "
                    f"{ver.get('warned', 0)} warned, {ver.get('failed', 0)} failed")
            if ver.get("retries"):
                line += f", after {ver['retries']} retries"
            blocking = decided.get("blocking_issues") or []
            if blocking:
                line += "; blocking: " + "; ".join(str(b).rstrip(".") for b in blocking[:2])
            lines.append(line + ".")
        model = r.get("model") or {}
        if model.get("calls"):
            model_id = str(model.get("model_id") or "a model")
            self._model_calls[model_id] = self._model_calls.get(model_id, 0) + int(model["calls"])
            if who not in self._model_agents:
                self._model_agents.append(who)
            lines.append(f"{name}'s {who} made {_plural(model['calls'], 'call')} to {model_id}"
                         + (f" on {model['provider']}" if model.get("provider") else "") + ".")
        if not lines:
            lines.append(self._understood_shape(r, name, who))
        return lines

    def _understood_shape(self, r: dict[str, Any], name: str, who: str) -> str:
        """A step that decided nothing new still says what it had in view.

        Counts only: the record's ``data`` also carries ``column_names``, and
        this line must not name a column or a file the story has not named.
        """
        data = r.get("data") or {}
        sampling = r.get("sampling") or {}
        seen = []
        if data.get("rows") is not None and data.get("columns") is not None:
            seen.append(f"{_n(data['rows'])} rows × {_plural(data['columns'], 'column')}")
        if data.get("quality_score") is not None:
            seen.append(f"quality {data['quality_score']}/100")
        if sampling.get("sampled") and sampling.get("working_rows") is not None:
            seen.append(f"a {sampling.get('strategy')} sample of {_n(sampling['working_rows'])} of "
                        f"{_n(sampling.get('input_rows'))} rows")
        step = f" at step {r['step_no']}" if r.get("step_no") is not None else ""
        if not seen:
            return f"{name}'s {who} recorded its understanding{step}, with no decision to add."
        return f"{name}'s {who} recorded its understanding{step}: " + ", ".join(seen) + "."

    def swarm_understanding(self, r: dict[str, Any]) -> list[str]:
        bits: list[str] = []
        if r.get("steps_total") is not None and r.get("clones") is not None:
            bits.append(f"{r['steps_total']} specialist steps across {r['clones']} clones")
        ver = r.get("verification") or {}
        if ver.get("checks_run"):
            bits.append(f"{_plural(ver['checks_run'], 'check')} run ({ver.get('passed', 0)} passed, "
                        f"{ver.get('warned', 0)} warned, {ver.get('failed', 0)} failed)")
        merge = r.get("merge") or {}
        if merge.get("agreement") and merge.get("verdict"):
            bits.append(f"the clones were {merge['agreement']} on {str(merge['verdict']).upper()}")
        model = r.get("model") or {}
        if model.get("calls"):
            bits.append(f"{_plural(model['calls'], 'model call')}")
        return ["Across the team: " + "; ".join(bits) + "."] if bits else []


#: Record type → handler. Extend to narrate record types this module does
#: not know; an unregistered type produces no line.
HANDLERS: dict[str, Handler] = {
    "swarm.requested": SwarmNarrative.requested,
    "swarm.sharded": SwarmNarrative.sharded,
    "clone.dispatched": SwarmNarrative.dispatched,
    "clone.step.started": SwarmNarrative.step_started,
    "clone.step.completed": SwarmNarrative.step_completed,
    "clone.returned": SwarmNarrative.returned,
    "swarm.merged": SwarmNarrative.merged,
    "swarm.finished": SwarmNarrative.finished,
    "agent.understanding": SwarmNarrative.agent_understanding,
    "swarm.understanding": SwarmNarrative.swarm_understanding,
}


def narrate(records: Iterable[dict[str, Any]], level: int = DEBUG) -> list[str]:
    """The story of ``records`` at ``level``, as plain sentences. Pure."""
    story = SwarmNarrative(level)
    lines: list[str] = []
    for record in records:
        lines.extend(story.feed(record))
    return lines
