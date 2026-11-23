// The live swarm demo: Avaloka clones herself, the clones attack their shards,
// and she converges their answers. Driven by the swarm API through
// useSwarmMission; drawn in her voice, with status said in words, not only colour.
//
// Used by the /swarm page and reachable from the agent-trace page. When opened
// from an analysis it offers that thread's dataset as the source.

import { useEffect, useMemo, useState } from "react";
import { cx } from "@/lib/utils/cx";
import { backendApi } from "@/lib/api/backendApi";
import { fmtCost, fmtMs } from "@/lib/agent-trace/layout";
import {
  SWARM_SAMPLES,
  type CloneView,
  type StepView,
  type SwarmCloneDoc,
  type SwarmEpisodes,
  type SwarmMissionDoc,
  type SwarmMissionRequest,
  type SwarmMissionSummary,
  type SwarmMissionView,
  type SwarmTelemetry,
  chargeLabel,
} from "@/lib/agent-trace/swarm";
import type { SwarmCapability } from "@/lib/agent-trace/types";
import { POLL_MS, useSwarmMission } from "@/lib/agent-trace/useSwarmMission";
import { SwarmLimitNote } from "./AgentTracePanel";

export type SwarmDemoApi = Pick<
  typeof backendApi,
  | "startSwarmMission"
  | "getSwarmRecords"
  | "getSwarmMission"
  | "listSwarmMissions"
  | "getSwarmClone"
  | "getSwarmEpisodes"
  | "getSwarmTelemetry"
  | "getCapabilities"
>;

export interface SwarmDemoPanelProps {
  /** The analysis this was opened from; its dataset becomes a source option. */
  threadId?: string | null;
  datasetName?: string | null;
  /** A finished mission to open at once (from the mission list or a link). */
  missionId?: string | null;
  className?: string;
  /** Injected in tests; defaults to the real backend client. */
  api?: SwarmDemoApi;
  pollMs?: number;
}

type Tab = "live" | "clones" | "episodes" | "telemetry";

function Chip({ tone = "neutral", children }: { tone?: "neutral" | "good" | "bad" | "brand" | "warn" | "purple"; children: React.ReactNode }) {
  const tones = {
    neutral: "bg-utility-neutral-50 text-utility-neutral-700 ring-utility-neutral-200",
    good: "bg-utility-green-50 text-utility-green-700 ring-utility-green-200",
    bad: "bg-utility-red-50 text-utility-red-700 ring-utility-red-200",
    brand: "bg-utility-brand-50 text-utility-brand-700 ring-utility-brand-200",
    warn: "bg-utility-orange-50 text-utility-orange-700 ring-utility-orange-200",
    purple: "bg-utility-purple-50 text-utility-purple-700 ring-utility-purple-200",
  } as const;
  return (
    <span className={cx("inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset", tones[tone])}>
      {children}
    </span>
  );
}

function verdictTone(v: string | null | undefined): "good" | "bad" | "neutral" {
  if (v === "pass" || v === "done") return "good";
  if (v === "fail" || v === "failed") return "bad";
  return "neutral";
}

const CLONE_STATUS: Record<CloneView["status"], { text: string; tone: "neutral" | "brand" | "good" | "bad" }> = {
  dispatched: { text: "dispatched", tone: "neutral" },
  working: { text: "working", tone: "brand" },
  done: { text: "returned", tone: "good" },
  failed: { text: "failed", tone: "bad" },
};

function fireflyName(f: string): string {
  return f.replace(/_/g, " ");
}

function StepRow({ step }: { step: StepView }) {
  const running = step.status === "running";
  return (
    <li
      className={cx(
        "rounded-md px-2 py-1.5",
        running ? "animate-pulse bg-utility-brand-50" : step.status === "failed" ? "bg-utility-red-50" : "bg-secondary",
      )}
      data-step-status={step.status}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="w-5 shrink-0 text-right font-mono text-[10px] text-tertiary">{step.stepNo}</span>
        <span className="text-xs font-semibold text-primary">{fireflyName(step.firefly)}</span>
        {step.humanRole ? <span className="text-xs text-tertiary">· {step.humanRole}</span> : null}
        <span className="ml-auto text-[11px] text-tertiary">
          {running ? "running…" : `${step.status === "failed" ? "failed" : "done"} · ${fmtMs(step.durationMs)}`}
          {!running && (step.computeCostUsd != null || step.modelCostUsd != null)
            ? ` · ${fmtCost((step.computeCostUsd ?? 0) + (step.modelCostUsd ?? 0))}`
            : ""}
        </span>
      </div>
      {step.output ? <p className="mt-0.5 pl-7 text-xs text-secondary">{step.output}</p> : null}
      {step.errorClass ? <p className="mt-0.5 pl-7 text-xs text-error-primary">{step.errorClass}</p> : null}
    </li>
  );
}

function CloneCard({ clone, train }: { clone: CloneView; train: boolean }) {
  const status = CLONE_STATUS[clone.status];
  return (
    <li
      className={cx(
        "rounded-xl border bg-primary p-3 shadow-xs",
        clone.selected ? "border-utility-brand-500 ring-2 ring-utility-brand-200" : "border-secondary",
      )}
      data-testid="clone-card"
      data-clone-id={clone.cloneId}
      aria-label={`${clone.name}: ${status.text}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-primary">{clone.name}</span>
          <span className="text-xs text-tertiary">shard {clone.shardId ?? "?"}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {clone.selected ? <Chip tone="brand">selected by the merge</Chip> : null}
          <Chip tone={status.tone}>{status.text}</Chip>
        </div>
      </div>
      <p className="mt-1 text-xs text-tertiary">
        {clone.rows != null ? `${clone.rows.toLocaleString()} rows` : "rows —"} · {clone.columns.length} columns
      </p>
      <ol className="mt-2 space-y-1" aria-label={`${clone.name} steps`}>
        {clone.steps.map((s) => (
          <StepRow key={s.stepId} step={s} />
        ))}
      </ol>
      {clone.status === "done" || clone.status === "failed" ? (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-secondary">
          <span>verdict</span>
          <Chip tone={verdictTone(clone.verdict)}>{clone.verdict ?? "—"}</Chip>
          {train ? (
            <span>
              model score <span className="font-mono text-primary">{clone.score != null ? clone.score.toPrecision(4) : "—"}</span>
            </span>
          ) : null}
          {clone.maxSafeDeploymentLevel != null ? <span>max safe level {clone.maxSafeDeploymentLevel}</span> : null}
          <span className="ml-auto text-tertiary">
            {fmtMs(clone.durationMs)} · {chargeLabel({ compute: clone.computeCostUsd, model: clone.modelCostUsd, total: clone.costUsd })}
          </span>
        </div>
      ) : null}
    </li>
  );
}

function Convergence({ view }: { view: SwarmMissionView }) {
  const merge = view.merge;
  if (!merge) return null;
  const nameOf = (id: string) => view.clones.find((c) => c.cloneId === id)?.name ?? id;
  return (
    <section className="rounded-xl border border-secondary bg-primary p-4 shadow-xs" aria-label="Convergence" data-testid="convergence">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-primary">Convergence</h3>
        <Chip tone={verdictTone(merge.verdict)}>verdict {merge.verdict ?? "—"}</Chip>
        {merge.rules.verdict ? <span className="text-xs text-tertiary">rule: {merge.rules.verdict}</span> : null}
        {merge.rules.profile ? <span className="text-xs text-tertiary">· profile: {merge.rules.profile}</span> : null}
        {merge.selectedModel ? <span className="text-xs text-tertiary">· model: {merge.rules.model}</span> : null}
        <span className="ml-auto text-xs text-tertiary">
          {merge.nRows != null ? `${merge.nRows.toLocaleString()} rows` : ""} · {merge.columns.length} columns
        </span>
      </div>
      <table className="mt-2 min-w-full text-xs" aria-label="Contributions">
        <thead>
          <tr className="text-left text-tertiary">
            <th className="py-1 pr-3 font-medium">Clone</th>
            <th className="pr-3 font-medium">Rows</th>
            <th className="pr-3 font-medium">Verdict</th>
            <th className="pr-3 font-medium">Compute / model</th>
            <th className="font-medium">Selected</th>
          </tr>
        </thead>
        <tbody>
          {merge.contributions.map((c) => (
            <tr
              key={c.cloneId}
              className={cx("border-t border-secondary", c.selected && "bg-utility-brand-50 font-semibold")}
              data-selected={c.selected ? "true" : "false"}
            >
              <td className="py-1 pr-3 text-primary">{nameOf(c.cloneId)}</td>
              <td className="pr-3">{c.rows?.toLocaleString() ?? "—"}</td>
              <td className="pr-3">
                <Chip tone={verdictTone(c.verdict)}>{c.verdict ?? "—"}</Chip>
              </td>
              <td className="pr-3">
                {(() => {
                  // The merge lists each clone's total; its split is on the clone's own record.
                  const own = view.clones.find((x) => x.cloneId === c.cloneId);
                  return chargeLabel({ compute: own?.computeCostUsd, model: own?.modelCostUsd, total: c.costUsd });
                })()}
              </td>
              <td>{c.selected ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {merge.selectedModel ? (
        <p className="mt-2 text-xs text-secondary">
          Selected model:{" "}
          <code className="font-mono">
            {Object.entries(merge.selectedModel)
              .filter(([, v]) => v != null && typeof v !== "object")
              .map(([k, v]) => `${k}=${String(v)}`)
              .join(" ")}
          </code>
        </p>
      ) : null}
      {view.finished ? (
        <p className="mt-2 text-xs text-secondary" data-testid="finished">
          <Chip tone={view.finished.status === "failed" ? "bad" : "good"}>{view.finished.status}</Chip> in {fmtMs(view.finished.durationMs)}
          {view.finished.economics ? <Economics economics={view.finished.economics} /> : null}
        </p>
      ) : null}
    </section>
  );
}

function Economics({ economics }: { economics: Record<string, unknown> }) {
  const n = (k: string) => (typeof economics[k] === "number" ? (economics[k] as number) : null);
  const parts = [
    n("estimated_manual_effort_hours") != null ? `${n("estimated_manual_effort_hours")} h of manual effort (rough heuristic estimate, not measured)` : null,
    n("compute_cost_usd") != null || n("model_cost_usd") != null || n("total_mission_cost_usd") != null
      ? chargeLabel({ compute: n("compute_cost_usd"), model: n("model_cost_usd"), charge: n("mission_charge_usd"), total: n("total_mission_cost_usd") })
      : null,
    n("human_review_hours") != null ? `${n("human_review_hours")} h of review left to a human` : null,
    n("economic_multiplier") != null ? `multiplier ${n("economic_multiplier")}×` : null,
  ].filter(Boolean);
  return parts.length ? <span className="ml-2 text-tertiary">· {parts.join(" · ")}</span> : null;
}

function Narration({ view }: { view: SwarmMissionView }) {
  return (
    <section className="rounded-xl border border-secondary bg-primary p-4 shadow-xs" aria-label="Avaloka" data-testid="narration">
      <div className="flex items-center gap-2">
        <span className="inline-flex size-7 items-center justify-center rounded-full bg-utility-brand-50 text-xs font-bold text-utility-brand-700" aria-hidden>
          A
        </span>
        <h3 className="text-sm font-semibold text-primary">Avaloka</h3>
        <Chip tone={view.status === "running" ? "brand" : view.status === "failed" ? "bad" : view.status === "done" ? "good" : "neutral"}>
          {view.status === "idle" ? "ready" : view.status}
        </Chip>
        {view.goal ? <span className="text-xs text-tertiary">“{view.goal}”</span> : null}
      </div>
      <ol className="mt-2 space-y-1" role="log" aria-live="polite" aria-relevant="additions">
        {view.narration.map((line, i) => (
          <li
            key={`${line.seq}-${i}`}
            role={line.tone === "warn" ? "note" : undefined}
            className={cx(
              "rounded-md px-2 py-1 text-sm",
              line.tone === "warn" && "bg-utility-orange-50 text-utility-orange-700",
              line.tone === "bad" && "bg-utility-red-50 text-utility-red-700",
              line.tone === "good" && "text-primary",
              line.tone === "neutral" && "text-secondary",
            )}
          >
            {line.text}
          </li>
        ))}
        {view.status === "idle" && !view.narration.length ? (
          <li className="text-sm text-tertiary">Pick a dataset and a goal, choose how many copies of me to send, and start.</li>
        ) : null}
      </ol>
    </section>
  );
}

function MissionList({ missions, onOpen, current }: { missions: SwarmMissionSummary[]; onOpen: (id: string) => void; current: string | null }) {
  if (!missions.length) return <p className="text-sm text-tertiary">No missions yet.</p>;
  return (
    <ul className="divide-y divide-secondary rounded-xl border border-secondary bg-primary" aria-label="Missions">
      {missions.map((m) => (
        <li key={m.mission_id} className={cx("flex flex-wrap items-center gap-2 px-3 py-2", m.mission_id === current && "bg-secondary")}>
          <Chip tone={m.status === "done" ? "good" : m.status === "failed" ? "bad" : "brand"}>{m.status ?? "?"}</Chip>
          <span className="text-xs text-tertiary">{m.kind}</span>
          <span className="min-w-0 flex-1 truncate text-sm text-primary">{m.goal}</span>
          <span className="text-xs text-tertiary">
            {m.clones} clone{m.clones === 1 ? "" : "s"}
            {m.verdict ? ` · ${m.verdict}` : ""}
          </span>
          <button
            type="button"
            className="rounded-md border border-secondary px-2 py-0.5 text-xs font-semibold text-secondary hover:bg-primary_hover"
            onClick={() => onOpen(m.mission_id)}
          >
            Open
          </button>
        </li>
      ))}
    </ul>
  );
}

function Json({ value }: { value: unknown }) {
  return <pre className="max-h-[60vh] overflow-auto rounded-lg border border-secondary bg-secondary px-3 py-2 font-mono text-xs text-primary">{JSON.stringify(value, null, 2)}</pre>;
}

/** The per-clone document, the RL episodes, or the telemetry projection of a mission. */
function MissionDetail({ tab, missionId, clones, api }: { tab: Tab; missionId: string; clones: CloneView[]; api: SwarmDemoApi }) {
  const [cloneId, setCloneId] = useState<string | null>(clones[0]?.cloneId ?? null);
  const [data, setData] = useState<SwarmCloneDoc | SwarmEpisodes | SwarmTelemetry | SwarmMissionDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!cloneId && clones[0]) setCloneId(clones[0].cloneId);
  }, [clones, cloneId]);
  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    const load =
      tab === "clones"
        ? cloneId
          ? api.getSwarmClone(missionId, cloneId)
          : api.getSwarmMission(missionId)
        : tab === "episodes"
          ? api.getSwarmEpisodes(missionId)
          : api.getSwarmTelemetry(missionId);
    load.then((d) => live && setData(d)).catch((e: Error) => live && setError(e?.message || "Could not load."));
    return () => {
      live = false;
    };
  }, [api, tab, missionId, cloneId]);
  return (
    <div className="space-y-2" role="tabpanel">
      {tab === "clones" ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-tertiary">Clone</span>
          {clones.map((c) => (
            <button
              key={c.cloneId}
              type="button"
              aria-pressed={c.cloneId === cloneId}
              className={cx(
                "rounded-md px-2 py-0.5 text-xs font-semibold",
                c.cloneId === cloneId ? "bg-secondary text-primary" : "text-tertiary hover:bg-primary_hover",
              )}
              onClick={() => setCloneId(c.cloneId)}
            >
              {c.name}
            </button>
          ))}
        </div>
      ) : null}
      {tab === "episodes" ? (
        <p className="text-xs text-tertiary">One episode per clone. `reward` is null on purpose: the learned layer assigns it.</p>
      ) : null}
      {tab === "telemetry" ? (
        <p className="text-xs text-tertiary">
          Exactly what telemetry would send for this mission, and nothing is sent: agent names, timings, outcomes and the fan-out. No goal,
          no column, no output, no path.
        </p>
      ) : null}
      {error ? <p className="text-sm text-error-primary">{error}</p> : data ? <Json value={data} /> : <p className="text-sm text-tertiary">Loading…</p>}
    </div>
  );
}

export function SwarmDemoPanel({ threadId, datasetName, missionId: initialMissionId, className, api = backendApi, pollMs = POLL_MS }: SwarmDemoPanelProps) {
  const mission = useSwarmMission({ api, pollMs });
  const [swarm, setSwarm] = useState<SwarmCapability | null>(null);
  const [missions, setMissions] = useState<SwarmMissionSummary[]>([]);
  const [tab, setTab] = useState<Tab>("live");

  const [source, setSource] = useState<string>(threadId ? "thread" : "churn");
  const [goal, setGoal] = useState("What drives churn, and how confident can I be?");
  const [kind, setKind] = useState<"analyze" | "train">("analyze");
  const [workers, setWorkers] = useState(2);
  const [target, setTarget] = useState("churned");
  const [metric, setMetric] = useState("");

  const cap = swarm?.limits?.max_clones_per_mission ?? null;
  const maxWorkers = cap ?? 16;

  useEffect(() => {
    let live = true;
    api
      .getCapabilities()
      .then((caps) => live && setSwarm(caps?.features?.swarm ?? null))
      .catch(() => live && setSwarm(null));
    return () => {
      live = false;
    };
  }, [api]);

  const refreshMissions = useMemo(
    () => () =>
      api
        .listSwarmMissions()
        .then((r) => setMissions(r.missions ?? []))
        .catch(() => undefined),
    [api],
  );
  useEffect(() => {
    void refreshMissions();
  }, [refreshMissions]);
  // A finished mission lands in the list.
  useEffect(() => {
    if (mission.view.status === "done" || mission.view.status === "failed") void refreshMissions();
  }, [mission.view.status, refreshMissions]);
  useEffect(() => {
    if (initialMissionId) void mission.open(initialMissionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- open once for the id the page was given
  }, [initialMissionId]);
  useEffect(() => {
    if (cap != null && workers > cap) setWorkers(cap);
  }, [cap, workers]);

  const submit = (ev: React.FormEvent) => {
    ev.preventDefault();
    const body: SwarmMissionRequest = {
      source: source === "thread" && threadId ? { thread_id: threadId } : { sample: source },
      goal,
      kind,
      workers,
      ...(kind === "train" ? { target, ...(metric ? { metric } : {}) } : {}),
    };
    setTab("live");
    void mission.start(body);
  };

  const view = mission.view;
  const busy = mission.starting || view.status === "running";
  const current = view.missionId ?? mission.started?.mission_id ?? null;
  const tabs: Array<{ id: Tab; label: string }> = [
    { id: "live", label: "Live" },
    { id: "clones", label: "Clone documents" },
    { id: "episodes", label: "RL episodes" },
    { id: "telemetry", label: "Telemetry (what would leave)" },
  ];

  return (
    <div className={cx("space-y-4", className)} data-testid="swarm-demo">
      <form onSubmit={submit} className="grid gap-3 rounded-xl border border-secondary bg-primary p-4 shadow-xs md:grid-cols-[1fr_2fr]" aria-label="Start a mission">
        <label className="flex flex-col gap-1 text-xs text-tertiary">
          Dataset
          <select
            className="rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            disabled={busy}
          >
            {threadId ? <option value="thread">{datasetName ? `This analysis: ${datasetName}` : "This analysis's dataset"}</option> : null}
            {SWARM_SAMPLES.map((s) => (
              <option key={s} value={s}>
                {s} (bundled sample)
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-tertiary">
          Goal
          <input
            className="rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            disabled={busy}
            required
          />
        </label>
        <div className="flex flex-wrap items-end gap-3 md:col-span-2">
          <label className="flex flex-col gap-1 text-xs text-tertiary">
            Kind
            <select
              className="rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
              value={kind}
              onChange={(e) => setKind(e.target.value as "analyze" | "train")}
              disabled={busy}
            >
              <option value="analyze">analyze</option>
              <option value="train">train</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-tertiary">
            Clones{cap != null ? ` (up to ${cap})` : ""}
            <input
              type="number"
              min={1}
              max={maxWorkers}
              className="w-20 rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
              value={workers}
              onChange={(e) => setWorkers(Math.max(1, Math.min(maxWorkers, Number(e.target.value) || 1)))}
              disabled={busy}
            />
          </label>
          {kind === "train" ? (
            <>
              <label className="flex flex-col gap-1 text-xs text-tertiary">
                Target column
                <input
                  className="w-36 rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  disabled={busy}
                  required
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-tertiary">
                Metric (optional)
                <input
                  className="w-28 rounded-md border border-secondary bg-primary px-2 py-1.5 text-sm text-primary"
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  disabled={busy}
                />
              </label>
            </>
          ) : null}
          <button
            type="submit"
            disabled={busy || !goal.trim()}
            className="rounded-lg bg-brand-solid px-3 py-1.5 text-sm font-semibold text-white shadow-xs hover:bg-brand-solid_hover disabled:opacity-50"
          >
            {mission.starting ? "Starting…" : view.status === "running" ? "Running…" : "Start"}
          </button>
          {swarm?.limits?.summary ? <span className="text-xs text-tertiary">{swarm.limits.summary}</span> : null}
        </div>
      </form>

      {mission.error ? (
        <p role="alert" className="text-sm text-error-primary">
          {mission.error}
        </p>
      ) : null}
      {mission.started?.grant?.limited && mission.started.grant.message ? (
        <SwarmLimitNote swarm={{ enabled: true, limited: true, message: mission.started.grant.message }} />
      ) : null}
      {mission.started?.persistence === "memory" ? (
        <p className="text-xs text-tertiary">This mission is held in memory: without Redis it is gone when the API restarts.</p>
      ) : null}

      {current ? (
        <div role="tablist" aria-label="Mission views" className="inline-flex rounded-lg border border-secondary bg-primary p-1 shadow-xs">
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={cx("rounded-md px-3 py-1.5 text-sm font-semibold transition", tab === t.id ? "bg-secondary text-primary" : "text-tertiary")}
            >
              {t.label}
            </button>
          ))}
        </div>
      ) : null}

      {tab === "live" || !current ? (
        <div className="space-y-4" role="tabpanel">
          <Narration view={view} />
          {view.clones.length ? (
            <ul className="grid gap-3 md:grid-cols-2" aria-label="Clones">
              {view.clones.map((c) => (
                <CloneCard key={c.cloneId} clone={c} train={view.kind === "train"} />
              ))}
            </ul>
          ) : null}
          <Convergence view={view} />
        </div>
      ) : (
        <MissionDetail tab={tab} missionId={current} clones={view.clones} api={api} />
      )}

      <section className="space-y-2" aria-label="Mission list">
        <h3 className="text-sm font-semibold text-primary">Missions</h3>
        <MissionList missions={missions} current={current} onOpen={(id) => { setTab("live"); void mission.open(id); }} />
      </section>
    </div>
  );
}
