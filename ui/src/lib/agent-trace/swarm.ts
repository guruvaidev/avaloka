// The swarm trace as the UI reads it: the records `/api/swarm/missions/{id}/records`
// serves, and a pure reducer that folds them into the view the demo draws.
//
// The backend owns the record shapes (avaloka/swarm_trace.py,
// docs/design/AGENT_TRACE.md §6); these types describe only what the UI reads.
// Records are append-only with a dense `seq` starting at 1, so the reducer is
// incremental: `reduceSwarm(newRecords, previousView)` applies only the records
// past the view's cursor, and a page received twice changes nothing.

import type { SwarmLimits } from "./types";

export type SwarmRecordType =
  | "swarm.requested"
  | "swarm.sharded"
  | "clone.dispatched"
  | "clone.step.started"
  | "clone.step.completed"
  | "clone.returned"
  | "swarm.merged"
  | "swarm.finished";

export interface SwarmRecord {
  seq: number;
  ts: string;
  type: SwarmRecordType | string;
  mission_id: string;
  clone_id?: string;
  step_id?: string;
  [k: string]: unknown;
}

export interface SwarmRecordsPage {
  mission_id: string;
  status: "running" | "done" | "failed" | string | null;
  records: SwarmRecord[];
  cursor: number;
  done: boolean;
}

export interface SwarmMissionRequest {
  source: { sample: string } | { thread_id: string };
  goal: string;
  kind: "analyze" | "train";
  workers?: number;
  max_parallel?: number;
  target?: string;
  metric?: string;
}

/** The POST response's grant (`SwarmGrant.as_dict()`), not the record's. */
export interface SwarmStartGrant {
  requested_clones: number;
  clones: number;
  requested_parallel?: number | null;
  parallel?: number | null;
  limited: boolean;
  message?: string | null;
  limits?: SwarmLimits;
}

export interface SwarmStartResponse {
  mission_id: string;
  status: "running" | string;
  grant: SwarmStartGrant;
  limits: SwarmLimits;
  source?: { kind: string; sample?: string; thread_id?: string };
  persistence?: "redis" | "memory" | string;
}

export interface SwarmMissionSummary {
  mission_id: string;
  status: "running" | "done" | "failed" | string | null;
  kind?: string | null;
  goal?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  clones: number;
  verdict?: string | null;
}

export interface SwarmCloneStepDoc {
  step_id: string;
  step_no?: number | null;
  firefly?: string | null;
  human_role?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  status?: "running" | "ok" | "failed" | string | null;
  output?: string | null;
  duration_ms?: number | null;
  manual_minutes?: number | null;
  compute_cost_usd?: number | null;
  model_cost_usd?: number | null;
  metadata?: Record<string, unknown>;
  error_class?: string | null;
  error?: string | null;
}

export interface SwarmCloneDoc {
  clone_id: string;
  name?: string | null;
  shard_id?: number | null;
  status?: "running" | "done" | "failed" | string | null;
  rows?: number | null;
  received?: Record<string, unknown>;
  dispatched_at?: string | null;
  returned_at?: string | null;
  steps: SwarmCloneStepDoc[];
  verdict?: string | null;
  max_safe_deployment_level?: number | null;
  model?: Record<string, unknown> | null;
  sampling?: Record<string, unknown> | null;
  cost_usd?: number | null;
  duration_ms?: number | null;
  error_class?: string | null;
  error?: string | null;
  mission_id?: string;
  records?: SwarmRecord[];
}

export interface SwarmMissionDoc {
  schema_version: string;
  mission_id: string;
  kind?: string | null;
  goal?: string | null;
  source?: string | null;
  status: "running" | "done" | "failed" | string;
  started_at?: string | null;
  ended_at?: string | null;
  limits?: SwarmLimits | null;
  grant?: Record<string, unknown> | null;
  sharding?: Record<string, unknown> | null;
  clones: SwarmCloneDoc[];
  merge?: Record<string, unknown> | null;
  economics?: Record<string, unknown> | null;
  duration_ms?: number | null;
  error_class?: string | null;
  persistence?: "redis" | "memory" | string;
  persistence_note?: string | null;
  records?: SwarmRecord[];
}

export interface SwarmEpisodes {
  mission_id: string;
  status: string | null;
  episodes: Array<Record<string, unknown>>;
}

export interface SwarmTelemetry {
  mission_id: string;
  status: string | null;
  events: Array<Record<string, unknown>>;
}

// ------------------------------------------------------------------ the view

export interface StepView {
  stepId: string;
  stepNo: number;
  firefly: string;
  humanRole: string | null;
  status: "running" | "ok" | "failed";
  output: string | null;
  durationMs: number | null;
  manualMinutes: number | null;
  computeCostUsd: number | null;
  modelCostUsd: number | null;
  metadata: Record<string, unknown>;
  errorClass: string | null;
}

export interface CloneView {
  cloneId: string;
  name: string;
  shardId: number | null;
  rows: number | null;
  columns: string[];
  status: "dispatched" | "working" | "done" | "failed";
  steps: StepView[];
  verdict: string | null;
  maxSafeDeploymentLevel: number | null;
  /** The trained model's summary (`score`, `name`…) on a training mission. */
  model: Record<string, unknown> | null;
  score: number | null;
  costUsd: number | null;
  /** The clone's nominal local-CPU charge and its model calls (clone.returned). */
  computeCostUsd: number | null;
  modelCostUsd: number | null;
  durationMs: number | null;
  errorClass: string | null;
  selected: boolean;
}

export interface ContributionView {
  cloneId: string;
  rows: number | null;
  verdict: string | null;
  selected: boolean;
  costUsd: number | null;
}

export interface MergeView {
  rules: { profile?: string; verdict?: string; model?: string };
  verdict: string | null;
  maxSafeDeploymentLevel: number | null;
  selectedModel: Record<string, unknown> | null;
  selectedCloneId: string | null;
  contributions: ContributionView[];
  nRows: number | null;
  columns: string[];
}

export interface SwarmMissionView {
  missionId: string | null;
  status: "idle" | "running" | "done" | "failed";
  kind: string | null;
  goal: string | null;
  request: {
    requestedClones: number | null;
    grantedClones: number | null;
    grantedParallel: number | null;
    limited: boolean;
    message: string | null;
    limits: SwarmLimits | null;
  } | null;
  sharding: { clones: number; totalRows: number | null; rowsPerClone: number[]; columns: string[] } | null;
  clones: CloneView[];
  merge: MergeView | null;
  finished: { status: string; durationMs: number | null; economics: Record<string, unknown> | null; errorClass: string | null } | null;
  /** The last `seq` folded in; the next poll asks for records after it. */
  cursor: number;
  recordCount: number;
  /** What Avaloka says as the fleet works, in her voice, oldest first. */
  narration: Array<{ seq: number; text: string; tone: "neutral" | "warn" | "good" | "bad" }>;
}

export function emptySwarm(): SwarmMissionView {
  return {
    missionId: null,
    status: "idle",
    kind: null,
    goal: null,
    request: null,
    sharding: null,
    clones: [],
    merge: null,
    finished: null,
    cursor: 0,
    recordCount: 0,
    narration: [],
  };
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);
const obj = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

function plural(n: number | null, word: string): string {
  return `${n ?? "?"} ${word}${n === 1 ? "" : "s"}`;
}

function durationText(ms: number | null): string {
  if (ms == null) return "—";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** The view after `records`, applied in `seq` order on top of `prev`. Pure. */
export function reduceSwarm(records: SwarmRecord[], prev: SwarmMissionView = emptySwarm()): SwarmMissionView {
  const fresh = records.filter((r) => typeof r.seq === "number" && r.seq > prev.cursor).sort((a, b) => a.seq - b.seq);
  if (!fresh.length) return prev;
  // Copy what will be written to; the previous view is never changed.
  const view: SwarmMissionView = {
    ...prev,
    clones: prev.clones.map((c) => ({ ...c, steps: c.steps.map((s) => ({ ...s })) })),
    narration: [...prev.narration],
  };
  const say = (seq: number, text: string, tone: SwarmMissionView["narration"][number]["tone"] = "neutral") =>
    view.narration.push({ seq, text, tone });
  const clone = (id: string): CloneView => {
    let c = view.clones.find((x) => x.cloneId === id);
    if (!c) {
      c = {
        cloneId: id, name: id, shardId: null, rows: null, columns: [], status: "dispatched", steps: [],
        verdict: null, maxSafeDeploymentLevel: null, model: null, score: null, costUsd: null,
        computeCostUsd: null, modelCostUsd: null, durationMs: null,
        errorClass: null, selected: false,
      };
      view.clones.push(c);
    }
    return c;
  };

  for (const r of fresh) {
    view.cursor = r.seq;
    view.recordCount += 1;
    if (!view.missionId && r.mission_id) view.missionId = r.mission_id;
    switch (r.type) {
      case "swarm.requested": {
        const granted = num(r.granted_clones);
        const limited = r.limited === true;
        view.status = "running";
        view.kind = str(r.kind);
        view.goal = str(r.goal);
        view.request = {
          requestedClones: num(r.requested_clones),
          grantedClones: granted,
          grantedParallel: num(r.granted_parallel),
          limited,
          message: str(r.message),
          limits: (obj(r.limits) as SwarmLimits | null) ?? null,
        };
        say(r.seq, `Cloning myself into ${granted ?? "?"} ${granted === 1 ? "copy" : "copies"} — each takes a shard of the data and works it the way I would.`);
        if (limited && view.request.message) say(r.seq, view.request.message, "warn");
        break;
      }
      case "swarm.sharded": {
        const clones = num(r.clones) ?? 0;
        const rowsPerClone = Array.isArray(r.rows_per_clone) ? (r.rows_per_clone as unknown[]).map((v) => num(v) ?? 0) : [];
        view.sharding = { clones, totalRows: num(r.total_rows), rowsPerClone, columns: strs(r.columns) };
        say(r.seq, `Sharded ${num(r.total_rows)?.toLocaleString() ?? "?"} rows into ${plural(clones, "shard")} of ${rowsPerClone.map((n) => n.toLocaleString()).join(" / ")} rows; ${plural(view.sharding.columns.length, "column")}.`);
        break;
      }
      case "clone.dispatched": {
        const c = clone(String(r.clone_id));
        const received = obj(r.received) ?? {};
        c.name = str(r.name) ?? c.cloneId;
        c.shardId = num(r.shard_id);
        c.rows = num(received.rows);
        c.columns = strs(received.columns);
        c.status = "dispatched";
        say(r.seq, `${c.name} takes shard ${c.shardId ?? "?"}: ${plural(c.rows, "row")}, ${plural(c.columns.length, "column")}.`);
        break;
      }
      case "clone.step.started": {
        const c = clone(String(r.clone_id));
        c.status = "working";
        const stepId = String(r.step_id ?? `${c.cloneId}/s${r.step_no ?? c.steps.length + 1}`);
        if (!c.steps.some((s) => s.stepId === stepId)) {
          c.steps.push({
            stepId, stepNo: num(r.step_no) ?? c.steps.length + 1, firefly: str(r.firefly) ?? "step",
            humanRole: str(r.human_role), status: "running", output: null, durationMs: null, manualMinutes: null,
            computeCostUsd: null, modelCostUsd: null, metadata: {}, errorClass: null,
          });
        }
        break;
      }
      case "clone.step.completed": {
        const c = clone(String(r.clone_id));
        const stepId = String(r.step_id ?? `${c.cloneId}/s${r.step_no ?? c.steps.length}`);
        let s = c.steps.find((x) => x.stepId === stepId);
        if (!s) {
          s = { stepId, stepNo: num(r.step_no) ?? c.steps.length + 1, firefly: "step", humanRole: null, status: "running",
                output: null, durationMs: null, manualMinutes: null, computeCostUsd: null, modelCostUsd: null, metadata: {}, errorClass: null };
          c.steps.push(s);
        }
        s.firefly = str(r.firefly) ?? s.firefly;
        s.humanRole = str(r.human_role) ?? s.humanRole;
        s.status = r.outcome === "failed" ? "failed" : "ok";
        s.output = str(r.output);
        s.durationMs = num(r.duration_ms);
        s.manualMinutes = num(r.manual_minutes);
        s.computeCostUsd = num(r.compute_cost_usd);
        s.modelCostUsd = num(r.model_cost_usd);
        s.metadata = obj(r.metadata) ?? {};
        s.errorClass = str(r.error_class);
        if (s.status === "failed") say(r.seq, `${c.name}: ${s.firefly} failed${s.errorClass ? ` (${s.errorClass})` : ""}.`, "bad");
        break;
      }
      case "clone.returned": {
        const c = clone(String(r.clone_id));
        c.status = r.status === "failed" ? "failed" : "done";
        c.verdict = str(r.verdict);
        c.maxSafeDeploymentLevel = num(r.max_safe_deployment_level);
        c.model = obj(r.model);
        c.score = num(c.model?.score);
        c.costUsd = num(r.cost_usd);
        c.computeCostUsd = num(r.compute_cost_usd);
        c.modelCostUsd = num(r.model_cost_usd);
        c.durationMs = num(r.duration_ms);
        c.errorClass = str(r.error_class);
        for (const s of c.steps) if (s.status === "running") s.status = c.status === "failed" ? "failed" : "ok";
        say(
          r.seq,
          c.status === "failed"
            ? `${c.name} came back without an answer${c.errorClass ? ` (${c.errorClass})` : ""}.`
            : `${c.name} is back in ${durationText(c.durationMs)}: verdict ${c.verdict ?? "—"}${c.score != null ? `, score ${c.score}` : ""}.`,
          c.status === "failed" ? "bad" : "good",
        );
        break;
      }
      case "swarm.merged": {
        const rules = obj(r.rules) ?? {};
        const contributions = (Array.isArray(r.contributions) ? (r.contributions as unknown[]) : [])
          .map(obj)
          .filter((x): x is Record<string, unknown> => !!x)
          .map((x) => ({
            cloneId: String(x.clone_id), rows: num(x.rows), verdict: str(x.verdict), selected: x.selected === true, costUsd: num(x.cost_usd),
          }));
        const selected = contributions.find((x) => x.selected)?.cloneId ?? null;
        view.merge = {
          rules: { profile: str(rules.profile) ?? undefined, verdict: str(rules.verdict) ?? undefined, model: str(rules.model) ?? undefined },
          verdict: str(r.verdict),
          maxSafeDeploymentLevel: num(r.max_safe_deployment_level),
          selectedModel: obj(r.selected_model),
          selectedCloneId: selected,
          contributions,
          nRows: num(r.n_rows),
          columns: strs(r.columns),
        };
        for (const c of view.clones) c.selected = c.cloneId === selected;
        say(
          r.seq,
          `Converging ${plural(contributions.length, "answer")} over ${num(r.n_rows)?.toLocaleString() ?? "?"} rows: verdict ${view.merge.verdict ?? "—"} (${rules.verdict ?? "merge"} rule)` +
            (selected ? `; ${view.clones.find((c) => c.cloneId === selected)?.name ?? selected}'s model is the one I stand behind.` : "."),
          view.merge.verdict === "fail" ? "bad" : "good",
        );
        break;
      }
      case "swarm.finished": {
        const status = str(r.status) ?? "done";
        view.status = status === "failed" ? "failed" : "done";
        view.finished = { status, durationMs: num(r.duration_ms), economics: obj(r.economics), errorClass: str(r.error_class) };
        say(r.seq, status === "failed"
          ? `The mission failed${view.finished.errorClass ? ` (${view.finished.errorClass})` : ""}.`
          : `Done in ${durationText(view.finished.durationMs)}.`, status === "failed" ? "bad" : "good");
        break;
      }
      default:
        break;
    }
  }
  return view;
}

/** The same view, rebuilt from a stored mission document's records. */
export function viewFromMission(doc: SwarmMissionDoc): SwarmMissionView {
  const view = reduceSwarm(doc.records ?? []);
  if (!view.missionId) view.missionId = doc.mission_id;
  if (view.status === "idle" && doc.status) view.status = doc.status === "failed" ? "failed" : doc.status === "done" ? "done" : "running";
  return view;
}

export const SWARM_SAMPLES = ["churn", "salaries", "healthcare", "youtube", "football", "taxi"] as const;

/** "$0.0012", "$0.01"; a positive amount under $0.00005 is "under $0.0001", never "$0". */
export function fmtSmallUsd(usd: number): string {
  if (usd > 0 && usd < 0.00005) return "under $0.0001";
  return `$${usd.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}`;
}

/**
 * A clone's (or a mission's) charge in words, as the CLI narration says it
 * (avaloka/swarm_narration.py): the compute figure is a nominal local-CPU rate
 * (fireflies.base.LOCAL_CPU_USD_PER_SECOND), not money spent, so it is never a
 * bare "cost". Model calls and a mission charge are named only when non-zero.
 * Records without the split (older missions) say what the total is made of.
 */
export function chargeLabel(parts: { compute?: number | null; model?: number | null; charge?: number | null; total?: number | null }): string {
  const { compute, model, charge, total } = parts;
  if (compute == null && model == null) {
    return total == null ? "—" : `${fmtSmallUsd(total)}: a nominal compute rate plus any model calls`;
  }
  const said: string[] = [];
  if (compute) said.push(`compute ${fmtSmallUsd(compute)} at a nominal CPU rate`);
  if (model) said.push(`model calls ${fmtSmallUsd(model)}`);
  if (charge) said.push(`mission charge ${fmtSmallUsd(charge)}`);
  return said.length ? said.join(" · ") : "no charge";
}
