// Shapes served by GET /threads/{id}/trace, GET /threads/{id}/planner-graph?format=json,
// GET /api/agent-metrics and the swarm entry of GET /api/capabilities.
// The backend owns them (app/trace/build.py, docs/design/AGENT_TRACE.md); these
// types only describe what the UI reads, so every field is optional-tolerant.

export interface InferenceCall {
  agent?: string;
  provider?: string | null;
  model?: string | null;
  tokens_in?: number | null;
  tokens_out?: number | null;
  cost_usd?: number | null;
  cost_source?: "provider" | "price_table" | null;
  latency_ms?: number | null;
  outcome?: "success" | "failed" | string;
  error_class?: string | null;
  tool_calls?: number;
  /** HTTP status of a failed call that moved the request to the next tier. */
  status?: number | null;
  /** Why the call failed, from the recorded status and message (app/core/model_tiers). */
  reason?: string | null;
  /** Set when a fallback tier answered in place of the configured model. */
  answered_by?: {
    configured_model?: string;
    answered_model?: string;
    provider?: string;
    reason?: string;
    note?: string;
  } | null;
}

export interface TraceOperation {
  op: string;
  formula?: string;
  line?: number | null;
  group_by?: string[];
  aggregations?: string[];
  column?: string;
  /** The columns this operation touches (a derived column comes first). */
  columns?: string[];
  method?: string;
  source?: string | null;
  target?: string | null;
}

export interface TrainedModel {
  name: string;
  type?: string | null;
  target?: string | null;
  features: string[];
  metrics: Record<string, number>;
  runtime_s?: number | null;
  status: "trained" | "failed" | string;
}

export interface EvaluationFacts {
  task?: string | null;
  primary_metric?: string | null;
  metrics: Record<string, number>;
  baseline: Record<string, number>;
  beats_baseline?: boolean | null;
  lift_over_baseline?: number | null;
  n_folds?: number | null;
  n_rows?: number | null;
  splitter?: string | null;
}

export interface SmallTable {
  columns: string[];
  rows: unknown[][];
  total_rows: number;
  truncated: boolean;
}

export interface TraceDecision {
  type: string;
  value?: string | number | boolean | null;
  reason?: string | null;
  /** Reply segments of `reason` when a model wrote it (commentary, c-strict). */
  reason_segments?: unknown;
  confidence?: number | null;
  check?: string;
  router?: string;
  chosen?: string;
  signals?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface TraceCalculation {
  operation?: string;
  code?: string | null;
  operations?: TraceOperation[];
  pseudocode?: string | null;
  data_scope?: Record<string, unknown>;
  values?: Record<string, unknown> | null;
  result?: SmallTable | null;
  findings?: Array<Record<string, unknown>>;
  answer?: string | null;
  /** The answer's reply segments: the provenance its message carried. */
  answer_segments?: unknown;
  charts?: Array<{ type?: string; title?: string; intent?: string; reason?: string | null }>;
  attempt?: number;
  reason?: string | null;
  models?: TrainedModel[];
  evaluation?: EvaluationFacts | null;
}

export interface TraceStep {
  kind: "agent" | "decision";
  step_id: string;
  parent_id: string | null;
  turn_id: string;
  seq: number;
  node: string;
  agent?: string;
  role?: string;
  clone?: string | null;
  started_at?: string;
  ended_at?: string;
  at?: string;
  after?: string | null;
  latency_ms?: number;
  status?: "ok" | "error";
  error_class?: string | null;
  inputs?: { question?: string | null; data_scope?: Record<string, unknown> };
  decision?: TraceDecision | null;
  calculation?: TraceCalculation | null;
  gates?: Array<{ gate: string; status?: string; reason?: string | null; reason_segments?: unknown }>;
  inference?: InferenceCall[];
  written?: string[];
  handoff_note?: string;
}

export interface TraceHandoff {
  from: string;
  from_step: string;
  to: string;
  to_step: string;
  parent_id: string | null;
  passed: string[];
  route?: string | null;
  reason?: string | null;
}

export interface SwarmClone {
  clone: string;
  name?: string;
  role?: string;
  focus?: string;
  status: "done" | "failed" | "not_run";
  agent_steps: string[];
  latency_ms?: number | null;
}

export interface InferenceTotals {
  llm_calls: number;
  llm_failures: number;
  tokens_in: number;
  tokens_out: number;
  llm_latency_ms: number;
  cost_usd: number | null;
  unpriced_calls: number;
}

export interface TraceTurn {
  turn_id: string;
  question?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  latency_ms?: number;
  data_scope?: Record<string, unknown>;
  steps: TraceStep[];
  handoffs: TraceHandoff[];
  swarm?: { clones: SwarmClone[]; message?: string | null };
  totals: InferenceTotals & { agent_steps: number; decisions: number; errors: number };
}

export interface AgentMetric {
  node: string;
  agent: string;
  role?: string;
  activations: number;
  succeeded: number;
  failed: number;
  retries: number;
  latency_ms: { total: number; p50: number | null; max: number | null };
  llm_calls: number;
  llm_failures: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number | null;
  unpriced_calls: number;
  models: string[];
}

export interface SwarmLimits {
  edition: string;
  label: string;
  max_clones_per_mission: number | null;
  max_parallel_clones: number | null;
  execution: string;
  cluster_fan_out: boolean;
  preview: boolean;
  summary: string;
}

export interface AgentTrace {
  schema_version: string;
  thread_id: string;
  turns: TraceTurn[];
  agents: AgentMetric[];
  totals: InferenceTotals;
  swarm_limits?: SwarmLimits | null;
}

export type PlanNodeType =
  | "question"
  | "agent"
  | "decision"
  | "gate"
  | "data"
  | "operation"
  | "check"
  | "result"
  | "evidence"
  | "train"
  | "evaluate";

export interface PlanNode {
  id: string;
  type: PlanNodeType;
  label: string;
  lane: "agents" | "code" | "data" | string;
  layer: number;
  order: number;
  step_id?: string;
  node?: string;
  status?: string;
  subtitle?: string | null;
  value?: unknown;
  reason?: string | null;
  formula?: string;
  op?: string;
  metrics?: { latency_ms?: number; llm_calls?: number; tokens?: number; cost_usd?: number | null; model?: string };
  values?: Record<string, unknown> | null;
  table?: SmallTable | null;
  signals?: Record<string, unknown>;
  clone?: string | null;
  group?: string;
  /** Column names an operation, load or train node touches. */
  columns?: string[] | null;
  /** Where an operation came from: executed, validated or generated code. */
  source?: "executed" | "validated" | "generated" | string | null;
  evaluation?: EvaluationFacts | null;
}

export interface PlanEdge {
  id: string;
  source: string;
  target: string;
  type: "flow" | "route" | "handoff" | "data" | "check" | "retry" | "evidence" | "gate" | string;
  label?: string;
}

export interface PlanGraph {
  schema_version: string;
  thread_id?: string;
  turn_id?: string | null;
  nodes: PlanNode[];
  edges: PlanEdge[];
}

export interface SwarmCapability {
  enabled: boolean;
  reason?: string;
  limits?: SwarmLimits | null;
  limited?: boolean;
  message?: string | null;
}

/** `features.swarm_console`: the live /swarm page and the swarm/trace routes.
 *  Gated by edition (Professional and Enterprise); the swarm itself is not. */
export interface SwarmConsoleCapability {
  enabled: boolean;
  reason?: "open" | "commercial" | string;
  page?: string;
  message?: string | null;
}

export interface ApiCapabilities {
  features?: { swarm?: SwarmCapability; swarm_console?: SwarmConsoleCapability };
}
