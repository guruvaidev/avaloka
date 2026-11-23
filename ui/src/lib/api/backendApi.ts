import { supabase } from "@/integrations/supabase/external-client";
import type { AgentMetric, AgentTrace, ApiCapabilities, PlanGraph } from "@/lib/agent-trace/types";
import type { UploadFormatStatus } from "@/lib/upload-formats";
import type { DeferredProfileResponse } from "@/lib/deferred-profile";
import type {
  SwarmCloneDoc,
  SwarmEpisodes,
  SwarmMissionDoc,
  SwarmMissionRequest,
  SwarmMissionSummary,
  SwarmRecordsPage,
  SwarmStartResponse,
  SwarmTelemetry,
} from "@/lib/agent-trace/swarm";
import { BACKEND_API_BASE as API_BASE } from "@/lib/api/backend-config";
// The configured MCP bases (runtime env, else localhost); never a hardcoded tunnel.
import { MCP_API_BASE, MCP_TOOL_BASE } from "@/lib/api/mcp-config";

export const SERVICE_RESTARTING_MESSAGE =
  "The Avaloka backend is temporarily unavailable and may be restarting or rolling out. " +
  "Wait for the deployment to become ready, then retry your message.";

export class BackendApiError extends Error {
  readonly status?: number;
  readonly code?: string;
  readonly retryable: boolean;

  constructor(
    message: string,
    options: { status?: number; code?: string; retryable?: boolean } = {},
  ) {
    super(message);
    this.name = "BackendApiError";
    this.status = options.status;
    this.code = options.code;
    this.retryable = options.retryable ?? false;
  }
}

const isTransientGatewayFailure = (status: number) =>
  status === 502 || status === 503 || status === 504;

const describesUnavailableUpstream = (message: string) =>
  /(bad gateway|service unavailable|upstream|langgraph (?:network error|timeout)|connection refused)/i.test(message);


// Single dataset upload response (legacy format)
interface UploadResponse {
  dataset_id: string;
  session_id: string;
  thread_id: string;
  schema: string[];
  samples: any[];
  ddl_schema: string;
  rows_sampled: number;
  visualization_config?: any;
  visualization_status?: string;
  // Sampling / profiling fields for large datasets
  portfolio_samples?: Record<string, any[]>;
  available_samples?: string[];
  profiling_result?: Record<string, any>;
  // "pending" while the data profile runs after the upload returned; poll
  // getDatasetProfile until it is full_profile, failed or skipped.
  profiling_status?: string | null;
  profiling_error?: string | null;
  sample_statistics?: Record<string, any>;
  sample_status?: string;
  background_task_id?: string;
}

// Multi-dataset upload response format
/** One entry of GET /threads/{id}/datasets. */
import type { JobNoteBatch, ThreadJobNotesResponse } from "@/lib/job-notes";

export interface ThreadDatasetItem {
  dataset_id: string;
  filename?: string | null;
  alias?: string | null;
  sheet_name?: string | null;
  source_workbook?: string | null;
  rows?: number | null;
  selected?: boolean;
  available?: boolean;
}

export interface DatasetItem {
  dataset_id: string;
  filename: string;
  alias: string;
  columns: string[];
  rows: any[];
  visualization_config?: any;
  visualization_status?: string;
  portfolio_samples?: Record<string, any[]>;
  available_samples?: string[];
  profiling_result?: Record<string, any>;
  profiling_status?: string | null;
  profiling_error?: string | null;
  sample_statistics?: Record<string, any>;
}

export interface MultiUploadResponse {
  session_id: string;
  thread_id: string;
  datasets: DatasetItem[];
}

interface ThreadResponse {
  thread_id: string;
  created_at: string;
  metadata: any;
  assistant_id: string;
}

interface MessageResponse {
  messages: Array<{
    role: string;
    content: string;
  }>;
  planner_definition: any;
  // Planner graph signals: the button shows only when status === "success"
  // and a display URL came back for this turn.
  planner_graph_status?: string;
  planner_graph_display_url?: string | null;
  ready_to_summarize: boolean;
  ready_to_code: boolean;
  coder_definition: any;
  output_file_data: any;
  output_json: any;
  output_tables?: Array<{
    title: string;
    rows: Record<string, unknown>[];
    action_ids?: string[];
    group_by?: string[];
    metric_columns?: string[];
  }>;
  multi_action_status?: "complete" | "incomplete";
  multi_action_actions?: Array<{ id: string; operation: string; column?: string | null; group_by: string[] }>;
  multi_action_missing?: Array<{ label: string }>;
  multi_action_notes?: string[];
  task_info?: { task_id: string; next_due_at?: number; max_runs?: number; [k: string]: unknown };
  // Background-job notes delivered at this turn's end (one coalesced batch per
  // flush, written by code on the API). Rendered after the reply; never
  // persisted again by the UI, the API wrote the transcript row.
  job_notes?: JobNoteBatch[] | null;
  analysis_fidelity?: "quick_sample" | "portfolio_samples" | "entire_dataset";
  selected_sample_name?: string;
  analysis_task_id?: string;
  // Slow-turn signals: "running" means poll /pending-turn; kind says what it is.
  pending_status?: "running" | string | null;
  pending_kind?: "analysis" | "training" | string | null;
  // Long-running (training) turn signals
  training_status?: string;
  training_completed?: boolean;
  training_metrics?: any;
  training_result?: any;
  mlflow_run_id?: string;
  // Set by the proxy when the upstream call exceeded its timeout
  error?: string;
  fallback?: boolean;
  execution_context?: { mode: string; mode_label: string; sample?: string | null };
  // The agent trace recorded for this turn (GET trace_url).
  trace_turn_id?: string | null;
  trace_url?: string | null;
  /** Set when a fallback model tier answered instead of the configured model. */
  answered_by?: {
    configured_model: string;
    answered_model: string;
    provider: string;
    reason: string;
    note?: string;
  } | null;
  datasets?: Array<{
    dataset_id: string;
    filename?: string;
    alias?: string;
  }>;
  active_dataset_ids?: string[];
  visualization_config?: any;
  visualization_status?: string;
  visualization_configs?: Record<string, any>;
  visualization_statuses?: Record<string, string>;
}

interface TaskResultResponse {
  status: "SUCCESS" | "FAILED" | "CANCELLED" | "PENDING" | "STARTED" | "RETRY" | string;
  result?: {
    output_file_data: any;
    output_json: any;
    execution_context?: { mode: string; mode_label: string; sample?: string | null };
  };
  traceback?: string;
  next_due_at?: number;
}

interface HealthResponse {
  status: string;
  graph_ready: boolean;
  langgraph_url: string;
  assistant_id: string;
  upstream_reachable: boolean;
  upstream_timeout_s: number;
}

interface DatabaseSessionResponse {
  session_id: string;
  thread_id: string;
  connection_id: string;
  expires_in: number;
}

interface BucketListResponse {
  backend: string;
  bucket: string;
  prefix: string;
  folders?: Array<{ name: string; prefix: string }>;
  objects: Array<{ key: string; size?: string; updated?: string }>;
}

interface DatabaseQueryResponse {
  dataset_id: string;
  thread_id: string;
  session_id: string;
  results: any[];
  schema: Array<{ name: string; type: string }>;
  row_count: number;
  execution_time_ms: number;
}

export interface PendingTurnResponse {
  status: "running" | "done" | "error" | "superseded" | "none" | string;
  message?: string;
  result?: any;
  /** On "superseded": the id of the turn that replaced this one. */
  superseded_by?: string | null;
}

/**
 * A reply is a scheduled task when task_info.task_id is set and the turn is
 * not still running. Scheduled tasks deliver results outside the chat turn
 * (notifications / Scheduled Analysis), so they never show loaders or poll.
 */
export function isScheduledTaskReply(res: any): boolean {
  if (!res || typeof res !== "object") return false;
  const taskId = res.task_info?.task_id;
  return Boolean(taskId) && String(res.pending_status ?? "").toLowerCase() !== "running";
}

/**
 * A chat turn is "deferred" only when the backend says pending_status ===
 * "running" (slow turns, model training), or when our proxy gave up waiting
 * on a synchronous response. Both resolve later via /pending-turn.
 */
export function isDeferredTurn(res: any): boolean {
  if (!res || typeof res !== "object") return false;
  if (res.fallback === true && res.error === "UPSTREAM_TIMEOUT") return true;
  if (isScheduledTaskReply(res)) return false;
  return String(res.pending_status ?? "").toLowerCase() === "running";
}

const _previewInflight: Map<string, Promise<any>> = new Map();
const _previewControllers: Map<string, AbortController> = new Map();

// --- inference-over-chat helpers -------------------------------------------
const _inferenceThreads: Map<string, string> = new Map();
let _inferenceSessionId: string | null = null;

function getInferenceSessionId(): string {
  if (_inferenceSessionId) return _inferenceSessionId;
  if (typeof window !== "undefined") {
    try {
      const existing =
        window.localStorage.getItem("avaloka_session_id") ||
        window.localStorage.getItem("avaloka.session_id");
      if (existing) {
        _inferenceSessionId = existing;
        return existing;
      }
    } catch {
      /* ignore */
    }
  }
  _inferenceSessionId =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `inf-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return _inferenceSessionId;
}

/** Normalize message content that may arrive as a string or as content blocks. */
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content))
    return content
      .map((c) => (typeof c === "string" ? c : typeof (c as any)?.text === "string" ? (c as any).text : ""))
      .filter(Boolean)
      .join("\n");
  if (content && typeof content === "object") {
    const t = (content as any).text ?? (content as any).content;
    if (typeof t === "string") return t;
  }
  return "";
}

const PREDICTION_KEY = /^(prediction|predicted|predicted_value|prediction_value|y_pred|output|result|value)$/i;

/** Walk a structured payload (output_json etc.) looking for a prediction-ish value. */
function parsePredictionFromJson(payload: unknown, depth = 0): unknown {
  if (payload === null || payload === undefined || depth > 5) return null;
  if (Array.isArray(payload)) {
    for (const item of payload) {
      const found = parsePredictionFromJson(item, depth + 1);
      if (found !== null && found !== undefined) return found;
    }
    return null;
  }
  if (typeof payload === "object") {
    const rec = payload as Record<string, unknown>;
    for (const [k, v] of Object.entries(rec)) {
      if (PREDICTION_KEY.test(k) && (typeof v === "number" || typeof v === "string")) {
        const n = Number(String(v).replace(/[$,]/g, ""));
        return Number.isFinite(n) ? n : v;
      }
    }
    for (const v of Object.values(rec)) {
      const found = parsePredictionFromJson(v, depth + 1);
      if (found !== null && found !== undefined) return found;
    }
  }
  return null;
}

/** Pull the numeric prediction out of an assistant reply, e.g. "Prediction: median_house_value = 452600". */
function parsePredictionFromReply(reply: string): unknown {
  if (!reply) return null;
  const patterns = [
    /prediction[^\n=:]*[=:]\s*\$?(-?[\d,]+(?:\.\d+)?)/i,
    /predicted[^\n=:]*[=:]\s*\$?(-?[\d,]+(?:\.\d+)?)/i,
    /\bresult\s*[=:]\s*\$?(-?[\d,]+(?:\.\d+)?)/i,
    // "The predicted median_house_value is $411,071.00" / "... is approximately 411071"
    /predicted[^\n.]*?\bis\b\s*(?:approximately\s*|about\s*|around\s*)?\$?(-?[\d,]+(?:\.\d+)?)/i,
    /prediction[^\n.]*?\bis\b\s*(?:approximately\s*|about\s*|around\s*)?\$?(-?[\d,]+(?:\.\d+)?)/i,
    /\bestimated[^\n.]*?\bis\b\s*(?:approximately\s*|about\s*|around\s*)?\$?(-?[\d,]+(?:\.\d+)?)/i,
  ];
  for (const re of patterns) {
    const m = reply.match(re);
    if (m) {
      const n = Number(m[1].replace(/,/g, ""));
      if (Number.isFinite(n)) return n;
    }
  }
  // Last resort: a lone currency/number token on a line that mentions value/price/prediction
  const line = reply
    .split(/\n+/)
    .find((l) => /(predict|value|price|estimate)/i.test(l) && /-?[\d,]+(\.\d+)?/.test(l));
  if (line) {
    const m = line.match(/\$?(-?[\d,]+(?:\.\d+)?)/);
    if (m) {
      const n = Number(m[1].replace(/,/g, ""));
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}



/**
 * A token pasted into the browser for a laptop demo against an API started
 * with a chosen SUPABASE_JWT_SECRET (docs/design/SWARM_DEMO.md). Read only
 * when set; the server still verifies it. Nothing else reads this key.
 */
export const DEV_TOKEN_KEY = "avaloka-dev-token";

function devToken(): string | null {
  try {
    return typeof window !== "undefined" ? window.localStorage.getItem(DEV_TOKEN_KEY) : null;
  } catch {
    return null;
  }
}

// Exported for the inference panel (src/api/inference.ts), as on 1.6.
export async function getAccessToken(): Promise<string> {
  const pasted = devToken();
  if (pasted) return pasted;
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session?.access_token) {
    if (typeof window !== "undefined") {
      window.location.href = "/";
    }
    throw new Error("Not authenticated - please sign in");
  }
  return session.access_token;
}

function proxyUrl(): string {
  return typeof window !== "undefined"
    ? `${window.location.origin}/api/public/backend-proxy`
    : `/api/public/backend-proxy`;
}

/**
 * Same-origin proxy fetch — routes ALL backend calls through
 * /api/public/backend-proxy to avoid CORS issues with the upstream host.
 * `path` is the backend path (e.g. "/api/upload" or "/threads/x/messages").
 */
export async function proxyFetch(path: string, init: RequestInit & { sessionId?: string } = {}): Promise<Response> {
  const { sessionId, headers: initHeaders, ...rest } = init;
  const headers = new Headers(initHeaders as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", API_BASE);
  if (sessionId) headers.set("X-Avaloka-Session", sessionId);
  return fetch(proxyUrl(), { ...rest, headers });
}

// MCP calls reuse the same backend-proxy, but forward to the MCP base.
// The proxy runs server-side, so the in-cluster MCP URL resolves there.
export async function mcpProxyFetch(base: string, path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", base);
  return fetch(proxyUrl(), { ...init, headers });
}

export type AppNotificationKind =
  | "analysis_done"
  | "analysis_failed"
  | "training_done"
  | "training_failed"
  | "batch_job_done"
  | "batch_job_failed"
  | "insights_ready"
  | "insights_failed"
  | string;

export interface AppNotification {
  id: string;
  kind: AppNotificationKind;
  title: string;
  body: string | null;
  link: { thread_id?: string; dataset_id?: string; deferred_id?: string; job_id?: string } | null;
  created_at: string;
  read_at: string | null;
}

export interface AppNotificationsResponse {
  items: AppNotification[];
  unread_count: number;
}

export interface UploadLimits {
  plan?: "free" | "professional" | "enterprise" | string;
  max_file_bytes?: number | null;
  max_total_bytes?: number | null;
  max_files?: number | null;
  max_file_mb?: number | null;
  max_total_mb?: number | null;
  batch_jobs?: boolean;
  /** True when the plan could not be checked (hosted, lookup failed): not an upgrade case. */
  plan_check_unavailable?: boolean;
  /** Per gated feature: on the plan, not on the plan, or not checkable right now. */
  feature_states?: Record<string, "allowed" | "not_on_plan" | "unavailable">;
  [k: string]: unknown;
}

export type UploadPhase = "uploading" | "processing";
export interface UploadProgressInfo {
  phase: UploadPhase;
  loaded: number;
  total: number;
}

/** Fail an upload only after this long with no bytes sent. */
const UPLOAD_STALL_MS = 60_000;
/** Server processing wait after the last byte: 60 s + 1 s per MB, capped at 180 s. */
function processingTimeoutMs(totalBytes: number) {
  const mb = totalBytes / (1024 * 1024);
  return Math.min(180_000, 60_000 + Math.ceil(mb) * 1000);
}

async function getTraceJson<T>(path: string, what: string): Promise<T> {
  const accessToken = await getAccessToken();
  const response = await proxyFetch(path, {
    method: "GET",
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!response.ok) {
    const err = new Error(`Failed to fetch ${what}: ${response.statusText}`) as Error & { status?: number; code?: string };
    err.status = response.status;
    if (response.status === 403) {
      // The swarm console gate (app/api/swarm_gate.py) names itself in detail.code.
      try {
        const json = await response.json();
        if (json?.detail?.code) err.code = String(json.detail.code);
      } catch {
        /* a bare 403 */
      }
    }
    throw err;
  }
  return response.json() as Promise<T>;
}

async function postTraceJson<T>(path: string, body: unknown, what: string): Promise<T> {
  const accessToken = await getAccessToken();
  const response = await proxyFetch(path, {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    // A 400 carries the reason (an unknown sample, a missing target column); say it.
    let detail = response.statusText;
    try {
      const json = await response.json();
      if (typeof json?.detail === "string") detail = json.detail;
    } catch {
      /* not JSON */
    }
    const err = new Error(`Failed to ${what}: ${detail}`) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }
  return response.json() as Promise<T>;
}

function turnQuery(turnId?: string | null, sep = "?"): string {
  return turnId ? `${sep}turn=${encodeURIComponent(turnId)}` : "";
}

/** GET /api/llm-models: the language model per agent role and its fallback chain. */
export type LlmModelsResponse = {
  roles: Array<{
    role: string;
    agent: string;
    provider: string;
    model: string;
    chain: Array<{ provider: string; model: string; armed: boolean; providers_only: string[] | null }>;
    routing: Record<string, unknown> | null;
  }>;
  note?: string;
};

export const backendApi = {
  /** Every agent step of one analysis (app/trace). */
  async getAgentTrace(threadId: string, turnId?: string | null): Promise<AgentTrace> {
    return getTraceJson<AgentTrace>(
      `/threads/${encodeURIComponent(threadId)}/trace${turnQuery(turnId)}`,
      "agent trace",
    );
  },

  /** The granular, typed planner graph of one recorded turn. */
  async getPlanGraph(threadId: string, turnId?: string | null): Promise<PlanGraph> {
    return getTraceJson<PlanGraph>(
      `/threads/${encodeURIComponent(threadId)}/planner-graph?format=json${turnQuery(turnId, "&")}`,
      "planner graph",
    );
  },

  /** Per-agent calls, latency, tokens, cost and outcomes across the caller's analyses. */
  async getAgentMetrics(): Promise<{ threads: number; turns: number; agents: AgentMetric[] }> {
    return getTraceJson(`/api/agent-metrics`, "agent metrics");
  },

  /** Feature discovery; ``features.swarm`` carries the edition's swarm limits,
   *  ``features.swarm_console`` whether the live /swarm page is open to this caller,
   *  ``features.upload_format_status`` what /api/upload takes and defers. */
  async getCapabilities(): Promise<ApiCapabilities & {
    features?: {
      upload_formats?: string[];
      upload_format_status?: Record<string, UploadFormatStatus>;
    };
  }> {
    return getTraceJson(`/api/capabilities`, "capabilities");
  },

  // ---- Swarm missions (app/trace/swarm_service.py; owner-only, Bearer auth) ----

  /** Start a coordinated mission; returns at once, then poll getSwarmRecords. */
  async startSwarmMission(body: SwarmMissionRequest): Promise<SwarmStartResponse> {
    return postTraceJson<SwarmStartResponse>(`/api/swarm/missions`, body, "start the swarm mission");
  },

  async listSwarmMissions(): Promise<{ missions: SwarmMissionSummary[] }> {
    return getTraceJson(`/api/swarm/missions`, "swarm missions");
  },

  async getSwarmMission(missionId: string, opts: { records?: boolean } = {}): Promise<SwarmMissionDoc> {
    return getTraceJson<SwarmMissionDoc>(
      `/api/swarm/missions/${encodeURIComponent(missionId)}${opts.records ? "?records=1" : ""}`,
      "swarm mission",
    );
  },

  /** Records with seq > after (seq starts at 1); `done` once the mission finished and the tail is drained. */
  async getSwarmRecords(missionId: string, after = 0, limit = 500): Promise<SwarmRecordsPage> {
    return getTraceJson<SwarmRecordsPage>(
      `/api/swarm/missions/${encodeURIComponent(missionId)}/records?after=${after}&limit=${limit}`,
      "swarm records",
    );
  },

  async getSwarmClone(missionId: string, cloneId: string): Promise<SwarmCloneDoc> {
    return getTraceJson<SwarmCloneDoc>(
      `/api/swarm/missions/${encodeURIComponent(missionId)}/clones/${encodeURIComponent(cloneId)}`,
      "swarm clone",
    );
  },

  /** The RL view: one episode per clone, reward left null. */
  async getSwarmEpisodes(missionId: string): Promise<SwarmEpisodes> {
    return getTraceJson<SwarmEpisodes>(`/api/swarm/missions/${encodeURIComponent(missionId)}/episodes`, "swarm episodes");
  },

  /** Exactly the records telemetry would send for this mission; nothing is sent. */
  async getSwarmTelemetry(missionId: string): Promise<SwarmTelemetry> {
    return getTraceJson<SwarmTelemetry>(`/api/swarm/missions/${encodeURIComponent(missionId)}/telemetry`, "swarm telemetry");
  },

  async getPlannerGraph(threadId: string): Promise<Blob> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/planner-graph`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) {
      const err = new Error(`Failed to fetch planner graph: ${response.statusText}`) as Error & { status?: number };
      err.status = response.status;
      throw err;
    }
    return response.blob();
  },

  async checkHealth(): Promise<HealthResponse> {
    const response = await proxyFetch(`/health`, { method: "GET" });
    if (!response.ok) throw new Error(`Health check failed: ${response.statusText}`);
    return response.json();
  },

  /** Plan upload limits for the signed-in user. Unwraps `{ plan, limits: {...} }` (flat responses also accepted). */
  async getUploadLimits(): Promise<UploadLimits> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/limits`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Limits check failed: ${response.statusText}`);
    const json: any = await response.json();
    if (json?.fallback === true || (json?.error && !json?.limits && json?.max_file_bytes == null)) {
      throw new Error("Limits unavailable");
    }
    const nested = json?.limits && typeof json.limits === "object" ? json.limits : json;
    return {
      ...nested,
      plan: json?.plan ?? nested?.plan,
      plan_check_unavailable: json?.plan_check_unavailable === true,
      feature_states: json?.feature_states ?? nested?.feature_states,
    };
  },

  async getVersion(): Promise<{ version: string }> {
    const response = await proxyFetch(`/version`, { method: "GET" });
    if (!response.ok) throw new Error(`Version check failed: ${response.statusText}`);
    return response.json();
  },

  /**
   * Health of the database query service: GET /health on MCP_TOOL_BASE, the
   * MCP server that List tables and queries call, through the same proxy.
   * (It used to invoke a "check-mcp-health" edge function that no install
   * ships -- ui/supabase/functions has none -- so it read unhealthy even
   * while registration and queries worked.) An unreachable upstream comes
   * back from the proxy as 200 + { fallback: true }, so only the server's own
   * { status: "healthy" } counts as healthy.
   */
  async checkMcpHealth(): Promise<{ status: string; healthy: boolean }> {
    try {
      const response = await mcpProxyFetch(MCP_TOOL_BASE, "/health", { method: "GET" });
      const data: any = await response.json().catch(() => null);
      if (data?.fallback === true) return { status: "unreachable", healthy: false };
      if (!response.ok) return { status: `unhealthy (HTTP ${response.status})`, healthy: false };
      if (data?.status === "healthy") return { status: "healthy", healthy: true };
      return { status: "unhealthy", healthy: false };
    } catch {
      return { status: "unreachable", healthy: false };
    }
  },

  async createThread(sessionId: string): Promise<ThreadResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads`, {
      method: "POST",
      sessionId,
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error(`Failed to create thread: ${response.statusText}`);
    return response.json();
  },

  async uploadFile(
    file: File,
    sessionId?: string,
    backend?: string,
    connectionId?: string,
    attachToThreadId?: string,
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (sessionId) formData.append("session_id", sessionId);
    if (backend) formData.append("backend", backend);
    if (connectionId) formData.append("connection_id", connectionId);
    if (attachToThreadId) formData.append("attach_to_thread_id", attachToThreadId);

    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/upload`, {
      method: "POST",
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      const err: any = new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      err.status = response.status;
      throw err;
    }

    return response.json();
  },

  /** Attach already-uploaded datasets to an existing chat thread. */
  async attachDatasetsToThread(threadId: string, datasetIds: string[]): Promise<void> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/datasets`, {
      method: "POST",
      headers: {
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ dataset_ids: datasetIds }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      const err: any = new Error(
        errorData.detail ||
          (response.status === 404 ? "Chat or dataset not found" : `Attach failed: ${response.statusText}`),
      );
      err.status = response.status;
      throw err;
    }
  },

  /** Attach every dataset of the given analyses to a chat thread. */
  async attachAnalysesToThread(
    threadId: string,
    analysisIds: string[],
  ): Promise<{ thread_id?: string; dataset_ids?: string[]; datasets?: unknown[] }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/datasets`, {
      method: "POST",
      headers: {
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ analysis_ids: analysisIds }),
    });
    const data = await response.json().catch(() => ({}) as any);
    if (!response.ok) {
      const detail = (data as any)?.detail;
      const err: any = new Error(
        typeof detail === "string" ? detail : typeof detail?.message === "string" ? detail.message : `Request failed (${response.status})`,
      );
      err.status = response.status;
      err.detail = detail;
      throw err;
    }
    return data;
  },

  /** List datasets attached to a chat thread (for the chips under the chat). */
  async listThreadDatasets(threadId: string): Promise<ThreadDatasetItem[]> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/datasets`, {
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
    });
    if (!response.ok) {
      const err: any = new Error(`Failed to load thread datasets: ${response.statusText}`);
      err.status = response.status;
      throw err;
    }
    const data = await response.json().catch(() => [] as any);
    return Array.isArray(data) ? data : (data?.datasets ?? []);
  },

  /** Save which datasets the chat uses (PUT /threads/{id}/selected-datasets, at least one id). */
  async setSelectedDatasets(threadId: string, datasetIds: string[]): Promise<ThreadDatasetItem[]> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/selected-datasets`, {
      method: "PUT",
      headers: {
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ dataset_ids: datasetIds }),
    });
    const data = await response.json().catch(() => ({}) as any);
    if (!response.ok) {
      const err: any = new Error(typeof data?.detail === "string" ? data.detail : `Saving selection failed (${response.status})`);
      err.status = response.status;
      throw err;
    }
    return Array.isArray(data) ? data : (data?.datasets ?? []);
  },

  async uploadFiles(
    files: File[],
    backend?: string,
    connectionId?: string,
    onProgress?: (p: UploadProgressInfo) => void,
    attachToThreadId?: string,
  ): Promise<MultiUploadResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    if (backend) formData.append("backend", backend);
    if (connectionId) formData.append("connection_id", connectionId);
    if (attachToThreadId) formData.append("attach_to_thread_id", attachToThreadId);

    const accessToken = await getAccessToken();
    const totalBytes = files.reduce((s, f) => s + f.size, 0);

    // XHR gives real byte-level progress. No fixed total timeout: we only fail
    // after 60 s without progress, then wait a size-based window for processing.
    return new Promise<MultiUploadResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      let processTimer: ReturnType<typeof setTimeout> | null = null;
      const clearTimers = () => {
        if (stallTimer) clearTimeout(stallTimer);
        if (processTimer) clearTimeout(processTimer);
        stallTimer = processTimer = null;
      };
      const fail = (message: string, extra: Record<string, unknown> = {}) => {
        clearTimers();
        const err: any = new Error(message);
        Object.assign(err, extra);
        reject(err);
      };
      const armStall = () => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(() => {
          xhr.abort();
          fail("Upload stalled — check your connection and retry", { code: "upload_stalled" });
        }, UPLOAD_STALL_MS);
      };
      const startProcessing = () => {
        if (processTimer) return;
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = null;
        onProgress?.({ phase: "processing", loaded: totalBytes, total: totalBytes });
        processTimer = setTimeout(() => {
          xhr.abort();
          fail("Processing is taking longer than expected. Your data may still be processing — please retry.", {
            code: "processing_timeout",
          });
        }, processingTimeoutMs(totalBytes));
      };

      xhr.open("POST", proxyUrl());
      xhr.setRequestHeader("X-Backend-Path", "/api/upload");
      xhr.setRequestHeader("X-Backend-Base", API_BASE);
      if (accessToken) xhr.setRequestHeader("Authorization", `Bearer ${accessToken}`);

      xhr.upload.onprogress = (e) => {
        const total = e.lengthComputable ? e.total : totalBytes;
        onProgress?.({ phase: "uploading", loaded: e.loaded, total });
        if (e.loaded >= total) startProcessing();
        else armStall();
      };
      xhr.upload.onload = () => startProcessing();
      xhr.onerror = () => fail("The server didn't respond. Your request may still be running — please retry in a moment.", { network: true });
      xhr.onload = () => {
        clearTimers();
        let json: any = null;
        try {
          json = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          json = null;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(json as MultiUploadResponse);
          return;
        }
        const detail = json?.detail;
        const message =
          typeof detail === "string"
            ? detail
            : typeof detail?.message === "string"
              ? detail.message
              : xhr.status >= 500
                ? "The server didn't respond. Your request may still be running — please retry in a moment."
                : `Batch upload failed (${xhr.status})`;
        const err: any = new Error(message);
        err.status = xhr.status;
        err.detail = detail;
        reject(err);
      };

      armStall();
      xhr.send(formData);
    });
  },

  async sendMessage(
    threadId: string,
    sessionId: string,
    content: string,
    datasetIds?: string[],
    _analysisFidelity?: "quick_sample" | "portfolio_samples" | "entire_dataset",
    _selectedSampleName?: string,
    metadata?: { dataset_id?: string | null; analysis_id?: string | null },
  ): Promise<MessageResponse> {
    try {
      this.abortAllPreviews();
    } catch {
      /* ignore */
    }

    const payload: any = {
      role: "user",
      content,
      stream: false,
      ...(datasetIds && datasetIds.length > 0 && { dataset_ids: datasetIds }),
      // The analysis mode and sample are not sent: the server ignores them in a
      // chat message, and the "Analysis on" switch writes them to the session
      // through /api/datasets/{id}/fidelity. The parameters stay for legacy callers.
      ...(metadata && { metadata: {
        ...(metadata.dataset_id && { dataset_id: metadata.dataset_id }),
        ...(metadata.analysis_id && { analysis_id: metadata.analysis_id }),
      } }),
    };

    const accessToken = await getAccessToken();
    let response: Response;
    try {
      response = await proxyFetch(`/threads/${threadId}/messages`, {
        method: "POST",
        sessionId,
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    } catch (cause) {
      console.error("Backend API network error while sending message:", cause);
      throw new BackendApiError(SERVICE_RESTARTING_MESSAGE, {
        code: "SERVICE_RESTARTING",
        retryable: true,
      });
    }

    if (!response.ok) {
      let errorMessage = `Message failed: ${response.statusText}`;
      let errorCode: string | undefined;
      let retryable = false;
      let hasStructuredError = false;
      try {
        const errorData = await response.json();
        if (typeof errorData.detail === "string") {
          errorMessage = errorData.detail;
          hasStructuredError = true;
        } else if (errorData.detail && typeof errorData.detail === "object") {
          errorMessage = errorData.detail.message || errorData.detail.summary || errorMessage;
          errorCode = errorData.detail.code;
          retryable = Boolean(errorData.detail.retryable);
          hasStructuredError = true;
        } else if (errorData.error) {
          errorMessage = errorData.error;
          hasStructuredError = true;
        }
      } catch {
        /* ignore */
      }

      if (
        isTransientGatewayFailure(response.status) &&
        (!hasStructuredError || describesUnavailableUpstream(errorMessage))
      ) {
        errorMessage = SERVICE_RESTARTING_MESSAGE;
        errorCode = "SERVICE_RESTARTING";
        retryable = true;
      }

      throw new BackendApiError(errorMessage, {
        status: response.status,
        code: errorCode,
        retryable,
      });
    }

    return response.json();
  },

  async listNotifications(opts: { limit?: number; unread?: boolean } = {}): Promise<AppNotificationsResponse> {
    const accessToken = await getAccessToken();
    const params = new URLSearchParams({ limit: String(opts.limit ?? 20) });
    if (opts.unread) params.set("unread", "true");
    const response = await proxyFetch(`/api/notifications?${params.toString()}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Notifications fetch failed: ${response.status}`);
    return response.json();
  },

  async markNotificationRead(id: string): Promise<{ ok: boolean }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/notifications/${encodeURIComponent(id)}/read`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Mark read failed: ${response.status}`);
    return response.json();
  },

  async markAllNotificationsRead(): Promise<{ updated: number }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/notifications/read-all`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Mark all read failed: ${response.status}`);
    return response.json();
  },

  /**
   * Poll the backend for the result of a long-running (deferred) chat turn.
   * Returns the raw envelope: { status: "running" | "done" | "error" | "superseded" | "none", ... }
   * Only for turns where isDeferredTurn() is true — never for scheduled tasks.
   */
  /**
   * Deliver and return the thread's finished-job notes. When a turn is in
   * flight the API keeps them queued (`deferred: true`); they then arrive with
   * that turn's reply as `job_notes`.
   */
  async getThreadJobNotes(threadId: string, sessionId: string): Promise<ThreadJobNotesResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/job-notes`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Job notes fetch failed: ${response.statusText}`);
    return response.json();
  },

  async getPendingTurn(
    threadId: string,
    sessionId: string,
    deferredId?: string | null,
  ): Promise<PendingTurnResponse> {
    const accessToken = await getAccessToken();
    const qs = deferredId ? `?deferred_id=${encodeURIComponent(deferredId)}` : "";
    const response = await proxyFetch(`/threads/${threadId}/pending-turn${qs}`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Pending turn fetch failed: ${response.statusText}`);
    return response.json();
  },

  /**
   * Poll /pending-turn every `intervalMs` until the turn resolves or the
   * overall budget runs out. Never throws on transient network errors.
   */
  async waitForPendingTurn(
    threadId: string,
    sessionId: string,
    deferredId?: string | null,
    opts: { intervalMs?: number; timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<PendingTurnResponse> {
    const intervalMs = opts.intervalMs ?? 5_000;
    const timeoutMs = opts.timeoutMs ?? 12 * 60_000;
    const deadline = Date.now() + timeoutMs;
    let consecutiveErrors = 0;

    while (Date.now() < deadline) {
      if (opts.signal?.aborted) return { status: "none" };
      await new Promise((r) => setTimeout(r, intervalMs));
      if (opts.signal?.aborted) return { status: "none" };
      try {
        const data = await this.getPendingTurn(threadId, sessionId, deferredId);
        consecutiveErrors = 0;
        const status = String(data?.status ?? "").toLowerCase();
        if (status === "running" || status === "pending" || status === "") continue;
        return { ...data, status: status as PendingTurnResponse["status"] };
      } catch (err: any) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 5) {
          return { status: "error", message: err?.message || "Lost connection while waiting for the result." };
        }
      }
    }
    return {
      status: "error",
      message: "Training is taking longer than expected. It may still finish in the background — try again shortly.",
    };
  },

  async fetchThreadHistory(threadId: string): Promise<{ messages: Array<{ role: string; content: string }> }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}/messages`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
    });
    if (!response.ok) throw new Error(`Failed to fetch history: ${response.statusText}`);
    return response.json();
  },

  async deleteThread(threadId: string): Promise<void> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/threads/${threadId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to delete thread: ${response.statusText}`);
  },

  async getTaskResult(taskId: string, sessionId: string, index: number = 0): Promise<TaskResultResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks/${taskId}/result/${index}`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Task result fetch failed: ${response.statusText}`);
    return response.json();
  },

  async listTasks(sessionId: string): Promise<any[]> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to list tasks: ${response.statusText}`);
    const data = await response.json();
    return Array.isArray(data) ? data : (data?.tasks ?? []);
  },

  async getTaskInfo(taskId: string, sessionId: string): Promise<any> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks/${taskId}/info`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Task info fetch failed: ${response.statusText}`);
    return response.json();
  },

  async getTaskStatus(taskId: string, sessionId: string): Promise<any> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks/${taskId}/status`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Task status fetch failed: ${response.statusText}`);
    return response.json();
  },

  async getTaskRuns(taskId: string, sessionId: string): Promise<any[]> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks/${taskId}/runs`, {
      method: "GET",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Task runs fetch failed: ${response.status} ${response.statusText}`);
    const data = await response.json();
    if (Array.isArray(data)) return data;
    return data?.runs ?? data?.items ?? [];
  },

  async deleteTask(taskId: string, sessionId: string): Promise<void> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/tasks/${taskId}`, {
      method: "DELETE",
      sessionId,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to cancel task: ${response.statusText}`);
  },

  abortAllPreviews() {
    _previewControllers.forEach((ctrl) => {
      try {
        ctrl.abort();
      } catch {
        /* ignore */
      }
    });
    _previewControllers.clear();
    _previewInflight.clear();
  },

  abortPreview(datasetId: string) {
    const prefix = `${datasetId}:`;
    _previewControllers.forEach((ctrl, key) => {
      if (key !== datasetId && !key.startsWith(prefix)) return;
      ctrl.abort();
      _previewControllers.delete(key);
      _previewInflight.delete(key);
    });
  },

  async getDatasetPreview(datasetId: string, sessionId: string, forceRefresh = false, threadId?: string | null): Promise<any> {
    const previewKey = `${datasetId}:${sessionId}:${threadId ?? ""}`;
    if (!forceRefresh && _previewInflight.has(previewKey)) return _previewInflight.get(previewKey)!;
    if (forceRefresh) this.abortPreview(datasetId);

    const promise = (async () => {
      const controller = new AbortController();
      _previewControllers.set(previewKey, controller);
      const timeoutId = setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), 60_000);

      try {
        const accessToken = await getAccessToken();
        const query = new URLSearchParams({ session_id: sessionId });
        if (threadId) query.set("thread_id", threadId);
        if (forceRefresh) query.set("_", String(Date.now()));
        const path = `/datasets/${encodeURIComponent(datasetId)}/preview?${query.toString()}`;
        const response = await proxyFetch(path, {
          method: "GET",
          sessionId,
          headers: { Authorization: `Bearer ${accessToken}` },
          signal: controller.signal,
          ...(forceRefresh ? { cache: "no-store" as RequestCache } : {}),
        });
        if (!response.ok) throw new Error(`Failed to fetch dataset preview: ${response.statusText}`);
        return response.json();
      } catch (err: any) {
        if (err?.name === "AbortError" || err?.name === "TimeoutError") return null;
        throw err;
      } finally {
        clearTimeout(timeoutId);
        _previewControllers.delete(previewKey);
      }
    })();

    _previewInflight.set(previewKey, promise);
    promise.finally(() => _previewInflight.delete(previewKey));
    return promise;
  },

  /** The upload's background data profile (GET /api/datasets/{id}/profile). */
  async getDatasetProfile(datasetId: string): Promise<DeferredProfileResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/datasets/${encodeURIComponent(datasetId)}/profile`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Profile fetch failed: ${response.statusText}`);
    return response.json();
  },

  /** Uploaded datasets for the signed-in user (GET /datasets). */
  async listDatasets(): Promise<any> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/datasets`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to list datasets: ${response.statusText}`);
    }
    return response.json();
  },

  /** Preview for an uploaded dataset (GET /datasets/{id}/preview), no session required. */
  async previewUploadedDataset(datasetId: string): Promise<any> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/datasets/${encodeURIComponent(datasetId)}/preview`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to load dataset preview: ${response.statusText}`);
    }
    return response.json();
  },

  /** Delete an uploaded dataset (DELETE /datasets/{id}). */
  async deleteDataset(datasetId: string): Promise<void> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/datasets/${encodeURIComponent(datasetId)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok && response.status !== 204) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to delete dataset: ${response.statusText}`);
    }
  },

  async listBucketObjects(params: {
    backend: "gcs" | "gs" | "s3" | "azure" | "az";
    bucket: string;
    prefix?: string;
    connectionId?: string;
  }): Promise<BucketListResponse> {
    const accessToken = await getAccessToken();
    const qs = new URLSearchParams({
      backend: params.backend,
      bucket: params.bucket,
      prefix: params.prefix ?? "",
    });
    if (params.connectionId) qs.set("connection_id", params.connectionId);
    const response = await proxyFetch(`/buckets/list?${qs.toString()}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to list bucket objects: ${response.statusText}`);
    }
    return response.json();
  },

  async registerExistingStorage(body: {
    storage_uri: string;
    key: string;
    connection_id: string;
    schema_json?: string | null;
  }): Promise<UploadResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/register-existing-storage`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to register cloud file: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Register an ENTIRE FOLDER as one dataset (Hive-partitioned Parquet / Iceberg).
   * NOTE: storage_uri is the BUCKET ROOT only — the prefix goes in `folder`.
   */
  async registerExistingFolder(body: {
    storage_uri: string;
    folder: string;
    connection_id: string;
  }): Promise<UploadResponse & { table_type?: string }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/register-existing-folder`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      const message =
        (typeof errorData.detail === "string" ? errorData.detail : errorData.detail?.message) ||
        errorData.message ||
        `Failed to register folder: ${response.statusText}`;
      const err = new Error(message) as Error & { status?: number };
      err.status = response.status;
      throw err;
    }
    return response.json();
  },

  /**
   * Create a cloud storage connection. The keys go to the backend over HTTPS,
   * which seals them under DB_ENCRYPTION_KEY and writes only the
   * *_ciphertext / *_iv columns of `cloud_datasets`: since migration
   * 20261116130000 the table refuses plaintext access_key/secret_key, so the
   * UI never writes a key itself (app/api/cloud_connections.py, build_router).
   */
  async createCloudConnection(payload: {
    name: string;
    provider: "aws" | "azure" | "gcp";
    region?: string;
    endpoint_url?: string;
    project_id?: string;
    access_key?: string;
    secret_key: string;
    bucket_name: string;
  }): Promise<{ id: string; name: string; provider: string; bucket_name: string }> {
    if (payload.provider === "gcp") {
      try {
        const parsed = JSON.parse(payload.secret_key) as Record<string, unknown>;
        if (payload.project_id && !parsed.project_id) parsed.project_id = payload.project_id;
        payload = { ...payload, secret_key: JSON.stringify(parsed) };
      } catch {
        throw new Error("Service account key must be valid JSON.");
      }
    }
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/cloud-connections`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        name: payload.name.trim() || `${payload.provider.toUpperCase()} connection`,
        provider: payload.provider,
        region: payload.region ?? null,
        endpoint_url: payload.endpoint_url ?? null,
        project_id: payload.project_id ?? null,
        bucket_name: payload.bucket_name.trim(),
        access_key: payload.access_key ?? null,
        secret_key: payload.secret_key,
      }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      const message =
        (typeof errorData.detail === "string" ? errorData.detail : errorData.detail?.message) ||
        `Failed to save the connection: ${response.statusText}`;
      throw new Error(message);
    }
    return response.json();
  },

  // No registerDatabase / listDatabaseTables here: they posted to
  // /api/database/register and /api/database/tables, which no backend route
  // serves, and nothing called them. Databases are registered by
  // DatabaseConnectModal through the onboarding API (/customers/register).

  async createDatabaseSession(connectionId: string): Promise<DatabaseSessionResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/database/connect`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({ connection_id: connectionId }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Session creation failed: ${response.statusText}`);
    }
    return response.json();
  },

  async executeDatabaseQuery(connectionId: string, sessionId: string, query: string): Promise<DatabaseQueryResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/v1/database/query`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({ connection_id: connectionId, session_id: sessionId, query }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Query execution failed: ${response.statusText}`);
    }
    return response.json();
  },

  async getAnalysisCode(
    analysisId: string,
  ): Promise<{ available: boolean; signed_url?: string; object_key?: string } | null> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/analysis/${analysisId}/code`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (response.status === 404) return null;
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || errorData.error || `Failed to fetch code: ${response.statusText}`);
    }
    return response.json();
  },

  async fetchSignedCode(signedUrl: string): Promise<string> {
    const response = await fetch(signedUrl);
    if (!response.ok) throw new Error(`Failed to download code: ${response.status} ${response.statusText}`);
    return response.text();
  },

  async sendInsightFeedback(
    analysisId: string,
    body: { insight_id: string; feedback_type: "positive" | "negative"; comment: string | null },
  ): Promise<any> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/analysis/${analysisId}/feedback`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || errorData.error || `Feedback failed: ${response.statusText}`);
    }
    return response.json().catch(() => ({}));
  },

  async getLlmModels(): Promise<LlmModelsResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/llm-models`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to load language models: ${response.statusText}`);
    return response.json();
  },

  async listModels(): Promise<{ models: BackendModel[] }> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/models`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to list models: ${response.statusText}`);
    return response.json();
  },

  async getModel(runId: string, sessionId?: string): Promise<BackendModelDetail> {
    const accessToken = await getAccessToken();
    let sid = sessionId;
    if (!sid && typeof window !== "undefined") {
      try {
        sid =
          window.localStorage.getItem("avaloka_session_id") ||
          window.localStorage.getItem("avaloka.session_id") ||
          undefined;
      } catch {
        /* ignore */
      }
    }
    const response = await proxyFetch(`/api/models/${encodeURIComponent(runId)}`, {
      method: "GET",
      sessionId: sid,
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Failed to fetch model: ${response.statusText}`);
    return response.json();
  },

  /**
   * Inference runs through the normal thread/chat pipeline (same as the old UI):
   * create/reuse a thread, post a natural-language "Run inference with …" message,
   * then parse the assistant reply for the prediction value.
   */
  async runInferenceViaThread(
    runId: string,
    features: Record<string, unknown>,
    datasetId?: string | null,
    sessionIdOverride?: string | null,
  ): Promise<{ prediction: unknown; reply: string; probabilities?: unknown; rawResponse: MessageResponse }> {
    const sessionId = sessionIdOverride || getInferenceSessionId();
    if (!datasetId) {
      throw new Error("This model has no linked training dataset, so inference can't run.");
    }
    const threadKey = `${runId}::${sessionId}::${datasetId ?? ""}`;
    let threadId = _inferenceThreads.get(threadKey);
    if (!threadId) {
      const thread = await this.createThread(sessionId);
      threadId = thread.thread_id;
      _inferenceThreads.set(threadKey, threadId);
    }

    const parts = Object.entries(features).map(([k, v]) => `${k}=${v}`);
    const content =
      parts.length > 1
        ? `Run inference with ${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}.`
        : `Run inference with ${parts.join("")}.`;

    const res = await this.sendMessage(
      threadId,
      sessionId,
      content,
      [datasetId],
      undefined,
      undefined,
      { dataset_id: datasetId },
    );
    const msgs = (res.messages ?? []) as Array<{ role: string; content: unknown }>;
    const reply =
      [...msgs]
        .reverse()
        .map((m) => (m.role !== "user" ? contentToText(m.content) : ""))
        .find((t) => t.trim().length > 0) ?? "";

    const prediction =
      parsePredictionFromReply(reply) ??
      parsePredictionFromJson((res as any).output_json) ??
      parsePredictionFromJson((res as any).output_file_data);

    const probabilities = [res.output_json, res.output_file_data, res]
      .map((value) => {
        if (typeof value !== "string") return value;
        try { return JSON.parse(value); } catch { return null; }
      })
      .map((value) => Array.isArray(value) ? value[0] : value)
      .find((value) => value && typeof value === "object" && ("probabilities" in value || "class_probabilities" in value)) as
      | { probabilities?: unknown; class_probabilities?: unknown }
      | undefined;
    return {
      prediction,
      reply,
      probabilities: probabilities?.probabilities ?? probabilities?.class_probabilities,
      rawResponse: res,
    };
  },


  /** One Instant Analysis run (GET /api/instant/{run_id}): verdict, report, export links. */
  async getInstantRun(runId: string): Promise<unknown> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/instant/${encodeURIComponent(runId)}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!response.ok) throw new Error(`Run fetch failed: ${response.statusText}`);
    return response.json();
  },

  /**
   * A run's self-contained HTML export, as text. A backend path
   * ("/api/instant/{run_id}/export.html") goes through the proxy with the
   * bearer; an absolute URL (a signed object-store link) is fetched as is.
   */
  async getInstantExport(url: string): Promise<string> {
    const response = /^https?:\/\//i.test(url)
      ? await fetch(url, { method: "GET" })
      : await proxyFetch(url, {
          method: "GET",
          headers: { Authorization: `Bearer ${await getAccessToken()}` },
        });
    if (!response.ok) throw new Error(`Export fetch failed: ${response.statusText}`);
    return response.text();
  },

  _mcpApi: MCP_API_BASE,

  _mcpTool: MCP_TOOL_BASE,
};

export interface BackendModel {
  run_id: string;
  name?: string;
  model_type?: string;
  version?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export interface BackendTrainingHistoryPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  [key: string]: unknown;
}

export interface BackendModelDetail extends BackendModel {
  description?: string;
  endpoint?: string;
  final_train_loss?: number;
  final_val_loss?: number;
  final_accuracy?: number;
  final_f1_score?: number;
  final_mae?: number;
  final_rmse?: number;
  final_r2_score?: number;
  rows_processed?: number;
  num_epochs_trained?: number;
  num_features?: number;
  num_classes?: number;
  training_history?: BackendTrainingHistoryPoint[];
}

export interface BackendInferenceResult {
  prediction?: unknown;
  model_name?: string;
  model_version?: string | number;
  model_type?: string;
  probabilities?: unknown;
  [key: string]: unknown;
}