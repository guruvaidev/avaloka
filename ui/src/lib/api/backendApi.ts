import { supabase } from "@/integrations/supabase/external-client";

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
  sample_statistics?: Record<string, any>;
  sample_status?: string;
  background_task_id?: string;
}

// Multi-dataset upload response format
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
  task_info?: { task_id: string; next_due_at?: number };
  analysis_fidelity?: "quick_sample" | "portfolio_samples" | "entire_dataset";
  selected_sample_name?: string;
  analysis_task_id?: string;
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

interface DatabaseRegistrationResponse {
  connection_id: string;
  customer_id: string;
  status: string;
  message: string;
}

interface DatabaseTablesResponse {
  tables: Array<{
    name: string;
    columns: string[];
    row_count?: number;
    schema?: string;
  }>;
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

import { BACKEND_API_BASE as API_BASE } from "@/lib/api/backend-config";

export interface PendingTurnResponse {
  status: "running" | "done" | "error" | "superseded" | "none" | string;
  message?: string;
  result?: any;
}

/**
 * A chat turn is "deferred" when the backend accepted it but is still
 * training in the background, or when our proxy gave up waiting on a
 * synchronous response. Both cases resolve later via /pending-turn.
 */
export function isDeferredTurn(res: any): boolean {
  if (!res || typeof res !== "object") return false;
  if (res.fallback === true && res.error === "UPSTREAM_TIMEOUT") return true;
  const hasRows = Array.isArray(res.output_json) && res.output_json.length > 0;
  return (
    String(res.training_status ?? "").toLowerCase() === "running" &&
    res.training_completed !== true &&
    !hasRows
  );
}
const MCP_API_BASE = "https://avaloka-mcp.ngrok.app";
const MCP_TOOL_BASE = "https://avaloka-mcp-tool.ngrok.app";

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



async function getAccessToken(): Promise<string> {
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
async function proxyFetch(path: string, init: RequestInit & { sessionId?: string } = {}): Promise<Response> {
  const { sessionId, headers: initHeaders, ...rest } = init;
  const headers = new Headers(initHeaders as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", API_BASE);
  if (sessionId) headers.set("X-Avaloka-Session", sessionId);
  return fetch(proxyUrl(), { ...rest, headers });
}

// MCP calls reuse the same backend-proxy, but forward to the MCP base.
// The proxy runs server-side, so the in-cluster MCP URL resolves there.
async function mcpProxyFetch(base: string, path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", base);
  return fetch(proxyUrl(), { ...init, headers });
}

export { mcpProxyFetch, proxyFetch };

export const backendApi = {
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

  async getVersion(): Promise<{ version: string }> {
    const response = await proxyFetch(`/version`, { method: "GET" });
    if (!response.ok) throw new Error(`Version check failed: ${response.statusText}`);
    return response.json();
  },

  async checkMcpHealth(): Promise<{ status: string; healthy: boolean }> {
    try {
      const response = await supabase.functions.invoke("check-mcp-health", { method: "GET" });
      if (response.error) return { status: "unhealthy", healthy: false };
      const data = response.data;
      return { status: data?.status || "healthy", healthy: data?.healthy !== false };
    } catch {
      return { status: "unhealthy", healthy: false };
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

  async uploadFile(file: File, sessionId?: string, backend?: string, connectionId?: string): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (sessionId) formData.append("session_id", sessionId);
    if (backend) formData.append("backend", backend);
    if (connectionId) formData.append("connection_id", connectionId);

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

  async uploadFiles(files: File[], backend?: string, connectionId?: string): Promise<MultiUploadResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    if (backend) formData.append("backend", backend);
    if (connectionId) formData.append("connection_id", connectionId);

    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/upload`, {
      method: "POST",
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      const err: any = new Error(errorData.detail || `Batch upload failed: ${response.statusText}`);
      err.status = response.status;
      throw err;
    }

    return response.json();
  },

  async sendMessage(
    threadId: string,
    sessionId: string,
    content: string,
    datasetIds?: string[],
    analysisFidelity?: "quick_sample" | "portfolio_samples" | "entire_dataset",
    selectedSampleName?: string,
    metadata?: { dataset_id?: string | null },
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
      ...(analysisFidelity && { analysis_fidelity: analysisFidelity }),
      ...(selectedSampleName && { selected_sample_name: selectedSampleName }),
      ...(metadata?.dataset_id && { metadata: { dataset_id: metadata.dataset_id } }),
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

  /**
   * Poll the backend for the result of a long-running (deferred) chat turn.
   * Returns the raw envelope: { status: "running" | "done" | "error" | "superseded" | "none", ... }
   */
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

  async getDatasetPreview(datasetId: string, sessionId: string, forceRefresh = false): Promise<any> {
    const previewKey = `${datasetId}:${sessionId}`;
    if (!forceRefresh && _previewInflight.has(previewKey)) return _previewInflight.get(previewKey)!;
    if (forceRefresh) this.abortPreview(datasetId);

    const promise = (async () => {
      const controller = new AbortController();
      _previewControllers.set(previewKey, controller);
      const timeoutId = setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), 60_000);

      try {
        const accessToken = await getAccessToken();
        const query = new URLSearchParams({ session_id: sessionId });
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
   * Create a cloud storage connection row in `cloud_datasets`. Matches the
   * existing production pattern (see src/lib/gcp-connection.ts) which stores
   * plaintext access_key/secret_key so the backend can initialise the cloud
   * SDK on /buckets/list and /api/register-existing-storage. When
   * DB_ENCRYPTION_KEY is set server-side, the backend rotates values into the
   * *_ciphertext / *_iv columns transparently.
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
    const { supabase } = await import("@/integrations/supabase/client");
    const { getCachedAuthUser } = await import("@/lib/auth-user");
    const uid = (await getCachedAuthUser())?.id;
    if (!uid) throw new Error("Not signed in");

    if (payload.provider === "gcp") {
      try {
        const parsed = JSON.parse(payload.secret_key) as Record<string, unknown>;
        if (payload.project_id && !parsed.project_id) parsed.project_id = payload.project_id;
        payload = { ...payload, secret_key: JSON.stringify(parsed) };
      } catch {
        throw new Error("Service account key must be valid JSON.");
      }
    }

    // For GCP the `access_key` column stores the Project ID (matches
    // gcp-connection.ts + backend get_cloud_connection).
    const access_key =
      payload.provider === "gcp"
        ? (payload.project_id ?? "").trim()
        : (payload.access_key ?? "").trim();

    const { data, error } = await supabase
      .from("cloud_datasets")
      .insert({
        user_id: uid,
        name: payload.name.trim() || `${payload.provider.toUpperCase()} connection`,
        provider: payload.provider,
        region: payload.region ?? null,
        endpoint_url: payload.endpoint_url ?? null,
        bucket_name: payload.bucket_name.trim(),
        access_key,
        secret_key: payload.secret_key,
      })
      .select("id,name,provider,bucket_name")
      .single();

    if (error) throw new Error(error.message);
    return data as { id: string; name: string; provider: string; bucket_name: string };
  },




  async registerDatabase(connectionData: {
    connection_name: string;
    db_type: string;
    host: string;
    port: number;
    database: string;
    username: string;
    password: string;
    use_ssl: boolean;
  }): Promise<DatabaseRegistrationResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/database/register`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify(connectionData),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Database registration failed: ${response.statusText}`);
    }
    return response.json();
  },

  async listDatabaseTables(connectionId: string): Promise<DatabaseTablesResponse> {
    const accessToken = await getAccessToken();
    const response = await proxyFetch(`/api/database/tables`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({ connection_id: connectionId }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}) as any);
      throw new Error(errorData.detail || `Failed to fetch tables: ${response.statusText}`);
    }
    return response.json();
  },

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
  ): Promise<{ prediction: unknown; reply: string }> {
    const sessionId = sessionIdOverride || getInferenceSessionId();
    if (!datasetId) {
      throw new Error("Select a dataset to provide inference session context.");
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

    return { prediction, reply };
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
