import { queryOptions } from "@tanstack/react-query";
import { z } from "zod";
import { getAccessToken, proxyFetch } from "@/lib/api/backendApi";
import { getModelMetrics } from "@/lib/modelMetrics";

/* ------------------------------------------------------------------ types */
export type InputRow = Record<string, string | number | boolean | null>;
export interface Feature { name: string; numeric: boolean; min?: number; max?: number; categories?: string[] }
export interface InferenceModel {
  key: string;
  runId?: string;
  name?: string;
  taskType?: string;
  isRegression: boolean;
  isClassification: boolean;
  version?: string;
  createdAt?: Date;
  description?: string;
  rowsProcessed?: number;
  epochs?: number;
  numFeatures?: number;
  /** Raw input columns (before encoding). */
  inputFeatureCount?: number;
  /** Columns after encoding, only when known and different from the input count. */
  encodedFeatureCount?: number;
  numClasses?: number;
  trainLoss?: number;
  valLoss?: number;
  mae?: number;
  rmse?: number;
  r2?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  baselineMae?: number;
  splitter?: string;
  target?: string;
  serviceRunning: boolean;
  /** pending or deploying: the worker is bringing the endpoint up. */
  serviceStarting: boolean;
  /** ready, but the serving pod is scaled to zero; the first prediction wakes it. */
  serviceAsleep: boolean;
  /** Endpoint lifecycle from Postgres: pending | deploying | ready | failed | stopped. */
  endpointStatus?: EndpointStatus;
  endpointError?: string;
  registeredName?: string;
  registryVersion?: number;
  usable: boolean;
  features: Feature[];
}
export type EndpointStatus = "pending" | "deploying" | "ready" | "failed" | "stopped";
export const ENDPOINT_STATUSES: readonly EndpointStatus[] = ["pending", "deploying", "ready", "failed", "stopped"];
export interface EndpointState { run_id?: string; status: EndpointStatus | null; asleep?: boolean; servable?: boolean; url?: string | null; server?: string | null; error?: string | null; registered_name?: string | null; version?: number | null; alias?: string | null; updated_at?: string | null }
export interface Prediction { value: string | number | boolean | null; probabilities: [string, number][] }
/** 202 from the API: the deploy runs in the worker; poll the endpoint. ("running" is the pre-2026-11 shape.) */
export interface ConfigureResponse { status: "deploying" | "running"; job_id?: string; endpoint?: EndpointState; inference_service_details?: Record<string, unknown> }
export interface StopResponse { status: "stopping" | "stopped"; job_id?: string; endpoint?: EndpointState }
export interface MlJob { id: string; kind: string; status: "queued" | "running" | "succeeded" | "failed" | "cancelled"; terminal: boolean; run_id?: string | null; error?: string | null }
export class InferenceApiError extends Error {
  constructor(public status: number, public detail: unknown, public call?: string) { super("Inference request failed"); }
}
/** The worker reported the deploy failed (or stopped) while we were polling. */
export class EndpointFailed extends InferenceApiError {
  constructor(public runId: string, public endpoint: EndpointState) {
    super(502, endpoint.error || "The inference endpoint could not be started.", "configure-inference-service");
    this.message = endpoint.error || "The inference endpoint could not be started.";
  }
}
/* ---------------------------------------------------------- safe readers */
export function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
export function toNumber(value: unknown): number | undefined {
  if (typeof value === "boolean" || value === null || value === undefined) return undefined;
  if (typeof value === "string" && value.trim() === "") return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}
function toText(value: unknown): string | undefined {
  if (value === null || value === undefined || typeof value === "object") return undefined;
  const s = String(value).trim();
  return s ? s : undefined;
}
function toDate(value: unknown): Date | undefined {
  if (value === null || value === undefined || value === "") return undefined;
  const d = new Date(typeof value === "number" ? value : String(value));
  return Number.isNaN(d.getTime()) ? undefined : d;
}
function firstNumber(...values: unknown[]) { for (const v of values) { const n = toNumber(v); if (n !== undefined) return n; } return undefined; }
function firstText(...values: unknown[]) { for (const v of values) { const s = toText(v); if (s !== undefined) return s; } return undefined; }

/* -------------------------------------------------------- feature schema */
const HOUSING_RANGES: Record<string, [number, number]> = {
  longitude: [-124.35, -114.31], latitude: [32.54, 41.95], housing_median_age: [1, 52], total_rooms: [2, 39320],
  total_bedrooms: [1, 6445], population: [3, 35682], households: [1, 6082], median_income: [0.5, 15.0],
};
const HOUSING_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"];
export const HOUSING_FEATURES: Feature[] = [
  ...Object.entries(HOUSING_RANGES).map(([name, [min, max]]) => ({ name, numeric: true, min, max })),
  { name: "ocean_proximity", numeric: false, categories: HOUSING_CATEGORIES },
];
export const HOUSING_EXAMPLE_ROW: Record<string, string> = {
  longitude: "-122.23", latitude: "37.88", housing_median_age: "41", total_rooms: "880", total_bedrooms: "129",
  population: "322", households: "126", median_income: "8.3252", ocean_proximity: "NEAR BAY",
};
export const isHousingModel = (target?: string, name?: string) => /median_house_value/i.test(`${target ?? ""} ${name ?? ""}`);
/** Fills missing ranges/categories from the housing defaults; schema values always win. */
export function withHousingDefaults(features: Feature[]): Feature[] {
  return features.map((f) => {
    const r = HOUSING_RANGES[f.name];
    if (r && f.numeric) return { ...f, min: f.min ?? r[0], max: f.max ?? r[1] };
    if (f.name === "ocean_proximity" && !f.categories?.length) return { ...f, numeric: false, categories: HOUSING_CATEGORIES };
    return f;
  });
}

/** Reads a feature schema from model details; returns [] when none is present. */
export function parseFeatureSchema(raw: unknown, target?: string): Feature[] {
  try {
    const model = asRecord(raw);
    const config = asRecord(model.config ?? model.model_config);
    const candidate = model.feature_schema ?? model.input_schema ?? model.features ?? model.feature_columns ?? model.feature_names ?? model.input_features ?? config.feature_schema ?? config.features ?? config.feature_columns;
    const schema = asRecord(candidate);
    const items: unknown[] = Array.isArray(candidate)
      ? candidate
      : Array.isArray(schema.features)
        ? schema.features
        : Object.entries(asRecord(schema.properties ?? candidate)).map(([name, data]) => ({ name, ...(typeof data === "string" ? { type: data } : asRecord(data)) }));
    const typeMap = asRecord(model.feature_types ?? model.dtypes ?? config.feature_types);
    const ranges = asRecord(model.feature_statistics ?? model.feature_stats ?? model.training_ranges ?? model.feature_ranges);
    const categoryMap = asRecord(model.categories ?? model.category_mappings ?? config.categories);
    const out: Feature[] = [];
    for (const item of items) {
      const spec = typeof item === "string" ? { name: item } : asRecord(item);
      const name = toText(spec.name ?? spec.feature ?? spec.column);
      if (!name || name === target || out.some((f) => f.name === name)) continue;
      const range = asRecord(ranges[name]);
      const categories = spec.categories ?? spec.enum ?? spec.values ?? categoryMap[name];
      const type = toText(spec.type ?? spec.dtype ?? typeMap[name]) ?? "";
      const list = Array.isArray(categories) ? categories.map((c) => toText(c)).filter((c): c is string => !!c) : undefined;
      const numeric = !list?.length && (/int|float|double|number|numeric|decimal|real|long/i.test(type) || (!type && !/proximity|category|class|label|name/i.test(name)));
      out.push({ name, numeric, min: toNumber(spec.min ?? spec.minimum ?? range.min), max: toNumber(spec.max ?? spec.maximum ?? range.max), ...(list?.length ? { categories: list } : {}) });
    }
    return out;
  } catch (error) {
    console.warn("[inference] feature schema unreadable", error);
    return [];
  }
}

/* ------------------------------------------------------------ normalizers */
function normalizeOne(raw: unknown, index: number): InferenceModel {
  const m = asRecord(raw);
  const report = asRecord(m.evaluation_report);
  const baseline = asRecord(report.baseline);
  const modelMetrics = getModelMetrics(m);
  const runId = firstText(m.run_id, m.mlflow_run_id);
  const name = firstText(m.model_name, m.name);
  const rawType = firstText(m.model_type, m.task_type)?.toLowerCase();
  const taskType = rawType ? rawType.charAt(0).toUpperCase() + rawType.slice(1) : undefined;
  const { mae, rmse, r2 } = modelMetrics;
  const isClassification = rawType === "classification";
  const isRegression = rawType === "regression" || (!isClassification && (mae !== undefined || r2 !== undefined));
  const target = firstText(m.target_column, m.target, asRecord(m.config ?? m.model_config).target_column, report.target);
  let features = parseFeatureSchema(m, target);
  if (isHousingModel(target, name)) features = features.length ? withHousingDefaults(features) : HOUSING_FEATURES;
  const numFeatures = toNumber(m.num_features);
  const inputFeatureCount = firstNumber(m.input_feature_count, m.num_input_features) ?? (features.length || numFeatures);
  const encodedRaw = firstNumber(m.encoded_feature_count, m.num_encoded_features, m.num_features_after_encoding) ?? numFeatures;
  const encodedFeatureCount = encodedRaw !== undefined && encodedRaw !== inputFeatureCount ? encodedRaw : undefined;
  return {
    key: runId ?? `index-${index}`,
    runId,
    name,
    taskType,
    isRegression,
    isClassification,
    version: firstText(m.model_version, m.version),
    createdAt: toDate(m.created_at),
    description: firstText(m.model_description, m.description),
    rowsProcessed: toNumber(m.rows_processed),
    epochs: toNumber(m.num_epochs_trained),
    numFeatures,
    inputFeatureCount,
    encodedFeatureCount,
    numClasses: toNumber(m.num_classes),
    trainLoss: modelMetrics.trainLoss,
    valLoss: modelMetrics.valLoss,
    mae,
    rmse,
    r2,
    accuracy: modelMetrics.accuracy,
    precision: modelMetrics.precision,
    recall: modelMetrics.recall,
    f1: modelMetrics.f1,
    baselineMae: toNumber(baseline.mae),
    splitter: modelMetrics.splitter,
    target: target ?? (/median_house_value/i.test(name ?? "") ? "median_house_value" : undefined),
    ...endpointFields(m),
    usable: Boolean(runId && name),
    features,
  };
}

/** Endpoint state from Postgres wins; the legacy artifact flag is the fallback for pre-registry runs. */
export function endpointFields(m: Record<string, unknown>): Pick<InferenceModel, "serviceRunning" | "serviceStarting" | "serviceAsleep" | "endpointStatus" | "endpointError" | "registeredName" | "registryVersion"> {
  const endpoint = asRecord(m.endpoint);
  const registry = asRecord(m.registry);
  const rawStatus = toText(endpoint.status);
  const endpointStatus = (ENDPOINT_STATUSES as readonly string[]).includes(rawStatus ?? "") ? (rawStatus as EndpointStatus) : undefined;
  const legacyRunning = m.inference_service_details !== null && m.inference_service_details !== undefined && m.inference_service_details !== false;
  return {
    serviceRunning: endpointStatus ? endpointStatus === "ready" : legacyRunning,
    serviceStarting: endpointStatus === "pending" || endpointStatus === "deploying",
    serviceAsleep: endpointStatus === "ready" && endpoint.asleep === true,
    endpointStatus,
    endpointError: toText(endpoint.error),
    registeredName: toText(registry.registered_name ?? endpoint.registered_name),
    registryVersion: toNumber(registry.version ?? endpoint.version),
  };
}

let loggedKeys = false;
export function normalizeModels(raw: unknown): InferenceModel[] {
  const items: unknown[] = Array.isArray(raw) ? raw : Array.isArray(asRecord(raw).models) ? (asRecord(raw).models as unknown[]) : [];
  if (!loggedKeys && items.length) { loggedKeys = true; console.debug("[inference] first model keys", Object.keys(asRecord(items[0]))); }
  const out: InferenceModel[] = [];
  items.forEach((item, i) => {
    try { out.push(normalizeOne(item, i)); }
    catch (error) { console.warn(toText(asRecord(item).run_id ?? asRecord(item).mlflow_run_id) ?? `index ${i}`, error); }
  });
  return out.sort((a, b) => (b.createdAt?.getTime() ?? 0) - (a.createdAt?.getTime() ?? 0));
}
export function normalizeModelDetails(raw: unknown): InferenceModel | null {
  try { return normalizeOne(asRecord(raw).model ?? raw, 0); }
  catch (error) { console.warn("[inference] model details unreadable", error); return null; }
}

function scalar(v: unknown): Prediction["value"] {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string" || typeof v === "boolean") return v;
  try { return JSON.stringify(v); } catch { return null; }
}
/** Maps a value, array of values or array of objects to exactly one entry per submitted row, in order. */
export function normalizeResult(result: unknown, rowCount: number, target?: string): Prediction[] {
  const bag = asRecord(result);
  const source = Array.isArray(result) ? result : Array.isArray(bag.predictions) ? bag.predictions : Array.isArray(bag.result) ? bag.result : [result];
  return Array.from({ length: Math.max(0, rowCount) }, (_, i) => {
    const v = source[i];
    const r = asRecord(v);
    const isObj = Object.keys(r).length > 0;
    const probs = r.probabilities ?? r.class_probabilities;
    const probabilities: [string, number][] = (Array.isArray(probs) ? probs.map((n, j) => [String(j), n] as const) : Object.entries(asRecord(probs)))
      .flatMap(([k, val]) => { const n = toNumber(val); return n === undefined ? [] : [[String(k), n] as [string, number]]; });
    const value = isObj ? r.prediction ?? r.predicted_value ?? r.predicted_class ?? r.class ?? (target ? r[target] : undefined) ?? r.value ?? Object.values(r).find((x) => typeof x === "number") : v;
    return { value: scalar(value), probabilities };
  });
}

/* -------------------------------------------------------------- formatters */
/** Shows a version as-is, prefixing "v" only when missing. */
export function formatVersion(version?: string): string | undefined {
  if (!version) return undefined;
  return /^v/i.test(version) ? version : `v${version}`;
}
export function formatNumber(v: unknown, digits = 2): string {
  const n = toNumber(v);
  return n === undefined ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function formatPercent(v: unknown, digits = 2): string {
  const n = toNumber(v);
  return n === undefined ? "—" : `${(n * 100).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
}
export function formatCurrency(v: unknown): string {
  const n = toNumber(v);
  return n === undefined ? "—" : n.toLocaleString("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
export function formatDate(v: unknown): string {
  const d = v instanceof Date ? (Number.isNaN(v.getTime()) ? undefined : v) : toDate(v);
  return d ? d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" }) : "—";
}
export function formatRelativeTime(v: unknown, now = Date.now()): string {
  const d = v instanceof Date ? (Number.isNaN(v.getTime()) ? undefined : v) : toDate(v);
  if (!d) return "—";
  const s = Math.round((now - d.getTime()) / 1000);
  const abs = Math.abs(s);
  if (abs < 60) return "just now";
  const [n, unit] = abs < 3600 ? [Math.round(abs / 60), "minute"] : abs < 86400 ? [Math.round(abs / 3600), "hour"] : abs < 86400 * 30 ? [Math.round(abs / 86400), "day"] : [0, ""];
  if (!unit) return d.toLocaleDateString("en-US", { dateStyle: "medium" });
  return s >= 0 ? `${n} ${unit}${n === 1 ? "" : "s"} ago` : `in ${n} ${unit}${n === 1 ? "" : "s"}`;
}
export const shortRunId = (runId?: string) => (runId ? runId.slice(0, 8) : "—");

/* ------------------------------------------------------------------ transport */
const runIdSchema = z.string().trim().min(1).max(512);
const rowSchema = z.record(z.union([z.string().max(10000), z.number().finite(), z.boolean(), z.null()]));
const bodySchema = z.union([rowSchema, z.array(rowSchema).min(1).max(10000)]);
async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const token = await getAccessToken();
  const call = path.split("/").pop();
  let sessionId: string | undefined;
  if (typeof window !== "undefined") {
    try { sessionId = window.localStorage.getItem("avaloka_session_id") || window.localStorage.getItem("avaloka.session_id") || undefined; } catch { /* storage unavailable */ }
  }
  let response: Response;
  try {
    response = await proxyFetch(path, { method, sessionId, headers: { Authorization: `Bearer ${token}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  } catch { throw new InferenceApiError(0, null, call); }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new InferenceApiError(response.status, data && typeof data === "object" && "detail" in data ? (data as { detail: unknown }).detail : data, call);
  return data as T;
}
function modelPath(runId: string) { return `/api/models/${encodeURIComponent(runIdSchema.parse(runId))}`; }
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
export interface PollOptions { intervalMs?: number; timeoutMs?: number }
/** Follows a 202 deploy: polls the endpoint until it is ready (resolves), failed/stopped (throws EndpointFailed) or times out. */
export async function waitForEndpoint(runId: string, { intervalMs = 2000, timeoutMs = 5 * 60_000 }: PollOptions = {}): Promise<EndpointState> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const endpoint = await inferenceApi.getEndpoint(runId);
    if (endpoint.status === "ready") return endpoint;
    if (endpoint.status === "failed" || endpoint.status === "stopped") throw new EndpointFailed(runId, endpoint);
    if (Date.now() >= deadline) throw new EndpointFailed(runId, { ...endpoint, error: "Timed out waiting for the inference endpoint." });
    await sleep(intervalMs);
  }
}
export const inferenceApi = {
  listModels: () => request<unknown>("/api/models"),
  getModel: (runId: string) => request<unknown>(modelPath(runId)),
  getEndpoint: (runId: string) => request<EndpointState>(`${modelPath(runId)}/endpoint`),
  getJob: (jobId: string) => request<MlJob>(`/api/ml-jobs/${encodeURIComponent(runIdSchema.parse(jobId))}`),
  predict: (runId: string, rows: InputRow | InputRow[]) => request<{ result?: unknown }>(`${modelPath(runId)}/predict`, "POST", bodySchema.parse(rows)),
  /** Asks for the deploy (202) and resolves only when the endpoint is ready. A pre-2026-11 server answers 200 "running" and resolves at once. */
  startService: async (runId: string, poll: PollOptions = {}): Promise<ConfigureResponse> => {
    const accepted = await request<ConfigureResponse>(`${modelPath(runId)}/configure-inference-service`, "POST");
    // Poll only on an explicit 202 "deploying"; anything else is the old synchronous shape.
    if (asRecord(accepted).status !== "deploying") return accepted;
    const endpoint = await waitForEndpoint(runId, poll);
    return { ...accepted, status: "deploying", endpoint };
  },
  stopService: (runId: string) => request<StopResponse>(`${modelPath(runId)}/stop-inference-service`, "POST"),
};
const queryDefaults = { retry: 1, staleTime: 60_000, refetchOnWindowFocus: false, throwOnError: false } as const;
export const modelsQuery = () => queryOptions({ queryKey: ["inference", "models"], queryFn: async () => normalizeModels(await inferenceApi.listModels()), ...queryDefaults });
export const modelQuery = (runId: string) => queryOptions({ queryKey: ["inference", "model", runId], queryFn: async () => normalizeModelDetails(await inferenceApi.getModel(runId)), enabled: Boolean(runId), ...queryDefaults });
