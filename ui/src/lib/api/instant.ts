// Instant Analysis (/instant): "upload a file or try sample data → see a data
// profile preview → ask one question", and the report page (/instant/:runId)
// the question lands on. Everything the two pages need from the backend goes
// through here, so PR-6 swaps the question stub for the real recipe endpoint
// in one place.
//
// The preview's numbers are the API's: rows, columns, per-column type,
// missing share and distinct count come from the upload's sample_statistics
// (app/agents/sampling_async.py) and the quality concerns from the landed
// profile (GET /api/datasets/{id}/profile). Nothing here measures the data.
import { backendApi, type UploadLimits } from "@/lib/api/backendApi";
import { deferredProfileOutcome, type DeferredProfileResponse } from "@/lib/deferred-profile";

/** The same accept list as BulkUploadModal's file input. */
export const INSTANT_ACCEPT = ".csv,.xlsx,.xls";

/** Per-file cap until /api/limits answers — the same fallback UploadModal uses. */
export const FALLBACK_MAX_FILE_BYTES = 100 * 1024 * 1024;

export interface InstantLimits {
  plan: string;
  maxFileBytes: number;
  maxFileMb: number;
}

export interface SampleDataset {
  id: string;
  label: string;
  /** The file's name as uploaded (app/sample_data/<file>). */
  file: string;
  /** Where the UI serves it from (ui/public/samples/<file>). */
  path: string;
}

/** app/sample_data/*.csv, copied to ui/public/samples/ (each well under 1 MB). */
export const SAMPLE_DATASETS: SampleDataset[] = [
  { id: "salaries", label: "Salaries", file: "salaries.csv", path: "/samples/salaries.csv" },
  {
    id: "football",
    label: "Football team stats",
    file: "2021-2022 Football Team Stats.csv",
    path: "/samples/2021-2022%20Football%20Team%20Stats.csv",
  },
  {
    id: "youtube",
    label: "YouTube channels",
    file: "Global_YouTube_Statistics.csv",
    path: "/samples/Global_YouTube_Statistics.csv",
  },
];

export const EXAMPLE_QUESTIONS = [
  "What drives the biggest differences in this data?",
  "Which columns move together?",
  "Where are the outliers, and are they real?",
];

export interface ColumnProfile {
  name: string;
  type: string;
  /** Percent of rows missing, as the API reported it (0–100), or null when it did not. */
  missingPct: number | null;
  distinct: number | null;
}

export interface DataProfilePreview {
  rows: number | null;
  columns: number | null;
  columnsTable: ColumnProfile[];
  /** Quality issues the upload's statistics already name (high-missing, constant columns). */
  issues: string[];
}

export type InstantProfileStatus = "pending" | "ready" | "failed" | "skipped";

export interface InstantUpload {
  datasetId: string;
  fileName: string;
  profileStatus: InstantProfileStatus;
  profileError: string | null;
  preview: DataProfilePreview;
}

/** Plan limits for the drop zone, with the same fallback the upload modal uses. */
export async function fetchInstantLimits(
  api: Pick<typeof backendApi, "getUploadLimits"> = backendApi,
): Promise<InstantLimits> {
  let limits: UploadLimits | null = null;
  try {
    limits = await api.getUploadLimits();
  } catch {
    limits = null;
  }
  const maxFileBytes =
    typeof limits?.max_file_bytes === "number" ? limits.max_file_bytes : FALLBACK_MAX_FILE_BYTES;
  const maxFileMb =
    typeof limits?.max_file_mb === "number"
      ? limits.max_file_mb
      : Math.round((maxFileBytes / (1024 * 1024)) * 10) / 10;
  return { plan: String(limits?.plan ?? "free"), maxFileBytes, maxFileMb };
}

/** Why a file cannot be uploaded from the drop zone, or null when it can. */
export function rejectFile(
  file: File,
  limits: InstantLimits,
  accept = INSTANT_ACCEPT,
): string | null {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  const allowed = accept
    .split(",")
    .map((s) => s.trim().replace(/^\./, "").toLowerCase())
    .filter(Boolean);
  if (!ext || !allowed.includes(ext)) {
    return `${file.name} is not a ${allowed.map((a) => a.toUpperCase()).join(", ")} file.`;
  }
  if (file.size > limits.maxFileBytes) {
    return `${file.name} is larger than the ${limits.maxFileMb} MB per-file limit of the ${limits.plan} plan.`;
  }
  return null;
}

const asNumber = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const asRecord = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
const asStrings = (v: unknown): string[] =>
  Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string" && x.trim().length > 0)
    : [];

/**
 * The preview from an upload response. Reads sample_statistics (the sampler's
 * column_statistics and data_shape), or the quick profiling_result when the
 * statistics are absent. missing_pct is preferred; missing_ratio is the same
 * number as a fraction, shown as a percent.
 */
export function previewFromUpload(up: unknown): DataProfilePreview {
  const u = asRecord(up) ?? {};
  const stats = asRecord(u.sample_statistics) ?? asRecord(u.profiling_result) ?? {};
  const shape = asRecord(stats.data_shape) ?? {};
  const colStats = asRecord(stats.column_statistics) ?? {};
  const columnsTable: ColumnProfile[] = Object.entries(colStats).map(([name, s]) => {
    const st = asRecord(s) ?? {};
    const pct = asNumber(st.missing_pct);
    const ratio = asNumber(st.missing_ratio);
    return {
      name,
      type: typeof st.type === "string" && st.type ? st.type : "unknown",
      missingPct: pct ?? (ratio == null ? null : Math.round(ratio * 1000) / 10),
      distinct: asNumber(st.n_unique),
    };
  });
  const dq = asRecord(stats.data_quality) ?? {};
  const issues: string[] = [];
  const highMissing = asStrings(dq.columns_with_high_missing);
  if (highMissing.length) issues.push(`High missing share: ${highMissing.join(", ")}`);
  const constant = asStrings(dq.constant_columns);
  if (constant.length) issues.push(`Constant columns: ${constant.join(", ")}`);
  return {
    rows: asNumber(shape.rows) ?? asNumber(u.estimated_rows),
    columns: asNumber(shape.columns) ?? (columnsTable.length ? columnsTable.length : null),
    columnsTable,
    issues,
  };
}

/** The landed profile's data-quality concerns, as the profiler wrote them. */
export function issuesFromProfile(res: DeferredProfileResponse | null | undefined): string[] {
  const insights = asRecord(asRecord(res?.profiling_result)?.data_quality_insights);
  return asStrings(insights?.concerns);
}

export function profileStatusOf(status: unknown): InstantProfileStatus {
  const s = String(status ?? "").toLowerCase();
  if (s === "pending") return "pending";
  if (s === "failed" || s === "error") return "failed";
  if (s === "skipped" || s === "") return "skipped";
  return "ready";
}

/** POST /api/upload with one file; the preview is built from the response. */
export async function uploadForInstant(
  file: File,
  api: Pick<typeof backendApi, "uploadFile"> = backendApi,
): Promise<InstantUpload> {
  const up = asRecord(await api.uploadFile(file)) ?? {};
  return {
    datasetId: String(up.dataset_id),
    fileName: file.name,
    profileStatus: profileStatusOf(up.profiling_status),
    profileError: typeof up.profiling_error === "string" ? up.profiling_error : null,
    preview: previewFromUpload(up),
  };
}

/**
 * Poll GET /api/datasets/{id}/profile until the profile lands (or fails), and
 * return that response. Null when the wait was abandoned or the budget ran
 * out; the preview from the upload stands either way.
 */
export async function waitForInstantProfile(
  datasetId: string,
  opts: {
    intervalMs?: number;
    timeoutMs?: number;
    signal?: AbortSignal;
    api?: Pick<typeof backendApi, "getDatasetProfile">;
  } = {},
): Promise<DeferredProfileResponse | null> {
  const api = opts.api ?? backendApi;
  const intervalMs = opts.intervalMs ?? 3_000;
  const deadline = Date.now() + (opts.timeoutMs ?? 5 * 60_000);
  let consecutiveErrors = 0;
  while (Date.now() < deadline) {
    if (opts.signal?.aborted) return null;
    try {
      const res = await api.getDatasetProfile(datasetId);
      consecutiveErrors = 0;
      if (deferredProfileOutcome(res)) return res;
    } catch {
      consecutiveErrors += 1;
      if (consecutiveErrors >= 5) return null;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return null;
}

/** A sample CSV from the UI's public assets, as a File the upload path takes. */
export async function loadSampleFile(
  sample: SampleDataset,
  fetchImpl: typeof fetch = fetch,
): Promise<File> {
  const res = await fetchImpl(sample.path);
  if (!res.ok) throw new Error(`Sample ${sample.label} is unavailable (${res.status})`);
  // Bytes, not res.blob(): a Blob from another realm (jsdom's File over
  // Node's Response) is stringified to "[object Blob]" by the File constructor.
  const bytes = new Uint8Array(await res.arrayBuffer());
  return new File([bytes], sample.file, { type: "text/csv" });
}

// ---- the one question ------------------------------------------------------

/** PR-6 adds this route (app/api/instant.py); until then askInstant answers "coming soon". */
export const INSTANT_ASK_PATH = "/api/instant";

export interface InstantAskRequest {
  dataset_id: string;
  question: string;
  /**
   * The answer to a question-back (verdict "ask"): the recipe parameter the
   * run asked about and the option the user chose from `InstantRun.ask`.
   */
  clarify?: { parameter: string; option: string };
}

export type InstantAskResponse =
  | { status: "coming_soon"; dataset_id: string; question: string }
  // PR-6: the recipe run — the run the report page loads, and the recipe the router chose.
  | { status: "ready"; dataset_id: string; question: string; run_id: string; recipe: string };

export type InstantAskReady = Extract<InstantAskResponse, { status: "ready" }>;

/**
 * Submit the one question for a dataset. Placeholder: resolves to the
 * "coming soon" state without a network call. PR-6 replaces the body with a
 * proxyFetch to INSTANT_ASK_PATH and keeps the signature.
 */
export async function askInstant(req: InstantAskRequest): Promise<InstantAskResponse> {
  const question = req.question.trim();
  if (!question) throw new Error("Ask one question first.");
  return { status: "coming_soon", dataset_id: req.dataset_id, question };
}

// ---- the run and its report ------------------------------------------------

/** GET `${INSTANT_RUN_PATH}/{run_id}`: the run PR-6 stores (verdict, report, export links). */
export const INSTANT_RUN_PATH = INSTANT_ASK_PATH;

/** A run ends in exactly one of these (1.7-INSTANT-ANALYSIS.md §2.2); code writes it. */
export type InstantVerdict = "pass" | "qualified" | "ask" | "refuse";
export const INSTANT_VERDICTS: readonly InstantVerdict[] = ["pass", "qualified", "ask", "refuse"];

export type DataCheckStatus = "passed" | "qualified" | "failed";
export type CompletionState = "answered" | "partial" | "open";

// The Outcome Report 1.1 fields the page reads (avaloka/outcomes/outcome-report.schema.json).
// Figures, KPIs and charts are rendered only by the export inside the frame,
// so they stay unknown here: nothing on the page formats a number.

/** One row of the GateReport: a gate, its verdict, and the caveat kind it raised. */
export interface DataCheck {
  gate: string;
  status: DataCheckStatus;
  figure?: string;
  caveat_kind?: string;
  reason?: string;
}

export interface Caveat {
  kind: string;
  text: string;
}

export interface Completion {
  state: CompletionState;
  missing?: string[];
  open_questions?: string[];
}

export interface Method {
  summary: string;
  steps?: string[];
  [key: string]: unknown;
}

export interface Provenance {
  recipe?: { id: string; version: string };
  engine?: Record<string, string>;
  llm_used?: boolean;
  [key: string]: unknown;
}

export interface OutcomeReport11 {
  schema_version?: string;
  headline?: { text: string; bullets?: { so_what: string; now_what: string }[] };
  figures?: unknown[];
  kpis?: unknown[];
  charts?: unknown[];
  data_checks?: DataCheck[];
  caveats?: Caveat[];
  completion?: Completion;
  method?: Method;
  provenance?: Provenance;
  [key: string]: unknown;
}

export interface InstantRun {
  run_id: string;
  dataset_id: string;
  recipe: { id: string; version: string };
  verdict: InstantVerdict;
  reason?: string;
  /** The question as the run took it, when the backend stores it. */
  question?: string;
  report?: OutcomeReport11;
  /** The self-contained HTML export (backend path or absolute URL). */
  export_url?: string;
  notebook_url?: string;
  markdown_url?: string;
  /** True when the run is a free preview and the full report is paywalled. */
  locked?: boolean;
  /** Verdict "ask": the one question back, with the options the recipe accepts. */
  ask?: { parameter: string; text: string; options: string[] };
  /** Verdict "refuse": the closed-list reason and the plain sentence for it. */
  refusal?: { reason: string; text: string };
}

const asString = (v: unknown): string | undefined =>
  typeof v === "string" && v.length > 0 ? v : undefined;

const isCheckStatus = (v: unknown): v is DataCheckStatus =>
  v === "passed" || v === "qualified" || v === "failed";

const isCompletionState = (v: unknown): v is CompletionState =>
  v === "answered" || v === "partial" || v === "open";

/** The report fields the page reads, each kept only when it has the schema's shape. */
export function readReport(raw: unknown): OutcomeReport11 | undefined {
  const r = asRecord(raw);
  if (!r) return undefined;
  const checks: DataCheck[] = [];
  for (const c of Array.isArray(r.data_checks) ? r.data_checks : []) {
    const rec = asRecord(c);
    const gate = asString(rec?.gate);
    if (rec && gate && isCheckStatus(rec.status)) {
      checks.push({
        gate,
        status: rec.status,
        figure: asString(rec.figure),
        caveat_kind: asString(rec.caveat_kind),
        reason: asString(rec.reason),
      });
    }
  }
  const caveats: Caveat[] = [];
  for (const c of Array.isArray(r.caveats) ? r.caveats : []) {
    const rec = asRecord(c);
    const kind = asString(rec?.kind);
    const text = asString(rec?.text);
    if (kind && text) caveats.push({ kind, text });
  }
  const comp = asRecord(r.completion);
  const completion: Completion | undefined =
    comp && isCompletionState(comp.state)
      ? {
          state: comp.state,
          missing: asStrings(comp.missing),
          open_questions: asStrings(comp.open_questions),
        }
      : undefined;
  const meth = asRecord(r.method);
  const summary = asString(meth?.summary);
  const method: Method | undefined =
    meth && summary ? { ...meth, summary, steps: asStrings(meth.steps) } : undefined;
  return { ...r, data_checks: checks, caveats, completion, method };
}

/** The run as the backend answered it, or an error naming what was unreadable. */
export function readRun(raw: unknown): InstantRun {
  const r = asRecord(raw);
  const run_id = asString(r?.run_id);
  const dataset_id = asString(r?.dataset_id);
  const verdict = r?.verdict;
  if (!r || !run_id || !dataset_id || !INSTANT_VERDICTS.includes(verdict as InstantVerdict)) {
    throw new Error("This run could not be read.");
  }
  const recipe = asRecord(r.recipe);
  const ask = asRecord(r.ask);
  const refusal = asRecord(r.refusal);
  return {
    run_id,
    dataset_id,
    recipe: { id: asString(recipe?.id) ?? "", version: asString(recipe?.version) ?? "" },
    verdict: verdict as InstantVerdict,
    reason: asString(r.reason),
    question: asString(r.question),
    report: readReport(r.report),
    export_url: asString(r.export_url),
    notebook_url: asString(r.notebook_url),
    markdown_url: asString(r.markdown_url),
    locked: r.locked === true,
    ask: ask
      ? {
          parameter: asString(ask.parameter) ?? "",
          text: asString(ask.text) ?? "",
          options: asStrings(ask.options),
        }
      : undefined,
    refusal: refusal
      ? { reason: asString(refusal.reason) ?? "", text: asString(refusal.text) ?? "" }
      : undefined,
  };
}

/** GET the run; the page shows the verdict and reads the report's text fields. */
export async function fetchInstantRun(
  runId: string,
  api: Pick<typeof backendApi, "getInstantRun"> = backendApi,
): Promise<InstantRun> {
  return readRun(await api.getInstantRun(runId));
}

/** The HTML export as text, for the sandboxed frame; nothing here parses it. */
export async function fetchInstantExport(
  url: string,
  api: Pick<typeof backendApi, "getInstantExport"> = backendApi,
): Promise<string> {
  return api.getInstantExport(url);
}

export const instantApi = {
  fetchInstantLimits,
  uploadForInstant,
  waitForInstantProfile,
  loadSampleFile,
  askInstant,
  fetchInstantRun,
  fetchInstantExport,
};

export type InstantApi = typeof instantApi;
