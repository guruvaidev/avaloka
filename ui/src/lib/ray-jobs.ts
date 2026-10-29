import { queryOptions } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/external-client";
import { BACKEND_API_BASE } from "./api/backend-config";

export type JobStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "STOPPED";
export type JobKind = "data_transfer" | "mta_training" | "inference";
export interface RayJob {
  submission_id: string;
  kind: JobKind;
  status: JobStatus;
  message: string | null;
  thread_id: string | null;
  analysis_id: string | null;
  source: string | null;
  destination: string | null;
  rows_written: number | null;
  rows_inserted?: number | null;
  rows?: number | null;
  row_count?: number | null;
  result?: { rows_inserted?: number | null } | null;
  start_time?: number | string | null;
  timezone?: string | null;
  tz?: string | null;
  runtime_s: number | null;
  created_at: string | null;
  ended_at: string | null;
}
export interface RayStatus { configured: boolean; reachable: boolean; ray_version?: string; nodes?: number | null; timezone?: string | null; tz?: string | null }
export interface RayLogs { submission_id: string; lines: string[]; truncated: boolean }
export interface RayFilters { status: string; kind: string }
export class RayApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    const value = detail && typeof detail === "object" ? (detail as Record<string, unknown>).message : detail;
    super(typeof value === "string" ? value : status === 404 ? "Job not found" : status === 410 ? "Logs are no longer available on the Ray cluster" : "Ray cluster unavailable");
  }
  get upgradeRequired() {
    return this.status === 403 && !!this.detail && typeof this.detail === "object" && (this.detail as Record<string, unknown>).reason === "plan_upgrade_required";
  }
}
export type RayResult<T> = { data: T; error: null } | { data: null; error: RayApiError };

export interface JobResult { rowsInserted?: number; table?: string }
export function parseJobResult(log: string): JobResult {
  let result: JobResult = {};
  for (const line of log.split(/\r?\n/)) {
    const match = /AVALOKA_RESULT\b.*\brows_inserted=(\d+)\b/.exec(line);
    if (!match) continue;
    const rowsInserted = Number(match[1]);
    if (!Number.isSafeInteger(rowsInserted)) continue;
    const table = /\btable=([^\s]+)/.exec(line)?.[1];
    result = table ? { rowsInserted, table } : { rowsInserted };
  }
  return result;
}
export function displayJobId(id: string): string {
  return id.replace(/^([a-z]+)-(?:\1-)+/, "$1-");
}
export function jobRows(job: RayJob, result?: JobResult): number | undefined {
  return [job.rows_inserted, job.rows, job.row_count, job.result?.rows_inserted, job.rows_written, result?.rowsInserted]
    .find((value): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0);
}
export const rayJobResultsOptions = queryOptions({
  queryKey: ["ray-jobs", "reported-results"],
  queryFn: () => ({} as Record<string, JobResult>),
  initialData: {} as Record<string, JobResult>,
  enabled: false,
  staleTime: Infinity,
  gcTime: Infinity,
});
export function jobStartTime(job: RayJob): number | undefined {
  for (const value of [job.start_time, job.created_at]) {
    const time = typeof value === "number" ? value : typeof value === "string" ? Date.parse(value) : NaN;
    if (Number.isFinite(time) && Number.isFinite(new Date(time).getTime())) return time;
  }
  return undefined;
}
export function localJobStart(job: RayJob, locale?: string, timeZone?: string): string | undefined {
  const time = jobStartTime(job);
  if (time === undefined) return undefined;
  const options: Intl.DateTimeFormatOptions = {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
    hourCycle: "h23", timeZoneName: "short", ...(timeZone ? { timeZone } : {}),
  };
  const formatter = new Intl.DateTimeFormat(locale, options);
  const parts = formatter.formatToParts(time);
  const zone = parts.find((part) => part.type === "timeZoneName");
  if (zone && /^GMT[+−-]/.test(zone.value)) {
    const generic = new Intl.DateTimeFormat(locale, { ...options, timeZoneName: "shortGeneric" })
      .formatToParts(time).find((part) => part.type === "timeZoneName")?.value;
    if (generic && !/^GMT[+−-]/.test(generic)) zone.value = generic;
  }
  return parts.map((part) => part.value).join("");
}
export function logClockNote(job: RayJob, cluster?: RayStatus): string {
  const zone = job.timezone || job.tz || cluster?.timezone || cluster?.tz;
  return zone ? `Log times are the Ray cluster's clock (${zone}).` : "Log times are the Ray cluster's clock, not your local time.";
}

// Mirror /api/limits transport without changing the shared backend client.
export async function rayRequest<T>(path: string, method = "GET", signal?: AbortSignal): Promise<T> {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session?.access_token) {
    if (typeof window !== "undefined" && path !== "/api/limits") window.location.href = "/";
    throw new RayApiError(401, "Session expired");
  }
  const response = await fetch(`${window.location.origin}/api/public/backend-proxy`, {
    method, signal,
    headers: {
      Authorization: `Bearer ${session.access_token}`,
      "X-Backend-Path": path,
      "X-Backend-Base": BACKEND_API_BASE,
    },
  });
  if (response.status === 401) window.location.href = "/";
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new RayApiError(response.status, body?.detail);
  if (body?.fallback === true || body?.error) throw new RayApiError(503, body?.detail);
  return body as T;
}
async function result<T>(path: string, signal?: AbortSignal): Promise<RayResult<T>> {
  try { return { data: await rayRequest<T>(path, "GET", signal), error: null }; }
  catch (error) {
    if (signal?.aborted) throw error;
    return { data: null, error: error instanceof RayApiError ? error : new RayApiError(503, null) };
  }
}
export interface RayLimits { features?: { data_transfer?: boolean }; limits?: { features?: { data_transfer?: boolean } } }
export function hasDataTransfer(limits: RayLimits | null | undefined) {
  return (limits?.features?.data_transfer ?? limits?.limits?.features?.data_transfer) === true;
}
const RAY_JOB_PLANS = new Set(["professional", "pro", "enterprise"]);
function planName(plan: unknown): string {
  if (typeof plan === "string") return plan;
  if (plan && typeof plan === "object") {
    const p = plan as Record<string, unknown>;
    for (const k of ["id", "slug", "tier", "name", "plan"]) {
      if (typeof p[k] === "string") return p[k] as string;
    }
  }
  return "";
}
/** Ray jobs are available on Professional and Enterprise plans, or whenever the backend reports data_transfer. */
export function canUseRayJobs(plan: unknown, limits: RayLimits | null | undefined): boolean {
  if (hasDataTransfer(limits)) return true;
  const name = planName(plan).toLowerCase().replace(/[\s_-]*plan$/, "").trim();
  return RAY_JOB_PLANS.has(name);
}
export const rayLimitsOptions = queryOptions({
  queryKey: ["ray-jobs", "limits"],
  queryFn: ({ signal }) => result<RayLimits>("/api/limits", signal),
  staleTime: 60_000,
});
export const rayStatusOptions = queryOptions({
  queryKey: ["ray-jobs", "status"],
  staleTime: 0,
  queryFn: ({ signal }) => result<RayStatus>("/api/ray/status", signal),
  refetchInterval: 30_000,
  refetchIntervalInBackground: false,
  refetchOnWindowFocus: true,
});
export interface RayJobsResult {
  data: { jobs: RayJob[] } | null;
  error: RayApiError | null;
  refreshFailures: number;
  refreshedAt?: number;
}
export function rayJobsOptions(filters: RayFilters) {
  const query = new URLSearchParams({ kind: filters.kind, status: filters.status, limit: "50" });
  return queryOptions({
    queryKey: ["ray-jobs", "list", filters],
    staleTime: 0,
    queryFn: async ({ signal, client, queryKey }): Promise<RayJobsResult> => {
      const previous = client.getQueryData<RayJobsResult>(queryKey);
      const response = await result<{ jobs: RayJob[] }>(`/api/ray/jobs?${query}`, signal);
      if (response.error) return {
        data: previous?.data ?? null, error: response.error,
        refreshFailures: (previous?.refreshFailures ?? 0) + 1,
        refreshedAt: previous?.refreshedAt,
      };
      const jobs = [...response.data.jobs].sort((a, b) => (jobStartTime(b) ?? 0) - (jobStartTime(a) ?? 0));
      return { data: { jobs }, error: null, refreshFailures: 0, refreshedAt: Date.now() };
    },
    retry: false,
    refetchInterval: (query) => (query.state.data?.refreshFailures ?? 0) >= 3 ? 30_000
      : query.state.data?.data?.jobs.some(isActiveJob) ? 5000 : 30_000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
}
export function isActiveJob(job: RayJob) { return job.status === "PENDING" || job.status === "RUNNING"; }
export function matchesJobId(submissionId: string, chatId: string) { return !!chatId && submissionId.endsWith(chatId); }
export function jobTitle(job: RayJob) { return `${job.source ?? "—"} → ${job.destination ?? "—"}`; }
export function shortenPath(path: string, max = 34) {
  if (path.length <= max) return path;
  const side = Math.floor((max - 1) / 2);
  return `${path.slice(0, side)}…${path.slice(-side)}`;
}
export function durationText(job: RayJob, now = Date.now()) {
  const start = job.created_at ? Date.parse(job.created_at) : NaN;
  const seconds = job.status === "RUNNING" && Number.isFinite(start) ? Math.max(0, (now - start) / 1000) : job.runtime_s;
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} m ${Math.floor(seconds % 60)} s`;
}
export function relativeStarted(date: string | null, now = Date.now()) {
  if (!date || !Number.isFinite(Date.parse(date))) return "—";
  const minutes = Math.max(0, Math.floor((now - Date.parse(date)) / 60_000));
  if (!minutes) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} h ago`;
  if (minutes < 2880) return "Yesterday";
  return `${Math.floor(minutes / 1440)} days ago`;
}
export function rayJobMarkdown(content: string) {
  // Keep existing links and fenced/inline code intact; link only ordinary text.
  return content.split(/(```[\s\S]*?```|`[^`]*`|!?\[[^\]]*\]\([^)]*\))/g).map((part, i) => i % 2 ? part : part.replace(/\bdta-[0-9a-f]{16}\b/g, (id) => `[${id}](/ray-jobs?job=${id})`)).join("");
}