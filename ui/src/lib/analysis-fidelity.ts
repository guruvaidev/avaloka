import { getAccessToken, proxyFetch } from "./api/backendApi";

export type AnalysisFidelity = "quick_sample" | "portfolio_samples" | "entire_dataset";
export type FidelityOption = {
  value: AnalysisFidelity;
  label: string;
  enabled: boolean;
  detail: string;
  reason?: string | null;
  status?: "ready" | "building" | "not_built";
  samples?: string[];
  runs_as_job?: boolean;
};
export type FidelityState = {
  dataset_id: string;
  analysis_fidelity: AnalysisFidelity;
  selected_sample_name: string | null;
  file_size_bytes: number;
  is_large_dataset: boolean;
  analysis_task_id: string | null;
  options: FidelityOption[];
  background_task_id?: string | null;
};
export type FidelityReply = {
  analysis_fidelity?: string;
  selected_sample_name?: string | null;
};
export class FidelityError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
export const FIDELITY_POLL_MS = 10_000;
export function needsFidelityPolling(state: FidelityState) {
  return Boolean(state.background_task_id) || state.options.some((o) => o.value === "portfolio_samples" && o.status === "building");
}
export function fidelityReplyDiffers(state: FidelityState, reply: FidelityReply) {
  return (reply.analysis_fidelity !== undefined && reply.analysis_fidelity !== state.analysis_fidelity)
    || (reply.selected_sample_name !== undefined && reply.selected_sample_name !== state.selected_sample_name);
}
export function isFullDatasetTask(reply: FidelityReply & { analysis_task_id?: string | null }, current?: string) {
  return Boolean(reply.analysis_task_id) && (reply.analysis_fidelity ?? current) === "entire_dataset";
}
async function request<T>(datasetId: string, sessionId: string, suffix: string, body?: object, signal?: AbortSignal): Promise<T> {
  const token = await getAccessToken();
  const response = await proxyFetch(`/api/datasets/${encodeURIComponent(datasetId)}/${suffix}`, {
    method: body ? "PUT" : "GET",
    sessionId,
    headers: { Authorization: `Bearer ${token}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new FidelityError(typeof error?.detail === "string" ? error.detail : `Could not update analysis mode (${response.status})`, response.status);
  }
  return response.json();
}
export const fidelityApi = {
  get: (id: string, session: string, signal?: AbortSignal) => request<FidelityState>(id, session, "fidelity", undefined, signal),
  put: (id: string, session: string, body: { analysis_fidelity?: AnalysisFidelity; selected_sample_name?: string }, signal?: AbortSignal) => request<FidelityState>(id, session, "fidelity", body, signal),
  status: (id: string, session: string, signal?: AbortSignal) => request<{ state: "NONE" | "PENDING" | "RUNNING" | "SUCCESS" | "FAILURE" }>(id, session, "background-task-status", undefined, signal),
};