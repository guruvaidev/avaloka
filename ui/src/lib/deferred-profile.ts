// The upload's data profile (and the Auto Insights written with it) can take
// minutes on a local model, so the upload answers first with data-only charts
// and profiling_status "pending". These helpers wait for the profile to land
// (GET /api/datasets/{id}/profile) and say what to show when it does.

export interface DeferredProfileResponse {
  dataset_id?: string;
  profiling_status?: string | null;
  profiling_error?: string | null;
  profiling_result?: Record<string, any> | null;
  visualization_config?: any;
  visualization_status?: string | null;
}

export type DeferredProfileOutcome =
  | { kind: "ready"; visualizationConfig: any | null }
  | { kind: "failed"; message: string }
  | { kind: "gone" };

export const PROFILE_FAILED_FALLBACK =
  "The data profile could not be built. Charts, tables and questions still work.";

const isPending = (status: unknown) => String(status ?? "").toLowerCase() === "pending";

/** The dataset whose profile is still running after this upload, if any. */
export function pendingProfileDatasetId(up: any): string | null {
  if (!up) return null;
  if (isPending(up.profiling_status) && up.dataset_id) return String(up.dataset_id);
  const first = Array.isArray(up.datasets) ? up.datasets[0] : null;
  if (first && isPending(first.profiling_status) && first.dataset_id) return String(first.dataset_id);
  return null;
}

/** What a finished profile response means for the page. */
export function deferredProfileOutcome(res: DeferredProfileResponse | null): DeferredProfileOutcome | null {
  if (!res) return null;
  const status = String(res.profiling_status ?? "").toLowerCase();
  if (status === "pending" || status === "") return null;
  if (status === "failed") {
    return { kind: "failed", message: res.profiling_error || PROFILE_FAILED_FALLBACK };
  }
  const vizReady = !res.visualization_status || res.visualization_status === "ready";
  return { kind: "ready", visualizationConfig: vizReady ? (res.visualization_config ?? null) : null };
}

/**
 * Poll `fetchOnce` until the profile lands. Network errors and the overall
 * budget end the wait quietly ("gone"): the notification bell still reports
 * the outcome, and a failed profile is never reported as "gone".
 */
export async function waitForDeferredProfile(
  fetchOnce: () => Promise<DeferredProfileResponse | null>,
  opts: { intervalMs?: number; timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<DeferredProfileOutcome> {
  const intervalMs = opts.intervalMs ?? 4_000;
  const deadline = Date.now() + (opts.timeoutMs ?? 20 * 60_000);
  let consecutiveErrors = 0;
  while (Date.now() < deadline) {
    if (opts.signal?.aborted) return { kind: "gone" };
    await new Promise((r) => setTimeout(r, intervalMs));
    if (opts.signal?.aborted) return { kind: "gone" };
    try {
      const outcome = deferredProfileOutcome(await fetchOnce());
      consecutiveErrors = 0;
      if (outcome) return outcome;
    } catch {
      consecutiveErrors += 1;
      if (consecutiveErrors >= 5) return { kind: "gone" };
    }
  }
  return { kind: "gone" };
}

/**
 * The dataset chips with a finished profile swapped in for the chip that was
 * waiting on it. The Auto Insights modal reads each dataset's config from its
 * chip, so updating only the page's config left the modal on the upload-time
 * charts forever (rweb18b: 3 uploads, 6+ minutes after the API had the full
 * profile). A chip whose config changed since the upload (another flow wrote
 * it) is left alone, as the page's own config is.
 *
 * The chip is recognised by its dataset id and its upload-time config's
 * CONTENT. Chips restored from the sessionStorage batch (dataset-batch.ts) are
 * JSON copies, never the object the upload produced: matching by identity
 * swapped nothing live (rweb18c: 360 s, modal reopened, still 6 upload charts).
 */
export function chipsWithFinishedProfile<T extends { id: string; visualizationConfig?: unknown; visualizationStatus?: string }>(
  chips: T[],
  datasetId: string,
  uploadConfig: unknown,
  finished: unknown,
): T[] {
  let changed = false;
  const next = chips.map((chip) => {
    if (chip.id !== datasetId) return chip;
    if (chip.visualizationConfig != null && !sameConfig(chip.visualizationConfig, uploadConfig)) return chip;
    changed = true;
    return { ...chip, visualizationConfig: finished, visualizationStatus: "ready" };
  });
  return changed ? next : chips;
}

/** The same viz config: the same object, or an equal JSON copy of it. */
function sameConfig(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}
