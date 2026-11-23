/**
 * What a Save & Execute response says actually happened.
 *
 * The endpoint answers HTTP 200 for a run that failed too: the outcome is in
 * `status` (with `available: false`). Only a run the server reports as a
 * success may be shown as "Saved & executed". The failure wording is written
 * here, in the user's terms; the server's detail is never shown.
 */
export type SaveExecuteOutcome =
  | { kind: "saved" }
  | { kind: "needs_fix" }
  | { kind: "failed"; message: string };

// The statuses save-and-execute returns for a run that produced a new version
// (server.py: the executor's own status, defaulting to "success").
const SAVED_STATUSES = new Set(["success", "succeeded", "completed", "dryrun"]);

const FAILURE_MESSAGES: Record<string, string> = {
  execution_failed: "Your code didn't finish, so nothing was saved. The current version is unchanged.",
  persistence_failed: "Your code ran, but the new version couldn't be saved. Please try again.",
};

const DEFAULT_FAILURE = "Save & Execute didn't complete, so nothing was saved.";

export function saveExecuteOutcome(body: unknown): SaveExecuteOutcome {
  const b = (body && typeof body === "object" ? body : {}) as Record<string, unknown>;
  const status = typeof b.status === "string" ? b.status.trim().toLowerCase() : "";
  if (status === "validation_failed" || b.validation_passed === false) {
    return { kind: "needs_fix" };
  }
  if (b.available === false || !SAVED_STATUSES.has(status)) {
    return { kind: "failed", message: FAILURE_MESSAGES[status] ?? DEFAULT_FAILURE };
  }
  return { kind: "saved" };
}
