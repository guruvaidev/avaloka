/**
 * The plan could not be checked (a hosted deployment whose plan lookup failed):
 * GET /api/limits answers `plan_check_unavailable: true`, and the gated routes
 * answer 503 with reason "plan_check_unavailable". Neither means the user's plan
 * lacks anything, so nothing here offers an upgrade.
 */
export const PLAN_CHECK_UNAVAILABLE_TITLE = "Plan check unavailable";
export const PLAN_CHECK_UNAVAILABLE_MESSAGE =
  "Your plan could not be checked right now, so this selection cannot be accepted yet. Please try again in a minute.";
/** Fallback when a 503 plan_check_unavailable carries no message of its own. */
export const PLAN_CHECK_UNAVAILABLE_RETRY =
  "Plan check unavailable: your plan could not be checked right now. Please try again in a minute.";
