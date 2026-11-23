// Closed label maps for the report page (/instant/:runId). Keyed by the ids
// the recipes write: provenance.recipe.id (the ten recipes of
// 1.7-INSTANT-ANALYSIS.md §2.5) and data_checks[].gate (the GateReport, §2.1).
//
// Every string here is words only. The Evidence panel shows nothing but these
// labels, a status word and the report's own caveat sentence: the numbers a
// gate measured live inside the export, which the backend verified, and
// tests/unit/instant-labels.test.ts holds each label digit-free.
import type { CompletionState, DataCheckStatus, InstantVerdict } from "@/lib/api/instant";

export const RECIPE_IDS = [
  "profile",
  "describe",
  "trend",
  "compare_periods",
  "segment",
  "cohort_retention",
  "correlate",
  "regress",
  "anomalies",
  "ab_test",
] as const;

export type RecipeId = (typeof RECIPE_IDS)[number];

export const RECIPE_TITLES: Record<RecipeId, string> = {
  profile: "Data profile",
  describe: "Typical value, spread and shape",
  trend: "Movement over time",
  compare_periods: "Change between two periods",
  segment: "Which groups matter",
  cohort_retention: "Cohort retention",
  correlate: "Which measures move together",
  regress: "What explains the measure",
  anomalies: "Unusual points and periods",
  ab_test: "Did arm B differ from arm A",
};

/** The recipe's title, or the id spelled out when the backend sends one the map lacks. */
export function recipeTitle(id: string): string {
  return (RECIPE_TITLES as Record<string, string>)[id] ?? humanise(id);
}

/** data_checks[].gate → the row label. Each gate is one source-data check. */
export const GATE_LABELS: Record<string, string> = {
  // shared
  duplicates: "Duplicate rows",
  constant: "Constant column",
  min_rows: "Enough rows",
  missing_measure: "Missing values in the measure",
  fanout_join: "Fan-out from a join",
  // describe, segment
  small_groups: "Small groups",
  high_cardinality: "Too many groups to read",
  // trend, compare_periods, anomalies
  date_parse: "Dates parsed",
  period_gaps: "Gaps in the period sequence",
  min_periods: "Enough periods",
  period_overlap: "Periods kept apart",
  empty_period: "Both periods hold rows",
  flagged_share: "Share of points flagged",
  // cohort_retention
  min_cohorts: "Enough cohorts",
  id_column: "An id column to follow",
  // correlate, regress
  simpson: "Reversal within groups",
  leakage: "Target leakage",
  vif: "Collinearity among drivers",
  residual_normality: "Residuals near normal",
  heteroscedasticity: "Even spread of residuals",
  // ab_test
  sample_ratio: "Arm sizes against the planned split",
  power: "Power at this sample size",
  assignment: "Rows assigned to arms at random",
  // execution (PR-11)
  sampling: "Whole data read, not a sample",
  budget: "Within the run budget",
};

/** The gate's label, or its id spelled out when the backend sends one the map lacks. */
export function gateLabel(gate: string): string {
  return GATE_LABELS[gate] ?? humanise(gate);
}

export const STATUS_WORDS: Record<DataCheckStatus, string> = {
  passed: "Passed",
  qualified: "Qualified",
  failed: "Failed",
};

/** The verdict pill. "Verified" is never said: the report says what was checked, in the Evidence panel. */
export const VERDICT_LABELS: Record<InstantVerdict, string> = {
  pass: "Answered",
  qualified: "Answered with caveats",
  ask: "One question back",
  refuse: "Not answered",
};

export const COMPLETION_WORDS: Record<CompletionState, string> = {
  answered: "Answered",
  partial: "Partly answered",
  open: "Open",
};

/** "period_gaps" → "Period gaps": a fallback only, and digit-free as long as the id is. */
function humanise(id: string): string {
  const words = id.replace(/_+/g, " ").trim();
  return words ? words[0]!.toUpperCase() + words.slice(1) : "Check";
}
