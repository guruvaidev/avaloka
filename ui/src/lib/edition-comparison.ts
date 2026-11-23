// The Billing tab's plan comparison and plan cards, and the plan gates that must
// agree with them. Whether a plan includes a capability comes only from the
// generated matrix; this file holds the words for it and nothing else.
// scripts/gen_edition_matrix.py regenerates the matrix from app/core/editions.py.
import {
  EDITION_CAPABILITIES,
  EDITION_CAPABILITY_STATUS,
  EDITION_MATRIX,
  EDITION_PLANS,
  type CapabilityStatus,
  type EditionCapability,
  type EditionPlan,
} from "@/lib/edition-matrix.gen";

/** What a plan's cell says: included, entitled but not shipped yet, or not included. */
export type PlanCell = "yes" | "coming" | "no";

type Matrix = Record<EditionPlan, Record<string, boolean>>;
type Statuses = Record<string, CapabilityStatus>;

/** The one rule. A "coming" capability is never "yes", whatever the matrix grants. */
export function planCell(
  plan: EditionPlan,
  capability: string,
  matrix: Matrix = EDITION_MATRIX,
  statuses: Statuses = EDITION_CAPABILITY_STATUS,
): PlanCell {
  if (!matrix[plan][capability]) return "no";
  return statuses[capability] === "shipped" ? "yes" : "coming";
}

/** True only when the plan has the capability and it has shipped. Plan gates use this. */
export function planIncludes(plan: EditionPlan, capability: EditionCapability): boolean {
  return planCell(plan, capability) === "yes";
}

/**
 * The plan a gate should judge. `undefined` (the plan lookup has not answered)
 * is null, so a gate does not lock anyone out on a failed lookup. An
 * unrecognised plan name is Free, the least-privileged plan.
 */
export function normalizePlan(tier: string | null | undefined): EditionPlan | null {
  if (tier === undefined || tier === null) return null;
  return (EDITION_PLANS as readonly string[]).includes(tier) ? (tier as EditionPlan) : "free";
}

export const COMPARISON_SECTIONS = [
  { id: "analysis", title: "Analysis", group: "analysis" },
  { id: "execution", title: "Execution and scheduling", group: "analysis" },
  { id: "ml", title: "Machine learning", group: "analysis" },
  { id: "connectivity", title: "Connectivity", group: "connectivity" },
  { id: "collaboration", title: "Reports and collaboration", group: "collaboration" },
] as const;

export type ComparisonSectionId = (typeof COMPARISON_SECTIONS)[number]["id"];
export type ComparisonGroup = (typeof COMPARISON_SECTIONS)[number]["group"];

export const COMPARISON_TABS: { id: "all" | ComparisonGroup; label: string }[] = [
  { id: "all", label: "View all" },
  { id: "analysis", label: "Analysis" },
  { id: "connectivity", label: "Connectivity" },
  { id: "collaboration", label: "Collaboration" },
];

/** `card: true` also lists the capability on the card of the first plan that gets it. */
type ComparisonLabel = { section: ComparisonSectionId; label: string; card?: true } | { hiddenBecause: string };

/**
 * One entry per capability. A capability left out is a compile error here and a
 * thrown error in comparisonSections, so a new flag cannot silently miss the table.
 * The words state what a plan includes; they make no claim the flag does not.
 */
export const COMPARISON_LABELS: Record<EditionCapability, ComparisonLabel> = {
  data_analysis: { section: "analysis", label: "Conversational data analysis", card: true },
  export_generated_code: { section: "analysis", label: "Export the generated code", card: true },
  local_execution: { section: "analysis", label: "Local execution" },
  file_connectors: { section: "analysis", label: "File connectors", card: true },
  auto_insights: { section: "analysis", label: "Auto Insights", card: true },
  // 1.7 (docs/design/1.7-DATA-PLATFORM.md part C): the personal catalog is open
  // on every plan; the shared one is Enterprise (owner ruling 2026-11-01).
  dataset_catalog: { section: "analysis", label: "Personal dataset catalog with schema versions", card: true },
  ray_distributed: { section: "execution", label: "Ray distributed execution", card: true },
  // Shipped with 2.0 (editions.py CAPABILITY_STATUS); open on every plan, the
  // clone count is a limit, not a gate.
  swarm_intelligence: { section: "execution", label: "Swarm: parallel sub-analyses of one mission" },
  entire_dataset: { section: "execution", label: "Entire-dataset and large-scale runs" },
  scheduler: { section: "execution", label: "Scheduler", card: true },
  batch_jobs: { section: "execution", label: "Batch jobs" },
  ml_training: { section: "ml", label: "Model training", card: true },
  ml_inference: { section: "ml", label: "Model inference" },
  database_connectors: { section: "connectivity", label: "Database and SQL warehouse connectors", card: true },
  cloud_connectors: { section: "connectivity", label: "Cloud storage connectors", card: true },
  cloud_provisioning: { section: "connectivity", label: "Automated cloud provisioning" },
  scheduled_delivery: { section: "connectivity", label: "Scheduled delivery of results" },
  reports: { section: "collaboration", label: "Reports", card: true },
  team_collaboration: { section: "collaboration", label: "Teams and collaborator invites", card: true },
  org_structure: { section: "collaboration", label: "Departments, roles, branches and app users", card: true },
  share_analysis: { section: "collaboration", label: "Share analyses", card: true },
  shared_catalog: { section: "collaboration", label: "Shared dataset catalog with stewards and access requests", card: true },
  comment_on_analysis: { section: "collaboration", label: "Comment on analyses" },
  notifications: { section: "collaboration", label: "Notifications" },
  sso_audit: { section: "collaboration", label: "SSO and audit log", card: true },
};

export type ComparisonRow = {
  capability: EditionCapability;
  label: string;
  cells: Record<EditionPlan, PlanCell>;
};

export type ComparisonSection = {
  id: ComparisonSectionId;
  title: string;
  group: ComparisonGroup;
  rows: ComparisonRow[];
};

/** The table's sections. Throws on any capability with no entry in `labels`. */
export function comparisonSections(
  capabilities: readonly string[] = EDITION_CAPABILITIES,
  labels: Record<string, ComparisonLabel | undefined> = COMPARISON_LABELS,
  matrix: Matrix = EDITION_MATRIX,
  statuses: Statuses = EDITION_CAPABILITY_STATUS,
): ComparisonSection[] {
  const unlabelled = capabilities.filter((capability) => !labels[capability]);
  if (unlabelled.length > 0) {
    throw new Error(`No Billing comparison label for: ${unlabelled.join(", ")}`);
  }
  return COMPARISON_SECTIONS.map((section) => ({
    ...section,
    rows: capabilities.flatMap((capability) => {
      const entry = labels[capability];
      if (!entry || !("section" in entry) || entry.section !== section.id) return [];
      const cells = Object.fromEntries(
        EDITION_PLANS.map((plan) => [plan, planCell(plan, capability, matrix, statuses)]),
      ) as Record<EditionPlan, PlanCell>;
      return [{ capability: capability as EditionCapability, label: entry.label, cells }];
    }),
  }));
}

/**
 * A plan card's feature lines: the card capabilities this plan has that the plan
 * before it does not (the card's heading says "Everything in <previous plan>").
 * A capability that has not shipped says so.
 */
export function planCardFeatures(plan: EditionPlan): string[] {
  const previous = EDITION_PLANS[EDITION_PLANS.indexOf(plan) - 1];
  return EDITION_CAPABILITIES.flatMap((capability) => {
    const entry = COMPARISON_LABELS[capability];
    if (!("card" in entry) || !entry.card) return [];
    const cell = planCell(plan, capability);
    if (cell === "no" || (previous && planCell(previous, capability) !== "no")) return [];
    return [cell === "coming" ? `${entry.label} (coming soon)` : entry.label];
  });
}

/**
 * What the Free upgrade prompt says a paid plan adds, as phrases keyed by
 * capability. Each must be one Free lacks and Professional includes; a unit
 * test holds the list to that, so the prompt claims nothing the matrix does not.
 */
export const UPGRADE_PROMPT_PHRASES: Partial<Record<EditionCapability, string>> = {
  team_collaboration: "invite people and form teams",
  comment_on_analysis: "comment on analyses",
  share_analysis: "share analyses",
  scheduler: "schedule analyses",
  reports: "use Reports",
};

/** "Upgrade to a, b and c." built from UPGRADE_PROMPT_PHRASES. */
export function upgradePromptText(): string {
  const phrases = Object.values(UPGRADE_PROMPT_PHRASES);
  const list = phrases.length > 1 ? `${phrases.slice(0, -1).join(", ")} and ${phrases[phrases.length - 1]}` : phrases[0];
  return `Upgrade to ${list}.`;
}
