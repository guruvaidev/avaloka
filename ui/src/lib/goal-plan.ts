// Goal first: the first message of a new chat is taken as the user's goal, and
// the chat answers with a short plan before anything runs. Nothing is sent to
// the backend until the user says go; the backend planner (plan_etl) then plans
// the real steps for that goal, exactly as it does for any other turn.
//
// The plan is code-written from the goal text and the attached datasets, so
// every step it names must be one the 1.6 pipeline really performs. It is a
// preview: the goal goes to the backend unchanged, and plan_etl decides
// the exact steps (docs/design/1.6-CHATBOT-INTRO.md, "Plan steps").

export type GoalKind = "analyse" | "train" | "forecast";

export type PlanStepKey = "profile" | "clean" | "join" | "analyse" | "train" | "verify";

export type PlanDataset = { id: string; name: string };

export type GoalPlan = {
  goal: string;
  kind: GoalKind;
  datasets: PlanDataset[];
  steps: { key: PlanStepKey; label: string }[];
};

const TRAIN_RE = /\b(train|training|model|classif\w*|predict\w*|regress\w*|fine-?tune)\b/i;
const FORECAST_RE =
  /\b(forecast\w*|projection|project(ed)?\s+\w+|next\s+(month|quarter|year|week)s?|future|trend\w*)\b/i;

export function classifyGoal(goal: string): GoalKind {
  if (FORECAST_RE.test(goal)) return "forecast";
  if (TRAIN_RE.test(goal)) return "train";
  return "analyse";
}

function listNames(datasets: PlanDataset[]): string {
  const names = datasets.map((d) => d.name);
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function buildGoalPlan(goal: string, datasets: PlanDataset[]): GoalPlan {
  const kind = classifyGoal(goal);
  const names = listNames(datasets);
  const steps: GoalPlan["steps"] = [
    {
      key: "profile",
      label: `Start from the profile of ${names}: column types, missing values and summary statistics.`,
    },
    {
      key: "clean",
      label: "Handle the missing values and mixed types the profile shows, in the code I write.",
    },
  ];
  if (datasets.length > 1) {
    steps.push({ key: "join", label: `Find the columns ${names} share and join them on those.` });
  }
  if (kind === "train") {
    steps.push({
      key: "train",
      label:
        "Train a classification or regression model; I'll show you a training plan to confirm first.",
    });
  } else if (kind === "forecast") {
    steps.push({ key: "analyse", label: "Analyse the trend over time with code I write and run." });
  } else {
    steps.push({
      key: "analyse",
      label: "Analyse the data for your goal with code I write and run.",
    });
  }
  steps.push({
    key: "verify",
    label: "Test that code on a sample of your data and check its output before the full run.",
  });
  return { goal, kind, datasets, steps };
}

/** Markdown for the plan message: datasets and steps, then the go prompt. */
export function planMessage(plan: GoalPlan): string {
  const lines = [
    `Here's my plan for: "${plan.goal}"`,
    "",
    `**Datasets:** ${plan.datasets.map((d) => d.name).join(", ")}`,
    "",
    ...plan.steps.map((step, i) => `${i + 1}. ${step.label}`),
    "",
    "Nothing runs until you say go. Reply **go** to start, or tell me a different goal.",
  ];
  return lines.join("\n");
}

const GO_RE =
  /^\s*(go|go ahead|yes|y|yep|yeah|ok|okay|proceed|start|run it|do it|sure)\s*[.!]*\s*$/i;

/** A reply that approves the pending plan. Anything else is a new goal. */
export function isGoAhead(text: string): boolean {
  return GO_RE.test(text);
}

export type GoalGateDecision =
  { action: "run"; content: string } | { action: "plan"; plan: GoalPlan };

/**
 * What to do with an outgoing message.
 * - goalFirst: the chat is new (no turn has run yet in this thread).
 * - pending: the plan waiting for the user's go, if any. A reply that is not
 *   a go replaces the goal and gets a fresh plan; it never runs.
 */
export function decideOutgoing(
  text: string,
  opts: { goalFirst: boolean; pending: GoalPlan | null; datasets: PlanDataset[] },
): GoalGateDecision {
  if (opts.pending) {
    if (isGoAhead(text)) return { action: "run", content: opts.pending.goal };
    return {
      action: "plan",
      plan: buildGoalPlan(text, opts.datasets.length ? opts.datasets : opts.pending.datasets),
    };
  }
  // No datasets yet: the existing turn path asks for an upload.
  if (!opts.goalFirst || !opts.datasets.length) return { action: "run", content: text };
  return { action: "plan", plan: buildGoalPlan(text, opts.datasets) };
}
