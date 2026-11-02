import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import {
  GoalPlanActions,
  useGoalFirstGate,
  type GoalChatLine,
} from "../../src/components/analysis/GoalPlanActions";
import {
  buildGoalPlan,
  classifyGoal,
  decideOutgoing,
  isGoAhead,
  planMessage,
} from "../../src/lib/goal-plan";

const SALES = { id: "d1", name: "sales.csv" };
const STORES = { id: "d2", name: "stores.csv" };

// runTurn stands in for the backend turn (and so for the model): the gate must
// not call it before the user's go.
function setup(opts: { isNewChat?: boolean; datasets?: { id: string; name: string }[] } = {}) {
  const runTurn = vi.fn(async (_content: string, _opts?: { showUser?: boolean }) => true);
  const lines: GoalChatLine[] = [];
  const appendLines = (l: GoalChatLine[]) => void lines.push(...l);
  const hook = renderHook(
    ({ isNewChat }) =>
      useGoalFirstGate({
        runTurn,
        appendLines,
        datasets: opts.datasets ?? [SALES, STORES],
        isNewChat,
      }),
    { initialProps: { isNewChat: opts.isNewChat ?? true } },
  );
  return { runTurn, lines, hook };
}

describe("goal -> plan -> confirm", () => {
  it("the first goal gets a plan naming the datasets and steps, and nothing runs", async () => {
    const { runTurn, lines, hook } = setup();
    await act(async () => {
      await hook.result.current.send("Which stores grew revenue fastest?");
    });
    expect(runTurn).not.toHaveBeenCalled();
    expect(hook.result.current.pending?.goal).toBe("Which stores grew revenue fastest?");
    expect(lines.map((l) => l.role)).toEqual(["user", "ai"]);
    const plan = lines[1].content;
    expect(plan).toContain("**Datasets:** sales.csv, stores.csv");
    for (const step of ["profile", "missing values", "join", "Analyse", "sample of your data"]) {
      expect(plan).toContain(step);
    }
    expect(plan).toContain("Nothing runs until you say go");
  });

  it("a reply that is not a go replaces the goal and still runs nothing", async () => {
    const { runTurn, lines, hook } = setup();
    await act(async () => {
      await hook.result.current.send("Revenue by region");
    });
    await act(async () => {
      await hook.result.current.send("Actually, train a model to predict churn");
    });
    expect(runTurn).not.toHaveBeenCalled();
    expect(hook.result.current.pending?.kind).toBe("train");
    expect(lines.filter((l) => l.role === "ai")).toHaveLength(2);
  });

  it("typing go runs the original goal once, without re-posting it", async () => {
    const { runTurn, lines, hook } = setup();
    await act(async () => {
      await hook.result.current.send("Revenue by region");
    });
    await act(async () => {
      await hook.result.current.send("go");
    });
    expect(runTurn).toHaveBeenCalledTimes(1);
    expect(runTurn).toHaveBeenCalledWith("Revenue by region", { showUser: false });
    expect(lines.at(-1)).toEqual({ role: "user", content: "go" });
    expect(hook.result.current.pending).toBeNull();
  });

  it("the Go button approves the pending plan", async () => {
    const { runTurn, hook } = setup();
    await act(async () => {
      await hook.result.current.send("Revenue by region");
    });
    await act(async () => {
      await hook.result.current.approve();
    });
    expect(runTurn).toHaveBeenCalledWith("Revenue by region", { showUser: false });
  });

  it("messages after the first turn go straight through", async () => {
    const { runTurn, hook } = setup({ isNewChat: false });
    await act(async () => {
      await hook.result.current.send("now split it by month");
    });
    expect(runTurn).toHaveBeenCalledWith("now split it by month");
  });

  it("with no dataset attached the turn runs, so the existing upload prompt shows", async () => {
    const { runTurn, hook } = setup({ datasets: [] });
    await act(async () => {
      await hook.result.current.send("hello");
    });
    expect(runTurn).toHaveBeenCalledWith("hello");
  });

  it("GoalPlanActions calls onGo", () => {
    const onGo = vi.fn();
    render(<GoalPlanActions onGo={onGo} />);
    fireEvent.click(screen.getByRole("button", { name: "Go" }));
    expect(onGo).toHaveBeenCalledTimes(1);
  });
});

describe("goal plan helpers", () => {
  it("classifies goals", () => {
    expect(classifyGoal("forecast next quarter sales")).toBe("forecast");
    expect(classifyGoal("train a model to predict churn")).toBe("train");
    expect(classifyGoal("average order value by region")).toBe("analyse");
  });

  it("joins only when there is more than one dataset", () => {
    expect(buildGoalPlan("x", [SALES]).steps.map((s) => s.key)).toEqual([
      "profile",
      "clean",
      "analyse",
      "verify",
    ]);
    expect(buildGoalPlan("x", [SALES, STORES]).steps.map((s) => s.key)).toEqual([
      "profile",
      "clean",
      "join",
      "analyse",
      "verify",
    ]);
  });

  it("never promises a forecast horizon 1.6 cannot produce", () => {
    const text = planMessage(buildGoalPlan("forecast next year", [SALES]));
    expect(text).not.toMatch(/project it forward|horizon|future/i);
  });

  it("recognises go replies only", () => {
    for (const t of ["go", "Go ahead", "yes", "OK!", "proceed"]) expect(isGoAhead(t)).toBe(true);
    for (const t of ["go by region", "no", "yes but by month", ""])
      expect(isGoAhead(t)).toBe(false);
  });

  it("a changed goal is planned against the datasets attached now", () => {
    const pending = buildGoalPlan("a", [SALES]);
    const d = decideOutgoing("b", { goalFirst: true, pending, datasets: [SALES, STORES] });
    expect(d.action === "plan" && d.plan.datasets).toEqual([SALES, STORES]);
  });
});
