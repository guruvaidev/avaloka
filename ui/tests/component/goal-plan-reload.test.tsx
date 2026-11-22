import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { useGoalFirstGate, type GoalChatLine } from "../../src/components/analysis/GoalPlanActions";
import * as goalPlan from "../../src/lib/goal-plan";

const SALES = { id: "d1", name: "sales.csv" };

// One page view of the analysis chat: unmounting and mounting again with the
// same thread id is what a page reload does to this hook (sessionStorage
// survives the reload; React state does not).
function pageView(threadId: string | null, isNewChat = true) {
  const runTurn = vi.fn(async (_content: string, _opts?: { showUser?: boolean }) => true);
  const lines: GoalChatLine[] = [];
  const hook = renderHook(
    ({ key }) =>
      useGoalFirstGate({
        runTurn,
        appendLines: (l) => void lines.push(...l),
        datasets: [SALES],
        isNewChat,
        storageKey: key,
      } as Parameters<typeof useGoalFirstGate>[0]),
    { initialProps: { key: threadId } },
  );
  return { runTurn, lines, hook };
}

afterEach(() => sessionStorage.clear());

describe("the goal-first plan survives a page reload", () => {
  it("after a reload the Go action is restored and approves the original goal", async () => {
    const first = pageView("thread-1");
    await act(async () => {
      await first.hook.result.current.send("Average subscribers by category");
    });
    expect(first.hook.result.current.pending?.goal).toBe("Average subscribers by category");
    first.hook.unmount();

    const reloaded = pageView("thread-1");
    // The Go button renders while pending is set (analysis.tsx).
    expect(reloaded.hook.result.current.pending?.goal).toBe("Average subscribers by category");
    await act(async () => {
      await reloaded.hook.result.current.approve();
    });
    expect(reloaded.runTurn).toHaveBeenCalledWith("Average subscribers by category", {
      showUser: false,
    });
  });

  it('after a reload a typed "go" confirms the pending plan instead of planning the goal "go"', async () => {
    const first = pageView("thread-1");
    await act(async () => {
      await first.hook.result.current.send("Average subscribers by category");
    });
    first.hook.unmount();

    const reloaded = pageView("thread-1");
    await act(async () => {
      await reloaded.hook.result.current.send("go");
    });
    expect(reloaded.runTurn).toHaveBeenCalledTimes(1);
    expect(reloaded.runTurn).toHaveBeenCalledWith("Average subscribers by category", {
      showUser: false,
    });
    expect(reloaded.lines.some((l) => l.content.includes('plan for: "go"'))).toBe(false);
    expect(reloaded.hook.result.current.pending).toBeNull();
  });

  it("the thread id arriving after mount keeps a plan made before it, and restores one stored for it", async () => {
    // Plan made while the thread id is still unknown, then the id arrives.
    const view = pageView(null);
    await act(async () => {
      await view.hook.result.current.send("Revenue by region");
    });
    view.hook.rerender({ key: "thread-2" });
    expect(view.hook.result.current.pending?.goal).toBe("Revenue by region");
    view.hook.unmount();
    // Reload: mounts with no id yet, then the id is read from the session.
    const reloaded = pageView(null);
    expect(reloaded.hook.result.current.pending).toBeNull();
    reloaded.hook.rerender({ key: "thread-2" });
    expect(reloaded.hook.result.current.pending?.goal).toBe("Revenue by region");
  });

  it("switching to another thread shows that thread's plan, not this one's", async () => {
    const view = pageView("thread-a");
    await act(async () => {
      await view.hook.result.current.send("Revenue by region");
    });
    view.hook.rerender({ key: "thread-b" });
    expect(view.hook.result.current.pending).toBeNull();
    view.hook.rerender({ key: "thread-a" });
    expect(view.hook.result.current.pending?.goal).toBe("Revenue by region");
  });

  it("approving clears the stored plan, so a later reload has nothing to restore", async () => {
    const first = pageView("thread-1");
    await act(async () => {
      await first.hook.result.current.send("Revenue by region");
    });
    await act(async () => {
      await first.hook.result.current.send("go");
    });
    first.hook.unmount();
    expect(pageView("thread-1").hook.result.current.pending).toBeNull();
  });
});

describe('a bare "go" with no plan waiting', () => {
  it("runs nothing and is not planned as a goal; it says there is nothing to approve", async () => {
    const { runTurn, lines, hook } = pageView("thread-1");
    await act(async () => {
      await hook.result.current.send("go");
    });
    expect(runTurn).not.toHaveBeenCalled();
    expect(hook.result.current.pending).toBeNull();
    expect(lines.map((l) => l.role)).toEqual(["user", "ai"]);
    expect(lines[1].content).toMatch(/no plan waiting/i);
    expect(lines[1].content).not.toContain('plan for: "go"');
  });

  it("a real goal still gets a plan", async () => {
    const { runTurn, hook } = pageView("thread-1");
    await act(async () => {
      await hook.result.current.send("go by region and sum revenue");
    });
    expect(runTurn).not.toHaveBeenCalled();
    expect(hook.result.current.pending?.goal).toBe("go by region and sum revenue");
  });

  it("after the first turn a bare yes still reaches the backend (e.g. a training-plan confirm)", async () => {
    const { runTurn, hook } = pageView("thread-1", false);
    await act(async () => {
      await hook.result.current.send("yes");
    });
    expect(runTurn).toHaveBeenCalledWith("yes");
  });
});

// #687: an explicit provisioning request gets a backend confirmation prompt
// ("Reply yes to go ahead") held in pending_clarification. The goal-first gate
// must hand that reply to the backend, never answer it locally.
describe("a backend confirmation after the goal was approved", () => {
  it("the user's yes reaches the backend in the same page view", async () => {
    const { runTurn, hook } = pageView("thread-1");
    await act(async () => {
      await hook.result.current.send("deploy this to GCP");
    });
    await act(async () => {
      await hook.result.current.approve();
    });
    expect(runTurn).toHaveBeenLastCalledWith("deploy this to GCP", { showUser: false });
    // The goal line is on screen now, so the page no longer counts the chat as new.
    const later = pageView("thread-1", false);
    await act(async () => {
      await later.hook.result.current.send("yes");
    });
    expect(later.runTurn).toHaveBeenCalledWith("yes");
  });

  it("after a reload mid-confirmation, yes still reaches the backend (nothing is restored)", async () => {
    const first = pageView("thread-1");
    await act(async () => {
      await first.hook.result.current.send("deploy this to GCP");
    });
    await act(async () => {
      await first.hook.result.current.approve();
    });
    first.hook.unmount();
    // Reload: the approved goal is in the persisted chat, so the chat is not new.
    const reloaded = pageView("thread-1", false);
    expect(reloaded.hook.result.current.pending).toBeNull();
    await act(async () => {
      await reloaded.hook.result.current.send("yes");
    });
    expect(reloaded.runTurn).toHaveBeenCalledWith("yes");
    expect(reloaded.lines).toEqual([]);
  });
});

describe("restored plan lines and route wiring", () => {
  it("puts the goal and plan back only when the hydrated chat lacks them", () => {
    const restore = (goalPlan as Record<string, unknown>).planLinesToRestore as
      | typeof goalPlan.planLinesToRestore
      | undefined;
    expect(typeof restore).toBe("function");
    const plan = goalPlan.buildGoalPlan("Revenue by region", [SALES]);
    const text = goalPlan.planMessage(plan);
    expect(restore!(plan, [])).toEqual([
      { role: "user", content: "Revenue by region" },
      { role: "ai", content: text },
    ]);
    expect(restore!(plan, [{ role: "ai", content: text }])).toEqual([]);
    expect(restore!(null, [])).toEqual([]);
  });

  it("the analysis page keys the gate by thread id instead of dropping the plan on thread change", () => {
    const page = readFileSync(resolve(__dirname, "../../src/routes/analysis.tsx"), "utf8");
    expect(page).toMatch(/useGoalFirstGate\(\{[\s\S]{0,400}storageKey: threadId/);
    expect(page).not.toMatch(/useEffect\(\(\) => resetGoalGate\(\), \[threadId/);
    expect(page).toMatch(/planLinesToRestore\(goalGate\.pending, messages\)/);
  });
});
