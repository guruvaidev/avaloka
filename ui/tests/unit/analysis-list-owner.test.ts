import { describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/supabase/client", () => ({
  supabase: { auth: { onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }) } },
}));

import { analysesKey, fetchOwnedStandaloneAnalyses, standaloneAnalysesKey } from "../../src/lib/analyses";
import { assistantOpenOnView } from "../../src/lib/insights-assistant-visibility";

function fakeClient(rows: any[]) {
  const calls: [string, ...unknown[]][] = [];
  const q: any = {
    select: (...a: unknown[]) => (calls.push(["select", ...a]), q),
    eq: (...a: unknown[]) => (calls.push(["eq", ...a]), q),
    is: (...a: unknown[]) => (calls.push(["is", ...a]), q),
    order: () => Promise.resolve({ data: rows, error: null }),
  };
  return { client: { from: (t: string) => (calls.push(["from", t]), q) } as any, calls };
}

describe("Analysis section list", () => {
  it("filters the query by the current user's owner id", async () => {
    const { client, calls } = fakeClient([]);
    await fetchOwnedStandaloneAnalyses(client, "profile-b");
    expect(calls).toContainEqual(["eq", "owner_id", "profile-b"]);
  });

  it("excludes a project-shared analysis owned by someone else", async () => {
    const { client } = fakeClient([
      { id: "1", owner_id: "profile-a", name: "Finance/Equities Trading" },
      { id: "2", owner_id: "profile-b", name: "Mine" },
    ]);
    const rows = await fetchOwnedStandaloneAnalyses(client, "profile-b");
    expect(rows.map((r) => r.id)).toEqual(["2"]);
  });

  it("returns nothing without a signed-in profile", async () => {
    const { client, calls } = fakeClient([{ id: "1", owner_id: "profile-a" }]);
    expect(await fetchOwnedStandaloneAnalyses(client, null)).toEqual([]);
    expect(calls).toEqual([]);
  });

  it("keys the list cache by user id", () => {
    expect(standaloneAnalysesKey("user-a")).toContain("user-a");
    expect(standaloneAnalysesKey("user-a")).not.toEqual(standaloneAnalysesKey("user-b"));
  });

  it("leaves the Projects list key unchanged", () => {
    expect(analysesKey("p1")).toEqual(["analyses", "p1"]);
  });
});

describe("Insights Assistant default visibility", () => {
  it("opens on Stack, closed on Overview and Column statistics, closed on Stack after dismissal", () => {
    expect(assistantOpenOnView("stack", false)).toBe(true);
    expect(assistantOpenOnView("grid", false)).toBe(false);
    expect(assistantOpenOnView("columns", false)).toBe(false);
    expect(assistantOpenOnView("stack", true)).toBe(false);
  });
});
