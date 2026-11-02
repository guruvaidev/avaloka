// NEW-BG-01, owner ruling 2026-11-01 (revised; supersedes the 2026-10-31
// Free/Pro rule): collaboration is a paid offering, and Professional gets
// referrals.
//   Free          shares (email results + public link); no teams; no invites.
//   Professional  invites collaborators (any plan, including Free); forms teams.
//   Enterprise    everything, including departments, roles, branches and users.
// Every flag fails closed while the plan loads or when there is no plan.
import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const plan = vi.hoisted(() => ({ value: "free" as string | undefined, loading: false }));

vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({
    data: plan.value === undefined ? undefined : { plan: plan.value },
    isLoading: plan.loading,
  }),
}));

import { usePlanFeatures } from "@/lib/use-plan-features";

function features(tier: string | undefined, loading = false) {
  plan.value = tier;
  plan.loading = loading;
  return renderHook(() => usePlanFeatures()).result.current;
}

const FLAGS = ["canShare", "canUseTeams", "canInviteCollaborators", "canManageOrgEntities"] as const;
const pick = (f: ReturnType<typeof usePlanFeatures>) => Object.fromEntries(FLAGS.map((k) => [k, f[k]]));

describe("usePlanFeatures: the revised plan matrix", () => {
  it("Free shares, but gets no teams, no invites and no org structure", () => {
    expect(pick(features("free"))).toEqual({
      canShare: true,
      canUseTeams: false,
      canInviteCollaborators: false,
      canManageOrgEntities: false,
    });
  });

  it("Professional invites collaborators and forms teams, but has no org structure", () => {
    const f = features("professional");
    expect(pick(f)).toEqual({
      canShare: true,
      canUseTeams: true,
      canInviteCollaborators: true,
      canManageOrgEntities: false,
    });
    expect(f.canUseAllConfigSections).toBe(false); // Models and Teams only
    expect(f.canUseProjects).toBe(true);
  });

  it("Enterprise gets everything", () => {
    const f = features("enterprise");
    expect(pick(f)).toEqual({
      canShare: true,
      canUseTeams: true,
      canInviteCollaborators: true,
      canManageOrgEntities: true,
    });
    expect(f.canUseAllConfigSections).toBe(true);
  });

  it.each([
    ["loading", undefined, true],
    ["loading with a cached Enterprise plan", "enterprise", true],
    ["no plan data", undefined, false],
    ["an unknown plan", "platinum", false],
  ])("fails closed: %s", (_label, tier, loading) => {
    const f = features(tier as string | undefined, loading as boolean);
    expect([f.canUseTeams, f.canInviteCollaborators, f.canManageOrgEntities]).toEqual([false, false, false]);
  });
});
