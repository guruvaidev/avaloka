// NEW-BG-01, owner ruling 2026-11-01 (revised; supersedes the 2026-10-31
// Free/Pro rule): collaboration is a paid offering, and Professional gets
// referrals.
//   Free          shares (email results + public link); sees Teams in
//                 Configurations but cannot create a team; no invites.
//   Professional  invites collaborators (any plan, including Free); creates teams.
//   Enterprise    everything, including departments, roles, branches and users.
// Every paid flag fails closed while the plan loads or when there is no plan.
// Since the Professional teams change (4140dd96), canUseTeams means "Teams is
// visible in Configurations" on every plan; creating a team and inviting
// members is canCreateTeams.
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

const FLAGS = ["canShare", "canUseTeams", "canCreateTeams", "canInviteCollaborators", "canManageOrgEntities"] as const;
const pick = (f: ReturnType<typeof usePlanFeatures>) => Object.fromEntries(FLAGS.map((k) => [k, f[k]]));

describe("usePlanFeatures: the revised plan matrix", () => {
  it("Free shares and sees Teams, but creates no team, sends no invites and has no org structure", () => {
    expect(pick(features("free"))).toEqual({
      canShare: true,
      canUseTeams: true,
      canCreateTeams: false,
      canInviteCollaborators: false,
      canManageOrgEntities: false,
    });
  });

  it("Professional invites collaborators and forms teams, but has no org structure", () => {
    const f = features("professional");
    expect(pick(f)).toEqual({
      canShare: true,
      canUseTeams: true,
      canCreateTeams: true,
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
      canCreateTeams: true,
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
    expect([f.canCreateTeams, f.canInviteCollaborators, f.canManageOrgEntities, f.canUseAllConfigSections]).toEqual([false, false, false, false]);
    expect(f.canUseTeams).toBe(true); // viewing Teams is not a paid feature
  });
});
