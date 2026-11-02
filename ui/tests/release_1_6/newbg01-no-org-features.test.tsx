/**
 * NEW-BG-01 - plan rules for organisation and team features.
 *
 * Owner decision 2026-10-31 redefined the bug (the billing organisation created
 * at signup stays); the owner revised the rules on 2026-11-01:
 *  - Free: can share (email results, public link); cannot form teams or invite
 *    collaborators.
 *  - Professional: can invite collaborators (including Free users, the referral
 *    path) and can form teams whose members are Professional or higher.
 *  - Enterprise: everything. Departments, roles, branches and app users
 *    (org entities) stay Enterprise-only.
 *  - While the plan is loading, nothing org- or team-related is granted.
 *
 * Capabilities come from usePlanFeatures (ui/src/lib/use-plan-features.ts); the
 * Configurations route (org entities) is wrapped in PlanGate
 * (ui/src/components/dashboard/PlanGate.tsx). The plan-rules follow-up (hotfix/1.6-plan-rules-pro-collab) made
 * the formerly open halves pass; they are plain tests now.
 */
import { render, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const plan = vi.hoisted(() => ({ value: "free" as string, loading: false }));
const navigate = vi.hoisted(() => vi.fn());

vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () =>
    plan.loading ? { data: undefined, isLoading: true } : { data: { plan: plan.value }, isLoading: false },
}));
vi.mock("@tanstack/react-router", () => ({ useNavigate: () => navigate }));

import { PlanGate } from "@/components/dashboard/PlanGate";
import { usePlanFeatures } from "@/lib/use-plan-features";

function capabilities(tier: string, loading = false) {
  plan.value = tier;
  plan.loading = loading;
  const { result } = renderHook(() => usePlanFeatures());
  const f = result.current;
  return {
    canShare: f.canShare,
    canUseTeams: f.canUseTeams,
    canInviteCollaborators: f.canInviteCollaborators,
    canManageOrgEntities: f.canManageOrgEntities,
  };
}

describe("NEW-BG-01 plan rules for organisation and team features", () => {
  beforeEach(() => {
    navigate.mockReset();
    plan.loading = false;
  });

  it("NEW-BG-01: Enterprise keeps sharing, teams, collaborator invites and org entities (control)", () => {
    expect(capabilities("enterprise")).toEqual({
      canShare: true,
      canUseTeams: true,
      canInviteCollaborators: true,
      canManageOrgEntities: true,
    });
  });

  it("NEW-BG-01: Free and Professional keep sharing (email results and public link)", () => {
    expect(capabilities("free").canShare).toBe(true);
    expect(capabilities("professional").canShare).toBe(true);
  });

  // NEW-BG-01 (owner revision 2026-11-01), fixed by hotfix/1.6-plan-rules-pro-collab.
  it("NEW-BG-01: Free gets no teams, collaborator invites or org entities", () => {
    expect(capabilities("free")).toEqual({
      canShare: true,
      canUseTeams: false,
      canInviteCollaborators: false,
      canManageOrgEntities: false,
    });
  });

  // NEW-BG-01 (owner revision 2026-11-01), fixed by hotfix/1.6-plan-rules-pro-collab: Professional invites
  // collaborators and forms teams; org entities stay Enterprise-only.
  it("NEW-BG-01: Professional gets teams and collaborator invites but no org entities", () => {
    expect(capabilities("professional")).toEqual({
      canShare: true,
      canUseTeams: true,
      canInviteCollaborators: true,
      canManageOrgEntities: false,
    });
  });

  // NEW-BG-01, fixed by hotfix/1.6-plan-rules-pro-collab: nothing is granted until the plan has loaded.
  it("NEW-BG-01: nothing org- or team-related is granted while the plan is loading", () => {
    expect(capabilities("enterprise", true)).toMatchObject({
      canUseTeams: false,
      canInviteCollaborators: false,
      canManageOrgEntities: false,
    });
  });

  it("NEW-BG-01: the org-entity Configurations route is closed to Free users (PlanGate renders nothing and redirects)", () => {
    plan.value = "free";
    const { container } = render(
      <PlanGate>
        <div>roles, branches, divisions, users, teams</div>
      </PlanGate>,
    );
    expect(container.textContent).toBe("");
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ to: "/settings" }));
  });
});
