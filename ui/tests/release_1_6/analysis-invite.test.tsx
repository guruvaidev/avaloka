// @vitest-environment happy-dom
// UI-B06 (UI Bugs) recorded that a standalone analysis (one NOT inside a
// project) is shared by link or email only. That rule has since been replaced:
//  - 2f3a4645 added the organisation member invite ("Team member") to the
//    standalone analysis share panel (Enterprise; resource_collaborators rows,
//    which analysis_share_access_level honours);
//  - 4140dd96 added Team collaboration for Professional standalone analyses
//    (analysis_shares; 20261106170000_standalone_analysis_team_sharing).
// The link and email paths stay on every plan.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { screen, waitFor } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// One stable server-fn double: a new function per render would re-fire the
// popover's load effect on every render.
const SETTINGS = { viewerProfileId: "p-me", isOwner: true, teams: [{ id: "team-1", name: "Analytics Squad" }], members: [], shares: [] };
const serverFn = vi.hoisted(() =>
  vi.fn(async (args?: { data?: { resourceType?: string } }) => (args?.data?.resourceType ? SETTINGS : [])),
);
const tier = vi.hoisted(() => ({ value: "enterprise" as "enterprise" | "professional" }));
vi.mock("@tanstack/react-start", () => ({ useServerFn: () => serverFn }));
// The Professional sharing panel (4140dd96) imports these server functions;
// their module builds auth middleware at load time, which this harness does not
// provide. The panel calls them only through useServerFn, mocked above.
vi.mock("@/lib/analysis-sharing.functions", () => ({
  getAnalysisShareSettings: vi.fn(),
  createAnalysisShare: vi.fn(),
  updateAnalysisShare: vi.fn(),
  removeAnalysisShare: vi.fn(),
}));
vi.mock("@/lib/configurations.functions", () => ({ listAppUsers: vi.fn() }));
vi.mock("@/lib/use-plan-features", () => ({
  ANALYSIS_LINKS_NEED_PRO: "plan note",
  COMMENTS_NEED_PRO: "plan note",
  usePlanFeatures: () => ({
    canShare: true,
    canInviteCollaborators: true,
    canUseTeams: true,
    canShareAnalysisLink: true,
    loading: false,
    isEnterprise: tier.value === "enterprise",
    isProfessional: tier.value === "professional",
  }),
}));
vi.mock("@/lib/auth-user", () => ({ getCachedAuthUser: vi.fn(async () => ({ id: "user-1" })) }));
vi.mock("@/integrations/supabase/client", async () =>
  (await import("./support/w4a-supabase")).supabaseClientMock({
    organization_teams: [{ id: "team-1", name: "Analytics Squad" }],
    app_users: [{ id: "user-2", email: "member@example.com", team_id: "team-1" }],
  }),
);
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }));

import { ShareCollaboratorsButton } from "@/components/dashboard/ShareCollaboratorsButton";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

async function openStandalone() {
  await act(async () => root.render(<ShareCollaboratorsButton analysisId="analysis-1" projectId={null} open onOpenChange={() => undefined} />));
  await waitFor(() => expect(document.querySelector(".invite-collaborators-dropdown")).not.toBeNull());
  await act(async () => new Promise((r) => setTimeout(r, 20)));
}

describe("UI-B06 share outside a project", () => {
  it("UI-B06 (revised): Enterprise invites organisation members to a standalone analysis, with link and email", async () => {
    tier.value = "enterprise";
    await openStandalone();
    expect(screen.queryByText("Team member")).not.toBeNull();
    expect(document.querySelector('section[aria-label="Standalone analysis access"]')).toBeNull();
    expect(screen.queryByText("Share analysis")).not.toBeNull();
    expect(screen.queryByText("Share link")).not.toBeNull();
    expect(screen.queryByText("Email Analysis")).not.toBeNull();
  });

  it("UI-B06 (revised): Professional shares a standalone analysis through Team collaboration, with link and email", async () => {
    tier.value = "professional";
    await openStandalone();
    expect(screen.queryByText("Team member")).toBeNull();
    expect(document.querySelector('section[aria-label="Standalone analysis access"]')).not.toBeNull();
    await waitFor(() => expect(screen.queryByText("Analytics Squad")).not.toBeNull());
    expect(serverFn).toHaveBeenCalledWith({ data: { analysisId: "analysis-1", resourceType: "analysis" } });
    expect(screen.queryByText("Share analysis")).not.toBeNull();
    expect(screen.queryByText("Share link")).not.toBeNull();
    expect(screen.queryByText("Email Analysis")).not.toBeNull();
  });
});
