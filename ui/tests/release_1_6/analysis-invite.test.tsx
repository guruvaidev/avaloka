// @vitest-environment happy-dom
// UI-B06 (UI Bugs): sharing an analysis that is NOT inside a project shows no
// Team / Team member pickers (collaboration on a standalone analysis is by
// link or email only). The org has a team, so a team picker, if rendered,
// would list it.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { screen, waitFor } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// One stable server-fn double: a new function per render would re-fire the
// popover's load effect on every render.
const serverFn = vi.hoisted(() => vi.fn(async () => []));
vi.mock("@tanstack/react-start", () => ({ useServerFn: () => serverFn }));
vi.mock("@/lib/configurations.functions", () => ({ listAppUsers: vi.fn() }));
vi.mock("@/lib/use-plan-features", () => ({
  usePlanFeatures: () => ({ canShare: true, canInviteCollaborators: true, canUseTeams: true, loading: false }),
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

describe("UI-B06 share outside a project", () => {
  it("UI-B06: the share panel of a standalone analysis has no Team or Team member selectors", async () => {
    await act(async () => root.render(<ShareCollaboratorsButton analysisId="analysis-1" projectId={null} open onOpenChange={() => undefined} />));
    await waitFor(() => expect(document.querySelector(".invite-collaborators-dropdown")).not.toBeNull());
    // Let the collaborator/team load settle so a team list would have rendered.
    await act(async () => new Promise((r) => setTimeout(r, 20)));
    expect(screen.queryByText("Team member")).toBeNull();
    expect(screen.queryByText("Team")).toBeNull();
    expect(screen.queryByPlaceholderText("Search teams…")).toBeNull();
    expect(screen.queryByText("Analytics Squad")).toBeNull();
    // What a standalone analysis does offer: a link and an email send.
    expect(screen.queryByText("Share analysis")).not.toBeNull();
    expect(screen.queryByText("Share link")).not.toBeNull();
    expect(screen.queryByText("Email Analysis")).not.toBeNull();
  });
});
