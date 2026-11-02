// @vitest-environment happy-dom
// NEW-BG-01, owner ruling 2026-11-01: Free shares (the share link stays) but
// gets no collaborator invites; Professional and Enterprise invite any user,
// including Free ones (the referral path) -- the candidate list is not
// filtered by the invitee's plan.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({ tier: "free" as string }));

vi.mock("@tanstack/react-start", () => ({ useServerFn: (fn: unknown) => fn }));
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: m.tier }, isLoading: false }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock("@/lib/configurations.functions", () => ({
  listAppUsers: async () => [
    { id: "u-free", name: "Fran Free", email: "fran@example.com", role: "Analyst", status: "Active" },
    { id: "u-pro", name: "Pat Pro", email: "pat@example.com", role: "Analyst", status: "Active" },
  ],
}));
vi.mock("@/integrations/supabase/client", () => {
  const chain: any = new Proxy({}, {
    get: (_t, p) => (p === "then" ? (res: (v: unknown) => void) => res({ data: [], error: null }) : () => chain),
  });
  return {
    supabase: {
      from: () => chain,
      rpc: async () => ({ data: null, error: null }),
      auth: {
        getSession: async () => ({ data: { session: null } }),
        getUser: async () => ({ data: { user: null } }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => undefined } } }),
      },
    },
  };
});
vi.mock("@/lib/resource-collaborators", async (orig) => {
  const real = await orig<typeof import("../../src/lib/resource-collaborators")>();
  return {
    ...real,
    ensureResourceShareLink: async () => ({ token: "tok", isPublic: false }),
    loadResourceCollaboratorRows: async () => [],
  };
});

import { InviteCollaboratorsPopover } from "../../src/components/dashboard/InviteCollaboratorsPopover";
import { filterActiveAppUsers } from "../../src/lib/resource-collaborators";

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

async function render(tier: string) {
  m.tier = tier;
  await act(async () => root.render(<InviteCollaboratorsPopover open onClose={() => undefined} projectId="p1" />));
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe("invite UI by plan", () => {
  it("Free keeps the share link but sees no collaborator invite", async () => {
    await render("free");
    expect(screen.getByText("Share link")).toBeTruthy();
    expect(screen.queryByText("Team member")).toBeNull();
    expect(screen.queryByText("Send invites")).toBeNull();
  });

  it.each(["professional", "enterprise"])("%s sees the collaborator invite and the share link", async (tier) => {
    await render(tier);
    expect(screen.getByText("Share link")).toBeTruthy();
    expect(screen.getByText("Team member")).toBeTruthy();
    expect(screen.getByText("Send invites")).toBeTruthy();
  });

  it("the invitee list is not filtered by plan: a Free user can be invited", () => {
    const users = [
      { id: "u-free", name: "Fran Free", email: "fran@example.com", status: "Active" },
      { id: "u-pro", name: "Pat Pro", email: "pat@example.com", status: "Active" },
    ] as any;
    expect(filterActiveAppUsers(users, "").map((u: any) => u.id)).toEqual(["u-free", "u-pro"]);
  });
});
