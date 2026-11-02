// @vitest-environment happy-dom
// NEW-BG-01, owner ruling 2026-11-01: a Professional team's members must be on
// Professional or higher; adding a Free user is refused with an upgrade
// message. An Enterprise team takes any member of the organisation.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({
  tier: "professional" as string,
  eligibility: vi.fn(),
  upsertTeam: vi.fn(),
}));

vi.mock("@tanstack/react-start", () => ({ useServerFn: (fn: unknown) => fn }));
// Radix popovers do not open under happy-dom; render their content inline so the
// picker's own logic is what is under test.
vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: m.tier }, isLoading: false }) }));
vi.mock("@/lib/configurations.functions", () => ({
  listAppUsers: async () => [
    { id: "u-pro", name: "Pat Pro", email: "pat@example.com", role: "Analyst", avatar: null, status: "Active" },
    { id: "u-free", name: "Fran Free", email: "fran@example.com", role: "Analyst", avatar: null, status: "Active" },
  ],
  listBranches: async () => [{ name: "HQ", location: "" }],
  upsertTeam: m.upsertTeam,
  getTeamMemberEligibility: m.eligibility,
}));

import { NewTeamModal } from "../../src/components/dashboard/NewTeamModal";
import { TEAM_MEMBERS_NEED_PRO } from "../../src/lib/team-member-eligibility";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;

beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  m.eligibility.mockReset();
  m.eligibility.mockImplementation(async ({ data }: { data: { appUserIds: string[] } }) =>
    Object.fromEntries(data.appUserIds.map((id) => [id, id === "u-pro"])),
  );
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

async function openPicker() {
  await act(async () =>
    root.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <NewTeamModal open onOpenChange={() => undefined} />
      </QueryClientProvider>,
    ),
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

const option = (id: string) => document.querySelector(`[data-testid="team-member-option-${id}"]`) as HTMLElement;

describe("team-member picker on a Professional plan", () => {
  beforeEach(() => {
    m.tier = "professional";
  });

  it("asks the server which members qualify, and marks the Free one", async () => {
    await openPicker();
    expect(m.eligibility).toHaveBeenCalledWith({ data: { appUserIds: ["u-free", "u-pro"] } });
    expect(option("u-pro").getAttribute("aria-disabled")).toBe("false");
    expect(option("u-free").getAttribute("aria-disabled")).toBe("true");
    expect(option("u-free").textContent).toContain("Needs Professional");
  });

  it("refuses to add a Free user and shows the upgrade message", async () => {
    await openPicker();
    await act(async () => fireEvent.click(option("u-free")));
    expect(screen.getByRole("alert").textContent).toBe(TEAM_MEMBERS_NEED_PRO);
    expect(screen.queryByText("Members With Access")).toBeNull();
  });

  it("adds a Professional member", async () => {
    await openPicker();
    await act(async () => fireEvent.click(option("u-pro")));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("Members With Access")).toBeTruthy();
  });
});

describe("team-member picker on Enterprise", () => {
  it("takes any member of the organisation without asking for eligibility", async () => {
    m.tier = "enterprise";
    await openPicker();
    expect(m.eligibility).not.toHaveBeenCalled();
    expect(option("u-free").getAttribute("aria-disabled")).toBe("false");
    await act(async () => fireEvent.click(option("u-free")));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("Members With Access")).toBeTruthy();
  });
});
