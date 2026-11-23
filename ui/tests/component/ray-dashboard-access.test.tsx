// @vitest-environment happy-dom
import { act, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { fireEvent, screen } from "@testing-library/dom";
import { expect, it, vi } from "vitest";
import { IconRail } from "../../src/components/dashboard/IconRail";
import { RayJobText } from "../../src/components/analysis/RayJobLink";
import { RayRequestError } from "../../src/lib/ray-dashboard-session";

const mocks = vi.hoisted(() => ({ status: vi.fn(), open: vi.fn(), stop: vi.fn(), id: "professional-user", listener: undefined as ((event: string, session: unknown) => void) | undefined }));
vi.mock("../../src/lib/ray-dashboard-session", async (original) => ({ ...(await original<object>()), rayApi: { status: mocks.status }, openRayDashboard: mocks.open, stopRayDashboardSessions: mocks.stop }));
vi.mock("../../src/lib/api/backendApi", () => ({ getAccessToken: vi.fn(), proxyFetch: vi.fn() }));
vi.mock("../../src/integrations/supabase/client", () => ({ supabase: { auth: {
  getSession: async () => ({ data: { session: { user: { id: mocks.id } } } }),
  onAuthStateChange: (listener: typeof mocks.listener) => { mocks.listener = listener; return { data: { subscription: { unsubscribe: vi.fn() } } }; },
} } }));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: null }) }));
// IconRail renders a router <Link> for the Inference item (since the inference panel).
vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn(), useRouterState: () => "/proanalysis", Link: ({ children, to, ...p }: { children: React.ReactNode; to: string }) => <a href={to} {...p}>{children}</a> }));
vi.mock("../../src/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "professional" } }) }));
vi.mock("../../src/lib/use-user-mgmt-perms", () => ({ useProjectsPerms: () => ({ canView: true }), useDashboardPerms: () => ({ canView: true }) }));
vi.mock("../../src/components/dashboard/NotificationsPopover", () => ({ NotificationsPopover: () => null }));
vi.mock("../../src/components/dashboard/UpgradeGate", () => ({ UpgradePlanDialog: ({ open }: { open: boolean }) => open ? <div role="dialog">Upgrade your plan</div> : null }));
vi.mock("react-aria-components", () => ({ Button: () => null }));
vi.mock("../../src/components/base/avatar/avatar", () => ({ Avatar: () => null }));
vi.mock("../../src/components/base/dropdown/dropdown", () => ({ Dropdown: { Root: () => null } }));
vi.mock("../../src/components/base/buttons/button-utility", () => ({ ButtonUtility: ({ tooltip, onClick }: { tooltip: string; onClick: () => void }) => <button aria-label={tooltip} onClick={onClick} /> }));
vi.mock("../../src/components/base/buttons/button", () => ({ Button: ({ children, onClick }: { children: React.ReactNode; onClick: () => void }) => <button onClick={onClick}>{children}</button> }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("shares one status call across Strict Mode sidebar and chat; handles login, upgrade and other errors", async () => {
  mocks.status.mockResolvedValue(new Response());
  const host = document.createElement("div"); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<StrictMode><IconRail /><RayJobText content="Job dta-4f1c0a9e72b35d18" /></StrictMode>));
    expect(mocks.status).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Ray dashboard" }));
    expect(mocks.open).toHaveBeenCalledWith(expect.any(Function), undefined);
    fireEvent.click(screen.getByRole("button", { name: "Open in Ray dashboard" }));
    expect(mocks.open).toHaveBeenLastCalledWith(expect.any(Function), "dta-4f1c0a9e72b35d18");
    await act(async () => { mocks.listener?.("TOKEN_REFRESHED", { user: { id: mocks.id } }); await new Promise((resolve) => setTimeout(resolve, 10)); });
    expect(mocks.status).toHaveBeenCalledTimes(1);
    await act(async () => { mocks.listener?.("SIGNED_OUT", null); await new Promise((resolve) => setTimeout(resolve, 10)); });
    expect(screen.queryByRole("button", { name: "Ray dashboard" })).toBeNull();
    mocks.status.mockRejectedValue(new RayRequestError(403, "plan_upgrade_required"));
    await act(async () => { mocks.listener?.("SIGNED_IN", { user: { id: "free-user" } }); await new Promise((resolve) => setTimeout(resolve, 10)); });
    expect(mocks.status).toHaveBeenCalledTimes(2);
    expect(host.querySelector('[data-rail-item="ray-dashboard"]')?.getAttribute("data-locked")).toBe("true");
    expect(screen.queryByRole("button", { name: "Open in Ray dashboard" })).toBeNull();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Ray dashboard" })));
    expect(screen.getByRole("dialog").textContent).toBe("Upgrade your plan");
    mocks.status.mockRejectedValue(new RayRequestError(500));
    await act(async () => { mocks.listener?.("SIGNED_IN", { user: { id: "other-user" } }); await new Promise((resolve) => setTimeout(resolve, 10)); });
    expect(screen.queryByRole("button", { name: "Ray dashboard" })).toBeNull();
  } finally { act(() => root.unmount()); host.remove(); }
});