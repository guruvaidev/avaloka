// @vitest-environment happy-dom
// Owner ruling 2026-11-02: the Billing table's Reports and Auto Insights rows must
// match the gates the app actually applies. Each test renders the real gate for
// every plan and compares what it lets through with the table's cell.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({ tier: "free" as string, navigate: vi.fn() }));

vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({ data: { plan: m.tier }, isLoading: false }),
}));
vi.mock("@tanstack/react-router", () => ({
  createFileRoute: () => (options: unknown) => ({ options }),
  useNavigate: () => m.navigate,
  useParams: () => ({ reportId: "r1" }),
  useRouter: () => ({ navigate: m.navigate }),
  Link: ({ children }: { children?: unknown }) => children,
}));
// The pages behind the gates are not under test; each renders a marker.
vi.mock("@/components/dashboard/ReportsSidebar", () => ({ ReportsSidebar: () => <div data-gate-open="reports" /> }));
vi.mock("@/components/dashboard/ReportsView", () => ({ ReportsView: () => null }));
vi.mock("@/lib/reports", () => ({
  deleteReport: vi.fn(),
  fetchReportProjects: vi.fn(async () => []),
  fetchReports: vi.fn(async () => []),
}));
vi.mock("@/integrations/supabase/client", async () =>
  (await import("../release_1_6/support/w4a-supabase")).supabaseClientMock({}),
);
vi.mock("@/lib/auth-user", () => ({ getCachedAuthUser: async () => null }));
vi.mock("@/components/insights-avatar/InsightsAvatarContext", () => ({
  InsightsAvatarProvider: () => <div data-gate-open="auto-insights" />,
  useInsightsAvatar: () => ({}),
  HIGHLIGHT_CLASS: "",
}));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Route as ReportsIndexRoute } from "@/routes/_authenticated/reports.index";
import { railLockedIds } from "@/components/dashboard/IconRail";
import { AutoInsightsModal } from "@/components/dashboard/AutoInsightsModal";
import { comparisonSections } from "@/lib/edition-comparison";
import { EDITION_PLANS, type EditionPlan } from "@/lib/edition-matrix.gen";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  m.navigate.mockReset();
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

function tableTick(capability: string, plan: EditionPlan): boolean {
  const row = comparisonSections()
    .flatMap((s) => s.rows)
    .find((r) => r.capability === capability);
  expect(row, `${capability} is a table row`).toBeTruthy();
  return row!.cells[plan] === "yes";
}

async function opens(node: React.ReactNode, marker: string): Promise<boolean> {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  await act(async () => {
    root.render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
  });
  return document.querySelector(`[data-gate-open="${marker}"]`) !== null;
}

describe("the table's ticks are the app's gates", () => {
  it.each(EDITION_PLANS)("Reports on %s: the /reports route gate lets in exactly the plans the table ticks", async (plan) => {
    m.tier = plan;
    const Page = (ReportsIndexRoute as unknown as { options: { component: () => React.ReactNode } }).options.component;
    const open = await opens(<Page />, "reports");
    expect(open).toBe(tableTick("reports", plan));
    expect(m.navigate.mock.calls.length > 0).toBe(!tableTick("reports", plan));
  });

  it.each(EDITION_PLANS)("Reports on %s: the rail locks Reports exactly when the table does not tick it", (plan) => {
    expect(railLockedIds(plan === "free", true, plan).has("reports")).toBe(!tableTick("reports", plan));
  });

  it.each(EDITION_PLANS)("Auto Insights on %s: the modal opens exactly when the table ticks it", async (plan) => {
    m.tier = plan;
    const open = await opens(<AutoInsightsModal {...({ open: true, onOpenChange: () => {} } as never)} />, "auto-insights");
    expect(open).toBe(tableTick("auto_insights", plan));
  });
});
