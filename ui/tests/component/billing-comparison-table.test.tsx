// @vitest-environment happy-dom
// Owner ruling 2026-11-01: the Billing tab's comparison table is generated from
// the edition matrix. Renders the real BillingTab; only server functions and
// Supabase are stubbed. The owner's rows are spelled out here, with no import of
// the generated matrix, so they fail on assertions against a hand-written table.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const summary = vi.hoisted(() => ({
  profile: { subscription_active: false, selected_plan: null, full_name: "Test User", currency: "USD" },
  customerAccount: null,
  subscription: null,
  license: null,
  trialDaysRemaining: null,
  invoices: [],
  planChanges: [],
  serialNumber: null,
}));

vi.mock("@tanstack/react-start", () => ({ useServerFn: (fn: unknown) => fn }));
vi.mock("@/lib/enterprise.functions", () => ({ getBillingSummary: vi.fn(async () => summary), upgradeToEnterprise: vi.fn() }));
vi.mock("@/lib/billing.functions", () => ({
  createEnterpriseCheckoutSession: vi.fn(),
  setupTrialPaymentMethod: vi.fn(),
  cancelSubscriptionAtPeriodEnd: vi.fn(),
  openBillingPortal: vi.fn(),
}));
vi.mock("@/lib/gateway-payment-method.functions", () => ({ getActiveGatewayPaymentMethod: vi.fn(async () => null) }));
vi.mock("@/lib/paypal.functions", () => ({ createPayPalSubscription: vi.fn() }));
vi.mock("@/lib/paypal-method.functions", () => ({ changePayPalPaymentMethod: vi.fn() }));
vi.mock("@/lib/paypal-finalize.functions", () => ({ finalizeEnterprisePayPalSubscription: vi.fn() }));
vi.mock("@/lib/stripe-cards.functions", () => ({ createStripeSetupIntent: vi.fn(), saveStripePaymentMethod: vi.fn() }));
vi.mock("@/components/dashboard/sales-inquiry.server", () => ({ submitSalesInquiry: vi.fn() }));
vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({ data: { plan: "free", planRow: { billing_interval: "monthly" } }, isLoading: false }),
}));
vi.mock("@stripe/stripe-js", () => ({ loadStripe: vi.fn(async () => null) }));
vi.mock("@/integrations/supabase/client", async () =>
  (await import("../release_1_6/support/w4a-supabase")).supabaseClientMock({
    plans: [
      { id: "p0", plan_type: "free", name: "Free", billing_interval: "monthly", price: 0 },
      { id: "p1", plan_type: "professional", name: "Professional", billing_interval: "monthly", price: 49 },
      { id: "p2", plan_type: "enterprise", name: "Enterprise", billing_interval: "monthly", price: 199 },
    ],
  }),
);
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }));

import { BillingTab } from "@/components/dashboard/BillingTab";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
let host: HTMLElement;
beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

async function renderTable(): Promise<HTMLElement> {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  await act(async () => {
    root.render(
      <QueryClientProvider client={client}>
        <BillingTab />
      </QueryClientProvider>,
    );
  });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  const heading = [...document.querySelectorAll("h3")].find((h) => h.textContent === "Overall Comparison");
  expect(heading, "the Overall Comparison block").toBeTruthy();
  return heading!.parentElement!.parentElement!;
}

type Cell = "yes" | "no" | "coming";

/** Every row as label -> [free, professional, enterprise], read from what is on screen. */
function readRows(table: HTMLElement): Map<string, Cell[]> {
  const rows = [...table.querySelectorAll<HTMLElement>("div.grid")].filter(
    (row) => row.children.length === 4 && row.querySelector(":scope > div.flex.justify-center") !== null,
  );
  return new Map(
    rows.map((row) => [
      row.children[0].textContent!.trim(),
      [...row.children].slice(1).map((cell): Cell => {
        if (cell.querySelector("svg")) {
          expect(cell.textContent, "a tick cell has no other text").toBe("");
          return "yes";
        }
        if (cell.textContent === "Coming soon") return "coming";
        expect(cell.textContent).toBe("—");
        return "no";
      }),
    ]),
  );
}

const FREE_PRO_ENT = (free: Cell, pro: Cell, ent: Cell) => [free, pro, ent];

describe("Billing comparison table, owner's rows", () => {
  it.each([
    ["Teams and collaborator invites", FREE_PRO_ENT("no", "yes", "yes")],
    ["Departments, roles, branches and app users", FREE_PRO_ENT("no", "no", "yes")],
    ["Share analyses", FREE_PRO_ENT("no", "yes", "yes")],
    ["Comment on analyses", FREE_PRO_ENT("no", "yes", "yes")],
    ["Notifications", FREE_PRO_ENT("no", "yes", "yes")],
    // Owner rulings 2026-11-02.
    ["SSO and audit log", FREE_PRO_ENT("no", "no", "coming")],
    ["Auto Insights", FREE_PRO_ENT("yes", "yes", "yes")],
    ["Reports", FREE_PRO_ENT("no", "yes", "yes")],
  ])("%s", async (label, expected) => {
    expect(readRows(await renderTable()).get(label)).toEqual(expected);
  });

  it("says every plan shares reports by email or public link, and Pro team members need Pro", async () => {
    const text = (await renderTable()).textContent!;
    expect(text).toContain("Every plan can share a report by email or public link.");
    expect(text).toContain("On Professional, team members need Professional or higher.");
  });

  it.each(["Team collaboration", "Teams and groups", "Group workloads", "Basic analytics", "Advanced analytics"])(
    "the unmapped or contradicted row %s is gone",
    async (label) => {
      expect(readRows(await renderTable()).has(label)).toBe(false);
    },
  );

  it("every row on screen is a capability in the matrix, with the matrix's cells", async () => {
    const { comparisonSections } = await import("@/lib/edition-comparison");
    const expected = new Map(
      comparisonSections()
        .flatMap((s) => s.rows)
        .map((row) => [row.label, [row.cells.free, row.cells.professional, row.cells.enterprise]]),
    );
    expect(expected.size).toBeGreaterThan(0);
    expect(readRows(await renderTable())).toEqual(expected);
  });
});

/** A plan card's feature lines, found by the card's plan name. */
function cardLines(name: string): string[] {
  const card = [...document.querySelectorAll<HTMLElement>("div.rounded-2xl")].find(
    (c) => c.querySelector("span.text-sm.font-semibold")?.textContent === name,
  );
  expect(card, `${name} plan card`).toBeTruthy();
  return [...card!.querySelectorAll("li")].map((li) => li.textContent!.trim());
}

describe("Billing plan cards", () => {
  it("Auto Insights is on the Free card, not only on Professional's", async () => {
    await renderTable();
    expect(cardLines("Free")).toContain("Auto Insights");
    expect(cardLines("Professional")).not.toContain("Auto Insights");
  });

  it("every card line is a row of the table that the plan includes", async () => {
    const rows = readRows(await renderTable());
    (["Free", "Professional", "Enterprise"] as const).forEach((name, i) => {
      for (const line of cardLines(name)) {
        const coming = line.endsWith(" (coming soon)");
        const cells = rows.get(line.replace(/ \(coming soon\)$/, ""));
        expect(cells, `${name} card line "${line}" is a table row`).toBeTruthy();
        expect(cells![i], `${name}: ${line}`).toBe(coming ? "coming" : "yes");
      }
    });
  });

  it("the Enterprise card says SSO and audit log is coming, not included", async () => {
    await renderTable();
    expect(cardLines("Enterprise")).toContain("SSO and audit log (coming soon)");
  });
});
