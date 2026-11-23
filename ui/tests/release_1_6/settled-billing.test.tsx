// @vitest-environment happy-dom
// 1.6 release suite: Billing rows settled from the shipped code
// (UI Bugs UI-B26 "fixed": prices shown in US dollars; UI-B27 "Not an issue":
// the Professional checkout asks for company name and VAT like Enterprise).
// Renders the real BillingTab; only server functions and Supabase are stubbed.
// See docs/test-reports/1.6-traceability.csv.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const summary = vi.hoisted(() => ({
  profile: { subscription_active: false, selected_plan: null, full_name: "Jane Doe", currency: "USD" },
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
  (await import("./support/w4a-supabase")).supabaseClientMock({
    plans: [
      { id: "p0", plan_type: "free", name: "Free", billing_interval: "monthly", price: 0 },
      { id: "p1", plan_type: "professional", name: "Professional", billing_interval: "monthly", price: 49 },
      { id: "p2", plan_type: "enterprise", name: "Enterprise", billing_interval: "monthly", price: 199 },
    ],
  }),
);
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }));

import { BillingTab } from "@/components/dashboard/BillingTab";
import { validateBillingDetails } from "@/lib/billing-details";

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

async function renderBilling() {
  await act(async () =>
    root.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <BillingTab />
      </QueryClientProvider>,
    ),
  );
  for (let i = 0; i < 20 && !document.body.textContent?.includes("Simple, transparent pricing"); i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
  }
  expect(document.body.textContent).toContain("Simple, transparent pricing");
}

describe("UI-B26 plan prices", () => {
  it("UI-B26: plan cards show prices in US dollars ($49, $199), never rupees", async () => {
    await renderBilling();
    const text = document.body.textContent ?? "";
    expect(text).toContain("$49");
    expect(text).toContain("$199");
    expect(text).not.toMatch(/₹|\bRs\.?\s?\d|\bINR\b/);
  });
});

describe("UI-B27 Professional checkout billing fields", () => {
  it("UI-B27: choosing Professional opens billing info asking for Company Name and VAT Number (agreed: not an issue)", async () => {
    await renderBilling();
    // "Get Started" is the Professional CTA only (Free says "Get started", Enterprise
    // is contact-sales); it appears on the plan card and in the comparison block,
    // and both open the same Professional buy-now flow.
    const getStarted = [...document.querySelectorAll("button")].filter((b) => b.textContent?.trim() === "Get Started");
    expect(getStarted.length, "Professional CTA").toBeGreaterThanOrEqual(1);
    await act(async () => {
      getStarted[0].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    const labels = [...document.querySelectorAll("label")].map((l) => l.textContent?.replace(/\*/g, "").trim());
    expect(labels).toContain("Company Name");
    expect(labels).toContain("VAT Number");
  });

  it("UI-B27: the shared billing validator requires company name and VAT for every paid plan", () => {
    const { errors } = validateBillingDetails({
      company_name: "",
      billing_address: "1 Example Street, Springfield 12345",
      vat_number: "",
      auto_pay_authorized: true,
    });
    expect(errors.company_name).toBeTruthy();
    expect(errors.vat_number).toBeTruthy();
    expect(errors.billing_address).toBeUndefined();
  });
});
