// @vitest-environment happy-dom
// UI-B28 (UI Bugs): on Settings > Billing, an active subscription shows the
// saved Country, Billing Email and Phone number (they were blank).
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { screen, waitFor } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const summary = vi.hoisted(() => ({
  profile: {
    subscription_active: true,
    selected_plan: "enterprise",
    phone: "+1 555 0100",
    country: "Canada",
    billing_email: "billing@example.com",
    full_name: "Jane Doe",
    currency: "USD",
    monthly_price_cents: 4900,
  },
  customerAccount: { company_name: "Example Corp", billing_address: "1 Example Street", vat_number: "VAT123" },
  subscription: { status: "active" },
  license: null,
  trialDaysRemaining: null,
  invoices: [],
  planChanges: [],
  serialNumber: "SN-1",
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
  useActivePlan: () => ({ data: { plan: "enterprise", planRow: { billing_interval: "monthly" } }, isLoading: false }),
}));
vi.mock("@stripe/stripe-js", () => ({ loadStripe: vi.fn(async () => null) }));
vi.mock("@/integrations/supabase/client", async () => (await import("./support/w4a-supabase")).supabaseClientMock());
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }));

import { BillingTab } from "@/components/dashboard/BillingTab";

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

describe("UI-B28 billing details", () => {
  it("UI-B28: country, billing email and phone number of an active subscription are displayed", async () => {
    await act(async () =>
      root.render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <BillingTab />
        </QueryClientProvider>,
      ),
    );
    // Wait for the active-subscription view (company name is a field that was never blank).
    await waitFor(() => expect(screen.queryByDisplayValue("Example Corp")).not.toBeNull());
    expect(screen.queryByDisplayValue("+1 555 0100")).not.toBeNull();
    expect(screen.queryByDisplayValue("billing@example.com")).not.toBeNull();
    expect(screen.queryAllByText("Canada").length).toBeGreaterThan(0);
  });
});
