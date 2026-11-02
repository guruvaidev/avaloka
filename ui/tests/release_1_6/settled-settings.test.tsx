// @vitest-environment happy-dom
// 1.6 release suite: Settings rows settled from the shipped code
// (UI Bugs UI-B23 "Deployed", UI-B24 "It is a Feature").
// Renders the real Settings route component; only the network edges
// (Supabase, server functions, the integrations API) are stubbed.
// See docs/test-reports/1.6-traceability.csv.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const plan = vi.hoisted(() => ({ current: "free" as "free" | "professional" | "enterprise" }));

vi.mock("@tanstack/react-start", () => ({ useServerFn: (fn: unknown) => fn }));
vi.mock("@/lib/settings.functions", () => ({
  updateOrganizationProfile: vi.fn(async () => null),
  getOrganizationProfile: vi.fn(async () => null),
}));
vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({ data: { plan: plan.current, isOrgOwner: true }, isSuccess: true, isError: false, isLoading: false }),
}));
vi.mock("@/lib/theme", () => ({ setTheme: vi.fn() }));
vi.mock("@/lib/api/integrations", () => ({
  integrationsApi: { list: vi.fn(async () => []), patchGithub: vi.fn(), disconnectGithub: vi.fn() },
}));
vi.mock("@/components/dashboard/IntegrationConfigModal", () => ({ IntegrationConfigModal: () => null }));
vi.mock("@/components/dashboard/BillingTab", () => ({ BillingTab: () => null }));
vi.mock("@/integrations/supabase/client", async () => (await import("./support/w4a-supabase")).supabaseClientMock());
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }));

import { Route } from "@/routes/_authenticated/settings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const SettingsPage = (Route as unknown as { options: { component: () => JSX.Element } }).options.component;

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
  window.history.replaceState(null, "", "/");
});

async function renderSettings(tab: string) {
  window.history.replaceState(null, "", `/settings?tab=${tab}`);
  await act(async () =>
    root.render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SettingsPage />
      </QueryClientProvider>,
    ),
  );
  // Let the profile load effect settle (it awaits the Supabase stubs).
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

const tabLabels = () =>
  [...host.querySelectorAll("button")]
    .map((b) => b.textContent?.trim() ?? "")
    .filter((t) => ["My details", "Profile", "Appearance", "Email", "Notifications", "Integrations", "Billing"].includes(t));

describe("UI-B23 Profile tab by plan", () => {
  it("UI-B23: the Profile (company) tab is not offered on the Free or Professional plan", async () => {
    for (const p of ["free", "professional"] as const) {
      plan.current = p;
      await renderSettings("my-details");
      expect(tabLabels(), `plan ${p}`).not.toContain("Profile");
      expect(tabLabels(), `plan ${p}`).toContain("My details");
      act(() => root.unmount());
      root = createRoot(host);
    }
  });

  it("UI-B23: the Profile (company) tab is offered on the Enterprise plan", async () => {
    plan.current = "enterprise";
    await renderSettings("my-details");
    expect(tabLabels()).toContain("Profile");
  });
});

describe("UI-B24 Integrations catalogue", () => {
  it("UI-B24: GitHub is connectable; Outlook, Slack and Jira are listed as view-only (not yet connectable)", async () => {
    plan.current = "professional";
    await renderSettings("integrations");
    const cards = [...host.querySelectorAll("div.rounded-xl.border")].filter((c) =>
      ["GitHub", "Outlook", "Slack", "Atlassian JIRA"].includes(c.querySelector("span.text-sm.font-semibold.text-foreground")?.textContent ?? ""),
    );
    const byName = Object.fromEntries(cards.map((c) => [c.querySelector("span.text-sm.font-semibold.text-foreground")!.textContent!, c]));
    expect(Object.keys(byName).sort()).toEqual(["Atlassian JIRA", "GitHub", "Outlook", "Slack"]);

    const action = (c: Element) => [...c.querySelectorAll("button")].map((b) => b.textContent?.trim()).filter(Boolean).pop();
    expect(action(byName.GitHub)).toBe("Connect");
    for (const name of ["Outlook", "Slack", "Atlassian JIRA"]) {
      expect(action(byName[name]), name).toBe("View integration");
      const toggle = byName[name].querySelector('button[role="switch"]') as HTMLButtonElement;
      expect(toggle.disabled, `${name} toggle disabled`).toBe(true);
      expect(toggle.getAttribute("aria-checked"), `${name} off`).toBe("false");
    }
  });
});
