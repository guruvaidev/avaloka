// @vitest-environment happy-dom
// Owner ruling 2026-11-02: share links to an ANALYSIS are Professional or
// higher. Report sharing, and emailing an analysis's results, stay open on
// every plan.
//   * usePlanFeatures().canShareAnalysisLink is the UI flag, failing closed.
//   * The analysis share panel shows the plan note instead of Generate, and no
//     visibility control, without it; "Email Analysis" stays.
//   * POST /api/analysis/:id/share-link refuses a caller below Professional
//     with a true message, before any share-link row is read or written.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const plan = vi.hoisted(() => ({ value: "free" as string | undefined, loading: false }));
const collab = vi.hoisted(() => ({ value: false, rpcs: [] as string[] }));
const adminTables = vi.hoisted(() => ({ touched: [] as string[] }));

vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({
    data: plan.value === undefined ? undefined : { plan: plan.value },
    isLoading: plan.loading,
  }),
}));
const serverFn = vi.hoisted(() => vi.fn(async () => []));
vi.mock("@tanstack/react-start", () => ({ useServerFn: () => serverFn }));
// The Professional sharing panel (4140dd96) imports these server functions;
// their module builds auth middleware at load time, which this harness does not
// provide. The panel calls them only through useServerFn, mocked above.
// The panel itself renders on Professional; it is not what this file tests.
vi.mock("@/components/dashboard/AnalysisStandaloneSharing", () => ({ AnalysisStandaloneSharing: () => null }));
vi.mock("@/lib/analysis-sharing.functions", () => ({
  getAnalysisShareSettings: vi.fn(),
  createAnalysisShare: vi.fn(),
  updateAnalysisShare: vi.fn(),
  removeAnalysisShare: vi.fn(),
}));
vi.mock("@/lib/configurations.functions", () => ({ listAppUsers: vi.fn() }));
vi.mock("@/lib/auth-user", () => ({ getCachedAuthUser: vi.fn(async () => ({ id: "user-1" })) }));
vi.mock("@/integrations/supabase/client", async () =>
  (await import("../release_1_6/support/w4a-supabase")).supabaseClientMock({}),
);
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn(), message: vi.fn() } }));

// The route module: createFileRoute returns its options so the handler can be called.
vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-router")>()),
  createFileRoute: () => (opts: unknown) => ({ options: opts }),
}));
// The caller's own client: answers the plan RPCs.
vi.mock("@supabase/supabase-js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@supabase/supabase-js")>()),
  createClient: () => ({
    rpc: async (name: string, args?: { p_action?: string; p_required?: string }) => {
      collab.rpcs.push(args?.p_action ? `${name}:${args.p_action}:${args.p_required}` : name);
      if (name === "collab_features_enabled") return { data: collab.value, error: null };
      return { data: null, error: null };
    },
  }),
}));
// The admin client: the caller owns the analysis; every table touched is recorded.
function adminChain(table: string): any {
  const rows: Record<string, unknown> = {
    profiles: { id: "p1" },
    analyses: { id: "a1", owner_id: "p1" },
    resource_share_links: { share_token: "tok", link_access: "view" },
  };
  const c: any = new Proxy(
    {},
    {
      get: (_t, prop) => {
        if (prop === "then") return undefined;
        if (prop === "maybeSingle" || prop === "single") return async () => ({ data: rows[table] ?? null, error: null });
        return () => c;
      },
    },
  );
  return c;
}
vi.mock("@/integrations/supabase/client.server", () => ({
  supabaseAdmin: {
    auth: { getUser: async () => ({ data: { user: { id: "u1" } }, error: null }) },
    from: (t: string) => {
      adminTables.touched.push(t);
      return adminChain(t);
    },
  },
}));

import { usePlanFeatures } from "@/lib/use-plan-features";
import { ShareCollaboratorsButton } from "@/components/dashboard/ShareCollaboratorsButton";
import { Route as ShareLinkRoute } from "@/routes/api/analysis.$analysisId.share-link";

// Pinned literally so a missing constant fails on an assertion.
const NOTE = "Share links to an analysis are included in the Professional and Enterprise plans. You can still email the results.";
const REFUSED =
  "Analysis share links need a Professional plan or higher. Reports can still be shared by email and public link on every plan.";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  plan.value = "free";
  plan.loading = false;
  collab.value = false;
  collab.rpcs.length = 0;
  adminTables.touched.length = 0;
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

describe("usePlanFeatures().canShareAnalysisLink", () => {
  it.each([
    ["free", false],
    ["professional", true],
    ["enterprise", true],
    [undefined, false],
  ])("plan %s -> %s", (tier, expected) => {
    plan.value = tier as string | undefined;
    expect(renderHook(() => usePlanFeatures()).result.current.canShareAnalysisLink).toBe(expected);
  });

  it("fails closed while the plan loads", () => {
    plan.value = "enterprise";
    plan.loading = true;
    expect(renderHook(() => usePlanFeatures()).result.current.canShareAnalysisLink).toBe(false);
  });
});

async function openPanel() {
  await act(async () =>
    root.render(<ShareCollaboratorsButton analysisId="analysis-1" projectId={null} open onOpenChange={() => undefined} />),
  );
  await waitFor(() => expect(document.querySelector(".invite-collaborators-dropdown")).not.toBeNull());
  await act(async () => new Promise((r) => setTimeout(r, 20)));
}

describe("analysis share panel by plan", () => {
  it("Free: plan note, no Generate, no visibility control; email stays", async () => {
    plan.value = "free";
    await openPanel();
    expect(screen.queryByText(NOTE)).not.toBeNull();
    expect(screen.queryByText("Generate")).toBeNull();
    expect(screen.queryByText("Private")).toBeNull();
    expect(screen.queryByText("Email Analysis")).not.toBeNull();
  });

  it.each([["professional"], ["enterprise"]])("%s: Generate and no plan note", async (tier) => {
    plan.value = tier;
    await openPanel();
    expect(screen.queryByText(NOTE)).toBeNull();
    expect(screen.queryByText("Generate")).not.toBeNull();
    expect(screen.queryByText("Email Analysis")).not.toBeNull();
  });
});

function post() {
  const handler = (ShareLinkRoute as any).options.server.handlers.POST;
  return handler({
    request: new Request("https://app.test/api/analysis/a1/share-link", {
      method: "POST",
      headers: { authorization: "Bearer t" },
      body: JSON.stringify({ access_level: "view" }),
    }),
    params: { analysisId: "a1" },
  }) as Promise<Response>;
}

describe("POST /api/analysis/:id/share-link", () => {
  it("refuses a caller below Professional with 403 and the true message, touching no share link", async () => {
    collab.value = false;
    const res = await post();
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ error: REFUSED });
    expect(adminTables.touched).not.toContain("resource_share_links");
    expect(collab.rpcs).toEqual(["collab_features_enabled"]);
  });

  it("lets a Professional caller through, recording the check at the Professional level", async () => {
    collab.value = true;
    const res = await post();
    expect(res.status).toBe(200);
    expect(((await res.json()) as { token?: string }).token).toBe("tok");
    expect(adminTables.touched).toContain("resource_share_links");
    expect(collab.rpcs).toEqual(["collab_features_enabled", "log_org_feature_bypass:createAnalysisShareLink:professional"]);
  });
});
