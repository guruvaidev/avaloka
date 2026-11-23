// @vitest-environment happy-dom
// A plan that could not be checked is not a plan without the feature.
// GET /api/limits answers plan_check_unavailable: true when a hosted plan
// lookup failed (its "plan" then reads Free). Over a limit, the upload dialog
// must say the plan check is unavailable and offer no upgrade; a real Free plan
// still gets the upgrade prompt.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

const limits = vi.hoisted(() => ({ data: undefined as unknown }));
const invalidate = vi.hoisted(() => vi.fn());

vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: limits.data }),
  useQueryClient: () => ({ invalidateQueries: invalidate, setQueryData: vi.fn() }),
}));
vi.mock("@/lib/api/backendApi", () => ({
  backendApi: { getUploadLimits: vi.fn(), uploadFiles: vi.fn(() => new Promise(() => {})) },
}));
vi.mock("@/lib/analyses", () => ({ analysesKey: () => ["analyses"] }));
vi.mock("@/lib/analysis-messages", () => ({ createAnalysis: vi.fn() }));
vi.mock("@/lib/api/project-dashboard", () => ({ projectDashboardKey: () => ["project-dashboard"] }));
vi.mock("@/lib/pending-dashboard-analysis", () => ({ setPendingDashboardAnalysis: vi.fn() }));
vi.mock("@/lib/auto-insights-bot", () => ({ startAutoInsightsBot: vi.fn(), finishAutoInsightsBot: vi.fn() }));
vi.mock("sonner", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), message: vi.fn(), success: vi.fn() }) }));
vi.mock("@/components/database/ConnectCloudModal", () => ({ ConnectCloudModal: () => null }));

import { UploadModal } from "../../src/components/dashboard/UploadModal";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  limits.data = undefined;
  invalidate.mockClear();
});

const flush = () => act(async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
});

/** Free-plan limits of 8 bytes per file, so the 30-byte CSV below is over the limit. */
const FREE_LIMITS = { plan: "free", max_file_bytes: 8, max_total_bytes: 8, max_files: 1, batch_jobs: false };

async function openOverLimit() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  const file = new File(["region,amount\nnorth,10\nsouth,20\n"], "regional_sales.csv", { type: "text/csv" });
  act(() => root!.render(<UploadModal open onOpenChange={() => {}} initialFiles={[file]} />));
  await flush();
  return document.body.textContent ?? "";
}

describe("upload limit dialog when the plan could not be checked", () => {
  it("a real Free plan over the limit still gets the upgrade prompt", async () => {
    limits.data = { ...FREE_LIMITS, plan_check_unavailable: false };
    const text = await openOverLimit();
    expect(text).toContain("Free plan limit reached");
    expect(text).toContain("Upgrade to Professional");
  });

  it("an unchecked plan says so, offers no upgrade, and re-checks next time", async () => {
    limits.data = { ...FREE_LIMITS, plan_check_unavailable: true };
    const text = await openOverLimit();
    expect(text).toContain("Plan check unavailable");
    expect(text).toContain("try again");
    expect(text).not.toContain("plan limit reached");
    expect(text.toLowerCase()).not.toContain("upgrade");
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["upload-limits"] });
  });
});
