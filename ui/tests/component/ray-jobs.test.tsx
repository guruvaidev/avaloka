// @vitest-environment happy-dom
import { act, Suspense, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen, waitFor, within } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RayJobsPage } from "../../src/components/ray-jobs/RayJobsPage";
import { ClusterStatusPill } from "../../src/components/ray-jobs/RayJobPresentation";
import { canUseRayJobs, displayJobId, durationText, hasDataTransfer, isActiveJob, matchesJobId, rayJobMarkdown, rayJobsOptions, rayStatusOptions, type RayJob } from "../../src/lib/ray-jobs";

vi.mock("../../src/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: vi.fn(async () => ({ data: { session: { access_token: "test-token" } } })) } } }));
vi.mock("@tanstack/react-router", async (original) => ({ ...await original<object>(), Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>, useNavigate: () => vi.fn() }));
const notify = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }));
vi.mock("sonner", () => ({ toast: notify }));
vi.mock("../../src/components/dashboard/UpgradeGate", () => ({ UpgradePlanDialog: () => null }));
// Scroll-lock's CJS sidecar otherwise loads a second React in this test runtime.
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
const base: RayJob = {
  submission_id: "dta-dta-4f1c0a9e72b35d18", kind: "data_transfer", status: "RUNNING", message: "Transferring rows",
  thread_id: "thread", analysis_id: "analysis", source: "patient_data.csv", destination: "transfers/patient_data_oct25.csv",
  rows_written: 1234, runtime_s: 45.6, created_at: new Date(Date.now() - 300_000).toISOString(), ended_at: null,
};
let jobs: RayJob[];
let upgrade = false;
let stopConflict = false;
let unavailable = false;
let expiredLogs = false;
const calls: { path: string; method: string }[] = [];
let client: QueryClient;
let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
function render(content: React.ReactNode) {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  act(() => root?.render(content));
}
beforeEach(() => {
  jobs = [base, { ...base, submission_id: "dta-dta-e8bc6d2358664de9", status: "SUCCEEDED" }];
  upgrade = stopConflict = unavailable = expiredLogs = false;
  calls.length = 0;
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  vi.stubGlobal("fetch", vi.fn(async (_url, init: RequestInit) => {
    const headers = new Headers(init.headers);
    const path = headers.get("X-Backend-Path") ?? "";
    calls.push({ path, method: init.method ?? "GET" });
    expect(headers.get("Authorization")).toBe("Bearer test-token");
    let body: unknown;
    let status = 200;
    if (path === "/api/limits") body = { features: { data_transfer: true } };
    else if (upgrade) { status = 403; body = { detail: { reason: "plan_upgrade_required", message: "Upgrade to use data transfer" } }; }
    else if (unavailable) { status = 503; body = { detail: "Unavailable" }; }
    else if (path === "/api/ray/status") body = { configured: true, reachable: true, ray_version: "2.49.2", nodes: 2 };
    else if (path.includes("/logs")) { status = expiredLogs ? 410 : 200; body = expiredLogs ? { detail: "Logs are no longer available on the Ray cluster" } : { submission_id: base.submission_id, lines: ["Reading source", "Transferred 1,234 rows"], truncated: true }; }
    else if (path.endsWith("/stop")) {
      jobs = jobs.map((job) => ({ ...job, status: "SUCCEEDED" }));
      status = stopConflict ? 409 : 200;
      body = stopConflict ? { detail: "Already finished" } : { submission_id: base.submission_id, stopped: true, status: "STOPPED" };
    } else if (path.startsWith("/api/ray/jobs?")) body = { jobs };
    else body = jobs.find((job) => path.endsWith(job.submission_id)) ?? base;
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  }));
});
afterEach(() => { act(() => root?.unmount()); host?.remove(); root = null; host = null; client.clear(); vi.unstubAllGlobals(); vi.clearAllMocks(); });
function mount(jobId = "") {
  return render(<QueryClientProvider client={client}><Suspense fallback={<p>Loading</p>}><RayJobsPage filters={{ status: "", kind: "" }} jobId={jobId} onFiltersChange={vi.fn()} onJobChange={vi.fn()} /></Suspense></QueryClientProvider>);
}
describe("Ray jobs", () => {
  it.each([
    [{ configured: true, reachable: true, ray_version: "2.49.2", nodes: 2 }, "Connected · Ray 2.49.2 · 2 nodes"],
    [{ configured: true, reachable: false }, "Cluster unreachable"],
    [{ configured: false, reachable: false }, "No Ray cluster configured"],
    [{ configured: true, reachable: true, ray_version: "2.49.2", nodes: null }, "Connected · Ray 2.49.2"],
  ])("renders cluster state %#", (status, text) => { render(<ClusterStatusPill status={status} />); expect(screen.getByRole("status").textContent).toBe(text); });
  it("renders fetched jobs and hides Stop on finished jobs", async () => {
    mount();
    await screen.findByText(displayJobId(base.submission_id));
    expect(screen.getAllByText("1,234")).toHaveLength(2);
    const rows = screen.getAllByRole("row");
    expect(within(rows[1]).getByRole("button", { name: "Stop", exact: true })).toBeTruthy();
    expect(within(rows[2]).queryByRole("button", { name: "Stop", exact: true })).toBeNull();
    expect(calls.some((call) => call.path === "/api/ray/jobs?kind=&status=&limit=50")).toBe(true);
  });
  it("shows the 403 upgrade card and no table", async () => {
    upgrade = true; mount();
    await screen.findByText("Upgrade to use data transfer");
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByRole("button", { name: "Upgrade plan" })).toBeTruthy();
  });
  it("refreshes after 409 and reports Job already finished", async () => {
    stopConflict = true; mount();
    const stopButton = await screen.findByRole("button", { name: "Stop", exact: true });
    act(() => fireEvent.click(stopButton));
    expect(await screen.findByText("Rows already written stay written.")).toBeTruthy();
    act(() => fireEvent.click(screen.getByRole("button", { name: "Stop job" })));
    await waitFor(() => expect(notify).toHaveBeenCalledWith("Job already finished"));
    await waitFor(() => expect(calls.some((call) => call.path === `/api/ray/jobs/${base.submission_id}`)).toBe(true));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop", exact: true })).toBeNull());
  });
  it("matches the chat ID suffix, highlights the row and opens its logs", async () => {
    mount("dta-4f1c0a9e72b35d18");
    await screen.findByText("Showing the last 500 lines");
    const row = screen.getAllByText(displayJobId(base.submission_id)).find((el) => el.closest("tr"))?.closest("tr");
    expect(row?.getAttribute("data-highlighted")).toBe("true");
    expect(screen.getByLabelText("Job logs").textContent).toContain("Transferred 1,234 rows");
    expect(calls.some((call) => call.path === `/api/ray/jobs/${base.submission_id}/logs?tail=500`)).toBe(true);
  });
  it("shows a retry state for 503 without crashing", async () => {
    unavailable = true; mount();
    await screen.findByRole("alert");
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });
  it("copies the original backend ID despite the collapsed display", async () => {
    const writeText = vi.fn(async () => {});
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    mount();
    const copy = await screen.findByRole("button", { name: `Copy ${base.submission_id}` });
    act(() => fireEvent.click(copy));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("dta-dta-4f1c0a9e72b35d18"));
  });
  it("uses only opened-panel logs, then preserves 20 rows in the list after closing", async () => {
    const id = "dta-dta-b72196bb48ee461a";
    const destination = "gs://avaloka-dta-destination/transfers/patient_data_oct25c.csv";
    jobs = [{ ...base, submission_id: id, status: "SUCCEEDED", rows_written: null, destination, start_time: "2026-10-25T08:20:00Z" }];
    const originalFetch = fetch;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (new Headers(init.headers).get("X-Backend-Path")?.includes("/logs")) {
        calls.push({ path: new Headers(init.headers).get("X-Backend-Path") ?? "", method: init.method ?? "GET" });
        return new Response(JSON.stringify({ submission_id: id, lines: ["2026-10-25 01:20:10,205 INFO start", "AVALOKA_RESULT rows_inserted=20 table=transfers/patient_data_oct25c.csv"], truncated: false }));
      }
      return originalFetch(url, init);
    }));
    function App() {
      const [selected, setSelected] = useState("");
      return <RayJobsPage filters={{ status: "", kind: "" }} jobId={selected} onJobChange={setSelected} onFiltersChange={vi.fn()} />;
    }
    render(<QueryClientProvider client={client}><Suspense fallback={<p>Loading</p>}><App /></Suspense></QueryClientProvider>);
    const button = await screen.findByRole("button", { name: "Logs", exact: true });
    expect(calls.filter((call) => call.path.includes("/logs"))).toHaveLength(0);
    act(() => fireEvent.click(button));
    await screen.findByText(`20 rows transferred to ${destination}`);
    expect(screen.getByText("Log times are the Ray cluster's clock, not your local time.")).toBeTruthy();
    expect(screen.getByLabelText("Job logs").textContent).toContain("2026-10-25 01:20:10,205");
    expect(screen.getByText(/^Started /).textContent).toContain("2026");
    act(() => fireEvent.click(screen.getByRole("button", { name: "Close", exact: true })));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("20").closest("tr")?.textContent).toContain("dta-b72196bb48ee461a");
    expect(calls.filter((call) => call.path.includes("/logs"))).toHaveLength(1);
    expect(screen.getByText("20").closest("tr")?.querySelector("td[title]")?.getAttribute("title")).toContain("2026");
  });
  it("shows expired logs inside the drawer", async () => {
    expiredLogs = true; mount(base.submission_id);
    await screen.findByText("Logs are no longer available on the Ray cluster");
    expect(screen.getByRole("button", { name: "Download" }).hasAttribute("disabled")).toBe(true);
  });
  it("requires exactly true data_transfer eligibility", () => {
    expect(hasDataTransfer({ features: { data_transfer: true } })).toBe(true);
    expect(hasDataTransfer({ features: { data_transfer: false } })).toBe(false);
    expect(hasDataTransfer({})).toBe(false);
  });
  it("unlocks Ray jobs on Professional and Enterprise only", () => {
    expect(canUseRayJobs("professional", null)).toBe(true);
    expect(canUseRayJobs("Professional Plan", null)).toBe(true);
    expect(canUseRayJobs({ id: "enterprise" }, null)).toBe(true);
    expect(canUseRayJobs("Enterprise Plan", undefined)).toBe(true);
    expect(canUseRayJobs("free", null)).toBe(false);
    expect(canUseRayJobs({ id: "starter" }, { features: { data_transfer: false } })).toBe(false);
    expect(canUseRayJobs(undefined, undefined)).toBe(false);
    expect(canUseRayJobs("free", { features: { data_transfer: true } })).toBe(true);
    expect(canUseRayJobs(null, { limits: { features: { data_transfer: true } } })).toBe(true);
  });
  it("uses 5 second active-job polling and 30 second cluster polling", () => {
    const interval = rayJobsOptions({ status: "", kind: "" }).refetchInterval;
    if (typeof interval !== "function") throw new Error("Expected conditional polling");
    expect(interval({ state: { data: { data: { jobs: [base] } } } } as Parameters<typeof interval>[0])).toBe(5000);
    expect(interval({ state: { data: { data: { jobs: [{ ...base, status: "SUCCEEDED" }] } } } } as Parameters<typeof interval>[0])).toBe(30000);
    expect(rayStatusOptions.refetchInterval).toBe(30000);
    expect(isActiveJob({ ...base, status: "PENDING" })).toBe(true);
  });
  it("formats live and finished durations and suffixes precisely", () => {
    expect(durationText({ ...base, status: "SUCCEEDED", runtime_s: 45.6 })).toBe("45.6 s");
    expect(durationText({ ...base, status: "SUCCEEDED", runtime_s: 130 })).toBe("2 m 10 s");
    expect(durationText({ ...base, created_at: "2026-10-25T00:00:00Z" }, Date.parse("2026-10-25T00:02:10Z"))).toBe("2 m 10 s");
    expect(matchesJobId(base.submission_id, "dta-4f1c0a9e72b35d18")).toBe(true);
    expect(matchesJobId(base.submission_id, "dta-e8bc6d2358664de9")).toBe(false);
  });
  it("links matching IDs without altering existing markdown links or code", () => {
    const id = "dta-4f1c0a9e72b35d18";
    expect(rayJobMarkdown(`Job ${id}`)).toBe(`Job [${id}](/ray-jobs?job=${id})`);
    expect(rayJobMarkdown(`\`${id}\` [${id}](/existing)`)).toBe(`\`${id}\` [${id}](/existing)`);
  });
});
