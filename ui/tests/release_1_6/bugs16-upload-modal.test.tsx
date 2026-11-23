// @vitest-environment happy-dom
// 1.6 release suite: Bugs-1.6 upload drop zone and "Select File & Start
// Analysis" modal rows (BG18, BG20, BG21, BG23, BG26).
// The real UploadModal is driven through a real upload: only the network call
// (backendApi.uploadFiles) and app-level side effects are stubbed.
// See docs/test-reports/1.6-traceability.csv.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

const upload = vi.hoisted(() => ({ resolve: null as null | ((value: unknown) => void) }));

vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: undefined }),
  useQueryClient: () => ({ invalidateQueries: vi.fn(), setQueryData: vi.fn() }),
}));
vi.mock("@/lib/api/backendApi", () => ({
  backendApi: {
    getUploadLimits: vi.fn(),
    // Resolves only when the test says so, so step 1 can be observed first.
    uploadFiles: vi.fn(
      (_files: File[], _a: unknown, _b: unknown, onProgress?: (p: { phase: string; loaded: number; total: number }) => void) =>
        new Promise((resolve) => {
          onProgress?.({ phase: "uploading", loaded: 1, total: 1 });
          upload.resolve = resolve;
        }),
    ),
  },
}));
vi.mock("@/lib/analyses", () => ({ analysesKey: () => ["analyses"] }));
vi.mock("@/lib/analysis-messages", () => ({ createAnalysis: vi.fn() }));
vi.mock("@/lib/api/project-dashboard", () => ({ projectDashboardKey: () => ["project-dashboard"] }));
vi.mock("@/lib/pending-dashboard-analysis", () => ({ setPendingDashboardAnalysis: vi.fn() }));
vi.mock("@/lib/auto-insights-bot", () => ({ startAutoInsightsBot: vi.fn(), finishAutoInsightsBot: vi.fn() }));
vi.mock("sonner", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), message: vi.fn(), success: vi.fn() }) }));
vi.mock("@/components/database/ConnectCloudModal", () => ({ ConnectCloudModal: () => null }));

import { UploadModal } from "../../src/components/dashboard/UploadModal";
import { UploadDropzone } from "../../src/components/dashboard/UploadDropzone";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  upload.resolve = null;
});

const flush = () => act(async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
});

const UPLOAD_RESPONSE = {
  session_id: "session-1",
  thread_id: "thread-1",
  datasets: [
    {
      dataset_id: "dataset-1",
      filename: "regional_sales.csv",
      columns: ["region", "amount"],
      rows: [{ region: "north", amount: 10 }, { region: "south", amount: 20 }],
      total_rows: 2,
    },
  ],
};

/** Opens the modal with one CSV and stops while the upload is in flight (step 1). */
async function openWithFile() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  const file = new File(["region,amount\nnorth,10\nsouth,20\n"], "regional_sales.csv", { type: "text/csv" });
  act(() => root!.render(<UploadModal open onOpenChange={() => {}} initialFiles={[file]} />));
  await flush();
  expect(upload.resolve, "upload request issued").not.toBeNull();
  // The dialog renders in a portal on document.body.
  return document.body;
}

async function finishUpload() {
  await act(async () => upload.resolve!(UPLOAD_RESPONSE));
  await flush();
}

const buttonByText = (scope: ParentNode, text: string) =>
  [...scope.querySelectorAll("button")].find((b) => b.textContent?.replace(/\s+/g, " ").trim() === text);

describe("Bugs-1.6 upload drop zone", () => {
  it("BG18: the drop zone has a solid primary 'Browse files' button inside it", () => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    act(() => root!.render(<UploadDropzone />));
    const browse = buttonByText(host, "Browse files");
    expect(browse, "Browse files button").toBeDefined();
    // Primary (solid brand) styling, not a link or ghost button.
    expect(browse!.className).toContain("bg-brand-solid");
    expect(browse!.className).toContain("text-white");
    // It sits inside the dashed drop target itself.
    expect(browse!.closest(".border-dashed")).not.toBeNull();
  });
});

describe("Bugs-1.6 Select File & Start Analysis modal", () => {
  it("BG26: the modal is a two-step flow, upload first and review second", async () => {
    const body = await openWithFile();
    expect(body.querySelector('[aria-label="Step 1 of 2"]')).not.toBeNull();
    expect(body.textContent).toContain("or drag and drop file");
    // Review content (profiling, issues, cleaning) is not on the upload step.
    expect(body.querySelector("h2")?.textContent ?? "").not.toContain("AI Insights");

    await finishUpload();
    expect(body.querySelector('[aria-label="Step 2 of 2"]')).not.toBeNull();
    expect(body.querySelector('[data-upload-step="2"]')).not.toBeNull();
  });

  it("BG20: after the file reaches 100% the drop zone is gone and a compact status bar offers 'Replace file'", async () => {
    const body = await openWithFile();
    expect(body.textContent).toContain("or drag and drop file");
    await finishUpload();
    expect(body.textContent).not.toContain("or drag and drop file");
    const replace = buttonByText(body, "Replace file");
    expect(replace, "Replace file action").toBeDefined();
    // The status bar names the file and says it is ready.
    const bar = replace!.parentElement!;
    expect(bar.textContent).toContain("regional_sales.csv");
    expect(bar.textContent).toContain("Ready");
  });

  it("BG21: Skip and Clean & Start sit in a footer pinned to the bottom of the modal", async () => {
    const body = await openWithFile();
    await finishUpload();
    const skip = buttonByText(body, "Skip");
    const clean = buttonByText(body, "Clean & Start");
    expect(skip, "Skip").toBeDefined();
    expect(clean, "Clean & Start").toBeDefined();
    for (const action of [skip!, clean!]) {
      const footer = action.closest("footer");
      expect(footer, `${action.textContent} inside a footer`).not.toBeNull();
      const classes = footer!.className.split(/\s+/);
      expect(classes).toEqual(expect.arrayContaining(["sticky", "bottom-0", "shrink-0"]));
      // The footer is outside the scrolling body, so it never scrolls away.
      expect(footer!.closest(".overflow-y-auto")).toBeNull();
    }
  });

  it("BG23: the review step heading reads 'AI Insights', never 'Ai Insights'", async () => {
    const body = await openWithFile();
    await finishUpload();
    const headings = [...body.querySelectorAll("h2")].map((h) => h.textContent?.trim());
    expect(headings).toContain("AI Insights");
    expect(body.textContent).not.toMatch(/\bAi Insights\b/);
  });
});
