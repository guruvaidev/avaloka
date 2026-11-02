// @vitest-environment happy-dom
// UI-B02 (UI Bugs): "+ Add" below the chat adds a dataset to the thread.
// Two halves: the Add dialog uploads into THIS thread and reports the new
// datasets back, and the chips under the chat merge the thread's dataset list
// and select the newly added dataset so the next message includes it.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mergeThreadDatasetChips, nextThreadDatasetSelection, type DatasetChip } from "@/lib/analysis-workspace";

const mocks = vi.hoisted(() => ({ uploadFiles: vi.fn(), attach: vi.fn() }));
vi.mock("@/lib/api/backendApi", () => ({
  backendApi: { uploadFiles: mocks.uploadFiles, attachAnalysesToThread: mocks.attach },
}));
vi.mock("@/lib/api/data-set-cards", () => ({ dataSetCardsKey: ["data-set-cards"], fetchDataSetCards: vi.fn(async () => []) }));
vi.mock("@/integrations/supabase/client", async () => (await import("./support/w4a-supabase")).supabaseClientMock());
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), warning: vi.fn(), success: vi.fn() } }));

import { AttachDatasetsModal } from "@/components/dashboard/AttachDatasetsModal";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  mocks.uploadFiles.mockReset();
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

describe("UI-B02 + Add dialog", () => {
  it("UI-B02: uploading a file in the Add dialog sends it to the current thread and hands the new dataset back", async () => {
    const uploaded = { dataset_id: "ds-new", filename: "returns.csv" };
    mocks.uploadFiles.mockResolvedValue({ datasets: [uploaded] });
    const onAttached = vi.fn(async () => undefined);
    const onOpenChange = vi.fn();
    await act(async () =>
      root.render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <AttachDatasetsModal
            open
            onOpenChange={onOpenChange}
            threadId="thread-1"
            currentAnalysisId="analysis-1"
            chipDatasetIds={["ds-existing"]}
            onAttached={onAttached}
          />
        </QueryClientProvider>,
      ),
    );
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Upload new" })));
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["a,b\n1,2\n"], "returns.csv", { type: "text/csv" });
    await act(async () => fireEvent.change(input, { target: { files: [file] } }));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Upload and add" })));

    expect(mocks.uploadFiles).toHaveBeenCalledTimes(1);
    const [files, , , , threadArg] = mocks.uploadFiles.mock.calls[0];
    expect((files as File[]).map((f) => f.name)).toEqual(["returns.csv"]);
    expect(threadArg).toBe("thread-1");
    expect(onAttached).toHaveBeenCalledWith({ uploadedDatasets: [uploaded] });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});

describe("UI-B02 chips after + Add", () => {
  const existing: DatasetChip = { id: "ds-existing", name: "sales.csv", available: true, threadId: "thread-1" };

  it("UI-B02: the refreshed thread list adds the new dataset as a chip and keeps the existing one", () => {
    const { merged, newIds } = mergeThreadDatasetChips([existing], [
      { dataset_id: "ds-existing", filename: "sales.csv", available: true },
      { dataset_id: "ds-new", filename: "returns.csv", available: true },
    ], "thread-1");
    expect(merged.map((c) => c.id)).toEqual(["ds-existing", "ds-new"]);
    expect(merged[1]).toMatchObject({ name: "returns.csv", threadId: "thread-1", available: true });
    expect(newIds).toEqual(["ds-new"]);
  });

  it("UI-B02: the newly added dataset is selected, so the next message includes it", () => {
    const { newIds, unavailable, serverSelected } = mergeThreadDatasetChips([existing], [
      { dataset_id: "ds-existing", filename: "sales.csv" },
      { dataset_id: "ds-new", filename: "returns.csv" },
    ], "thread-1");
    expect(nextThreadDatasetSelection(["ds-existing"], serverSelected, unavailable, newIds, true)).toEqual([
      "ds-existing",
      "ds-new",
    ]);
  });
});
