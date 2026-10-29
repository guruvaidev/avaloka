// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CloudStorageBrowserModal } from "../../src/components/database/CloudStorageBrowserModal";
import { BucketFolderName } from "../../src/components/database/BucketFolderName";

const mocks = vi.hoisted(() => ({ list: vi.fn(), navigate: vi.fn() }));
vi.mock("@tanstack/react-router", () => ({ useNavigate: () => mocks.navigate }));
vi.mock("../../src/lib/api/backendApi", () => ({ backendApi: { listBucketObjects: mocks.list, getUploadLimits: vi.fn(async () => ({})) } }));
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));

let root: Root;
const connection = { id: "fixture-connection", name: "test_autoinsight", provider: "gcp" as const, bucket_name: "fixture-bucket" };
const csvFolder = { name: "4936ddf2-7832-49de-9ef3-2eff957b3761.csv", prefix: "4936ddf2-7832-49de-9ef3-2eff957b3761.csv/" };
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  mocks.list.mockReset();
  mocks.list.mockResolvedValue({ bucket: "fixture-bucket", prefix: "", folders: [csvFolder, { name: "exports", prefix: "exports/" }], objects: [{ key: "report.csv", size: "1024" }, { key: "report.json", size: "2048" }] });
});
afterEach(() => { act(() => root.unmount()); document.body.innerHTML = ""; });

async function renderPicker() {
  await act(async () => root.render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><CloudStorageBrowserModal open connection={connection} onOpenChange={() => undefined} /></QueryClientProvider>));
}

describe("folder navigation and selection", () => {
  it("clicking a folder name opens the original backend prefix", async () => {
    const open = vi.fn();
    await act(async () => root.render(<BucketFolderName folder={csvFolder} onOpen={open} />));
    fireEvent.click(screen.getByRole("button", { name: /4936ddf2/ }));
    expect(open).toHaveBeenCalledWith(csvFolder.prefix);
  });
  it("the name and Open action both load the same folder through the existing listing call", async () => {
    await renderPicker();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: /4936ddf2/ })));
    expect(mocks.list).toHaveBeenLastCalledWith({ backend: "gcs", bucket: "fixture-bucket", prefix: csvFolder.prefix, connectionId: connection.id });
    await act(async () => fireEvent.click(screen.getAllByRole("button", { name: "Open" })[0]));
    expect(mocks.list).toHaveBeenLastCalledWith({ backend: "gcs", bucket: "fixture-bucket", prefix: csvFolder.prefix, connectionId: connection.id });
  });
  it("double-clicking a folder row opens it but double-clicking its checkbox never navigates", async () => {
    await renderPicker();
    const checkbox = screen.getByRole("checkbox", { name: `Select folder ${csvFolder.name}` });
    const before = mocks.list.mock.calls.length;
    await act(async () => fireEvent.doubleClick(checkbox));
    expect(mocks.list.mock.calls.length).toBe(before);
    const row = checkbox.closest("tr");
    if (!row) throw new Error("Missing folder row");
    await act(async () => fireEvent.doubleClick(row));
    expect(mocks.list).toHaveBeenLastCalledWith(expect.objectContaining({ prefix: csvFolder.prefix }));
  });
  it("checkbox selection does not navigate and updates folder/file actions while preserving exclusive selection", async () => {
    await renderPicker();
    expect((screen.getByRole("button", { name: "Use files" }) as HTMLButtonElement).disabled).toBe(true);
    const before = mocks.list.mock.calls.length;
    await act(async () => fireEvent.click(screen.getByRole("checkbox", { name: `Select folder ${csvFolder.name}` })));
    expect(mocks.list.mock.calls.length).toBe(before);
    expect(screen.getByText("1 folder selected")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Use folder" }) as HTMLButtonElement).disabled).toBe(false);
    await act(async () => fireEvent.click(screen.getByRole("checkbox", { name: "Select folder exports" })));
    expect(screen.getByText("2 folders selected")).toBeTruthy();
    await act(async () => fireEvent.click(screen.getByRole("checkbox", { name: "Select report.csv" })));
    expect(screen.getByText("1 file selected")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Use file" })).toBeTruthy();
    await act(async () => fireEvent.click(screen.getByRole("checkbox", { name: "Select report.json" })));
    expect(screen.getByText("2 files selected")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Use files" })).toBeTruthy();
  });
});