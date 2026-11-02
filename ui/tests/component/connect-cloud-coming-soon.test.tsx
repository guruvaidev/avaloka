// @vitest-environment happy-dom
//
// Cloud Connect database tiles must not fake authentication.
//
// Every database tile except BigQuery used to "authenticate" on a 900 ms timer:
// any non-empty password without "fail" moved the dialog to the "Select Data"
// step, with mock files, and no backend was ever called. Tiles with no real
// connector now render "Coming soon" with no credential form, and the only way
// to reach "Select Data" is a resolved backend call.
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  createGcp: vi.fn(),
  finalizeGcp: vi.fn(),
  dsCreate: vi.fn(),
  insertFiles: vi.fn(),
}));
vi.mock("@/lib/api/backendApi", () => ({ backendApi: { listBucketObjects: mocks.list } }));
vi.mock("@/lib/gcp-connection", () => ({
  buildGcsDatasetPath: (bucket: string, folder: string) => `gs://${bucket}/${folder}`,
  parseGcsDatasetPath: (raw: string) => {
    const [bucket, ...rest] = raw.replace("gs://", "").split("/");
    return { bucket, prefix: rest.join("/") };
  },
  createGcpCloudConnection: mocks.createGcp,
  finalizeGcpCloudConnection: mocks.finalizeGcp,
}));
vi.mock("@/lib/data-sources", () => ({
  useDataSourceMutations: () => ({
    create: { mutateAsync: mocks.dsCreate, isPending: false },
    insertFiles: { mutateAsync: mocks.insertFiles, isPending: false },
  }),
}));
// The storage-tile modals talk to Supabase directly; they are not under test.
vi.mock("@/components/database/CloudStorageConnectionsListModal", () => ({ CloudStorageConnectionsListModal: () => null }));
vi.mock("@/components/database/CloudStorageBrowserModal", () => ({ CloudStorageBrowserModal: () => null }));
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));

import { ConnectCloudModal } from "../../src/components/database/ConnectCloudModal";
import { PROVIDERS } from "../../src/components/database/providers";

const never = () => new Promise<never>(() => undefined);
const wait = (ms: number) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });

beforeEach(() => {
  for (const m of Object.values(mocks)) m.mockReset();
  mocks.list.mockImplementation(never);
  mocks.createGcp.mockImplementation(never);
});

function renderModal() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={qc}><ConnectCloudModal open onOpenChange={() => undefined} /></QueryClientProvider>);
}

function tile(name: string): HTMLButtonElement {
  const img = screen.getByAltText(name);
  const btn = img.closest("button");
  if (!btn) throw new Error(`no tile button for ${name}`);
  return btn as HTMLButtonElement;
}

/** Select a tile, continue, fill whatever credential form appears, press Authenticate. */
async function tryToConnect(name: string) {
  fireEvent.click(tile(name));
  const cont = screen.getByRole("button", { name: /Continue/ }) as HTMLButtonElement;
  if (!cont.disabled) fireEvent.click(cont);
  const dialog = screen.getByRole("dialog");
  for (const input of Array.from(dialog.querySelectorAll<HTMLInputElement>("input"))) {
    if (input.type === "radio") continue;
    if (input.type === "file") {
      const key = new File(['{"type":"service_account"}'], "key.json", { type: "application/json" });
      await act(async () => { fireEvent.change(input, { target: { files: [key] } }); });
    } else {
      fireEvent.change(input, { target: { value: input.type === "number" ? "5432" : "correct-horse-battery" } });
    }
  }
  for (const select of Array.from(dialog.querySelectorAll<HTMLSelectElement>("select"))) {
    fireEvent.change(select, { target: { value: select.options[1]?.value ?? "" } });
  }
  await wait(50);
  const auth = screen.queryByRole("button", { name: /Authenticate/ });
  if (auth) await act(async () => { fireEvent.click(auth); });
  // Longer than the 900 ms timer that used to fake success.
  await wait(1200);
}

const reachedFiles = () => screen.queryByText("Select Data") !== null;

describe("Cloud Connect tiles without a connector", () => {
  it.each(["Snowflake", "Databricks"])("%s renders Coming soon with no password field", async (name) => {
    renderModal();
    expect(tile(name).textContent).toContain("Coming soon");
    fireEvent.click(tile(name));
    const cont = screen.getByRole("button", { name: /Continue/ }) as HTMLButtonElement;
    if (!cont.disabled) fireEvent.click(cont);
    expect(screen.getByRole("dialog").querySelector('input[type="password"]')).toBeNull();
    expect(screen.queryByText("Authenticate & Connect")).toBeNull();
  });
});

describe("no tile reaches Select Data without a resolved backend call", () => {
  it.each(PROVIDERS.map((p) => p.name))("%s", async (name) => {
    renderModal();
    await tryToConnect(name);
    expect(reachedFiles()).toBe(false);
    expect(screen.queryByText("Doc.01")).toBeNull();
  });

  it("Google BigQuery reaches Select Data once the backend lists files (the check can see the step)", async () => {
    mocks.createGcp.mockResolvedValue({ id: "conn-1" });
    mocks.list.mockResolvedValue({ objects: [{ key: "exports/orders.csv", size: "10 KB" }] });
    renderModal();
    await tryToConnect("Google BigQuery");
    expect(mocks.list).toHaveBeenCalledWith(expect.objectContaining({ backend: "gcs", connectionId: "conn-1" }));
    expect(reachedFiles()).toBe(true);
  });

  it("Google BigQuery stays on the credential step when the backend rejects", async () => {
    mocks.createGcp.mockRejectedValue(new Error("403 from backend"));
    renderModal();
    await tryToConnect("Google BigQuery");
    expect(reachedFiles()).toBe(false);
    expect(screen.getByText("403 from backend")).toBeTruthy();
  });
});
