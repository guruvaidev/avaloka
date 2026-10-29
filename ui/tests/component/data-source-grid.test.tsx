// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

// The connection modals need a router/query-client context; the tile grid does not.
vi.mock("@/components/database/CloudStorageBrowserModal", () => ({ CloudStorageBrowserModal: () => null }));
vi.mock("@/components/database/CloudStorageConnectionsListModal", () => ({ CloudStorageConnectionsListModal: () => null }));
vi.mock("@/components/database/DatabaseConnectionsListModal", () => ({ DatabaseConnectionsListModal: () => null }));

import { DataSourceGrid } from "../../src/components/dashboard/DataSourceGrid";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

function mount() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<DataSourceGrid />));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const ALL_NAMES = [
  "MySQL",
  "PostgreSQL",
  "SQL Server",
  "MariaDB",
  "MongoDB",
  "Google Cloud Storage",
  "Amazon S3",
  "Azure Blob Storage",
];

describe("DataSourceGrid connector tiles", () => {
  it("every tile renders a logo img with alt text and a visible name", () => {
    const el = mount();
    // Expand so the cloud storage tiles are shown too.
    const viewMore = [...el.querySelectorAll("button")].find((b) => b.textContent?.includes("View More"));
    expect(viewMore).toBeDefined();
    act(() => {
      viewMore!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    for (const name of ALL_NAMES) {
      const img = el.querySelector(`img[alt="${name}"]`);
      expect(img, `logo img for ${name}`).not.toBeNull();
      expect(img!.getAttribute("src"), `logo src for ${name}`).toBeTruthy();
      const tile = img!.closest("button");
      expect(tile, `tile for ${name}`).not.toBeNull();
      expect(tile!.textContent, `visible name for ${name}`).toContain(name);
    }
  });

  it("all tiles share the same fixed size classes", () => {
    const el = mount();
    const viewMore = [...el.querySelectorAll("button")].find((b) => b.textContent?.includes("View More"));
    act(() => {
      viewMore!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    const tiles = ALL_NAMES.map((name) => el.querySelector(`img[alt="${name}"]`)!.closest("button")!);
    for (const tile of tiles) {
      expect(tile.className).toContain("h-[96px]");
      expect(tile.className).toContain("bg-card");
      expect(tile.className).toContain("border-border");
    }
  });

  it("logos sit on a light badge in dark mode (no CSS filter inversion)", () => {
    const el = mount();
    const img = el.querySelector('img[alt="MySQL"]')!;
    const badge = img.parentElement!;
    expect(badge.className).toContain("dark:bg-white");
    expect(badge.className).toContain("rounded-md");
    expect(img.className).not.toContain("invert");
    expect(img.style.filter).toBe("");
  });
});
