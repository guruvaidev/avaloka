// @vitest-environment happy-dom
// 1.6 release suite: Bugs-1.6 BG39 (data view rows per page).
// See docs/test-reports/1.6-traceability.csv.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { DataGrid, PAGE_SIZE_KEY } from "../../src/components/dashboard/data-view/DataGrid";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

const columns = ["record_id", "region", "amount"];
const rows = Array.from({ length: 100 }, (_, i) => ({ record_id: i + 1, region: i % 2 ? "north" : "south", amount: (i + 1) * 10 }));

function mount() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<DataGrid rows={rows} columns={columns} />));
  return host;
}

beforeEach(() => {
  try { localStorage.removeItem(PAGE_SIZE_KEY); } catch { /* no storage */ }
});

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

describe("Bugs-1.6 BG39 data view density", () => {
  it("BG39: data grid defaults to 25 rows per page with a 10/25/50/100 page-size selector", () => {
    const el = mount();
    expect(el.querySelectorAll("tbody tr").length).toBe(25);
    expect(el.textContent).toContain("Showing 1–25 of 100");

    const select = el.querySelector('select[aria-label="Rows per page"]') as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    expect([...select!.options].map((o) => Number(o.value))).toEqual([10, 25, 50, 100]);
    expect(Number(select!.value)).toBe(25);
  });

  it("BG39: choosing 50 rows per page shows 50 rows", () => {
    const el = mount();
    const select = el.querySelector('select[aria-label="Rows per page"]') as HTMLSelectElement;
    act(() => {
      select.value = "50";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(el.querySelectorAll("tbody tr").length).toBe(50);
  });
});
