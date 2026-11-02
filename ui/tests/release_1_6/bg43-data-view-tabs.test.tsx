// @vitest-environment happy-dom
// BG43 (Bugs-1.6): the Data View tab that shows the dataset's STRUCTURE
// (columns, types, nulls) must be named "Schema", "Data Dictionary" or
// "Column Summary", not the ambiguous "Data Preview", and must be a separate
// tab from the data table (the rows). Renders DataViewTabs, moved verbatim
// from routes/analysis.tsx. Replaces the tautological check in
// tests/unit/workspace-layout.test.tsx, which passes the tab names in itself.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { fireEvent, screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DataViewTabs } from "@/components/dashboard/data-view/DataViewTabs";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

const STRUCTURAL_NAMES = ["Schema", "Data Dictionary", "Column Summary"];

describe("BG43 data view tab names", () => {
  it("BG43: the structural tab is named Schema / Data Dictionary / Column Summary, never 'Data Preview'", async () => {
    await act(async () => root.render(<DataViewTabs view="sample" resultCount={0} onSelect={() => undefined} />));
    const names = screen.getAllByRole("tab").map((t) => t.textContent?.trim() ?? "");
    const structural = names.filter((n) => STRUCTURAL_NAMES.includes(n));
    expect(structural).toHaveLength(1);
    expect(names.some((n) => /preview/i.test(n))).toBe(false);
  });

  it("BG43: the structural tab is distinct from the data table tab and opens its own view", async () => {
    const onSelect = vi.fn();
    await act(async () => root.render(<DataViewTabs view="sample" resultCount={2} onSelect={onSelect} />));
    const tabs = screen.getAllByRole("tab");
    const structural = tabs.find((t) => STRUCTURAL_NAMES.includes(t.textContent?.trim() ?? ""));
    const table = tabs.find((t) => t.textContent?.trim() === "Data sample");
    expect(structural).toBeDefined();
    expect(table).toBeDefined();
    expect(structural).not.toBe(table);
    // The table tab is the selected one; the structural tab is a different view.
    expect(table!.getAttribute("aria-selected")).toBe("true");
    expect(structural!.getAttribute("aria-selected")).toBe("false");
    expect(structural!.getAttribute("title")).toMatch(/structure/i);
    await act(async () => fireEvent.click(structural!));
    await act(async () => fireEvent.click(table!));
    expect(onSelect.mock.calls.map((c) => c[0])).toEqual(["preview", "sample"]);
  });
});
