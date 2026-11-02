// @vitest-environment happy-dom
// 1.6 release suite: Bugs-1.6 Auto Insights rows (BG02, BG29, BG30).
// Renders the real AutoInsightsModal; only the chart drawing is stubbed
// (recharts' CJS build loads a second React copy under Vitest).
// See docs/test-reports/1.6-traceability.csv.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

vi.mock("@/components/dashboard/dynamicChart", () => ({
  DynamicChart: () => null,
  // Two insight slides, as normalizeVizConfig would produce from a config.
  normalizeVizConfig: (config: unknown) =>
    config
      ? [
          { id: "c1", title: "Average Sales by Region", type: "bar", data: [{ region: "north", amount: 10 }], xKey: "region", series: [{ dataKey: "amount", name: "amount" }], insights: ["North leads."] },
          { id: "c2", title: "Sales Trend by Month", type: "line", data: [{ month: "2024-01", amount: 5 }], xKey: "month", series: [{ dataKey: "amount", name: "amount" }], insights: ["Steady growth."] },
        ]
      : [],
}));

import { AutoInsightsModal } from "../../src/components/dashboard/AutoInsightsModal";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

const CONFIG = {
  dataset: { filename: "regional_sales.csv", total_rows: 1200, columns: ["region", "month", "amount"] },
  charts: [],
};

function mountModal() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<AutoInsightsModal open onOpenChange={() => {}} config={CONFIG} datasetName="regional_sales.csv" />));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const buttons = (scope: ParentNode) => [...scope.querySelectorAll("button")];
const click = (el: Element) => act(() => { el.dispatchEvent(new MouseEvent("click", { bubbles: true })); });

describe("Bugs-1.6 Auto Insights", () => {
  it("BG02: no 'Open Code' control is offered on insight cards when there is no analysis code", () => {
    const el = mountModal();
    const card = el.querySelector("section.ai-chart-card");
    expect(card, "stack-view insight card").not.toBeNull();
    const codeControl = (scope: ParentNode) =>
      buttons(scope).find((b) => /\bcode\b/i.test(`${b.textContent ?? ""} ${b.getAttribute("aria-label") ?? ""}`));
    expect(codeControl(el)).toBeUndefined();

    // Same on the Overview grid.
    const overview = buttons(el).find((b) => b.textContent?.trim() === "Overview");
    expect(overview).toBeDefined();
    click(overview!);
    expect(el.querySelector(".ai-overview-chart-grid")).not.toBeNull();
    expect(codeControl(el)).toBeUndefined();
  });

  it("BG29: the insights header names the dataset with its row and column counts", () => {
    const el = mountModal();
    const header = el.querySelector("header.ai-header");
    expect(header).not.toBeNull();
    expect(header!.querySelector("strong")?.textContent).toBe("regional_sales.csv");
    expect(header!.textContent).toContain("1,200 rows · 3 columns");
  });

  it("BG30: Previous/Next arrows and the 'N of M' indicator live inside the insight card", () => {
    const el = mountModal();
    const card = el.querySelector("section.ai-chart-card") as HTMLElement;
    expect(card).not.toBeNull();
    expect(card.querySelector("h3")?.textContent).toBe("Average Sales by Region");
    const previous = card.querySelector('button[aria-label="Previous insight"]');
    const next = card.querySelector('button[aria-label="Next insight"]');
    expect(previous, "Previous inside the card").not.toBeNull();
    expect(next, "Next inside the card").not.toBeNull();
    expect(card.textContent).toContain("1 of 2");

    // The arrows drive this card.
    click(next!);
    const after = el.querySelector("section.ai-chart-card") as HTMLElement;
    expect(after.querySelector("h3")?.textContent).toBe("Sales Trend by Month");
    expect(after.textContent).toContain("2 of 2");
  });
});
