// @vitest-environment happy-dom
// 1.6 release suite: Auto Insights rows settled from the shipped code
// (Bugs-1.6 BG03, BG04, BG32; UI Bugs UI-B09). Renders the real
// AutoInsightsModal; only chart drawing is stubbed (recharts' CJS build loads
// a second React copy under Vitest), as in bugs16-auto-insights.test.tsx.
// See docs/test-reports/1.6-traceability.csv.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

vi.mock("@/components/dashboard/dynamicChart", () => ({
  DynamicChart: () => null,
  // Two aggregated insight slides, as normalizeVizConfig would produce.
  normalizeVizConfig: (config: unknown) =>
    config
      ? [
          { id: "c1", title: "Total Sales by Region", type: "bar", data: [{ region: "north", amount: 10 }, { region: "south", amount: 7 }], xKey: "region", series: [{ dataKey: "amount", name: "amount" }], insights: ["North leads."] },
          { id: "c2", title: "Sales Trend by Month", type: "line", data: [{ month: "2024-01", amount: 5 }], xKey: "month", series: [{ dataKey: "amount", name: "amount" }], insights: ["Steady growth."] },
        ]
      : [],
}));

import { AutoInsightsModal } from "../../src/components/dashboard/AutoInsightsModal";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

const CONFIG = {
  dataset: { filename: "regional_sales.csv", total_rows: 1200, columns: ["order_id", "region", "month", "amount"] },
  charts: [],
};
// Raw uploaded rows (one per order) -- what the chart aggregates.
const RAW_SAMPLES = [
  { order_id: "A-1", region: "north", month: "2024-01", amount: 6 },
  { order_id: "A-2", region: "north", month: "2024-01", amount: 4 },
  { order_id: "A-3", region: "south", month: "2024-01", amount: 7 },
];

function mount(props: Record<string, unknown> = {}) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<AutoInsightsModal open onOpenChange={() => {}} config={CONFIG} datasetName="regional_sales.csv" {...props} />));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const buttons = (scope: ParentNode) => [...scope.querySelectorAll("button, [role='button']")];
const click = (el: Element) => act(() => { el.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
const label = (b: Element) => `${b.textContent ?? ""} ${b.getAttribute("aria-label") ?? ""}`;

describe("Auto Insights settled rows", () => {
  it("BG03: no dead 'Explore more' control is offered in the stack or overview views", () => {
    const el = mount();
    expect(el.querySelector("section.ai-chart-card"), "stack view rendered").not.toBeNull();
    expect(buttons(el).filter((b) => /explore\s*more/i.test(label(b)))).toEqual([]);
    const overview = buttons(el).find((b) => b.textContent?.trim() === "Overview");
    expect(overview).toBeDefined();
    click(overview!);
    expect(el.querySelector(".ai-overview-chart-grid"), "overview rendered").not.toBeNull();
    expect(buttons(el).filter((b) => /explore\s*more/i.test(label(b)))).toEqual([]);
  });

  it("BG32: the stack view shows one active insight card with a titled navigator, not blank stacked cards", () => {
    const el = mount();
    expect(el.querySelectorAll("section.ai-chart-card")).toHaveLength(1);
    const nav = el.querySelector('nav[aria-label="Insight navigator"]');
    expect(nav).not.toBeNull();
    const titles = [...nav!.querySelectorAll("strong")].map((s) => s.textContent);
    expect(titles).toEqual(["Total Sales by Region", "Sales Trend by Month"]);
  });

  it("UI-B09: the insights assistant (avatar) is available on Auto Insights and every card offers 'Explain this chart'", () => {
    const el = mount();
    const launcher = el.querySelector('button[aria-label="Ask about these charts"]');
    expect(launcher, "avatar launcher").not.toBeNull();
    expect(el.querySelector("section.ai-chart-card [aria-label='Explain this chart']"), "per-card explain").not.toBeNull();
    click(launcher!);
    expect(el.querySelector('aside[aria-label="Insights Assistant"]'), "assistant panel opens").not.toBeNull();
  });

  // BG04 is recorded Merged, but InsightCard's "Data table" prefers the raw
  // `samples` rows over the chart's own aggregated rows whenever samples are
  // passed (ui/src/components/dashboard/AutoInsightsModal.tsx, tableRows).
  it.fails("BG04 [open defect, strict]: an insight's Data table shows the aggregated rows behind its chart, not raw sample rows", () => {
    const el = mount({ samples: RAW_SAMPLES });
    const tableToggle = buttons(el).find((b) => b.textContent?.trim() === "Data table");
    expect(tableToggle).toBeDefined();
    click(tableToggle!);
    const headers = [...el.querySelectorAll("section.ai-chart-card th")].map((th) => th.textContent?.trim()).filter((t) => t && t !== "#");
    expect(headers).toEqual(["region", "amount"]);
    expect(el.querySelector("section.ai-chart-card")!.textContent).not.toContain("A-1");
  });
});
