// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

vi.mock("@/components/dashboard/dynamicChart", () => ({
  DynamicChart: () => null,
  normalizeVizConfig: () => [] as never[],
}));

// Column statistics uses a react-aria Checkbox that loads a second React copy under Vitest.
vi.mock("@/components/dashboard/auto-insights/AutoInsightsPanels", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  ColumnStatisticsTab: () => null,
}));

// AutoInsightsModal reads the plan (Auto Insights is on every plan); no QueryClient here.
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "free" }, isLoading: false }) }));
import { AutoInsightsModal } from "../../src/components/dashboard/AutoInsightsModal";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

function mount() {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => {
    root!.render(<AutoInsightsModal open={true} onOpenChange={() => {}} config={{ slides: [] } as never} datasetName="Sales" />);
  });
  return host;
}
const panel = () => host!.querySelector('aside[aria-label="Insights Assistant"]');
const avatarBtn = () => host!.querySelector('button[aria-label="Open Insights Assistant"]') as HTMLButtonElement | null;
const tab = (label: string) => Array.from(host!.querySelectorAll("button")).find((b) => b.textContent === label)!;
const click = (el: Element) => act(() => { (el as HTMLElement).click(); });
const exactlyOne = () => expect(Number(!!panel()) + Number(!!avatarBtn())).toBe(1);

describe("Insights Assistant panel", () => {
  it("has no hard-coded white background or forced-light class", () => {
    mount();
    const root = panel()!;
    expect(root).not.toBeNull();
    expect(root.className).toContain("bg-card");
    for (const el of [root, ...root.querySelectorAll("*")]) {
      const cls = el.getAttribute("class") ?? "";
      expect(cls).not.toMatch(/\bbg-white\b|#fff\b|text-gray-|bg-gray-|\blight\b/);
      expect(el.getAttribute("data-theme")).not.toBe("light");
    }
  });

  it("Stack opens with the panel open", () => {
    mount();
    expect(panel()).not.toBeNull();
    expect(avatarBtn()).toBeNull();
  });

  it("Overview and Column statistics open with the panel closed and the avatar shown", () => {
    mount();
    click(tab("Overview"));
    expect(panel()).toBeNull();
    expect(avatarBtn()).not.toBeNull();
    expect(avatarBtn()!.title).toBe("Ask about these charts");
    click(tab("Column statistics"));
    expect(panel()).toBeNull();
    exactlyOne();
  });

  it("closing shows the avatar, clicking the avatar reopens, and Stack stays closed after a dismissal", () => {
    mount();
    click(host!.querySelector('button[aria-label="Close assistant"]')!);
    expect(panel()).toBeNull();
    exactlyOne();
    click(tab("Overview"));
    click(tab("Stack"));
    expect(panel()).toBeNull();
    click(avatarBtn()!);
    expect(panel()).not.toBeNull();
    exactlyOne();
  });
});

describe("Overview section order", () => {
  it("renders Dataset overview before Charts, with Charts last", () => {
    const src = readFileSync(resolve(__dirname, "../../src/components/dashboard/AutoInsightsModal.tsx"), "utf8");
    const content = src.slice(src.indexOf('<div className="ai-overview-content">'));
    const overview = content.indexOf("<DatasetOverview ");
    const charts = content.indexOf("ai-overview-charts-heading");
    expect(overview).toBeGreaterThan(-1);
    expect(overview).toBeLessThan(charts);
    expect(content.indexOf("<DatasetOverview ", charts)).toBe(-1);
  });
});
