// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

// Charts are irrelevant to theming; recharts' CJS build loads a second React copy under Vitest.
vi.mock("@/components/dashboard/dynamicChart", () => ({
  DynamicChart: () => null,
  normalizeVizConfig: () => [] as never[],
}));

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

describe("Auto Insights theming", () => {
  it("root has no forced-light class, attribute or color-scheme", () => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    act(() => {
      root!.render(<AutoInsightsModal open={true} onOpenChange={() => {}} config={{ slides: [] } as never} datasetName="Dummy_Sales_Data_2" />);
    });

    const aiRoot = host.querySelector(".ai-modal");
    expect(aiRoot).not.toBeNull();

    const suspects = [aiRoot as Element, ...aiRoot!.querySelectorAll("*")];
    for (const el of suspects) {
      const classes = (el.getAttribute("class") ?? "").split(/\s+/);
      expect(classes).not.toContain("light");
      expect(classes).not.toContain("theme-light");
      expect(el.getAttribute("data-theme")).not.toBe("light");
      expect(el.getAttribute("style") ?? "").not.toMatch(/color-scheme:\s*light/);
    }
  });

  it("root uses theme variables, not a fixed light background", () => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    act(() => {
      root!.render(<AutoInsightsModal open={true} onOpenChange={() => {}} config={{ slides: [] } as never} datasetName="Dummy_Sales_Data_2" />);
    });
    const aiRoot = host.querySelector(".ai-modal") as HTMLElement;
    expect(aiRoot).not.toBeNull();
    // The ai-modal rule reads var(--color-ai-canvas) / var(--color-ai-text),
    // which have dark overrides — no inline fixed colour may be present.
    expect(aiRoot.style.background).toBe("");
    expect(aiRoot.style.color).toBe("");
  });
});
