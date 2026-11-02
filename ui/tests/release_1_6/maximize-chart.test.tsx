// @vitest-environment happy-dom
// UI-B29 (UI Bugs): a chart in the analysis workspace can be maximized: its
// "Maximize chart" control opens the same chart in a large dialog.
// Renders ChartCarousel, moved verbatim from routes/analysis.tsx.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { fireEvent, screen, waitFor } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { normalizeChart, type Slide } from "@/components/dashboard/dynamicChart";

vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));

import { ChartCarousel } from "@/components/analysis/ChartCarousel";

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

const slide: Slide = {
  ...normalizeChart({ type: "bar", title: "Sales by region", labels: ["north", "south"], values: [30, 20] })!,
  subtitle: "North leads",
};

describe("UI-B29 maximize chart", () => {
  it("UI-B29: the Maximize control opens the chart in a large dialog titled with the chart", async () => {
    await act(async () => root.render(<ChartCarousel items={[{ slide, index: 0, cardKey: "auto::sales::0" }]} />));
    expect(screen.queryByRole("dialog")).toBeNull();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Maximize Sales by region chart" })));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeNull());
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Sales by region");
    expect(dialog.textContent).toContain("North leads");
  });
});
