// Renders the real DynamicChart with real Recharts. jsdom has no layout, so
// ResponsiveContainer would measure 0x0 and draw nothing; the mock below hands
// the chart a fixed size and leaves every other Recharts export untouched.
import { render, screen } from "@testing-library/react";
import { cloneElement, type ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  DynamicChart,
  normalizeChart,
  normalizeVizConfig,
} from "@/components/dashboard/dynamicChart";
import { deriveVizFromRows } from "@/lib/derive-viz";

vi.mock("recharts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("recharts")>()),
  ResponsiveContainer: ({ children }: { children: ReactElement }) =>
    cloneElement(children, { width: 640, height: 320 } as object),
}));

/** Text of every axis title Recharts drew (tick labels excluded). */
function axisTitles(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll("text.recharts-label")).map(
    (node) => node.textContent ?? "",
  );
}

const ROWS = [
  { region: "North", revenue: 10 },
  { region: "South", revenue: 40 },
  { region: "East", revenue: 25 },
];

describe("DynamicChart", () => {
  it("draws one bar per data point", () => {
    const [slide] = normalizeVizConfig(deriveVizFromRows(ROWS));
    const { container } = render(<DynamicChart slide={slide} />);
    expect(container.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(3);
    expect(screen.getByText("South")).toBeInTheDocument();
  });

  it("draws a pie legend with each slice's share", () => {
    const slide = normalizeChart({ type: "pie", labels: ["Open", "Closed"], values: [3, 1] })!;
    render(<DynamicChart slide={slide} />);
    expect(screen.getByText("3 (75.0%)")).toBeInTheDocument();
    expect(screen.getByText("1 (25.0%)")).toBeInTheDocument();
  });

  it("says so, instead of drawing an empty frame, when a chart has no data", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const slide = normalizeChart({ type: "bar", title: "Nothing to plot", insights: ["Kept."] })!;
    expect(slide.insights).toEqual(["Kept."]);
    const { container } = render(<DynamicChart slide={slide} />);
    expect(
      screen.getByText("This chart couldn't be drawn from the available data."),
    ).toBeInTheDocument();
    expect(container.querySelector("svg")).toBeNull();
  });

  it("omits axis titles in compact mode", () => {
    const [slide] = normalizeVizConfig(deriveVizFromRows(ROWS));
    const { container } = render(<DynamicChart slide={slide} compact />);
    expect(axisTitles(container)).toEqual([]);
  });

  describe("axis titles", () => {
    it("shows the titles derive-viz chose for a client-derived bar chart", () => {
      const [slide] = normalizeVizConfig(deriveVizFromRows(ROWS));
      const { container } = render(<DynamicChart slide={slide} />);
      expect(axisTitles(container)).toEqual(["Region", "Total revenue"]);
    });

    it("shows the titles derive-viz chose for a client-derived line chart", () => {
      const [slide] = normalizeVizConfig(
        deriveVizFromRows([
          { order_date: "2024-01-01", total: 10 },
          { order_date: "2024-02-01", total: 50 },
        ]),
      );
      expect(slide.type).toBe("line");
      const { container } = render(<DynamicChart slide={slide} />);
      expect(axisTitles(container)).toEqual(["Order date", "Average total"]);
    });

    it("titles an encoded chart with readable names, not raw keys behind an X:/Y: prefix", () => {
      const slide = normalizeChart(
        {
          type: "bar",
          encodings: {
            x: { field: "sales_region" },
            y: { field: "net_revenue", aggregate: "sum" },
          },
        },
        [
          { sales_region: "North", net_revenue: 10 },
          { sales_region: "South", net_revenue: 40 },
        ],
      )!;
      const { container } = render(<DynamicChart slide={slide} />);
      expect(axisTitles(container)).toEqual(["Sales region", "Total net revenue"]);
    });
  });
});
