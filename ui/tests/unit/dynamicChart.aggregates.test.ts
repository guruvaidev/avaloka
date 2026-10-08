// The aggregate a chart spec asks for must be the aggregate that gets plotted.
//
// A bar/line spec with `encodings.y = { field, aggregate }` and no precomputed
// points is aggregated client-side from the preview rows (buildFromEncodings in
// dynamicChart.tsx). The fixture is skewed on purpose: every aggregate yields a
// different number per group, so plotting the wrong one cannot pass by accident.
import { describe, expect, it, vi } from "vitest";
import { normalizeChart } from "@/components/dashboard/dynamicChart";

const SAMPLES = [
  { region: "North", revenue: 1 },
  { region: "North", revenue: 2 },
  { region: "North", revenue: 100 },
  { region: "South", revenue: 10 },
  { region: "South", revenue: 20 },
  { region: "South", revenue: 90 },
];

//                          North      South
const EXPECTED: Record<string, [number, number]> = {
  sum: [103, 120],
  mean: [34.3333, 40],
  avg: [34.3333, 40],
  average: [34.3333, 40],
  median: [2, 20],
  min: [1, 10],
  max: [100, 90],
};

function spec(aggregate?: string) {
  return {
    id: "c1",
    type: "bar",
    title: "Revenue by region",
    encodings: {
      x: { field: "region", type: "nominal" },
      y: { field: "revenue", type: "quantitative", ...(aggregate ? { aggregate } : {}) },
    },
  };
}

/** Plotted value per region, whatever key the builder stored it under. */
function plotted(aggregate?: string): Record<string, number> {
  const slide = normalizeChart(spec(aggregate), SAMPLES);
  expect(slide, "chart was dropped").not.toBeNull();
  expect(slide!.undrawable, "chart was marked undrawable").toBeFalsy();
  const valueKey = slide!.series[0].dataKey;
  return Object.fromEntries(slide!.data.map((row) => [String(row[slide!.xKey]), row[valueKey]]));
}

describe("normalizeChart: client-side aggregation from encodings", () => {
  it.each(Object.entries(EXPECTED))(
    "aggregate %s plots that aggregate",
    (aggregate, [north, south]) => {
      expect(plotted(aggregate)).toEqual({ North: north, South: south });
    },
  );

  it("is case-insensitive about the aggregate name", () => {
    expect(plotted("SUM")).toEqual({ North: 103, South: 120 });
  });

  it("defaults to the mean when the spec names no aggregate", () => {
    expect(plotted()).toEqual({ North: 34.3333, South: 40 });
  });

  it("counts rows per category for aggregate count", () => {
    const slide = normalizeChart(
      { type: "bar", encodings: { x: { field: "region" }, y: { aggregate: "count" } } },
      [...SAMPLES, { region: "North", revenue: 5 }],
    );
    expect(slide!.data).toEqual([
      { category: "North", value: 4 },
      { category: "South", value: 3 },
    ]);
  });

  // An aggregate the builder does not implement must never be drawn as if it
  // were a mean: the chart would carry the requested label and the wrong
  // numbers. Dropping it or marking it undrawable are both acceptable.
  it.each(["p95", "stddev", "variance", "distinct"])(
    "does not plot unsupported aggregate %s as a silent mean",
    (aggregate) => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      const slide = normalizeChart(spec(aggregate), SAMPLES);
      const drawn = slide && !slide.undrawable ? slide.data : [];
      expect(drawn).toEqual([]);
    },
  );
});
