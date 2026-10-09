import { describe, expect, it } from "vitest";

import type { Slide } from "@/components/dashboard/dynamicChart";
import {
  getLineRenderData,
  getPieRenderData,
  getRenderedPoints,
} from "@/lib/chart-points";

function makeSlide(overrides: Partial<Slide>): Slide {
  return {
    title: "Test chart",
    type: "bar",
    data: [],
    xKey: "x",
    series: [{ dataKey: "y", name: "Value" }],
    insights: [],
    ...overrides,
  };
}

describe("getLineRenderData", () => {
  it("sorts ISO dates and averages duplicate days", () => {
    const slide = makeSlide({
      type: "line",
      xKey: "date",
      series: [{ dataKey: "revenue" }],
      data: [
        { date: "2024-01-03", revenue: 30 },
        { date: "2024-01-01", revenue: 10 },
        { date: "2024-01-01", revenue: 20 },
      ],
    });

    expect(getLineRenderData(slide).data).toEqual([
      { date: "1 Jan", revenue: 15 },
      { date: "3 Jan", revenue: 30 },
    ]);
  });

  it("groups long date ranges by month", () => {
    const slide = makeSlide({
      type: "area",
      xKey: "date",
      series: [{ dataKey: "amount" }],
      data: [
        { date: "2024-01-01", amount: 10 },
        { date: "2024-01-20", amount: 20 },
        { date: "2024-06-01", amount: 30 },
      ],
    });

    expect(getLineRenderData(slide).data).toEqual([
      { date: "Jan 2024", amount: 15 },
      { date: "Jun 2024", amount: 30 },
    ]);
  });

  it("does not treat ID-like values as dates", () => {
    const rows = [
      { label: "sku-1", value: 10 },
      { label: "sku-2", value: 20 },
    ];

    const slide = makeSlide({
      type: "line",
      xKey: "label",
      data: rows,
    });

    expect(getLineRenderData(slide).data).toBe(rows);
  });

  it("does not transform non-line charts", () => {
    const rows = [{ date: "2024-01-01", value: 10 }];
    const slide = makeSlide({ type: "bar", data: rows });

    expect(getLineRenderData(slide).data).toBe(rows);
  });
});

describe("getPieRenderData", () => {
  it("groups categories beyond ten slices into Other", () => {
    const slide = makeSlide({
      type: "pie",
      xKey: "category",
      series: [{ dataKey: "amount" }],
      data: Array.from({ length: 12 }, (_, index) => ({
        category: `C${index + 1}`,
        amount: index + 1,
      })),
    });

    expect(getPieRenderData(slide)).toEqual([
      { name: "C12", value: 12 },
      { name: "C11", value: 11 },
      { name: "C10", value: 10 },
      { name: "C9", value: 9 },
      { name: "C8", value: 8 },
      { name: "C7", value: 7 },
      { name: "C6", value: 6 },
      { name: "C5", value: 5 },
      { name: "C4", value: 4 },
      { name: "Other", value: 6 },
    ]);
  });

  it("drops missing labels and nonnumeric values", () => {
    const slide = makeSlide({
      type: "donut",
      xKey: "category",
      series: [{ dataKey: "amount" }],
      data: [
        { category: "A", amount: 2 },
        { category: "B", amount: "3" },
        { category: "C", amount: "invalid" },
        { category: null, amount: 4 },
      ],
    });

    expect(getPieRenderData(slide)).toEqual([
      { name: "A", value: 2 },
      { name: "B", value: 3 },
    ]);
  });
});

describe("getRenderedPoints", () => {
  it("preserves backend histogram range labels", () => {
    const slide = makeSlide({
      type: "bar",
      chartKind: "histogram",
      xKey: "bin",
      series: [{ dataKey: "count" }],
      data: [
        { bin: "0–10", count: 4 },
        { bin: "10–20", count: 7 },
      ],
    });

    expect(getRenderedPoints(slide)).toEqual([
      { x: "0–10", y: 4 },
      { x: "10–20", y: 7 },
    ]);
  });

  it("constructs ranges from numeric histogram starts", () => {
    const slide = makeSlide({
      type: "bar",
      chartKind: "histogram",
      xKey: "start",
      series: [{ dataKey: "count" }],
      data: [
        { start: 0, count: 2 },
        { start: 5, count: 4 },
        { start: 10, count: 1 },
      ],
    });

    expect(getRenderedPoints(slide)).toEqual([
      { x: "0.00–5.00", y: 2 },
      { x: "5.00–10.00", y: 4 },
      { x: "10.00–15.00", y: 1 },
    ]);
  });

  it("limits avatar chart context to sixty points", () => {
    const slide = makeSlide({
      data: Array.from({ length: 75 }, (_, index) => ({
        x: index,
        y: index * 2,
      })),
    });

    const points = getRenderedPoints(slide);

    expect(points).toHaveLength(60);
    expect(points.at(-1)).toEqual({ x: 59, y: 118 });
  });
});