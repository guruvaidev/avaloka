// @vitest-environment node
//
// The emailed PDFs' charts are now drawn locally instead of being rendered by
// quickchart.io. planChart is pure — config in, drawing primitives out — so the
// geometry, the labelling and the "nothing escapes the box" property are all
// testable without pdf-lib and without a network.
import { describe, expect, it } from "vitest";

import {
  type Box,
  formatValue,
  hexToRgb,
  niceTicks,
  parseConfig,
  planChart,
  type Prim,
  slicePath,
} from "../../supabase/functions/_shared/chart-render.ts";

// Stand-in for font.widthOfTextAtSize: proportional enough to exercise
// centring and ellipsizing.
const measure = (text: string, size: number, bold: boolean) => text.length * size * (bold ? 0.58 : 0.5);

const BOX: Box = { x: 48, y: 400, width: 499, height: 230 };

const barConfig = (values: Array<number | null>, labels = ["a", "b", "c"]) => ({
  type: "bar",
  data: { labels, datasets: [{ label: "revenue", data: values, backgroundColor: "#1565ef" }] },
  options: {
    plugins: { title: { display: true, text: "Revenue by region" }, legend: { display: false } },
    scales: { y: { title: { display: true, text: "revenue" } }, x: { title: { display: true, text: "region" } } },
  },
});

const rects = (p: Prim[]) => p.filter((x) => x.kind === "rect") as Extract<Prim, { kind: "rect" }>[];
const texts = (p: Prim[]) => p.filter((x) => x.kind === "text") as Extract<Prim, { kind: "text" }>[];
const lines = (p: Prim[]) => p.filter((x) => x.kind === "line") as Extract<Prim, { kind: "line" }>[];
const circles = (p: Prim[]) => p.filter((x) => x.kind === "circle") as Extract<Prim, { kind: "circle" }>[];
const paths = (p: Prim[]) => p.filter((x) => x.kind === "path") as Extract<Prim, { kind: "path" }>[];

describe("planChart: bar", () => {
  it("draws one bar per non-null value and skips the nulls", () => {
    const plan = planChart(barConfig([10, null, 30]), BOX, measure);
    // Bars are the only rects in a single-series bar chart (no legend).
    expect(rects(plan)).toHaveLength(2);
  });

  it("scales bar height to the value", () => {
    const plan = planChart(barConfig([10, 20, 40]), BOX, measure);
    const bars = rects(plan).sort((a, b) => a.x - b.x);
    expect(bars).toHaveLength(3);
    // 20 is twice 10, 40 is four times 10 — on a zero-based axis.
    expect(bars[1].h / bars[0].h).toBeCloseTo(2, 1);
    expect(bars[2].h / bars[0].h).toBeCloseTo(4, 1);
  });

  it("puts negative bars below the zero line and positive bars above it", () => {
    const plan = planChart(barConfig([-20, 20]), BOX, measure);
    const bars = rects(plan).sort((a, b) => a.x - b.x);
    expect(bars).toHaveLength(2);
    // The negative bar's top edge is the zero line; the positive bar's bottom is.
    expect(bars[0].y + bars[0].h).toBeCloseTo(bars[1].y, 1);
  });

  it("renders the title, the axis titles and the category labels", () => {
    const plan = planChart(barConfig([1, 2, 3]), BOX, measure);
    const all = texts(plan).map((t) => t.text);
    expect(all).toContain("Revenue by region");
    expect(all).toContain("region");
    expect(all).toContain("revenue");
    expect(all).toContain("a");
    expect(all).toContain("c");
  });

  it("groups multi-series bars side by side and draws a legend", () => {
    const plan = planChart({
      type: "bar",
      data: {
        labels: ["a", "b"],
        datasets: [
          { label: "north", data: [1, 2], backgroundColor: "#1565ef" },
          { label: "south", data: [3, 4], backgroundColor: "#16a34a" },
        ],
      },
      options: { plugins: { title: { display: true, text: "T" }, legend: { display: true } } },
    }, BOX, measure);
    const labels = texts(plan).map((t) => t.text);
    expect(labels).toContain("north");
    expect(labels).toContain("south");
    // 4 bars + 2 legend swatches.
    expect(rects(plan)).toHaveLength(6);
  });

  it("thins category labels instead of overprinting them", () => {
    const many = Array.from({ length: 50 }, (_, i) => `label-${i}`);
    const plan = planChart(barConfig(many.map((_, i) => i), many), BOX, measure);
    const drawn = texts(plan).filter((t) => t.text.startsWith("label-"));
    expect(drawn.length).toBeGreaterThan(0);
    expect(drawn.length).toBeLessThan(many.length);
  });
});

describe("planChart: line, area and scatter", () => {
  const lineConfig = (fill: boolean) => ({
    type: "line",
    data: { labels: ["a", "b", "c", "d"], datasets: [{ label: "v", data: [1, 4, 2, 8], borderColor: "#1565ef", fill }] },
    options: { plugins: { title: { display: true, text: "Trend" } } },
  });

  it("joins the points with one segment per gap", () => {
    const plan = planChart(lineConfig(false), BOX, measure);
    // Gridlines are lines too, so count the ones drawn in the series colour.
    const series = lines(plan).filter((l) => l.thickness === 1.4);
    expect(series).toHaveLength(3);
    expect(circles(plan)).toHaveLength(4);
  });

  it("fills under the line only when the config asks for an area", () => {
    expect(rects(planChart(lineConfig(false), BOX, measure))).toHaveLength(0);
    const filled = rects(planChart(lineConfig(true), BOX, measure));
    expect(filled.length).toBeGreaterThan(0);
    expect(filled.every((r) => r.opacity !== undefined && r.opacity < 0.5)).toBe(true);
  });

  it("plots one marker per scatter point and labels the x axis numerically", () => {
    const plan = planChart({
      type: "scatter",
      data: { datasets: [{ label: "pts", data: [{ x: 1, y: 2 }, { x: 3, y: 9 }, { x: 5, y: 4 }], backgroundColor: "#1565ef" }] },
      options: { plugins: { title: { display: true, text: "Scatter" } }, scales: { x: { title: { text: "x" } }, y: { title: { text: "y" } } } },
    }, BOX, measure);
    expect(circles(plan)).toHaveLength(3);
    expect(texts(plan).map((t) => t.text)).toContain("Scatter");
  });

  it("drops scatter points whose coordinates are not numbers", () => {
    const plan = planChart({
      type: "scatter",
      data: { datasets: [{ label: "pts", data: [{ x: 1, y: 2 }, { x: "n/a", y: 3 }, { x: 4, y: null }] }] },
      options: {},
    }, BOX, measure);
    expect(circles(plan)).toHaveLength(1);
  });
});

describe("planChart: pie and doughnut", () => {
  const pieConfig = (type: string, data: number[]) => ({
    type,
    data: {
      labels: ["alpha", "beta", "gamma"],
      datasets: [{ data, backgroundColor: ["#1565ef", "#16a34a", "#f97316"] }],
    },
    options: { plugins: { title: { display: true, text: "Share" } } },
  });

  it("draws one filled path per positive slice, with its share in the legend", () => {
    const plan = planChart(pieConfig("pie", [50, 30, 20]), BOX, measure);
    expect(paths(plan)).toHaveLength(3);
    const legend = texts(plan).map((t) => t.text);
    expect(legend).toContain("alpha 50.0%");
    expect(legend).toContain("beta 30.0%");
    expect(legend).toContain("gamma 20.0%");
  });

  it("skips zero and negative slices rather than drawing a degenerate wedge", () => {
    const plan = planChart(pieConfig("pie", [50, 0, -10]), BOX, measure);
    expect(paths(plan)).toHaveLength(1);
    expect(texts(plan).map((t) => t.text)).toContain("alpha 100.0%");
  });

  it("gives a doughnut an inner radius, so its slice path has two arcs", () => {
    const pie = paths(planChart(pieConfig("pie", [60, 40]), BOX, measure))[0];
    const ring = paths(planChart(pieConfig("doughnut", [60, 40]), BOX, measure))[0];
    expect((pie.d.match(/A /g) ?? []).length).toBe(1);
    expect((ring.d.match(/A /g) ?? []).length).toBe(2);
  });

  it("returns nothing for a pie with no positive data, so the PDF leaves the space out", () => {
    expect(planChart(pieConfig("pie", [0, 0, 0]), BOX, measure)).toEqual([]);
  });

  it("slicePath closes a full circle with two arcs rather than one unrenderable one", () => {
    const full = slicePath(40, 0, 0, Math.PI * 2);
    expect((full.match(/A /g) ?? []).length).toBe(2);
    expect(full.endsWith("Z")).toBe(true);
  });
});

describe("planChart: boundaries and refusals", () => {
  it("keeps every primitive inside the box it was given", () => {
    const configs = [
      barConfig([10, -40, 300]),
      {
        type: "line",
        data: { labels: ["a", "b", "c"], datasets: [{ label: "v", data: [5, 50, 500], fill: true }] },
        options: { plugins: { title: { display: true, text: "A very long chart title that will not fit on one line at all" } } },
      },
      {
        type: "pie",
        data: { labels: ["one", "two"], datasets: [{ data: [1, 2], backgroundColor: ["#1565ef", "#16a34a"] }] },
        options: { plugins: { title: { display: true, text: "Share" } } },
      },
    ];
    for (const config of configs) {
      for (const p of planChart(config, BOX, measure)) {
        switch (p.kind) {
          case "rect":
            expect(p.x).toBeGreaterThanOrEqual(BOX.x - 0.5);
            expect(p.x + p.w).toBeLessThanOrEqual(BOX.x + BOX.width + 0.5);
            expect(p.y).toBeGreaterThanOrEqual(BOX.y - 0.5);
            expect(p.y + p.h).toBeLessThanOrEqual(BOX.y + BOX.height + 0.5);
            break;
          case "line":
            expect(Math.min(p.x1, p.x2)).toBeGreaterThanOrEqual(BOX.x - 0.5);
            expect(Math.max(p.x1, p.x2)).toBeLessThanOrEqual(BOX.x + BOX.width + 0.5);
            expect(Math.min(p.y1, p.y2)).toBeGreaterThanOrEqual(BOX.y - 0.5);
            expect(Math.max(p.y1, p.y2)).toBeLessThanOrEqual(BOX.y + BOX.height + 0.5);
            break;
          case "text":
            expect(p.x).toBeGreaterThanOrEqual(BOX.x - 0.5);
            expect(p.x + measure(p.text, p.size, p.bold)).toBeLessThanOrEqual(BOX.x + BOX.width + 0.5);
            break;
          case "circle":
            expect(p.cx - p.r).toBeGreaterThanOrEqual(BOX.x - 0.5);
            expect(p.cx + p.r).toBeLessThanOrEqual(BOX.x + BOX.width + 0.5);
            break;
        }
      }
    }
  });

  it("ellipsizes a title too wide for the box instead of overflowing it", () => {
    const long = "x".repeat(400);
    const plan = planChart({
      type: "bar",
      data: { labels: ["a"], datasets: [{ label: "v", data: [1] }] },
      options: { plugins: { title: { display: true, text: long } } },
    }, BOX, measure);
    const title = texts(plan).find((t) => t.bold);
    expect(title).toBeDefined();
    expect(title!.text.endsWith("...")).toBe(true);
    expect(title!.text.length).toBeLessThan(long.length);
  });

  it("returns an empty plan rather than an empty frame when there is nothing to draw", () => {
    expect(planChart({}, BOX, measure)).toEqual([]);
    expect(planChart({ type: "bar", data: { labels: [], datasets: [] } }, BOX, measure)).toEqual([]);
    expect(planChart({ type: "bar", data: { labels: ["a"], datasets: [{ label: "v", data: [1] }] } }, {
      ...BOX,
      width: 10,
      height: 10,
    }, measure)).toEqual([]);
  });

  it("parseConfig refuses a config with no datasets", () => {
    expect(parseConfig({ type: "bar", data: { labels: ["a"] } })).toBeNull();
    expect(parseConfig({ type: "bar" })).toBeNull();
    expect(parseConfig({ type: "scatter", data: { datasets: [{ label: "x", data: [] }] } })).toBeNull();
  });
});

describe("axis helpers", () => {
  it("niceTicks produces round numbers that span the data and include zero", () => {
    const ticks = niceTicks(0, 93);
    expect(ticks[0]).toBe(0);
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(93);
    expect(ticks.every((t) => Number.isFinite(t))).toBe(true);

    const negative = niceTicks(-40, 60);
    expect(negative[0]).toBeLessThanOrEqual(-40);
    expect(negative).toContain(0);
  });

  it("niceTicks survives a flat series and a non-finite range", () => {
    expect(niceTicks(5, 5).length).toBeGreaterThan(1);
    expect(niceTicks(NaN, NaN)).toEqual([0, 1]);
  });

  it("formatValue is compact and keeps small values readable", () => {
    expect(formatValue(0)).toBe("0");
    expect(formatValue(1200)).toBe("1.2k");
    expect(formatValue(3_400_000)).toBe("3.4M");
    expect(formatValue(2_000_000_000)).toBe("2B");
    expect(formatValue(0.25)).toBe("0.25");
    expect(formatValue(-1500)).toBe("-1.5k");
  });

  it("hexToRgb reads the palette the config carries and falls back cleanly", () => {
    expect(hexToRgb("#1565ef", [0, 0, 0])).toEqual([0x15 / 255, 0x65 / 255, 0xef / 255]);
    expect(hexToRgb("rgba(1,2,3,0.5)", [0.5, 0.5, 0.5])).toEqual([0.5, 0.5, 0.5]);
    expect(hexToRgb(undefined, [0.1, 0.2, 0.3])).toEqual([0.1, 0.2, 0.3]);
  });
});
