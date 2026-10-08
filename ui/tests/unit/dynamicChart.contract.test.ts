// Pins the backend -> chart contract for precomputed points.
//
// The backend (app/agents/visualization_agent.py, _ground_chart) attaches
//   derived_data = { points, x_key, y_key, ... }
// where every point carries its label under x_key and its number under y_key
// (_keyed_points). The component instead looks for derived_data.category_field
// / value_field -- names the backend has never sent -- and then falls back to
// the encoding field names. That fallback only works while x_key/y_key happen
// to equal the encoding fields. These tests state the contract directly so the
// coincidence cannot hide a break.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { normalizeChart } from "@/components/dashboard/dynamicChart";

/** Mirror of _keyed_points(): generic aliases plus the chart's own keys. */
function keyedPoints(points: { x: string; y: number }[], xKey: string, yKey: string) {
  return points.map((p) => ({ ...p, label: p.x, name: p.x, value: p.y, [xKey]: p.x, [yKey]: p.y }));
}

const RAW_POINTS = [
  { x: "Emergency", y: 5 },
  { x: "Elective", y: 3 },
  { x: "Urgent", y: 1 },
];

const AS_PLOTTED = [
  { category: "Emergency", value: 5 },
  { category: "Elective", value: 3 },
  { category: "Urgent", value: 1 },
];

function backendChart(
  type: string,
  encodings: Record<string, unknown>,
  xKey: string,
  yKey: string,
) {
  return {
    id: "chart_1",
    type,
    title: "Admissions",
    encodings,
    derived_data: { points: keyedPoints(RAW_POINTS, xKey, yKey), x_key: xKey, y_key: yKey },
  };
}

describe("derived_data.points contract", () => {
  // The UI can be checked out without the backend (e.g. a ui-only image build
  // context); the cross-check only exists in the full repo, and is reported as
  // skipped rather than passed when it cannot run.
  const agent = resolve(__dirname, "../../../app/agents/visualization_agent.py");
  it.skipIf(!existsSync(agent))("backend still declares point keys as x_key / y_key", () => {
    const source = readFileSync(agent, "utf8");
    expect(source).toContain('derived["x_key"] = x_key');
    expect(source).toContain('derived["y_key"] = y_key');
    expect(source).not.toMatch(/derived\["(category_field|value_field)"\]/);
  });

  it("draws a count bar as emitted today (y has no field, y_key is 'count')", () => {
    const chart = backendChart(
      "bar",
      {
        x: { field: "Admission Type", type: "nominal" },
        y: { aggregate: "count", type: "quantitative" },
      },
      "Admission Type",
      "count",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });

  it("draws a measure bar as emitted today (y_key equals the y field)", () => {
    const chart = backendChart(
      "bar",
      {
        x: { field: "Admission Type", type: "nominal" },
        y: { field: "Billing Amount", type: "quantitative", aggregate: "sum" },
      },
      "Admission Type",
      "Billing Amount",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });

  it("draws a count pie as emitted today (theta rewritten to the 'count' key)", () => {
    const chart = backendChart(
      "pie",
      {
        color: { field: "Admission Type", type: "nominal" },
        theta: { field: "count", type: "quantitative", aggregate: "sum" },
      },
      "Admission Type",
      "count",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });

  it("keeps the backend's point order instead of re-sorting", () => {
    const chart = backendChart(
      "bar",
      { x: { field: "t" }, y: { aggregate: "count" } },
      "t",
      "count",
    );
    chart.derived_data.points.reverse();
    expect(normalizeChart(chart)!.data.map((row) => row.category)).toEqual([
      "Urgent",
      "Elective",
      "Emergency",
    ]);
  });

  // _ground_chart discards a y field that is the label column itself, or that
  // is not a real column, and counts rows instead: y_key becomes "count" while
  // encodings.y.field is left untouched for bar and line charts (only pies get
  // their theta rewritten). The encoding field is then the wrong key to read.
  it("reads values from y_key when the y encoding names the label column", () => {
    const chart = backendChart(
      "bar",
      { x: { field: "Admission Type" }, y: { field: "Admission Type", aggregate: "count" } },
      "Admission Type",
      "count",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });

  it("reads values from y_key when the y encoding names a column that does not exist", () => {
    const chart = backendChart(
      "bar",
      { x: { field: "Admission Type" }, y: { field: "Patients", aggregate: "sum" } },
      "Admission Type",
      "count",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });

  it("reads labels from x_key and values from y_key, not from encoding field names", () => {
    const chart = backendChart(
      "bar",
      { x: { field: "admission_type" }, y: { field: "billing_amount", aggregate: "sum" } },
      "Admission Type",
      "Billing Amount",
    );
    expect(normalizeChart(chart)!.data).toEqual(AS_PLOTTED);
  });
});
