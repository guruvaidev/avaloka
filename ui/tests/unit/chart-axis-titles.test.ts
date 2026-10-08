import { describe, expect, it } from "vitest";
import { aggregateTitle, chartAxisTitles, fieldTitle } from "@/lib/chart-axis-titles";
import { normalizeVizConfig } from "@/components/dashboard/dynamicChart";
import { deriveVizFromResultTable, deriveVizFromRows } from "@/lib/derive-viz";

describe("fieldTitle", () => {
  it.each([
    ["order_date", "Order date"],
    ["billing-amount", "Billing amount"],
    ["unitPrice", "Unit Price"],
    ["  ", "Value"],
  ])("%j -> %j", (input, expected) => {
    expect(fieldTitle(input)).toBe(expected);
  });
});

describe("aggregateTitle", () => {
  it.each([
    ["count", "revenue", "Count of records"],
    ["sum", "revenue", "Total revenue"],
    ["mean", "unit_price", "Average unit price"],
    ["AVG", "revenue", "Average revenue"],
    ["median", "revenue", "Median revenue"],
    ["min", "revenue", "Minimum revenue"],
    ["max", "revenue", "Maximum revenue"],
    ["sum", "GDP", "Total GDP"],
    [null, "revenue", "Revenue"],
    [undefined, null, "Value"],
  ])("%j of %j -> %j", (aggregate, field, expected) => {
    expect(aggregateTitle(aggregate, field)).toBe(expected);
  });
});

describe("chartAxisTitles", () => {
  const series = [{ dataKey: "revenue", name: "revenue" }];

  it("prefers explicit titles over anything inferred", () => {
    expect(
      chartAxisTitles({
        xKey: "region",
        series,
        xAxisTitle: " Sales region ",
        yAxisTitle: "Total revenue",
      }),
    ).toEqual({ x: "Sales region", y: "Total revenue" });
  });

  it("titles a histogram by bin start and record count", () => {
    expect(chartAxisTitles({ xKey: "age", series, chartKind: "histogram" })).toEqual({
      x: "Age (bin start)",
      y: "Count of records",
    });
  });

  it("names the aggregate on the value axis", () => {
    expect(
      chartAxisTitles({
        xKey: "category",
        xField: "region",
        series,
        aggregate: "median",
        yField: "revenue",
      }),
    ).toEqual({ x: "Region", y: "Median revenue" });
  });

  it("falls back to the series names", () => {
    expect(
      chartAxisTitles({
        xKey: "month",
        series: [{ dataKey: "units_sold" }, { dataKey: "returns", name: "Returned units" }],
      }),
    ).toEqual({ x: "Month", y: "Units sold / Returned units" });
  });
});

// derive-viz.ts writes xAxisTitle / yAxisTitle onto every chart it builds. The
// slide is what the chart component receives, so the titles only reach the
// screen if normalizeVizConfig carries them across.
describe("client-derived axis titles survive normalizeVizConfig", () => {
  it("for a chart derived from rows", () => {
    const viz = deriveVizFromRows([
      { region: "North", revenue: 10 },
      { region: "South", revenue: 40 },
    ]);
    expect(viz.charts[0]).toMatchObject({ xAxisTitle: "Region", yAxisTitle: "Total revenue" });
    const [slide] = normalizeVizConfig(viz);
    expect(chartAxisTitles(slide)).toEqual({ x: "Region", y: "Total revenue" });
    expect(slide).toMatchObject({ xAxisTitle: "Region", yAxisTitle: "Total revenue" });
  });

  it("for a chart derived from a grouped result table", () => {
    const viz = deriveVizFromResultTable({
      group_by: ["region", "tier"],
      metric_columns: ["sum_revenue"],
      rows: [
        { region: "North", tier: "Gold", sum_revenue: 7 },
        { region: "North", tier: "Silver", sum_revenue: 3 },
      ],
    });
    const [slide] = normalizeVizConfig(viz);
    expect(slide).toMatchObject({ xAxisTitle: "Region / Tier", yAxisTitle: "Sum revenue" });
  });
});
