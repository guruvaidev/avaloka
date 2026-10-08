// derive-viz builds the client-side chart config shown when the backend reply
// carries rows but no visualization_config. Values here are small integers so
// the assertions do not depend on the machine's number-formatting locale.
import { describe, expect, it } from "vitest";
import { deriveVizFromResultTable, deriveVizFromRows } from "@/lib/derive-viz";

describe("deriveVizFromRows", () => {
  it.each([
    ["null", null],
    ["a non-array", { region: "North" }],
    ["an empty array", []],
    ["rows that are not objects", [1, 2, 3]],
    ["rows with no keys", [{}, {}]],
    ["rows whose every value is blank", [{ a: "", b: null }]],
  ])("returns null for %s", (_label, rows) => {
    expect(deriveVizFromRows(rows)).toBeNull();
  });

  it("marks its output as client-derived", () => {
    expect(deriveVizFromRows([{ region: "North" }, { region: "South" }])?.source).toBe(
      "client_derived",
    );
  });

  describe("category + number", () => {
    const rows = [
      { region: "North", revenue: 10 },
      { region: "South", revenue: 40 },
      { region: "North", revenue: 20 },
      { region: "East", revenue: "1,000" },
      { region: "", revenue: 999 },
      { region: "West", revenue: null },
    ];

    it("sums the number per category and sorts descending", () => {
      const [bar] = deriveVizFromRows(rows).charts;
      expect(bar.type).toBe("bar");
      expect(bar.xKey).toBe("region");
      expect(bar.data).toEqual([
        { region: "East", revenue: 1000 },
        { region: "South", revenue: 40 },
        { region: "North", revenue: 30 },
      ]);
    });

    it("labels the value axis with the aggregate it actually applied", () => {
      const [bar] = deriveVizFromRows(rows).charts;
      expect(bar.xAxisTitle).toBe("Region");
      expect(bar.yAxisTitle).toBe("Total revenue");
      expect(bar.series).toEqual([{ dataKey: "revenue", name: "Total revenue" }]);
    });

    it("keeps only the ten largest categories", () => {
      // Letter-only labels: see the ID-like label test below for why.
      const many = Array.from("ABCDEFGHIJKLMN", (letter, i) => ({
        team: `Team ${letter}`,
        units: i + 1,
      }));
      const [bar] = deriveVizFromRows(many).charts;
      expect(bar.data).toHaveLength(10);
      expect(bar.data[0]).toEqual({ team: "Team N", units: 14 });
      expect(bar.data[9]).toEqual({ team: "Team E", units: 5 });
    });

    // The date sniffing hands any string with a digit and a "-", "/", ":" or
    // "T" to Date.parse, and V8 parses "sku-4", "Q-1" and "Store 12" as dates.
    // An ID-like label column must still be charted as categories.
    it.each([
      ["sku-", ["sku-1", "sku-2", "sku-3", "sku-4"]],
      ["quarter", ["Q-1", "Q-2", "Q-3", "Q-4"]],
      ["path-like", ["item/1", "item/2", "item/3", "item/4"]],
    ])("treats ID-like %s labels as categories, not dates", (_label, labels) => {
      const rows = labels.map((label, i) => ({ label, units: (i + 1) * 10 }));
      const [chart] = deriveVizFromRows(rows).charts;
      expect(chart.type).toBe("bar");
      expect(chart.data[0]).toEqual({ label: labels[3], units: 40 });
    });
  });

  describe("date + number", () => {
    const rows = [
      { order_date: "2024-03-01", total: 30 },
      { order_date: "2024-01-01", total: 10 },
      { order_date: "2024-02-01", total: 50 },
    ];

    it("draws a line in chronological order", () => {
      const line = deriveVizFromRows(rows).charts[0];
      expect(line.type).toBe("line");
      expect(line.data.map((row: { order_date: string }) => row.order_date)).toEqual([
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
      ]);
      expect(line.xAxisTitle).toBe("Order date");
      expect(line.yAxisTitle).toBe("Average total");
      expect(line.aggregate).toBe("mean");
      expect(line.yField).toBe("total");
    });

    it("describes the direction from the first to the last date, not row order", () => {
      const line = deriveVizFromRows(rows).charts[0];
      expect(line.subtitle).toBe("total increased from 10 to 30.");
      const falling = deriveVizFromRows([...rows, { order_date: "2024-04-01", total: 5 }])
        .charts[0];
      expect(falling.subtitle).toBe("total decreased from 10 to 5.");
    });

    it("does not mistake plain numbers for dates", () => {
      const charts = deriveVizFromRows([
        { year: 2023, total: 1 },
        { year: 2024, total: 2 },
      ]).charts;
      expect(charts.map((chart: { type: string }) => chart.type)).toEqual(["scatter"]);
    });
  });

  describe("two numbers", () => {
    it("draws a scatter of the first two numeric columns", () => {
      const [scatter] = deriveVizFromRows([
        { height: 150, weight: 50 },
        { height: 160, weight: "60" },
        { height: 170, weight: null },
      ]).charts;
      expect(scatter.type).toBe("scatter");
      expect(scatter.xKey).toBe("height");
      expect(scatter.data).toEqual([
        { height: 150, weight: 50 },
        { height: 160, weight: 60 },
      ]);
      expect(scatter.subtitle).toBe("2 observations compare weight with height.");
    });

    it("caps a scatter at 500 points", () => {
      const rows = Array.from({ length: 620 }, (_, i) => ({ a: i, b: i * 2 }));
      expect(deriveVizFromRows(rows).charts[0].data).toHaveLength(500);
    });

    it("shows a single all-numeric row as a metric bar, never a one-point scatter", () => {
      const viz = deriveVizFromRows([{ mean_price: 12, median_price: 9 }]);
      expect(viz.charts).toHaveLength(1);
      const [bar] = viz.charts;
      expect(bar.type).toBe("bar");
      expect(bar.xAxisTitle).toBe("Metric");
      expect(bar.yAxisTitle).toBe("Price");
      expect(bar.data).toEqual([
        { metric: "mean price", value: 12 },
        { metric: "median price", value: 9 },
      ]);
    });

    it("titles the value axis generically when the metrics measure different things", () => {
      const [bar] = deriveVizFromRows([{ mean_price: 12, max_weight: 9 }]).charts;
      expect(bar.yAxisTitle).toBe("Calculated value");
    });
  });

  describe("category only", () => {
    it("falls back to a count per category, most frequent first", () => {
      const [bar] = deriveVizFromRows([
        { status: "Open" },
        { status: "Closed" },
        { status: "Open" },
        { status: "" },
      ]).charts;
      expect(bar.data).toEqual([
        { status: "Open", count: 2 },
        { status: "Closed", count: 1 },
      ]);
      expect(bar.yAxisTitle).toBe("Count of records");
      expect(bar.subtitle).toBe("Open is the most frequent value with 2 rows.");
    });

    it("returns null when the only category has a single distinct value", () => {
      expect(deriveVizFromRows([{ status: "Open" }, { status: "Open" }])).toBeNull();
    });
  });
});

describe("deriveVizFromResultTable", () => {
  const grouped = {
    title: "Sales summary",
    group_by: ["region"],
    metric_columns: ["sum_revenue", "mean_units"],
    rows: [
      { region: "North", sum_revenue: 30, mean_units: 4 },
      { region: "South", sum_revenue: 40, mean_units: 2 },
      { region: null, sum_revenue: 5, mean_units: "n/a" },
    ],
  };

  it("plots each metric against the grouping without re-aggregating", () => {
    const viz = deriveVizFromResultTable(grouped);
    expect(viz.source).toBe("client_derived");
    expect(viz.charts).toHaveLength(2);
    const [revenue, units] = viz.charts;
    expect(revenue.data).toEqual([
      { region: "South", sum_revenue: 40 },
      { region: "North", sum_revenue: 30 },
      { region: "(missing)", sum_revenue: 5 },
    ]);
    // The non-numeric "n/a" row is dropped rather than plotted as zero.
    expect(units.data).toEqual([
      { region: "North", mean_units: 4 },
      { region: "South", mean_units: 2 },
    ]);
  });

  it("titles axes from the group and metric columns", () => {
    const [revenue] = deriveVizFromResultTable(grouped).charts;
    expect(revenue.title).toBe("sum revenue by region");
    expect(revenue.xAxisTitle).toBe("Region");
    expect(revenue.yAxisTitle).toBe("Sum revenue");
    expect(revenue.subtitle).toBe("Showing all 3 groups from Sales summary.");
  });

  it("joins several grouping columns into one label", () => {
    const [chart] = deriveVizFromResultTable({
      group_by: ["region", "tier"],
      metric_columns: ["total"],
      rows: [
        { region: "North", tier: "Gold", total: 7 },
        { region: "North", tier: "Silver", total: 3 },
      ],
    }).charts;
    expect(chart.xKey).toBe("region / tier");
    expect(chart.xAxisTitle).toBe("Region / Tier");
    expect(chart.data).toEqual([
      { "region / tier": "North / Gold", total: 7 },
      { "region / tier": "North / Silver", total: 3 },
    ]);
  });

  it("shows the ten highest groups and says how many there were", () => {
    const rows = Array.from({ length: 12 }, (_, i) => ({ sku: `sku-${i}`, total: i }));
    const [chart] = deriveVizFromResultTable({
      group_by: ["sku"],
      metric_columns: ["total"],
      rows,
    }).charts;
    expect(chart.data).toHaveLength(10);
    expect(chart.data[0]).toEqual({ sku: "sku-11", total: 11 });
    expect(chart.subtitle).toBe("Showing the 10 highest of 12 groups from the result table.");
  });

  it("ignores group and metric columns that are not in the rows", () => {
    const viz = deriveVizFromResultTable({
      group_by: ["region", "ghost"],
      metric_columns: ["total", "phantom"],
      rows: [
        { region: "North", total: 7 },
        { region: "South", total: 3 },
      ],
    });
    expect(viz.charts).toHaveLength(1);
    expect(viz.charts[0].xKey).toBe("region");
  });

  it("plots an ungrouped single-row result as one bar per metric", () => {
    const [bar] = deriveVizFromResultTable({
      title: "Overall",
      metric_columns: ["mean_price", "median_price", "note"],
      rows: [{ mean_price: 12, median_price: 9, note: "ok" }],
    }).charts;
    expect(bar.title).toBe("Overall");
    expect(bar.xAxisTitle).toBe("Statistic");
    expect(bar.yAxisTitle).toBe("Price");
    expect(bar.data).toEqual([
      { metric: "mean price", value: 12 },
      { metric: "median price", value: 9 },
    ]);
  });

  it("falls back to row inference, prefixing the table title", () => {
    const viz = deriveVizFromResultTable({
      title: "Tickets",
      rows: [{ status: "Open" }, { status: "Closed" }, { status: "Open" }],
    });
    expect(viz.charts[0].title).toBe("Tickets: status distribution");
  });

  it("returns null when no metric value is numeric", () => {
    expect(
      deriveVizFromResultTable({
        group_by: ["region"],
        metric_columns: ["total"],
        rows: [{ region: "North", total: "n/a" }],
      }),
    ).toBeNull();
  });
});
