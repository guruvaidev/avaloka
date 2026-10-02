import assert from "node:assert/strict";
import { test } from "node:test";
import { collectAutoInsightHighlights, collectChartFindings } from "../src/lib/auto-insight-highlights.ts";

test("multi-sheet highlights keep one labeled point per sheet", () => {
  const groups = [
    {
      id: "sales",
      name: "book [Sales].csv",
      samples: [
        { ID: 1, "Unnamed: 5": "x", "Bug status": "Open" },
        { ID: 2, "Unnamed: 5": "x", "Bug status": "Open" },
        { ID: 3, "Unnamed: 5": "x", "Bug status": "Open" },
        { ID: 4, "Unnamed: 5": "y", "Bug status": "Open" },
        { ID: 5, "Unnamed: 5": "y", "Bug status": "Closed" },
      ],
      slides: [
        { subtitle: "First chart description", insights: [] },
        { subtitle: "Second chart description", insights: [] },
      ],
    },
    {
      id: "inventory",
      name: "book [Inventory].csv",
      samples: [
        { Category: "Hardware", Revenue: 10 },
        { Category: "Hardware", Revenue: 20 },
        { Category: "Hardware", Revenue: 30 },
        { Category: "Software", Revenue: 40 },
      ],
      slides: [
        { title: "Inventory levels by category", subtitle: "First chart description", insights: [] },
        { subtitle: "Inventory by location", insights: [] },
      ],
    },
    { id: "empty", name: "book [Empty].csv", slides: [{ title: "Counts by Unnamed: 5", subtitle: "Shows counts", insights: [] }] },
  ];

  assert.deepEqual(collectAutoInsightHighlights(groups), [
    { datasetId: "sales", datasetName: "book [Sales].csv", text: "In 5 sampled rows with Bug status, “Open” is most common (4, 80%)." },
    { datasetId: "inventory", datasetName: "book [Inventory].csv", text: "In 4 sampled rows with Category, “Hardware” is most common (3, 75%)." },
  ]);
});

test("numeric sheets report a sample statistic and ignore sequential identifiers", () => {
  const [highlight] = collectAutoInsightHighlights([{
    id: "numeric",
    name: "book [Numeric].csv",
    samples: [10, 20, 30, 40, 50].map((Revenue, i) => ({ ID: i + 1, Revenue })),
    slides: [{ title: "Distribution of Revenue", insights: [] }],
  }]);
  assert.equal(highlight.text, "In 5 sampled rows, the median Revenue is 30 (range 10–50).");
});

test("a saved sheet can derive a finding from chart data when preview rows are absent", () => {
  const [highlight] = collectAutoInsightHighlights([{
    id: "saved",
    name: "book [Bugs].csv",
    slides: [
      { type: "bar", xField: "Unnamed: 5", xKey: "name", aggregate: "count", series: [{ dataKey: "value", name: "Count" }], data: [{ name: "x", value: 8 }, { name: "y", value: 2 }], insights: [] },
      { type: "bar", xField: "Bug status", xKey: "name", aggregate: "count", series: [{ dataKey: "value", name: "Count" }], data: [{ name: "Open", value: 7 }, { name: "Closed", value: 3 }], insights: [] },
    ],
  }]);
  assert.equal(highlight.text, "In the plotted sample, “Open” is the most common Bug status (7 of 10, 70%).");
});

test("Key Insights omit chart descriptions and keep genuine findings", () => {
  assert.deepEqual(
    collectChartFindings([
      { subtitle: "Shows counts by categories of ID.", insights: [] },
      { subtitle: "Plots Count against Count.1.", insights: ["Revenue rose 8%.", "Revenue rose 8%."] },
      { subtitle: "Shows the distribution of Count.1.", insights: ["  Margin fell 2%.  "] },
    ]),
    ["Revenue rose 8%.", "Margin fell 2%."],
  );
});
