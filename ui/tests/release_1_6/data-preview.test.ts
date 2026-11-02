// UI-B07 / UI-B33 (UI Bugs): Data Preview is available for every dataset in
// the thread (it follows the selected dataset), and the Data Table and the
// Data Preview (Column Summary) show the same rows and columns.
// Tests workspaceDatasetFromChip and dataViewSource, moved verbatim from
// routes/analysis.tsx.
import { describe, expect, it } from "vitest";
import { dataViewSource, workspaceDatasetFromChip, type DatasetChip } from "@/lib/analysis-workspace";

const salesRows = [
  { region: "north", sales: 10 },
  { region: "south", sales: 20 },
  { region: "east", sales: 30 },
];
const returnRows = [
  { sku: "A-1", returned: 2, reason: "damaged" },
  { sku: "B-2", returned: 1, reason: "late" },
];
const sales: DatasetChip = { id: "ds-a", name: "sales.csv", samples: salesRows, schema: ["region", "sales"], totalRows: 3000 };
const returns: DatasetChip = { id: "ds-b", name: "returns.csv", samples: returnRows, rowsSampled: 2 };

describe("UI-B07 Data Preview for every dataset", () => {
  it("UI-B07: selecting each dataset gives a preview of THAT dataset's own rows and columns", () => {
    const a = workspaceDatasetFromChip(sales, sales.samples!);
    const b = workspaceDatasetFromChip(returns, returns.samples!);
    expect(a).toMatchObject({ filename: "sales.csv", schema: ["region", "sales"], samples: salesRows, total_rows: 3000 });
    expect(b).toMatchObject({ filename: "returns.csv", schema: ["sku", "returned", "reason"], samples: returnRows, rows_sampled: 2 });
    const previewA = dataViewSource(a, "sample", false, []);
    const previewB = dataViewSource(b, "sample", false, []);
    expect(previewA.dataViewColumns).toEqual(["region", "sales"]);
    expect(previewB.dataViewColumns).toEqual(["sku", "returned", "reason"]);
    expect(previewB.dataViewRows).toBe(returnRows);
  });
});

describe("UI-B33 Data Table and Data Preview agree", () => {
  const dataset = workspaceDatasetFromChip(sales, sales.samples!);
  const resultRows = [{ region: "north", total: 10 }];

  it("UI-B33: on the data sample, the preview shows exactly the table's rows and columns", () => {
    const v = dataViewSource(dataset, "sample", true, resultRows);
    expect(v.dataViewRows).toBe(v.sampleRows);
    expect(v.dataViewColumns).toEqual(v.sampleColumns);
    expect(v.dataViewRows).toHaveLength(3);
  });

  it("UI-B33: after switching to an analysis result, the preview follows that result table", () => {
    const v = dataViewSource(dataset, "results", true, resultRows);
    expect(v.dataViewRows).toBe(resultRows);
    expect(v.dataViewColumns).toEqual(["region", "total"]);
    // The data-sample table itself is unchanged.
    expect(v.sampleRows).toBe(salesRows);
  });
});
