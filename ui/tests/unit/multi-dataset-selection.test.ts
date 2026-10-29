import { describe, expect, it } from "vitest";
import {
  canStartSelection,
  initialSelection,
  PLACEHOLDER_COLUMN_NAME,
  realColumnNames,
  selectionPayload,
  uploadKind,
  workbookHeading,
} from "../../src/lib/upload-review";
import { chipLabel, noChartsText, toggleSelection, visibleChips } from "../../src/lib/multi-dataset-view";

const sheets = Array.from({ length: 10 }, (_, i) => ({
  dataset_id: `s${i}`,
  filename: `Avaloka-1.6-Master-Test-Plan-2 (1) [${i}].csv`,
  source_workbook: "Avaloka-1.6-Master-Test-Plan-2 (1).xlsx",
}));
const files = [
  { dataset_id: "m", filename: "mortgage_loans_test.csv", source_workbook: null },
  { dataset_id: "y", filename: "Global YouTube Statistics.csv", source_workbook: null },
];

describe("Key Metrics placeholder filter", () => {
  it("hides generated names and keeps real columns", () => {
    expect(realColumnNames(["ID", "column_3", "Unnamed: 2", "unnamed 4", "Pri", "Column_12"])).toEqual(["ID", "Pri"]);
    expect(PLACEHOLDER_COLUMN_NAME.test("column_name")).toBe(false);
    expect(realColumnNames(["column_1", "column_2"])).toEqual([]);
  });
});

describe("selection rules", () => {
  it("one workbook selects all sheets and sends no PUT", () => {
    const kind = uploadKind(sheets, 1);
    expect(kind).toBe("workbook");
    const ids = initialSelection(kind, sheets.map((d) => d.dataset_id));
    expect(ids).toHaveLength(10);
    expect(canStartSelection(kind, ids)).toBe(true);
    expect(selectionPayload(kind, ids)).toBeNull();
  });

  it("several files select nothing and keep the buttons disabled until a file is ticked", () => {
    const kind = uploadKind(files, 2);
    expect(kind).toBe("files");
    const ids = initialSelection(kind, files.map((d) => d.dataset_id));
    expect(ids).toEqual([]);
    expect(canStartSelection(kind, ids)).toBe(false);
    expect(canStartSelection(kind, ["m"])).toBe(true);
  });

  it("sends the ticked file's ids in the PUT body", () => {
    expect(selectionPayload("files", ["m"])).toEqual({ dataset_ids: ["m"] });
    expect(selectionPayload("files", [])).toBeNull();
  });

  it("one file with one sheet stays unchanged", () => {
    expect(uploadKind([files[0]], 1)).toBe("single");
    expect(selectionPayload("single", ["m"])).toBeNull();
  });

  it("decides the kind after a reload without a file count", () => {
    expect(uploadKind(sheets)).toBe("workbook");
    expect(uploadKind(files)).toBe("files");
  });
});

describe("chips and headings", () => {
  it("labels workbook sheets by sheet name and shows 3 + more beyond 4 chips", () => {
    expect(chipLabel({ name: "Avaloka_1_6_Ma", sheetName: "3. Agents", sourceWorkbook: "x.xlsx" })).toBe("3. Agents");
    expect(chipLabel({ name: "mortgage.csv", sheetName: null, sourceWorkbook: null })).toBe("mortgage.csv");
    expect(visibleChips(sheets, false)).toEqual({ shown: sheets.slice(0, 3), hidden: 7 });
    expect(visibleChips(sheets, true).hidden).toBe(0);
    expect(visibleChips(files, false).hidden).toBe(0);
  });

  it("never switches off the last selected chip", () => {
    expect(toggleSelection(["a"], "a")).toEqual(["a"]);
    expect(toggleSelection(["a", "b"], "a")).toEqual(["b"]);
    expect(toggleSelection(["a"], "b")).toEqual(["a", "b"]);
  });

  it("titles a workbook analysis with its readable name and sheet count", () => {
    expect(workbookHeading("Avaloka-1.6-Master-Test-Plan-2 (1).xlsx", 10)).toBe("Avaloka 1.6 Master Test Plan 2 · 10 sheets");
  });

  it("explains a sheet with no charts", () => {
    expect(noChartsText([{ title: "Records by column_3", reason: "charts a column with no name in the file (column_3)" }])).toBe(
      "No charts: charts a column with no name in the file (column_3)",
    );
    expect(noChartsText(undefined)).toBe("No charts for this sheet.");
  });
});
