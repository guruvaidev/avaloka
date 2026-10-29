import { describe, expect, it } from "vitest";
import {
  applyFilters,
  cellAlign,
  choosePinnedColumn,
  columnKinds,
  formatValue,
  inferColumnKind,
  sortRows,
  summarizeColumns,
} from "../../src/lib/data-view";

const rows = [
  { loan_id: "L3", activity_year: 2024, loan_type: "FHA", loan_amount: 250000, interest_rate: 6.5, census_tract: "06037101110", derived_msa_md: "31084" },
  { loan_id: "L1", activity_year: 2018, loan_type: "conventional", loan_amount: 900000, interest_rate: null, census_tract: "06037101120", derived_msa_md: "31084" },
  { loan_id: "L2", activity_year: 2021, loan_type: "VA", loan_amount: null, interest_rate: 3.1, census_tract: "", derived_msa_md: "41860" },
  { loan_id: "L4", activity_year: 2019, loan_type: "Conventional", loan_amount: 40000, interest_rate: 4.2, census_tract: "06037101130", derived_msa_md: "41860" },
];
const kinds = columnKinds(rows, Object.keys(rows[0]));

describe("sorting", () => {
  it("sorts numbers numerically, descending, empty last", () => {
    expect(sortRows(rows, "loan_amount", "desc", "number").map((r) => r.loan_amount)).toEqual([900000, 250000, 40000, null]);
  });
  it("keeps empty values last when ascending too", () => {
    expect(sortRows(rows, "loan_amount", "asc", "number").map((r) => r.loan_amount)).toEqual([40000, 250000, 900000, null]);
  });
  it("sorts years 2018 → 2024", () => {
    expect(sortRows(rows, "activity_year", "asc", kinds.activity_year).map((r) => r.activity_year)).toEqual([2018, 2019, 2021, 2024]);
  });
  it("sorts text case-insensitively", () => {
    expect(sortRows(rows, "loan_type", "asc", "text").map((r) => r.loan_type)).toEqual(["conventional", "Conventional", "FHA", "VA"]);
  });
  it("sorts dates by date", () => {
    const dated = [{ d: "2024-02-01" }, { d: "" }, { d: "2023-12-31" }];
    expect(sortRows(dated, "d", "asc", "date").map((r) => r.d)).toEqual(["2023-12-31", "2024-02-01", ""]);
  });
});

describe("filtering", () => {
  it("keeps only selected values", () => {
    expect(applyFilters(rows, { loan_type: { values: ["FHA", "VA"] } }).map((r) => r.loan_id)).toEqual(["L3", "L2"]);
  });
  it("applies numeric min/max and drops empty values", () => {
    expect(applyFilters(rows, { loan_amount: { min: 100000, max: 300000 } }).map((r) => r.loan_id)).toEqual(["L3"]);
  });
  it("applies contains case-insensitively", () => {
    expect(applyFilters(rows, { loan_type: { contains: "conv" } })).toHaveLength(2);
  });
  it("cleared filters return all rows", () => {
    expect(applyFilters(rows, { loan_type: {} })).toHaveLength(4);
  });
});

describe("decimal formatting", () => {
  it("formats computed decimals to at most 2 places", () => {
    expect(formatValue(34307922673.6667, "number", "total_views")).toBe("34,307,922,673.67");
    expect(formatValue(6.5, "number", "interest_rate")).toBe("6.5");
  });
  it("never formats codes or years as numbers", () => {
    expect(formatValue("06037101110", "identifier", "census_tract")).toBe("06037101110");
    expect(formatValue(2024, "number", "activity_year")).toBe("2024");
  });
});

describe("alignment rule", () => {
  it("right-aligns measures and left-aligns codes, IDs and text", () => {
    expect(cellAlign(kinds.loan_amount)).toBe("right");
    expect(cellAlign(kinds.interest_rate)).toBe("right");
    expect(cellAlign(kinds.loan_type)).toBe("left");
    expect(cellAlign(kinds.census_tract)).toBe("left");
    expect(cellAlign(kinds.derived_msa_md)).toBe("left");
    expect(cellAlign(kinds.loan_id)).toBe("left");
  });
  it("treats numeric-looking codes as identifiers even when dtype says integer", () => {
    expect(inferColumnKind("zip_code", [94110], { name: "zip_code", dtype: "int64" })).toBe("identifier");
    expect(inferColumnKind("x", [1], { name: "x", dtype: "int64", role: "identifier" })).toBe("identifier");
  });
  it("never adds separators to codes or years", () => {
    expect(formatValue("31084", "identifier", "derived_msa_md")).toBe("31084");
    expect(formatValue(2024, "number", "activity_year")).toBe("2024");
    expect(formatValue(250000, "number", "loan_amount")).toBe("250,000");
  });
});

describe("pinned column", () => {
  it("prefers id / *_id columns", () => {
    expect(choosePinnedColumn(["loan_type", "loan_id", "activity_year"])).toBe("loan_id");
  });
  it("uses the time column when no id exists", () => {
    expect(choosePinnedColumn(["loan_type", "activity_year"])).toBe("activity_year");
  });
  it("uses a profile key/time role", () => {
    expect(choosePinnedColumn(["a", "period"], [{ name: "period", role: "time" }])).toBe("period");
  });
  it("falls back to the first column", () => {
    expect(choosePinnedColumn(["loan_type", "income"])).toBe("loan_type");
  });
});


describe("column summary", () => {
  it("counts missing values and types", () => {
    const rate = summarizeColumns(rows, ["interest_rate", "activity_year"], kinds);
    expect(rate[0]).toMatchObject({ type: "Decimal", nonNull: 3, missing: 1 });
    expect(rate[1].type).toBe("Integer");
  });
});
