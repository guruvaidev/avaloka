import { describe, expect, it } from "vitest";
import { changedFeatures, recentSummary, safeErrorDetails, chunksOf, keepRecent, parseCsv, parseJson, predictionRange, rowWarnings, validateRows, baselineImprovement, type InferenceRun } from "./inference";
import { HOUSING_EXAMPLE_ROW, HOUSING_FEATURES, normalizeModels, normalizeResult, parseFeatureSchema } from "@/api/inference";

const features = parseFeatureSchema({ feature_schema: [{ name: "median_income", type: "number", min: 1, max: 15 }, { name: "total_rooms", type: "number" }, { name: "total_bedrooms", type: "number" }, { name: "population", type: "number" }, { name: "households", type: "number" }] });
const row = { median_income: 5, total_rooms: 100, total_bedrooms: 80, population: 200, households: 100 };
describe("Inference input rules", () => {
  it("blocks missing values", () => { expect(rowWarnings({ ...row, median_income: "" }, features).required).toEqual(["median_income"]); });
  it("keeps non-required JSON input warnings advisory", () => { const parsed = parseJson(JSON.stringify({ ...row, median_income: "unknown" }), features); expect(rowWarnings(parsed[0], features)).toMatchObject({ required: [], warnings: ["median_income: Enter a number"] }); });
  it("warns above training max without blocking", () => { const r = rowWarnings({ ...row, median_income: 16 }, features); expect(r.fields.median_income).toEqual(["Above training max 15"]); expect(r.required).toEqual([]); });
  it("warns below training min", () => { expect(rowWarnings({ ...row, median_income: 0 }, features).fields.median_income).toEqual(["Below training min 1"]); });
  it("warns when bedrooms exceed rooms", () => { expect(rowWarnings({ ...row, total_bedrooms: 101 }, features).fields.total_bedrooms).toContain("Can't exceed total_rooms"); });
  it("warns when households exceed population", () => { expect(rowWarnings({ ...row, households: 201 }, features).fields.households).toContain("Can't exceed population"); });
  it("rejects an empty array", () => { expect(() => parseJson("[]", features)).toThrow("Add at least one row"); });
  it("names the missing feature by row", () => { expect(() => parseJson(JSON.stringify([row, { ...row, median_income: null }]), features)).toThrow("Row 2 is missing median_income"); });
  it("converts CSV numbers", () => { expect(parseCsv("median_income,total_rooms,total_bedrooms,population,households\n5,100,80,200,100", features)).toEqual([row]); });
  it("reports missing CSV columns", () => { expect(() => parseCsv("other\n1", features)).toThrow("Missing columns: median_income"); });
  it("caps at 10,000 rows", () => { expect(validateRows(Array.from({ length: 10000 }, () => row), features)).toHaveLength(10000); expect(() => validateRows(Array.from({ length: 10001 }, () => row), features)).toThrow("For more than 10,000 rows, split the file."); });
  it("chunks by 500 in order", () => { const input = Array.from({ length: 1201 }, (_, i) => i); expect(chunksOf(input).map((c) => c.length)).toEqual([500, 500, 201]); expect(chunksOf(input).flat()).toEqual(input); });
  it("computes MAE ranges", () => { expect(predictionRange(367199.55, 37043.67)).toEqual([330155.88, 404243.22]); });
  it("(a) computes the exact requested typical and wider ranges", () => {
    expect(predictionRange(219035.95, 37043.67)).toEqual([181992.28, 256079.62]);
    expect(predictionRange(219035.95, 55006.76)).toEqual([164029.19, 274042.71]);
    expect(predictionRange(10, 20)).toEqual([0, 30]);
  });
  it("(b) preserves three JSON/CSV rows and both inconsistent-input warnings", () => {
    const input = [HOUSING_EXAMPLE_ROW, { ...HOUSING_EXAMPLE_ROW, median_income: "7" }, { ...HOUSING_EXAMPLE_ROW, total_rooms: "120", households: "326" }];
    const rows = parseJson(JSON.stringify(input), HOUSING_FEATURES);
    expect(rows.map((r) => r.total_rooms)).toEqual([880, 880, 120]);
    expect(rowWarnings(rows[2], HOUSING_FEATURES).warnings).toEqual(["total_bedrooms: Can't exceed total_rooms", "households: Can't exceed population"]);
    expect(changedFeatures(rows, 2)).toEqual(["total_rooms", "households", "median_income"]);
    const csv = Object.keys(input[0]).join(",") + "\n" + input.map((r) => Object.values(r).join(",")).join("\n");
    expect(rowWarnings(parseCsv(csv, HOUSING_FEATURES)[2], HOUSING_FEATURES).warnings).toHaveLength(2);
  });
  it("formats recent single and batch predictions with warning counts", () => {
    const model = normalizeModels([{ model_name: "median_house_value_regressor", run_id: "r" }])[0];
    const run: InferenceRun = { id: "r", time: 0, duration: 1, model, rows: [{}], predictions: normalizeResult(219035.95, 1), warnings: [[]] };
    expect(recentSummary(run)).toBe("$219,035.95");
    expect(recentSummary({ ...run, rows: [{}, {}], predictions: normalizeResult([219035.95, 396177.87], 2), warnings: [[], ["a", "b"]] })).toBe("$219,035.95 – $396,177.87 · 2 warnings");
  });
  it("exposes only short safe server details, status and the failed call", () => {
    expect(safeErrorDetails({ status: 502, call: "configure-inference-service", detail: "Service unavailable" })).toEqual({ status: "502", call: "configure-inference-service", detail: "Service unavailable" });
    expect(safeErrorDetails({ detail: "x".repeat(301) }).detail).toBeUndefined();
    expect(safeErrorDetails({ detail: "Traceback: secret" }).detail).toBeUndefined();
  });
  it("computes baseline improvement", () => { expect(baselineImprovement(75, 100)).toBe(0.25); expect(baselineImprovement(75, undefined)).toBeUndefined(); });
  it("keeps the last 20 session runs", () => { const runs = Array.from({ length: 20 }, (_, i) => ({ id: String(i) } as InferenceRun)); const r = keepRecent(runs, { id: "new" } as InferenceRun); expect(r).toHaveLength(20); expect(r[0].id).toBe("new"); });
});
