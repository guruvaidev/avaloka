// Moved from src/api/inference.test.ts: vitest's include globs run tests/unit/**, not src/** (contract e9_17).
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ proxy: vi.fn(), token: vi.fn() }));
vi.mock("@/lib/api/backendApi", () => ({ proxyFetch: mocks.proxy, getAccessToken: mocks.token }));
import { formatCurrency, formatDate, formatNumber, formatPercent, formatRelativeTime, formatVersion, inferenceApi, InferenceApiError, normalizeModelDetails, normalizeModels, normalizeResult } from "@/api/inference";

const real = (i: number) => ({ model_name: "median_house_value_model", model_type: "regression", model_version: String(i), mlflow_run_id: `run${i}abcdef${i}`, created_at: new Date(2026, 0, 1 + (i % 28)).toISOString(), rows_processed: 16512, num_epochs_trained: 50, num_features: 9, final_train_loss: 0.12345, final_val_loss: 0.2, final_mae: 37043.6712, final_rmse: 55006.76, final_r2_score: 0.77, evaluation_report: { metrics: { mae: 1, rmse: 2, r2: 0.5 }, splitter: "random", baseline: { mae: 90000, rmse: 110000 } }, inference_service_details: i % 2 ? { url: "x" } : null });

describe("Model normalization", () => {
  it("(a) normalizes ~100 real runs keyed by mlflow_run_id, newest first", () => {
    const models = normalizeModels({ models: Array.from({ length: 100 }, (_, i) => real(i)) });
    expect(models).toHaveLength(100);
    expect(models.every((m) => m.usable && m.runId?.startsWith("run"))).toBe(true);
    expect(models[0].taskType).toBe("Regression");
    expect(models[0].mae).toBeCloseTo(37043.6712);
    expect(models[0].baselineMae).toBe(90000);
    expect(new Set(models.map((m) => m.key)).size).toBe(100);
    expect((models[0].createdAt?.getTime() ?? 0) >= (models[99].createdAt?.getTime() ?? 0)).toBe(true);
  });
  it("(b) handles empty and malformed lists", () => { expect(normalizeModels({ models: [] })).toEqual([]); expect(normalizeModels(null)).toEqual([]); expect(normalizeModels("oops")).toEqual([]); expect(normalizeModels({ models: "x" })).toEqual([]); });
  it("(c) treats an item with only run_id as not ready", () => { const [m] = normalizeModels([{ run_id: "only" }]); expect(m.runId).toBe("only"); expect(m.usable).toBe(false); expect(m.name).toBeUndefined(); });
  it("(d) parses metrics sent as strings and falls back to evaluation_report", () => { const [m] = normalizeModels([{ run_id: "r", model_name: "n", final_mae: "12.5", evaluation_report: { metrics: { rmse: "20", r2: "0.5" } } }]); expect(m.mae).toBe(12.5); expect(m.rmse).toBe(20); expect(m.r2).toBe(0.5); });
  it("(e) ignores invalid created_at", () => { const [m] = normalizeModels([{ run_id: "r", model_name: "n", created_at: "not a date" }]); expect(m.createdAt).toBeUndefined(); expect(formatDate(m.createdAt)).toBe("—"); expect(formatRelativeTime(m.createdAt)).toBe("—"); });
  it("skips null items without throwing", () => { expect(normalizeModels([null, 5, { run_id: "r", model_name: "n" }]).filter((m) => m.usable)).toHaveLength(1); });
  it("falls back to housing features for median_house_value", () => { expect(normalizeModelDetails(real(1))?.features.map((f) => f.name)).toContain("ocean_proximity"); });
});

describe("Result normalization (f)", () => {
  it("maps a bare number", () => { expect(normalizeResult(206855.5, 1)).toEqual([{ value: 206855.5, probabilities: [] }]); });
  it("maps an array of values in order", () => { expect(normalizeResult([1, 2, 3], 3).map((p) => p.value)).toEqual([1, 2, 3]); });
  it("maps an array of objects with probabilities", () => { const r = normalizeResult([{ prediction: "yes", probabilities: { yes: 0.8, no: 0.2 } }, { prediction: "no" }], 2); expect(r.map((p) => p.value)).toEqual(["yes", "no"]); expect(r[0].probabilities).toEqual([["yes", 0.8], ["no", 0.2]]); });
  it("always returns one entry per submitted row", () => { expect(normalizeResult([1], 3).map((p) => p.value)).toEqual([1, null, null]); });
});

describe("Formatters", () => {
  it("formats with separators and dashes for missing input", () => { expect(formatNumber(37043.6712)).toBe("37,043.67"); expect(formatNumber(55006.76)).toBe("55,006.76"); expect(formatNumber(undefined)).toBe("—"); expect(formatPercent(0.7712)).toBe("77.12%"); expect(formatCurrency(206855.5)).toBe("$206,855.50"); expect(formatCurrency("x")).toBe("—"); });
});

describe("Inference endpoint contract", () => {
  beforeEach(() => { mocks.proxy.mockReset(); mocks.token.mockResolvedValue("tok"); mocks.proxy.mockImplementation(async () => Response.json({ result: [1] })); });
  it("sends bearer auth", async () => { await inferenceApi.listModels(); expect(mocks.proxy).toHaveBeenCalledWith("/api/models", expect.objectContaining({ method: "GET", headers: { Authorization: "Bearer tok" } })); });
  it("encodes run ids", async () => { await inferenceApi.getModel("run/id"); expect(mocks.proxy.mock.calls[0][0]).toBe("/api/models/run%2Fid"); });
  it("sends one row unwrapped and batches as arrays", async () => { await inferenceApi.predict("abc", { x: 5 }); await inferenceApi.predict("abc", [{ x: 5 }, { x: 6 }]); expect(mocks.proxy.mock.calls[0][1].body).toBe('{"x":5}'); expect(mocks.proxy.mock.calls[1][1].body).toBe('[{"x":5},{"x":6}]'); });
  it("rejects empty arrays before sending", () => { expect(() => inferenceApi.predict("abc", [])).toThrow(); expect(mocks.proxy).not.toHaveBeenCalled(); });
  it("(g) surfaces 409 so the page can configure and retry", async () => { mocks.proxy.mockResolvedValueOnce(Response.json({ detail: "not configured" }, { status: 409 })); await expect(inferenceApi.predict("abc", { x: 1 })).rejects.toMatchObject({ status: 409 }); await inferenceApi.startService("abc"); expect(mocks.proxy.mock.calls[1][0]).toBe("/api/models/abc/configure-inference-service"); await expect(inferenceApi.predict("abc", { x: 1 })).resolves.toEqual({ result: [1] }); });
  it("(h) keeps plan_upgrade_required details", async () => { mocks.proxy.mockResolvedValue(Response.json({ detail: { reason: "plan_upgrade_required", message: "Upgrade", allowed_plans: ["enterprise"] } }, { status: 403 })); await expect(inferenceApi.predict("abc", { x: 1 })).rejects.toMatchObject({ status: 403, detail: { reason: "plan_upgrade_required", allowed_plans: ["enterprise"] } }); });
  it("hides network exception text", async () => { mocks.proxy.mockRejectedValue(new Error("secret")); await expect(inferenceApi.listModels()).rejects.toEqual(new InferenceApiError(0, null, "models")); });
});

describe("inference follow-ups", () => {
  it("shows version as-is and adds v only when missing", () => {
    expect(formatVersion("v1.0")).toBe("v1.0");
    expect(formatVersion("2")).toBe("v2");
  });
  it("uses 9 input features with 14 after encoding for housing models", () => {
    const [m] = normalizeModels([{ mlflow_run_id: "r", model_name: "median_house_value_model", num_features: 14 }]);
    expect(m.inputFeatureCount).toBe(9);
    expect(m.encodedFeatureCount).toBe(14);
    expect(m.features.find((f) => f.name === "total_rooms")).toMatchObject({ min: 2, max: 39320 });
    expect(m.features.find((f) => f.name === "ocean_proximity")?.categories).toEqual(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]);
  });
  it("gives unknown models no form fields", () => {
    expect(normalizeModels([{ mlflow_run_id: "r", model_name: "Survived_Classifier" }])[0].features).toEqual([]);
  });
});
