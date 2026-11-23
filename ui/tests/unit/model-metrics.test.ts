// Moved from src/lib/modelMetrics.test.ts: vitest's include globs run tests/unit/**, not src/** (contract e9_17).
import { describe, expect, it } from "vitest";
import { getModelMetrics } from "@/lib/modelMetrics";
import { normalizeModelDetails } from "@/api/inference";

export const metricFixture = { mlflow_run_id: "eeaa471263e243deb73a023b3790c37f", model_name: "median_house_value_regressor", model_type: "regression", model_version: "v1.0", num_features: 14, metrics: { final_mae: "37043.67", final_rmse: "55006.76", final_r2_score: "0.7743", final_train_loss: "0.2779", final_val_loss: "0.2268" } };
describe("Shared model metrics", () => {
  it("(e) reads the Configurations nested final metrics identically for Inference", () => {
    const metrics = getModelMetrics(metricFixture);
    expect(metrics).toMatchObject({ mae: 37043.67, rmse: 55006.76, r2: 0.7743, trainLoss: 0.2779, valLoss: 0.2268 });
    expect(normalizeModelDetails(metricFixture)).toMatchObject(metrics);
  });
  it("keeps root values first, then nested metrics, then evaluation metrics", () => {
    expect(getModelMetrics({ final_mae: "1", metrics: { final_mae: "2", rmse: "3" }, evaluation_report: { metrics: { mae: "4", rmse: "5", r2: "0.5" } } })).toMatchObject({ mae: 1, rmse: 3, r2: 0.5 });
  });
  it("(f) keeps missing, blank and invalid values optional without crashing", () => {
    for (const value of [null, {}, { metrics: { final_mae: "NaN", final_rmse: "Infinity", r2: "", train_loss: false, val_loss: null } }]) {
      expect(getModelMetrics(value)).toMatchObject({ mae: undefined, rmse: undefined, r2: undefined, trainLoss: undefined, valLoss: undefined });
    }
  });
});