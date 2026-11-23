/** The single metric mapping for Configurations and Inference model details. */
export function getModelMetrics(details: unknown) {
  const record = (v: unknown): Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
  const outer = record(details);
  const raw = outer.model ? record(outer.model) : outer;
  const metrics = record(raw.metrics);
  const report = record(raw.evaluation_report);
  const reportMetrics = record(report.metrics);
  const read = (...keys: string[]): number | undefined => {
    for (const source of [raw, metrics, reportMetrics]) {
      for (const key of keys) {
        const value = source[key];
        if (value === null || value === undefined || typeof value === "boolean" || (typeof value === "string" && !value.trim())) continue;
        try { const n = Number(value); if (Number.isFinite(n)) return n; } catch { /* optional malformed metric */ }
      }
    }
    return undefined;
  };
  const splitter = report.splitter ?? raw.validation_splitter ?? raw.splitter;
  const splitterRecord = record(splitter);
  const splitterText = typeof splitter === "string" ? splitter : splitterRecord.name ?? splitterRecord.type ?? splitterRecord.strategy;
  return {
    mae: read("final_mae", "mae", "mean_absolute_error"),
    rmse: read("final_rmse", "rmse", "root_mean_squared_error"),
    r2: read("final_r2_score", "final_r2", "r2_score", "r2", "r squared"),
    trainLoss: read("final_train_loss", "train_loss"),
    valLoss: read("final_val_loss", "val_loss", "validation_loss"),
    accuracy: read("final_accuracy", "accuracy"),
    precision: read("final_precision", "precision"),
    recall: read("final_recall", "recall"),
    f1: read("final_f1_score", "f1_score", "f1"),
    splitter: typeof splitterText === "string" && splitterText.trim() ? splitterText : undefined,
  };
}