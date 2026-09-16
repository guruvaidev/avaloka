import { ArrowLeft, Loader2, AlertCircle } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

import { Button } from "@/components/ui/button";
import { backendApi, type BackendModelDetail, type BackendTrainingHistoryPoint } from "@/lib/api/backendApi";
import { InferencePanel } from "@/components/dashboard/InferencePanel";


interface Props {
  runId: string;
  onBack: () => void;
}

function titleCase(v: string): string {
  return v ? v.charAt(0).toUpperCase() + v.slice(1) : v;
}

function formatMetric(n: unknown): string | null {
  if (typeof n !== "number" || !Number.isFinite(n)) return null;
  return n.toFixed(4);
}

function formatCount(n: unknown): string | null {
  if (typeof n === "number" && Number.isFinite(n)) return Math.trunc(n).toLocaleString();
  if (typeof n === "string" && n !== "" && Number.isFinite(Number(n))) return Math.trunc(Number(n)).toLocaleString();
  return null;
}

function pickString(raw: Record<string, unknown>, ...keys: string[]): string | null {
  for (const k of keys) {
    const v = raw[k];
    if (v !== undefined && v !== null && v !== "") return String(v);
  }
  return null;
}

function pickNumber(raw: Record<string, unknown>, ...keys: string[]): number | undefined {
  for (const k of keys) {
    const v = raw[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v !== "" && Number.isFinite(Number(v))) return Number(v);
  }
  return undefined;
}

function extractHistory(raw: Record<string, unknown>): BackendTrainingHistoryPoint[] {
  const candidates = [
    raw.training_history,
    raw.history,
    (raw.metrics as Record<string, unknown> | undefined)?.training_history,
  ];
  for (const c of candidates) {
    if (Array.isArray(c) && c.length > 0) {
      return c
        .map((p, i) => {
          const r = (p ?? {}) as Record<string, unknown>;
          const epoch = pickNumber(r, "epoch", "step", "index") ?? i + 1;
          const train_loss = pickNumber(r, "train_loss", "training_loss", "loss");
          const val_loss = pickNumber(r, "val_loss", "validation_loss", "val_loss_value");
          const accuracy = pickNumber(r, "accuracy", "val_accuracy", "acc");
          const f1_score = pickNumber(r, "f1_score", "f1", "val_f1_score", "val_f1");
          return { epoch, train_loss, val_loss, accuracy, f1_score };
        })
        .filter((p) => p.train_loss !== undefined || p.val_loss !== undefined || p.accuracy !== undefined || p.f1_score !== undefined);
    }
  }
  return [];
}

export function ModelDetailView({ runId, onBack }: Props) {
  const q = useQuery<BackendModelDetail>({
    queryKey: ["model", runId],
    queryFn: () => backendApi.getModel(runId),
  });

  const raw = (q.data ?? {}) as Record<string, unknown>;
  const metricsBag = ((raw.metrics as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>;

  const name = pickString(raw, "name", "model_name", "run_name", "registered_model_name", "display_name");
  const description = pickString(raw, "description", "summary", "notes");
  const modelType = pickString(raw, "model_type", "task_type", "type");
  const version = pickString(raw, "version", "latest_version", "model_version", "current_version");

  const trainLoss = pickNumber(raw, "final_train_loss", "train_loss") ?? pickNumber(metricsBag, "final_train_loss", "train_loss");
  const valLoss = pickNumber(raw, "final_val_loss", "val_loss", "validation_loss") ?? pickNumber(metricsBag, "final_val_loss", "val_loss", "validation_loss");
  const accuracy = pickNumber(raw, "final_accuracy", "accuracy") ?? pickNumber(metricsBag, "final_accuracy", "accuracy");
  const f1 = pickNumber(raw, "final_f1_score", "f1_score", "f1") ?? pickNumber(metricsBag, "final_f1_score", "f1_score", "f1");

  const r2 = pickNumber(raw, "final_r2", "r2_score", "r2", "r squared") ?? pickNumber(metricsBag, "final_r2", "r2_score", "r2");
  const rmse = pickNumber(raw, "final_rmse", "rmse", "root_mean_squared_error") ?? pickNumber(metricsBag, "final_rmse", "rmse", "root_mean_squared_error");
  const mae = pickNumber(raw, "final_mae", "mae", "mean_absolute_error") ?? pickNumber(metricsBag, "final_mae", "mae", "mean_absolute_error");

  const rowsProcessed = pickNumber(raw, "rows_processed") ?? pickNumber(metricsBag, "rows_processed");
  const numEpochs = pickNumber(raw, "num_epochs_trained", "epochs_trained", "num_epochs") ?? pickNumber(metricsBag, "num_epochs_trained");
  const numFeatures = pickNumber(raw, "num_features") ?? pickNumber(metricsBag, "num_features");
  const numClasses = pickNumber(raw, "num_classes") ?? pickNumber(metricsBag, "num_classes");

  const history = q.data ? extractHistory(raw) : [];
  const hasHistory = history.length > 0;

  const isRegression = /regress/i.test(modelType ?? "") || r2 !== undefined || rmse !== undefined || mae !== undefined;

  const metricTiles = (
    isRegression
      ? [
          { label: "Average Error (MAE)", value: formatMetric(mae) },
          { label: "Large-Error Score (RMSE)", value: formatMetric(rmse) },
          {
            label: "Explained Variation (R²)",
            value: typeof r2 === "number" && Number.isFinite(r2) ? `${(r2 * 100).toFixed(2)}%` : null,
          },
          { label: "Final Train Loss", value: formatMetric(trainLoss) },
          { label: "Final Validation Loss", value: formatMetric(valLoss) },
        ]

      : [
          { label: "Final Train Loss", value: formatMetric(trainLoss) },
          { label: "Final Validation Loss", value: formatMetric(valLoss) },
          { label: "Final Accuracy", value: formatMetric(accuracy) },
          { label: "Final F1 Score", value: formatMetric(f1) },
        ]
  ).filter((m) => m.value !== null);


  const runInfoTiles = [
    { label: "Rows Processed", value: formatCount(rowsProcessed) },
    { label: "Number of Epochs", value: formatCount(numEpochs) },
    { label: "Number of Features", value: formatCount(numFeatures) },
    { label: "Number of Classes", value: formatCount(numClasses) },
  ].filter((m) => m.value !== null);

  return (
    <div className="flex flex-1 flex-col gap-4 p-4 sm:p-6">
      <div className="flex items-center gap-3 rounded-2xl border border-border bg-white px-5 py-4">
        <Button variant="ghost" size="icon" onClick={onBack} aria-label="Back" className="h-8 w-8">
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h1 className="text-base font-semibold text-foreground">
          {name ?? (q.isLoading ? "Loading…" : "—")}
        </h1>
      </div>

      {q.isLoading ? (
        <div className="flex flex-1 items-center justify-center rounded-2xl border border-border bg-white p-10">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : q.isError ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-2xl border border-border bg-white p-10 text-center">
          <AlertCircle className="h-6 w-6 text-red-500" />
          <div className="text-sm font-medium text-foreground">Failed to load model</div>
          <div className="text-xs text-muted-foreground">
            {q.error instanceof Error ? q.error.message : "Unknown error"}
          </div>
          <Button variant="outline" size="sm" onClick={() => q.refetch()}>
            Retry
          </Button>
        </div>
      ) : q.data ? (
        <div className="flex flex-1 flex-col gap-4">
          <div className="rounded-2xl border border-border bg-white p-6">
            <div className="grid grid-cols-2 gap-x-8 gap-y-5">
              {modelType && <Field label="Task Type" value={titleCase(modelType)} />}
              {version && <Field label="Latest version" value={`Version ${version}`} />}
              {description && <Field label="Description" value={description} full />}
            </div>

            {metricTiles.length > 0 && (
              <>
                <h2 className="mt-7 text-base font-semibold text-foreground">Metrics</h2>
                <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {metricTiles.map((m) => (
                    <div key={m.label} className="rounded-xl border border-border bg-[#f9fafb] p-4">
                      <div className="text-xs text-muted-foreground">{m.label}</div>
                      <div className="mt-1 text-xl font-semibold text-foreground">{m.value}</div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {runInfoTiles.length > 0 && (
              <>
                <h2 className="mt-7 text-base font-semibold text-foreground">Run Info</h2>
                <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {runInfoTiles.map((m) => (
                    <div key={m.label} className="rounded-xl border border-border bg-[#f9fafb] p-4">
                      <div className="text-xs text-muted-foreground">{m.label}</div>
                      <div className="mt-1 text-xl font-semibold text-foreground">{m.value}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {hasHistory && (() => {
            const hasAccuracy = history.some((p) => p.accuracy !== undefined);
            const hasF1 = history.some((p) => p.f1_score !== undefined);
            return (
            <div className="rounded-2xl border border-border bg-white p-6">
              <h2 className="text-base font-semibold text-foreground">Training metrics</h2>
              <div className="mt-4 h-[340px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 10, right: 16, left: 0, bottom: 24 }}>
                    <CartesianGrid stroke="#eef2f7" />
                    <XAxis
                      dataKey="epoch"
                      tick={{ fontSize: 11, fill: "#6b7280" }}
                      axisLine={{ stroke: "#e5e7eb" }}
                      label={{ value: "Epoch", position: "insideBottom", offset: -10, fontSize: 11, fill: "#6b7280" }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#6b7280" }}
                      axisLine={{ stroke: "#e5e7eb" }}
                      label={{ value: "Value", angle: -90, position: "insideLeft", fontSize: 11, fill: "#6b7280" }}
                    />
                    <Tooltip />
                    <Legend verticalAlign="bottom" iconType="square" />
                    <Line type="monotone" dataKey="train_loss" name="Training loss" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
                    <Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#fca5a5" strokeWidth={2} dot={{ r: 2 }} />
                    {hasAccuracy && <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} />}
                    {hasF1 && <Line type="monotone" dataKey="f1_score" name="F1 score" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2 }} />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            );
          })()}

          <InferencePanel runId={runId} raw={raw} />
        </div>

      ) : null}
    </div>
  );
}

function Field({ label, value, full }: { label: string; value: string; full?: boolean }) {
  return (
    <div className={full ? "col-span-2" : undefined}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-medium text-foreground">{value}</div>
    </div>
  );
}
