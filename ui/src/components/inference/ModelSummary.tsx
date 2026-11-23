import { Link } from "@tanstack/react-router";
import { Copy, Loader2, Play, Square } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { formatDate, formatNumber, formatPercent, formatVersion, type InferenceModel } from "@/api/inference";
import { baselineImprovement } from "@/lib/inference";

export function ServiceBadge({ running, starting = false, failed = false, asleep = false }: { running: boolean; starting?: boolean; failed?: boolean; asleep?: boolean }) {
  const cls = starting ? "inf-status-starting" : running ? "inf-status-running" : failed ? "inf-status-warning" : "inf-status-stopped";
  return (
    <span className={`inf-badge ${cls}`} role="status">
      <span aria-hidden="true" className="size-1.5 rounded-full bg-current" />
      {starting ? "Starting…" : running ? (asleep ? "Ready (sleeping)" : "Service running") : failed ? "Service failed" : "Service stopped"}
    </span>
  );
}

export async function copyText(value: string, message: string) {
  try { await navigator.clipboard.writeText(value); toast(message); }
  catch { toast.error("Copy unavailable. Select and copy the text."); }
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-inf-tile p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}

export function ModelSummary({ model, starting, busy, onStart, onStop }: { model: InferenceModel; starting: boolean; busy: boolean; onStart: () => void; onStop: () => void }) {
  const tiles: { label: string; value: string }[] = model.isClassification
    ? [
        { label: "Accuracy", value: formatPercent(model.accuracy) },
        { label: "Precision", value: formatPercent(model.precision) },
        { label: "Recall", value: formatPercent(model.recall) },
        { label: "F1", value: formatPercent(model.f1) },
      ]
    : [
        { label: "Average error (MAE)", value: formatNumber(model.mae) },
        { label: "Large-error score (RMSE)", value: formatNumber(model.rmse) },
        { label: "Explained variation (R²)", value: formatPercent(model.r2) },
        { label: "Training rows", value: formatNumber(model.rowsProcessed, 0) },
      ];
  const shown = tiles.filter((t) => t.value !== "—");
  const beat = model.isRegression ? baselineImprovement(model.mae, model.baselineMae) : undefined;
  const featureCount = model.inputFeatureCount;
  const footer = [
    model.target && `Target ${model.target}`,
    featureCount !== undefined && `${formatNumber(featureCount, 0)} input features${model.encodedFeatureCount !== undefined ? ` (${formatNumber(model.encodedFeatureCount, 0)} after encoding)` : ""}`,
    model.epochs !== undefined && `${formatNumber(model.epochs, 0)} epochs`,
    model.splitter && `Validation ${model.splitter}`,
    model.trainLoss !== undefined && `Train loss ${formatNumber(model.trainLoss, 4)}`,
    model.valLoss !== undefined && `Validation loss ${formatNumber(model.valLoss, 4)}`,
  ].filter((x): x is string => typeof x === "string");

  return (
    <section className="inf-card" aria-label="Model summary">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="break-all font-mono text-base font-semibold">{model.name ?? "Unnamed run"}</h2>
            {model.version && <span className="inf-badge inf-status-stopped">{formatVersion(model.version)}</span>}
            {model.taskType && <span className="inf-badge inf-status-starting">{model.taskType}</span>}
            <ServiceBadge running={model.serviceRunning} starting={starting || model.serviceStarting} failed={model.endpointStatus === "failed"} asleep={model.serviceAsleep} />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              Run <code className="font-mono text-foreground">{model.runId ?? "—"}</code>
              {model.runId && (
                <Button variant="ghost" size="icon" className="size-11" aria-label="Copy run ID" onClick={() => copyText(model.runId ?? "", "Run ID copied")}><Copy className="size-4" /></Button>
              )}
            </span>
            <span>Trained {formatDate(model.createdAt)}</span>
            {model.runId && (
              <Link to="/configurations" search={{ run_id: model.runId }} className="inline-flex min-h-11 items-center font-medium text-primary underline-offset-4 hover:underline">View in Configurations</Link>
            )}
          </div>
        </div>
        {model.serviceRunning ? (
          <Button variant="outline" className="min-h-11" disabled={busy} onClick={onStop}><Square />Stop service</Button>
        ) : (
          <Button variant="outline" className="min-h-11" disabled={busy || starting} onClick={onStart}>{starting ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : <Play />}Start service</Button>
        )}
      </div>
      {shown.length > 0 && <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">{shown.map((t) => <Tile key={t.label} {...t} />)}</div>}
      {beat !== undefined && <p className="mt-3 text-sm inf-success-text">Beats baseline (mean prediction) by {formatPercent(beat, 1)}</p>}
      {footer.length > 0 && <p className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">{footer.join(" · ")}</p>}
    </section>
  );
}
