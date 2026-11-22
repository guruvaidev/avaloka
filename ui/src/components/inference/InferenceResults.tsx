import { Copy, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatCurrency, formatNumber, formatPercent, type Prediction } from "@/api/inference";
import { changedFeatures, download, exportResults, predictionRange, type InferenceRun } from "@/lib/inference";
import { copyText } from "./ModelSummary";

function formatValue(run: InferenceRun, value: Prediction["value"]): string {
  if (value === null) return "—";
  if (typeof value !== "number") return String(value);
  return run.model.target === "median_house_value" ? formatCurrency(value) : formatNumber(value);
}
function formatRange(run: InferenceRun, range?: [number, number], short = false) {
  if (range && short && run.model.target === "median_house_value") return range.map((n) => n.toLocaleString("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 0 }).replace("K", "k")).join(" – ");
  return range ? `${formatValue(run, range[0])} – ${formatValue(run, range[1])}` : "—";
}
function formatDuration(ms: number) { return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`; }

export function InferenceResults({ run, selected, onSelect }: { run: InferenceRun | null; selected: number; onSelect: (row: number) => void }) {
  if (!run || run.predictions.length === 0) {
    return (
      <section className="inf-card" aria-live="polite" aria-label="Results">
        <h2 className="font-semibold">Results</h2>
        <p className="py-6 text-sm text-muted-foreground">Run inference to see predictions here.</p>
      </section>
    );
  }
  const index = Math.min(Math.max(0, selected), run.predictions.length - 1);
  const prediction = run.predictions[index];
  const warnings = run.warnings[index] ?? [];
  const target = run.model.target ?? "value";
  const typical = predictionRange(prediction?.value, run.model.mae);
  const wider = predictionRange(prediction?.value, run.model.rmse);
  const spread = Math.max(run.model.mae ?? 0, run.model.rmse ?? 0);
  const typicalPct = spread > 0 && run.model.mae !== undefined ? (run.model.mae / spread) * 100 : 0;
  const rowCount = run.rows.length;

  return (
    <section className="inf-card" aria-live="polite" aria-label="Results">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold">Results</h2>
          <p className="mt-1 text-xs text-muted-foreground">{rowCount.toLocaleString("en-US")} {rowCount === 1 ? "row" : "rows"} · {new Date(run.time).toLocaleTimeString("en-US")} · {formatDuration(run.duration)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" className="min-h-11" onClick={() => copyText(JSON.stringify(run.predictions.map((p) => p.value), null, 2), "JSON copied")}><Copy />Copy JSON</Button>
          <Button variant="outline" className="min-h-11" onClick={() => download("inference-results.csv", exportResults(run), "text/csv")}><Download />Export CSV</Button>
        </div>
      </div>

      <div className="mt-5 rounded-lg bg-inf-tile p-4">
        <p className="text-xs text-muted-foreground">Row {index + 1} · predicted {target}</p>
        <p className="mt-1 break-all text-3xl font-semibold tabular-nums">{formatValue(run, prediction?.value ?? null)}</p>
        {typical && (
          <>
            <div className="inf-range mt-4" aria-hidden="true"><span style={{ left: `${50 - typicalPct / 2}%`, width: `${typicalPct}%` }} /><i /></div>
            <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
              <div><dt className="text-muted-foreground">Typical range (± MAE)</dt><dd className="font-medium tabular-nums">{formatRange(run, typical)}</dd></div>
              {wider && <div><dt className="text-muted-foreground">Wider range (± RMSE)</dt><dd className="font-medium tabular-nums">{formatRange(run, wider)}</dd></div>}
            </dl>
          </>
        )}
        {run.model.isRegression && run.model.mae === undefined && <p className="mt-3 text-xs text-muted-foreground">Error range unavailable for this run.</p>}
        {prediction && prediction.probabilities.length > 0 && (
          <div className="mt-4 space-y-2">
            {prediction.probabilities.map(([label, p]) => (
              <div key={label} className="text-xs">
                <div className="flex justify-between"><span>{label}</span><span className="tabular-nums">{formatPercent(p, 1)}</span></div>
                <div className="mt-1 h-2 rounded bg-inf-active"><div className="h-2 rounded bg-inf-blue" style={{ width: `${Math.min(100, Math.max(0, p * 100))}%` }} /></div>
              </div>
            ))}
          </div>
        )}
        {warnings.length > 0 && (
          <div className="inf-warning mt-4" role="note">
            <p className="font-semibold">Input looks inconsistent. Prediction may be unreliable.</p>
            <ul className="mt-1 list-disc pl-4">{warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          </div>
        )}
      </div>

      <div className="mt-5 max-h-96 overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-card text-muted-foreground">
            <tr><th className="px-3 py-2 font-medium">Row</th><th className="px-3 py-2 font-medium">Prediction</th>{run.model.mae !== undefined && <th className="px-3 py-2 font-medium">Typical range</th>}{rowCount > 1 && <th className="px-3 py-2 font-medium">Changed vs previous row</th>}<th className="px-3 py-2 font-medium">Status</th></tr>
          </thead>
          <tbody>
            {run.predictions.slice(0, 1000).map((p, i) => {
              const changed = changedFeatures(run.rows, i);
              const w = run.warnings[i]?.length ?? 0;
              return (
                <tr key={i} className={`border-t border-border ${i === index ? "bg-inf-active" : ""}`}>
                  <td className="px-1"><Button variant="link" className="min-h-11 px-2" aria-pressed={i === index} onClick={() => onSelect(i)}>Row {i + 1}</Button></td>
                  <td className="px-3 tabular-nums">{formatValue(run, p.value)}</td>
                  {run.model.mae !== undefined && <td className="whitespace-nowrap px-3 tabular-nums">{formatRange(run, predictionRange(p.value, run.model.mae), true)}</td>}
                  {rowCount > 1 && <td className="px-3">{changed.join(", ") || "—"}</td>}
                  <td className="px-3">{w ? <span className="inf-badge inf-status-warning">{w} {w === 1 ? "warning" : "warnings"}</span> : <span className="inf-badge inf-status-running">OK</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {run.predictions.length > 1000 && <p className="px-3 py-2 text-xs text-muted-foreground">Showing the first 1,000 rows. Export CSV for all {run.predictions.length.toLocaleString("en-US")}.</p>}
      </div>
    </section>
  );
}
