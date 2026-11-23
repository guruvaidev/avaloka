import { useState } from "react";
import type { CrossValidation, CvFold } from "@/api/inference";

/** Counts with row-% underneath; rows are the actual class, columns the predicted one. */
export function ConfusionMatrix({ cv }: { cv: CrossValidation }) {
  const [fold, setFold] = useState<number | "all">("all");
  if (cv.status === "skipped" || cv.status === "failed") {
    return <p className="text-sm text-muted-foreground" data-testid="cv-note">Cross-validation {cv.status}{cv.reason ? `: ${cv.reason}` : "."}</p>;
  }
  if (!cv.aggregate) return null;
  const isClassification = Array.isArray((cv.aggregate as { matrix?: unknown }).matrix);
  if (!isClassification) {
    return (
      <div data-testid="cv-regression">
        <p className="text-sm text-muted-foreground">Confusion matrices apply to classification models. Fold metrics ({cv.k}-fold, {cv.splitter}):</p>
        <table className="mt-2 w-full text-xs tabular-nums"><thead><tr><th className="text-left">Fold</th><th className="text-right">RMSE</th><th className="text-right">MAE</th><th className="text-right">R²</th></tr></thead>
          <tbody>{(cv.folds as CvFold[]).map((f) => <tr key={f.fold}><td>{f.fold + 1}</td><td className="text-right">{fmt(f.rmse)}</td><td className="text-right">{fmt(f.mae)}</td><td className="text-right">{fmt(f.r2)}</td></tr>)}</tbody></table>
      </div>
    );
  }
  const source = fold === "all" ? cv.aggregate : (cv.folds as CvFold[])[fold];
  const labels = source.labels ?? [];
  const matrix = source.matrix ?? [];
  const normalized = source.normalized ?? [];
  return (
    <div data-testid="cv-matrix">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium">Confusion matrix</span>
        <span className="text-muted-foreground">({cv.k}-fold, {cv.splitter}; accuracy {pct(cv.aggregate.accuracy_mean)} ± {pct(cv.aggregate.accuracy_std)})</span>
        <select aria-label="Fold" className="ml-auto rounded border bg-background px-2 py-1 text-xs" value={fold} onChange={(e) => setFold(e.target.value === "all" ? "all" : Number(e.target.value))}>
          <option value="all">All folds (sum)</option>
          {(cv.folds as CvFold[]).map((f) => <option key={f.fold} value={f.fold}>Fold {f.fold + 1}</option>)}
        </select>
      </div>
      <table className="w-full border-collapse text-xs tabular-nums" role="table">
        <thead><tr><th className="p-1 text-left text-muted-foreground">actual ╲ predicted</th>{labels.map((l) => <th key={l} className="p-1 text-right">{l}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="p-1 text-left font-medium">{labels[i]}</th>
              {row.map((count, j) => {
                const share = normalized[i]?.[j] ?? 0;
                return (
                  <td key={j} data-testid={`cv-cell-${i}-${j}`} className="p-1 text-right align-top" style={{ background: `color-mix(in oklch, var(--inf-blue) ${Math.round(share * 60)}%, transparent)` }}>
                    <div className="font-semibold">{count}</div>
                    <div className="text-[10px] text-muted-foreground">{pct(share)}</div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11px] text-muted-foreground">Total {source.n} rows{fold === "all" ? ` across ${cv.k} folds` : ""}.</p>
    </div>
  );
}
const fmt = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v.toLocaleString("en-US", { maximumFractionDigits: 3 }) : "—");
const pct = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
