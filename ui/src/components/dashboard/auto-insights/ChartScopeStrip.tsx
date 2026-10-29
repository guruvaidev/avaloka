import { Check, ChevronDown, Info } from "lucide-react";
import { Dialog, DialogTrigger, Popover } from "react-aria-components";
import { Button } from "@/components/base/buttons/button";

type Sample = string | { name?: string; sample_name?: string; id?: string };
type ScopeDataset = {
  charts_scope?: string;
  rows_charted?: number;
  rows_sampled?: number;
  total_rows?: number;
  selected_sample_name?: string;
  available_samples?: Sample[];
};

const count = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
const rows = (value: number) => new Intl.NumberFormat("en-US").format(value);
const share = (value: number, total: number | undefined) => total != null && total > 0 ? (value / total * 100).toFixed(1) : undefined;
export function sampleLabel(name: string): string {
  if (name === "full") return "Full sample";
  if (name.startsWith("strat_")) return `Stratified by ${name.slice(6)}`;
  const readable = name.replace(/_/g, " ");
  return readable.charAt(0).toUpperCase() + readable.slice(1);
}

export function ChartScopeStrip({ dataset = {} }: { dataset?: ScopeDataset }) {
  const total = count(dataset.total_rows);
  const charted = count(dataset.rows_charted);
  const sampled = count(dataset.rows_sampled);
  const full = dataset.charts_scope === "full_file";
  const sample = dataset.charts_scope === "sample";
  const chartPercent = charted == null ? undefined : share(charted, total);
  const samplePercent = sampled == null ? undefined : share(sampled, total);
  const name = dataset.selected_sample_name;
  const available = (Array.isArray(dataset.available_samples) ? dataset.available_samples : []).flatMap((item) => {
    const key = typeof item === "string" ? item : item?.name ?? item?.sample_name ?? item?.id;
    return typeof key === "string" && key.trim() ? [key] : [];
  });
  const allRows = total ?? charted;
  const numbers = full
    ? `Numbers computed on all ${allRows == null ? "rows (count not supplied)" : `${rows(allRows)} rows`}`
    : sample
      ? charted == null ? "Numbers from a sample (row count not supplied)"
        : `Numbers from a sample of ${rows(charted)}${total == null ? "" : ` of ${rows(total)}`} rows${chartPercent == null ? "" : ` (${chartPercent}%)`}`
      : "Chart numbers scope not supplied";
  const tooltip = full || sample
    ? `Charts are picked using a sample of the data; the values you see are then recalculated on ${full ? "all rows" : "the sample"}.`
    : "The dataset does not specify whether chart values use all rows or a sample.";
  return <div className="ai-chart-scope" aria-label="Chart data scope">
    <span className={`ai-scope-pill ${full ? "ai-scope-full" : sample ? "ai-scope-sample" : "ai-scope-neutral"}`} title={tooltip} tabIndex={0}>
      {full ? <Check size={13} aria-hidden="true" /> : <Info size={13} aria-hidden="true" />}<span>{numbers}</span>
    </span>
    {sampled != null && <span className="ai-scope-pill ai-scope-neutral">{name
      ? `Chart chosen from sample: ${sampleLabel(name)} · ${rows(sampled)} rows${samplePercent == null ? "" : ` (${samplePercent}% of file)`}`
      : `Chart chosen from a ${rows(sampled)}-row sample${samplePercent == null ? "" : ` (${samplePercent}% of file)`}`}</span>}
    {available.length >= 2 && <DialogTrigger>
      <Button color="secondary" className="ai-scope-pill ai-scope-options" iconTrailing={ChevronDown}>{available.length} samples available</Button>
      <Popover placement="bottom start" className="ai-sample-popover">
        <Dialog aria-label="Available samples" className="ai-sample-dialog"><ul>{available.map((key, index) => <li key={`${key}-${index}`}><span>{sampleLabel(key)}</span>{key === name && <Check size={14} aria-label="Current sample" />}</li>)}</ul></Dialog>
      </Popover>
    </DialogTrigger>}
  </div>;
}