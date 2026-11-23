import Papa from "papaparse";
import { formatCurrency, formatNumber, toNumber, type Feature, type InferenceModel, type InputRow, type Prediction } from "@/api/inference";

export interface InferenceRun { id: string; time: number; duration: number; model: InferenceModel; rows: InputRow[]; predictions: Prediction[]; warnings: string[][] }

const isBlank = (v: unknown) => v === undefined || v === null || (typeof v === "string" && v.trim() === "");

export function rowWarnings(row: InputRow, features: Feature[]): { required: string[]; warnings: string[]; fields: Record<string, string[]> } {
  const fields: Record<string, string[]> = {};
  const required: string[] = [];
  for (const f of features) {
    const v = row[f.name];
    const messages: string[] = [];
    if (isBlank(v)) { required.push(f.name); messages.push("Required"); }
    else if (f.numeric) {
      const n = toNumber(v);
      if (n === undefined) messages.push("Enter a number");
      else {
        if (f.max !== undefined && n > f.max) messages.push(`Above training max ${f.max}`);
        if (f.min !== undefined && n < f.min) messages.push(`Below training min ${f.min}`);
      }
    }
    fields[f.name] = messages;
  }
  for (const [a, b] of [["total_bedrooms", "total_rooms"], ["households", "population"]] as const) {
    const left = toNumber(row[a]);
    const right = toNumber(row[b]);
    if (left !== undefined && right !== undefined && left > right) (fields[a] ??= []).push(`Can't exceed ${b}`);
  }
  const warnings: string[] = [];
  for (const [name, messages] of Object.entries(fields)) for (const m of messages) if (m !== "Required") warnings.push(`${name}: ${m}`);
  return { required, warnings, fields };
}

export function validateRows(value: unknown, features: Feature[]): InputRow[] {
  const rows = Array.isArray(value) ? value : [value];
  if (!rows.length) throw new Error("Add at least one row");
  if (rows.length > 10000) throw new Error("For more than 10,000 rows, split the file.");
  return rows.map((raw, i) => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Row ${i + 1} must be an object`);
    const row = raw as Record<string, unknown>;
    for (const f of features) {
      if (isBlank(row[f.name])) throw new Error(`Row ${i + 1} is missing ${f.name}`);
    }
    const out: InputRow = {};
    for (const [key, v] of Object.entries(row)) {
      if (v !== null && typeof v !== "string" && typeof v !== "number" && typeof v !== "boolean") throw new Error(`Row ${i + 1}: ${key} must be a single value`);
      out[key] = features.some((f) => f.name === key && f.numeric) ? toNumber(v) ?? v : v;
    }
    return out;
  });
}

export function parseJson(text: string, features: Feature[]): InputRow[] {
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("Invalid JSON: check commas, quotes and brackets"); }
  return validateRows(value, features);
}

export function parseCsv(text: string, features: Feature[]): InputRow[] {
  const parsed = Papa.parse<Record<string, string>>(text, { header: true, skipEmptyLines: "greedy", transformHeader: (h) => h.trim() });
  const error = parsed.errors.find((e) => e.code !== "UndetectableDelimiter");
  if (error) throw new Error(`CSV row ${(error.row ?? 0) + 1}: ${error.message}`);
  const columns = parsed.meta.fields ?? [];
  const missing = features.filter((f) => !columns.includes(f.name));
  if (missing.length) throw new Error(`Missing columns: ${missing.map((f) => f.name).join(", ")}`);
  return validateRows(parsed.data, features);
}

export const chunksOf = <T,>(rows: T[], size = 500): T[][] => Array.from({ length: Math.ceil(rows.length / size) }, (_, i) => rows.slice(i * size, (i + 1) * size));
export const keepRecent = (runs: InferenceRun[], run: InferenceRun) => [run, ...runs].slice(0, 20);
export function predictionRange(value: unknown, error: number | undefined): [number, number] | undefined {
  const n = toNumber(value);
  return n === undefined || error === undefined || !Number.isFinite(error) ? undefined : [Math.max(0, Math.round((n - error) * 100) / 100), Math.max(0, Math.round((n + error) * 100) / 100)];
}
export function changedFeatures(rows: InputRow[], index: number): string[] {
  const current = rows[index];
  const previous = rows[index - 1];
  if (!current || !previous) return [];
  return [...new Set([...Object.keys(current), ...Object.keys(previous)])].filter((name) => current[name] !== previous[name]);
}
export function recentSummary(run: InferenceRun): string {
  const format = (value: Prediction["value"]) => typeof value === "number" ? run.model.target === "median_house_value" ? formatCurrency(value) : formatNumber(value) : value === null ? "—" : String(value);
  if (run.rows.length === 1) return format(run.predictions[0]?.value ?? null);
  const numbers = run.predictions.flatMap((p) => typeof p.value === "number" ? [p.value] : []);
  const summary = numbers.length ? `${format(Math.min(...numbers))} – ${format(Math.max(...numbers))}` : [...new Set(run.predictions.map((p) => format(p.value)))].join(" / ");
  const warnings = run.warnings.reduce((count, row) => count + row.length, 0);
  return `${summary}${warnings ? ` · ${warnings} warnings` : ""}`;
}
export function safeErrorDetails(error: unknown) {
  const e = error as { status?: unknown; call?: unknown; detail?: unknown } | null;
  return {
    status: typeof e?.status === "number" ? e.status === 0 ? "Network error" : String(e.status) : "Unavailable",
    call: typeof e?.call === "string" && ["inference", "configure-inference-service", "stop-inference-service"].includes(e.call) ? e.call : "inference",
    detail: typeof e?.detail === "string" && e.detail.length <= 300 && !/\b(traceback|stack trace)\b|\n\s*(?:at |File ")/i.test(e.detail) ? e.detail : undefined,
  };
}
export function baselineImprovement(mae?: number, baselineMae?: number): number | undefined {
  if (mae === undefined || baselineMae === undefined || baselineMae === 0) return undefined;
  return (baselineMae - mae) / baselineMae;
}
export function exportResults(run: InferenceRun): string {
  return Papa.unparse(run.rows.map((row, i) => {
    const value = run.predictions[i]?.value ?? null;
    const typical = predictionRange(value, run.model.mae);
    const wider = predictionRange(value, run.model.rmse);
    return { ...row, prediction: value, typical_min: typical?.[0], typical_max: typical?.[1], wider_min: wider?.[0], wider_max: wider?.[1], warnings: run.warnings[i]?.join("; ") ?? "" };
  }), { escapeFormulae: true });
}
export function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
