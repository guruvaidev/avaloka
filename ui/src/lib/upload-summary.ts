/**
 * Builds the first assistant chat message shown when an analysis opens right
 * after an upload or cloud import (or is reopened with an empty thread).
 * Uses ONLY the upload response / persisted dataset context — no API call.
 * Every part is optional: a failing part is logged and skipped, never the message.
 */
function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch (err) {
    console.error("[auto-insight-message]", err);
    return fallback;
  }
}

function vizConfig(d: any): any {
  return d?.visualization_config ?? d?.visualization_configs ?? d?.viz_config ?? null;
}

function columnCount(d: any): number | null {
  const shapeCols = d?.sample_statistics?.data_shape?.columns;
  if (typeof shapeCols === "number") return shapeCols;
  const cols = d?.columns ?? d?.schema;
  if (Array.isArray(cols)) return cols.length;
  if (cols && typeof cols === "object") return Object.keys(cols).length;
  const sample = Array.isArray(d?.samples) ? d.samples?.[0] : Array.isArray(d?.rows) ? d.rows?.[0] : null;
  return sample && typeof sample === "object" ? Object.keys(sample).length : null;
}

function positiveNumber(value: unknown): number | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Real row count: total_rows or sample_statistics.data_shape.rows — never rows_sampled. */
function totalRows(d: any): number | null {
  return (
    positiveNumber(d?.total_rows) ??
    positiveNumber(d?.sample_statistics?.data_shape?.rows) ??
    positiveNumber(vizConfig(d)?.dataset?.total_rows)
  );
}

/** Preview lengths only reject ambiguous legacy counts; they never supply a count. */
function sampledRows(d: any): number | null {
  const cfg = vizConfig(d);
  const authoritative = positiveNumber(cfg?.dataset?.rows_sampled) ?? positiveNumber(d?.sample_statistics?.sample_size);
  if (authoritative != null) return authoritative;
  const legacy = positiveNumber(d?.rows_sampled);
  const matchesPreview = [d?.samples, d?.rows].some((p) => Array.isArray(p) && p.length === legacy);
  return matchesPreview ? null : legacy;
}

const fmt = (n: number) => n.toLocaleString("en-US");

function insightSentences(d: any): string[] {
  const charts = Array.isArray(vizConfig(d)?.charts) ? vizConfig(d).charts : [];
  return charts
    .map((c: any) => (typeof c?.insight === "string" ? c.insight.trim() : ""))
    .flatMap((insight: string) => insight.split(/(?<=[.!?])\s+/))
    .filter(Boolean);
}

export function buildUploadSummaryMessage(up: any, fallbackName?: string): string | null {
  if (!up || typeof up !== "object") return null;
  const datasets: any[] = Array.isArray(up?.datasets) && up.datasets.length ? up.datasets : [up];
  const blocks: string[] = [];
  let insightsLeft = 3;
  const seenSentences = new Set<string>();
  const seenPrefixes = new Set<string>();

  datasets.forEach((d, i) => {
    const merged = i === 0 ? { ...up, ...(d ?? {}) } : (d ?? {});
    const name =
      merged?.filename || (i === 0 ? fallbackName : null) || merged?.alias || merged?.dataset_id || "Dataset";
    const rows = safe(() => totalRows(merged), null);
    const cols = safe(() => columnCount(merged), null);
    const lines: string[] = [
      `${name}: ${rows != null ? fmt(rows) : "Unknown"} rows, ${cols != null ? fmt(cols) : "Unknown"} columns.`,
    ];

    const sentences = safe(
      () =>
        insightSentences(merged)
          .filter((sentence: string) => {
            const prefix = sentence.slice(0, 40);
            if (seenSentences.has(sentence) || seenPrefixes.has(prefix)) return false;
            seenSentences.add(sentence);
            seenPrefixes.add(prefix);
            return true;
          })
          .slice(0, insightsLeft),
      [] as string[],
    );
    insightsLeft -= sentences.length;
    if (sentences.length) lines.push(sentences.join(" "));

    const scope = safe(() => {
      if (rows == null) return null;
      if (vizConfig(merged)?.dataset?.charts_scope === "full_file") return `Auto Insights use all ${fmt(rows)} rows.`;
      const sampled = sampledRows(merged);
      if (sampled == null) return null;
      return sampled < rows
        ? `Auto Insights are based on a random sample of ${fmt(sampled)} of ${fmt(rows)} rows; chat answers use the full file.`
        : `Auto Insights use all ${fmt(rows)} rows.`;
    }, null);
    if (scope) lines.push(scope);
    blocks.push(lines.join("\n\n"));
  });

  return blocks.length ? blocks.join("\n\n") : null;
}
