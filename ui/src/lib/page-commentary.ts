// Page commentary: what the chat says about the Auto Insights on this page.
//
// 1.6 has no way to label model-written words as unchecked commentary (2.0's
// reply_segments has not been ported), so every sentence here is written by
// code from figures the profiler computed: row and column counts, column
// types, missing cells and the chart count. Model-written profile fields
// (domain, readiness_score, readiness_reasoning, red_flags, quick_wins,
// analysis_suggestions, predictive_value, column meanings) are never read.

export type CommentaryLine = { key: string; text: string };

export const PAGE_COMMENTARY_LABEL = "Computed from your data profile";

type ProfileColumn = { name?: unknown; dtype?: unknown; missing_pct?: unknown };

function finite(value: unknown): number | null {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(n: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(n);
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${fmt(n)} ${n === 1 ? one : many}`;
}

function kindOf(dtype: unknown): "number" | "date" | "category" {
  const t = String(dtype ?? "").toLowerCase();
  if (/date|time/.test(t)) return "date";
  if (/int|float|double|decimal|number|numeric/.test(t)) return "number";
  return "category";
}

/** Columns more than this share empty are called out as needing cleaning. */
export const GAPPY_COLUMN_PCT = 20;

/**
 * Code-written commentary for an Auto Insights profile (the visualization
 * config the page already renders). Returns no lines when the profile has no
 * computed figures, rather than guessing.
 */
export function pageCommentary(
  config: unknown,
  opts: { datasetCount?: number } = {},
): CommentaryLine[] {
  if (!config || typeof config !== "object") return [];
  const cfg = config as Record<string, unknown>;
  const dataset = (cfg.dataset ?? {}) as Record<string, unknown>;
  const columns: ProfileColumn[] = Array.isArray(cfg.column_stats) ? cfg.column_stats : [];
  const lines: CommentaryLine[] = [];
  let hasGappy = false;

  const totalRows = finite(dataset.total_rows);
  const sampledRows = finite(dataset.rows_sampled);
  const rows = totalRows ?? sampledRows;
  const colCount = finite(dataset.column_count) ?? (columns.length || null);
  if (rows != null && colCount != null) {
    const scope = totalRows == null ? " (sampled)" : "";
    lines.push({
      key: "shape",
      text: `This dataset has ${plural(rows, "row")}${scope} and ${plural(colCount, "column")}.`,
    });
  }

  if (columns.length) {
    const counts = { number: 0, category: 0, date: 0 };
    for (const c of columns) counts[kindOf(c.dtype)]++;
    const parts = [
      counts.number && plural(counts.number, "numeric column"),
      counts.category && plural(counts.category, "category column"),
      counts.date && plural(counts.date, "date column"),
    ].filter(Boolean);
    if (parts.length) lines.push({ key: "types", text: `Column types: ${parts.join(", ")}.` });

    const withMissing = columns.filter((c) => finite(c.missing_pct) != null);
    if (withMissing.length) {
      const gappy = withMissing
        .filter((c) => (finite(c.missing_pct) as number) > GAPPY_COLUMN_PCT)
        .sort((a, b) => (finite(b.missing_pct) as number) - (finite(a.missing_pct) as number));
      if (gappy.length) {
        hasGappy = true;
        const worst = gappy
          .slice(0, 3)
          .map((c) => `${String(c.name)} (${fmt(finite(c.missing_pct) as number)}% empty)`)
          .join(", ");
        lines.push({
          key: "missing",
          text: `${plural(gappy.length, "column")} ${gappy.length === 1 ? "is" : "are"} more than ${GAPPY_COLUMN_PCT}% empty: ${worst}.`,
        });
      } else if (withMissing.every((c) => (finite(c.missing_pct) as number) === 0)) {
        lines.push({ key: "missing", text: "No column has empty values." });
      } else {
        lines.push({ key: "missing", text: `No column is more than ${GAPPY_COLUMN_PCT}% empty.` });
      }
    }
  }

  const charts = Array.isArray(cfg.charts) ? cfg.charts.length : 0;
  if (charts)
    lines.push({ key: "charts", text: `Auto Insights drew ${plural(charts, "chart")} for it.` });

  // Next steps follow from the figures above, never from model suggestions.
  const next: string[] = [];
  if (hasGappy) next.push("clean the emptiest columns");
  if ((opts.datasetCount ?? 0) > 1) next.push("join the attached datasets");
  if (columns.some((c) => kindOf(c.dtype) === "date")) next.push("look at trends over time");
  if (lines.length && next.length)
    lines.push({ key: "next", text: `Possible next steps: ${next.join("; ")}.` });

  return lines;
}
