// Analysis-result runs for the Data View: rebuilds the per-run result list
// from persisted thread messages and provides pure helpers for navigation,
// multi-table pills, prompt previews and CSV file names. No fetching here —
// everything works on messages/rows already loaded by the analysis route.

import type { AnalysisMessage } from "@/lib/analysis-messages";

export type OutputFile = { filename: string; content: string };

export type ParsedTable = { title?: string; headers: string[]; rows: Record<string, unknown>[] };

export type AnalysisResultTable = { name?: string; rows: Record<string, unknown>[] };

// Each analysis run's tabular output, persisted per run (parallel to charts).
export type TableRun = {
  id: string;
  sectionId: string;
  title: string;
  rows: Record<string, unknown>[];
  /** Code that produced THIS result (from coder_definition.code). */
  code?: string | null;
  /** Version stamp for this result, used by GET /analysis/{id}/code?prompt_ts= */
  promptTs?: string | null;
  /** CSV payload produced by this result. */
  file?: OutputFile | null;
  /** The user's prompt that produced this table (empty/missing -> "Analysis {i}"). */
  prompt?: string | null;
  /** ISO time the analysis ran. */
  ranAt?: string | null;
  /** Chat message id of the user prompt, for "Open in chat". */
  messageId?: string | null;
  /** Additional tables returned by the same analysis (rendered as pills). */
  extraTables?: AnalysisResultTable[];
};

/** Split a pipe-delimited markdown table row into cells. */
function splitPipeRow(line: string): string[] {
  const trimmed = line.trim();
  const withoutOuterPipes = trimmed.replace(/^\|/, "").replace(/\|$/, "");
  return withoutOuterPipes.split("|").map((c) => c.trim());
}

/**
 * Parse a single markdown pipe table from a block of text.
 * Returns null if the block does not look like a header + separator + body.
 */
function parseMarkdownTableBlock(block: string): ParsedTable | null {
  const lines = block
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l) => l.trim().length > 0);
  if (lines.length < 3) return null;

  const headerCells = splitPipeRow(lines[0]!);
  if (headerCells.length === 0 || headerCells.every((c) => c.length === 0)) return null;

  const separator = lines[1]!;
  if (!/^\s*\|?[\s:|-]+\|\s*$/.test(separator) || !/-/.test(separator)) return null;

  const rows: Record<string, unknown>[] = [];
  for (let i = 2; i < lines.length; i++) {
    const cells = splitPipeRow(lines[i]!);
    if (cells.length === 0 || cells.every((c) => c.length === 0)) continue;
    const row: Record<string, unknown> = {};
    headerCells.forEach((h, idx) => {
      const key = h.trim() || `col_${idx + 1}`;
      row[key] = cells[idx] ?? "";
    });
    rows.push(row);
  }
  if (rows.length === 0) return null;

  return { headers: headerCells, rows };
}

/**
 * Find all markdown pipe tables in `raw` and parse them into structured rows.
 * Returns an empty array when no tables are found.
 */
export function parseMarkdownTables(raw: string): ParsedTable[] {
  if (!raw) return [];
  const tables: ParsedTable[] = [];
  // Match a header row, a separator row containing '-', and any following body rows.
  const tableRegex = /(^|\n)[ \t]*\|?[^\n]*\|[^\n]*\n[ \t]*\|?[\s:|-]*-[\s:|-]*\n(?:[ \t]*\|?[^\n]*\|[^\n]*\n?)*/g;
  let match: RegExpExecArray | null;
  const seen = new Set<string>();
  while ((match = tableRegex.exec(raw)) !== null) {
    const block = match[0].replace(/^\n/, "");
    const parsed = parseMarkdownTableBlock(block);
    if (!parsed) continue;
    const key = JSON.stringify({ headers: parsed.headers, rows: parsed.rows.slice(0, 3) });
    if (seen.has(key)) continue;
    seen.add(key);
    tables.push(parsed);
  }
  return tables;
}

/**
 * Extract tabular payloads out of assistant text so the result table only ever
 * appears in the Data table panel — never duplicated inside the chat bubble.
 * Returns the cleaned text plus any parsed markdown tables.
 */
export function extractTabularContent(raw: string): { text: string; tables: ParsedTable[] } {
  if (!raw) return { text: "", tables: [] };
  let text = raw;

  // Fenced blocks that are only CSV / JSON row dumps.
  text = text.replace(/```[a-zA-Z]*\s*\n([\s\S]*?)```/g, (block, inner: string) => {
    const body = inner.trim();
    if (!body) return "";
    const looksJsonRows = /^\[\s*[{[]/.test(body) && /[}\]]\s*\]$/.test(body);
    const lines = body.split("\n").filter((l) => l.trim());
    const looksCsv =
      lines.length > 1 &&
      lines.every((l) => (l.match(/,/g)?.length ?? 0) >= 1) &&
      new Set(lines.map((l) => l.match(/,/g)?.length ?? 0)).size <= 2;
    return looksJsonRows || looksCsv ? "" : block;
  });

  // Markdown pipe tables (header + --- separator + body rows).
  const tables = parseMarkdownTables(text);
  text = text.replace(
    /(^|\n)[ \t]*\|?[^\n]*\|[^\n]*\n[ \t]*\|?[\s:|-]*-[\s:|-]*\n(?:[ \t]*\|?[^\n]*\|[^\n]*\n?)*/g,
    "$1",
  );

  // Standalone "N rows × M columns" preambles (e.g. "Transformed data — 20 rows × 7 columns.").
  text = text.replace(/(^|\n)[^\n]*\b\d+\s*rows?\s*[x×*]\s*\d+\s*columns?\b[^\n]*(?=\n|$)/gi, "$1");

  // Leftover orphan pipe rows.
  text = text
    .split("\n")
    .filter((l) => !/^\s*\|.*\|\s*$/.test(l))
    .join("\n");

  return { text: text.replace(/\n{3,}/g, "\n\n").trim(), tables };
}

/** Backwards-compatible helper that only returns the text portion. */
export function stripTabularContent(raw: string): string {
  return extractTabularContent(raw).text;
}

export function tableRunFingerprint(rows: unknown[]): string {
  try {
    return JSON.stringify(rows).slice(0, 5000);
  } catch {
    return String(rows.length);
  }
}

export function mergeTableRun(prev: TableRun[], run: TableRun): TableRun[] {
  const fp = tableRunFingerprint(run.rows);
  if (prev.some((r) => r.id === run.id || tableRunFingerprint(r.rows) === fp)) return prev;
  return [...prev, run];
}

/** Tolerantly read a backend `output_tables` payload into named tables. */
export function parseOutputTables(raw: unknown): AnalysisResultTable[] {
  if (!Array.isArray(raw)) return [];
  const tables: AnalysisResultTable[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const t = item as { name?: unknown; title?: unknown; rows?: unknown };
    if (!Array.isArray(t.rows) || t.rows.length === 0) continue;
    tables.push({
      name: typeof t.name === "string" && t.name.trim() ? t.name : typeof t.title === "string" && t.title.trim() ? t.title : undefined,
      rows: t.rows as Record<string, unknown>[],
    });
  }
  return tables;
}

/** Extra tables for a run, deduplicated against its main rows. */
export function extraTablesFor(mainRows: Record<string, unknown>[], candidates: AnalysisResultTable[]): AnalysisResultTable[] {
  const mainFp = tableRunFingerprint(mainRows);
  const seen = new Set<string>([mainFp]);
  const extras: AnalysisResultTable[] = [];
  for (const table of candidates) {
    const fp = tableRunFingerprint(table.rows);
    if (seen.has(fp)) continue;
    seen.add(fp);
    extras.push(table);
  }
  return extras;
}

// Rebuild the per-run table history from persisted analysis messages so the
// Analysis results view survives reloads exactly like the charts do.
export function extractTableRunsFromMessages(rows: AnalysisMessage[], defaultSectionId: string): TableRun[] {
  let runs: TableRun[] = [];
  let lastPrompt = "";
  let lastPromptId: string | null = null;
  for (const row of rows) {
    if (row.role === "user") {
      lastPrompt = String(row.content ?? "").trim();
      lastPromptId = row.id;
      continue;
    }
    if (row.role !== "assistant") continue;
    const out = row.output as {
      output_json?: unknown[];
      output_tables?: unknown;
      extracted_tables?: ParsedTable[];
      section_id?: string;
      code?: string | null;
      prompt_ts?: string | null;
      output_file?: OutputFile | null;
    } | null;

    // Pasted results are stored as their own message kind and rendered in a
    // dedicated section below the analysis — never as a normal result page.
    if (
      out &&
      ((out as { pasted_analysis?: unknown }).pasted_analysis || (out as { pasted_analyses?: unknown }).pasted_analyses)
    )
      continue;

    const base = {
      sectionId: typeof out?.section_id === "string" ? out.section_id : defaultSectionId,
      title: lastPrompt || `Result ${runs.length + 1}`,
      prompt: lastPrompt || null,
      ranAt: typeof row.created_at === "string" ? row.created_at : null,
      messageId: lastPromptId,
      code: typeof out?.code === "string" ? out.code : null,
      promptTs: typeof out?.prompt_ts === "string" ? out.prompt_ts : null,
      file: out?.output_file ?? null,
    };

    // Backend-provided multi-table payload wins when present.
    const outputTables = parseOutputTables(out?.output_tables);
    if (outputTables.length) {
      runs = mergeTableRun(runs, {
        ...base,
        id: row.id,
        rows: outputTables[0]!.rows,
        extraTables: outputTables.slice(1),
      });
      continue;
    }

    // Structured backend rows; markdown tables from the same reply become
    // additional tables (pills), never separate analyses.
    const hasStructured = !!(out && Array.isArray(out.output_json) && out.output_json.length > 0);
    if (hasStructured) {
      const mainRows = out!.output_json as Record<string, unknown>[];
      const extractedExtras = Array.isArray(out!.extracted_tables)
        ? out!.extracted_tables.map((t) => ({ name: t.title, rows: t.rows }))
        : [];
      runs = mergeTableRun(runs, {
        ...base,
        id: row.id,
        rows: mainRows,
        extraTables: extraTablesFor(mainRows, extractedExtras),
      });
      continue;
    }

    // Tables embedded in the assistant's markdown content (e.g. "list cloud
    // connections"): the first table is the result, the rest are extra tables.
    const extracted = extractTabularContent(row.content ?? "");
    if (extracted.tables.length > 0) {
      const mainRows = extracted.tables[0]!.rows;
      runs = mergeTableRun(runs, {
        ...base,
        id: `${row.id}-tables`,
        rows: mainRows,
        extraTables: extraTablesFor(
          mainRows,
          extracted.tables.slice(1).map((t) => ({ name: t.title, rows: t.rows })),
        ),
        code: null,
        promptTs: null,
        file: null,
      });
    }
  }
  return runs;
}

/** All tables of a run: the main table first, then its extra tables. */
export function tablesForRun(run: TableRun): AnalysisResultTable[] {
  return [{ rows: run.rows }, ...(run.extraTables ?? [])];
}

/** Keep a result index inside [0, count - 1]; 0 when there are no results. */
export function clampAnalysisIndex(index: number, count: number): number {
  if (count <= 0) return 0;
  return Math.min(Math.max(index, 0), count - 1);
}

/** "9:15 PM" style time for a run; empty string when unknown. */
export function formatRunTime(iso?: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

/** The prompt shown in the results header; falls back to "Analysis {i}". */
export function resultDisplayPrompt(run: TableRun, index: number): string {
  const raw = (run.prompt ?? run.title ?? "").trim();
  if (!raw || /^result\s+\d+$/i.test(raw) || raw === "Restored result" || /^pasted result/i.test(raw)) {
    return `Analysis ${index + 1}`;
  }
  return raw;
}

/** First `max` characters of a prompt for the jump dropdown. */
export function promptPreview(text: string, max = 60): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

function csvBaseName(dataset: string): string {
  const base = (dataset || "dataset").replace(/\.[A-Za-z0-9]+$/, "").trim().replace(/\s+/g, "_");
  return base || "dataset";
}

/** "{dataset}_sample.csv" */
export function sampleCsvName(dataset: string): string {
  return `${csvBaseName(dataset)}_sample.csv`;
}

/** "{dataset}_analysis_{i}.csv" — i is 1-based. */
export function analysisCsvName(dataset: string, index: number): string {
  return `${csvBaseName(dataset)}_analysis_${index}.csv`;
}
