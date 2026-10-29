/** Display-only upload review helpers. Never infer backend cleaning behavior. */
export const CLEANING_DISCLOSURE = "No cleaning options are sent by this action. The backend has not supplied its cleaning rules.";
export const CLEANING_TOOLTIP = "Starts with the uploaded dataset; no cleaning options are sent and backend cleaning rules are unavailable.";

export function readableDatasetName(filename: string): string {
  return filename.replace(/\.(csv|xlsx?|json)$/i, "")
    .replace(/(?:\s*\(\d+\)|__+\d+|[ _-]+copy(?:[ _-]*\d+)?)+$/, "")
    .replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim()
    .split(" ").map((word) => /^(AI|SQL|API|USA|UK)$/i.test(word) ? word.toUpperCase() : /^youtube$/i.test(word) ? "YouTube" : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(" ")
    .slice(0, 80) || "Dataset";
}

export function reviewDatasetName(dataset: any): string {
  const profile = dataset?.visualization_config?.profile_overview ?? dataset?.profiling_result ?? {};
  const title = profile.dataset_title ?? profile.title ?? dataset?.dataset_title ?? profile.domain;
  if (typeof title === "string" && title.trim()) return title.trim().slice(0, 80);
  return readableDatasetName(dataset?.source_workbook ? `${dataset.source_workbook.replace(/\.[^.]+$/, "")} ${dataset.sheet_name ?? ""}` : dataset?.filename ?? dataset?.alias ?? "Dataset");
}

export function canReviewUpload(fileCount: number, readyCount: number, progress: number, uploading: boolean): boolean {
  return fileCount > 0 && readyCount === fileCount && progress >= 100 && !uploading;
}

export type ReviewIssue = {
  id: string;
  title: string;
  summary: string;
  details: string[];
  columns?: { name: string; count: number | null; percent: number | null }[];
};
const finite = (value: unknown): number | null => value == null || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
const text = (value: unknown): string | null => typeof value === "string" && value.trim() ? value.trim() : null;
export const issueHasDetails = (issue: ReviewIssue): boolean => issue.details.length > 0 || Boolean(issue.columns?.length);

/** Counts come exclusively from profile metadata, never guessed from preview rows. */
export function reviewIssues(dataset: any): ReviewIssue[] {
  const viz = dataset?.visualization_config ?? {};
  const profile = dataset?.profiling_result ?? viz.profiling_result ?? {};
  const overview = viz.profile_overview ?? {};
  const quality = profile.data_quality ?? {};
  const result: ReviewIssue[] = [];
  const duplicates = profile.duplicates ?? quality.duplicates ?? {};
  const duplicateCount = finite(profile.duplicate_rows ?? quality.duplicate_rows ?? duplicates.count ?? viz.dataset?.duplicate_rows);
  const examples = Array.isArray(duplicates.example_row_numbers) ? duplicates.example_row_numbers : Array.isArray(profile.duplicate_row_numbers) ? profile.duplicate_row_numbers : [];
  if (duplicateCount != null && duplicateCount > 0) result.push({
    id: "duplicates", title: "Duplicate rows found", summary: `${duplicateCount.toLocaleString()} duplicate rows`,
    details: [`The profile reports ${duplicateCount.toLocaleString()} duplicate rows.`, ...(examples.length ? [`Example row numbers: ${examples.slice(0, 5).join(", ")}`] : [])],
  });
  const stats = Array.isArray(viz.column_stats) ? viz.column_stats : Array.isArray(profile.column_stats) ? profile.column_stats : [];
  const columns = stats.map((column: any) => ({
    name: String(column.name ?? column.column ?? "Column"),
    count: finite(column.missing_count ?? column.null_count ?? column.empty_cells),
    percent: finite(column.missing_pct ?? column.missing_percent),
  })).filter((column: { count: number | null; percent: number | null }) => (column.count ?? 0) > 0 || (column.percent ?? 0) > 0);
  const missingCount = finite(profile.missing_cells ?? quality.missing_cells ?? viz.dataset?.missing_cells ?? overview.missing_cells);
  if (columns.length || (missingCount != null && missingCount > 0)) result.push({
    id: "missing", title: "Empty cells found",
    summary: `${columns.length ? `${columns.length} columns with empty cells` : "Empty cells reported"}${missingCount != null ? ` · ${missingCount.toLocaleString()} cells` : ""}`,
    details: missingCount != null ? [`The profile reports ${missingCount.toLocaleString()} empty cells.`] : [], columns,
  });
  const flags = [...(Array.isArray(profile.red_flags) ? profile.red_flags : []), ...(Array.isArray(profile.concerns) ? profile.concerns : []), ...(Array.isArray(overview.dataset_issues) ? overview.dataset_issues : [])];
  for (const [index, flag] of flags.entries()) {
    const title = text(flag) ?? text(flag?.title) ?? text(flag?.issue);
    if (!title || result.some((issue) => issue.title === title)) continue;
    const detail = text(flag?.details) ?? text(flag?.description);
    result.push({ id: `flag-${index}`, title, summary: text(flag?.summary) ?? "Reported by the profile", details: detail ? [detail] : [] });
  }
  return result;
}
/** Generated placeholder names (column_3, Unnamed: 2) are not real column names. */
export const PLACEHOLDER_COLUMN_NAME = /^(column_\d+|unnamed:?\s*\d+)$/i;
export const realColumnNames = (columns: string[]): string[] =>
  columns.filter((name) => typeof name === "string" && !PLACEHOLDER_COLUMN_NAME.test(name.trim()));

export type UploadKind = "single" | "workbook" | "files";
type KindDataset = { dataset_id?: string; filename?: string | null; source_workbook?: string | null };

/** Upload kind from the datasets (upload response or GET thread datasets) and, when known, the uploaded file count. */
export function uploadKind(datasets: KindDataset[], fileCount?: number): UploadKind {
  const list = Array.isArray(datasets) ? datasets : [];
  const files = fileCount ?? new Set(list.map((d) => d.source_workbook ?? d.filename ?? d.dataset_id)).size;
  if (files > 1) return "files";
  const workbook = list[0]?.source_workbook;
  if (list.length > 1 && workbook && list.every((d) => d.source_workbook === workbook)) return "workbook";
  return "single";
}

/** Workbook: all sheets. Several files: nothing until the user ticks. Single: unchanged (all). */
export function initialSelection(kind: UploadKind, datasetIds: string[]): string[] {
  return kind === "files" ? [] : [...datasetIds];
}

export const canStartSelection = (kind: UploadKind, selectedIds: string[]): boolean =>
  kind !== "files" || selectedIds.length > 0;

/** Body for PUT /threads/{id}/selected-datasets; null when no call is needed (single file or one workbook). */
export function selectionPayload(kind: UploadKind, selectedIds: string[]): { dataset_ids: string[] } | null {
  if (kind !== "files" || !selectedIds.length) return null;
  return { dataset_ids: Array.from(new Set(selectedIds)) };
}

/** Readable workbook title for a group of sheets: "Avaloka 1.6 Master Test Plan 2 · 10 sheets". */
export function workbookHeading(sourceWorkbook: string, sheetCount: number): string {
  return `${readableDatasetName(sourceWorkbook)} · ${sheetCount} sheet${sheetCount === 1 ? "" : "s"}`;
}

/** Notes about skipped sheets belong to the summary card, once. */
export const isSkippedSheetNote = (note: string): boolean => /\bskipped\b/i.test(note) && /\bsheet\b/i.test(note);
