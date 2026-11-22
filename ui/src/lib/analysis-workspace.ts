// Pure helpers for the analysis workspace, moved verbatim out of
// routes/analysis.tsx (which re-imports them) so the 1.6 release suite can
// test them: ui/tests/release_1_6/{chat-add-dataset,multi-upload-charts,data-preview}.test.ts.
// Behaviour is unchanged; see docs/test-reports/1.6-traceability.csv
// (UI-B02, UI-B03, UI-B04, UI-B07, UI-B33).
import type { ThreadDatasetItem } from "@/lib/api/backendApi";
import { deriveVizFromRows } from "@/lib/derive-viz";
import { chipLabel } from "@/lib/multi-dataset-view";

export type DatasetChip = {
  id: string;
  name: string;
  sheetName?: string | null;
  /** Original workbook file when this dataset is one of its sheets. */
  sourceWorkbook?: string | null;
  rows?: number | null;
  available?: boolean;
  sessionId?: string;
  threadId?: string;
  schema?: unknown;
  samples?: Record<string, unknown>[];
  rowsSampled?: number;
  totalRows?: number;
  sizeMb?: number;
  visualizationConfig?: unknown;
  visualizationStatus?: string;
};

export type UploadedDataset = {
  filename: string;
  schema: string[];
  samples: Record<string, unknown>[];
  rows_sampled: number;
  total_rows?: number;
  size_mb?: number;
};

export function datasetSizeMb(source: any): number | undefined {
  const mb =
    typeof source?.file_size_mb === "number"
      ? source.file_size_mb
      : typeof source?.file_size_bytes === "number"
        ? source.file_size_bytes / (1024 * 1024)
        : undefined;
  return mb && mb > 0 ? mb : undefined;
}

export function datasetTotalRows(source: any): number | undefined {
  const value =
    source?.total_rows ??
    source?.visualization_config?.dataset?.total_rows ??
    source?.visualizationConfig?.dataset?.total_rows;
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : undefined;
}

export function normalizeDatasetSchema(schema: unknown, samples: unknown[]): string[] {
  if (Array.isArray(schema)) {
    const columns = schema
      .map((column) =>
        typeof column === "string"
          ? column
          : column && typeof column === "object" && "name" in column
            ? String((column as { name: unknown }).name)
            : "",
      )
      .filter(Boolean);
    if (columns.length) return columns;
  }
  if (schema && typeof schema === "object") return Object.keys(schema);
  const first = samples[0];
  if (first && typeof first === "object" && !Array.isArray(first)) return Object.keys(first);
  if (Array.isArray(first)) return first.map((_, index) => `col_${index + 1}`);
  return [];
}

/** Merge GET /threads/{id}/datasets into the chips already shown (UI-B02: "+ Add"). Moved verbatim from routes/analysis.tsx refreshThreadDatasets. */
export function mergeThreadDatasetChips(
  prevChips: DatasetChip[],
  items: ThreadDatasetItem[] | null | undefined,
  threadId: string,
): { merged: DatasetChip[]; newIds: string[]; unavailable: Set<string>; serverSelected: string[] } {
  const fresh: DatasetChip[] = (items ?? [])
    .filter((d) => d?.dataset_id)
    .map((d) => ({
      id: d.dataset_id,
      name: d.alias || d.filename || d.dataset_id,
      sheetName: d.sheet_name ?? null,
      sourceWorkbook: d.source_workbook ?? null,
      rows: d.rows ?? null,
      available: d.available !== false,
      threadId,
    }));
  const serverSelected = (items ?? [])
    .filter((d) => d?.dataset_id && d.selected === true && d.available !== false)
    .map((d) => d.dataset_id);
  const prevIds = new Set(prevChips.map((c) => c.id));
  const byId = new Map(prevChips.map((c) => [c.id, c] as const));
  for (const c of fresh) {
    const existing = byId.get(c.id);
    byId.set(
      c.id,
      existing
        ? {
            ...existing,
            name: c.name,
            sheetName: c.sheetName,
            sourceWorkbook: c.sourceWorkbook,
            rows: c.rows,
            available: c.available,
          }
        : c,
    );
  }
  const merged = Array.from(byId.values());
  const newIds = fresh.filter((c) => !prevIds.has(c.id) && c.available !== false).map((c) => c.id);
  const unavailable = new Set(merged.filter((c) => c.available === false).map((c) => c.id));
  return { merged, newIds, unavailable, serverSelected };
}

/** Selection after "+ Add" refreshes the chips. Moved verbatim from routes/analysis.tsx refreshThreadDatasets. */
export function nextThreadDatasetSelection(
  prev: string[],
  serverSelected: string[],
  unavailable: Set<string>,
  newIds: string[],
  selectNew?: boolean,
): string[] {
  // The saved selection (GET selected: true) is authoritative after a reload.
  const base = serverSelected.length ? serverSelected : prev;
  const kept = base.filter((id) => !unavailable.has(id));
  return selectNew ? Array.from(new Set([...kept, ...newIds])) : kept;
}

export type AutoInsightsDataset = { id: string; name: string; config: any; status?: string; samples?: any[] };

/** Per-dataset tabs for the Auto Insights modal (grouped uploads).
 *  One entry per DISTINCT dataset_id, each carrying its OWN config + OWN rows.
 *  Never pairs one dataset's chart spec with another dataset's rows.
 *  Moved verbatim from routes/analysis.tsx (autoInsightsDatasets useMemo). */
export function buildAutoInsightsDatasets(
  datasetChips: DatasetChip[],
  persistedDataset: UploadedDataset | null,
  selectedDatasetIds: string[],
): AutoInsightsDataset[] {
  const byId = new Map<string, { id: string; name: string; config: any; status?: string; samples?: any[] }>();
  const multi = datasetChips.length > 1;
  datasetChips.forEach((d) => {
    if (!d?.id || byId.has(d.id)) return;
    // Unselected chips stay in the analysis, but their insights are hidden until selected.
    if (multi && selectedDatasetIds.length && !selectedDatasetIds.includes(d.id)) return;
    // Own rows only: chip samples, or the persisted preview when it belongs
    // to THIS dataset (same filename), otherwise none.
    const ownSamples =
      (Array.isArray(d.samples) && d.samples.length ? d.samples : undefined) ??
      (persistedDataset?.filename && persistedDataset.filename === d.name ? persistedDataset.samples : undefined);
    const ownConfig = d.visualizationConfig ?? (ownSamples?.length ? deriveVizFromRows(ownSamples) : null);
    byId.set(d.id, {
      id: d.id,
      name: multi ? chipLabel(d) : d.name,
      config: ownConfig,
      status: d.visualizationStatus ?? (ownConfig ? "ready" : undefined),
      samples: ownSamples,
    });
  });
  return Array.from(byId.values());
}

/** The Data View dataset for a selected chip that already carries its rows (UI-B07).
 *  Moved verbatim from routes/analysis.tsx selectDatasetChip. */
export function workspaceDatasetFromChip(dataset: DatasetChip, samples: Record<string, unknown>[]): UploadedDataset {
  const schema = normalizeDatasetSchema(dataset.schema, samples);
  return {
    filename: dataset.name,
    schema,
    samples,
    rows_sampled: dataset.rowsSampled ?? samples.length,
    total_rows: dataset.totalRows ?? datasetTotalRows(dataset.visualizationConfig),
    size_mb: dataset.sizeMb,
  };
}

/** Rows/columns for the Data table (sample) and for the Chart / Column Summary
 *  views, which follow the last active sample/results table (UI-B33).
 *  Moved verbatim from routes/analysis.tsx. */
export function dataViewSource(
  viewDataset: UploadedDataset | null,
  lastDataView: "sample" | "results",
  hasSelectedRun: boolean,
  selectedResultRows: Record<string, unknown>[],
): {
  sampleRows: Record<string, unknown>[];
  sampleColumns: string[];
  dataViewRows: Record<string, unknown>[];
  dataViewColumns: string[];
} {
  const sampleRows: Record<string, unknown>[] = viewDataset?.samples ?? [];
  const sampleColumns: string[] = (() => {
    if (viewDataset?.schema?.length) return viewDataset.schema;
    const keys = new Set<string>();
    sampleRows.slice(0, 50).forEach((row) => Object.keys(row ?? {}).forEach((key) => keys.add(key)));
    return Array.from(keys);
  })();
  const dataViewRows: Record<string, unknown>[] =
    lastDataView === "results" && hasSelectedRun ? selectedResultRows : sampleRows;
  const dataViewColumns: string[] = (() => {
    if (lastDataView === "sample" && viewDataset?.schema?.length) return viewDataset.schema;
    const keys = new Set<string>();
    dataViewRows.slice(0, 50).forEach((row) => Object.keys(row ?? {}).forEach((key) => keys.add(key)));
    return Array.from(keys);
  })();
  return { sampleRows, sampleColumns, dataViewRows, dataViewColumns };
}
