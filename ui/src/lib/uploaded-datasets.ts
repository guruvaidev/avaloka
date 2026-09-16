import { useQuery } from "@tanstack/react-query";
import { backendApi } from "@/lib/api/backendApi";

/** A dataset uploaded directly by the user (file or folder). */
export type UploadedDataset = {
  dataset_id: string;
  name: string;
  filename: string | null;
  alias: string | null;
  columnCount: number | null;
  columns: string[];
  sizeBytes: number | null;
  createdAt: string | null;
  sessionId: string | null;
  groupSessionId: string | null;
  sourceKind: string | null;
};

/** A collapsed group of datasets uploaded together. */
export type UploadedDatasetGroup = {
  key: string;
  title: string;
  subtitle: string;
  members: UploadedDataset[];
  createdAt: string | null;
  totalSizeBytes: number | null;
  totalColumnCount: number | null;
};

function pickNumber(...values: unknown[]): number | null {
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() && Number.isFinite(Number(v))) return Number(v);
    if (Array.isArray(v)) return v.length;
  }
  return null;
}

function pickString(...values: unknown[]): string | null {
  for (const v of values) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return null;
}

/** Normalize a raw backend dataset row defensively — field names vary. */
export function mapUploadedDataset(raw: Record<string, any>): UploadedDataset | null {
  const metadata = raw.metadata && typeof raw.metadata === "object" ? raw.metadata : {};
  const id = pickString(raw.dataset_id, raw.id, raw.datasetId, metadata.dataset_id, metadata.datasetId);
  if (!id) return null;

  const filename = pickString(raw.filename, raw.file_name, metadata.filename, metadata.file_name);
  const apiName = pickString(raw.name, raw.dataset_name, raw.title, metadata.name, metadata.dataset_name);
  const alias = pickString(raw.alias, metadata.alias);
  // name → alias → filename (no synthetic fallback here; UI decides the placeholder)
  const name = apiName ?? alias ?? filename ?? id;
  const rawColumns = Array.isArray(raw.columns) ? raw.columns : [];

  return {
    dataset_id: id,
    name,
    filename,
    alias: alias ?? apiName,
    columnCount: rawColumns.length || pickNumber(
      raw.column_count,
      raw.num_columns,
      raw.n_columns,
      raw.schema?.columns,
    ),
    columns: rawColumns,
    sizeBytes: pickNumber(raw.size_bytes, raw.size, raw.file_size, raw.bytes),
    // Strictly the API's created_at — never a client-side "now"
    createdAt: pickString(raw.created_at, metadata.created_at),


    sessionId: pickString(
      raw.session_id,
      raw.sessionId,
      raw.avaloka_session_id,
      metadata.session_id,
      metadata.sessionId,
    ),
    groupSessionId: pickString(
      raw.group_session_id,
      raw.groupSessionId,
      metadata.group_session_id,
      metadata.groupSessionId,
    ),
    sourceKind: pickString(raw.source_kind, raw.sourceKind, metadata.source_kind, metadata.sourceKind),
  };
}

export function formatBytes(bytes: number | null): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;

  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/** Collapse datasets that were uploaded together into a single group card. */
export function groupUploadedDatasets(datasets: UploadedDataset[]): UploadedDatasetGroup[] {
  const byKey = new Map<string, UploadedDataset[]>();
  for (const d of datasets) {
    const key = d.groupSessionId || d.dataset_id;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key)!.push(d);
  }

  const groups: UploadedDatasetGroup[] = [];
  for (const [key, members] of byKey.entries()) {
    const sorted = [...members].sort(
      (a, b) => new Date(a.createdAt ?? 0).getTime() - new Date(b.createdAt ?? 0).getTime(),
    );
    const first = sorted[0];
    const isMulti = sorted.length > 1;
    const realName = first.name && first.name !== first.dataset_id ? first.name : null;
    const firstLabel = realName || first.alias || first.filename || "Untitled dataset";

    const title = isMulti ? `${firstLabel} + ${sorted.length - 1} more` : firstLabel;
    const subtitle = isMulti
      ? first.groupSessionId
        ? `ID: ${first.groupSessionId}`
        : `ID: ${first.dataset_id}`
      : `ID: ${first.dataset_id}`;

    let maxCreated = 0;
    let totalSize = 0;
    let hasSize = false;
    let totalColumns = 0;
    let hasColumns = false;
    for (const m of sorted) {
      const t = new Date(m.createdAt ?? 0).getTime();
      if (t > maxCreated) maxCreated = t;
      if (m.sizeBytes != null) {
        totalSize += m.sizeBytes;
        hasSize = true;
      }
      if (m.columnCount != null) {
        totalColumns += m.columnCount;
        hasColumns = true;
      }
    }

    groups.push({
      key,
      title,
      subtitle,
      members: sorted,
      createdAt: maxCreated ? new Date(maxCreated).toISOString() : null,
      totalSizeBytes: hasSize ? totalSize : null,
      totalColumnCount: hasColumns ? totalColumns : null,
    });
  }

  return groups.sort(
    (a, b) => new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime(),
  );
}

export function useUploadedDatasets(enabled = true) {
  return useQuery({
    enabled,
    queryKey: ["uploaded-datasets"] as const,
    staleTime: 30_000,
    queryFn: async (): Promise<UploadedDataset[]> => {
      const res = await backendApi.listDatasets();
      const rows: any[] = Array.isArray(res)
        ? res
        : Array.isArray(res?.datasets)
          ? res.datasets
          : Array.isArray(res?.items)
            ? res.items
            : Array.isArray(res?.data)
              ? res.data
              : [];
      return rows
        .map((r) => mapUploadedDataset(r ?? {}))
        .filter((d): d is UploadedDataset => !!d)
        .sort(
          (a, b) =>
            new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime(),
        );
    },
  });
}
