import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

/** One uploaded file/dataset belonging to an analysis group. */
export type AnalysisDataset = {
  dataset_id: string;
  filename: string | null;
  alias: string | null;
  /** The analyses row that owns this dataset (mother or child upload). */
  analysisId: string;
};

const BATCH_KEY = "analysis:batch";

/**
 * Label for an analysis group so multi-file uploads don't look like a single
 * file. Single-file groups keep the plain name.
 */
export function analysisGroupName(baseName: string, datasetCount: number): string {
  const name = (baseName ?? "").trim();
  if (!name || datasetCount <= 1) return name;
  const extra = datasetCount - 1;
  return `${name} + ${extra} more`;
}

/** Display label for a dataset row: filename → alias → dataset_id. */
export function analysisDatasetLabel(d: AnalysisDataset): string {
  const raw = (d.filename || d.alias || d.dataset_id || "").trim();
  const tail = raw.split("/").filter(Boolean).pop() ?? raw;
  return tail || d.dataset_id;
}

/**
 * Datasets remembered client-side for a grouped (multi-file) upload. The chips
 * are written by the analysis page via `analysis:batch:<analysisId>`.
 */
function readStoredDatasets(analysisId: string): AnalysisDataset[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = sessionStorage.getItem(`${BATCH_KEY}:${analysisId}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { chips?: { id?: string; name?: string }[] };
    return (parsed?.chips ?? [])
      .filter((c) => !!c?.id)
      .map((c) => ({
        dataset_id: String(c.id),
        filename: c.name ?? null,
        alias: c.name ?? null,
        analysisId,
      }));
  } catch {
    return [];
  }
}

function mergeById(...lists: AnalysisDataset[][]): AnalysisDataset[] {
  const byId = new Map<string, AnalysisDataset>();
  for (const list of lists) {
    for (const d of list) {
      if (!d.dataset_id) continue;
      const prev = byId.get(d.dataset_id);
      if (!prev) {
        byId.set(d.dataset_id, d);
      } else {
        byId.set(d.dataset_id, {
          ...prev,
          filename: prev.filename ?? d.filename,
          alias: prev.alias ?? d.alias,
        });
      }
    }
  }
  return Array.from(byId.values());
}

/**
 * Datasets for a set of analysis groups, keyed by the mother analysis id.
 * Sourced from the `analyses` rows themselves (mother + child uploads) and
 * merged with the grouped-upload chips kept in sessionStorage.
 */
export function useAnalysesDatasets(analysisIds: string[]) {
  const ids = useMemo(() => Array.from(new Set(analysisIds)).sort(), [analysisIds]);

  const query = useQuery({
    queryKey: ["analysis-datasets", ids] as const,
    enabled: ids.length > 0,
    staleTime: 60_000,
    queryFn: async (): Promise<Record<string, AnalysisDataset[]>> => {
      const [own, children] = await Promise.all([
        supabase.from("analyses").select("id,dataset_id,filename,name").in("id", ids),
        supabase
          .from("analyses")
          .select("id,dataset_id,filename,name,parent_analysis_id")
          .in("parent_analysis_id", ids)
          .order("created_at", { ascending: true }),
      ]);

      const out: Record<string, AnalysisDataset[]> = {};
      for (const id of ids) out[id] = [];

      for (const row of (own.data ?? []) as any[]) {
        if (!row.dataset_id) continue;
        out[row.id]?.push({
          dataset_id: row.dataset_id,
          filename: row.filename ?? null,
          alias: row.name ?? null,
          analysisId: row.id,
        });
      }
      for (const row of (children.data ?? []) as any[]) {
        if (!row.dataset_id || !row.parent_analysis_id) continue;
        out[row.parent_analysis_id]?.push({
          dataset_id: row.dataset_id,
          filename: row.filename ?? null,
          alias: row.name ?? null,
          analysisId: row.id,
        });
      }
      return out;
    },
  });

  const byAnalysis = useMemo(() => {
    const map: Record<string, AnalysisDataset[]> = {};
    for (const id of ids) {
      map[id] = mergeById(query.data?.[id] ?? [], readStoredDatasets(id));
    }
    return map;
  }, [ids, query.data]);

  return { byAnalysis, isLoading: query.isLoading, error: query.error as Error | null };
}
