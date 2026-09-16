// Shared builder for Auto Insights groups (one group per DISTINCT dataset_id).
// Consumed by both the Auto Insights modal and the inline chart panel on the
// analysis page so the two views can never diverge.
import { normalizeVizConfig, type Slide } from "@/components/dashboard/dynamicChart";

export interface InsightDataset {
  id: string;
  name: string;
  config?: any;
  status?: string;
  samples?: any[];
}

export interface InsightGroup {
  id: string;
  name: string;
  slides: Slide[];
  samples?: any[];
  outputRows?: Record<string, unknown>[] | null;
  errored: boolean;
  pending: boolean;
}

export function insightChartKey(slide: Slide) {
  return (
    String(slide.title ?? "").trim().toLowerCase() ||
    JSON.stringify({ type: slide.type, xKey: slide.xKey, series: slide.series.map((s) => s.dataKey) })
  );
}

export function dedupeInsightSlides(slides: Slide[]) {
  const seen = new Set<string>();
  return slides.filter((slide) => {
    const key = insightChartKey(slide);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function isErrorStatus(s?: string) {
  const v = String(s ?? "").toLowerCase();
  return v === "error" || v === "failed";
}

export function buildInsightGroups(opts: {
  datasets?: InsightDataset[];
  config?: any;
  status?: string;
  samples?: any[];
  outputRows?: Record<string, unknown>[] | null;
  datasetName?: string;
  primaryDatasetId?: string | null;
  /** Slides to show when there is no config at all (modal demo fallback). */
  fallbackSlides?: Slide[];
}): InsightGroup[] {
  const { config, status, samples, outputRows, datasetName, primaryDatasetId, fallbackSlides } = opts;

  const byId = new Map<string, InsightDataset>();
  (Array.isArray(opts.datasets) ? opts.datasets : []).forEach((d) => {
    if (!d?.id || byId.has(d.id)) return;
    byId.set(d.id, d);
  });
  const tabs = Array.from(byId.values());

  if (tabs.length) {
    return tabs.map((t) => {
      const shared = !!primaryDatasetId && t.id === primaryDatasetId;
      const st = t.status ?? (shared ? status : undefined);
      const errored = isErrorStatus(st);
      const cfg = errored ? null : (t.config ?? (shared ? config : null));
      const smp = t.samples?.length ? t.samples : shared ? samples : undefined;
      const own = dedupeInsightSlides(normalizeVizConfig(cfg, smp));
      return {
        id: t.id,
        name: t.name,
        slides: own,
        samples: smp,
        outputRows: shared ? outputRows : undefined,
        errored,
        pending: !errored && st === "pending" && own.length === 0,
      };
    });
  }

  const errored = isErrorStatus(status);
  const own = dedupeInsightSlides(normalizeVizConfig(errored ? null : config, samples));
  return [
    {
      id: "__single__",
      name: datasetName ?? "",
      slides: own.length ? own : errored ? [] : (fallbackSlides ?? []),
      samples,
      outputRows,
      errored,
      pending: !errored && status === "pending" && own.length === 0,
    },
  ];
}
