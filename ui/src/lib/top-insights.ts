/** Cross-dataset "top insights" picker shared by the Chart tab and the chat findings. Pure, display-only. */
export type TopInsightChart = {
  importance?: number | null;
  rank?: number | null;
  title?: string | null;
  headline?: string | null;
  insight?: string | null;
  [key: string]: unknown;
};

export type TopInsightDataset = {
  id: string;
  name: string;
  rows?: number | null;
  visualization_config?: { charts?: TopInsightChart[] | null } | null;
};

export type TopInsightPick = { datasetId: string; datasetName: string; chart: TopInsightChart; score: number };

const finite = (value: unknown): number | null =>
  value == null || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);

/** Rank of a chart: its own `rank`, else its 1-based position in the list. */
export function chartRank(chart: TopInsightChart, index: number): number {
  return finite(chart?.rank) ?? index + 1;
}

/** `importance` when present, else 100 − 10 × (rank − 1). */
export function chartScore(chart: TopInsightChart, index: number): number {
  return finite(chart?.importance) ?? 100 - 10 * (chartRank(chart, index) - 1);
}

export function pickTopInsights(
  datasets: TopInsightDataset[],
  { limit = 5, perDatasetCap = 2 }: { limit?: number; perDatasetCap?: number } = {},
): TopInsightPick[] {
  const candidates = (Array.isArray(datasets) ? datasets : []).flatMap((dataset, datasetIndex) => {
    const charts = Array.isArray(dataset?.visualization_config?.charts) ? dataset.visualization_config!.charts! : [];
    return charts.map((chart, index) => ({
      datasetId: dataset.id,
      datasetName: dataset.name,
      chart,
      score: chartScore(chart, index),
      rank: chartRank(chart, index),
      rows: finite(dataset.rows) ?? 0,
      datasetIndex,
    }));
  });
  candidates.sort(
    (a, b) => b.score - a.score || b.rows - a.rows || a.rank - b.rank || a.datasetIndex - b.datasetIndex,
  );
  const taken = new Set<number>();
  const counts = new Map<string, number>();
  // Pass 1: respect the per-dataset cap while other datasets still have charts.
  candidates.forEach((candidate, index) => {
    if (taken.size >= limit) return;
    const count = counts.get(candidate.datasetId) ?? 0;
    if (count >= perDatasetCap) return;
    counts.set(candidate.datasetId, count + 1);
    taken.add(index);
  });
  // Pass 2: every dataset is exhausted under the cap — fill from what is left.
  candidates.forEach((_, index) => {
    if (taken.size < limit) taken.add(index);
  });
  return candidates
    .filter((_, index) => taken.has(index))
    .map(({ datasetId, datasetName, chart, score }) => ({ datasetId, datasetName, chart, score }));
}

/** Finding text for a pick: headline, else insight, else title. */
export function pickFindingText(chart: TopInsightChart): string {
  const pick = [chart?.headline, chart?.insight, chart?.title].find((v) => typeof v === "string" && v.trim());
  return typeof pick === "string" ? pick.trim() : "";
}
