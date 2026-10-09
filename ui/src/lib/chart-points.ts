// Single source of truth for the data each Auto Insights chart actually renders.
// Used by DynamicChart AND the Insights Avatar so explanations match the screen.
import type { Slide } from "@/components/dashboard/dynamicChart";

export type RenderedPoint = { x: string | number; y: number | string | null };

const ISO_DATE = /^\d{4}-\d{2}-\d{2}([T ][\d:.]+(Z|[+-]\d{2}:?\d{2})?)?$/;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function parseIso(v: unknown): number | null {
  if (typeof v !== "string" || !ISO_DATE.test(v.trim())) return null;
  const t = Date.parse(v.trim());
  return Number.isFinite(t) ? t : null;
}

/** Line/area data, with ISO-date x values sorted and bucketed by day or month. */
export function getLineRenderData(slide: Slide): { data: any[]; xKey: string } {
  const xKey = slide.xKey;
  const rows = Array.isArray(slide.data) ? slide.data : [];
  if ((slide.type !== "line" && slide.type !== "area") || !rows.length) return { data: rows, xKey };
  const times = rows.map((r) => parseIso(r?.[xKey]));
  if (times.some((t) => t === null)) return { data: rows, xKey };

  const ts = times as number[];
  const monthly = Math.max(...ts) - Math.min(...ts) > 90 * 86_400_000;
  const buckets = new Map<number, { label: string; sums: Record<string, number>; counts: Record<string, number> }>();
  rows.forEach((row, i) => {
    const d = new Date(ts[i]);
    const y = d.getUTCFullYear();
    const m = d.getUTCMonth();
    const key = monthly ? Date.UTC(y, m, 1) : Date.UTC(y, m, d.getUTCDate());
    const label = monthly ? `${MONTHS[m]} ${y}` : `${d.getUTCDate()} ${MONTHS[m]}`;
    const b = buckets.get(key) ?? { label, sums: {}, counts: {} };
    for (const s of slide.series) {
      const n = Number(row?.[s.dataKey]);
      if (!Number.isFinite(n)) continue;
      b.sums[s.dataKey] = (b.sums[s.dataKey] ?? 0) + n;
      b.counts[s.dataKey] = (b.counts[s.dataKey] ?? 0) + 1;
    }
    buckets.set(key, b);
  });
  const data = Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, b]) => {
      const out: Record<string, unknown> = { [xKey]: b.label };
      for (const s of slide.series) {
        const c = b.counts[s.dataKey];
        out[s.dataKey] = c ? Number((b.sums[s.dataKey] / c).toFixed(4)) : null;
      }
      return out;
    });
  return { data, xKey };
}

/** Pie/donut slices, grouping tiny categories into "Other" beyond 10 slices. */
export function getPieRenderData(slide: Slide): { name: string; value: number }[] {
  const key = slide.series[0]?.dataKey ?? "value";
  const nameKey = slide.xKey;
  const raw = (slide.data ?? []).filter((d) => d && d[nameKey] != null && Number.isFinite(Number(d[key])));
  let pieData = raw.map((d) => ({ name: String(d[nameKey]), value: Number(d[key]) }));
  const MAX_SLICES = 10;
  if (pieData.length > MAX_SLICES) {
    const sorted = [...pieData].sort((a, b) => b.value - a.value);
    const top = sorted.slice(0, MAX_SLICES - 1);
    const otherVal = sorted.slice(MAX_SLICES - 1).reduce((s, r) => s + r.value, 0);
    pieData = [...top, { name: "Other", value: otherVal }];
  }
  return pieData;
}

const MAX_POINTS = 60;

export function getRenderedPoints(slide: Slide): RenderedPoint[] {
  if (slide.type === "pie" || slide.type === "donut") {
    return getPieRenderData(slide).slice(0, MAX_POINTS).map((d) => ({ x: d.name, y: d.value }));
  }
  const yKey = slide.series[0]?.dataKey ?? "y";
  const rows = Array.isArray(slide.data) ? slide.data : [];
  if (slide.chartKind === "histogram") {
    // Precomputed bins already carry their exact backend range labels.
    if (rows.every((r) => typeof r?.[slide.xKey] === "string")) {
      return rows.slice(0, MAX_POINTS).map((r) => ({ x: r[slide.xKey], y: Number(r?.[yKey]) }));
    }
    const starts = rows.map((r) => Number(r?.[slide.xKey]));
    const width = starts.length > 1 && Number.isFinite(starts[1] - starts[0]) ? starts[1] - starts[0] : 0;
    return rows.slice(0, MAX_POINTS).map((r, i) => {
      const s = starts[i];
      const x = Number.isFinite(s) ? `${s.toFixed(2)}–${(s + width).toFixed(2)}` : String(r?.[slide.xKey]);
      return { x, y: Number(r?.[yKey]) };
    });
  }
  const data = slide.type === "line" || slide.type === "area" ? getLineRenderData(slide).data : rows;
  return data.slice(0, MAX_POINTS).map((r) => ({ x: r?.[slide.xKey] ?? null, y: r?.[yKey] ?? null }));
}
