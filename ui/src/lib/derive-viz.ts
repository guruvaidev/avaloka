// Client-side fallback that turns raw analysis rows (output_json) into a
// small viz_config compatible with `normalizeVizConfig` / `DynamicChart`.
// Used when the backend reply has no `visualization_config` so each user
// prompt still gets a Chart view alongside the Data table.

type Row = Record<string, unknown>;

type ColKind = "number" | "date" | "category" | "unknown";

type ColStats = {
  key: string;
  kind: ColKind;
  uniques: number;
};

const SAMPLE = 50;
const TOP_K = 10;
const SCATTER_CAP = 500;

function parseNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const t = v.trim().replace(/,/g, "");
    if (t === "") return null;
    const n = Number(t);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function looksLikeDate(v: unknown): boolean {
  if (v instanceof Date) return !Number.isNaN(v.getTime());
  if (typeof v !== "string") return false;
  if (!/\d/.test(v)) return false;
  // require a separator typical of dates to avoid matching plain numbers
  if (!/[-/:T]/.test(v)) return false;
  const t = Date.parse(v);
  return Number.isFinite(t);
}

function classify(rows: Row[], key: string): ColStats {
  const sample = rows.slice(0, SAMPLE);
  let nNum = 0;
  let nDate = 0;
  let nVal = 0;
  const seen = new Set<string>();
  for (const r of sample) {
    const v = r?.[key];
    if (v == null || v === "") continue;
    nVal++;
    seen.add(String(v));
    if (parseNumber(v) != null) nNum++;
    else if (looksLikeDate(v)) nDate++;
  }
  if (nVal === 0) return { key, kind: "unknown", uniques: 0 };
  const ratio = (n: number) => n / nVal;
  let kind: ColKind = "category";
  if (ratio(nNum) >= 0.8) kind = "number";
  else if (ratio(nDate) >= 0.8) kind = "date";
  return { key, kind, uniques: seen.size };
}

function topKBy(rows: Row[], xKey: string, yKey: string) {
  const groups = new Map<string, { sum: number; count: number }>();
  for (const r of rows) {
    const x = r?.[xKey];
    if (x == null || x === "") continue;
    const y = parseNumber(r?.[yKey]);
    if (y == null) continue;
    const k = String(x);
    const g = groups.get(k) ?? { sum: 0, count: 0 };
    g.sum += y;
    g.count += 1;
    groups.set(k, g);
  }
  const rowsOut = Array.from(groups.entries()).map(([k, g]) => ({
    [xKey]: k,
    [yKey]: Number((g.sum).toFixed(4)),
  }));
  rowsOut.sort((a, b) => (b[yKey] as number) - (a[yKey] as number));
  return rowsOut.slice(0, TOP_K);
}

function displayValue(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
}

export function deriveVizFromRows(rows: unknown): any | null {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const first = rows[0];
  if (!first || typeof first !== "object") return null;
  const data = rows as Row[];
  const keys = Object.keys(first as Row);
  if (!keys.length) return null;

  const cols = keys.map((k) => classify(data, k));
  const numbers = cols.filter((c) => c.kind === "number");
  const cats = cols.filter((c) => c.kind === "category" && c.uniques > 1);
  const dates = cols.filter((c) => c.kind === "date");

  const charts: any[] = [];

  // 1) date + number -> line
  if (dates[0] && numbers[0]) {
    const xKey = dates[0].key;
    const yKey = numbers[0].key;
    const series = [...data]
      .map((r) => ({ [xKey]: String(r?.[xKey] ?? ""), [yKey]: parseNumber(r?.[yKey]) }))
      .filter((r) => r[xKey] && r[yKey] != null) as Row[];
    series.sort((a, b) => Date.parse(String(a[xKey])) - Date.parse(String(b[xKey])));
    if (series.length) {
      const firstValue = series[0]?.[yKey] as number;
      const lastValue = series[series.length - 1]?.[yKey] as number;
      const direction = lastValue > firstValue ? "increased" : lastValue < firstValue ? "decreased" : "remained stable";
      charts.push({
        type: "line",
        title: `${yKey} over ${xKey}`,
        subtitle: `${yKey} ${direction} from ${displayValue(firstValue)} to ${displayValue(lastValue)}.`,
        xKey,
        data: series,
        series: [{ dataKey: yKey, name: yKey }],
        insights: [`Across the displayed period, ${yKey} ${direction} by ${displayValue(Math.abs(lastValue - firstValue))}.`],
      });
    }
  }

  // 2) category + number -> bar (top-K)
  if (cats[0] && numbers[0]) {
    const xKey = cats[0].key;
    const yKey = numbers[0].key;
    const bars = topKBy(data, xKey, yKey);
    if (bars.length) {
      const leader = bars[0];
      const leaderName = String(leader?.[xKey] ?? "the leading category");
      const leaderValue = Number(leader?.[yKey] ?? 0);
      charts.push({
        type: "bar",
        title: `Top ${xKey} by ${yKey}`,
        subtitle: `${leaderName} has the highest ${yKey} at ${displayValue(leaderValue)}.`,
        xKey,
        data: bars,
        series: [{ dataKey: yKey, name: yKey }],
        insights: [`${leaderName} leads the displayed ${xKey} categories by ${yKey}.`],
      });
    }
  }

  // 3) two numbers -> scatter
  if (numbers.length >= 2 && charts.length < 3) {
    const xKey = numbers[0].key;
    const yKey = numbers[1].key;
    const points = data
      .map((r) => ({ [xKey]: parseNumber(r?.[xKey]), [yKey]: parseNumber(r?.[yKey]) }))
      .filter((r) => r[xKey] != null && r[yKey] != null)
      .slice(0, SCATTER_CAP) as Row[];
    if (points.length) {
      charts.push({
        type: "scatter",
        title: `${yKey} vs ${xKey}`,
        subtitle: `${points.length} observations compare ${yKey} with ${xKey}.`,
        xKey,
        data: points,
        series: [{ dataKey: yKey, name: yKey }],
        insights: [`The chart shows how ${yKey} varies across ${xKey} for ${points.length} observations.`],
      });
    }
  }

  // 4) fallback: single category (count) -> bar of counts
  if (charts.length === 0 && cats[0]) {
    const xKey = cats[0].key;
    const counts = new Map<string, number>();
    for (const r of data) {
      const v = r?.[xKey];
      if (v == null || v === "") continue;
      const k = String(v);
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    const bars = Array.from(counts.entries())
      .map(([k, c]) => ({ [xKey]: k, count: c }))
      .sort((a, b) => (b.count as number) - (a.count as number))
      .slice(0, TOP_K);
    if (bars.length) {
      const leader = bars[0];
      const leaderName = String(leader?.[xKey] ?? "the leading category");
      const leaderCount = Number(leader?.count ?? 0);
      charts.push({
        type: "bar",
        title: `${xKey} distribution`,
        subtitle: `${leaderName} is the most frequent value with ${displayValue(leaderCount)} rows.`,
        xKey,
        data: bars,
        series: [{ dataKey: "count", name: "count" }],
        insights: [`${leaderName} is the largest ${xKey} group in the pasted result.`],
      });
    }
  }

  if (!charts.length) return null;
  return { charts, source: "client_derived" };
}
