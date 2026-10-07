// Client-side fallback that turns raw analysis rows (output_json) into a
// small viz_config compatible with `normalizeVizConfig` / `DynamicChart`.
// Used when the backend reply has no `visualization_config` so each user
// prompt still gets a Chart view alongside the Data table.
import { aggregateTitle, fieldTitle } from "@/lib/chart-axis-titles";

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

// Date.parse alone is far too lenient: V8 reads "sku-4", "Q-1", "item/7" and
// "Store 12" as dates. Only strings shaped like a date are handed to it.
const NUMERIC_DATE =
  /^(?:\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?:[T ]\d{1,2}:\d{2}.*)?$/;
const MONTH_NAME =
  /(?:^|[^a-z])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[^a-z]|$)/i;

function looksLikeDate(v: unknown): boolean {
  if (v instanceof Date) return !Number.isNaN(v.getTime());
  if (typeof v !== "string") return false;
  const s = v.trim();
  if (!/\d/.test(s)) return false;
  if (!NUMERIC_DATE.test(s) && !MONTH_NAME.test(s)) return false;
  return Number.isFinite(Date.parse(s));
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

function summaryValueAxisTitle(metrics: string[]): string {
  if (metrics.length === 1) return fieldTitle(metrics[0]);
  const sources = metrics.map((metric) => metric.replace(/^(?:mean|median|sum|avg|average|min|max|count)_/i, ""));
  return sources.every((source) => source === sources[0]) ? fieldTitle(sources[0]) : "Calculated value";
}

export type ResultTableForViz = {
  title?: string;
  rows: Row[];
  group_by?: string[];
  metric_columns?: string[];
};

// Structured action tables already contain aggregated values. Plot each
// metric directly against its own grouping instead of aggregating the rows
// again or combining unrelated groupings into one chart.
export function deriveVizFromResultTable(table: ResultTableForViz): any | null {
  const groups = table.group_by?.filter((key) => table.rows[0] && key in table.rows[0]) ?? [];
  const metrics = table.metric_columns?.filter((key) => table.rows[0] && key in table.rows[0]) ?? [];
  if (!groups.length && metrics.length && table.rows.length === 1) {
    const plottedMetrics = metrics.filter((key) => parseNumber(table.rows[0][key]) != null);
    const data = plottedMetrics
      .map((key) => ({ metric: key.replace(/_/g, " "), value: parseNumber(table.rows[0][key]) }))
      .filter((row): row is { metric: string; value: number } => row.value != null);
    if (data.length) return {
      source: "client_derived",
      charts: [{
        type: "bar",
        title: table.title ?? "Overall metric values",
        subtitle: data.map(({ metric, value }) => `${metric}: ${displayValue(value)}`).join("; "),
        xKey: "metric",
        xAxisTitle: data.length === 1 ? "Metric" : "Statistic",
        yAxisTitle: summaryValueAxisTitle(plottedMetrics),
        data,
        series: [{ dataKey: "value", name: summaryValueAxisTitle(plottedMetrics) }],
        insights: ["These values come directly from the overall result table."],
      }],
    };
  }
  if (!groups.length || !metrics.length) {
    const fallback = deriveVizFromRows(table.rows);
    if (!fallback || !table.title) return fallback;
    return {
      ...fallback,
      charts: fallback.charts.map((chart: any) => ({ ...chart, title: `${table.title}: ${chart.title}` })),
    };
  }

  const xKey = groups.join(" / ");
  const charts = metrics.flatMap((metric) => {
    const data = table.rows
      .map((row) => {
        const value = parseNumber(row[metric]);
        if (value == null) return null;
        const label = groups.map((key) => String(row[key] ?? "(missing)")).join(" / ");
        return { [xKey]: label, [metric]: value };
      })
      .filter((row): row is Record<string, string | number> => row != null)
      .sort((a, b) => Number(b[metric]) - Number(a[metric]));
    if (!data.length) return [];
    const shown = data.slice(0, TOP_K);
    const metricLabel = metric.replace(/_/g, " ");
    return [{
      type: "bar",
      title: `${metricLabel} by ${groups.join(" and ")}`,
      subtitle: data.length > TOP_K
        ? `Showing the ${TOP_K} highest of ${data.length} groups from ${table.title ?? "the result table"}.`
        : `Showing all ${data.length} groups from ${table.title ?? "the result table"}.`,
      xKey,
      xAxisTitle: groups.map(fieldTitle).join(" / "),
      yAxisTitle: fieldTitle(metric),
      data: shown,
      series: [{ dataKey: metric, name: metricLabel }],
      insights: ["Values are plotted directly from this grouped result table."],
    }];
  });
  return charts.length ? { charts, source: "client_derived" } : null;
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

  // A single result row with several numeric columns is a set of summary
  // metrics, not a set of observations. A scatter plot of those columns would
  // contain one point and imply a relationship the data cannot establish.
  if (data.length === 1 && numbers.length >= 2 && numbers.length === cols.length) {
    const metricRows = numbers
      .map(({ key }) => ({ metric: key.replace(/_/g, " "), value: parseNumber(data[0]?.[key]) }))
      .filter((row): row is { metric: string; value: number } => row.value != null);
    if (metricRows.length >= 2) {
      return {
        source: "client_derived",
        charts: [{
          type: "bar",
          title: "Requested metric values",
          subtitle: metricRows.map(({ metric, value }) => `${metric}: ${displayValue(value)}`).join("; "),
          xKey: "metric",
          xAxisTitle: "Metric",
          yAxisTitle: summaryValueAxisTitle(numbers.map(({ key }) => key)),
          data: metricRows,
          series: [{ dataKey: "value", name: summaryValueAxisTitle(numbers.map(({ key }) => key)) }],
          insights: ["These are separate summary calculations from the same result table."],
        }],
      };
    }
  }

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
        xAxisTitle: fieldTitle(xKey),
        yAxisTitle: aggregateTitle("mean", yKey),
        aggregate: "mean",
        yField: yKey,
        data: series,
        series: [{ dataKey: yKey, name: aggregateTitle("mean", yKey) }],
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
        xAxisTitle: fieldTitle(xKey),
        yAxisTitle: aggregateTitle("sum", yKey),
        data: bars,
        series: [{ dataKey: yKey, name: aggregateTitle("sum", yKey) }],
        insights: [`${leaderName} leads the displayed ${xKey} categories by ${yKey}.`],
      });
    }
  }

  // 3) two numbers -> scatter
  if (data.length >= 2 && numbers.length >= 2 && charts.length < 3) {
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
        xAxisTitle: fieldTitle(xKey),
        yAxisTitle: fieldTitle(yKey),
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
        xAxisTitle: fieldTitle(xKey),
        yAxisTitle: "Count of records",
        data: bars,
        series: [{ dataKey: "count", name: "Count of records" }],
        insights: [`${leaderName} is the largest ${xKey} group in the pasted result.`],
      });
    }
  }

  if (!charts.length) return null;
  return { charts, source: "client_derived" };
}
