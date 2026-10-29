import { useState, type ReactNode } from "react";
import { scatterScale, niceAxis, logarithmicAxis, measureLabel, binLabel, subsetLabels } from "@/lib/chart-display";
import { Button } from "@/components/base/buttons/button";
import { applyChartSemantics, axisKind, scalePolicy, extremePointsNote, plainCategory, type AxisKind, type ColumnProfile } from "@/lib/chart-semantics";
import {
  ComposedChart,
  Area,
  Line,
  BarChart,
  Bar,
  LabelList,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
} from "recharts";
import { getLineRenderData, getPieRenderData } from "@/lib/chart-points";
import { chartAxisTitles } from "@/lib/chart-axis-titles";

export type ChartType = "line" | "area" | "bar" | "pie" | "donut" | "scatter";

export type Series = { dataKey: string; name?: string; color?: string };

export type Slide = {
  title: string;
  subtitle?: string;
  type: ChartType;
  data: any[];
  xKey: string;
  series: Series[];
  insights: string[];
  headline?: string | null;
  insightFacts?: string | null;
  /** Metadata from visualization_config (used by the Insights Avatar). */
  id?: string;
  chartKind?: "bar" | "line" | "pie" | "scatter" | "histogram";
  xField?: string | null;
  yField?: string | null;
  aggregate?: string | null;
  reason?: string | null;
  /** Explicit axis labels (encodings titles / "Count"). */
  xLabel?: string;
  yLabel?: string;
  /** Human-readable titles preserved from client-derived chart specs. */
  xAxisTitle?: string;
  yAxisTitle?: string;
  /** Backend values are already expressed as percentages, not fractions. */
  yFormat?: "percent";
  rateValue?: string;
  /** Chart couldn't be drawn; title + insights are still shown. */
  undrawable?: boolean;
  authoritativePoints?: boolean;
  xKind?: AxisKind;
  yKind?: AxisKind;
  pointsHidden?: number;
  measuredRows?: number;
  correlation?: number;
  pointsShown?: number;
};

export const PALETTE = ["#1565ef", "#f97066", "#12b76a", "#f79009", "#7a5af8", "#06aed4", "#ee46bc"];

function formatNumberCompact(n: number | string): string {
  return measureLabel(n);
}

function tooltipCompactFormatter(value: any, name: any) {
  if (typeof value === "number") return [formatNumberCompact(value), name];
  return [value, name];
}

function resolveFieldKey(samples: any[], field: string): string | null {
  if (!samples?.length || !field) return null;
  const row = samples[0];
  if (!row || typeof row !== "object") return null;
  const keys = Object.keys(row as object);
  if (keys.includes(field)) return field;
  const norm = field.trim().toLowerCase();
  return keys.find((k) => k.trim().toLowerCase() === norm) ?? null;
}

function asArray<T>(v: T | T[] | undefined | null): T[] {
  if (v == null) return [];
  return Array.isArray(v) ? v : [v];
}

function pickType(raw: any): ChartType {
  if (raw?.encodings?.x?.bin) return "bar";
  const t = String(raw?.type ?? raw?.chart_type ?? raw?.kind ?? "line").toLowerCase();
  if (t.includes("scatter")) return "scatter";
  if (t.includes("bar") || t.includes("column") || t.includes("hist")) return "bar";
  if (t.includes("donut")) return "donut";
  if (t.includes("pie")) return "pie";
  if (t.includes("area")) return "area";
  return "line";
}

const SUPPORTED_AGGREGATES = new Set(["", "count", "sum", "mean", "avg", "average", "median", "min", "max"]);

function buildFromEncodings(raw: any, samples: any[]): { data: any[]; xKey: string; series: Series[] } | null {
  const enc = raw?.encodings;
  if (!enc || !samples?.length) return null;
  const xFieldRaw = enc.x?.field;
  if (!xFieldRaw) return null;
  const xField = resolveFieldKey(samples, String(xFieldRaw));
  if (!xField) return null;
  const aggregate = String(enc.y?.aggregate ?? "").toLowerCase();
  // Never silently render an unsupported aggregate as a mean. Doing so gives
  // the chart a correct-looking label with incorrect values.
  if (!SUPPORTED_AGGREGATES.has(aggregate)) return null;
  const topK = raw?.config?.top_k as number | undefined;
  let yFieldRaw = enc.y?.field as string | undefined;
  let yField = yFieldRaw ? (resolveFieldKey(samples, yFieldRaw) ?? undefined) : undefined;
  const countKey = "__count__";

  if (
    enc.x?.bin ||
    String(raw?.type ?? raw?.chart_type ?? raw?.kind ?? "")
      .toLowerCase()
      .includes("hist")
  ) {
    if (applyChartSemantics({ type: "bar", chartKind: "histogram", title: "", data: samples, xKey: xField, xField, series: [{ dataKey: countKey }] }) === null) return null;
    const nbins = Number(raw?.config?.nbins) || 20;
    const nums = samples.map((r) => Number(r?.[xField])).filter((n) => Number.isFinite(n));
    if (!nums.length) return null;
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const width = max > min ? (max - min) / nbins : 1;
    const bins = new Array(nbins).fill(0);
    for (const n of nums) {
      let idx = width > 0 ? Math.floor((n - min) / width) : 0;
      if (idx >= nbins) idx = nbins - 1;
      bins[idx] += 1;
    }
    const data = bins.map((count, i) => ({
      [xField]: width > 0 ? Number((min + i * width).toFixed(2)) : min,
      [countKey]: count,
    }));
    return {
      data,
      xKey: xField,
      series: [{ dataKey: countKey, name: "Count", color: PALETTE[0] }],
    };
  }

  if (!yField) {
    if (aggregate === "count") yField = countKey;
    else return null;
  }
  const resolvedY = yField;

  if (pickType(raw) === "scatter") {
    const data = samples
      .map((r) => ({ [xField]: r?.[xField], [resolvedY]: r?.[resolvedY] }))
      .filter((r) => r[xField] != null && r[resolvedY] != null);
    return { data, xKey: xField, series: [{ dataKey: resolvedY, name: resolvedY, color: PALETTE[0] }] };
  }

  const groups = new Map<string, { sum: number; count: number; vals: number[] }>();
  for (const row of samples) {
    const xv = row?.[xField];
    if (xv == null) continue;
    const key = String(xv);
    const g = groups.get(key) ?? { sum: 0, count: 0, vals: [] };
    g.count += 1;
    if (yField !== countKey) {
      const yv = Number(row?.[yField]);
      if (Number.isFinite(yv)) {
        g.sum += yv;
        g.vals.push(yv);
      }
    }
    groups.set(key, g);
  }
  let rows = Array.from(groups.entries()).map(([k, g]) => {
    let v: number;
    if (yField === countKey || aggregate === "count" || xField === yField) v = g.count;
    else if (aggregate === "sum") v = g.sum;
    else if (aggregate === "mean" || aggregate === "avg" || aggregate === "average")
      v = g.vals.length ? g.sum / g.vals.length : 0;
    else if (aggregate === "median") {
      const sorted = [...g.vals].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      v = !sorted.length ? 0 : sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    } else if (aggregate === "min") v = g.vals.length ? Math.min(...g.vals) : 0;
    else if (aggregate === "max") v = g.vals.length ? Math.max(...g.vals) : 0;
    // aggregate === "": no aggregate named, default to the mean.
    else v = g.vals.length ? g.sum / g.vals.length : g.count;
    return { [xField]: k, [yField]: Number(v.toFixed(4)) };
  });
  if (topK && rows.length > topK) rows = rows.slice(0, topK);
  return {
    data: rows,
    xKey: xField,
    series: [{ dataKey: resolvedY, name: resolvedY === countKey ? "Count" : resolvedY, color: PALETTE[0] }],
  };
}

function pickInsights(raw: any): string[] {
  const src = raw?.insights ?? raw?.key_insights ?? raw?.takeaways ?? raw?.notes ?? raw?.insight ?? [];
  return asArray(src)
    .map((it: any) => (typeof it === "string" ? it : (it?.text ?? it?.insight ?? it?.message)))
    .filter(Boolean) as string[];
}

export function normalizeChart(raw: any, samples?: any[], columns: ColumnProfile[] = []): Slide | null {
  const field = raw?.encodings?.x?.field ?? raw?.xKey ?? raw?.derived_data?.x_key ?? "x";
  const histogram = raw?.encodings?.x?.bin || /hist/i.test(raw?.type ?? raw?.chart_type ?? raw?.kind ?? "");
  const fieldKind = axisKind(field, (samples ?? []).map((point) => point[field]), columns);
  if (histogram && (fieldKind === "identifier" || fieldKind === "year")) return null;
  let slide = normalizeChartInner(raw, samples);
  if (!slide) {
    if (!raw || typeof raw !== "object") return null;
    // Never drop a chart: keep its title + insights even when it can't be drawn.
    console.warn("[AutoInsights] Chart could not be drawn", { id: raw?.id, type: raw?.type ?? raw?.chart_type });
    slide = {
      title: raw.title ?? raw.name ?? raw.label ?? "Auto Insight",
      subtitle: raw.subtitle ?? raw.description ?? raw.reason ?? undefined,
      type: pickType(raw),
      data: [],
      xKey: "category",
      series: [{ dataKey: "value", name: "Value", color: PALETTE[0] }],
      insights: pickInsights(raw),
      undrawable: true,
    };
  }
  const enc = raw?.encodings ?? {};
  const t = String(raw?.type ?? raw?.chart_type ?? raw?.kind ?? "").toLowerCase();
  const kind: Slide["chartKind"] =
    enc?.x?.bin || t.includes("hist")
      ? "histogram"
      : slide.type === "donut" || slide.type === "pie"
        ? "pie"
        : slide.type === "area"
          ? "line"
          : slide.type;
  const str = (v: unknown) => (v == null || v === "" ? null : String(v));
  return applyChartSemantics({
    ...slide,
    authoritativePoints: Array.isArray(raw?.derived_data?.points),
    pointsHidden: raw?.derived_data?.points_hidden,
    measuredRows: raw?.derived_data?.n,
    correlation: raw?.derived_data?.r,
    pointsShown: raw?.derived_data?.points_shown,
    id: raw?.id != null ? String(raw.id) : undefined,
    chartKind: kind,
    xField: str(slide.xField ?? enc.x?.field ?? enc.color?.field ?? raw?.xKey ?? raw?.derived_data?.x_key),
    yField: str(slide.yField ?? enc.y?.field ?? enc.theta?.field ?? raw?.derived_data?.y_key ?? slide.series[0]?.dataKey),
    aggregate: str(enc.y?.aggregate ?? enc.theta?.aggregate),
    reason: str(raw?.reason),
    headline: str(raw?.headline),
    insightFacts: str(raw?.insight_facts),
    xAxisTitle: str(raw?.xAxisTitle ?? raw?.x_axis_title ?? slide.xAxisTitle ?? slide.xLabel) ?? undefined,
    yAxisTitle: str(raw?.yAxisTitle ?? raw?.y_axis_title ?? slide.yAxisTitle ?? slide.yLabel) ?? undefined,
    yFormat: raw?.config?.y_format === "percent" || enc.y?.aggregate === "rate" ? "percent" : undefined,
    rateValue: str(enc.y?.value) ?? undefined,
  }, columns);
}

const MISSING_LABEL = "(missing)";

function categoryLabel(v: unknown): string {
  if (v == null) return MISSING_LABEL;
  const s = String(v).trim();
  if (!s || ["nan", "null", "none", "undefined", "nat"].includes(s.toLowerCase())) return MISSING_LABEL;
  return s;
}

/**
 * Robust categorical builder for bar / line / pie charts (never histograms or scatter).
 * Points always use fixed keys "category" and "value" so they can never collide.
 * Returns null when this builder doesn't apply, so legacy paths keep working.
 */
function buildCategorical(
  raw: any,
  samples: any[] | undefined,
  type: ChartType,
): {
  data: any[];
  xKey: string;
  series: Series[];
  xLabel: string;
  yLabel: string;
} | null {
  const t = String(raw?.type ?? raw?.chart_type ?? raw?.kind ?? "").toLowerCase();
  const enc = raw?.encodings ?? {};
  if (type === "scatter" || t.includes("hist") || enc?.x?.bin) return null;
  const isPie = type === "pie" || type === "donut";
  const catFieldRaw = isPie ? (enc.color?.field ?? enc.x?.field ?? enc.theta?.field) : enc.x?.field;
  const valEnc = isPie ? (enc.theta ?? enc.y) : enc.y;
  const valFieldRaw: string | undefined = valEnc?.field;
  const aggregate = String(valEnc?.aggregate ?? (isPie ? "count" : "")).toLowerCase();
  const isCount = aggregate === "count" || !valFieldRaw || (!!catFieldRaw && valFieldRaw === catFieldRaw);
  const xLabel = String(enc.x?.title ?? (isPie ? enc.color?.title : undefined) ?? catFieldRaw ?? "Category");
  const yLabel = String(valEnc?.title ?? (isCount ? "Count" : (valFieldRaw ?? "Value")));
  const series: Series[] = [{ dataKey: "value", name: yLabel, color: PALETTE[0] }];

  // Rule 3: precomputed points win, in given order.
  const dd = raw?.derived_data;
  if (dd && Array.isArray(dd.points) && dd.points.length) {
    // The backend contract names these fields x_key/y_key. Prefer those keys
    // over the original encodings because grounding may rewrite the value key.
    const cf = dd.x_key ?? dd.xKey ?? dd.category_field ?? catFieldRaw;
    const vf = dd.y_key ?? dd.yKey ?? dd.value_field ?? valFieldRaw ?? "count";
    const data = dd.points.map((p: any) => ({
      category: categoryLabel((cf ? p?.[cf] : undefined) ?? p?.x ?? p?.label ?? p?.name),
      value: Number((vf ? p?.[vf] : undefined) ?? p?.y ?? p?.value ?? p?.count) || 0,
      ...(type === "bar" ? { n: p?.n } : {}),
    }));
    return { data, xKey: "category", series, xLabel, yLabel };
  }

  if (!catFieldRaw || !samples?.length) return null;
  // Aggregated value bar/line charts keep using the existing encodings builder.
  if (!isCount && !isPie) return null;
  const catField = resolveFieldKey(samples, String(catFieldRaw));
  if (!catField) return null;
  const valField = !isCount && valFieldRaw ? resolveFieldKey(samples, String(valFieldRaw)) : null;
  const groups = new Map<string, { sum: number; count: number }>();
  for (const row of samples) {
    const key = categoryLabel(row?.[catField]);
    const g = groups.get(key) ?? { sum: 0, count: 0 };
    g.count += 1;
    if (valField) {
      const n = Number(row?.[valField]);
      if (Number.isFinite(n)) g.sum += n;
    }
    groups.set(key, g);
  }
  let data = Array.from(groups.entries()).map(([category, g]) => ({
    category,
    value:
      valField && aggregate === "sum"
        ? g.sum
        : valField && /mean|avg|average/.test(aggregate)
          ? g.count
            ? g.sum / g.count
            : 0
          : g.count,
  }));
  if (type === "bar" || isPie) data.sort((a, b) => b.value - a.value);
  const topK = raw?.config?.top_k as number | undefined;
  if (topK && data.length > topK) {
    if (isPie) {
      const other = data.slice(topK).reduce((s, r) => s + r.value, 0);
      data = other > 0 ? [...data.slice(0, topK), { category: "Other", value: other }] : data.slice(0, topK);
    } else data = data.slice(0, topK);
  }
  if (!data.length) return null;
  return { data, xKey: "category", series, xLabel, yLabel };
}

function normalizeChartInner(raw: any, samples?: any[]): Slide | null {
  if (!raw || typeof raw !== "object") return null;
  try {
    const type = pickType(raw);
    const title = raw.title ?? raw.name ?? raw.label ?? "Auto Insight";
    const subtitle = raw.subtitle ?? raw.description ?? raw.reason ?? undefined;

    // Backend points are authoritative. Histograms, scatters and empty point
    // lists use them as-is; bar, line and pie charts go through buildCategorical,
    // which reads x_key/y_key and keeps the backend's order as {category, value}.
    const dd = raw.derived_data;
    const isHistogram = String(raw.type ?? raw.chart_type ?? raw.kind ?? "").toLowerCase().includes("hist") || raw.encodings?.x?.bin;
    if (Array.isArray(dd?.points) && (isHistogram || type === "scatter" || dd.points.length === 0)) {
      const first = dd.points[0] ?? {};
      const xKey = dd.x_key ?? dd.xKey ?? ("x" in first ? "x" : raw.encodings?.x?.field ?? raw.xKey ?? "x");
      const yKey = dd.y_key ?? dd.yKey ?? ("count" in first ? "count" : "y" in first ? "y" : raw.encodings?.y?.field ?? "value");
      return {
        title, subtitle, type: isHistogram ? "bar" : type,
        data: dd.points, xKey,
        series: [{ dataKey: yKey, name: isHistogram ? "Count" : raw.encodings?.y?.title ?? raw.encodings?.y?.field ?? yKey, color: PALETTE[0] }],
        insights: pickInsights(raw),
        xLabel: raw.encodings?.x?.title ?? raw.encodings?.x?.field,
        yLabel: isHistogram || raw.encodings?.y?.aggregate === "count" ? "Count" : raw.encodings?.y?.title ?? raw.encodings?.y?.field,
      };
    }

    const cat = buildCategorical(raw, samples, type);
    if (cat) {
      return {
        title,
        subtitle,
        type,
        data: cat.data,
        xKey: cat.xKey,
        series: cat.series,
        insights: pickInsights(raw),
        xLabel: cat.xLabel,
        yLabel: cat.yLabel,
      };
    }

    let data: any[] = [];
    let xKey = raw.xKey ?? raw.x_key ?? raw.xAxis ?? raw.x_field ?? "x";
    let series: Series[] = [];

    // Pie/donut shape: { labels: [...], values: [...] } or { categories/names + values }
    if (type === "pie" || type === "donut") {
      const labels = raw.labels ?? raw.categories ?? raw.names ?? raw.keys;
      const values = raw.values ?? raw.data_values ?? raw.counts;
      if (Array.isArray(labels) && Array.isArray(values) && labels.length && labels.length === values.length) {
        data = labels.map((l: any, i: number) => ({
          name: String(l),
          value: Number(values[i]) || 0,
        }));
        xKey = "name";
        series = [{ dataKey: "value", name: "Value", color: PALETTE[0] }];
        return { title, subtitle, type, data, xKey, series, insights: pickInsights(raw) };
      }
      // Encodings-based pie: color.field is the category, theta aggregates (usually count)
      const enc = raw.encodings;
      if (enc && samples?.length) {
        const catFieldRaw = enc.color?.field ?? enc.theta?.field ?? enc.x?.field;
        const catField = catFieldRaw ? resolveFieldKey(samples, String(catFieldRaw)) : null;
        if (catField) {
          const aggregate = String(enc.theta?.aggregate ?? enc.y?.aggregate ?? "count").toLowerCase();
          const valFieldRaw = aggregate === "count" ? null : (enc.theta?.field ?? enc.y?.field);
          const valField =
            valFieldRaw && String(valFieldRaw) !== String(catFieldRaw)
              ? resolveFieldKey(samples, String(valFieldRaw))
              : null;
          const groups = new Map<string, { sum: number; count: number }>();
          for (const row of samples) {
            const cv = row?.[catField];
            if (cv == null || cv === "") continue;
            const key = String(cv);
            const g = groups.get(key) ?? { sum: 0, count: 0 };
            g.count += 1;
            if (valField) {
              const n = Number(row?.[valField]);
              if (Number.isFinite(n)) g.sum += n;
            }
            groups.set(key, g);
          }
          let rows = Array.from(groups.entries()).map(([name, g]) => ({
            name,
            value: aggregate === "sum" && valField ? g.sum : g.count,
          }));
          const topK = raw?.config?.top_k as number | undefined;
          rows.sort((a, b) => b.value - a.value);
          if (topK && rows.length > topK) {
            const top = rows.slice(0, topK);
            const otherVal = rows.slice(topK).reduce((s, r) => s + r.value, 0);
            rows = otherVal > 0 ? [...top, { name: "Other", value: otherVal }] : top;
          }
          if (rows.length) {
            data = rows;
            xKey = "name";
            series = [{ dataKey: "value", name: "Value", color: PALETTE[0] }];
            return { title, subtitle, type, data, xKey, series, insights: pickInsights(raw) };
          }
        }
      }
    }

    const derived = raw.derived_data && (raw.derived_data.rows ?? raw.derived_data.data);
    const dataSrc = Array.isArray(raw.data) ? raw.data : Array.isArray(derived) ? derived : null;

    if (dataSrc && dataSrc.length > 0 && typeof dataSrc[0] === "object") {
      data = dataSrc;
      const keys = Object.keys(data[0]);
      if (!keys.includes(xKey)) xKey = raw.xKey ?? keys[0];
      const rawSeries = raw.series ?? raw.y ?? raw.yKeys ?? raw.metrics;
      if (Array.isArray(rawSeries) && rawSeries.length) {
        series = rawSeries.map((s: any, i: number) =>
          typeof s === "string"
            ? { dataKey: s, name: s, color: PALETTE[i % PALETTE.length] }
            : {
                dataKey: s.dataKey ?? s.key ?? s.field ?? s.name,
                name: s.name ?? s.label ?? s.dataKey ?? s.key,
                color: s.color ?? PALETTE[i % PALETTE.length],
              },
        );
      } else {
        series = keys
          .filter((k) => k !== xKey)
          .map((k, i) => ({ dataKey: k, name: k, color: PALETTE[i % PALETTE.length] }));
      }
    } else if (raw.encodings && samples?.length) {
      const built = buildFromEncodings(raw, samples);
      if (built) {
        data = built.data;
        xKey = built.xKey;
        series = built.series;
      }
    } else if (Array.isArray(raw.x) && (Array.isArray(raw.y) || Array.isArray(raw.series))) {
      const xs: any[] = raw.x;
      const ySrc = raw.series ?? raw.y;
      const yArr =
        Array.isArray(ySrc?.[0]) || (ySrc?.[0] && typeof ySrc[0] === "object" && "values" in ySrc[0])
          ? ySrc
          : [{ name: raw.yLabel ?? "value", values: ySrc }];
      xKey = "x";
      series = yArr.map((s: any, i: number) => ({
        dataKey: s.name ?? `s${i}`,
        name: s.name ?? `Series ${i + 1}`,
        color: s.color ?? PALETTE[i % PALETTE.length],
      }));
      data = xs.map((xv: any, idx: number) => {
        const row: any = { x: xv };
        yArr.forEach((s: any, i: number) => {
          const values = Array.isArray(s) ? s : s.values;
          row[series[i].dataKey] = values?.[idx];
        });
        return row;
      });
    }

    if (!data.length || !series.length) return null;

    return { title, subtitle, type, data, xKey, series, insights: pickInsights(raw) };
  } catch {
    return null;
  }
}

export function normalizeVizConfig(config: any, samples?: any[], inheritedColumns: ColumnProfile[] = []): Slide[] {
  if (!config) return [];
  const columns = Array.isArray(config.columns) ? config.columns : Array.isArray(config.column_stats) ? config.column_stats : inheritedColumns;
  if (
    typeof config === "object" &&
    !Array.isArray(config) &&
    !config.charts &&
    !config.visualizations &&
    !config.type &&
    !config.data &&
    !config.encodings
  ) {
    const out: Slide[] = [];
    Object.entries(config).forEach(([key, value]) => {
      if (key !== "columns" && key !== "column_stats" && value && typeof value === "object") out.push(...normalizeVizConfig(value, samples, columns));
    });
    return out;
  }
  const list = (Array.isArray(config) && config) ||
    config.charts ||
    config.visualizations ||
    config.plots ||
    config.figures || [config];
  return asArray(list)
    .map((c) => normalizeChart(c, samples, columns))
    .filter(Boolean) as Slide[];
}

export function collectVizInsights(config: any, samples?: any[]): { summaries: string[]; insights: string[] } {
  if (!config) return { summaries: [], insights: [] };

  const slides = normalizeVizConfig(config, samples);
  const summaries: string[] = [];
  const insights: string[] = [];
  const seen = new Set<string>();

  const add = (list: string[], text?: string | null) => {
    const t = text?.trim();
    if (!t || seen.has(t)) return;
    seen.add(t);
    list.push(t);
  };

  for (const slide of slides) {
    add(summaries, slide.subtitle);
    slide.insights.forEach((item) => add(insights, item));
  }

  if (typeof config === "object" && config !== null) {
    add(summaries, config.llm_selection_reason);
    const warnings = config.warnings ?? config.bias_diagnostics?.warnings;
    if (Array.isArray(warnings)) {
      warnings.forEach((w) => add(insights, typeof w === "string" ? w : (w?.message ?? w?.text ?? w?.warning)));
    }
  }

  return { summaries, insights };
}

export function DynamicChart({
  slide,
  height = 230,
  compact = false,
  chartKey,
  stackPresentation = false,
  header,
}: {
  slide: Slide;
  height?: number;
  compact?: boolean;
  chartKey?: string;
  stackPresentation?: boolean;
  header?: ReactNode;
}) {
  const uid = (chartKey ?? slide.title ?? "chart").replace(/[^a-zA-Z0-9_-]/g, "_");
  const axisTitles = chartAxisTitles(slide);
  const yKey = slide.series[0]?.dataKey ?? "y";
  const histogram = slide.chartKind === "histogram";
  const xValues = slide.data.map((point) => point[slide.xKey]);
  const yValues = slide.data.flatMap((point) => slide.series.map((series) => point[series.dataKey]));
  const xPolicy = scalePolicy(xValues);
  const yPolicy = scalePolicy(yValues);
  const [scaleChoice, setScaleChoice] = useState<{ slide: Slide; mode: "linear" | "log" } | null>(null);
  const choice = scaleChoice?.slide === slide ? scaleChoice.mode : null;
  const xLog = scatterScale(slide.type, xValues, choice);
  const yLog = scatterScale(slide.type, yValues, choice);
  const xAxis = niceAxis(xValues);
  const yAxis = niceAxis(yValues, slide.type !== "scatter");
  const xLogAxis = logarithmicAxis(xValues);
  const yLogAxis = logarithmicAxis(yValues);
  // Readable axis titles from chartAxisTitles; rate charts say "% {value}",
  // histograms say "Count".
  const exactX = axisTitles.x;
  const exactY = axisTitles.y;
  const numericYTitle =
    slide.yFormat === "percent" ? `% ${slide.rateValue ?? slide.yField ?? exactY}` : histogram ? "Count" : exactY;
  const controls = slide.type === "scatter" && (xPolicy.automatic || yPolicy.automatic) ? (
    <div className="ai-scale-control" role="group" aria-label="Axis scale" onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
      {(["linear", "log"] as const).map((mode) => <Button key={mode} size="sm" color="tertiary" className="ai-scale-option" aria-label={`${mode === "log" ? "Log" : "Linear"} scale for ${slide.title}`} aria-pressed={mode === "log" ? xLog || yLog : !xLog && !yLog} onPress={() => setScaleChoice({ slide, mode })}>{mode === "log" ? "Log" : "Linear"}</Button>)}
    </div>
  ) : null;
  const hiddenNote = slide.type === "scatter" ? extremePointsNote(slide.pointsHidden, slide.measuredRows) : null;
  const categoryTooltip =
    slide.xKey === "category"
      ? {
          labelFormatter: () => "",
          formatter: (v: any, _n: any, p: any) => [formatNumberCompact(v), p?.payload?.category ?? ""],
          separator: ": ",
        }
      : { formatter: tooltipCompactFormatter };

  if (slide.undrawable || !slide.data?.length) {
    return (
      <div
        className="flex w-full items-center justify-center text-[12px] text-[#667085]"
        style={{ height: compact ? Math.min(height, 120) : height }}
      >
        This chart couldn't be drawn from the available data.
      </div>
    );
  }

  if (slide.type === "pie" || slide.type === "donut") {
    const pieData = getPieRenderData(slide);
    if (!pieData.length) return null;
    const total = pieData.reduce((s, r) => s + (r.value || 0), 0);
    const pct = (v: number) => `${(total > 0 ? (v / total) * 100 : 0).toFixed(1)}%`;
    const donut = stackPresentation || slide.type === "donut";

    return (
      <div
        className={
          stackPresentation
            ? `ai-pie-layout ${compact ? "ai-pie-compact" : ""}`
            : `flex w-full min-w-0 flex-wrap gap-3 overflow-visible ${height > 400 ? "h-full items-center" : "h-auto"}`
        }
        style={stackPresentation ? { height, minHeight: 320 } : height > 400 ? { height } : undefined}
      >
        <div
          className={
            stackPresentation
              ? "ai-pie-box"
              : `${height > 400 ? "h-full" : "h-[230px]"} min-h-[230px] min-w-[220px] flex-[1.2_1_0%] shrink-0`
          }
        >
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <RTooltip
                separator=": "
                formatter={(value: any, name: any) => [`${formatNumberCompact(value)} (${pct(Number(value))})`, name]}
              />
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                innerRadius={stackPresentation ? "55%" : Math.max(40, height / 4)}
                outerRadius={stackPresentation ? "80%" : Math.min(height > 400 ? 220 : 105, Math.max(60, height / 2.7))}
                stroke={stackPresentation ? "var(--color-ai-surface)" : undefined}
                strokeWidth={stackPresentation ? 2 : undefined}
                isAnimationActive={!stackPresentation}
                paddingAngle={stackPresentation ? 1 : 2}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                ))}
              </Pie>
              {stackPresentation && donut && (
                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="ai-pie-total">
                  {formatNumberCompact(total)}
                  <tspan x="50%" dy="18" className="ai-pie-total-label">
                    Total
                  </tspan>
                </text>
              )}
            </PieChart>
          </ResponsiveContainer>
        </div>
        {!compact ? (
          <div
            className={
              stackPresentation ? "ai-pie-legend" : "max-h-[230px] min-w-[160px] flex-1 shrink-0 overflow-y-auto pr-1"
            }
          >
            <ul className="flex w-full flex-col gap-1.5">
              {pieData.map((item, i) => (
                <li
                  key={`${item.name}-${i}`}
                  className={
                    stackPresentation
                      ? "ai-pie-legend-item"
                      : "flex min-w-0 items-center gap-2 text-[11px] text-[#475467]"
                  }
                >
                  <span
                    className="size-2 shrink-0 rounded-sm"
                    style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                  />
                  <span className="min-w-0 flex-1 break-words">{item.name}</span>
                  <span className="shrink-0 whitespace-nowrap text-[#101828]">
                    {formatNumberCompact(item.value)} ({pct(item.value)})
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    );
  }

  if (slide.type === "scatter") {
    return (
      <div className="ai-scatter-chart">
        <div className="ai-scatter-header">{header}{controls}</div>
        <div className="ai-scatter-plot" style={{ height: Math.max(height, 320) }}><ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 12, right: 16, left: 16, bottom: 12 }}
          >
            <CartesianGrid stroke="var(--chart-grid, #e4e7ec)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey={slide.xKey}
              scale={xLog ? "log" : "linear"}
              domain={xLog && xLogAxis ? xLogAxis.domain : xAxis.domain}
              ticks={xLog && xLogAxis ? xLogAxis.ticks : xAxis.ticks}
              tickFormatter={formatNumberCompact}
              tick={{ fill: "var(--chart-tick, #667085)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              height={compact ? 30 : 60}
              label={
                compact
                  ? undefined
                  : {
                      value: `${exactX}${xLog ? " (log scale)" : ""}`,
                      position: "insideBottom",
                      offset: 0,
                      fill: "var(--chart-axis-title, #344054)",
                      fontSize: 12,
                      fontWeight: 600,
                    }
              }
            />
            <YAxis
              type="number"
              dataKey={yKey}
              scale={yLog ? "log" : "linear"}
              domain={yLog && yLogAxis ? yLogAxis.domain : yAxis.domain}
              ticks={yLog && yLogAxis ? yLogAxis.ticks : yAxis.ticks}
              tick={{ fill: "var(--chart-tick, #667085)", fontSize: 10 }}
              tickFormatter={formatNumberCompact}
              tickLine={false}
              axisLine={false}
              width={compact ? 40 : 82}
              label={
                compact
                  ? undefined
                  : {
                      value: `${exactY}${yLog ? " (log scale)" : ""}`,
                      angle: -90,
                      position: "insideLeft",
                      offset: 0,
                      fill: "var(--chart-axis-title, #344054)",
                      fontSize: 12,
                      fontWeight: 600,
                      style: { textAnchor: "middle", lineHeight: 1.5 },
                    }
              }
            />
            <ZAxis range={[20, 20]} />
            <RTooltip cursor={{ strokeDasharray: "3 3" }} formatter={tooltipCompactFormatter} />
            <Scatter data={slide.data} fill={slide.series[0]?.color ?? PALETTE[0]} />
          </ScatterChart>
        </ResponsiveContainer></div>
        {hiddenNote && <p className="ai-extreme-note">{hiddenNote}</p>}
      </div>
    );
  }

  if (slide.type === "bar") {
    const margin = compact ? { top: 4, right: 4, left: 0, bottom: 0 } : { top: stackPresentation ? 28 : 12, right: 16, left: 16, bottom: 12 };
    const emphasisRanges =
      stackPresentation && slide.chartKind !== "histogram"
        ? new Map(
            slide.series.map((series) => {
              const values = slide.data.map((row) => Number(row[series.dataKey])).filter(Number.isFinite);
              return [series.dataKey, { highest: Math.max(...values), lowest: Math.min(...values) }];
            }),
          )
        : null;
    const tickSize = compact ? 9 : 10;
    const isPercent = slide.yFormat === "percent";
    const yName = numericYTitle;
    const formatValue = isPercent ? (value: number | string) => `${formatNumberCompact(value)}%` : formatNumberCompact;
    const barTooltip = isPercent
      ? {
          ...categoryTooltip,
          formatter: (value: any, name: any, item: any) => {
            const n = item?.payload?.n;
            const count = n == null || n === "" ? NaN : Number(n);
            const scope = Number.isFinite(count) ? ` of ${count.toLocaleString("en-US")} rows` : "";
            return [
              `${formatValue(value)}${scope}`,
              slide.xKey === "category" ? (item?.payload?.category ?? name) : name,
            ];
          },
        }
      : categoryTooltip;
    const histogramTicks = histogram ? subsetLabels(slide.data.map((row) => row[slide.xKey])) : undefined;
    const histogramTooltip =
      slide.chartKind === "histogram"
        ? {
            labelFormatter: (label: unknown) => String(label ?? ""),
            formatter: (value: any, name: any): [string, any] => [formatNumberCompact(value), name],
          }
        : barTooltip;
    return (
      <div className="flex w-full min-w-0 flex-col" style={{ height, minHeight: height }}>
        <div className="min-h-0 flex-1"><ResponsiveContainer width="100%" height="100%">
          <BarChart data={slide.data} margin={margin} barCategoryGap={histogram ? "2%" : "10%"}>
            <CartesianGrid stroke="var(--chart-grid, #e4e7ec)" strokeDasharray="3 3" />
            <XAxis
              dataKey={slide.xKey}
              type="category"
              scale="band"
              ticks={histogramTicks}
              tick={({ x, y, payload }: any) => {
                const full = histogram ? binLabel(payload.value) : plainCategory(payload.value);
                const short = full.length > 14 ? `${full.slice(0, 13)}…` : full;
                return <text x={x} y={y + 14} transform={histogram ? undefined : `rotate(-40, ${x}, ${y + 14})`} textAnchor={histogram ? "middle" : "end"} className="fill-ai-muted" fontSize={tickSize}><title>{String(payload.value)}</title>{histogram ? full : short}</text>;
              }}
              interval={0}
              tickLine={false}
              axisLine={false}
              hide={compact}
              height={compact ? 30 : histogram ? 62 : 108}
              label={
                compact
                  ? undefined
                  : {
                      value: `${exactX}${xLog ? " (log scale)" : ""}`,
                      position: "insideBottom",
                      offset: 0,
                      fill: "var(--chart-axis-title, #344054)",
                      fontSize: 12,
                      fontWeight: 600,
                    }
              }
            />
            <YAxis
              scale="linear"
              domain={yAxis.domain}
              ticks={yAxis.ticks}
              tick={{ fill: "var(--chart-tick, #667085)", fontSize: tickSize }}
              tickFormatter={formatValue}
              tickLine={false}
              axisLine={false}
              width={compact ? 40 : 82}
              label={
                compact
                  ? undefined
                  : {
                      value: yName,
                      angle: -90,
                      position: "insideLeft",
                      offset: 0,
                      fill: "var(--chart-axis-title, #344054)",
                      fontSize: 12,
                      fontWeight: 600,
                      style: { textAnchor: "middle", lineHeight: 1.5 },
                    }
              }
            />
            <RTooltip {...histogramTooltip} />
            {slide.series.map((s) => (
              <Bar key={s.dataKey} dataKey={s.dataKey} name={s.name} fill={s.color ?? PALETTE[0]} radius={[4, 4, 0, 0]}>
                {stackPresentation &&
                  slide.chartKind !== "histogram" &&
                  slide.data.length >= 2 &&
                  slide.data.map((point, index) => {
                    const value = Number(point[s.dataKey]);
                    const range = emphasisRanges?.get(s.dataKey);
                    const highest = range?.highest,
                      lowest = range?.lowest;
                    return (
                      <Cell
                        key={index}
                        fill={
                          value === highest
                            ? "var(--color-ai-primary)"
                            : value === lowest
                              ? "var(--color-ai-low-bar)"
                              : "var(--color-ai-bar)"
                        }
                      />
                    );
                  })}
                {stackPresentation && slide.chartKind !== "histogram" && (
                  <LabelList
                    dataKey={s.dataKey}
                    position="top"
                    formatter={(value: unknown) => {
                      const number = Number(value);
                      return (
                        new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(number) +
                        (isPercent ? "%" : "")
                      );
                    }}
                    className="fill-ai-muted"
                    fontSize={11}
                  />
                )}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer></div>
      </div>
    );
  }

  const lineData = getLineRenderData(slide).data;
  const hasNegative = lineData.some((d) =>
    slide.series.some((s) => typeof d[s.dataKey] === "number" && d[s.dataKey] < 0),
  );
  const yName = numericYTitle;
  return (
    <div className="w-full min-w-0" style={{ height, minHeight: height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={lineData}
          margin={compact ? { top: 8, right: 8, left: 0, bottom: 0 } : { top: 8, right: 16, left: 24, bottom: 56 }}
        >
          <defs>
            {slide.series.map((s, i) => (
              <linearGradient key={s.dataKey} id={`dc-grad-${uid}-${i}-${s.dataKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0.04} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid stroke="var(--chart-grid, #e4e7ec)" strokeDasharray="3 3" />
          <XAxis
            dataKey={slide.xKey}
            type="category"
            interval={slide.xKind === "year" ? 0 : undefined}
            tickFormatter={slide.xKind === "year" || slide.xKind === "identifier" ? plainCategory : undefined}
            tick={{ fill: "var(--chart-tick, #667085)", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            height={compact ? 30 : 60}
            label={
              compact
                ? undefined
                : {
                    value: exactX,
                    position: "insideBottom",
                    offset: 0,
                    fill: "var(--chart-axis-title, #344054)",
                    fontSize: 12,
                    fontWeight: 600,
                  }
            }
          />
          <YAxis
            domain={yAxis.domain}
            ticks={yAxis.ticks}
            tick={{ fill: "var(--chart-tick, #667085)", fontSize: 10 }}
            tickFormatter={formatNumberCompact}
            tickLine={false}
            axisLine={false}
            width={compact ? 40 : 82}
            label={
              compact
                ? undefined
                : {
                    value: yName || "Value",
                    angle: -90,
                    position: "insideLeft",
                    offset: 0,
                    fill: "var(--chart-axis-title, #344054)",
                    fontSize: 12,
                    fontWeight: 600,
                    style: { textAnchor: "middle", lineHeight: 1.5 },
                  }
            }
          />

          {hasNegative && <ReferenceLine y={0} stroke="#1565ef" strokeWidth={1} />}
          <RTooltip {...categoryTooltip} />
          {slide.series.map((s, i) =>
            slide.type === "line" ? (
              <Line
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={1.75}
                dot={{ fill: s.color, stroke: "#fff", strokeWidth: 2, r: 3 }}
              />
            ) : (
              <Area
                key={s.dataKey}
                type="linear"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={1.5}
                fill={`url(#dc-grad-${uid}-${i}-${s.dataKey})`}
                dot={{ fill: s.color, stroke: "#fff", strokeWidth: 2, r: 3 }}
              />
            ),
          )}
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-1 flex items-center justify-center gap-5 text-[11px] text-[#475467]">
        {slide.series.map((s) => (
          <span key={s.dataKey} className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-sm" style={{ background: s.color }} />
            {s.name ?? s.dataKey}
          </span>
        ))}
      </div>
    </div>
  );
}
