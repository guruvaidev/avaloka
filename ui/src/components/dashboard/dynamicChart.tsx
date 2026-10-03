import {
  ComposedChart,
  Area,
  Line,
  BarChart,
  Bar,
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
  /** Chart couldn't be drawn; title + insights are still shown. */
  undrawable?: boolean;
};

export const PALETTE = ["#1565ef", "#f97066", "#12b76a", "#f79009", "#7a5af8", "#06aed4", "#ee46bc"];

function formatNumberCompact(n: number | string): string {
  const num = typeof n === "string" ? Number(n) : n;
  if (num === null || num === undefined || !Number.isFinite(num)) return String(n);
  const abs = Math.abs(num);
  const sign = num < 0 ? "-" : "";
  if (abs >= 1_000_000_000) return `${sign}${(abs / 1_000_000_000).toFixed(1).replace(/\.0$/, "")}B`;
  if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (abs >= 1_000) return `${sign}${(abs / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(num);
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

function buildFromEncodings(raw: any, samples: any[]): { data: any[]; xKey: string; series: Series[] } | null {
  const enc = raw?.encodings;
  if (!enc || !samples?.length) return null;
  const xFieldRaw = enc.x?.field;
  if (!xFieldRaw) return null;
  const xField = resolveFieldKey(samples, String(xFieldRaw));
  if (!xField) return null;
  const aggregate = String(enc.y?.aggregate ?? "").toLowerCase();
  const topK = raw?.config?.top_k as number | undefined;
  let yFieldRaw = enc.y?.field as string | undefined;
  let yField = yFieldRaw ? (resolveFieldKey(samples, yFieldRaw) ?? undefined) : undefined;
  const countKey = "__count__";

  if (enc.x?.bin) {
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

  if (pickType(raw) === "scatter") {
    const data = samples
      .map((r) => ({ [xField]: Number(r?.[xField]), [yField!]: Number(r?.[yField!]) }))
      .filter((r) => Number.isFinite(r[xField]) && Number.isFinite(r[yField!]));
    return { data, xKey: xField, series: [{ dataKey: yField!, name: yField!, color: PALETTE[0] }] };
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
    else v = g.vals.length ? g.sum / g.vals.length : g.count;
    return { [xField]: k, [yField!]: Number(v.toFixed(4)) };
  });
  if (topK && rows.length > topK) rows = rows.slice(0, topK);
  return {
    data: rows,
    xKey: xField,
    series: [{ dataKey: yField!, name: yField === countKey ? "Count" : yField!, color: PALETTE[0] }],
  };
}

function pickInsights(raw: any): string[] {
  const src = raw?.insights ?? raw?.key_insights ?? raw?.takeaways ?? raw?.notes ?? [];
  return asArray(src)
    .map((it: any) => (typeof it === "string" ? it : (it?.text ?? it?.insight ?? it?.message)))
    .filter(Boolean) as string[];
}

export function normalizeChart(raw: any, samples?: any[]): Slide | null {
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
  const kind: Slide["chartKind"] = enc?.x?.bin || t.includes("hist")
    ? "histogram"
    : slide.type === "donut" || slide.type === "pie"
      ? "pie"
      : slide.type === "area"
        ? "line"
        : slide.type;
  const str = (v: unknown) => (v == null || v === "" ? null : String(v));
  return {
    ...slide,
    id: raw?.id != null ? String(raw.id) : undefined,
    chartKind: kind,
    xField: str(enc.x?.field ?? enc.color?.field),
    yField: str(enc.y?.field ?? enc.theta?.field),
    aggregate: str(enc.y?.aggregate ?? enc.theta?.aggregate),
    reason: str(raw?.reason),
  };
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
function buildCategorical(raw: any, samples: any[] | undefined, type: ChartType): {
  data: any[]; xKey: string; series: Series[]; xLabel: string; yLabel: string;
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
    const cf = dd.category_field ?? catFieldRaw;
    const vf = dd.value_field ?? valFieldRaw ?? "count";
    const data = dd.points.map((p: any) => ({
      category: categoryLabel(p?.[cf]),
      value: Number(p?.[vf]) || 0,
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
    value: valField && aggregate === "sum"
      ? g.sum
      : valField && /mean|avg|average/.test(aggregate)
        ? (g.count ? g.sum / g.count : 0)
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

    const cat = buildCategorical(raw, samples, type);
    if (cat) {
      return {
        title, subtitle, type, data: cat.data, xKey: cat.xKey, series: cat.series,
        insights: pickInsights(raw), xLabel: cat.xLabel, yLabel: cat.yLabel,
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
          const valField = valFieldRaw && String(valFieldRaw) !== String(catFieldRaw)
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

export function normalizeVizConfig(config: any, samples?: any[]): Slide[] {
  if (!config) return [];
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
    Object.values(config).forEach((v) => out.push(...normalizeVizConfig(v, samples)));
    if (out.length) return out;
  }
  const list = (Array.isArray(config) && config) ||
    config.charts ||
    config.visualizations ||
    config.plots ||
    config.figures || [config];
  return asArray(list)
    .map((c) => normalizeChart(c, samples))
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
}: {
  slide: Slide;
  height?: number;
  compact?: boolean;
  chartKey?: string;
}) {
  const uid = (chartKey ?? slide.title ?? "chart").replace(/[^a-zA-Z0-9_-]/g, "_");
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
      <div className="flex w-full items-center justify-center text-[12px] text-[#667085]" style={{ height: compact ? Math.min(height, 120) : height }}>
        This chart couldn't be drawn from the available data.
      </div>
    );
  }

  if (slide.type === "pie" || slide.type === "donut") {
    const pieData = getPieRenderData(slide);
    if (!pieData.length) return null;
    const total = pieData.reduce((s, r) => s + (r.value || 0), 0) || 1;
    const pct = (v: number) => `${((v / total) * 100).toFixed(1)}%`;

    return (
      <div className="flex h-auto w-full min-w-0 flex-wrap gap-3 overflow-visible">
        <div className="h-[230px] min-h-[230px] min-w-[220px] flex-[1.2_1_0%] shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <RTooltip
                separator=": "
                formatter={(value: any, name: any) => [
                  `${formatNumberCompact(value)} (${pct(Number(value))})`,
                  name,
                ]}
              />
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                innerRadius={Math.max(40, height / 4)}
                outerRadius={Math.max(60, height / 2.7)}
                paddingAngle={2}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        {!compact ? (
          <div className="max-h-[230px] min-w-[160px] flex-1 shrink-0 overflow-y-auto pr-1">
            <ul className="flex w-full flex-col gap-1.5">
              {pieData.map((item, i) => (
                <li key={`${item.name}-${i}`} className="flex min-w-0 items-center gap-2 text-[11px] text-[#475467]">
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

    const yKey = slide.series[0]?.dataKey ?? "y";
    const yName = slide.series[0]?.name ?? yKey;
    return (
      <div className="w-full min-w-0" style={{ height, minHeight: height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={compact ? { top: 8, right: 8, left: 0, bottom: 0 } : { top: 8, right: 16, left: 24, bottom: 56 }}>
            <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey={slide.xKey}
              tick={{ fill: "#667085", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              height={compact ? 30 : 60}
              label={compact ? undefined : { value: `X: ${slide.xLabel ?? slide.xKey}`, position: "insideBottom", offset: 0, fill: "#344054", fontSize: 12, fontWeight: 600 }}
            />
            <YAxis
              type="number"
              dataKey={yKey}
              tick={{ fill: "#667085", fontSize: 10 }}
              tickFormatter={formatNumberCompact}
              tickLine={false}
              axisLine={false}
              width={compact ? 40 : 72}
              label={compact ? undefined : { value: `Y: ${yName}`, angle: -90, position: "insideLeft", offset: 10, fill: "#344054", fontSize: 12, fontWeight: 600, style: { textAnchor: "middle" } }}
            />
            <ZAxis range={[20, 20]} />
            <RTooltip cursor={{ strokeDasharray: "3 3" }} formatter={tooltipCompactFormatter} />
            <Scatter data={slide.data} fill={slide.series[0]?.color ?? PALETTE[0]} />

          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }


  if (slide.type === "bar") {
    const margin = compact ? { top: 4, right: 4, left: -16, bottom: 0 } : { top: 8, right: 16, left: 24, bottom: 56 };
    const tickSize = compact ? 9 : 10;
    const yName = slide.yLabel ?? slide.series[0]?.name ?? slide.series[0]?.dataKey ?? "Value";
    return (
      <div className="w-full min-w-0" style={{ height, minHeight: height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={slide.data} margin={margin}>
            <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 3" />
            <XAxis
              dataKey={slide.xKey}
              tick={{ fill: "#667085", fontSize: tickSize }}
              tickLine={false}
              axisLine={false}
              hide={compact}
              height={compact ? 30 : 60}
              label={compact ? undefined : { value: `X: ${slide.xLabel ?? slide.xKey}`, position: "insideBottom", offset: 0, fill: "#344054", fontSize: 12, fontWeight: 600 }}
            />
            <YAxis
              tick={{ fill: "#667085", fontSize: tickSize }}
              tickFormatter={formatNumberCompact}
              tickLine={false}
              axisLine={false}
              width={compact ? 28 : 72}
              label={compact ? undefined : { value: `Y: ${yName}`, angle: -90, position: "insideLeft", offset: 10, fill: "#344054", fontSize: 12, fontWeight: 600, style: { textAnchor: "middle" } }}
            />
            <RTooltip {...categoryTooltip} />
            {slide.series.map((s) => (

              <Bar
                key={s.dataKey}
                dataKey={s.dataKey}
                name={s.name}
                fill={s.color ?? PALETTE[0]}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }


  const lineData = getLineRenderData(slide).data;
  const hasNegative = lineData.some((d) =>
    slide.series.some((s) => typeof d[s.dataKey] === "number" && d[s.dataKey] < 0),
  );
  const yName = slide.yLabel ?? slide.series[0]?.name ?? slide.series[0]?.dataKey ?? "";
  return (
    <div className="w-full min-w-0" style={{ height, minHeight: height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={lineData} margin={compact ? { top: 8, right: 8, left: 0, bottom: 0 } : { top: 8, right: 16, left: 24, bottom: 56 }}>
          <defs>
            {slide.series.map((s, i) => (
              <linearGradient key={s.dataKey} id={`dc-grad-${uid}-${i}-${s.dataKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0.04} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 3" />
          <XAxis
            dataKey={slide.xKey}
            tick={{ fill: "#667085", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            height={compact ? 30 : 60}
            label={compact ? undefined : { value: `X: ${slide.xLabel ?? slide.xKey}`, position: "insideBottom", offset: 0, fill: "#344054", fontSize: 12, fontWeight: 600 }}
          />
          <YAxis
            tick={{ fill: "#667085", fontSize: 10 }}
            tickFormatter={formatNumberCompact}
            tickLine={false}
            axisLine={false}
            width={compact ? 40 : 72}
            label={compact ? undefined : { value: `Y: ${yName || "Value"}`, angle: -90, position: "insideLeft", offset: 10, fill: "#344054", fontSize: 12, fontWeight: 600, style: { textAnchor: "middle" } }}
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
