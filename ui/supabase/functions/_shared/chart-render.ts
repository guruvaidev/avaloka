// Local chart rendering for the emailed PDFs.
//
// Both send-analysis-email and send-report-email used to POST the chart config
// — real data values, column names and series labels — to https://quickchart.io
// and embed the PNG it returned. That is a third-party sub-processor receiving
// customer data, and it appeared in no egress inventory and no DPA. The charts
// are now drawn as vector graphics with the pdf-lib both functions already
// embed, so nothing leaves the cluster and nothing has to be declared.
//
// `planChart` is pure: it turns a Chart.js-shaped config into a list of drawing
// primitives in PDF user space, which is what the unit tests exercise.
// `drawChartPlan` is the thin pdf-lib adapter that executes them.

export type Rgb = [number, number, number];

export type Prim =
  | { kind: "rect"; x: number; y: number; w: number; h: number; color: Rgb; opacity?: number }
  | { kind: "line"; x1: number; y1: number; x2: number; y2: number; thickness: number; color: Rgb }
  | { kind: "text"; text: string; x: number; y: number; size: number; bold: boolean; color: Rgb }
  | { kind: "circle"; cx: number; cy: number; r: number; color: Rgb }
  | { kind: "path"; d: string; originX: number; originY: number; color: Rgb };

/** Bottom-left origin, as pdf-lib uses. */
export type Box = { x: number; y: number; width: number; height: number };

/** Text width in points, e.g. pdf-lib's font.widthOfTextAtSize. */
export type Measure = (text: string, size: number, bold: boolean) => number;

const INK: Rgb = [0.06, 0.09, 0.16];
const MUTED: Rgb = [0.4, 0.45, 0.55];
const GRID: Rgb = [0.886, 0.91, 0.941];
const AXIS: Rgb = [0.74, 0.77, 0.82];

export const CHART_COLORS: Rgb[] = [
  [0.082, 0.396, 0.937], // #1565ef
  [0.086, 0.639, 0.29], // #16a34a
  [0.976, 0.451, 0.086], // #f97316
  [0.659, 0.333, 0.969], // #a855f7
  [0.863, 0.149, 0.149], // #dc2626
  [0.055, 0.647, 0.914], // #0ea5e9
  [0.792, 0.541, 0.016], // #ca8a04
  [0.033, 0.569, 0.698], // #0891b2
];

/** "#1565ef" -> [0.08, 0.4, 0.94]. Falls back to the palette on anything else. */
export function hexToRgb(value: unknown, fallback: Rgb): Rgb {
  if (typeof value !== "string") return fallback;
  const hex = value.trim().replace(/^#/, "");
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return fallback;
  return [
    parseInt(hex.slice(0, 2), 16) / 255,
    parseInt(hex.slice(2, 4), 16) / 255,
    parseInt(hex.slice(4, 6), 16) / 255,
  ];
}

type Dataset = { label: string; data: unknown; backgroundColor?: unknown; borderColor?: unknown; fill?: unknown };

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/** Axis ticks at 1/2/2.5/5 x 10^n, so labels read as round numbers. */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  let lo = Math.min(0, min);
  let hi = Math.max(0, max);
  if (lo === hi) hi = lo + 1;
  const rawStep = (hi - lo) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10) *
    magnitude;
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let t = lo; t <= hi + step / 1000; t += step) {
    ticks.push(Math.abs(t) < step / 1000 ? 0 : t);
  }
  return ticks;
}

/** Compact axis/value labels: 1.2k, 3.4M, 0.25. */
export function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${trim(v / 1e9)}B`;
  if (abs >= 1e6) return `${trim(v / 1e6)}M`;
  if (abs >= 1e3) return `${trim(v / 1e3)}k`;
  if (abs === 0) return "0";
  if (abs < 1) return trim(v, 2);
  return trim(v);
}

function trim(v: number, places = 1): string {
  const s = v.toFixed(places);
  return s.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function ellipsize(text: string, maxWidth: number, size: number, bold: boolean, measure: Measure): string {
  if (measure(text, size, bold) <= maxWidth) return text;
  let out = text;
  while (out.length > 1 && measure(`${out}...`, size, bold) > maxWidth) out = out.slice(0, -1);
  return `${out}...`;
}

type Series = { label: string; color: Rgb; values: (number | null)[] };

type Parsed = {
  type: "bar" | "line" | "area" | "scatter" | "pie" | "doughnut";
  title: string;
  xTitle: string;
  yTitle: string;
  labels: string[];
  series: Series[];
  points: { x: number; y: number }[];
  showLegend: boolean;
};

/** Read the Chart.js-shaped config buildChartJsConfig already produces. */
export function parseConfig(config: Record<string, unknown>): Parsed | null {
  const rawType = String(config.type ?? "bar").toLowerCase();
  const data = asRecord(config.data);
  if (!data) return null;
  const datasets = (Array.isArray(data.datasets) ? data.datasets : []) as Dataset[];
  if (!datasets.length) return null;

  const options = asRecord(config.options) ?? {};
  const plugins = asRecord(options.plugins) ?? {};
  const titleOpt = asRecord(plugins.title) ?? {};
  const legendOpt = asRecord(plugins.legend) ?? {};
  const scales = asRecord(options.scales) ?? {};
  const xScale = asRecord(scales.x) ?? {};
  const yScale = asRecord(scales.y) ?? {};

  const title = String(titleOpt.text ?? "");
  const xTitle = String((asRecord(xScale.title) ?? {}).text ?? "");
  const yTitle = String((asRecord(yScale.title) ?? {}).text ?? "");

  const labels = (Array.isArray(data.labels) ? data.labels : []).map((l) => String(l ?? ""));

  if (rawType === "scatter") {
    const raw = Array.isArray(datasets[0].data) ? (datasets[0].data as unknown[]) : [];
    const points: { x: number; y: number }[] = [];
    for (const p of raw) {
      const r = asRecord(p);
      const x = num(r?.x);
      const y = num(r?.y);
      if (x !== null && y !== null) points.push({ x, y });
    }
    if (!points.length) return null;
    return {
      type: "scatter",
      title,
      xTitle,
      yTitle,
      labels: [],
      series: [{
        label: String(datasets[0].label ?? ""),
        color: hexToRgb(datasets[0].backgroundColor, CHART_COLORS[0]),
        values: [],
      }],
      points,
      showLegend: false,
    };
  }

  const series: Series[] = datasets.map((d, i) => {
    const fallback = CHART_COLORS[i % CHART_COLORS.length];
    const swatch = Array.isArray(d.backgroundColor) ? d.backgroundColor[i] : d.backgroundColor;
    return {
      label: String(d.label ?? ""),
      color: hexToRgb(swatch ?? d.borderColor, fallback),
      values: (Array.isArray(d.data) ? (d.data as unknown[]) : []).map(num),
    };
  });

  if (rawType === "pie" || rawType === "doughnut") {
    const slices = (Array.isArray(datasets[0].data) ? (datasets[0].data as unknown[]) : []).map(num);
    const colors = Array.isArray(datasets[0].backgroundColor) ? datasets[0].backgroundColor : [];
    if (!slices.some((v) => v !== null && v > 0)) return null;
    return {
      type: rawType,
      title,
      xTitle: "",
      yTitle: "",
      labels,
      series: slices.map((v, i) => ({
        label: labels[i] ?? "",
        color: hexToRgb(colors[i], CHART_COLORS[i % CHART_COLORS.length]),
        values: [v],
      })),
      points: [],
      showLegend: true,
    };
  }

  const type = rawType === "line" ? (datasets.some((d) => d.fill === true) ? "area" : "line") : "bar";
  if (!labels.length) return null;
  return {
    type,
    title,
    xTitle,
    yTitle,
    labels,
    series,
    points: [],
    showLegend: legendOpt.display !== false && series.length > 1,
  };
}

/**
 * Turn a chart config into drawing primitives inside `box`.
 * Returns an empty list when there is nothing renderable, so the caller can
 * leave the space out of the PDF rather than draw an empty frame.
 */
export function planChart(config: Record<string, unknown>, box: Box, measure: Measure): Prim[] {
  const parsed = parseConfig(config);
  if (!parsed) return [];

  const out: Prim[] = [];
  const TITLE_SIZE = 11;
  const LABEL_SIZE = 7;
  const LEGEND_SIZE = 7.5;

  let top = box.y + box.height;

  if (parsed.title) {
    const text = ellipsize(parsed.title, box.width, TITLE_SIZE, true, measure);
    out.push({
      kind: "text",
      text,
      x: box.x + (box.width - measure(text, TITLE_SIZE, true)) / 2,
      y: top - TITLE_SIZE,
      size: TITLE_SIZE,
      bold: true,
      color: INK,
    });
    top -= TITLE_SIZE + 8;
  }

  if (parsed.showLegend && parsed.type !== "pie" && parsed.type !== "doughnut") {
    top = drawLegend(out, parsed.series, box, top, LEGEND_SIZE, measure);
  }

  if (parsed.type === "pie" || parsed.type === "doughnut") {
    return [...out, ...planPie(parsed, box, top, LEGEND_SIZE, measure)];
  }

  // --- cartesian plot area -------------------------------------------------
  const yLabels = axisLabels(parsed);
  const gutterLeft = Math.max(
    18,
    ...yLabels.map((t) => measure(t, LABEL_SIZE, false) + 6),
  ) + (parsed.yTitle ? 10 : 0);
  const gutterBottom = 16 + (parsed.xTitle ? 10 : 0);

  // The y-axis title gets its own row above the plot: drawn level with the top
  // tick it would sit on top of its label.
  if (parsed.yTitle) {
    const t = ellipsize(parsed.yTitle, box.width, LABEL_SIZE, false, measure);
    out.push({ kind: "text", text: t, x: box.x, y: top - LABEL_SIZE, size: LABEL_SIZE, bold: false, color: MUTED });
    top -= LABEL_SIZE + 3;
  }

  const plot: Box = {
    x: box.x + gutterLeft,
    y: box.y + gutterBottom,
    width: box.width - gutterLeft - 6,
    height: top - (box.y + gutterBottom),
  };
  if (plot.width <= 20 || plot.height <= 20) return out;

  const { ticks, lo, hi } = parsed.type === "scatter"
    ? verticalRange(parsed.points.map((p) => p.y))
    : verticalRange(parsed.series.flatMap((s) => s.values.filter((v): v is number => v !== null)));

  const toY = (v: number) => plot.y + ((v - lo) / (hi - lo)) * plot.height;

  // Gridlines and y tick labels.
  for (const t of ticks) {
    const gy = toY(t);
    out.push({ kind: "line", x1: plot.x, y1: gy, x2: plot.x + plot.width, y2: gy, thickness: 0.5, color: GRID });
    const label = formatValue(t);
    out.push({
      kind: "text",
      text: label,
      x: plot.x - 4 - measure(label, LABEL_SIZE, false),
      y: gy - LABEL_SIZE / 3,
      size: LABEL_SIZE,
      bold: false,
      color: MUTED,
    });
  }

  // Axes.
  out.push({ kind: "line", x1: plot.x, y1: plot.y, x2: plot.x + plot.width, y2: plot.y, thickness: 0.75, color: AXIS });
  out.push({ kind: "line", x1: plot.x, y1: plot.y, x2: plot.x, y2: plot.y + plot.height, thickness: 0.75, color: AXIS });
  if (lo < 0 && hi > 0) {
    const zero = toY(0);
    out.push({ kind: "line", x1: plot.x, y1: zero, x2: plot.x + plot.width, y2: zero, thickness: 0.75, color: AXIS });
  }

  if (parsed.type === "scatter") {
    planScatter(out, parsed, plot, toY, LABEL_SIZE, measure);
  } else if (parsed.type === "bar") {
    planBars(out, parsed, plot, toY, lo);
  } else {
    planLines(out, parsed, plot, toY, lo, parsed.type === "area");
  }

  if (parsed.type !== "scatter") {
    planCategoryLabels(out, parsed.labels, plot, LABEL_SIZE, measure);
  }

  // Axis titles.
  if (parsed.xTitle) {
    const t = ellipsize(parsed.xTitle, plot.width, LABEL_SIZE, false, measure);
    out.push({
      kind: "text",
      text: t,
      x: plot.x + (plot.width - measure(t, LABEL_SIZE, false)) / 2,
      y: box.y + 1,
      size: LABEL_SIZE,
      bold: false,
      color: MUTED,
    });
  }
  return out;
}

function axisLabels(parsed: Parsed): string[] {
  const values = parsed.type === "scatter"
    ? parsed.points.map((p) => p.y)
    : parsed.series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  return verticalRange(values).ticks.map(formatValue);
}

function verticalRange(values: number[]): { ticks: number[]; lo: number; hi: number } {
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const ticks = niceTicks(min, max);
  return { ticks, lo: ticks[0], hi: ticks[ticks.length - 1] };
}

function drawLegend(out: Prim[], series: Series[], box: Box, top: number, size: number, measure: Measure): number {
  const SWATCH = 6;
  const GAP = 10;
  let x = box.x;
  let y = top - size;
  let used = 0;
  for (const s of series) {
    const label = ellipsize(s.label || "series", 120, size, false, measure);
    const width = SWATCH + 3 + measure(label, size, false) + GAP;
    if (x + width > box.x + box.width && x > box.x) {
      x = box.x;
      y -= size + 4;
      used++;
      if (used > 2) break; // three rows of legend is already too many
    }
    out.push({ kind: "rect", x, y: y + 1, w: SWATCH, h: SWATCH, color: s.color });
    out.push({ kind: "text", text: label, x: x + SWATCH + 3, y: y + 1, size, bold: false, color: MUTED });
    x += width;
  }
  return y - 8;
}

function planBars(out: Prim[], parsed: Parsed, plot: Box, toY: (v: number) => number, lo: number): void {
  const groups = parsed.labels.length;
  const slot = plot.width / Math.max(1, groups);
  const inner = Math.max(1, slot * 0.72);
  const barWidth = inner / Math.max(1, parsed.series.length);
  const baseline = toY(Math.max(lo, 0));

  for (let g = 0; g < groups; g++) {
    const left = plot.x + g * slot + (slot - inner) / 2;
    parsed.series.forEach((s, si) => {
      const v = s.values[g];
      if (v === null || v === undefined) return;
      const top = toY(v);
      const h = Math.abs(top - baseline);
      out.push({
        kind: "rect",
        x: left + si * barWidth,
        y: Math.min(top, baseline),
        w: Math.max(0.6, barWidth - 1),
        h: Math.max(0.4, h),
        color: s.color,
      });
    });
  }
}

function planLines(
  out: Prim[],
  parsed: Parsed,
  plot: Box,
  toY: (v: number) => number,
  lo: number,
  fill: boolean,
): void {
  const n = parsed.labels.length;
  const step = n > 1 ? plot.width / (n - 1) : 0;
  const xAt = (i: number) => (n > 1 ? plot.x + i * step : plot.x + plot.width / 2);
  const baseline = toY(Math.max(lo, 0));

  for (const s of parsed.series) {
    if (fill) {
      // Fill as thin columns under the line: pdf-lib has no polygon fill, and a
      // column fill is indistinguishable at this width.
      for (let i = 0; i < n - 1; i++) {
        const a = s.values[i];
        const b = s.values[i + 1];
        if (a === null || b === null || a === undefined || b === undefined) continue;
        const slices = Math.max(1, Math.ceil(step));
        for (let k = 0; k < slices; k++) {
          const t = k / slices;
          const v = a + (b - a) * t;
          const yTop = toY(v);
          out.push({
            kind: "rect",
            x: xAt(i) + step * t,
            y: Math.min(yTop, baseline),
            w: step / slices + 0.3,
            h: Math.abs(yTop - baseline),
            color: s.color,
            opacity: 0.18,
          });
        }
      }
    }
    for (let i = 0; i < n - 1; i++) {
      const a = s.values[i];
      const b = s.values[i + 1];
      if (a === null || b === null || a === undefined || b === undefined) continue;
      out.push({
        kind: "line",
        x1: xAt(i),
        y1: toY(a),
        x2: xAt(i + 1),
        y2: toY(b),
        thickness: 1.4,
        color: s.color,
      });
    }
    if (n <= 40) {
      for (let i = 0; i < n; i++) {
        const v = s.values[i];
        if (v === null || v === undefined) continue;
        out.push({ kind: "circle", cx: xAt(i), cy: toY(v), r: 1.8, color: s.color });
      }
    }
  }
}

function planScatter(
  out: Prim[],
  parsed: Parsed,
  plot: Box,
  toY: (v: number) => number,
  size: number,
  measure: Measure,
): void {
  const xs = parsed.points.map((p) => p.x);
  const xTicks = niceTicks(Math.min(...xs), Math.max(...xs));
  const xLo = xTicks[0];
  const xHi = xTicks[xTicks.length - 1];
  const toX = (v: number) => plot.x + ((v - xLo) / (xHi - xLo || 1)) * plot.width;

  for (const t of xTicks) {
    const label = formatValue(t);
    out.push({
      kind: "text",
      text: label,
      x: toX(t) - measure(label, size, false) / 2,
      y: plot.y - size - 2,
      size,
      bold: false,
      color: MUTED,
    });
  }
  const color = parsed.series[0]?.color ?? CHART_COLORS[0];
  for (const p of parsed.points) {
    out.push({ kind: "circle", cx: toX(p.x), cy: toY(p.y), r: 2, color });
  }
}

function planCategoryLabels(out: Prim[], labels: string[], plot: Box, size: number, measure: Measure): void {
  if (!labels.length) return;
  const slot = plot.width / labels.length;
  // Show every nth label so they never overlap.
  const every = Math.max(1, Math.ceil((size * 4) / Math.max(1, slot)));
  for (let i = 0; i < labels.length; i++) {
    if (i % every !== 0) continue;
    const text = ellipsize(labels[i], Math.max(14, slot * every - 2), size, false, measure);
    out.push({
      kind: "text",
      text,
      x: plot.x + i * slot + slot / 2 - measure(text, size, false) / 2,
      y: plot.y - size - 2,
      size,
      bold: false,
      color: MUTED,
    });
  }
}

function planPie(parsed: Parsed, box: Box, top: number, size: number, measure: Measure): Prim[] {
  const out: Prim[] = [];
  const total = parsed.series.reduce((acc, s) => acc + Math.max(0, s.values[0] ?? 0), 0);
  if (total <= 0) return out;

  const height = top - box.y;
  const legendWidth = Math.min(
    box.width * 0.45,
    Math.max(70, ...parsed.series.map((s) => measure(`${s.label} 00.0%`, size, false) + 14)),
  );
  const diameter = Math.min(height - 6, box.width - legendWidth - 10);
  if (diameter <= 20) return out;

  const radius = diameter / 2;
  const cx = box.x + radius;
  const cy = box.y + height / 2;
  // SVG space for drawSvgPath: y grows downward from the path origin, which we
  // place at the top-left of the circle's bounding box.
  const originX = cx - radius;
  const originY = cy + radius;
  const inner = parsed.type === "doughnut" ? radius * 0.55 : 0;

  let angle = -Math.PI / 2; // start at 12 o'clock
  parsed.series.forEach((s) => {
    const value = Math.max(0, s.values[0] ?? 0);
    if (value <= 0) return;
    const sweep = (value / total) * Math.PI * 2;
    out.push({
      kind: "path",
      d: slicePath(radius, inner, angle, angle + sweep),
      originX,
      originY,
      color: s.color,
    });
    angle += sweep;
  });

  let ly = cy + Math.min(radius, (parsed.series.length * (size + 4)) / 2) - size;
  for (const s of parsed.series) {
    const value = Math.max(0, s.values[0] ?? 0);
    if (value <= 0) continue;
    const pct = `${((value / total) * 100).toFixed(1)}%`;
    const label = ellipsize(s.label || "—", legendWidth - measure(pct, size, false) - 16, size, false, measure);
    out.push({ kind: "rect", x: box.x + diameter + 10, y: ly, w: 6, h: 6, color: s.color });
    out.push({
      kind: "text",
      text: `${label} ${pct}`,
      x: box.x + diameter + 19,
      y: ly - 0.5,
      size,
      bold: false,
      color: MUTED,
    });
    ly -= size + 4;
  }

  return out;
}

/** One pie/doughnut slice as an SVG path in y-down space. */
export function slicePath(radius: number, inner: number, from: number, to: number): string {
  const pt = (r: number, a: number) =>
    `${(radius + r * Math.cos(a)).toFixed(3)} ${(radius + r * Math.sin(a)).toFixed(3)}`;
  const large = to - from > Math.PI ? 1 : 0;
  // A full circle cannot be one arc; split it.
  if (to - from >= Math.PI * 2 - 1e-6) {
    const half = from + Math.PI;
    return [
      `M ${pt(radius, from)}`,
      `A ${radius} ${radius} 0 0 1 ${pt(radius, half)}`,
      `A ${radius} ${radius} 0 0 1 ${pt(radius, from)}`,
      "Z",
    ].join(" ");
  }
  if (inner > 0) {
    return [
      `M ${pt(inner, from)}`,
      `L ${pt(radius, from)}`,
      `A ${radius} ${radius} 0 ${large} 1 ${pt(radius, to)}`,
      `L ${pt(inner, to)}`,
      `A ${inner} ${inner} 0 ${large} 0 ${pt(inner, from)}`,
      "Z",
    ].join(" ");
  }
  return [
    `M ${radius} ${radius}`,
    `L ${pt(radius, from)}`,
    `A ${radius} ${radius} 0 ${large} 1 ${pt(radius, to)}`,
    "Z",
  ].join(" ");
}

/** pdf-lib bindings. `rgb` and `page` are passed in so this module imports nothing. */
// deno-lint-ignore no-explicit-any
export function drawChartPlan(page: any, prims: Prim[], deps: {
  // deno-lint-ignore no-explicit-any
  font: any;
  // deno-lint-ignore no-explicit-any
  fontBold: any;
  // deno-lint-ignore no-explicit-any
  rgb: (r: number, g: number, b: number) => any;
}): void {
  for (const p of prims) {
    switch (p.kind) {
      case "rect":
        page.drawRectangle({
          x: p.x,
          y: p.y,
          width: p.w,
          height: p.h,
          color: deps.rgb(...p.color),
          ...(p.opacity === undefined ? {} : { opacity: p.opacity }),
        });
        break;
      case "line":
        page.drawLine({
          start: { x: p.x1, y: p.y1 },
          end: { x: p.x2, y: p.y2 },
          thickness: p.thickness,
          color: deps.rgb(...p.color),
        });
        break;
      case "text":
        page.drawText(p.text, {
          x: p.x,
          y: p.y,
          size: p.size,
          font: p.bold ? deps.fontBold : deps.font,
          color: deps.rgb(...p.color),
        });
        break;
      case "circle":
        page.drawCircle({ x: p.cx, y: p.cy, size: p.r, color: deps.rgb(...p.color) });
        break;
      case "path":
        page.drawSvgPath(p.d, { x: p.originX, y: p.originY, color: deps.rgb(...p.color), borderWidth: 0 });
        break;
    }
  }
}
