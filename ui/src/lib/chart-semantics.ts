export type ColumnProfile = { name?: string; field?: string; role?: string; is_year?: boolean };
export type AxisKind = "measure" | "category" | "year" | "identifier";

const CODE_SUFFIX = /(?:^|[_\s-])(id|no|num|number|code|uuid|key|phone|zip|zipcode|postcode|pin|tract|census_tract|fips|geoid|msamd|msa|cbsa)$/i;
const YEAR_NAME = /(?:^|[_\s-])year$/i;

export function plainCategory(value: unknown): string {
  // Strip only a zero fractional suffix; preserve leading zeroes in string codes.
  return String(value ?? "").replace(/^(\d+)\.0+$/, "$1");
}

export function axisKind(field: string, values: unknown[], columns: ColumnProfile[] = [], yearChart = false): AxisKind {
  const profile = columns.find((column) => (column.name ?? column.field)?.toLowerCase() === field.toLowerCase());
  if (profile?.role === "identifier" || CODE_SUFFIX.test(field)) return "identifier";
  if (profile?.is_year || ((YEAR_NAME.test(field) || yearChart) && values.length > 0 && values.every((value) => {
    const number = Number(value);
    return Number.isInteger(number) && number >= 1900 && number <= 2100;
  }))) return "year";
  const numeric = values.length > 0 && values.every((value) => value != null && value !== "" && Number.isFinite(Number(value)));
  const flag = numeric && values.every((value) => Number(value) === 0 || Number(value) === 1);
  if (flag || (profile?.role && profile.role !== "measure")) return "category";
  return numeric ? "measure" : "category";
}

export function scalePolicy(values: unknown[]) {
  const numbers = values.map((value) => value == null || value === "" ? NaN : Number(value));
  const positive = numbers.length > 0 && numbers.every((value) => Number.isFinite(value) && value > 0);
  const min = positive ? numbers.reduce((minimum, value) => Math.min(minimum, value), Infinity) : 0;
  const max = positive ? numbers.reduce((maximum, value) => Math.max(maximum, value), -Infinity) : 0;
  return { positive, automatic: positive && max / min >= 100, min, max };
}

export function logTicks(values: unknown[]): number[] | undefined {
  const policy = scalePolicy(values);
  if (!policy.positive) return undefined;
  const ticks: number[] = [];
  for (let power = Math.ceil(Math.log10(policy.min)); power <= Math.floor(Math.log10(policy.max)); power++) {
    ticks.push(10 ** power);
  }
  return ticks.length ? ticks : undefined;
}

export function extremePointsNote(hidden: unknown, n: unknown): string | null {
  if (typeof hidden !== "number" || !Number.isInteger(hidden) || hidden <= 0 || typeof n !== "number" || !Number.isInteger(n) || n < 0) return null;
  return `${hidden.toLocaleString("en-US")} extreme points left off so the rest is readable; the correlation uses all ${n.toLocaleString("en-US")} rows.`;
}

/** Apply only display semantics; never consult preview rows when points are present. */
export function applyChartSemantics<T extends {
  type: string; title: string; data: any[]; xKey: string;
  series: { dataKey: string; name?: string; color?: string }[];
  chartKind?: string; xField?: string | null; yField?: string | null;
  xAxisTitle?: string; yAxisTitle?: string;
}>(slide: T, columns: ColumnProfile[] = []): (T & { xKind: AxisKind; yKind: AxisKind }) | null {
  const yKey = slide.series[0]?.dataKey ?? "y";
  const xField = slide.xField ?? slide.xKey;
  const yField = slide.yField ?? yKey;
  const xKind = axisKind(xField, slide.data.map((point) => point[slide.xKey]), columns, /year/i.test(slide.chartKind ?? ""));
  const yKind = axisKind(yField, slide.data.map((point) => point[yKey]), columns);
  if (slide.type === "scatter" && (xKind !== "measure" || yKind !== "measure")) {
    if (!slide.data.length) return null;
    if (xKind !== "measure" && yKind !== "measure") return null;
    const categoryKey = xKind === "measure" ? yKey : slide.xKey;
    const measureKey = xKind === "measure" ? slide.xKey : yKey;
    const categoryField = xKind === "measure" ? yField : xField;
    const measureField = xKind === "measure" ? xField : yField;
    const groups = new Map<string, { sum: number; n: number }>();
    for (const point of slide.data) {
      const value = Number(point[measureKey]);
      if (!Number.isFinite(value) || point[categoryKey] == null) continue;
      const category = plainCategory(point[categoryKey]);
      const group = groups.get(category) ?? { sum: 0, n: 0 };
      group.sum += value;
      group.n++;
      groups.set(category, group);
    }
    const categoryKind = xKind === "measure" ? yKind : xKind;
    const data = Array.from(groups, ([category, group]) => ({ category, value: group.sum / group.n }));
    if (categoryKind === "year") data.sort((a, b) => Number(a.category) - Number(b.category));
    if (!data.length) return null;
    return { ...slide, type: "bar", chartKind: "bar", title: `Average ${measureField} by ${categoryField}`,
      xKey: "category", series: [{ dataKey: "value", name: measureField, color: slide.series[0]?.color }], data,
      xField: categoryField, yField: measureField, xAxisTitle: categoryField, yAxisTitle: measureField,
      xKind: categoryKind, yKind: "measure" };
  }
  if (slide.chartKind === "histogram" && (xKind === "identifier" || xKind === "year")) return null;
  // A code/year must never be placed on the dependent numeric axis.
  if (slide.type !== "pie" && slide.type !== "donut" && (yKind === "identifier" || yKind === "year")) return null;
  if (xKind === "identifier" || xKind === "year") {
    const data = slide.data.map((point) => ({ ...point, [slide.xKey]: plainCategory(point[slide.xKey]) }));
    if (xKind === "year") data.sort((a, b) => Number(a[slide.xKey]) - Number(b[slide.xKey]));
    return { ...slide, data, xKind, yKind };
  }
  return { ...slide, xKind, yKind };
}