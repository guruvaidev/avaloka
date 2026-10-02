type AxisSeries = { dataKey: string; name?: string };

export type AxisSlide = {
  xKey: string;
  series: AxisSeries[];
  chartKind?: string;
  xField?: string | null;
  yField?: string | null;
  aggregate?: string | null;
  xAxisTitle?: string;
  yAxisTitle?: string;
};

export function fieldTitle(value: string): string {
  const words = value
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Value";
}

export function aggregateTitle(aggregate: string | null | undefined, field?: string | null): string {
  const name = field ? fieldTitle(field) : "values";
  const measure = name === name.toUpperCase() ? name : name.charAt(0).toLowerCase() + name.slice(1);
  switch (aggregate?.toLowerCase()) {
    case "count": return "Count of records";
    case "sum": return `Total ${measure}`;
    case "mean":
    case "avg":
    case "average": return `Average ${measure}`;
    case "median": return `Median ${measure}`;
    case "min": return `Minimum ${measure}`;
    case "max": return `Maximum ${measure}`;
    default: return field ? name : "Value";
  }
}

export function chartAxisTitles(slide: AxisSlide): { x: string; y: string } {
  const first = slide.series[0];
  const xField = slide.xField || slide.xKey;
  const x = slide.xAxisTitle?.trim() || (slide.chartKind === "histogram"
    ? `${fieldTitle(xField)} (bin start)`
    : fieldTitle(xField));

  if (slide.yAxisTitle?.trim()) return { x, y: slide.yAxisTitle.trim() };
  if (slide.chartKind === "histogram" || slide.aggregate?.toLowerCase() === "count"
    || first?.dataKey === "__count__") {
    return { x, y: "Count of records" };
  }
  if (slide.aggregate && slide.yField) {
    return { x, y: aggregateTitle(slide.aggregate, slide.yField) };
  }
  const names = [...new Set(slide.series.map((series) => fieldTitle(series.name || series.dataKey)))];
  return { x, y: names.join(" / ") || "Value" };
}
