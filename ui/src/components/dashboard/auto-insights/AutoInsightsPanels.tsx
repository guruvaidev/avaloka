import { useState } from "react";
import { ChevronDown, ChevronLeft, ChevronRight, Search, X, ArrowUp, ArrowDown } from "lucide-react";
import { Button } from "@/components/base/buttons/button";
import { Checkbox } from "@/components/base/checkbox/checkbox";
import type { Slide } from "../dynamicChart";
import type { InsightGroup } from "@/lib/insight-groups";

export type InsightView = "grid" | "stack" | "columns";
export type Column = {
  name: string; dtype?: string; role?: string; meaning?: string;
  predictive_value?: "high" | "medium" | "low";
  missing_pct?: number; n_unique?: number; min?: unknown; max?: unknown;
  mean?: number; median?: number; n_months?: number; first?: unknown; last?: unknown;
  top_values?: { value: unknown; count?: number; pct?: number }[];
};

export function number(value: unknown, compact = false): string {
  if (value == null || value === "" || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2, ...(compact ? { notation: "compact" as const } : {}) }).format(Number(value));
}
export const percent = (value: unknown) => value == null ? "—" : `${Number(value).toFixed(1).replace(/\.0$/, "")}%`;
export const date = (value: unknown) => value == null ? "—" : String(value).slice(0, 7);
export function columnType(column: Column) {
  if (/date|time/.test((column.dtype ?? "").toLowerCase())) return "Date";
  if (/numeric|number|int|float|double|decimal/.test((column.dtype ?? "").toLowerCase())) return "Number";
  return "Category";
}
export function columnsOf(config: any): Column[] {
  const columns: Column[] = Array.isArray(config?.column_stats) ? config.column_stats : [];
  return columns.map((column) => {
    const fallback = config?.profiling_result?.column_explanations?.[column.name];
    const meaning = typeof column.meaning === "string" && column.meaning.trim()
      ? column.meaning : typeof fallback === "string" && fallback.trim() ? fallback : undefined;
    return { ...column, meaning };
  });
}
export function rawChart(config: any, slide: Slide) {
  const charts = Array.isArray(config?.charts) ? config.charts : [];
  return charts.find((chart: any) => slide.id && String(chart.id ?? chart.chart_id) === slide.id)
    ?? charts.find((chart: any) => chart.title === slide.title);
}
export function fieldsOf(config: any, slide: Slide): string[] {
  const encodings = rawChart(config, slide)?.encodings;
  const fields = encodings
    ? ["x", "y", "theta", "color"].flatMap((key) => (Array.isArray(encodings[key]) ? encodings[key] : [encodings[key]]).map((encoding: any) => encoding?.field))
    : [slide.xField, slide.yField];
  return [...new Set(fields.filter((field): field is string => typeof field === "string" && field.toLowerCase() !== "count" && field !== "__count__"))];
}
export function scopeCaption(config: any) {
  const full = config?.column_stats_scope === "full_file";
  const count = full ? config?.dataset?.total_rows : config?.dataset?.rows_sampled;
  return count == null ? "Statistics scope not supplied" : `Statistics on ${full ? "all" : "a sample of"} ${number(count)} rows`;
}
export function ValuePill({ value, suffix = false }: { value?: Column["predictive_value"]; suffix?: boolean }) {
  return value ? <span className={`ai-pill ai-value-${value}`}>{value}{suffix ? " value" : ""}</span> : null;
}

export function AutoInsightsHeader({ groups, active, view, onDataset, onView, onClose }: {
  groups: InsightGroup[]; active?: InsightGroup; view: InsightView;
  onDataset: (id: string) => void; onView: (view: InsightView) => void; onClose: () => void;
}) {
  const config = active?.config;
  const dataset = config?.dataset;
  const filename = dataset?.filename || active?.name || "Dataset";
  const sheet = dataset?.sheet_name ?? dataset?.sheet;
  const name = sheet && !String(filename).includes(String(sheet)) ? `${filename} · ${sheet}` : filename;
  const domain = config?.profile_overview?.domain;
  return <header className="ai-header ai-surface">
    <div className="ai-context"><h2 id="auto-insights-title">Auto Insights</h2><span className="ai-divider" />
      <div className="ai-dataset">
        {groups.length > 1 ? <div className="ai-dataset-select"><select aria-label="Dataset" value={active?.id ?? ""} onChange={(event) => onDataset(event.target.value)}>{groups.map((group) => <option key={group.id} value={group.id}>{group.config?.dataset?.filename ?? group.name}{group.config?.dataset?.sheet_name && !group.name.includes(group.config.dataset.sheet_name) ? ` · ${group.config.dataset.sheet_name}` : ""}</option>)}</select><ChevronDown size={16} /></div> : <strong title={name}>{name}</strong>}
        <p>{number(dataset?.total_rows ?? dataset?.rows_sampled)} {dataset?.total_rows == null && dataset?.rows_sampled != null ? "sample rows" : "rows"} · {number(Array.isArray(dataset?.columns) ? dataset.columns.length : dataset?.column_count ?? (columnsOf(config).length || null))} columns{domain ? ` · ${domain}` : ""}</p>
      </div>
    </div>
    <div className="ai-segments" role="group" aria-label="Auto Insights views">{([['grid', 'Overview'], ['stack', 'Stack'], ['columns', 'Column statistics']] as const).map(([key, label]) => <Button key={key} color="tertiary" className={`ai-control ai-tab ${view === key ? 'ai-active' : ''}`} aria-pressed={view === key} onPress={() => onView(key)}>{label}</Button>)}</div>
    <Button color="secondary" className="ai-control ai-close" aria-label="Close Auto Insights" iconLeading={X} onPress={onClose} />
  </header>;
}

function chartKind(slide: Slide, config: any) {
  if (slide.chartKind === "histogram") return "Histogram";
  if (slide.type === "bar") return "Bar · comparison";
  if (slide.type === "pie" || slide.type === "donut") return "Pie · share";
  if (slide.type === "scatter") return "Scatter · relationship";
  const field = rawChart(config, slide)?.encodings?.x?.field;
  const dateColumn = columnsOf(config).find((column) => column.name === field && columnType(column) === "Date");
  const years = dateColumn ? slide.data.map((point) => String(point[slide.xKey] ?? "").match(/^\d{4}/)?.[0]).filter((year): year is string => Boolean(year)).sort() : [];
  const range = years.length && years[0] !== years[years.length - 1] ? ` ${years[0]}–${years[years.length - 1]}` : "";
  return `Line · trend${range}`;
}
export function InsightNavigator({ slides, index, config, onSelect }: { slides: Slide[]; index: number; config: any; onSelect: (index: number) => void }) {
  return <nav className="ai-navigator ai-surface" aria-label="Insight navigator"><h3 className="ai-eyebrow">INSIGHTS · {slides.length}</h3><div className="ai-nav-list">{slides.map((slide, position) => <Button color="tertiary" key={`${slide.title}-${position}`} className={`ai-control ai-nav-item ${index === position ? 'ai-selected' : ''}`} aria-current={index === position ? "true" : undefined} onPress={() => onSelect(position)}><span className="ai-nav-number">{position + 1}</span><span className="ai-nav-copy"><strong>{slide.title}</strong><small>{chartKind(slide, config)}</small></span></Button>)}</div></nav>;
}
export function ChartNavigation({ index, total, onSelect }: { index: number; total: number; onSelect: (index: number) => void }) {
  return <><Button color="secondary" className="ai-control ai-arrow ai-previous" aria-label="Previous insight" isDisabled={index === 0} iconLeading={ChevronLeft} onPress={() => onSelect(index - 1)} /><Button color="primary" className="ai-control ai-arrow ai-next" aria-label="Next insight" isDisabled={index >= total - 1} iconLeading={ChevronRight} onPress={() => onSelect(index + 1)} /><div className="ai-progress"><div className="ai-dots">{Array.from({ length: total }, (_, position) => <Button color="tertiary" key={position} className={`ai-control ai-dot ${position === index ? 'ai-dot-active' : ''}`} aria-label={`Show insight ${position + 1}`} aria-current={position === index ? "true" : undefined} onPress={() => onSelect(position)} />)}</div><span>{index + 1} of {total} · use ← → keys</span></div></>;
}

export function ChartStatsRail({ config, slide, onAllColumns }: { config: any; slide: Slide; onAllColumns: () => void }) {
  const fields = fieldsOf(config, slide);
  const columns = columnsOf(config).filter((column) => fields.includes(column.name));
  return <aside className="ai-stats-rail ai-surface"><div className="ai-rail-heading"><h3 className="ai-eyebrow">COLUMNS IN THIS CHART</h3><Button color="link-color" className="ai-all-columns" onPress={onAllColumns}>All columns →</Button></div>
    {columns.map((column) => <section className="ai-column-block" key={column.name}><div className="ai-column-heading"><strong>{column.name}</strong><ValuePill value={column.predictive_value} suffix /></div>{column.meaning && <p className="ai-meaning">{column.meaning}</p>}<p className="ai-column-meta">{columnType(column)} · {percent(column.missing_pct)} missing · {number(column.n_unique)} distinct</p>
      {columnType(column) === "Number" ? <dl className="ai-stat-tiles">{(['Min', 'Median', 'Mean', 'Max'] as const).map((label) => <div key={label}><dt>{label}</dt><dd>{number(column[label.toLowerCase() as 'min' | 'median' | 'mean' | 'max'], true)}</dd></div>)}</dl> : columnType(column) === "Date" ? <p className="ai-date-range">{date(column.first ?? column.min)} → {date(column.last ?? column.max)}<small>{number(column.n_months)} months</small></p> : <div className="ai-top-values">{column.top_values?.slice(0, 3).map((item, index) => <div key={index}><span title={String(item.value)}>{String(item.value)}</span><meter min={0} max={100} value={item.pct ?? 0} aria-label={`${item.value}: ${percent(item.pct)}`} /><span>{percent(item.pct)}</span></div>)}</div>}
    </section>)}
    {!columns.length && <p className="ai-meaning">No column statistics available for this chart.</p>}<footer className="ai-scope">{scopeCaption(config)}</footer>
  </aside>;
}

function MiniSummary({ column, config }: { column: Column; config: any }) {
  const type = columnType(column);
  const histogram = type === "Number" && Array.isArray(config?.charts) ? config.charts.find((chart: any) => (String(chart.type).includes('hist') || chart.encodings?.x?.bin) && chart.encodings?.x?.field === column.name) : null;
  const values = type === "Category" ? column.top_values?.slice(0, 6).map((item) => item.pct ?? 0) : histogram?.derived_data?.points?.map((point: any) => point.count ?? point.y ?? 0);
  let text: string;
  if (type === "Category") {
    const top = column.top_values ?? [];
    text = top.slice(0, 3).map((item) => `${String(item.value)} ${percent(item.pct)}`).join(' · ') + (top.length > 3 ? ` · +${top.length - 3} more` : '');
  } else if (type === "Date") text = `${date(column.first ?? column.min)} → ${date(column.last ?? column.max)} · ${number(column.n_months)} months`;
  else text = `${number(column.min, true)} → ${number(column.max, true)} · median ${number(column.median, true)}`;
  const maximum = values?.length ? Math.max(...values.map(Number), 1) : 1;
  const median = Number(column.median), minimum = Number(column.min), max = Number(column.max);
  const tick = max > minimum && Number.isFinite(median) ? Math.max(0, Math.min(120, (median - minimum) / (max - minimum) * 120)) : null;
  return <div className="ai-summary"><svg className="ai-mini" width="120" height="26" viewBox="0 0 120 26" role="img" aria-label={text || "No summary available"}>{values?.length ? values.map((value: number, index: number) => { const width = 120 / values.length; const height = Math.max(1, Number(value) / maximum * 24); return <rect key={index} x={index * width} y={26 - height} width={Math.max(1, width - 2)} height={height} rx="1" />; }) : <><rect x="0" y="10" width="120" height="6" rx="3" />{type === "Number" && tick != null && <line x1={tick} x2={tick} y1="4" y2="22" />}</>}</svg><span>{text || "—"}</span></div>;
}

export function ColumnStatisticsTab({ config, slides, onChart }: { config: any; slides: Slide[]; onChart: (index: number) => void }) {
  const [showIds, setShowIds] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [selected, setSelected] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: 'missing_pct' | 'n_unique' | 'predictive_value'; asc: boolean } | null>(null);
  const source = columnsOf(config);
  const hidden = source.filter((column) => column.role?.toLowerCase() === 'identifier').length;
  const visible = source.filter((column) => showIds || column.role?.toLowerCase() !== 'identifier');
  const counts = [visible.length, ...['Number', 'Category', 'Date'].map((type) => visible.filter((column) => columnType(column) === type).length)];
  const rows = visible.filter((column) => (filter === 'All' || columnType(column) === filter) && `${column.name} ${column.meaning ?? ''}`.toLowerCase().includes(search.toLowerCase()));
  if (sort) rows.sort((a, b) => { const rank = { high: 3, medium: 2, low: 1 }; const value = (column: Column) => sort.key === 'predictive_value' ? rank[column.predictive_value ?? 'low'] : Number(column[sort.key] ?? -1); return (value(a) - value(b)) * (sort.asc ? 1 : -1); });
  const onSort = (key: 'missing_pct' | 'n_unique' | 'predictive_value') => setSort((prev) => ({ key, asc: prev?.key === key ? !prev.asc : true }));
  const choose = (column: Column) => { setSelected(column.name); const index = slides.findIndex((slide) => fieldsOf(config, slide).includes(column.name)); if (index >= 0) onChart(index); };
  return <section className="ai-columns-tab ai-surface"><div className="ai-toolbar"><div className="ai-filters">{(['All', 'Number', 'Category', 'Date'] as const).map((type, index) => <Button color="secondary" key={type} className={`ai-control ai-filter ${filter === type ? 'ai-active' : ''}`} aria-pressed={filter === type} onPress={() => setFilter(type)}>{['All', 'Numbers', 'Categories', 'Dates'][index]} {counts[index]}</Button>)}</div><div className="ai-search-tools"><Checkbox label="Show ID columns" isSelected={showIds} onChange={setShowIds} /><label className="ai-search"><Search size={16} /><input aria-label="Search columns" placeholder="Search columns" value={search} onChange={(event) => setSearch(event.target.value)} /></label></div></div>
    <div className="ai-table-scroll"><table className="ai-column-table"><thead><tr><th>Column</th><th>Type</th>{([['Predictive value', 'predictive_value'], ['Missing', 'missing_pct'], ['Distinct', 'n_unique']] as const).map(([label, key]) => <th key={key} aria-sort={sort?.key === key ? sort.asc ? 'ascending' : 'descending' : 'none'}><Button color="tertiary" className="ai-control ai-sort" onPress={() => onSort(key)}>{label}{sort?.key === key && (sort.asc ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}</Button></th>)}<th>Summary</th></tr></thead><tbody>{rows.map((column) => <tr key={column.name} className={selected === column.name ? 'ai-row-selected' : ''} onClick={() => choose(column)}><td><Button color="tertiary" className="ai-control ai-column-name" onPress={() => choose(column)}>{column.name}</Button>{column.meaning && <p title={column.meaning}>{column.meaning}</p>}</td><td><span className={`ai-pill ai-type-${columnType(column).toLowerCase()}`}>{columnType(column)}</span></td><td><ValuePill value={column.predictive_value} /></td><td>{percent(column.missing_pct)}</td><td>{number(column.n_unique)}</td><td><MiniSummary column={column} config={config} /></td></tr>)}</tbody></table>{!rows.length && <p className="ai-empty">{search ? `No columns match "${search}"` : 'No column statistics available for this dataset'}</p>}</div>
    <footer className="ai-table-footer"><span>{scopeCaption(config)}{!showIds && hidden > 0 ? ` · ${hidden} ID columns hidden` : ''}</span><span>Click a column to see the charts that use it</span></footer>
  </section>;
}