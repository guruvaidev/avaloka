/**
 * Project dashboard persistence — single module for tabs, widgets, and pin/save flows.
 *
 * ## Storage (live Supabase)
 * All dashboard state is stored on **`analyses.viz_config`** only:
 * - `dashboard_tabs[]` — tab names/ids from the LLM (`tabsFromVizConfig`)
 * - `dashboard_widgets[]` — pinned charts (`addInsightToDashboard`, drag-and-drop)
 *
 * Supabase calls in this file:
 * - `SELECT viz_config FROM analyses WHERE id = ?` (load)
 * - `UPDATE analyses SET viz_config = ? WHERE id = ?` (save)
 *
 * ## Not used (legacy Lovable schema — tables missing on live DB)
 * - `analysis_dashboards`, `dashboard_tabs`, `dashboard_graphs`
 * Do not re-introduce those tables; use `dashboard_widgets` in viz_config instead.
 *
 * ## Exported API (parity with Lovable project-dashboard.ts)
 * | Export | Purpose |
 * |--------|---------|
 * | `projectDashboardKey` | React Query key |
 * | `projectDashboardQueryKey` | React Query key (+ viz hint) |
 * | `tabsFromVizConfig` | Parse LLM tab list |
 * | `chartOptionsFromVizConfig` | Chart picker labels |
 * | `insightPayloadFromVizConfig` | Build pin payload from chart index |
 * | `graphKeyForChartIndex` | Stable chart id for dedupe / insights panel |
 * | `pinnedGraphKeysFromVizConfig` | Charts already on dashboard |
 * | `dashboardFromVizConfig` | Build tabs+widgets view model from viz_config |
 * | `fetchAnalysisDashboard` | Load dashboard (DB → localStorage fallback) |
 * | `ensureDashboardTabsFromVizConfig` | Ensure tabs exist (viz_config or local) |
 * | `seedDashboardFromVizConfig` | Auto-pin all charts on upload |
 * | `addInsightToDashboard` | Pin one chart to a slot |
 * | `dashboardWidgetsForHistory` | History panel entries |
 */
import type { QueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export type DashboardTab = {
  id: string;
  dashboardId: string;
  title: string;
  tabIndex: number;
};

export type DashboardWidget = {
  id: string;
  tabId: string;
  analysisId: string;
  title: string;
  chartType: string;
  config: Record<string, unknown>;
  position: number;
  placed: boolean;
  graphKey: string;
};

export type ProjectDashboard = {
  tabs: DashboardTab[];
  widgets: DashboardWidget[];
};

const EMPTY_DASHBOARD: ProjectDashboard = { tabs: [], widgets: [] };

export function isProjectDashboard(value: unknown): value is ProjectDashboard {
  if (!value || typeof value !== "object") return false;
  const candidate = value as ProjectDashboard;
  return Array.isArray(candidate.tabs) && Array.isArray(candidate.widgets);
}

export type AddInsightToDashboardResult = {
  dashboard: ProjectDashboard;
  vizConfig: Record<string, unknown>;
};

/** Accept legacy Lovable return (`ProjectDashboard`) or `{ dashboard, vizConfig }`. */
export function normalizeAddInsightResult(
  value: ProjectDashboard | AddInsightToDashboardResult,
): AddInsightToDashboardResult {
  if (isProjectDashboard(value)) {
    return { dashboard: value, vizConfig: {} };
  }
  const dashboard = isProjectDashboard(value.dashboard) ? value.dashboard : EMPTY_DASHBOARD;
  const vizConfig =
    value.vizConfig && typeof value.vizConfig === "object" && !Array.isArray(value.vizConfig)
      ? (value.vizConfig as Record<string, unknown>)
      : {};
  return { dashboard, vizConfig };
}

/** React Query cache may hold `ProjectDashboard` or a mistaken `{ dashboard, vizConfig }` wrapper. */
export function projectDashboardFromCache(value: unknown): ProjectDashboard | undefined {
  if (isProjectDashboard(value)) return value;
  if (value && typeof value === "object") {
    const wrapped = (value as AddInsightToDashboardResult).dashboard;
    if (isProjectDashboard(wrapped)) return wrapped;
  }
  return undefined;
}

export function setProjectDashboardCache(
  qc: QueryClient,
  analysisId: string,
  vizConfig: unknown | null | undefined,
  dashboard: ProjectDashboard,
) {
  const data = isProjectDashboard(dashboard) ? dashboard : EMPTY_DASHBOARD;
  qc.setQueryData(projectDashboardQueryKey(analysisId, vizConfig), data);
  qc.setQueryData(projectDashboardQueryKey(analysisId, null), data);
}

export type DashboardInsightPayload = {
  title: string;
  chartType: string;
  config: Record<string, unknown>;
  graphKey: string;
};

export const projectDashboardKey = (analysisId: string | null) => ["project-dashboard", analysisId] as const;

const LOCAL_KEY = "analysis-dashboard:v1";

type LocalStore = Record<string, ProjectDashboard>;

type StoredDashboardWidget = {
  tab_index: number;
  position: number;
  graph_key: string;
  title: string;
  chart_type: string;
  placed: boolean;
  config?: Record<string, unknown>;
  created_at?: string;
};

function storageKey(projectId: string, analysisId: string) {
  return `${projectId}:${analysisId}`;
}

function readLocal(): LocalStore {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(LOCAL_KEY) || "{}") as LocalStore;
  } catch {
    return {};
  }
}

function writeLocal(store: LocalStore) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LOCAL_KEY, JSON.stringify(store));
  } catch {
    /* ignore */
  }
}

function parseVizRecord(vizConfig: unknown): Record<string, unknown> {
  if (!vizConfig) return {};
  if (typeof vizConfig === "string") {
    try {
      const parsed = JSON.parse(vizConfig) as unknown;
      return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {};
    } catch {
      return {};
    }
  }
  return typeof vizConfig === "object" ? (vizConfig as Record<string, unknown>) : {};
}

export function analysisDashboardTabId(analysisId: string): string {
  return `analysis-${analysisId}`;
}

/** One dashboard tab per analysis — title is the analysis name, not LLM dashboard_tabs. */
export function analysisDashboardTab(analysisId: string, title: string): DashboardTab {
  return {
    id: analysisDashboardTabId(analysisId),
    dashboardId: analysisId,
    title: title.trim() || "Analysis",
    tabIndex: 0,
  };
}

export type VizDashboardTab = {
  id?: string;
  title: string;
  tab_index?: number;
  tabIndex?: number;
  chart_ids?: string[];
};

export function tabsFromVizConfig(vizConfig: unknown): VizDashboardTab[] {
  if (!vizConfig || typeof vizConfig !== "object") return [];
  const raw = (vizConfig as { dashboard_tabs?: unknown }).dashboard_tabs;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(
      (t): t is Record<string, unknown> =>
        t != null && typeof t === "object" && typeof (t as { title?: unknown }).title === "string",
    )
    .map((t, i) => ({
      id: t.id != null ? String(t.id) : undefined,
      title: String(t.title),
      tab_index: typeof t.tab_index === "number" ? t.tab_index : typeof t.tabIndex === "number" ? t.tabIndex : i,
      chart_ids: Array.isArray(t.chart_ids) ? t.chart_ids.map(String) : undefined,
    }))
    .sort((a, b) => (a.tab_index ?? 0) - (b.tab_index ?? 0));
}

type VizChart = Record<string, unknown>;

function chartsFromVizConfig(vizConfig: unknown): VizChart[] {
  if (!vizConfig || typeof vizConfig !== "object") return [];
  const charts = (vizConfig as { charts?: unknown }).charts;
  return Array.isArray(charts) ? (charts as VizChart[]) : [];
}

function storedWidgetsFromViz(vizConfig: unknown): StoredDashboardWidget[] {
  const record = parseVizRecord(vizConfig);
  const raw = record.dashboard_widgets;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => item != null && typeof item === "object")
    .map((item) => ({
      tab_index: typeof item.tab_index === "number" ? item.tab_index : 0,
      position: typeof item.position === "number" ? item.position : 0,
      graph_key: String(item.graph_key ?? item.graphKey ?? ""),
      title: String(item.title ?? "Chart"),
      chart_type: String(item.chart_type ?? item.chartType ?? "bar"),
      placed: item.placed !== false,
      config: item.config && typeof item.config === "object" ? (item.config as Record<string, unknown>) : undefined,
      created_at: typeof item.created_at === "string" ? item.created_at : undefined,
    }))
    .filter((w) => w.graph_key && w.placed);
}

function dashboardTabsFromViz(analysisId: string, vizConfig: unknown): DashboardTab[] {
  return tabsFromVizConfig(vizConfig).map((t, i) => {
    const tabIndex = t.tab_index ?? i;
    return {
      id: t.id ?? `preview-tab-${tabIndex}`,
      dashboardId: analysisId,
      title: t.title,
      tabIndex,
    };
  });
}

function widgetsFromVizConfig(analysisId: string, vizConfig: unknown, _tabs: DashboardTab[]): DashboardWidget[] {
  const analysisTabId = analysisDashboardTabId(analysisId);
  return storedWidgetsFromViz(vizConfig).map((w) => ({
    id: `viz-widget-${w.tab_index}-${w.position}-${w.graph_key}`,
    tabId: analysisTabId,
    analysisId,
    title: w.title,
    chartType: w.chart_type,
    config: w.config ?? {},
    position: w.position,
    placed: w.placed,
    graphKey: w.graph_key,
  }));
}

export function dashboardFromVizConfig(analysisId: string, vizConfig: unknown): ProjectDashboard {
  const tabs = dashboardTabsFromViz(analysisId, vizConfig);
  return {
    tabs,
    widgets: widgetsFromVizConfig(analysisId, vizConfig, tabs),
  };
}

function graphKeyForChart(chart: VizChart, index: number): string {
  return chart.id != null ? String(chart.id) : `viz:${chart.rank ?? index}`;
}

export function graphKeyForChartIndex(vizConfig: unknown, chartIndex: number): string | null {
  const charts = chartsFromVizConfig(vizConfig);
  const chart = charts[chartIndex];
  if (!chart) return null;
  return graphKeyForChart(chart, chartIndex);
}

export function pinnedGraphKeysFromVizConfig(vizConfig: unknown): Set<string> {
  const keys = new Set(storedWidgetsFromViz(vizConfig).map((w) => w.graph_key));
  const record = parseVizRecord(vizConfig);
  const legacy = record.pinned_graphs;
  if (Array.isArray(legacy)) {
    for (const k of legacy) if (k != null) keys.add(String(k));
  }
  return keys;
}

function chartToInsightPayload(
  chart: VizChart,
  vizConfig: Record<string, unknown>,
  chartIndex: number,
): DashboardInsightPayload {
  const chartId = graphKeyForChart(chart, chartIndex);
  return {
    title: String(chart.title ?? "Chart"),
    chartType: String(chart.type ?? "bar").toLowerCase(),
    config: { ...vizConfig, charts: [chart], focusChartId: chartId },
    graphKey: chartId,
  };
}

export function chartOptionsFromVizConfig(vizConfig: unknown): { index: number; title: string }[] {
  return chartsFromVizConfig(vizConfig).map((c, i) => ({
    index: i,
    title: String(c.title ?? c.name ?? `Chart ${i + 1}`),
  }));
}

export function insightPayloadFromVizConfig(vizConfig: unknown, chartIndex = 0): DashboardInsightPayload | null {
  const charts = chartsFromVizConfig(vizConfig);
  const chart = charts[chartIndex];
  if (!chart) return null;
  const vizRecord = parseVizRecord(vizConfig);
  return chartToInsightPayload(chart, vizRecord, chartIndex);
}

async function loadVizConfig(analysisId: string): Promise<unknown | null> {
  const { data, error } = await supabase.from("analyses").select("viz_config").eq("id", analysisId).maybeSingle();
  if (error) {
    console.warn("[project-dashboard] Failed to load viz_config", error);
    return null;
  }
  return data?.viz_config ?? null;
}

async function saveVizConfig(analysisId: string, vizConfig: Record<string, unknown>): Promise<void> {
  const { error } = await supabase.from("analyses").update({ viz_config: vizConfig }).eq("id", analysisId);
  if (error) throw error;
}

function saveLocalDashboard(projectId: string, analysisId: string, dashboard: ProjectDashboard) {
  const store = readLocal();
  store[storageKey(projectId, analysisId)] = dashboard;
  writeLocal(store);
}

function loadLocalDashboard(projectId: string, analysisId: string): ProjectDashboard | null {
  return readLocal()[storageKey(projectId, analysisId)] ?? null;
}

type TabSpec = { title: string; tab_index: number; id?: string };

/** Offline / Supabase-failure fallback — mirrors Lovable `ensureLocalDashboard`. */
function ensureLocalDashboard(projectId: string, analysisId: string, tabSpecs?: TabSpec[]): ProjectDashboard {
  const key = storageKey(projectId, analysisId);
  const store = readLocal();
  if (store[key]?.tabs.length) return store[key];

  const specs = tabSpecs?.length ? tabSpecs : [{ title: "Dashboard", tab_index: 0 }];
  const tabs: DashboardTab[] = specs.map(({ title, tab_index, id }) => ({
    id: id ? `local-${id}` : `local-tab-${tab_index}`,
    dashboardId: `local-dash-${analysisId}`,
    title,
    tabIndex: tab_index,
  }));
  const dashboard: ProjectDashboard = { tabs, widgets: [] };
  store[key] = dashboard;
  writeLocal(store);
  return dashboard;
}

function tabSpecsFromViz(vizTabs: VizDashboardTab[]): TabSpec[] {
  return vizTabs.map((t, i) => ({
    title: t.title,
    tab_index: t.tab_index ?? i,
    id: t.id,
  }));
}

function hasPlacedWidgets(dashboard: ProjectDashboard | null | undefined): boolean {
  return dashboard?.widgets.some((w) => w.placed) ?? false;
}

function seedWidgetsLocally(
  projectId: string,
  analysisId: string,
  record: Record<string, unknown>,
  vizTabs: VizDashboardTab[],
  charts: VizChart[],
): ProjectDashboard {
  const tabSpecs = tabSpecsFromViz(vizTabs);
  const dashboard = ensureLocalDashboard(projectId, analysisId, tabSpecs);
  if (hasPlacedWidgets(dashboard)) return dashboard;

  const tabs = dashboard.tabs;
  const tabIdMap = buildTabIndexMap(vizTabs);
  const positionsByTab = new Map<string, number>();
  const widgets: DashboardWidget[] = [...dashboard.widgets];

  for (let i = 0; i < charts.length; i++) {
    const chart = charts[i];
    const tabIndex = resolveTabIndexForChart(chart, vizTabs, tabIdMap, i);
    const tab = tabs.find((t) => t.tabIndex === tabIndex) ?? tabs[tabIndex] ?? tabs[0];
    if (!tab) continue;

    const pos = positionsByTab.get(tab.id) ?? 0;
    positionsByTab.set(tab.id, pos + 1);
    const insight = chartToInsightPayload(chart, record, i);
    widgets.push({
      id: `local-widget-${insight.graphKey}`,
      tabId: tab.id,
      analysisId,
      title: insight.title,
      chartType: insight.chartType,
      config: insight.config,
      position: pos,
      placed: true,
      graphKey: insight.graphKey,
    });
  }

  const updated = { ...dashboard, widgets };
  saveLocalDashboard(projectId, analysisId, updated);
  return updated;
}

function buildTabIndexMap(vizTabs: VizDashboardTab[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const vizTab of vizTabs) {
    const idx = vizTab.tab_index ?? 0;
    if (vizTab.id) map.set(vizTab.id, idx);
  }
  return map;
}

export function chartIndicesForDashboardTab(vizConfig: unknown, tab: Pick<DashboardTab, "id" | "tabIndex">): number[] {
  const vizTabs = tabsFromVizConfig(vizConfig);
  const charts = chartsFromVizConfig(vizConfig);
  if (!charts.length) return [];
  if (!vizTabs.length) return charts.map((_, i) => i);

  const tabIdMap = buildTabIndexMap(vizTabs);
  return charts.reduce<number[]>((indices, chart, i) => {
    const tabIndex = resolveTabIndexForChart(chart, vizTabs, tabIdMap, i);
    const vizTab = vizTabs[tabIndex];
    const matchesIndex = tabIndex === tab.tabIndex;
    const matchesId = vizTab?.id != null && vizTab.id === tab.id;
    if (matchesIndex || matchesId) indices.push(i);
    return indices;
  }, []);
}

function resolveTabIndexForChart(
  chart: VizChart,
  vizTabs: VizDashboardTab[],
  tabIdMap: Map<string, number>,
  fallbackIndex: number,
): number {
  const chartTabId = chart.tab_id ?? chart.tabId;
  if (chartTabId != null) {
    const mapped = tabIdMap.get(String(chartTabId));
    if (mapped != null) return mapped;
  }

  const chartId = chart.id != null ? String(chart.id) : null;
  if (chartId) {
    for (const vizTab of vizTabs) {
      if (vizTab.chart_ids?.includes(chartId) && vizTab.id) {
        const mapped = tabIdMap.get(vizTab.id);
        if (mapped != null) return mapped;
        return vizTab.tab_index ?? 0;
      }
    }
  }

  if (typeof chart.tab_index === "number") return chart.tab_index;
  if (typeof chart.tabIndex === "number") return chart.tabIndex;
  return fallbackIndex % Math.max(vizTabs.length, 1);
}

function resolveTab(dashboard: ProjectDashboard, tabValue: string): DashboardTab | undefined {
  return (
    dashboard.tabs.find((t) => t.id === tabValue) ??
    dashboard.tabs.find((t) => t.title.toLowerCase() === tabValue.toLowerCase()) ??
    dashboard.tabs.find((t) => t.title.toLowerCase().includes(tabValue.toLowerCase()))
  );
}

function resolveTabForSave(
  tabValue: string,
  tabs: DashboardTab[],
  vizTabs: VizDashboardTab[] = [],
): DashboardTab | undefined {
  const direct = resolveTab({ tabs, widgets: [] }, tabValue);
  if (direct) return direct;

  if (tabValue.startsWith("analysis-")) {
    const analysisId = tabValue.slice("analysis-".length);
    return (
      tabs.find((t) => t.id === tabValue) ?? {
        id: tabValue,
        dashboardId: analysisId,
        title: "",
        tabIndex: 0,
      }
    );
  }

  if (tabValue.startsWith("preview-tab-")) {
    const idx = Number.parseInt(tabValue.replace("preview-tab-", ""), 10);
    if (!Number.isNaN(idx)) {
      return tabs.find((t) => t.tabIndex === idx) ?? tabs[idx] ?? tabs[0];
    }
  }

  const vizTab = vizTabs.find((t) => t.id === tabValue);
  if (vizTab) {
    const idx = vizTab.tab_index ?? 0;
    return tabs.find((t) => t.tabIndex === idx) ?? tabs[idx] ?? tabs[0];
  }

  return tabs[0];
}

function computePosition(position: string, slotIndex?: number): number {
  if (position === "top") return 0;
  if (position === "after-recent") return 1;
  if (position === "custom" && slotIndex != null) return slotIndex;
  return 0;
}

export function projectDashboardQueryKey(analysisId: string | null, vizConfig?: unknown | null) {
  return [...projectDashboardKey(analysisId), vizConfig != null ? "viz" : "no-viz"] as const;
}

/** Load dashboard tabs + pinned charts from analyses.viz_config (no extra tables). */
export async function fetchAnalysisDashboard(
  projectId: string | null,
  analysisId: string | null,
  options?: { vizConfig?: unknown | null; autoSeed?: boolean },
): Promise<ProjectDashboard> {
  if (!analysisId) return { tabs: [], widgets: [] };

  const loaded = await loadVizConfig(analysisId);
  const vizConfig = loaded ?? options?.vizConfig ?? null;
  const record = parseVizRecord(vizConfig);

  if (!loaded && projectId) {
    const cached = loadLocalDashboard(projectId, analysisId);
    if (cached) return cached;
  }

  if (options?.autoSeed === true) {
    const seeded = await seedDashboardFromVizConfig({
      projectId,
      analysisId,
      vizConfig: record,
    });
    if (seeded) {
      if (projectId) saveLocalDashboard(projectId, analysisId, seeded);
      return seeded;
    }
  }

  let dashboard = dashboardFromVizConfig(analysisId, record);
  const vizTabs = tabsFromVizConfig(vizConfig);
  if (vizTabs.length && !dashboard.tabs.length) {
    const ensured = await ensureDashboardTabsFromVizConfig({
      projectId,
      analysisId,
      vizConfig: record,
    });
    if (ensured) dashboard = ensured;
  }

  if (projectId) saveLocalDashboard(projectId, analysisId, dashboard);
  return dashboard;
}

/** Ensure tabs exist in viz_config (or localStorage when DB write fails). */
export async function ensureDashboardTabsFromVizConfig(input: {
  projectId: string | null;
  analysisId: string;
  vizConfig: unknown;
}): Promise<ProjectDashboard | null> {
  const vizTabs = tabsFromVizConfig(input.vizConfig);
  const tabSpecs = tabSpecsFromViz(vizTabs);

  if (!vizTabs.length) {
    if (!input.projectId) return null;
    return ensureLocalDashboard(input.projectId, input.analysisId);
  }

  const record = parseVizRecord(input.vizConfig);
  if (!tabsFromVizConfig(record).length) {
    const nextViz = {
      ...record,
      dashboard_tabs: tabSpecs.map((t) => ({
        id: t.id,
        title: t.title,
        tab_index: t.tab_index,
      })),
    };
    try {
      await saveVizConfig(input.analysisId, nextViz);
      const saved = dashboardFromVizConfig(input.analysisId, nextViz);
      if (input.projectId) saveLocalDashboard(input.projectId, input.analysisId, saved);
      return saved;
    } catch (err) {
      console.warn("[project-dashboard] tab ensure save failed, using localStorage", err);
      if (!input.projectId) return dashboardFromVizConfig(input.analysisId, input.vizConfig);
      return ensureLocalDashboard(input.projectId, input.analysisId, tabSpecs);
    }
  }

  const dashboard = dashboardFromVizConfig(input.analysisId, record);
  if (input.projectId) saveLocalDashboard(input.projectId, input.analysisId, dashboard);
  return dashboard;
}

/** Optional: auto-pin all charts into viz_config.dashboard_widgets (used on upload when enabled). */
export async function seedDashboardFromVizConfig(input: {
  projectId: string | null;
  analysisId: string;
  vizConfig: unknown;
}): Promise<ProjectDashboard | null> {
  const vizTabs = tabsFromVizConfig(input.vizConfig);
  const charts = chartsFromVizConfig(input.vizConfig);
  if (!vizTabs.length || !charts.length) return null;

  const record = parseVizRecord(input.vizConfig);
  if (storedWidgetsFromViz(record).length) {
    return dashboardFromVizConfig(input.analysisId, record);
  }

  const tabs = dashboardTabsFromViz(input.analysisId, record);
  const widgets: StoredDashboardWidget[] = [];
  const now = new Date().toISOString();
  const tabIdMap = buildTabIndexMap(vizTabs);

  for (let i = 0; i < charts.length; i++) {
    const chart = charts[i];
    const tabIndex = resolveTabIndexForChart(chart, vizTabs, tabIdMap, i);
    const tab = tabs.find((t) => t.tabIndex === tabIndex) ?? tabs[tabIndex] ?? tabs[0];
    if (!tab) continue;

    const positionsInTab = widgets.filter((w) => w.tab_index === tab.tabIndex).length;
    const insight = chartToInsightPayload(chart, record, i);
    widgets.push({
      tab_index: tab.tabIndex,
      position: positionsInTab,
      graph_key: insight.graphKey,
      title: insight.title,
      chart_type: insight.chartType,
      placed: true,
      config: insight.config,
      created_at: now,
    });
  }

  const nextViz = { ...record, dashboard_widgets: widgets };
  try {
    await saveVizConfig(input.analysisId, nextViz);
  } catch (err) {
    console.warn("[project-dashboard] Supabase seed failed, using localStorage", err);
    if (!input.projectId) return null;
    return seedWidgetsLocally(input.projectId, input.analysisId, record, vizTabs, charts);
  }

  const dashboard = dashboardFromVizConfig(input.analysisId, nextViz);
  if (input.projectId) saveLocalDashboard(input.projectId, input.analysisId, dashboard);
  return dashboard;
}

/** Pin a chart to a dashboard slot — saved in analyses.viz_config.dashboard_widgets. */
export async function addInsightToDashboard(input: {
  projectId: string;
  analysisId: string;
  tab: string;
  position: string;
  slotIndex?: number;
  insight: DashboardInsightPayload;
  vizConfig?: unknown | null;
}): Promise<{ dashboard: ProjectDashboard; vizConfig: Record<string, unknown> }> {
  const pos = computePosition(input.position, input.slotIndex);
  const loadedViz = await loadVizConfig(input.analysisId);
  const record = parseVizRecord(loadedViz ?? input.vizConfig);
  const vizTabs = tabsFromVizConfig(record);
  let tabs = dashboardTabsFromViz(input.analysisId, record);
  if (!tabs.length && input.projectId) {
    tabs = ensureLocalDashboard(input.projectId, input.analysisId, tabSpecsFromViz(vizTabs)).tabs;
  }
  const tab = resolveTabForSave(input.tab, tabs, vizTabs);
  if (!tab) throw new Error("No dashboard tab available");

  const tabIndex = tab.tabIndex;
  const graphKey = input.insight.graphKey || `widget:${Date.now()}`;
  const now = new Date().toISOString();

  const nextStored = storedWidgetsFromViz(record)
    .filter((w) => !(w.tab_index === tabIndex && w.position === pos))
    .concat([
      {
        tab_index: tabIndex,
        position: pos,
        graph_key: graphKey,
        title: input.insight.title,
        chart_type: input.insight.chartType,
        placed: true,
        config: input.insight.config,
        created_at: now,
      },
    ])
    .sort((a, b) => a.tab_index - b.tab_index || a.position - b.position);

  const nextViz = { ...record, dashboard_widgets: nextStored };
  try {
    await saveVizConfig(input.analysisId, nextViz);
  } catch (err) {
    console.warn("[project-dashboard] viz_config save failed, caching locally", err);
    const updated = dashboardFromVizConfig(input.analysisId, nextViz);
    saveLocalDashboard(input.projectId, input.analysisId, updated);
    return { dashboard: updated, vizConfig: nextViz };
  }

  const updated = dashboardFromVizConfig(input.analysisId, nextViz);
  saveLocalDashboard(input.projectId, input.analysisId, updated);
  return { dashboard: updated, vizConfig: nextViz };
}

/** Remove a pinned chart from a dashboard slot. */
export async function removeInsightFromDashboard(input: {
  projectId: string;
  analysisId: string;
  tab: string;
  slotIndex: number;
  vizConfig?: unknown | null;
}): Promise<{ dashboard: ProjectDashboard; vizConfig: Record<string, unknown> }> {
  const loadedViz = await loadVizConfig(input.analysisId);
  const record = parseVizRecord(loadedViz ?? input.vizConfig);
  const vizTabs = tabsFromVizConfig(record);
  let tabs = dashboardTabsFromViz(input.analysisId, record);
  if (!tabs.length && input.projectId) {
    tabs = ensureLocalDashboard(input.projectId, input.analysisId, tabSpecsFromViz(vizTabs)).tabs;
  }
  const tab = resolveTabForSave(input.tab, tabs, vizTabs);
  if (!tab) throw new Error("No dashboard tab available");

  const tabIndex = tab.tabIndex;
  const nextStored = storedWidgetsFromViz(record).filter(
    (w) => !(w.tab_index === tabIndex && w.position === input.slotIndex),
  );
  const nextViz = { ...record, dashboard_widgets: nextStored };
  try {
    await saveVizConfig(input.analysisId, nextViz);
  } catch (err) {
    console.warn("[project-dashboard] viz_config save failed, caching locally", err);
    const updated = dashboardFromVizConfig(input.analysisId, nextViz);
    saveLocalDashboard(input.projectId, input.analysisId, updated);
    return { dashboard: updated, vizConfig: nextViz };
  }

  const updated = dashboardFromVizConfig(input.analysisId, nextViz);
  saveLocalDashboard(input.projectId, input.analysisId, updated);
  return { dashboard: updated, vizConfig: nextViz };
}

/** For history: pinned charts stored in viz_config. */
export function dashboardWidgetsForHistory(vizConfig: unknown): Array<{
  id: string;
  title: string;
  config: unknown;
  created_at: string;
}> {
  return storedWidgetsFromViz(vizConfig).map((w) => ({
    id: `viz-widget-${w.tab_index}-${w.position}-${w.graph_key}`,
    title: w.title,
    config: w.config ?? {},
    created_at: w.created_at ?? new Date().toISOString(),
  }));
}
