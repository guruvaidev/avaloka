import { useQuery } from "@tanstack/react-query";
import {
  fetchAnalysisDashboard,
  projectDashboardFromCache,
  projectDashboardQueryKey,
  tabsFromVizConfig,
  type DashboardTab,
  type DashboardWidget,
} from "@/lib/api/project-dashboard";

export function useAnalysisDashboard(projectId: string | null, analysisId: string | null, vizConfig?: unknown | null) {
  const query = useQuery({
    queryKey: projectDashboardQueryKey(analysisId, vizConfig),
    enabled: !!analysisId,
    queryFn: () => fetchAnalysisDashboard(projectId, analysisId),
  });

  const vizTabs = tabsFromVizConfig(vizConfig);
  const previewTabs: DashboardTab[] = vizTabs.map((t, tabIndex) => ({
    id: t.id ?? `preview-tab-${tabIndex}`,
    dashboardId: "",
    title: t.title,
    tabIndex: t.tab_index ?? tabIndex,
  }));

  const dashboard = projectDashboardFromCache(query.data);
  const tabs = dashboard?.tabs?.length ? dashboard.tabs : previewTabs.length ? previewTabs : [];
  const widgets = dashboard?.widgets ?? [];

  return { ...query, tabs, widgets };
}

export function widgetsForTab(widgets: DashboardWidget[], tabId: string) {
  return widgets.filter((w) => w.placed && w.tabId === tabId);
}
