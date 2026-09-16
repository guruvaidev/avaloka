import type { DragEvent } from "react";

export const DASHBOARD_INSIGHT_DRAG_TYPE = "application/vnd.avaloka.dashboard-insight";

export type DashboardInsightDragData = {
  chartIndex: number;
  analysisId?: string;
};

type InsightDragEvent = DragEvent<Element>;

export function setInsightDragData(event: InsightDragEvent, chartIndex: number, analysisId?: string) {
  event.dataTransfer.setData(
    DASHBOARD_INSIGHT_DRAG_TYPE,
    JSON.stringify({ chartIndex, analysisId } satisfies DashboardInsightDragData),
  );
  event.dataTransfer.effectAllowed = "move";
}

export function readInsightDragData(event: InsightDragEvent): DashboardInsightDragData | null {
  const raw = event.dataTransfer.getData(DASHBOARD_INSIGHT_DRAG_TYPE);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as DashboardInsightDragData;
    if (typeof parsed.chartIndex !== "number" || parsed.chartIndex < 0) return null;
    return parsed;
  } catch {
    return null;
  }
}
