import { useResourceAccess } from "@/lib/use-resource-access";
import { useState, type DragEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { X, File02, LayoutLeft, Expand06 } from "@untitledui/icons";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { cx } from "@/lib/utils/cx";
import { Button } from "@/components/base/buttons/button";
import { DASHBOARD_SLOT_LAYOUT, dashboardSlotColClass } from "@/lib/dashboard-layout";
import { readInsightDragData } from "@/lib/dashboard-drag";
import {
  addInsightToDashboard,
  fetchAnalysisDashboard,
  insightPayloadFromVizConfig,
  normalizeAddInsightResult,
  projectDashboardQueryKey,
  removeInsightFromDashboard,
  setProjectDashboardCache,
} from "@/lib/api/project-dashboard";
import { createReport } from "@/lib/reports";
import { toast } from "sonner";
import { DashboardEmptySlot } from "./DashboardEmptySlot";
import { DashboardMiniChart } from "./dashboardMiniChart";
import { widgetsForTab } from "./useAnalysisDashboard";
import { useDashboardPerms } from "@/lib/use-user-mgmt-perms";
import { useUpgradeGate } from "@/components/dashboard/UpgradeGate";


type Props = {
  projectId: string | null;
  analysisId: string | null;
  activeTabId: string;
  vizConfig?: unknown | null;
  resolveVizConfig?: (analysisId: string) => unknown | null | undefined;
  samples?: Record<string, unknown>[] | null;
  className?: string;
  onVizConfigUpdated?: (vizConfig: Record<string, unknown>, analysisId?: string) => void;
  insightsOpen?: boolean;
  onToggleInsights?: () => void;
};

export function ProjectDashboardView({
  projectId,
  analysisId,
  activeTabId,
  vizConfig,
  resolveVizConfig,
  samples,
  className,
  onVizConfigUpdated,
  insightsOpen,
  onToggleInsights,
}: Props) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const dashboardPerms = useDashboardPerms();
  const analysisAccess = useResourceAccess("analysis", analysisId);
  const canCreate = analysisAccess.canEdit && (dashboardPerms.loading || dashboardPerms.canCreate);
  const canEdit = analysisAccess.canEdit && (dashboardPerms.loading || dashboardPerms.canEdit);
  const { data: dashboard } = useQuery({
    queryKey: projectDashboardQueryKey(analysisId, vizConfig),
    enabled: !!analysisId,
    queryFn: () => fetchAnalysisDashboard(projectId, analysisId),
  });
  const widgets = dashboard?.widgets ?? [];
  const tabWidgets = widgetsForTab(widgets, activeTabId);

  const [dragOverSlot, setDragOverSlot] = useState<number | null>(null);
  const [savingSlot, setSavingSlot] = useState<number | null>(null);
  const [removingSlot, setRemovingSlot] = useState<number | null>(null);
  const [maximizedWidget, setMaximizedWidget] = useState<(typeof tabWidgets)[number] | null>(null);
  

  const handleGenerateReport = async () => {
    if (!tabWidgets.length) {
      toast.error("Add at least one chart to the dashboard before generating a report.");
      return;
    }
    if (!analysisId) {
      toast.error("Save the analysis before generating a report.");
      return;
    }
    try {
      const payload = {
        createdAt: Date.now(),
        tabId: activeTabId,
        source: "dashboard-tab" as const,
        widgets: tabWidgets,
        samples: samples ?? [],
      };
      sessionStorage.setItem("avaloka:generated-report", JSON.stringify(payload));

      const firstTitle = tabWidgets[0]?.title?.trim();
      await createReport({
        projectId,
        analysisId,
        title: firstTitle || "Generated report",
        keyInsights: payload,
      });

      toast.success("Report generated");
      navigate({ to: "/reports" });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not generate report");
    }
  };


  const handleDrop = async (slotIndex: number, event: DragEvent<Element>) => {
    event.preventDefault();
    setDragOverSlot(null);

    if (!projectId || !analysisId) {
      toast.error("Save the analysis to a project before adding charts");
      return;
    }

    const drag = readInsightDragData(event);
    if (!drag) return;

    const targetAnalysisId = drag.analysisId ?? analysisId;
    const targetViz =
      (drag.analysisId ? resolveVizConfig?.(drag.analysisId) : null) ??
      (targetAnalysisId === analysisId ? vizConfig : resolveVizConfig?.(targetAnalysisId)) ??
      vizConfig;

    if (!targetViz) {
      toast.error("Could not load chart data for this analysis");
      return;
    }

    const insight = insightPayloadFromVizConfig(targetViz, drag.chartIndex);
    if (!insight) {
      toast.error("Could not load chart data");
      return;
    }

    if (!activeTabId) {
      toast.error("No dashboard tab available");
      return;
    }

    setSavingSlot(slotIndex);
    try {
      const result = await addInsightToDashboard({
        projectId,
        analysisId: targetAnalysisId,
        tab: activeTabId,
        position: "custom",
        slotIndex,
        insight,
        vizConfig: targetViz,
      });
      const { dashboard: nextDashboard, vizConfig: nextViz } = normalizeAddInsightResult(result);
      setProjectDashboardCache(qc, targetAnalysisId, nextViz ?? targetViz, nextDashboard);
      onVizConfigUpdated?.(nextViz, targetAnalysisId);
      toast.success("Chart added to dashboard");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save chart");
    } finally {
      setSavingSlot(null);
    }
  };

  const handleRemove = async (widgetAnalysisId: string, slotIndex: number) => {
    if (!projectId || !activeTabId) return;
    setRemovingSlot(slotIndex);
    try {
      const targetViz =
        widgetAnalysisId === analysisId ? vizConfig : (resolveVizConfig?.(widgetAnalysisId) ?? vizConfig);
      const result = await removeInsightFromDashboard({
        projectId,
        analysisId: widgetAnalysisId,
        tab: activeTabId,
        slotIndex,
        vizConfig: targetViz,
      });
      const { dashboard: nextDashboard, vizConfig: nextViz } = normalizeAddInsightResult(result);
      setProjectDashboardCache(qc, widgetAnalysisId, nextViz, nextDashboard);
      onVizConfigUpdated?.(nextViz, widgetAnalysisId);
      toast.success("Chart removed from dashboard");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove chart");
    } finally {
      setRemovingSlot(null);
    }
  };

  const slotDropHandlers = (slotIndex: number) => (!canCreate ? {} : {
    onDragOver: (event: DragEvent<Element>) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      setDragOverSlot(slotIndex);
    },
    onDragLeave: () => {
      setDragOverSlot((current) => (current === slotIndex ? null : current));
    },
    onDrop: (event: DragEvent<Element>) => {
      void handleDrop(slotIndex, event);
    },
  });

  return (

    <div className={cx("flex min-h-0 flex-1 flex-col overflow-hidden", className)}>
      
      <div className="flex flex-wrap items-center gap-2 border-b border-secondary bg-primary px-4 py-3 sm:px-6">
        {onToggleInsights ? (
          <button
            type="button"
            onClick={onToggleInsights}
            aria-label={insightsOpen ? "Hide insights panel" : "Show insights panel"}
            aria-pressed={!!insightsOpen}
            title={insightsOpen ? "Hide insights" : "Show insights"}
            className={cx(
              "grid size-9 place-items-center rounded-lg border border-secondary bg-primary text-fg-secondary transition hover:bg-primary_hover",
              insightsOpen && "border-brand bg-brand-primary text-brand-secondary",
            )}
          >
            <LayoutLeft className="size-4" />
          </button>
        ) : null}
        <div className="ml-auto flex items-center gap-2">
          
          <Button
            size="sm"
            color="primary"
            iconLeading={File02}
            onClick={ handleGenerateReport}
            
            isDisabled={!tabWidgets.length}
          >
            Generate report
          </Button>
        </div>
      </div>
      <div className="grid min-w-0 flex-1 auto-rows-[minmax(280px,auto)] grid-cols-1 content-start gap-4 overflow-y-auto bg-secondary/40 p-4 sm:grid-cols-2 sm:p-6 2xl:grid-cols-6">

        {DASHBOARD_SLOT_LAYOUT.map(({ position, colSpan }) => {
          const widget = tabWidgets.find((w) => w.position === position);
          if (widget) {
            return (
              <div
                key={widget.id}
                {...slotDropHandlers(position)}
                className={cx(
                  dashboardSlotColClass(colSpan),
                  "group relative flex min-h-[280px] min-w-0 flex-col overflow-hidden rounded-xl border border-secondary bg-primary p-4",
                  dragOverSlot === position && "border-brand ring-1 ring-brand/30",
                  (savingSlot === position || removingSlot === position) && "opacity-60",
                )}
              >
                {canEdit ? (
                <button
                  type="button"
                  aria-label="Remove chart from dashboard"
                  disabled={removingSlot === position}
                  onClick={() => void handleRemove(widget.analysisId, position)}
                  className="absolute right-3 top-3 z-10 rounded-md p-1 text-fg-quaternary hover:bg-primary_hover hover:text-fg-secondary disabled:opacity-50"
                >
                  <X className="size-4" />
                </button>
                ) : null}
                <p className="pr-8 text-sm font-semibold text-primary">{widget.title}</p>
                <div className="mt-2 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
                  <DashboardMiniChart widget={widget} samples={samples} />
                </div>
                <button
                  type="button"
                  aria-label={`Maximize ${widget.title}`}
                  title="View larger chart"
                  onClick={() => setMaximizedWidget(widget)}
                  className="absolute left-1/2 top-1/2 z-20 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-xl border border-secondary bg-primary/90 text-secondary opacity-30 shadow-md backdrop-blur-sm transition hover:scale-105 hover:text-primary focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-brand group-hover:opacity-90"
                >
                  <Expand06 className="size-5" />
                </button>
              </div>
            );
          }
          return (
            <DashboardEmptySlot
              key={`slot-${position}`}
              colSpan={colSpan}
              dragOver={dragOverSlot === position}
              saving={savingSlot === position}
              disabled={savingSlot === position || !canCreate}
              {...slotDropHandlers(position)}
            />
          );
        })}
      </div>
      <Dialog open={!!maximizedWidget} onOpenChange={(open) => !open && setMaximizedWidget(null)}>
        <DialogContent className="grid h-[90vh] w-[94vw] max-w-none grid-rows-[auto_1fr] gap-3 overflow-hidden rounded-xl border-secondary bg-primary p-5 shadow-2xl dark:border-white/25">
          <DialogTitle className="truncate pr-10 text-base font-semibold text-primary">
            {maximizedWidget?.title ?? "Chart"}
          </DialogTitle>
          {maximizedWidget ? (
            <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
              <DashboardMiniChart
                widget={maximizedWidget}
                samples={samples}
                height={typeof window === "undefined" ? 590 : Math.max(360, window.innerHeight * 0.68)}
                compact={false}
              />
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
