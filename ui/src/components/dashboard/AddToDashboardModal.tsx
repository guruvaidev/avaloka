import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, X } from "@untitledui/icons";
import { Dialog, DialogPortal, DialogOverlay, DialogTitle } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cx } from "@/lib/utils/cx";
import { DASHBOARD_SLOT_LAYOUT, dashboardSlotColClass } from "@/lib/dashboard-layout";
import { toast } from "sonner";
import {
  addInsightToDashboard,
  analysisDashboardTabId,
  fetchAnalysisDashboard,
  insightPayloadFromVizConfig,
  normalizeAddInsightResult,
  projectDashboardQueryKey,
  setProjectDashboardCache,
  type DashboardInsightPayload,
} from "@/lib/api/project-dashboard";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId?: string | null;
  analysisId?: string | null;
  analysisTitle?: string | null;
  vizConfig?: unknown | null;
  chartIndex?: number;
  insight?: DashboardInsightPayload | null;
  onConfirm?: (tab: string, position: string, slotIndex?: number) => void;
  onVizConfigUpdated?: (vizConfig: Record<string, unknown>) => void;
};

const FALLBACK_TABS: { id: string; title: string }[] = [];

export function AddToDashboardModal({
  open,
  onOpenChange,
  projectId,
  analysisId,
  analysisTitle,
  vizConfig,
  chartIndex = 0,
  insight: insightProp,
  onConfirm,
  onVizConfigUpdated,
}: Props) {
  const qc = useQueryClient();
  const [tab, setTab] = useState("");
  const [position, setPosition] = useState("");
  const [slotIndex, setSlotIndex] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  const insight = useMemo(() => {
    if (vizConfig) return insightPayloadFromVizConfig(vizConfig, chartIndex);
    return insightProp ?? null;
  }, [vizConfig, chartIndex, insightProp]);

  const { data: dashboard } = useQuery({
    queryKey: projectDashboardQueryKey(analysisId ?? null, vizConfig),
    enabled: open && !!analysisId,
    queryFn: () => fetchAnalysisDashboard(projectId ?? null, analysisId ?? null),
  });

  const tabs = useMemo(() => {
    if (analysisId && analysisTitle) {
      return [{ id: analysisDashboardTabId(analysisId), title: analysisTitle }];
    }
    if (dashboard?.tabs.length) {
      return dashboard.tabs.map((t) => ({ id: t.id, title: t.title }));
    }
    return FALLBACK_TABS;
  }, [analysisId, analysisTitle, dashboard]);

  useEffect(() => {
    if (!open) return;
    if (!tab && tabs.length) setTab(tabs[0].id);
  }, [open, tab, tabs]);

  const isCustom = position === "custom";

  const customSlots = useMemo(() => {
    const activeTab = tabs.find((t) => t.id === tab) ?? tabs[0];
    const widgets = dashboard?.widgets.filter((w) => w.tabId === activeTab?.id && w.placed) ?? [];
    return DASHBOARD_SLOT_LAYOUT.map(({ position }) => {
      const filled = widgets.find((w) => w.position === position);
      return {
        position,
        label: filled?.title ?? "Empty",
        kind: filled ? ("filled" as const) : ("empty" as const),
      };
    });
  }, [dashboard, tab, tabs]);

  const handleConfirm = async () => {
    if (!tab || !position) {
      toast.error("Select a dashboard tab and position");
      return;
    }
    onConfirm?.(tab, position, slotIndex ?? undefined);

    if (!projectId || !analysisId || !insight) {
      toast.success("Added to dashboard");
      onOpenChange(false);
      return;
    }

    setSaving(true);
    try {
      const result = await addInsightToDashboard({
        projectId,
        analysisId,
        tab,
        position,
        slotIndex: slotIndex ?? undefined,
        insight,
        vizConfig,
      });
      const { dashboard, vizConfig: nextViz } = normalizeAddInsightResult(result);
      setProjectDashboardCache(qc, analysisId, nextViz ?? vizConfig, dashboard);
      onVizConfigUpdated?.(nextViz);
      toast.success("Added to dashboard");
      onOpenChange(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save widget");
    } finally {
      setSaving(false);
    }
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) setSlotIndex(null);
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogPortal>
        <DialogOverlay className="bg-black/30" />
        <DialogPrimitive.Content
          className={cx(
            "fixed left-1/2 top-1/2 z-50 grid w-full -translate-x-1/2 -translate-y-1/2 gap-0 rounded-2xl bg-white p-0 shadow-[0_24px_48px_-12px_rgba(16,24,40,0.18)] outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            isCustom ? "max-w-[600px]" : "max-w-[480px]",
          )}
        >
          <DialogTitle className="sr-only">Add to dashboard</DialogTitle>

          <div className="flex items-start gap-4 px-6 pb-5 pt-6">
            <div className="grid size-11 shrink-0 place-items-center rounded-[10px] border border-[#eaecf0] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.05)]">
              <Calendar className="size-5 text-[#475467]" />
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="text-base font-semibold leading-6 text-[#101828]">Add to dashboard</h2>
              <p className="mt-0.5 text-sm leading-5 text-[#475467]">
                Add the selected insights to the project dashboard
              </p>
            </div>
            <DialogPrimitive.Close
              className="grid size-7 shrink-0 cursor-pointer place-items-center rounded-md text-[#98a2b3] hover:bg-[#f2f4f7] hover:text-[#475467]"
              aria-label="Close"
            >
              <X className="size-5" />
            </DialogPrimitive.Close>
          </div>

          <div className="h-px bg-[#eaecf0]" />

          <div className="space-y-4 px-6 py-5">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-[#344054]">Dashboard Tab</label>
              <Select value={tab} onValueChange={setTab}>
                <SelectTrigger className="h-11 w-full rounded-lg border-[#d0d5dd] bg-white px-3.5 text-sm text-[#101828] shadow-[0_1px_2px_rgba(16,24,40,0.05)] data-[placeholder]:text-[#667085]">
                  <SelectValue placeholder="Select Tab" />
                </SelectTrigger>
                <SelectContent>
                  {tabs.map((t) => (
                    <SelectItem key={t.id} value={t.id}>
                      {t.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-[#344054]">Position</label>
              <Select
                value={position}
                onValueChange={(v) => {
                  setPosition(v);
                  setSlotIndex(null);
                }}
              >
                <SelectTrigger className="h-11 w-full rounded-lg border-[#d0d5dd] bg-white px-3.5 text-sm text-[#101828] shadow-[0_1px_2px_rgba(16,24,40,0.05)] data-[placeholder]:text-[#667085]">
                  <SelectValue placeholder="Select Position" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="top">Top of the Section</SelectItem>
                  <SelectItem value="after-recent">After Recent Widget</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {isCustom && (
              <div className="pt-1">
                <p className="text-sm font-medium text-[#344054]">Select Custom Position</p>
                <div className="mt-4 grid grid-cols-6 gap-3">
                  {customSlots.map((slot) => {
                    const layout = DASHBOARD_SLOT_LAYOUT.find((s) => s.position === slot.position)!;
                    const selected = slotIndex === slot.position;
                    const isEmpty = slot.kind === "empty";
                    return (
                      <button
                        key={slot.position}
                        type="button"
                        onClick={() => setSlotIndex(slot.position)}
                        className={cx(
                          dashboardSlotColClass(layout.colSpan),
                          "flex h-12 items-center justify-between rounded-lg border bg-white px-3 text-sm shadow-[0_1px_2px_rgba(16,24,40,0.05)] transition",
                          isEmpty
                            ? "border-[#1565ef] text-[#1565ef]"
                            : "border-[#eaecf0] text-[#101828] hover:border-[#d0d5dd]",
                          selected && "ring-2 ring-[#1565ef] ring-offset-1",
                        )}
                      >
                        <span className="truncate font-medium">{slot.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div className="h-px bg-[#eaecf0]" />

          <div className="flex justify-end gap-3 px-6 py-4">
            <button
              type="button"
              onClick={() => handleOpenChange(false)}
              className="inline-flex h-10 items-center justify-center rounded-lg border border-[#d0d5dd] bg-white px-4 text-sm font-semibold text-[#344054] shadow-[0_1px_2px_rgba(16,24,40,0.05)] hover:bg-[#f9fafb]"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={saving}
              onClick={() => void handleConfirm()}
              className="inline-flex h-10 items-center justify-center rounded-lg bg-[#1565ef] px-4 text-sm font-semibold text-white shadow-[0_1px_2px_rgba(16,24,40,0.05)] hover:bg-[#1257d6] disabled:opacity-60"
            >
              {saving ? "Saving…" : "Add to Dashboard"}
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPortal>
    </Dialog>
  );
}
