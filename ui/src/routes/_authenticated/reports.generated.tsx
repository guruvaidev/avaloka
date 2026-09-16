import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowLeft, Share07, Calendar } from "@untitledui/icons";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { DashboardMiniChart } from "@/components/dashboard/dashboardMiniChart";
import { DASHBOARD_SLOT_LAYOUT, dashboardSlotColClass } from "@/lib/dashboard-layout";
import type { DashboardWidget } from "@/lib/api/project-dashboard";
import { cx } from "@/lib/utils/cx";

export const Route = createFileRoute("/_authenticated/reports/generated")({
  head: () => ({
    meta: [
      { title: "Generated Report · Avaloka AI" },
      { name: "description", content: "Report generated from your dashboard charts." },
    ],
  }),
  component: GeneratedReportPage,
});

type StoredReport = {
  createdAt: number;
  tabId: string;
  widgets: DashboardWidget[];
  samples: Record<string, unknown>[];
};

function GeneratedReportPage() {
  const navigate = useNavigate();
  const [report, setReport] = useState<StoredReport | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("avaloka:generated-report");
      if (raw) setReport(JSON.parse(raw) as StoredReport);
    } catch {
      /* ignore */
    }
    setLoaded(true);
  }, []);

  const widgets = report?.widgets ?? [];
  const samples = report?.samples ?? [];
  const generatedAt = report?.createdAt ? new Date(report.createdAt) : null;

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-primary text-primary">
      <header className="flex items-center justify-between gap-3 border-b border-secondary bg-primary px-6 py-3">
        <div className="flex items-center gap-3">
          <ButtonUtility
            size="sm"
            color="tertiary"
            icon={ArrowLeft}
            tooltip="Back"
            onClick={() => navigate({ to: "/reports" })}
          />
          <div className="leading-tight">
            <h1 className="text-md font-semibold text-primary">Generated Report</h1>
            <p className="text-xs text-tertiary">
              {generatedAt
                ? `Created ${generatedAt.toLocaleString()}`
                : "From dashboard charts"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button color="primary" size="sm" iconLeading={Share07}>
            Share
          </Button>
          <Button color="secondary" size="sm" iconLeading={Calendar}>
            Schedule
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto bg-secondary/40 p-6">
        {!loaded ? null : widgets.length === 0 ? (
          <div className="mx-auto max-w-md rounded-xl border border-secondary bg-primary p-8 text-center">
            <p className="text-md font-semibold text-primary">No charts to report</p>
            <p className="mt-2 text-sm text-tertiary">
              Drag charts into the dashboard and click "Generate report" to build a report.
            </p>
            <div className="mt-4 flex justify-center">
              <Button size="sm" color="primary" onClick={() => navigate({ to: "/dashboard" })}>
                Go to dashboard
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid auto-rows-fr grid-cols-6 gap-4">
            {DASHBOARD_SLOT_LAYOUT.map(({ position, colSpan }) => {
              const widget = widgets.find((w) => w.position === position);
              if (!widget) return null;
              return (
                <div
                  key={widget.id}
                  className={cx(
                    dashboardSlotColClass(colSpan),
                    "flex min-h-[220px] flex-col rounded-xl border border-secondary bg-primary p-4",
                  )}
                >
                  <p className="text-sm font-semibold text-primary">{widget.title}</p>
                  <div className="mt-2 flex min-h-0 flex-1 flex-col">
                    <DashboardMiniChart widget={widget} samples={samples} height={220} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
