import { useMemo } from "react";
import type { DashboardWidget } from "@/lib/api/project-dashboard";
import { DynamicChart, normalizeChart } from "./dynamicChart";

function chartFromWidget(widget: DashboardWidget): Record<string, unknown> | null {
  const cfg = widget.config;
  if (!cfg || typeof cfg !== "object") return null;

  const focusId = (cfg as { focusChartId?: unknown }).focusChartId;
  const charts = Array.isArray((cfg as { charts?: unknown }).charts)
    ? ((cfg as { charts: Record<string, unknown>[] }).charts ?? [])
    : [];

  if (focusId != null) {
    const match = charts.find((c) => String(c.id) === String(focusId));
    if (match) return match;
  }

  const byKey = charts.find((c) => String(c.id) === widget.graphKey);
  if (byKey) return byKey;

  const idxMatch = /^chart:(\d+)$/.exec(widget.graphKey);
  if (idxMatch) {
    const idx = Number(idxMatch[1]);
    if (charts[idx]) return charts[idx];
  }

  const byTitle = charts.find((c) => String(c.title ?? "") === widget.title);
  if (byTitle) return byTitle;

  if (charts.length === 1) return charts[0];

  const withEncodings = charts.find((c) => c.encodings);
  if (withEncodings) return withEncodings;

  if ((cfg as { encodings?: unknown }).encodings) {
    return {
      id: widget.graphKey,
      title: widget.title,
      type: widget.chartType,
      encodings: (cfg as { encodings: unknown }).encodings,
      config: (cfg as { config?: unknown }).config,
    };
  }

  return charts[0] ?? null;
}

export function DashboardMiniChart({
  widget,
  samples,
  height = 160,
}: {
  widget: DashboardWidget;
  samples?: Record<string, unknown>[] | null;
  height?: number;
}) {
  const slide = useMemo(() => {
    const chart = chartFromWidget(widget);
    if (!chart || !samples?.length) return null;
    return normalizeChart(chart, samples);
  }, [widget, samples]);

  if (!slide) {
    return (
      <div className="flex h-full min-h-[88px] items-center justify-center text-xs text-tertiary">No chart data</div>
    );
  }

  return (
    <div className="min-h-0 flex-1">
      <DynamicChart slide={slide} height={height} compact chartKey={widget.graphKey} />
    </div>
  );
}
