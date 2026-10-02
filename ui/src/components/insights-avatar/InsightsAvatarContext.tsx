import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

type ExplainRequest = { chartId: string; nonce: number };

type Ctx = {
  focusedChartId: string | null;
  setFocusedChartId: (id: string | null) => void;
  highlightedIds: string[];
  setHighlightedIds: (ids: string[]) => void;
  explainRequest: ExplainRequest | null;
  requestExplain: (chartId: string) => void;
  /** Props to spread on a chart card for focus tracking + scroll targeting. */
  bindChart: (chartId: string) => Record<string, unknown>;
  isHighlighted: (chartId: string) => boolean;
};

const noop = () => {};
const InsightsAvatarCtx = createContext<Ctx>({
  focusedChartId: null,
  setFocusedChartId: noop,
  highlightedIds: [],
  setHighlightedIds: noop,
  explainRequest: null,
  requestExplain: noop,
  bindChart: () => ({}),
  isHighlighted: () => false,
});

export function InsightsAvatarProvider({ children }: { children: ReactNode }) {
  const [focusedChartId, setFocusedChartId] = useState<string | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<string[]>([]);
  const [explainRequest, setExplainRequest] = useState<ExplainRequest | null>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearHover = () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = null;
  };

  const requestExplain = useCallback((chartId: string) => {
    setFocusedChartId(chartId);
    setExplainRequest({ chartId, nonce: Date.now() + Math.random() });
  }, []);

  const bindChart = useCallback(
    (chartId: string) => ({
      "data-insight-chart-id": chartId,
      onMouseEnter: () => {
        clearHover();
        hoverTimer.current = setTimeout(() => setFocusedChartId(chartId), 1000);
      },
      onMouseLeave: clearHover,
      onClickCapture: () => setFocusedChartId(chartId),
    }),
    [],
  );

  const value = useMemo<Ctx>(
    () => ({
      focusedChartId,
      setFocusedChartId,
      highlightedIds,
      setHighlightedIds,
      explainRequest,
      requestExplain,
      bindChart,
      isHighlighted: (id) => highlightedIds.includes(id),
    }),
    [focusedChartId, highlightedIds, explainRequest, requestExplain, bindChart],
  );

  return <InsightsAvatarCtx.Provider value={value}>{children}</InsightsAvatarCtx.Provider>;
}

export function useInsightsAvatar() {
  return useContext(InsightsAvatarCtx);
}

export const HIGHLIGHT_CLASS = "!border-[#1a5fe8] ring-2 ring-[#1a5fe8] shadow-[0_0_0_6px_rgba(26,95,232,0.15),0_12px_32px_-8px_rgba(26,95,232,0.45)]";
