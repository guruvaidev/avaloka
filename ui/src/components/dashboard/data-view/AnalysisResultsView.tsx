// Analysis results view for the Data View: one header strip per chat-produced
// result table with prompt, run time, prev/next navigation, a jump dropdown and
// per-analysis table pills. Works only on runs passed in; never fetches.

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { BarChart3, ChevronLeft, ChevronRight, MessageSquare } from "lucide-react";
import { Button } from "@/components/base/buttons/button";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { cx } from "@/lib/utils/cx";
import { formatRunTime, resultDisplayPrompt, tablesForRun, type TableRun } from "@/lib/analysis-results";
import { DataGrid, type HighlightRequest } from "./DataGrid";
import type { ColumnKind, ColumnProfile } from "@/lib/data-view";

export function AnalysisResultsView({
  runs,
  index,
  onIndexChange,
  tableIndex,
  onTableChange,
  profiles = [],
  highlight,
  onChartColumn,
  onOpenInChat,
  onAskInChat,
}: {
  runs: TableRun[];
  /** 0-based index of the visible analysis. */
  index: number;
  onIndexChange: (index: number) => void;
  /** 0-based index of the visible table within the current analysis. */
  tableIndex: number;
  onTableChange: (index: number) => void;
  profiles?: ColumnProfile[];
  highlight?: HighlightRequest;
  onChartColumn?: (column: string, kind: ColumnKind) => void;
  onOpenInChat?: (run: TableRun) => void;
  onAskInChat?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [clamped, setClamped] = useState(false);
  const promptRef = useRef<HTMLParagraphElement | null>(null);
  const run = runs[index];
  const tables = run ? tablesForRun(run) : [];
  const safeTable = Math.min(tableIndex, Math.max(tables.length - 1, 0));
  const activeTable = tables[safeTable];

  const prompt = run ? resultDisplayPrompt(run, index) : "";
  useEffect(() => { setExpanded(false); }, [run?.id, prompt]);
  useEffect(() => {
    const element = promptRef.current;
    if (!element || expanded) return;
    const measure = () => setClamped(element.scrollHeight > element.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [prompt, expanded]);

  const changeRun = (next: number) => {
    setExpanded(false);
    onTableChange(0);
    onIndexChange(next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest("button, input, textarea, select, [role='combobox'], [role='option']")) return;
    if (event.key === "ArrowLeft" && index > 0) {
      event.preventDefault();
      changeRun(index - 1);
    } else if (event.key === "ArrowRight" && index < runs.length - 1) {
      event.preventDefault();
      changeRun(index + 1);
    }
  };

  if (!run) {
    return (
      <div className="flex flex-col items-center gap-3 px-4 py-14 text-center">
        <div className="grid size-12 place-items-center rounded-full bg-[#eaf1fd] text-[#1a5fe0]">
          <BarChart3 className="size-6" />
        </div>
        <p className="max-w-sm text-sm text-secondary">
          No analysis results yet. Ask a question in the chat, for example &lsquo;Average subscribers by category&rsquo;.
        </p>
        <button
          type="button"
          onClick={onAskInChat}
          className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
        >
          <MessageSquare className="size-4" /> Ask in chat
        </button>
      </div>
    );
  }

  const time = formatRunTime(run.ranAt);

  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
    <div role="region" aria-label="Analysis results" onKeyDown={onKeyDown} className="outline-none">
      <div className="analysis-result-header flex flex-wrap items-start justify-between gap-3 rounded-[10px] border border-border bg-card px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <p className="text-xs text-muted-foreground">
              Analysis {index + 1} of {runs.length}
            </p>
            {time ? <p className="text-xs text-muted-foreground">· {time}</p> : null}
          </div>
          <p
            className={cx("mt-1 text-sm text-foreground [overflow-wrap:anywhere]", !expanded && "line-clamp-2")}
            ref={promptRef}
          >
            {prompt}
          </p>
          <div className="mt-1 flex items-center gap-4">
            {clamped || expanded ? (
              <Button size="xs" color="link-color" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)} className="text-xs text-primary">
                {expanded ? "Show less" : "Show more"}
              </Button>
            ) : null}
            {onOpenInChat ? (
              <Button size="xs" color="link-color" iconLeading={MessageSquare} onClick={() => onOpenInChat(run)} className="gap-1.5 text-xs text-primary *:data-icon:size-3">
                Open in chat
              </Button>
            ) : null}
          </div>
        </div>
        {runs.length > 1 ? (
          <div className="flex max-w-full shrink-0 items-center gap-1.5">
            <Button size="xs" color="tertiary" aria-label="Previous analysis" isDisabled={index === 0} aria-disabled={index === 0} onClick={() => changeRun(index - 1)} className="size-8 bg-muted text-foreground hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
              <ChevronLeft className="size-4" />
            </Button>
            <Select value={String(index)} onValueChange={(value) => changeRun(Number(value))}>
              <SelectTrigger aria-label="Jump to analysis" className="h-8 w-auto min-w-[140px] max-w-[280px] bg-background text-foreground">
                <SelectValue>Analysis {index + 1}</SelectValue>
              </SelectTrigger>
              <SelectContent className="max-w-[min(480px,calc(100vw-32px))] border-border bg-popover text-popover-foreground dark:border-border dark:bg-popover dark:text-popover-foreground">
                {runs.map((item, i) => (
                  <SelectItem key={item.id} value={String(i)} title={resultDisplayPrompt(item, i)}>
                    {i + 1} · {resultDisplayPrompt(item, i).slice(0, 60)}{resultDisplayPrompt(item, i).length > 60 ? "…" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button size="xs" color="tertiary" aria-label="Next analysis" isDisabled={index === runs.length - 1} aria-disabled={index === runs.length - 1} onClick={() => changeRun(index + 1)} className="size-8 bg-muted text-foreground hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
              <ChevronRight className="size-4" />
            </Button>
          </div>
        ) : null}
      </div>

      {tables.length > 1 ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Tables in this analysis">
          {tables.map((table, i) => (
            <button
              key={i}
              type="button"
              role="tab"
              aria-selected={safeTable === i}
              onClick={() => onTableChange(i)}
              className={cx(
                "rounded-full px-3 py-1 text-xs font-semibold transition",
                safeTable === i ? "bg-[#1565ef] text-white" : "bg-[#e9edf5] text-tertiary hover:text-primary",
              )}
            >
              {table.name?.trim() || `Table ${i + 1}`}
            </button>
          ))}
        </div>
      ) : null}

      <div className="mt-3">
        <DataGrid
          key={`${run.id}:${safeTable}`}
          rows={activeTable?.rows ?? []}
          columns={activeTable ? Array.from(activeTable.rows.slice(0, 50).reduce((keys, row) => {
            Object.keys(row ?? {}).forEach((key) => keys.add(key));
            return keys;
          }, new Set<string>())) : []}
          profiles={profiles}
          highlight={highlight}
          onChartColumn={onChartColumn}
        />
      </div>
    </div>
  );
}
