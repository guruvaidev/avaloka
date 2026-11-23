import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { fidelityApi, FidelityError, fidelityReplyDiffers, needsFidelityPolling, FIDELITY_POLL_MS, type FidelityState, type FidelityReply, type AnalysisFidelity } from "@/lib/analysis-fidelity";

type Props = {
  datasetId: string;
  sessionId: string;
  threadId: string | null;
  chatResponse?: FidelityReply | null;
  onStateChange?: (state: FidelityState) => void;
  onSampleChange?: (datasetId: string) => Promise<void>;
};

export function FidelitySwitch({ datasetId, sessionId, threadId, chatResponse, onStateChange, onSampleChange }: Props) {
  const [state, setState] = useState<FidelityState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const stateRef = useRef(state);
  const callbacks = useRef({ onStateChange, onSampleChange });
  callbacks.current = { onStateChange, onSampleChange };
  const controllerRef = useRef<AbortController | null>(null);
  const commit = useCallback((next: FidelityState | null) => {
    stateRef.current = next;
    setState(next);
    if (next) callbacks.current.onStateChange?.(next);
  }, []);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const next = await fidelityApi.get(datasetId, sessionId, signal);
      if (!signal?.aborted) commit(next);
    } catch (error) {
      if (signal?.aborted) return;
      commit(null);
      if (!(error instanceof FidelityError && error.status === 404)) console.error("[FidelitySwitch] load failed", error);
    }
  }, [datasetId, sessionId, commit]);

  useEffect(() => {
    const controller = new AbortController();
    controllerRef.current = controller;
    commit(null);
    setLoading(true);
    setSaving(false);
    void refresh(controller.signal).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh, threadId, commit]);

  useEffect(() => {
    if (stateRef.current && chatResponse && fidelityReplyDiffers(stateRef.current, chatResponse)) {
      void refresh(controllerRef.current?.signal);
    }
  }, [chatResponse, refresh]);

  const polling = state ? needsFidelityPolling(state) : false;
  const pollingTaskId = state?.background_task_id;
  useEffect(() => {
    if (!polling) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await fidelityApi.status(datasetId, sessionId, controller.signal);
        if (controller.signal.aborted) return;
        if (["SUCCESS", "FAILURE", "NONE"].includes(result.state)) {
          if (result.state === "FAILURE") toast.error("Portfolio sample build failed");
          await refresh(controller.signal);
          return;
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        console.error("[FidelitySwitch] status failed", error);
      }
      timer = setTimeout(() => void poll(), FIDELITY_POLL_MS);
    };
    timer = setTimeout(() => void poll(), FIDELITY_POLL_MS);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [polling, pollingTaskId, datasetId, sessionId, refresh]);

  const change = async (mode: AnalysisFidelity, sample?: string) => {
    const previous = stateRef.current;
    const signal = controllerRef.current?.signal;
    if (!previous || saving) return;
    setSaving(true);
    commit({ ...previous, analysis_fidelity: mode, ...(sample !== undefined ? { selected_sample_name: sample } : {}) });
    try {
      const next = await fidelityApi.put(datasetId, sessionId, {
        analysis_fidelity: mode,
        ...(sample !== undefined ? { selected_sample_name: sample } : {}),
      }, signal);
      if (signal?.aborted) return;
      commit(next);
      toast.success(sample ? `${sample} selected` : `${next.options.find((o) => o.value === next.analysis_fidelity)?.label ?? "Analysis mode"} selected`);
      if (next.selected_sample_name !== previous.selected_sample_name) {
        try { await callbacks.current.onSampleChange?.(datasetId); }
        catch (error) { console.error("[FidelitySwitch] preview failed", error); toast.error("Sample selected, but its rows could not be refreshed"); }
      }
    } catch (error) {
      if (signal?.aborted) return;
      commit(previous);
      toast.error(error instanceof Error ? error.message : "Could not update analysis mode");
    } finally { if (!signal?.aborted) setSaving(false); }
  };

  if (loading) return <Skeleton className="h-9 w-44 shrink-0" aria-label="Loading analysis mode" />;
  if (!state) return null;
  const portfolio = state.options.find((o) => o.value === "portfolio_samples");
  const selected = state.options.find((o) => o.value === state.analysis_fidelity);
  const label = (option: NonNullable<typeof selected>) => `${option.label}${option.value === "portfolio_samples" && option.status === "building" ? " (building…)" : ""}`;
  return (
    <TooltipProvider>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <div className="flex items-center gap-2">
          <span className="whitespace-nowrap text-xs text-tertiary">Analysis on</span>
          <Select value={state.analysis_fidelity} onValueChange={(value) => void change(value as AnalysisFidelity)} disabled={saving}>
            <SelectTrigger aria-label="Analysis on" className="h-9 w-44 border-secondary bg-primary text-primary"><SelectValue>{selected ? label(selected) : state.analysis_fidelity}</SelectValue></SelectTrigger>
            <SelectContent className="max-w-[calc(100vw-2rem)] border-secondary bg-primary text-primary dark:border-secondary dark:bg-primary dark:text-primary">
              {state.options.map((option) => (
                <Tooltip key={option.value}>
                  <TooltipTrigger asChild>
                    <div>
                      <SelectItem value={option.value} disabled={!option.enabled} textValue={label(option)} className="focus:bg-primary_hover focus:text-primary">
                        <span className="block">{label(option)}</span>
                        <span className="block max-w-72 whitespace-normal text-xs text-tertiary">{option.detail}</span>
                      </SelectItem>
                    </div>
                  </TooltipTrigger>
                  {!option.enabled && option.reason ? <TooltipContent>{option.reason}</TooltipContent> : null}
                </Tooltip>
              ))}
            </SelectContent>
          </Select>
        </div>
        {state.analysis_fidelity === "portfolio_samples" && Boolean(portfolio?.samples?.length) ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-tertiary">Sample</span>
            <Select value={state.selected_sample_name ?? ""} disabled={saving} onValueChange={(sample) => void change("portfolio_samples", sample)}>
              <SelectTrigger aria-label="Sample" className="h-9 w-40 border-secondary bg-primary text-primary"><SelectValue placeholder="Select sample" /></SelectTrigger>
              <SelectContent className="border-secondary bg-primary text-primary dark:border-secondary dark:bg-primary dark:text-primary">
                {portfolio?.samples?.map((sample) => <SelectItem key={sample} value={sample}>{sample}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        ) : null}
        {selected?.runs_as_job ? <Tooltip><TooltipTrigger asChild><Badge tabIndex={0} variant="outline" className="border-secondary text-primary">Runs as background Ray job</Badge></TooltipTrigger><TooltipContent className="max-w-72">Each question is scheduled on the cluster. Results appear in the task panel when the job finishes.</TooltipContent></Tooltip> : null}
      </div>
    </TooltipProvider>
  );
}