import { useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { isActiveJob, jobRows, jobTitle, localJobStart, logClockNote, parseJobResult, rayJobResultsOptions, rayRequest, RayApiError, type RayJob, type RayLogs, type RayStatus } from "@/lib/ray-jobs";
import { JobIdentity, JobStatusBadge } from "./RayJobPresentation";
import { useRayJobsVisible } from "./useRayJobsLive";

export function RayLogsDrawer({ job, cluster, onClose, onUpgrade }: { job: RayJob | null; cluster?: RayStatus; onClose: () => void; onUpgrade: (error: RayApiError) => void }) {
  const body = useRef<HTMLPreElement>(null);
  const queryClient = useQueryClient();
  const reportedResults = useQuery(rayJobResultsOptions);
  const id = job?.submission_id;
  const visible = useRayJobsVisible(false);
  const previousJob = useRef<RayJob | null>(null);
  const logs = useQuery({
    queryKey: ["ray-jobs", "logs", id],
    staleTime: 0,
    queryFn: ({ signal }) => rayRequest<RayLogs>(`/api/ray/jobs/${encodeURIComponent(id ?? "")}/logs?tail=500`, "GET", signal),
    enabled: !!id && visible,
    retry: false,
    refetchInterval: job && isActiveJob(job) ? 5000 : false,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
  useEffect(() => {
    const previous = previousJob.current;
    if (!visible) return;
    previousJob.current = job;
    if (previous && job && previous.submission_id === job.submission_id && isActiveJob(previous) && !isActiveJob(job)) {
      const wasFetching = logs.isFetching;
      void logs.refetch({ cancelRefetch: false }).then((response) => {
        // If a tick was already in flight at completion, fetch once more after it.
        if (wasFetching && response.data && !response.error) void logs.refetch({ cancelRefetch: false });
      });
    }
  }, [job, visible, logs.refetch]);
  const parsed = logs.data ? parseJobResult(logs.data.lines.join("\n")) : undefined;
  const result = parsed?.rowsInserted !== undefined ? parsed : id ? reportedResults.data[id] : undefined;
  const rows = job ? jobRows(job, result) : undefined;
  const destination = job?.destination || result?.table;
  const start = job ? localJobStart(job) : undefined;
  useEffect(() => {
    if (!id || !logs.data) return;
    const result = parseJobResult(logs.data.lines.join("\n"));
    if (result.rowsInserted === undefined) return;
    queryClient.setQueryData(rayJobResultsOptions.queryKey, (old: Record<string, typeof result> | undefined) => ({ ...old, [id]: result }));
  }, [id, logs.data, queryClient]);
  useEffect(() => { if (body.current) body.current.scrollTop = body.current.scrollHeight; }, [logs.data]);
  useEffect(() => { if (logs.error instanceof RayApiError && logs.error.upgradeRequired) onUpgrade(logs.error); }, [logs.error, onUpgrade]);
  const download = () => {
    if (!logs.data || !id) return;
    const url = URL.createObjectURL(new Blob([logs.data.lines.join("\n")], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${id}.log`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast.success("Logs downloaded");
  };
  return <Sheet open={!!job} onOpenChange={(open) => { if (!open) onClose(); }}>
    <SheetContent className="flex w-full flex-col gap-0 border-border bg-card p-0 text-foreground sm:max-w-[560px]">
      <header className="shrink-0 border-b border-border p-5 pr-12">
        <SheetTitle className="break-words text-base">{job ? jobTitle(job) : "Job logs"}</SheetTitle>
        {job && <div className="mt-3 flex flex-wrap items-center gap-2"><JobStatusBadge status={job.status} /><JobIdentity job={job} /></div>}
        <SheetDescription className="mt-2 break-words">{rows !== undefined && destination ? `${rows.toLocaleString()} rows transferred to ${destination}` : job?.message ?? "Job logs"}</SheetDescription>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-2 p-5">
        {job && <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs text-muted-foreground"><span>{logClockNote(job, cluster)}</span>{start && <span>Started {start}</span>}</div>}
        {logs.data?.truncated && <p className="text-xs text-muted-foreground">Showing the last 500 lines</p>}
        {logs.error ? <div role="alert" className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 rounded-lg bg-ray-log-bg p-5 text-center text-ray-log-text">
          <p>{"status" in logs.error && logs.error.status === 503 ? "Ray cluster unavailable" : logs.error.message}</p>
          <Button variant="outline" className="text-foreground" onClick={() => logs.refetch({ cancelRefetch: false })}>Retry</Button>
        </div> : <pre ref={body} aria-label="Job logs" aria-busy={logs.isFetching} className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-all rounded-lg bg-ray-log-bg p-4 font-mono text-xs leading-6 text-ray-log-text">{logs.isPending ? "Loading logs…" : logs.data?.lines.join("\n") || "No logs yet."}</pre>}
      </div>
      <footer className="flex shrink-0 justify-end gap-2 border-t border-border p-4">
        <Button variant="outline" disabled={logs.isFetching} onClick={() => logs.refetch({ cancelRefetch: false })}><RefreshCw className={logs.isFetching ? "animate-spin motion-reduce:animate-none" : ""} />Refresh</Button>
        <Button className="bg-brand-solid text-primary_on-brand hover:bg-brand-solid_hover" disabled={!logs.data || !!logs.error} onClick={download}><Download />Download</Button>
      </footer>
    </SheetContent>
  </Sheet>;
}