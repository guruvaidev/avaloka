import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient, useSuspenseQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { AlertTriangle, LoaderCircle, RefreshCw, Server } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertDialog, AlertDialogContent, AlertDialogTitle, AlertDialogDescription, AlertDialogCancel, AlertDialogFooter } from "@/components/ui/alert-dialog";
import { TooltipProvider } from "@/components/ui/tooltip";
import { UpgradePlanDialog } from "@/components/dashboard/UpgradeGate";
import {
  durationText, hasDataTransfer, isActiveJob, matchesJobId, relativeStarted,
  RayApiError, rayJobsOptions, rayLimitsOptions, rayRequest, rayStatusOptions,
  jobRows, jobStartTime, localJobStart, rayJobResultsOptions, type RayFilters, type RayJob,
} from "@/lib/ray-jobs";
import { ClusterStatusPill, JobIdentity, JobPath, JobStatusBadge } from "./RayJobPresentation";
import { RayLogsDrawer } from "./RayLogsDrawer";
import { useRayJobsVisible } from "./useRayJobsLive";

export interface RayJobsPageProps {
  filters: RayFilters;
  jobId: string;
  onFiltersChange: (filters: RayFilters) => void;
  onJobChange: (id: string) => void;
}
export function RayJobsPage({ filters, jobId, onFiltersChange, onJobChange }: RayJobsPageProps) {
  const limits = useSuspenseQuery(rayLimitsOptions);
  const allowed = hasDataTransfer(limits.data.data);
  const queryClient = useQueryClient();
  const visible = useRayJobsVisible();
  const status = useQuery({ ...rayStatusOptions, enabled: allowed && visible });
  const list = useQuery({ ...rayJobsOptions(filters), enabled: allowed && visible });
  const jobs = list.data?.data?.jobs ?? [];
  const reportedResults = useQuery(rayJobResultsOptions);
  const matched = jobs.find((job) => matchesJobId(job.submission_id, jobId));
  const resolvedId = matched?.submission_id ?? (/^dta-[0-9a-f]{16}$/.test(jobId) ? `dta-${jobId}` : jobId);
  const detail = useQuery({
    queryKey: ["ray-jobs", "detail", jobId],
    staleTime: 0,
    queryFn: ({ signal }) => rayRequest<RayJob>(`/api/ray/jobs/${encodeURIComponent(resolvedId)}`, "GET", signal),
    enabled: allowed && visible && !!jobId && !list.isPending && !!list.data?.data,
    retry: false,
    refetchInterval: (query) => query.state.data && isActiveJob(query.state.data) ? 5000 : false,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
  const [upgradeOpen, setUpgradeOpen] = useState(false);
  const [stopping, setStopping] = useState<RayJob | null>(null);
  const [now, setNow] = useState(Date.now());
  const selected = jobId ? matched && (!detail.data || !isActiveJob(matched)) ? matched : detail.data ?? matched ?? null : null;
  const previousJobs = useRef<{ key: string; ids: Set<string> } | null>(null);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const filterKey = JSON.stringify(filters);
  useEffect(() => {
    if (!list.data?.data || list.data.error) return;
    const ids = new Set(list.data.data.jobs.map((job) => job.submission_id));
    const previous = previousJobs.current;
    previousJobs.current = { key: filterKey, ids };
    setNewIds(new Set(previous?.key === filterKey ? [...ids].filter((id) => !previous.ids.has(id)) : []));
    const timer = window.setTimeout(() => setNewIds(new Set()), 100);
    return () => window.clearTimeout(timer);
  }, [list.data, filterKey]);
  const previousSelected = useRef<RayJob | null>(null);
  useEffect(() => {
    const previous = previousSelected.current;
    previousSelected.current = selected;
    if (previous && selected && previous.submission_id === selected.submission_id && isActiveJob(previous) && !isActiveJob(selected)) {
      void detail.refetch({ cancelRefetch: false });
    }
  }, [selected, detail.refetch]);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [visible]);
  const stop = useMutation({
    mutationFn: (job: RayJob) => rayRequest<{ submission_id: string; stopped: boolean; status: RayJob["status"] }>(`/api/ray/jobs/${encodeURIComponent(job.submission_id)}/stop`, "POST"),
    onSuccess: (response, job) => {
      queryClient.setQueryData(rayJobsOptions(filters).queryKey, (old: typeof list.data) => old?.data ? { ...old, data: { jobs: old.data.jobs.map((row) => row.submission_id === job.submission_id ? { ...row, status: response.status } : row) } } : old);
      setStopping(null);
      toast.success(response.stopped ? "Job stopped" : "Job already finished");
      queryClient.invalidateQueries({ queryKey: ["ray-jobs", "list"] });
      queryClient.invalidateQueries({ queryKey: ["ray-jobs", "detail"] });
    },
    onError: async (error, job) => {
      if (error instanceof RayApiError && error.status === 409) {
        setStopping(null);
        toast("Job already finished");
        try {
          const fresh = await rayRequest<RayJob>(`/api/ray/jobs/${encodeURIComponent(job.submission_id)}`);
          queryClient.setQueryData(["ray-jobs", "detail", jobId], fresh);
        } catch { /* The list retry below remains available. */ }
        await queryClient.invalidateQueries({ queryKey: ["ray-jobs", "list"] });
      } else if (error instanceof RayApiError && error.upgradeRequired) {
        setStopping(null);
        queryClient.setQueryData(rayLimitsOptions.queryKey, { data: null, error });
      } else toast.error(error.message);
    },
  });
  const errors = [limits.data.error, list.data?.error, status.data?.error, detail.error];
  const upgradeError = errors.find((error) => error instanceof RayApiError && error.upgradeRequired);
  const unavailable = [limits.data.error, list.data?.data ? null : list.data?.error, status.data?.error, detail.error].find((error) => error && !(error instanceof RayApiError && (error.upgradeRequired || error.status === 404)));
  const retry = () => { limits.refetch({ cancelRefetch: false }); status.refetch({ cancelRefetch: false }); list.refetch({ cancelRefetch: false }); if (jobId) detail.refetch({ cancelRefetch: false }); };
  const actions = (job: RayJob) => <div className="flex gap-2 max-md:w-full md:justify-end">
    <Button variant="outline" className="max-md:flex-1" onClick={() => onJobChange(job.submission_id)}>Logs</Button>
    {isActiveJob(job) && <Button variant="outline" className="border-error-primary text-error-primary max-md:flex-1" onClick={() => setStopping(job)}>Stop</Button>}
  </div>;
  return <TooltipProvider><main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-background px-4 py-6 text-foreground sm:px-8 sm:py-8">
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div><h1 className="text-2xl font-semibold">Ray jobs</h1><p className="mt-1 text-sm text-muted-foreground">Data transfers and training jobs you started</p></div>
      {status.data?.data && <ClusterStatusPill status={status.data.data} />}
      {allowed && status.isPending && <Skeleton className="h-8 w-56 rounded-full" />}
    </header>
    {(upgradeError || (!allowed && !limits.data.error)) ? <section className="mt-8 max-w-lg rounded-lg border border-brand bg-brand-primary_alt p-6">
      <Server className="mb-4 size-8 text-brand-secondary" />
      <h2 className="font-semibold">Ray jobs requires an upgrade</h2>
      <p className="mt-2 text-sm text-muted-foreground">{upgradeError?.message ?? "Upgrade your plan to access data transfer and Ray jobs."}</p>
      <Button className="mt-5 bg-brand-solid text-primary_on-brand hover:bg-brand-solid_hover" onClick={() => setUpgradeOpen(true)}>Upgrade plan</Button>
    </section> : <>
      {unavailable && <div role="alert" className="mt-6 flex items-center justify-between gap-4 rounded-lg border border-border bg-warning-primary p-4 text-warning-primary"><span className="flex items-center gap-2"><AlertTriangle className="size-4 shrink-0" />Ray cluster unavailable</span><Button variant="outline" onClick={retry}>Retry</Button></div>}
      {allowed && <>
         <div className="mt-6 flex flex-wrap items-end gap-3">
          <Filter label="Status" value={filters.status} options={[["", "All"], ["RUNNING", "Running"], ["SUCCEEDED", "Succeeded"], ["FAILED", "Failed"], ["STOPPED", "Stopped"]]} onChange={(status) => onFiltersChange({ ...filters, status })} />
          <Filter label="Kind" value={filters.kind} options={[["", "All"], ["data_transfer", "Data transfer"], ["mta_training", "Training"]]} onChange={(kind) => onFiltersChange({ ...filters, kind })} />
           <div className="ml-auto flex h-10 items-center gap-2">
             {list.data?.refreshedAt !== undefined && <span className="text-xs text-muted-foreground">Updated {Math.max(0, Math.floor((now - list.data.refreshedAt) / 1000))}s ago</span>}
             <Button variant="ghost" size="icon" aria-label="Refresh jobs" title="Refresh jobs" disabled={list.isFetching} onClick={() => list.refetch({ cancelRefetch: false })}><RefreshCw className={list.isFetching ? "animate-spin motion-reduce:animate-none" : ""} /></Button>
           </div>
        </div>
         {list.data?.data && list.data.error && <p role="status" className="mt-2 text-xs text-muted-foreground">Couldn't refresh. Retrying…</p>}
        {detail.error instanceof RayApiError && detail.error.status === 404 && !selected && <p role="alert" className="mt-4 text-error-primary">Job not found</p>}
        <div className="mt-6 overflow-hidden rounded-lg border border-border md:bg-card max-md:border-0">
          <table className="w-full table-fixed max-md:block">
            <thead className="bg-muted/50 text-xs text-muted-foreground max-md:hidden"><tr>{["Status", "Job", "Rows", "Started", "Duration", "Actions"].map((label, i) => <th key={label} scope="col" className={`${i === 1 ? "w-[34%] text-left" : i === 5 ? "w-[17%] text-right" : "text-left"} px-4 py-3 font-medium`}>{label}</th>)}</tr></thead>
            <tbody className="max-md:flex max-md:flex-col max-md:gap-4">
              {list.isPending && Array.from({ length: 5 }, (_, i) => <tr key={i} className="border-t border-border max-md:block max-md:rounded-lg max-md:border max-md:p-4">{Array.from({ length: 6 }, (_, n) => <td key={n} className="px-4 py-5 max-md:block"><Skeleton className="h-4 w-full" /></td>)}</tr>)}
              {jobs.map((job) => <tr key={job.submission_id} data-new={newIds.has(job.submission_id)} data-highlighted={matchesJobId(job.submission_id, jobId)} className={`border-t border-border transition-colors ${newIds.has(job.submission_id) ? "duration-0" : "duration-[2000ms]"} motion-reduce:transition-none max-md:grid max-md:grid-cols-2 max-md:gap-3 max-md:rounded-lg max-md:border max-md:p-4 ${matchesJobId(job.submission_id, jobId) ? "bg-brand-primary_alt ring-1 ring-inset ring-brand" : newIds.has(job.submission_id) ? "bg-primary/5" : "bg-transparent max-md:bg-card"}`}>
                <td className="px-4 py-4 max-md:order-1 max-md:p-0"><JobStatusBadge status={job.status} /></td>
                <td className="min-w-0 px-4 py-4 max-md:order-3 max-md:col-span-2 max-md:p-0"><JobPath job={job} /><JobIdentity job={job} /></td>
                <td className="px-4 py-4 text-sm text-muted-foreground max-md:order-4 max-md:p-0"><span className="md:hidden">Rows </span>{jobRows(job, reportedResults.data[job.submission_id])?.toLocaleString() ?? "—"}</td>
                <td title={localJobStart(job)} className="px-4 py-4 text-sm text-muted-foreground max-md:order-2 max-md:p-0 max-md:text-right">{relativeStarted(jobStartTime(job) === undefined ? null : new Date(jobStartTime(job) ?? 0).toISOString(), now)}</td>
                <td className="px-4 py-4 text-sm text-muted-foreground max-md:order-5 max-md:p-0"><span className="md:hidden">Duration </span>{durationText(job, now)}</td>
                <td className="px-4 py-4 max-md:order-6 max-md:col-span-2 max-md:p-0">{actions(job)}</td>
              </tr>)}
            </tbody>
          </table>
          {!list.isPending && !list.data?.error && !jobs.length && <div className="flex flex-col items-center gap-4 py-16 text-center"><Server className="size-8 text-muted-foreground" /><p className="max-w-sm px-4 text-sm text-muted-foreground">No Ray jobs yet. Start a data transfer from chat and it will appear here.</p><Button asChild className="bg-brand-solid text-primary_on-brand hover:bg-brand-solid_hover"><Link to="/proanalysis">Open chat</Link></Button></div>}
        </div>
      </>}
    </>}
    <RayLogsDrawer job={upgradeError ? null : selected} cluster={status.data?.data ?? undefined} onClose={() => onJobChange("")} onUpgrade={(error) => queryClient.setQueryData(rayLimitsOptions.queryKey, { data: null, error })} />
    <AlertDialog open={!!stopping} onOpenChange={(open) => { if (!open && !stop.isPending) setStopping(null); }}>
      <AlertDialogContent className="border-border text-foreground"><AlertDialogTitle>Stop this job?</AlertDialogTitle><AlertDialogDescription>Rows already written stay written.</AlertDialogDescription>
        {stopping && <p className="break-all font-mono text-xs text-muted-foreground">{stopping.submission_id}</p>}
        <AlertDialogFooter><AlertDialogCancel disabled={stop.isPending}>Cancel</AlertDialogCancel><Button variant="destructive" disabled={stop.isPending} onClick={() => { if (stopping) stop.mutate(stopping); }}>{stop.isPending && <LoaderCircle className="animate-spin" />}Stop job</Button></AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
    <UpgradePlanDialog open={upgradeOpen} feature="Ray jobs" onClose={() => setUpgradeOpen(false)} />
  </main></TooltipProvider>;
}
function Filter({ label, value, options, onChange }: { label: string; value: string; options: string[][]; onChange: (value: string) => void }) {
  return <div className="min-w-0 flex-1 md:max-w-[150px]"><label className="mb-1 block text-xs text-muted-foreground">{label}</label>
    <Select value={value || "all"} onValueChange={(next) => onChange(next === "all" ? "" : next)}><SelectTrigger aria-label={label} className="h-10 bg-card"><SelectValue /></SelectTrigger><SelectContent className="border-border bg-popover text-popover-foreground dark:border-border dark:bg-popover dark:text-popover-foreground">{options.map(([id, text]) => <SelectItem key={id} value={id || "all"}>{text}</SelectItem>)}</SelectContent></Select>
  </div>;
}