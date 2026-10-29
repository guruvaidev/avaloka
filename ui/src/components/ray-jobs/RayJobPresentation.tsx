import { Copy, LoaderCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { JobStatus, RayJob, RayStatus } from "@/lib/ray-jobs";
import { displayJobId, jobTitle, shortenPath } from "@/lib/ray-jobs";

const statusStyles: Record<JobStatus, string> = {
  PENDING: "bg-muted text-muted-foreground",
  RUNNING: "bg-brand-primary_alt text-brand-secondary",
  SUCCEEDED: "bg-success-primary text-success-primary",
  FAILED: "bg-error-primary text-error-primary",
  STOPPED: "bg-warning-primary text-warning-primary",
};
export function JobStatusBadge({ status }: { status: JobStatus }) {
  return <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyles[status]}`}>
    {status === "RUNNING" && <LoaderCircle aria-hidden="true" className="size-3 animate-spin motion-reduce:animate-none" />}
    {status.charAt(0) + status.slice(1).toLowerCase()}
  </span>;
}
export function ClusterStatusPill({ status }: { status: RayStatus }) {
  const text = status.reachable
    ? `Connected${status.ray_version ? ` · Ray ${status.ray_version}` : ""}${status.nodes != null ? ` · ${status.nodes} nodes` : ""}`
    : status.configured ? "Cluster unreachable" : "No Ray cluster configured";
  return <span role="status" className="inline-flex max-w-full items-center gap-2 rounded-full border border-border bg-muted/50 px-3 py-1.5 text-xs text-foreground">
    <span aria-hidden="true" className={`size-2 shrink-0 rounded-full ${status.reachable ? "bg-success-solid" : status.configured ? "bg-warning-solid" : "bg-muted-foreground"}`} />{text}
  </span>;
}
export function JobIdentity({ job }: { job: RayJob }) {
  return <div className="flex min-w-0 items-center gap-1 text-xs text-muted-foreground">
    <span className="truncate font-mono" title={displayJobId(job.submission_id)}>{displayJobId(job.submission_id)}</span>
    <Button variant="ghost" size="icon" className="size-7 shrink-0" aria-label={`Copy ${job.submission_id}`} title="Copy job ID" onClick={async () => {
      try { await navigator.clipboard.writeText(job.submission_id); toast.success("Job ID copied"); }
      catch { toast.error("Could not copy job ID"); }
    }}><Copy /></Button>
  </div>;
}
export function JobPath({ job }: { job: RayJob }) {
  return <Tooltip><TooltipTrigger asChild>
    <span tabIndex={0} className="block truncate text-sm font-medium text-foreground">{shortenPath(job.source ?? "—")} → {shortenPath(job.destination ?? "—")}</span>
  </TooltipTrigger><TooltipContent className="max-w-sm break-all bg-popover text-popover-foreground border border-border">{jobTitle(job)}</TooltipContent></Tooltip>;
}