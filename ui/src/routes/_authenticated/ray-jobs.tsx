import { createFileRoute } from "@tanstack/react-router";
import { RayJobsPage } from "@/components/ray-jobs/RayJobsPage";
import { rayJobsOptions, rayLimitsOptions, rayStatusOptions, hasDataTransfer } from "@/lib/ray-jobs";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/_authenticated/ray-jobs")({
  validateSearch: (search: Record<string, unknown>) => ({
    status: ["RUNNING", "SUCCEEDED", "FAILED", "STOPPED"].includes(String(search.status)) ? String(search.status) : "",
    kind: ["data_transfer", "mta_training"].includes(String(search.kind)) ? String(search.kind) : "",
    job: typeof search.job === "string" ? search.job : "",
  }),
  loaderDeps: ({ search }) => ({ status: search.status, kind: search.kind }),
  loader: async ({ context, deps }) => {
    const limits = await context.queryClient.ensureQueryData(rayLimitsOptions);
    if (hasDataTransfer(limits.data)) await Promise.all([
      context.queryClient.ensureQueryData(rayStatusOptions),
      context.queryClient.ensureQueryData(rayJobsOptions(deps)),
    ]);
  },
  head: () => ({ meta: [
    { title: "Ray jobs · Avaloka" },
    { name: "description", content: "Monitor your Avaloka data transfer and training jobs on Ray." },
    { property: "og:title", content: "Ray jobs · Avaloka" },
    { property: "og:description", content: "Monitor your Avaloka data transfer and training jobs on Ray." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary" },
  ] }),
  component: RayJobsRoute,
  pendingComponent: () => <div className="p-8 text-muted-foreground">Loading Ray jobs…</div>,
  errorComponent: ({ reset }) => <div role="alert" className="p-8 text-foreground">Ray cluster unavailable <Button variant="outline" onClick={() => { window.location.reload(); reset(); }}>Retry</Button></div>,
  notFoundComponent: () => <div className="p-8 text-foreground">Job not found</div>,
});
function RayJobsRoute() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();
  return <RayJobsPage filters={{ status: search.status, kind: search.kind }} jobId={search.job}
    onFiltersChange={(filters) => navigate({ search: (previous) => ({ ...previous, ...filters }) })}
    onJobChange={(job) => navigate({ search: (previous) => ({ ...previous, job }), replace: true })} />;
}